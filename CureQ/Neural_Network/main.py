# -*- coding: utf-8 -*-
"""
MEA-classificatie (Axion 6x8) – per-well Conv1D pipeline
- Leest 2D RAW HDF5: [channels, time] = (768, T)
- Reshapes naar [rows=6, cols=8, electrodes=16, time]
- Spike-detectie per elektrode (bandpass + neg. drempel), OR -> binaire well-trace [R,C,T]
- 1 ms binning -> ~1000 Hz
- Segmenten per well (seq_len_ms), Conv1D-training (multiclass)
- Consensus (majority vote) per well voor rapportage

Run:
    pip install tensorflow h5py scipy scikit-learn numpy
    python train_mea_conv1d_per_well.py
"""

import os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import h5py
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, callbacks


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # === Pad naar je HDF5 ===
    h5_path: str = r"./Data/SCA1/20250128_SCA1_first_test(000).h5"

    # === HDF5 dataset key met raw analoge data ===
    key_raw: str = "Data/Recording_0/AnalogStream/Stream_0/ChannelData"

    # === MEA layout ===
    rows: int = 6
    cols: int = 8
    # Channels = rows*cols*electrodes_per_well; bij 768 ch -> 16 elektroden/well
    channel_layout: str = "well_major"
    """
    - "well_major": 16 kanalen van dezelfde well staan opeenvolgend, daarna volgende well, etc.
    - "electrode_major": alle e0 van alle wells eerst, dan e1 van alle wells, etc.
    """

    # === Sampling/segmentering ===
    sampling_hz: int = 20000  # raw sample rate (pas aan indien anders)
    seq_len_ms: int = 4000  # segmentduur
    bin_ms: int = 1  # 1 ms binning (reduceert 20kHz -> 1kHz)
    train_segments_per_well: int = 180
    val_segments_per_well: int = 60
    stride_ms_eval: int = 4000  # consensus stride

    # === Spike-detectie (op raw) ===
    do_spike_detect: bool = True
    bp_low_hz: float = 200.0
    bp_high_hz: float = 4000.0
    thresh_std: float = 5.0

    # === Training ===
    batch_size: int = 24
    lr: float = 1e-3
    epochs: int = 500
    patience: int = 50
    seed: int = 42

    # === Output ===
    outdir: str = "outputs"


cfg = Config()
os.makedirs(cfg.outdir, exist_ok=True)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


# -----------------------------
# Helpers
# -----------------------------
def build_label_map_for_columns(cols: int) -> Dict[int, str]:
    """
    Kolommen:
        1–2  -> HTT-CAG54
        3–4  -> CTRL
        5–6  -> HTT-CAG46
        7–8  -> CTRL
    (0-based indices)
    """
    lab = {}
    for c in range(cols):
        if c in (0, 1):
            lab[c] = "HTT-CAG54"
        elif c in (2, 3):
            lab[c] = "CTRL"
        elif c in (4, 5):
            lab[c] = "HTT-CAG46"
        elif c in (6, 7):
            lab[c] = "CTRL"
        else:
            lab[c] = "UNK"
    return lab


def robust_std(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def butter_bandpass(low, high, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def detect_spikes_from_raw(raw_rcte: np.ndarray, fs: int, low: float, high: float, thresh_std: float) -> np.ndarray:
    """
    raw_rcte: [R, C, E, T] float (µV)
    Retour:   [R, C, T] uint8 (OR over E)
    """
    R, C, E, T = raw_rcte.shape
    b, a = butter_bandpass(low, high, fs, order=2)
    out = np.zeros((R, C, T), dtype=np.uint8)
    for r in range(R):
        for c in range(C):
            well_bin = np.zeros(T, dtype=np.uint8)
            for e in range(E):
                sig = filtfilt(b, a, raw_rcte[r, c, e, :])
                sigma = robust_std(sig)
                thr = -thresh_std * sigma
                well_bin |= (sig < thr).astype(np.uint8)
            out[r, c, :] = well_bin
    return out


def bin_time(spk_rct: np.ndarray, fs: int, bin_ms: int = 1) -> np.ndarray:
    """
    Bint binaire spikes naar lagere tijdresolutie (OR binnen bin).
    spk_rct: [R,C,T] uint8
    Retour:  [R,C,Tbinned] uint8
    """
    factor = max(1, int(round(bin_ms * fs / 1000.0)))
    R, C, T = spk_rct.shape
    T_trim = (T // factor) * factor
    if T_trim == 0:
        raise ValueError("Te korte opname voor gekozen bin_ms.")
    x = spk_rct[:, :, :T_trim].reshape(R, C, -1, factor)
    return (x.sum(axis=3) > 0).astype(np.uint8)


def load_raw_as_rcte(h5_path: str) -> np.ndarray:
    """
    Leest 2D RAW [channels, time] en reshaped naar [R,C,E,T].
    Ondersteunt 'well_major' of 'electrode_major' kanaalvolgorde.
    """
    with h5py.File(h5_path, "r") as f:
        if cfg.key_raw not in f:
            raise KeyError(f"Dataset '{cfg.key_raw}' niet gevonden in {h5_path}")
        raw = f[cfg.key_raw][()]
    if raw.ndim != 2:
        raise ValueError(f"Verwacht 2D raw [channels,time], kreeg {raw.shape}")

    ch, T = raw.shape
    R, C = cfg.rows, cfg.cols
    wells = R * C
    if ch % wells != 0:
        raise ValueError(f"Channels ({ch}) niet deelbaar door aantal wells ({wells}).")
    E = ch // wells

    if cfg.channel_layout == "well_major":
        # [wells*E, T] -> [wells, E, T] -> [R, C, E, T]
        raw_rcte = raw.reshape(wells, E, T).reshape(R, C, E, T)
    elif cfg.channel_layout == "electrode_major":
        # [E*wells, T] -> [E, wells, T] -> [wells, E, T] -> [R,C,E,T]
        raw_e_wells_T = raw.reshape(E, wells, T)
        raw_rcte = np.transpose(raw_e_wells_T, (1, 0, 2)).reshape(R, C, E, T)
    else:
        raise ValueError("cfg.channel_layout moet 'well_major' of 'electrode_major' zijn.")
    return raw_rcte


def make_segments_per_well(spk_rct: np.ndarray, n_segments_per_well: int, seq_len: int) -> Tuple[
    np.ndarray, np.ndarray, List[str]]:
    """
    Maakt segmenten per well.
    Retourneert X: [N, seq_len, 1] uint8, y_idx: [N] int, classes: List[str]
    """
    R, C, T = spk_rct.shape
    col2label = build_label_map_for_columns(C)
    X_list, labels = [], []
    for r in range(R):
        for c in range(C):
            lab = col2label[c]
            if lab == "UNK":
                continue
            sig = spk_rct[r, c, :]
            if T < seq_len:
                continue
            starts = np.random.randint(0, T - seq_len + 1, size=n_segments_per_well)
            segs = np.stack([sig[s:s + seq_len] for s in starts], axis=0)  # [n, seq]
            X_list.append(segs[..., None])  # [n, seq, 1]
            labels += [lab] * segs.shape[0]

    if not X_list:
        raise RuntimeError("Geen segmenten gevonden (check data/seq_len_ms).")

    X = np.concatenate(X_list, axis=0).astype(np.uint8)
    classes = sorted(list(set(labels)))
    name2idx = {n: i for i, n in enumerate(classes)}
    y_idx = np.array([name2idx[n] for n in labels], dtype=np.int32)
    return X, y_idx, classes


def build_cnn_1d(n_classes: int, seq_len: int) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len, 1))
    x = inp
    for f in [64, 128, 256]:
        x = layers.Conv1D(f, kernel_size=5, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def consensus_eval_1d(spk_rct: np.ndarray, model: tf.keras.Model, classes: List[str],
                      seq_len: int, stride_ms: int, bin_ms: int, out_path: str) -> str:
    """
    Majority vote per well met stride in ms (op binned tijdas).
    """
    R, C, T = spk_rct.shape
    fs_bin = int(round(1000 / bin_ms))  # bij 1ms -> 1000 Hz
    stride = max(1, int(round(stride_ms * fs_bin / 1000.0)))

    col2label = build_label_map_for_columns(C)
    y_true, y_pred = [], []

    for r in range(R):
        for c in range(C):
            lab = col2label[c]
            if lab == "UNK":
                continue
            sig = spk_rct[r, c, :]
            if sig.shape[0] < seq_len:
                continue
            starts = list(range(0, sig.shape[0] - seq_len + 1, stride))
            batch = np.stack([sig[s:s + seq_len] for s in starts], axis=0)[..., None]
            probs = model.predict(batch.astype(np.float32), verbose=0)
            votes = probs.argmax(axis=1)
            pred_idx = np.bincount(votes, minlength=len(classes)).argmax()
            y_true.append(lab)
            y_pred.append(classes[pred_idx])

    cls_sorted = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=cls_sorted, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=cls_sorted)

    print("\n=== Consensus per well ===")
    print(report)
    print("Labels volgorde:", cls_sorted)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    with open(out_path, "w") as f:
        f.write(report + "\nLabels: " + ", ".join(cls_sorted) + "\n")
        f.write("Confusion (rows=true, cols=pred):\n")
        f.write(np.array2string(cm, separator=", "))

    return report


# -----------------------------
# Train pipeline
# -----------------------------
def train():
    # 1) Laad raw en maak spikes per well
    raw_rcte = load_raw_as_rcte(cfg.h5_path)  # [R,C,E,T]
    R, C, E, T = raw_rcte.shape
    print(f"RAW shape R,C,E,T = {raw_rcte.shape}  (verwacht 6,8,16,T)")

    if not cfg.do_spike_detect:
        raise ValueError("Deze pipeline verwacht spike-detectie (cfg.do_spike_detect=True).")

    spk_rct = detect_spikes_from_raw(raw_rcte, fs=cfg.sampling_hz,
                                     low=cfg.bp_low_hz, high=cfg.bp_high_hz,
                                     thresh_std=cfg.thresh_std)  # [R,C,T] uint8
    print("Spikes mean (ruwe, %):", float(spk_rct.mean()) * 100.0)

    # 2) 1 ms binning -> grote snelheids/geheugenwinst
    spk_rct = bin_time(spk_rct, fs=cfg.sampling_hz, bin_ms=cfg.bin_ms)  # [R,C,Tb]
    print("Na binning shape:", spk_rct.shape, "mean(%):", float(spk_rct.mean()) * 100.0)

    # 3) Segmenten per well
    seq_len = int(round(cfg.seq_len_ms / cfg.bin_ms))  # bij 1ms -> 4000 samples
    X_all, y_all, classes = make_segments_per_well(spk_rct, cfg.train_segments_per_well, seq_len)
    print("X_all:", X_all.shape, "y_all:", y_all.shape, "classes:", classes)

    # 4) Splits
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.20, random_state=cfg.seed, stratify=y_all)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.125, random_state=cfg.seed, stratify=ytr)  # 0.7/0.1/0.2

    # 5) tf.data (cast pas in de pipeline naar float32)
    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(10000, seed=cfg.seed) \
        .batch(cfg.batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y)) \
        .prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(cfg.batch_size) \
        .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
        .prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(cfg.batch_size) \
        .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
        .prefetch(tf.data.AUTOTUNE)

    # 6) Model
    model = build_cnn_1d(n_classes=len(classes), seq_len=seq_len)
    ckpt = os.path.join(cfg.outdir, "best_model.keras")
    cbs = [
        callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=cfg.patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=cbs)

    # 7) Bewaar
    model.save(os.path.join(cfg.outdir, "last_model.keras"))
    with open(os.path.join(cfg.outdir, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)
    with open(os.path.join(cfg.outdir, "classes.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    # 8) Snelle test op segment-niveau
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"[Segment] test acc = {acc:.4f}")

    # 9) Consensus per well (binned tijdas)
    report_path = os.path.join(cfg.outdir, "consensus_report.txt")
    _ = consensus_eval_1d(spk_rct, model, classes, seq_len, cfg.stride_ms_eval, cfg.bin_ms, report_path)
    print(f"Consensus-rapport geschreven naar: {report_path}")


if __name__ == "__main__":
    train()
