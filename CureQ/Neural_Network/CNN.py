# -*- coding: utf-8 -*-
"""
MEA-classificatie (CPU ONLY) – CTRL vs SCA1
Versie: RONDES METHODE (CLEANED)

Oplossing voor lege curves:
1. De plot-functie is robuuster gemaakt en checkt op NaN waarden.
2. CPU wordt geforceerd om 'optimizer crashes' in ronde 2 te voorkomen.
"""

import os

# ---------------------------------------------------------
# STAP 1: FORCEER CPU (AANGEZET TEGEN NAN CRASHES)
# ---------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json, csv, gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt
# Voorkomt GUI errors bij het plotten (headless mode)
plt.switch_backend("Agg")
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, callbacks

print("[System] TensorFlow Devices:", tf.config.list_physical_devices())


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    manifest_path: str = "./manifest.json"

    # --- Spike Exclusion ---
    spike_csv_path: str = "./analyses/activity_analyzed/measurements_spikes_bursts/electrode_spikes_over_measurement.csv"
    spike_cutoff_hz: float = 13.0
    # -----------------------

    # Training batch settings
    n_trainval_files: int = 8
    n_test_files: int = 2

    seed: int = 51

    key_raw: str = "Data/Recording_0/AnalogStream/Stream_0/ChannelData"
    rows: int = 6
    cols: int = 8
    channel_layout: str = "well_major"

    sampling_hz: int = 20000
    use_downsample: bool = True
    target_hz: int = 4000
    seq_len_ms: int = 4000

    # Segmentatie settings
    train_segments_per_well: int = 90
    val_segments_per_well: int = 30
    # test_segments_per_well is verwijderd (gebruikt stride_ms_eval)
    stride_ms_eval: int = 4000

    norm_per_electrode: bool = True

    # Model settings
    batch_size: int = 24
    lr: float = 1e-4
    epochs: int = 500
    patience: int = 20

    use_only_one_ctrl_line: bool = False
    ctrl_line_side: str = "left"
    channel_shuffle: bool = True

    outdir: str = "output_curves_130126_13hz_gpu"
    final_test_fraction: float = 0.2
    max_rounds: Optional[int] = 0  # 0/None = geen limiet


cfg = Config()
os.makedirs(cfg.outdir, exist_ok=True)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

CKPT_PATH = os.path.join(cfg.outdir, "best_model.keras")
LAST_PATH = os.path.join(cfg.outdir, "last_model.keras")
GLOBAL_SPLIT_PATH = os.path.join(cfg.outdir, "global_split.json")
USED_TRAIN_LOG = os.path.join(cfg.outdir, "used_train_files.json")


# -----------------------------
# Spike CSV Helpers
# -----------------------------
def load_spike_database(csv_path: str) -> Dict[Tuple[str, int, int], int]:
    if not os.path.exists(csv_path):
        print(f"[Warning] Spike CSV niet gevonden: {csv_path}. Er wordt niets geexcludeerd.")
        return {}

    spike_db = {}
    print(f"[Data] Spike CSV inlezen: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {}

        meas_ids = header[2:]
        count_records = 0
        for row in reader:
            if not row:
                continue
            try:
                well = int(row[0])
                elec = int(row[1])
                counts = row[2:]
                for m_idx, val_str in enumerate(counts):
                    if m_idx < len(meas_ids):
                        meas_id = meas_ids[m_idx]
                        try:
                            val = int(float(val_str))
                        except:
                            val = 0
                        spike_db[(meas_id, well, elec)] = val
                        count_records += 1
            except Exception:
                pass

    print(f"[Data] {count_records} spike-waarden ingeladen voor {len(meas_ids)} metingen.")
    return spike_db


def mask_noisy_electrodes_single(raw: np.ndarray, file_rec: Dict, spike_db: Dict, cutoff_hz: float, fs: int):
    if not spike_db:
        return

    meas_id = str(file_rec.get("id", ""))
    R, C, E, T = raw.shape
    duration_sec = T / fs
    masked_count = 0

    for r in range(R):
        for c in range(C):
            well_id = (r * cfg.cols) + (c + 1)
            for e_idx in range(E):
                elec_id = e_idx + 1
                key = (meas_id, well_id, elec_id)
                if key in spike_db:
                    count = spike_db[key]
                    hz = count / duration_sec

                    if hz > cutoff_hz:
                        raw[r, c, e_idx, :] = 0.0
                        masked_count += 1

    if masked_count > 0:
        print(f"      -> Totaal: {masked_count} elektroden op 0 gezet in dit bestand.")


# -----------------------------
# Manifest & Split helpers
# -----------------------------
def load_manifest(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest niet gevonden: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for rec in data:
        p = rec.get("path") or rec.get("path_rel") or rec.get("filename")
        if not p:
            continue
        p_abs = p if os.path.isabs(p) else os.path.abspath(p)
        if os.path.exists(p_abs):
            rec["_abs_path"] = p_abs
            valid.append(rec)
    return valid


def load_used_train_ids(path: str = USED_TRAIN_LOG) -> set:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(int(x) for x in json.load(f).get("used_train_ids", []))
    except:
        return set()


def save_used_train_ids(used_ids: set, path: str = USED_TRAIN_LOG) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"used_train_ids": sorted(int(x) for x in used_ids)}, f, indent=2)


def build_or_load_global_split(manifest: List[Dict], final_test_fraction: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    if os.path.exists(GLOBAL_SPLIT_PATH):
        print(f"[GlobalSplit] Laden: {GLOBAL_SPLIT_PATH}")
        with open(GLOBAL_SPLIT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        final_ids, train_ids = set(data["final_test_ids"]), set(data["trainval_ids"])
    else:
        print("[GlobalSplit] Nieuwe split maken.")
        ids_all = sorted(set(int(rec["id"]) for rec in manifest if rec.get("id") is not None))
        rng = np.random.RandomState(cfg.seed)
        idx = np.arange(len(ids_all))
        rng.shuffle(idx)
        n_final = max(1, int(round(len(ids_all) * final_test_fraction)))
        final_ids = {ids_all[i] for i in idx[:n_final]}
        train_ids = {ids_all[i] for i in idx[n_final:]}
        with open(GLOBAL_SPLIT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "final_test_fraction": final_test_fraction,
                    "final_test_ids": sorted(final_ids),
                    "trainval_ids": sorted(train_ids),
                },
                f,
                indent=2,
            )

    id2rec = {int(rec["id"]): rec for rec in manifest if rec.get("id") is not None}
    return [id2rec[i] for i in train_ids if i in id2rec], [id2rec[i] for i in final_ids if i in id2rec]


# ✅ AANGEPAST: pakt “wat er nog is” en stopt pas als er geen train files meer zijn
def pick_next_files(
    manifest: List[Dict], seed: int, n_trainval: int, n_test: int
) -> Tuple[List[Dict], List[Dict], bool]:
    """
    Pakt per ronde zoveel mogelijk ongebruikte files.
    - Stopt pas (done=True) als er geen trainval files meer te pakken zijn.
    - Testset mag leeg zijn (dan wordt evaluatie overgeslagen).
    """
    used_ids = load_used_train_ids()
    manifest_unused = [rec for rec in manifest if rec.get("id") not in used_ids]
    remaining = len(manifest_unused)

    if remaining == 0:
        return [], [], True

    # probeer test te pakken, maar houd minstens 1 file over voor training als dat kan
    n_test_eff = min(n_test, max(0, remaining - 1))
    n_train_eff = min(n_trainval, remaining - n_test_eff)

    if n_train_eff <= 0:
        return [], [], True

    rng = np.random.RandomState(seed)
    idx = np.arange(remaining)
    rng.shuffle(idx)

    test_files = [manifest_unused[i] for i in idx[:n_test_eff]]
    trainval_files = [manifest_unused[i] for i in idx[n_test_eff : n_test_eff + n_train_eff]]

    for rec in trainval_files + test_files:
        if rec.get("id") is not None:
            used_ids.add(int(rec["id"]))
    save_used_train_ids(used_ids)

    return trainval_files, test_files, False


# -----------------------------
# IO & Processing
# -----------------------------
def build_label_map_for_columns(cols: int) -> Dict[int, str]:
    lab = {}
    for c in range(cols):
        if c in (0, 1, 4, 5):
            lab[c] = "SCA1"
        elif c in (2, 3, 6, 7):
            lab[c] = "CTRL"
        else:
            lab[c] = "UNK"
    return lab


def get_allowed_cols_per_label(cols: int) -> Dict[str, set]:
    col2label = build_label_map_for_columns(cols)
    all_cols = {}
    for c in range(cols):
        lab = col2label.get(c, "UNK")
        if lab != "UNK":
            all_cols.setdefault(lab, set()).add(c)
    if cfg.use_only_one_ctrl_line and "CTRL" in all_cols:
        side = {2, 3} if cfg.ctrl_line_side == "left" else {6, 7}
        all_cols["CTRL"] &= side
    return all_cols


def load_raw_as_rcte(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        raw = f[cfg.key_raw][()]
    ch, T = raw.shape
    R, C = cfg.rows, cfg.cols
    if cfg.channel_layout == "well_major":
        return raw.reshape(R * C, ch // (R * C), T).reshape(R, C, ch // (R * C), T).astype(np.float32)
    else:
        return np.transpose(raw.reshape(ch // (R * C), R * C, T), (1, 0, 2)).reshape(R, C, ch // (R * C), T).astype(
            np.float32
        )


def maybe_downsample(raw: np.ndarray, fs: int, target: int) -> tuple:
    if not cfg.use_downsample or target >= fs:
        return raw, fs
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(fs, target)
    p, q = target // g, fs // g
    R, C, E, T = raw.shape
    out = np.empty((R, C, E, int(np.ceil(T * p / q))), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            out[r, c] = resample_poly(raw[r, c], p, q, axis=1)
    return out, target


def normalize_per_electrode(raw: np.ndarray) -> np.ndarray:
    raw = np.nan_to_num(raw, copy=False)
    med = np.median(raw, axis=-1, keepdims=True)
    mad = np.median(np.abs(raw - med), axis=-1, keepdims=True)
    return (raw - med) / (1.4826 * mad + 1e-7)


def wells_from_file(file_idx: int, C: int, R: int) -> List[Tuple]:
    col2lab = build_label_map_for_columns(C)
    return [(file_idx, r, c, col2lab[c]) for r in range(R) for c in range(C) if col2lab.get(c, "UNK") != "UNK"]


def filter_wells(wells: List[Tuple], allowed: Dict[str, set]) -> List[Tuple]:
    return [w for w in wells if w[3] in allowed and (allowed[w[3]] is None or w[2] in allowed[w[3]])]


def make_segments(raw_list, wells, n_seg, seq_len, name2idx=None, seed=42, shuffle=False):
    rng = np.random.RandomState(seed)
    X_l, lab_l = [], []
    for (fi, r, c, lab) in wells:
        sig = raw_list[fi][r, c]
        if np.all(sig == 0):
            continue
        if sig.shape[1] < seq_len:
            continue

        starts = rng.randint(0, sig.shape[1] - seq_len + 1, size=n_seg)
        batch = []
        for s in starts:
            seg = sig[:, s : s + seq_len]
            if shuffle:
                seg = seg[rng.permutation(seg.shape[0]), :]
            batch.append(seg.T)
        if batch:
            X_l.append(np.stack(batch))
            lab_l += [lab] * len(batch)

    if not X_l:
        raise RuntimeError("Geen segmenten gegenereerd (alles te kort of gemaskeerd?)")
    X = np.concatenate(X_l, axis=0)

    if name2idx is None:
        classes = sorted(list(set(lab_l)))
        name2idx = {n: i for i, n in enumerate(classes)}
    else:
        classes = [k for k, v in sorted(name2idx.items(), key=lambda x: x[1])]

    y = np.array([name2idx[l] for l in lab_l], dtype=np.int32)
    return X, y, classes, name2idx


# -----------------------------
# LIGHTWEIGHT Model (Aangepast)
# -----------------------------
def build_model(n_classes, seq_len, n_chan):
    inp = layers.Input(shape=(seq_len, n_chan))
    x = layers.Conv1D(16, 1, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Separable Convs
    for f in [16, 32]:
        x = layers.SeparableConv1D(
            f,
            5,
            padding="same",
            depthwise_regularizer=tf.keras.regularizers.l2(1e-3),
            pointwise_regularizer=tf.keras.regularizers.l2(1e-3),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)

    # Dilated Convs
    for d in [1, 2]:
        x = layers.Conv1D(32, 5, padding="same", dilation_rate=d, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)

    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-2))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# Plotting (Fixed)
# -----------------------------
def plot_curves(hist, out, tag):
    """
    Robuuste plot functie die checks doet op NaN waarden.
    """
    for metric in ["loss", "accuracy"]:
        data_train = hist.get(metric, [])
        data_val = hist.get(f"val_{metric}", [])

        if not data_train:
            print(f"[Plot] Geen data voor {metric} in {tag}")
            continue

        if np.any(np.isnan(data_train)):
            print(f"[Plot Warning] {tag} - {metric} bevat NaNs! Eerste 5: {data_train[:5]}")

        plt.figure(figsize=(8, 4))
        plt.plot(data_train, label=f"Train {metric}")
        if data_val:
            plt.plot(data_val, label=f"Val {metric}")

        plt.title(f"{metric.title()} ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{metric}_{tag}.png"))
        plt.close()


# -----------------------------
# Evaluatie
# -----------------------------
def consensus_eval_multifile(raw_list_test, test_files, fs, model, classes_train, seq_len_ms, stride_ms, outdir, round_tag):
    seq_len = int(round(seq_len_ms * fs / 1000.0))
    stride = max(1, int(round(stride_ms * fs / 1000.0)))
    col2label = build_label_map_for_columns(cfg.cols)
    allowed = set(classes_train)

    y_true, y_pred, rows_out = [], [], [("filename", "row", "col", "true_label", "pred_label")]
    print(f"\n[Evaluatie] Start consensus over {len(test_files)} testbestanden...")

    for i, raw in enumerate(raw_list_test):
        fname = os.path.basename(test_files[i].get("filename", f"file_{i}"))
        R, C, E, T = raw.shape
        for r in range(R):
            for c in range(C):
                lab = col2label.get(c, "UNK")
                if lab == "UNK" or lab not in allowed:
                    continue
                sig = raw[r, c]
                if np.all(sig == 0):
                    continue
                if sig.shape[1] < seq_len:
                    continue

                starts = list(range(0, sig.shape[1] - seq_len + 1, stride))
                if not starts:
                    continue

                batch = np.stack([sig[:, s : s + seq_len].T for s in starts]).astype(np.float32)
                probs = model.predict(batch, verbose=0)
                pred_lab = classes_train[np.bincount(probs.argmax(axis=1), minlength=len(classes_train)).argmax()]

                y_true.append(lab)
                y_pred.append(pred_lab)
                rows_out.append((fname, r, c, lab, pred_lab))

    if not y_true:
        print(f"[{round_tag}] Geen data voor evaluatie.")
        return

    cls_sorted = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=cls_sorted, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=cls_sorted)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n=== RESULTS ({round_tag}) ===")
    print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")
    print(report)

    with open(os.path.join(outdir, f"report_{round_tag}.txt"), "w") as f:
        f.write(f"Acc: {acc:.4f}\nPrec: {prec:.4f}\nF1: {f1:.4f}\n\n{report}\n\nCM:\n{np.array2string(cm)}")

    with open(os.path.join(outdir, f"consensus_{round_tag}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_out)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cls_sorted, yticklabels=cls_sorted)
    plt.title(f"Confusion Matrix - {round_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cm_{round_tag}.png"))
    plt.close()


# -----------------------------
# Main Helpers: Process Loop
# -----------------------------
def process_file_list(files: List[Dict], spike_db: Dict, desc: str) -> Tuple[List[np.ndarray], int]:
    processed_list = []
    final_fs = cfg.sampling_hz

    print(f"\n--- Start Processing {desc} ---")

    for i, f in enumerate(files):
        fname = f.get("filename", "Unknown")
        print(f"[{i+1}/{len(files)}] Loading: {fname} ...", end=" ", flush=True)

        try:
            raw = load_raw_as_rcte(f["_abs_path"])
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        mask_noisy_electrodes_single(raw, f, spike_db, cfg.spike_cutoff_hz, cfg.sampling_hz)
        raw, final_fs = maybe_downsample(raw, cfg.sampling_hz, cfg.target_hz)

        if cfg.norm_per_electrode:
            raw = normalize_per_electrode(raw)

        processed_list.append(raw)
        gc.collect()
        print("Done.")

    return processed_list, final_fs


# -----------------------------
# Main Loop
# -----------------------------
def train_one_round(trainval, test, idx, spike_db):
    print(f"\n========== RONDE {idx+1} ==========")

    print("\n--- SAMENVATTING BESTANDEN DEZE RONDE ---")
    print(f"Train/Val files ({len(trainval)}):")
    for f in trainval:
        print(f"  - {f.get('filename', 'Unknown')}")

    print(f"Test files ({len(test)}):")
    for f in test:
        print(f"  - {f.get('filename', 'Unknown')}")
    print("-----------------------------------------\n")

    raw_tr, fs = process_file_list(trainval, spike_db, "Train/Val Set")
    raw_te, _ = process_file_list(test, spike_db, "Test Set")

    if not raw_tr:
        print("Geen trainingsdata geladen. Stop ronde.")
        return

    E = raw_tr[0].shape[2]

    wells_all = [w for i in range(len(raw_tr)) for w in wells_from_file(i, cfg.cols, cfg.rows)]
    tr_wells, va_wells = train_test_split(
        wells_all, test_size=0.2, stratify=[w[3] for w in wells_all], random_state=cfg.seed + idx
    )
    tr_wells = filter_wells(tr_wells, get_allowed_cols_per_label(cfg.cols))

    seq_len = int(round(cfg.seq_len_ms * fs / 1000.0))

    print(f"[Training] Genereren segmenten...")
    Xtr, ytr, classes, n2i = make_segments(
        raw_tr, tr_wells, cfg.train_segments_per_well, seq_len, seed=cfg.seed + idx, shuffle=cfg.channel_shuffle
    )
    Xva, yva, _, _ = make_segments(
        raw_tr, va_wells, cfg.val_segments_per_well, seq_len, name2idx=n2i, seed=cfg.seed + idx + 1
    )

    print(f"Classes: {classes}, Train shape: {Xtr.shape}")

    del raw_tr
    gc.collect()

    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(5000).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(len(classes), seq_len, E)
    if os.path.exists(CKPT_PATH):
        model.load_weights(CKPT_PATH)

    cbs = [
        callbacks.ModelCheckpoint(CKPT_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=cfg.epochs, callbacks=cbs, class_weight=dict(enumerate(cw)))

    model.save(LAST_PATH)
    h_dict = {k: [float(x) for x in v] for k, v in hist.history.items()}
    with open(os.path.join(cfg.outdir, f"history_round{idx+1}.json"), "w") as f:
        json.dump(h_dict, f)

    plot_curves(h_dict, cfg.outdir, f"round{idx+1}")

    # ✅ AANGEPAST: evaluatie overslaan als er geen test files / test data is
    if test and raw_te:
        consensus_eval_multifile(
            raw_te, test, fs, model, classes, cfg.seq_len_ms, cfg.stride_ms_eval, cfg.outdir, f"round{idx+1}"
        )
    else:
        print(f"[round{idx+1}] Geen testbestanden beschikbaar deze ronde -> evaluatie overgeslagen.")

    del raw_te, Xtr, ytr, Xva, yva, ds_tr, ds_va
    gc.collect()
    print(f"Ronde {idx+1} klaar.")


def main():
    manifest = load_manifest(cfg.manifest_path)
    spike_db = load_spike_database(cfg.spike_csv_path)

    pool, final = build_or_load_global_split(manifest, cfg.final_test_fraction)
    print(f"Pool: {len(pool)}, Final: {len(final)}")

    i = 0
    while True:
        if cfg.max_rounds and i >= cfg.max_rounds:
            break

        tr, te, done = pick_next_files(pool, cfg.seed + i, cfg.n_trainval_files, cfg.n_test_files)
        if done:
            print("[Done] Geen train-bestanden meer over. Stoppen.")
            break

        train_one_round(tr, te, i, spike_db)
        i += 1


if __name__ == "__main__":
    main()
