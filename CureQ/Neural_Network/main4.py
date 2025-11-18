# -*- coding: utf-8 -*-
"""
MEA-classificatie (RAW) – CTRL vs SCA1
Manifest + random TRAIN/VAL/TEST + resume + QC-exclusions + constant channel budget

- Leest manifest.json (lijst records met minimaal 'id', 'filename', 'path')
- Bestanden in manifest waarvan het pad niet bestaat worden genegeerd
- inactive_electrodes.csv: manifest-ID's die niet naar een bestaand bestand verwijzen worden genegeerd
- Random file-split (cfg.seed, aantallen instelbaar)
- Labels: CTRL vs SCA1 (kolomlayout samengevoegd)
- Resume: laad best_model.keras + classes.json indien aanwezig, anders fresh start
- QC: inactive_electrodes.csv met kolom 'inactive_measurements' (manifest-IDs)
- Constant channel budget: elke well/meeting heeft per segment hetzelfde aantal actieve kanalen (E_eff)
"""

import os, json, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, callbacks


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # Manifest + QC
    manifest_path: str = "./manifest.json"
    inactive_csv_path: str = "./inactive_electrodes.csv"   # jouw CSV
    well_index_base: int = 1        # CSV 'well' is 1-based
    electrode_index_base: int = 1   # CSV 'electrode' is 1-based

    # Random bestandselectie
    n_trainval_files: int = 1
    n_test_files: int = 1
    seed: int = 42

    # HDF5 dataset key
    key_raw: str = "Data/Recording_0/AnalogStream/Stream_0/ChannelData"

    # MEA layout
    rows: int = 6
    cols: int = 8
    channel_layout: str = "well_major"  # of "electrode_major"

    # Sampling / segmentering
    sampling_hz: int = 20000
    use_downsample: bool = False
    target_hz: int = 2000
    seq_len_ms: int = 4000
    train_segments_per_well: int = 180
    val_segments_per_well: int = 60
    test_segments_per_well: int = 60
    stride_ms_eval: int = 4000

    # Normalisatie
    norm_per_electrode: bool = True

    # Training
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 500
    patience: int = 50

    # Kolom-selecties
    use_only_one_ctrl_line: bool = False
    ctrl_line_side: str = "left"
    include_cols_per_label: Optional[Dict[str, List[int]]] = None

    # Channel budget & augmentatie
    constant_channel_budget: bool = True   # zelfde aantal actieve kanalen per segment
    channel_shuffle: bool = True           # random permutatie van kanalen per segment

    # Output
    outdir: str = "outputs_raw_CTRL_vs_SCA1_manifest"

cfg = Config()
os.makedirs(cfg.outdir, exist_ok=True)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

CKPT_PATH    = os.path.join(cfg.outdir, "best_model.keras")
LAST_PATH    = os.path.join(cfg.outdir, "last_model.keras")
CLASSES_PATH = os.path.join(cfg.outdir, "classes.json")
SPLIT_PATH   = os.path.join(cfg.outdir, "file_split.json")


# -----------------------------
# Manifest helpers
# -----------------------------
def load_manifest(path: str) -> List[Dict]:
    """
    Laadt manifest.json.
    - Entries waarvan het bestand niet (meer) bestaat, worden genegeerd.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest niet gevonden: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest moet een lijst van records zijn.")

    valid = []
    skipped = 0
    for rec in data:
        p = rec.get("path") or rec.get("path_rel") or rec.get("filename")
        if not p:
            skipped += 1
            continue
        p_abs = p if os.path.isabs(p) else os.path.abspath(p)
        if os.path.exists(p_abs):
            rec["_abs_path"] = p_abs
            valid.append(rec)
        else:
            skipped += 1
            print(f"[Manifest] Ignore entry: bestand bestaat niet ({p_abs})")

    if skipped > 0:
        print(f"[Manifest] In totaal {skipped} manifest entries genegeerd (bestanden niet aanwezig).")

    return valid

def random_split_files(manifest: List[Dict], seed: int,
                       n_trainval: int, n_test: int) -> Tuple[List[Dict], List[Dict]]:
    if len(manifest) < n_trainval + n_test:
        raise ValueError(
            f"Te weinig bestaande bestanden ({len(manifest)}) voor split: "
            f"{n_trainval} trainval + {n_test} test."
        )

    rng = np.random.RandomState(seed)
    idx = np.arange(len(manifest))
    rng.shuffle(idx)
    test_idx = idx[:n_test]
    trainval_idx = idx[n_test:n_test+n_trainval]
    test_files = [manifest[i] for i in test_idx]
    trainval_files = [manifest[i] for i in trainval_idx]

    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "trainval": [{"id": r.get("id"), "filename": r.get("filename"), "path": r["_abs_path"]} for r in trainval_files],
            "test": [{"id": r.get("id"), "filename": r.get("filename"), "path": r["_abs_path"]} for r in test_files],
        }, f, indent=2)
    print(f"[Split] TRAIN/VAL={len(trainval_files)}, TEST={len(test_files)} → {SPLIT_PATH}")
    return trainval_files, test_files


# -----------------------------
# Label layout (CTRL vs SCA1)
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
    if cfg.include_cols_per_label is not None:
        return {lab: set(v) for lab, v in cfg.include_cols_per_label.items()}
    col2label = build_label_map_for_columns(cols)
    all_cols_per_label: Dict[str, set] = {}
    for c in range(cols):
        lab = col2label.get(c, "UNK")
        if lab != "UNK":
            all_cols_per_label.setdefault(lab, set()).add(c)
    if cfg.use_only_one_ctrl_line:
        ctrl_left, ctrl_right = {2, 3}, {6, 7}
        chosen = ctrl_left if cfg.ctrl_line_side.lower() == "left" else ctrl_right
        if "CTRL" in all_cols_per_label:
            all_cols_per_label["CTRL"] = all_cols_per_label["CTRL"].intersection(chosen)
    return all_cols_per_label


# -----------------------------
# IO & RAW helpers
# -----------------------------
def load_raw_as_rcte(h5_path: str) -> np.ndarray:
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
        raw_rcte = raw.reshape(wells, E, T).reshape(R, C, E, T)
    elif cfg.channel_layout == "electrode_major":
        raw_e_wells_T = raw.reshape(E, wells, T)
        raw_rcte = np.transpose(raw_e_wells_T, (1, 0, 2)).reshape(R, C, E, T)
    else:
        raise ValueError("cfg.channel_layout moet 'well_major' of 'electrode_major' zijn.")
    return raw_rcte.astype(np.float32)

def maybe_downsample(raw_rcte: np.ndarray, fs: int, target_hz: int) -> tuple[np.ndarray, int]:
    if not cfg.use_downsample or target_hz >= fs:
        return raw_rcte, fs
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(fs, target_hz)
    p, q = target_hz // g, fs // g
    R, C, E, T = raw_rcte.shape
    out = np.empty((R, C, E, int(np.ceil(T * p / q))), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            out[r, c] = resample_poly(raw_rcte[r, c], p, q, axis=1)
    return out, target_hz

def normalize_per_electrode(raw_rcte: np.ndarray) -> np.ndarray:
    med = np.median(raw_rcte, axis=-1, keepdims=True)
    mad = np.median(np.abs(raw_rcte - med), axis=-1, keepdims=True)
    sigma = 1.4826 * mad + 1e-12
    return (raw_rcte - med) / sigma


# -----------------------------
# QC: CSV → exclusions map
# -----------------------------
def well_id_to_row_col(well_id: int, rows: int, cols: int, base: int = 1) -> Tuple[int, int]:
    """well_id (base-1) → (row,col) in rij-major (links→rechts, top→down)."""
    idx0 = well_id - base
    if idx0 < 0 or idx0 >= rows * cols:
        raise ValueError(f"well_id buiten bereik: {well_id}")
    r = idx0 // cols
    c = idx0 % cols
    return r, c

def build_manifest_id_index(manifest: List[Dict]) -> Dict[int, Dict]:
    """
    Bouw mapping id -> manifest record.
    Alleen manifest entries die een bestaand bestand hebben (na load_manifest) worden meegenomen.
    """
    out = {}
    for rec in manifest:
        mid = rec.get("id")
        if mid is not None:
            out[int(mid)] = rec
    return out

def load_qc_exclusions(csv_path: str,
                       manifest: List[Dict],
                       rows: int,
                       cols: int,
                       well_base: int = 1,
                       elec_base: int = 1) -> Dict[Tuple[str, int, int], set]:
    """
    Bouwt dict:
        key = (abs_path, row, col)
        value = set({electrode_idx, ...})

    CSV-kolommen:
        well (1-based), electrode (1-based),
        inactive_measurements (bijv '6,19,23'),
        optioneel always_inactive (0/1).

    Manifest-IDs die verwijzen naar bestanden die niet (meer) bestaan worden genegeerd.
    """
    excl: Dict[Tuple[str, int, int], set] = {}
    if not os.path.exists(csv_path):
        print(f"[QC] Geen CSV gevonden op {csv_path} (skip).")
        return excl

    id2rec = build_manifest_id_index(manifest)
    if not id2rec:
        print("[QC] Geen valide manifest-IDs beschikbaar (alleen bestaande bestanden tellen).")

    def add_excl(abs_path: str, r: int, c: int, e: int):
        key = (abs_path, r, c)
        if key not in excl:
            excl[key] = set()
        excl[key].add(e)

    import pandas as pd
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        try:
            well_id = int(row["well"])
            elec_id = int(row["electrode"])
        except Exception:
            # verplichte kolommen ontbreken → rij negeren
            continue

        try:
            r, c = well_id_to_row_col(well_id, rows, cols, base=well_base)
        except ValueError:
            # well_id buiten bereik → negeren
            continue

        e_idx = elec_id - elec_base
        if e_idx < 0:
            continue

        always = int(row.get("always_inactive", 0)) if "always_inactive" in df.columns else 0
        inactive_str = str(row.get("inactive_measurements", "")).strip()

        # ALWAYS_INACTIVE: elektroden altijd inactive in alle *bestaande* manifest entries
        if always == 1:
            for rec in manifest:
                abs_path = rec["_abs_path"]
                add_excl(abs_path, r, c, e_idx)
            continue

        if inactive_str == "" or inactive_str.lower() == "nan":
            continue

        # Parse comma-lijst van manifest-IDs
        tokens = [t.strip() for t in inactive_str.split(",") if t.strip() != ""]
        for tok in tokens:
            try:
                mid = int(tok)
            except ValueError:
                continue
            if mid not in id2rec:
                # Dit is precies jouw wens: ID verwijst naar een niet-bestaand bestand → negeren
                print(f"[QC] Ignore inactive_measurements ID {mid}: geen bestaand bestand in manifest.")
                continue
            abs_path = id2rec[mid]["_abs_path"]
            add_excl(abs_path, r, c, e_idx)

    return excl

def apply_exclusions_to_raw_rcte(raw_rcte: np.ndarray,
                                 abs_path: str,
                                 exclusions: Dict[Tuple[str, int, int], set]):
    """Zet QC-uitgesloten elektroden op 0. In-place."""
    R, C, E, T = raw_rcte.shape
    for r in range(R):
        for c in range(C):
            key = (abs_path, r, c)
            if key not in exclusions:
                continue
            for e in exclusions[key]:
                if 0 <= e < E:
                    raw_rcte[r, c, e, :] = 0.0


# -----------------------------
# Constant channel budget helpers
# -----------------------------
def compute_E_eff(E: int, rows: int, cols: int,
                  manifest_sel: List[Dict],
                  exclusions_map: Dict[Tuple[str, int, int], set]) -> int:
    """
    Bepaal minimum aantal "niet-QC-gemaskerde" kanalen over alle (file,row,col).
    E_eff = min_well (E - #QC_exclusions).
    """
    E_eff = E
    for rec in manifest_sel:
        abs_path = rec["_abs_path"]
        for r in range(rows):
            for c in range(cols):
                key = (abs_path, r, c)
                excl = exclusions_map.get(key, set())
                excl_count = len(excl)
                active_here = max(0, E - excl_count)
                E_eff = min(E_eff, active_here)
    # minimaal 1 kanaal
    E_eff = max(1, E_eff)
    return E_eff

def make_segment_channel_mask(E: int, base_exclusions: set, E_eff: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Bouw mask: True = kanaal blijft "actief", False = maskeren (op 0 zetten).
    - base_exclusions: QC-uitgesloten elektroden (0-based indices).
    - E_eff: aantal actieve kanalen dat je wilt overhouden.
    """
    mask = np.ones(E, dtype=bool)
    for e in base_exclusions:
        if 0 <= e < E:
            mask[e] = False
    active_idx = np.flatnonzero(mask)
    if len(active_idx) <= E_eff:
        # al op/onder budget; laat zo
        return mask
    drop_count = len(active_idx) - E_eff
    drop_idxs = rng.choice(active_idx, size=drop_count, replace=False)
    mask[drop_idxs] = False
    return mask

def apply_mask_to_segment(sig_E_T: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    sig_E_T: shape (E, T). Mask False -> 0.
    """
    out = sig_E_T.copy()
    out[~mask, :] = 0.0
    return out

def maybe_shuffle_channels(sig_E_T: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    perm = rng.permutation(sig_E_T.shape[0])
    return sig_E_T[perm, :]


# -----------------------------
# Wells & segmenten (multi-file)
# -----------------------------
Well = Tuple[int, int, int, str]  # (file_idx, row, col, label)

def wells_from_file(file_idx: int, C: int, R: int) -> List[Well]:
    col2label = build_label_map_for_columns(C)
    out = []
    for r in range(R):
        for c in range(C):
            lab = col2label.get(c, "UNK")
            if lab != "UNK":
                out.append((file_idx, r, c, lab))
    return out

def filter_wells_for_training(wells: List[Well], allowed_cols_per_label: Dict[str, set]) -> List[Well]:
    keep = []
    for (fi, r, c, lab) in wells:
        if lab in allowed_cols_per_label:
            allowed = allowed_cols_per_label[lab]
            if (allowed is None) or (c in allowed):
                keep.append((fi, r, c, lab))
    return keep

def stratified_split_wells_all(wells: List[Well], seed: int, val_size: float = 0.125) -> Tuple[List[Well], List[Well]]:
    labels = np.array([lab for (_, _, _, lab) in wells])
    tr, va = train_test_split(wells, test_size=val_size, random_state=seed, stratify=labels)
    return tr, va

def make_segments_from_wells_raw_multifile(raw_list: List[np.ndarray],
                                           wells_subset: List[Well],
                                           n_segments_per_well: int,
                                           seq_len_samples: int,
                                           name2idx_out: Optional[Dict[str, int]] = None,
                                           seed: int = 42,
                                           exclusions_map: Optional[Dict[Tuple[str, int, int], set]] = None,
                                           manifest_sel: Optional[List[Dict]] = None,
                                           E_eff: Optional[int] = None,
                                           channel_shuffle: bool = False):
    rng = np.random.RandomState(seed)
    X_list, labels = [], []
    for (fi, r, c, lab) in wells_subset:
        sig = raw_list[fi][r, c, :, :]  # (E, T)
        E, T = sig.shape
        if T < seq_len_samples:
            continue
        if manifest_sel is not None:
            abs_path = manifest_sel[fi]["_abs_path"]
            key = (abs_path, r, c)
            base_excl = exclusions_map.get(key, set()) if exclusions_map is not None else set()
        else:
            base_excl = set()

        starts = rng.randint(0, T - seq_len_samples + 1, size=n_segments_per_well)
        seg_list = []
        for s in starts:
            seg = sig[:, s:s+seq_len_samples]  # (E, L)
            if E_eff is not None:
                mask = make_segment_channel_mask(E, base_excl, E_eff, rng)
                seg = apply_mask_to_segment(seg, mask)
            if channel_shuffle:
                seg = maybe_shuffle_channels(seg, rng)
            seg_list.append(seg.T)  # (L, E)
        if not seg_list:
            continue
        segs = np.stack(seg_list, axis=0)
        X_list.append(segs.astype(np.float32))
        labels += [lab] * segs.shape[0]

    if not X_list:
        raise RuntimeError("Geen segmenten gevonden (check wells/seq_len).")
    X = np.concatenate(X_list, axis=0)

    if name2idx_out is None:
        classes = sorted(list(set(labels)))
        name2idx = {n: i for i, n in enumerate(classes)}
    else:
        name2idx = name2idx_out
        classes = [None] * len(name2idx)
        for k, v in name2idx.items():
            classes[v] = k
    y_idx = np.array([name2idx[n] for n in labels], dtype=np.int32)
    return X, y_idx, classes

def make_segments_with_map_multifile(raw_list: List[np.ndarray],
                                     wells_subset: List[Well],
                                     n_segments_per_well: int,
                                     seq_len_samples: int,
                                     name2idx: Dict[str,int],
                                     seed: int = 42,
                                     exclusions_map: Optional[Dict[Tuple[str, int, int], set]] = None,
                                     manifest_sel: Optional[List[Dict]] = None,
                                     E_eff: Optional[int] = None,
                                     channel_shuffle: bool = False):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    skipped = 0
    for (fi, r, c, lab) in wells_subset:
        if lab not in name2idx:
            skipped += 1
            continue
        sig = raw_list[fi][r, c, :, :]
        E, T = sig.shape
        if T < seq_len_samples:
            continue
        if manifest_sel is not None:
            abs_path = manifest_sel[fi]["_abs_path"]
            key = (abs_path, r, c)
            base_excl = exclusions_map.get(key, set()) if exclusions_map is not None else set()
        else:
            base_excl = set()

        starts = rng.randint(0, T - seq_len_samples + 1, size=n_segments_per_well)
        seg_list = []
        for s in starts:
            seg = sig[:, s:s+seq_len_samples]
            if E_eff is not None:
                mask = make_segment_channel_mask(E, base_excl, E_eff, rng)
                seg = apply_mask_to_segment(seg, mask)
            if channel_shuffle:
                seg = maybe_shuffle_channels(seg, rng)
            seg_list.append(seg.T)
        if not seg_list:
            continue
        segs = np.stack(seg_list, axis=0)
        X_list.append(segs.astype(np.float32))
        y_list += [name2idx[lab]] * segs.shape[0]

    if not X_list:
        raise RuntimeError("Geen segmenten (mogelijk mismatch labels of te korte opname).")
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    if skipped > 0:
        print(f"[Seg] {skipped} wells overgeslagen (label niet in TRAIN mapping).")
    return X, y


# -----------------------------
# Model
# -----------------------------
def build_cnn_1d_multichan(n_classes: int, seq_len: int, n_chan: int) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len, n_chan), name="input")
    x = layers.Conv1D(64, 1, padding="same", use_bias=False, name="mix1")(inp)
    x = layers.BatchNormalization(name="bn_mix1")(x)
    x = layers.ReLU(name="relu_mix1")(x)
    for i, f in enumerate([128, 128, 64, 64], 1):
        x = layers.SeparableConv1D(f, 7 if i <= 2 else 5, padding="same",
                                   depthwise_regularizer=tf.keras.regularizers.l2(1e-5),
                                   pointwise_regularizer=tf.keras.regularizers.l2(1e-5),
                                   name=f"sepconv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.MaxPooling1D(2, name=f"pool_{i}")(x)
    for d in [1, 2, 4, 8]:
        x = layers.Conv1D(128, 5, padding="same", dilation_rate=d, use_bias=False, name=f"dil_conv_d{d}")(x)
        x = layers.BatchNormalization(name=f"bn_d{d}")(x)
        x = layers.ReLU(name=f"relu_d{d}")(x)
        x = layers.MaxPooling1D(2, name=f"pool_d{d}")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_128")(x)
    x = layers.Dropout(0.4, name="dropout_2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="mea_raw_multichan_CTRL_vs_SCA1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")]
    )
    return model


# -----------------------------
# Visuals
# -----------------------------
def plot_training_curves(history, outdir):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["accuracy"], label="Train acc")
    plt.plot(history["val_accuracy"], label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"), dpi=150); plt.close()


# -----------------------------
# Consensus-evaluatie
# -----------------------------
def consensus_eval_test_file_raw(raw_rcte_test: np.ndarray,
                                 fs: int,
                                 model: tf.keras.Model,
                                 classes_train: List[str],
                                 seq_len_ms: int,
                                 stride_ms: int,
                                 outdir: str,
                                 tag: str,
                                 abs_path: str,
                                 exclusions_map: Dict[Tuple[str, int, int], set],
                                 E_eff: Optional[int] = None,
                                 channel_shuffle: bool = False):
    R, C, E, T = raw_rcte_test.shape
    seq_len = int(round(seq_len_ms * fs / 1000.0))
    stride  = max(1, int(round(stride_ms * fs / 1000.0)))

    col2label = build_label_map_for_columns(C)
    allowed_labels = set(classes_train)

    rng = np.random.RandomState(cfg.seed + 999)  # vaste seed voor deterministische consensus

    y_true, y_pred = [], []
    rows_out = [("row", "col", "true_label", "pred_label")]

    for r in range(R):
        for c in range(C):
            lab = col2label.get(c, "UNK")
            if lab == "UNK" or lab not in allowed_labels:
                continue
            sig = raw_rcte_test[r, c, :, :]  # (E, T)
            if sig.shape[1] < seq_len:
                continue
            starts = list(range(0, sig.shape[1] - seq_len + 1, stride))
            if not starts:
                continue

            key = (abs_path, r, c)
            base_excl = exclusions_map.get(key, set())

            seg_list = []
            for s in starts:
                seg = sig[:, s:s+seq_len]
                if E_eff is not None:
                    mask = make_segment_channel_mask(E, base_excl, E_eff, rng)
                    seg = apply_mask_to_segment(seg, mask)
                if channel_shuffle:
                    seg = maybe_shuffle_channels(seg, rng)
                seg_list.append(seg.T)
            batch = np.stack(seg_list, axis=0).astype(np.float32)  # (B, L, E)

            probs = model.predict(batch, verbose=0)
            votes = probs.argmax(axis=1)
            pred_idx = np.bincount(votes, minlength=len(classes_train)).argmax()
            y_true.append(lab)
            y_pred.append(classes_train[pred_idx])
            rows_out.append((r, c, lab, classes_train[pred_idx]))

    if not y_true:
        print(f"[{tag}] Geen valide wells voor consensus.")
        return

    cls_sorted = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=cls_sorted, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=cls_sorted)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    print(f"\n=== Consensus per well ({tag}) ===")
    print(report)
    print("Labels volgorde:", cls_sorted)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cls_sorted, yticklabels=cls_sorted)
    plt.title(f"Confusion Matrix – {tag} (Counts)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_matrix_{tag}_counts.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=cls_sorted, yticklabels=cls_sorted)
    plt.title(f"Confusion Matrix – {tag} (Row-Normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_matrix_{tag}_normalized.png"), dpi=150)
    plt.close()

    with open(os.path.join(outdir, f"consensus_report_{tag}.txt"), "w") as f:
        f.write(report + "\nLabels (order): " + ", ".join(cls_sorted) + "\n")
        f.write("Confusion (rows=true, cols=pred):\n")
        f.write(np.array2string(cm, separator=", "))

    with open(os.path.join(outdir, f"per_well_consensus_{tag}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_out)


# -----------------------------
# Resume helpers
# -----------------------------
def save_classes(classes: List[str], path: str = CLASSES_PATH):
    with open(path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

def load_classes(path: str = CLASSES_PATH) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return list(data.get("classes", []))

def load_or_build_model(n_classes: int, seq_len: int, n_chan: int) -> tf.keras.Model:
    if os.path.exists(CKPT_PATH) and os.path.exists(CLASSES_PATH):
        print(f"[Resume] Laad bestaand model: {CKPT_PATH}")
        model = tf.keras.models.load_model(CKPT_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")]
        )
        exp_in = (None, seq_len, n_chan)
        if model.input_shape != exp_in:
            raise ValueError(f"Incompatibele input shape. Verwacht {exp_in}, kreeg {model.input_shape}.")
        return model
    print("[Init] Geen bestaand model, start from scratch.")
    return build_cnn_1d_multichan(n_classes=n_classes, seq_len=seq_len, n_chan=n_chan)


# -----------------------------
# Train pipeline
# -----------------------------
def train():
    # Manifest + split
    manifest = load_manifest(cfg.manifest_path)
    trainval_files, test_files = random_split_files(manifest, seed=cfg.seed,
                                                    n_trainval=cfg.n_trainval_files,
                                                    n_test=cfg.n_test_files)

    # RAW laden
    raw_list_trainval, raw_list_test = [], []
    E_ref = None
    for rec in trainval_files:
        arr = load_raw_as_rcte(rec["_abs_path"])
        if E_ref is None: E_ref = arr.shape[2]
        assert arr.shape[:2] == (cfg.rows, cfg.cols)
        assert arr.shape[2] == E_ref
        raw_list_trainval.append(arr)
    for rec in test_files:
        arr = load_raw_as_rcte(rec["_abs_path"])
        assert arr.shape[:2] == (cfg.rows, cfg.cols)
        assert arr.shape[2] == E_ref
        raw_list_test.append(arr)

    # Downsample (optioneel) + fs
    fs = cfg.sampling_hz
    for i in range(len(raw_list_trainval)):
        raw_list_trainval[i], fs_tr = maybe_downsample(raw_list_trainval[i], cfg.sampling_hz, cfg.target_hz)
        fs = fs_tr
    for i in range(len(raw_list_test)):
        raw_list_test[i], fs_te = maybe_downsample(raw_list_test[i], cfg.sampling_hz, cfg.target_hz)
        assert fs_te == fs

    # Normalisatie
    if cfg.norm_per_electrode:
        raw_list_trainval = [normalize_per_electrode(x) for x in raw_list_trainval]
        raw_list_test     = [normalize_per_electrode(x) for x in raw_list_test]

    # QC-exclusions
    exclusions = load_qc_exclusions(cfg.inactive_csv_path, manifest,
                                    rows=cfg.rows, cols=cfg.cols,
                                    well_base=cfg.well_index_base,
                                    elec_base=cfg.electrode_index_base)

    # QC op raw toepassen (zodat consensus-eval het ook ziet)
    for i, rec in enumerate(trainval_files):
        apply_exclusions_to_raw_rcte(raw_list_trainval[i], rec["_abs_path"], exclusions)
    for j, rec in enumerate(test_files):
        apply_exclusions_to_raw_rcte(raw_list_test[j], rec["_abs_path"], exclusions)

    # Channel budget bepalen
    E = E_ref
    if cfg.constant_channel_budget:
        E_eff_tr = compute_E_eff(E, cfg.rows, cfg.cols, trainval_files, exclusions)
        E_eff_te = compute_E_eff(E, cfg.rows, cfg.cols, test_files, exclusions)
        E_eff = min(E_eff_tr, E_eff_te)
    else:
        E_eff_tr = E_eff_te = E_eff = E
    print(f"[Channels] E={E} | E_eff(train)={E_eff_tr} | E_eff(test)={E_eff_te} | gebruikt E_eff={E_eff}")

    # Well-sets
    wells_all_trainval: List[Well] = []
    for i in range(len(raw_list_trainval)):
        wells_all_trainval += wells_from_file(i, cfg.cols, cfg.rows)
    wells_all_test: List[Well] = []
    for j in range(len(raw_list_test)):
        wells_all_test += wells_from_file(j, cfg.cols, cfg.rows)

    wells_train_all, wells_val = stratified_split_wells_all(wells_all_trainval, seed=cfg.seed, val_size=0.125)
    allowed_cols_train = get_allowed_cols_per_label(cfg.cols)
    wells_train = filter_wells_for_training(wells_train_all, allowed_cols_per_label=allowed_cols_train)

    # Segmentlengte
    seq_len_samples = int(round(cfg.seq_len_ms * fs / 1000.0))

    # Segmenten TRAIN
    Xtr, ytr, classes_tr = make_segments_from_wells_raw_multifile(
        raw_list_trainval, wells_train, cfg.train_segments_per_well, seq_len_samples,
        seed=cfg.seed,
        exclusions_map=exclusions, manifest_sel=trainval_files,
        E_eff=(E_eff if cfg.constant_channel_budget else None),
        channel_shuffle=cfg.channel_shuffle
    )
    print("TRAIN classes (sorted):", classes_tr)
    name2idx = {n: i for i, n in enumerate(classes_tr)}

    # Segmenten VAL
    Xva, yva = make_segments_with_map_multifile(
        raw_list_trainval, wells_val, cfg.val_segments_per_well, seq_len_samples, name2idx,
        seed=cfg.seed+1,
        exclusions_map=exclusions, manifest_sel=trainval_files,
        E_eff=(E_eff if cfg.constant_channel_budget else None),
        channel_shuffle=False
    )

    # Snelle segment-sanity op TEST (samengevoegd)
    Xte_list, yte_list = [], []
    for j in range(len(raw_list_test)):
        Xte_j, yte_j = make_segments_with_map_multifile(
            [raw_list_test[j]], wells_from_file(0, cfg.cols, cfg.rows),
            cfg.test_segments_per_well, seq_len_samples, name2idx,
            seed=cfg.seed+2+j,
            exclusions_map=exclusions, manifest_sel=[test_files[j]],
            E_eff=(E_eff if cfg.constant_channel_budget else None),
            channel_shuffle=False
        )
        Xte_list.append(Xte_j); yte_list.append(yte_j)
    Xte = np.concatenate(Xte_list, axis=0) if Xte_list else np.zeros((0, seq_len_samples, E), dtype=np.float32)
    yte = np.concatenate(yte_list, axis=0) if yte_list else np.zeros((0,), dtype=np.int32)

    print("TRAIN X:", Xtr.shape, "VAL X:", Xva.shape, "TEST(segm) X:", Xte.shape, "classes:", classes_tr)

    # tf.data
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(min(10000, Xtr.shape[0]), seed=cfg.seed).batch(cfg.batch_size).prefetch(AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(cfg.batch_size).prefetch(AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((Xte, yte)).batch(cfg.batch_size).prefetch(AUTOTUNE) if Xte.shape[0] > 0 else None

    # Resume / model
    old = load_classes()
    if old is not None:
        if sorted(old) != sorted(classes_tr):
            raise ValueError(f"Classes mismatch tussen opgeslagen {old} en huidige {classes_tr}.")
        classes_tr = old
        name2idx = {n: i for i, n in enumerate(classes_tr)}
    else:
        save_classes(classes_tr)

    model = load_or_build_model(len(classes_tr), seq_len_samples, raw_list_trainval[0].shape[2])

    # Callbacks & class weights
    cbs = [
        callbacks.ModelCheckpoint(CKPT_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=cfg.patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]
    cls_indices = np.arange(len(classes_tr))
    cw = compute_class_weight(class_weight="balanced", classes=cls_indices, y=ytr)
    class_weight = {int(i): float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weight)

    # Train
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=cbs, class_weight=class_weight)

    # Save + visuals
    model.save(LAST_PATH)
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}
    with open(os.path.join(cfg.outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, cfg.outdir)

    # Segment sanity test
    if test_ds is not None and Xte.shape[0] > 0:
        seg_loss, seg_acc, seg_top2 = model.evaluate(test_ds, verbose=0)
        print(f"[Segment] TEST(all) acc = {seg_acc:.4f} | top2 = {seg_top2:.4f}")

    # Consensus per testbestand (met zelfde E_eff + QC-mask)
    for j, rec in enumerate(test_files):
        tag = f"TEST_{j+1}_{os.path.basename(rec.get('filename','test.h5'))}"
        consensus_eval_test_file_raw(
            raw_list_test[j], fs, model, classes_tr, cfg.seq_len_ms, cfg.stride_ms_eval,
            cfg.outdir, tag, abs_path=rec["_abs_path"],
            exclusions_map=exclusions,
            E_eff=(E_eff if cfg.constant_channel_budget else None),
            channel_shuffle=False
        )

    print("Klaar. Output in:", cfg.outdir)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    train()
