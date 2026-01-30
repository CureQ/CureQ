"""MEA 1D-CNN training + evaluation pipeline (Axion H5).

This script runs repeated training rounds on a set of MEA recordings (H5), with:
- A fixed global train/val pool and a separate "final" file-level test split.
- Preprocessing aligned with CNN training (raw traces -> (6x8 wells x 16 electrodes),
  optional downsampling, robust MAD normalization, and masking of overactive electrodes
  based on a spike-database CSV).
- Per-well segmentation for training/validation (memmap cache to save RAM).
- A 1D-CNN model (Conv/SeparableConv + pooling) with early stopping and class weights.
- Consensus evaluation at well level on the final test files: per well, multiple segment
  predictions are aggregated via majority vote, including ROC-AUC on p(SCA1).

Outputs:
- Per round: best_model.keras, last_model.keras, history.json, loss/accuracy curves,
  confusion matrix, classification report and ROC curve.
- Across all rounds: overall metrics/plots + round_summaries (CSV/JSON).
"""
import os
import json, csv, gc, shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, callbacks
print('[System] TensorFlow Devices:', tf.config.list_physical_devices())

@dataclass
class Config:
    """Configuration for data loading, preprocessing, training and evaluation.
    
    Variables are grouped by pipeline component to make it clear which settings belong together.
    """

    # ----------------------------
    # Input files and data sources
    # ----------------------------
    # File locations for reading the dataset and (optional) spike information.
    manifest_path: str = './manifest.json'
    spike_csv_path: str = './analyses/activity_analyzed/measurements_spikes_bursts/electrode_spikes_over_measurement.csv'

    # ----------------------------
    # Dataset splits and round planning
    # ----------------------------
    # Settings for selecting train/val files per round and the fixed
    # hold-out file-level test split.
    n_trainval_files: int = 11
    n_test_files: int = 2
    min_trainval_files: int = 4
    final_test_fraction: float = 0.2
    max_rounds: Optional[int] = 0
    seed: int = 51

    # ----------------------------
    # Plate layout and labels
    # ----------------------------
    # Well-plate layout and conventions for column-to-label mapping (CTRL vs SCA1).
    rows: int = 6
    cols: int = 8
    use_only_one_ctrl_line: bool = False
    ctrl_line_side: str = 'left'
    pos_label: str = 'SCA1'

    # ----------------------------
    # Raw data read settings (H5)
    # ----------------------------
    # Parameters to locate the raw MEA time series in Axion H5 and reshape correctly.
    key_raw: str = 'Data/Recording_0/AnalogStream/Stream_0/ChannelData'
    channel_layout: str = 'well_major'
    sampling_hz: int = 20000

    # ----------------------------
    # Preprocessing: masking, downsampling and normalization
    # ----------------------------
    # - Masking: overactive electrodes based on spike counts (if spike_db is available).
    # - Downsampling: optionally reduce to target_hz.
    # - Normalization: robust median/MAD normalization per electrode.
    spike_cutoff_hz: float = 13.0
    use_downsample: bool = True
    target_hz: int = 4000
    norm_per_electrode: bool = True

    # ----------------------------
    # Segmentation and evaluation windows
    # ----------------------------
    # Segment lengths (ms) are converted to samples based on (downsampled) fs.
    seq_len_ms: int = 4000
    train_segments_per_well: int = 90
    val_segments_per_well: int = 30
    stride_ms_eval: int = 4000

    # ----------------------------
    # Model training (hyperparameters)
    # ----------------------------
    # Core settings for optimization, batching, and early stopping.
    batch_size: int = 24
    lr: float = 0.0001
    epochs: int = 500
    patience: int = 20

    # ----------------------------
    # Regularization and augmentation
    # ----------------------------
    # Channel shuffle: permute electrode order per training segment as regularization.
    channel_shuffle: bool = True

    # ----------------------------
    # Output and logging
    # ----------------------------
    # Output folder and paths for saving/reusing a fixed global split.
    outdir: str = 'output_170126'
    global_split_path: str = 'BT_cureq-data/Chen/global_split/global_split.json'

    # ----------------------------
    # Numerical safety and stability
    # ----------------------------
    # Clip values limit extreme outliers; mad_floor prevents division by (near) zero.
    raw_clip_abs: float = 1000000.0
    norm_clip_abs: float = 20.0
    mad_floor: float = 1e-06

    # ----------------------------
    # Memory and performance settings
    # ----------------------------
    # Memmap cache reduces RAM usage by storing segments on disk.
    use_memmap_cache: bool = True
    cache_dir_name: str = '_memmap_cache'
    tf_shuffle_buffer: int = 5000
    eval_pred_chunk: int = 64

cfg = Config()
os.makedirs(cfg.outdir, exist_ok=True)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)
GLOBAL_SPLIT_PATH = os.path.abspath(cfg.global_split_path)
USED_TRAINVAL_LOG = os.path.join(cfg.outdir, 'used_trainval_ids.json')
GLOBAL_BEST_PATH = os.path.join(cfg.outdir, 'best_model_GLOBAL.keras')
AGG: Dict[str, Any] = {'y_true': [], 'y_pred': [], 'score_vote': [], 'classes': None, 'round_summaries': []}

def load_spike_database(csv_path: str) -> Dict[Tuple[str, int, int], int]:
    """Read the spike-database CSV.
    
    The CSV contains spike counts per (well, electrode) across multiple measurements. The header
    is expected to contain measurement IDs starting from column 3. The output is a dict with keys
    (meas_id, well, elec) and as value the spike count.
    
    Returns:
        Dict[(meas_id, well, elec) -> spike_count]
    """
    if not os.path.exists(csv_path):
        print(f'[Warning] Spike CSV niet gevonden: {csv_path}. Er wordt niets geexcludeerd.')
        return {}
    spike_db = {}
    print(f'[Data] Spike CSV inlezen: {csv_path}')
    with open(csv_path, 'r', encoding='utf-8') as f:
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
                well = int(float(row[0]))
                elec = int(float(row[1]))
                counts = row[2:]
                for m_idx, val_str in enumerate(counts):
                    if m_idx < len(meas_ids):
                        meas_id = meas_ids[m_idx]
                        try:
                            val = int(float(val_str))
                        except Exception:
                            val = 0
                        spike_db[meas_id, well, elec] = val
                        count_records += 1
            except Exception:
                pass
    print(f'[Data] {count_records} spike-waarden ingeladen voor {len(meas_ids)} metingen.')
    return spike_db

def mask_noisy_electrodes_single(raw: np.ndarray, file_rec: Dict, spike_db: Dict, cutoff_hz: float, fs: int):
    """Mask overactive electrodes in a recording.
    
    For each electrode, the firing rate is estimated from spike counts in spike_db and the
    recording duration (T/fs). Electrodes above cfg.spike_cutoff_hz are set to 0.
    
    Args:
        raw: Array (rows, cols, electrodes, time)
        file_rec: Manifest record containing e.g. 'id'
        spike_db: Dict with spike counts
        cutoff_hz: Threshold in Hz
        fs: Sampling rate (Hz)
    """
    if not spike_db:
        return
    meas_id = str(file_rec.get('id', ''))
    R, C, E, T = raw.shape
    duration_sec = max(T / fs, 1e-09)
    masked_count = 0
    for r in range(R):
        for c in range(C):
            well_id = r * cfg.cols + (c + 1)
            for e_idx in range(E):
                elec_id = e_idx + 1
                key = (meas_id, well_id, elec_id)
                if key in spike_db:
                    hz = spike_db[key] / duration_sec
                    if hz > cutoff_hz:
                        raw[r, c, e_idx, :] = 0.0
                        masked_count += 1
    if masked_count > 0:
        print(f'      -> Totaal: {masked_count} elektroden op 0 gezet in dit bestand.')

def load_manifest(path: str) -> List[Dict]:
    """Load manifest.json and filter to existing paths.
    
    The manifest must contain a path per record via 'path'/'path_rel'/'filename'.
    For each valid record, an '_abs_path' field is added.
    
    Returns:
        List of valid manifest records.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Manifest niet gevonden: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    valid = []
    for rec in data:
        p = rec.get('path') or rec.get('path_rel') or rec.get('filename')
        if not p:
            continue
        p_abs = p if os.path.isabs(p) else os.path.abspath(p)
        if os.path.exists(p_abs):
            rec['_abs_path'] = p_abs
            valid.append(rec)
    return valid

def load_used_trainval_ids(path: str=USED_TRAINVAL_LOG) -> set:
    """Load the log file containing already-used train/val IDs.
    
    Returns:
        Set of used IDs (int).
    """
    if not os.path.exists(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set((int(x) for x in json.load(f).get('used_trainval_ids', [])))
    except Exception:
        return set()

def save_used_trainval_ids(used_ids: set, path: str=USED_TRAINVAL_LOG) -> None:
    """Save the set of used train/val IDs to JSON.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'used_trainval_ids': sorted((int(x) for x in used_ids))}, f, indent=2)

def build_or_load_global_split(manifest: List[Dict], final_test_fraction: float=0.2) -> Tuple[List[Dict], List[Dict]]:
    """Create or load a global file-level split.
    
    - If cfg.global_split_path exists: load trainval_ids and final_test_ids.
    - Otherwise: create a new split using cfg.final_test_fraction and save it.
    
    Returns:
        (pool_records, final_records)
    """
    if os.path.exists(GLOBAL_SPLIT_PATH):
        print(f'[GlobalSplit] Laden: {GLOBAL_SPLIT_PATH}')
        with open(GLOBAL_SPLIT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        final_ids, train_ids = (set(data['final_test_ids']), set(data['trainval_ids']))
    else:
        print(f'[GlobalSplit] Niet gevonden -> nieuwe split maken en opslaan naar: {GLOBAL_SPLIT_PATH}')
        ids_all = sorted(set((int(rec['id']) for rec in manifest if rec.get('id') is not None)))
        rng = np.random.RandomState(cfg.seed)
        idx = np.arange(len(ids_all))
        rng.shuffle(idx)
        n_final = max(1, int(round(len(ids_all) * final_test_fraction)))
        final_ids = {ids_all[i] for i in idx[:n_final]}
        train_ids = {ids_all[i] for i in idx[n_final:]}
        data = {'final_test_fraction': final_test_fraction, 'final_test_ids': sorted(final_ids), 'trainval_ids': sorted(train_ids)}
        os.makedirs(os.path.dirname(GLOBAL_SPLIT_PATH), exist_ok=True)
        with open(GLOBAL_SPLIT_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    id2rec = {int(rec['id']): rec for rec in manifest if rec.get('id') is not None}
    pool = [id2rec[i] for i in train_ids if i in id2rec]
    final = [id2rec[i] for i in final_ids if i in id2rec]
    return (pool, final)

def pick_next_train_files(pool: List[Dict], seed: int, n_trainval: int) -> Tuple[List[Dict], bool]:
    """Select the next set of train/val files from the pool.
    
    The selection avoids IDs that have already been used (USED_TRAINVAL_LOG). If fewer than
    cfg.min_trainval_files remain, a stop condition is returned.
    
    Returns:
        (trainval_files, done)
    """
    used_ids = load_used_trainval_ids()
    pool_unused = [rec for rec in pool if rec.get('id') not in used_ids]
    remaining = len(pool_unused)
    if remaining < cfg.min_trainval_files:
        print(f'[Stop] Nog {remaining} train/val bestanden over, dit is minder dan het minimum ({cfg.min_trainval_files}). Training stopt.')
        return ([], True)
    n_train_eff = min(n_trainval, remaining)
    rng = np.random.RandomState(seed)
    idx = np.arange(remaining)
    rng.shuffle(idx)
    trainval_files = [pool_unused[i] for i in idx[:n_train_eff]]
    for rec in trainval_files:
        if rec.get('id') is not None:
            used_ids.add(int(rec['id']))
    save_used_trainval_ids(used_ids)
    return (trainval_files, False)

def pick_test_files_from_final(final: List[Dict], seed: int, n_test: int) -> List[Dict]:
    """Choose (randomly) a subset of test files from the final split.
    """
    if not final:
        return []
    n_test_eff = min(n_test, len(final))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(final))
    rng.shuffle(idx)
    return [final[i] for i in idx[:n_test_eff]]

def build_label_map_for_columns(cols: int) -> Dict[int, str]:
    """Map plate columns to labels (CTRL/SCA1/UNK).
    
    This mapping encodes the experimental layout: which columns belong to which cell line.
    """
    lab = {}
    for c in range(cols):
        if c in (0, 1, 4, 5):
            lab[c] = 'SCA1'
        elif c in (2, 3, 6, 7):
            lab[c] = 'CTRL'
        else:
            lab[c] = 'UNK'
    return lab

def get_allowed_cols_per_label(cols: int) -> Dict[str, set]:
    """Determine which columns are allowed per label.
    
    Respects cfg.use_only_one_ctrl_line and cfg.ctrl_line_side to optionally include only one
    control line.
    """
    col2label = build_label_map_for_columns(cols)
    all_cols = {}
    for c in range(cols):
        lab = col2label.get(c, 'UNK')
        if lab != 'UNK':
            all_cols.setdefault(lab, set()).add(c)
    if cfg.use_only_one_ctrl_line and 'CTRL' in all_cols:
        side = {2, 3} if cfg.ctrl_line_side == 'left' else {6, 7}
        all_cols['CTRL'] &= side
    return all_cols

def load_raw_as_rcte(h5_path: str) -> np.ndarray:
    """Read raw ChannelData from H5 and reshape to (rows, cols, electrodes, time).
    
    - Reads cfg.key_raw.
    - Converts NaN/Inf -> 0 and clips to cfg.raw_clip_abs.
    - Reshaping follows cfg.channel_layout.
    
    Returns:
        np.ndarray float32 with shape (rows, cols, electrodes, time)
    """
    with h5py.File(h5_path, 'r') as f:
        raw = f[cfg.key_raw][()]
    raw = np.asarray(raw, dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, -cfg.raw_clip_abs, cfg.raw_clip_abs)
    ch, T = raw.shape
    R, C = (cfg.rows, cfg.cols)
    if cfg.channel_layout == 'well_major':
        out = raw.reshape(R * C, ch // (R * C), T).reshape(R, C, ch // (R * C), T)
    else:
        out = np.transpose(raw.reshape(ch // (R * C), R * C, T), (1, 0, 2)).reshape(R, C, ch // (R * C), T)
    return out.astype(np.float32, copy=False)

def maybe_downsample(raw: np.ndarray, fs: int, target: int) -> tuple:
    """Downsample a recording to cfg.target_hz (optional).
    
    Uses resample_poly for rational resampling. After resampling, NaN/Inf are removed and the
    data is clipped again.
    
    Returns:
        (downsampled_raw, new_fs)
    """
    if not cfg.use_downsample or target >= fs:
        return (raw, fs)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(fs, target)
    p, q = (target // g, fs // g)
    R, C, E, T = raw.shape
    out = np.empty((R, C, E, int(np.ceil(T * p / q))), dtype=np.float32)
    for r in range(R):
        for c in range(C):
            out[r, c] = resample_poly(raw[r, c], p, q, axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    out = np.clip(out, -cfg.raw_clip_abs, cfg.raw_clip_abs).astype(np.float32, copy=False)
    return (out, target)

def normalize_per_electrode(raw: np.ndarray) -> np.ndarray:
    """Robust per-electrode normalization based on median and MAD.
    
    For each electrode, (x - median) / (1.4826*MAD) is computed. MAD is lower-bounded by
    cfg.mad_floor to avoid instability. Output is clipped to cfg.norm_clip_abs.
    """
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    med = np.median(raw, axis=-1, keepdims=True)
    mad = np.median(np.abs(raw - med), axis=-1, keepdims=True)
    denom = 1.4826 * np.maximum(mad, cfg.mad_floor) + 1e-07
    x = (raw - med) / denom
    x = np.clip(x, -cfg.norm_clip_abs, cfg.norm_clip_abs)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return x.astype(np.float32, copy=False)

def wells_from_file(file_idx: int, C: int, R: int) -> List[Tuple]:
    """Generate (file_idx, row, col, label) tuples for all wells in a file.
    """
    col2lab = build_label_map_for_columns(C)
    return [(file_idx, r, c, col2lab[c]) for r in range(R) for c in range(C) if col2lab.get(c, 'UNK') != 'UNK']

def filter_wells(wells: List[Tuple], allowed: Dict[str, set]) -> List[Tuple]:
    """Filter wells based on the allowed columns per label.
    """
    return [w for w in wells if w[3] in allowed and (allowed[w[3]] is None or w[2] in allowed[w[3]])]

def process_file_list(files: List[Dict], spike_db: Dict, desc: str) -> Tuple[List[np.ndarray], int]:
    """Load and preprocess a list of H5 files.
    
    Pipeline per file:
    1) load_raw_as_rcte()
    2) mask_noisy_electrodes_single() using spike_db
    3) maybe_downsample()
    4) normalize_per_electrode() (optional)
    
    Returns:
        (processed_raw_list, final_fs)
    """
    processed_list = []
    final_fs = cfg.sampling_hz
    print(f'\n--- Start Processing {desc} ---')
    for i, f in enumerate(files):
        fname = f.get('filename', 'Unknown')
        print(f'[{i + 1}/{len(files)}] Loading: {fname} ...', end=' ', flush=True)
        try:
            raw = load_raw_as_rcte(f['_abs_path'])
        except Exception as e:
            print(f'FAILED: {e}')
            continue
        if not np.isfinite(raw).all():
            print('FAILED: non-finite values after load -> skipped')
            continue
        mask_noisy_electrodes_single(raw, f, spike_db, cfg.spike_cutoff_hz, cfg.sampling_hz)
        raw, final_fs = maybe_downsample(raw, cfg.sampling_hz, cfg.target_hz)
        if cfg.norm_per_electrode:
            raw = normalize_per_electrode(raw)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        processed_list.append(raw)
        gc.collect()
        print('Done.')
    return (processed_list, final_fs)

def _ensure_dir(p: str):
    """Create a directory if it does not yet exist.
    """
    os.makedirs(p, exist_ok=True)

def make_segments_to_memmap(raw_list: List[np.ndarray], wells: List[Tuple], n_seg: int, seq_len: int, out_x_path: str, out_y_path: str, name2idx: Optional[Dict[str, int]]=None, seed: int=42, shuffle_channels: bool=False) -> Tuple[int, List[str], Dict[str, int], Tuple[int, int, int]]:
    """Write segmented data to memmap files.
    
    For each valid well, n_seg segments of length seq_len are drawn. Segments are written to
    X (float32) and labels to y (int32). This avoids high RAM usage for large datasets.
    
    Args:
        raw_list: Preprocessed recordings (rows, cols, electrodes, time)
        wells: List of (file_idx, row, col, label)
        n_seg: Number of segments per well
        seq_len: Segment length in samples
        out_x_path/out_y_path: Files for memmap storage
        name2idx: Optional fixed class-index mapping (consistent between train and val)
        shuffle_channels: Optionally permute electrode order per segment (regularization)
    
    Returns:
        (N, classes, name2idx, x_shape)
    """
    rng = np.random.RandomState(seed)
    eligible = []
    E_ref = None
    for fi, r, c, lab in wells:
        sig = raw_list[fi][r, c]
        if E_ref is None:
            E_ref = sig.shape[0]
        if sig.shape[0] != E_ref:
            continue
        if np.all(sig == 0):
            continue
        if sig.shape[1] < seq_len:
            continue
        eligible.append((fi, r, c, lab))
    if not eligible:
        raise RuntimeError('Geen segmenten: geen geldige wells (alles te kort of gemaskeerd?)')
    if name2idx is None:
        classes = sorted(list(set([lab for *_, lab in eligible])))
        name2idx = {n: i for i, n in enumerate(classes)}
    else:
        classes = [k for k, v in sorted(name2idx.items(), key=lambda x: x[1])]
    N = len(eligible) * int(n_seg)
    x_shape = (N, int(seq_len), int(E_ref))
    Xmm = np.memmap(out_x_path, dtype=np.float32, mode='w+', shape=x_shape)
    ymm = np.memmap(out_y_path, dtype=np.int32, mode='w+', shape=(N,))
    write_idx = 0
    for fi, r, c, lab in eligible:
        sig = raw_list[fi][r, c]
        T = sig.shape[1]
        starts = rng.randint(0, T - seq_len + 1, size=n_seg)
        batch = np.empty((n_seg, seq_len, E_ref), dtype=np.float32)
        for j, s in enumerate(starts):
            seg = sig[:, s:s + seq_len]
            if shuffle_channels:
                seg = seg[rng.permutation(E_ref), :]
            batch[j] = seg.T
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        Xmm[write_idx:write_idx + n_seg] = batch
        ymm[write_idx:write_idx + n_seg] = int(name2idx[lab])
        write_idx += n_seg
    Xmm.flush()
    ymm.flush()
    del Xmm, ymm
    return (N, classes, name2idx, x_shape)

class MemmapBatchLoader:

    def __init__(self, x_path: str, y_path: str, x_shape: Tuple[int, int, int]):
        """Initialize a memmap loader.
        
        Args:
            x_path: Path to memmap with X (float32)
            y_path: Path to memmap with y (int32)
            x_shape: Shape of X (N, seq_len, n_chan)
        """
        self.x_path = x_path
        self.y_path = y_path
        self.x_shape = x_shape
        self._X = None
        self._y = None

    def _open(self):
        """Open memmap files lazily (only when the first batch is read).
        """
        if self._X is None:
            self._X = np.memmap(self.x_path, dtype=np.float32, mode='r', shape=self.x_shape)
        if self._y is None:
            self._y = np.memmap(self.y_path, dtype=np.int32, mode='r', shape=(self.x_shape[0],))

    def read_batch(self, idx_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Read a batch of indices from memmap and sanitize NaN/Inf.
        """
        self._open()
        idx_batch = np.asarray(idx_batch, dtype=np.int64)
        Xb = self._X[idx_batch]
        yb = self._y[idx_batch]
        Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
        return (Xb.astype(np.float32, copy=False), yb.astype(np.int32, copy=False))

    def close(self):
        """Close the loader by releasing memmap references.
        """
        self._X = None
        self._y = None

def dataset_from_memmap(x_path: str, y_path: str, x_shape: Tuple[int, int, int], batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    """Create a tf.data.Dataset that reads batches from memmap files.
    
    Reads indices from a range dataset, shuffles if training=True, and loads batches via
    tf.py_function from MemmapBatchLoader.
    """
    N = int(x_shape[0])
    loader = MemmapBatchLoader(x_path, y_path, x_shape)

    def _load_batch(idx_batch_tf):
        Xb, yb = tf.py_function(func=lambda x: loader.read_batch(x), inp=[idx_batch_tf], Tout=[tf.float32, tf.int32])
        Xb.set_shape((None, x_shape[1], x_shape[2]))
        yb.set_shape((None,))
        return (Xb, yb)
    ds = tf.data.Dataset.range(N)
    if training:
        ds = ds.shuffle(buffer_size=min(cfg.tf_shuffle_buffer, N), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(_load_batch, num_parallel_calls=1)
    ds = ds.prefetch(1)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds._memmap_loader = loader
    return ds

def build_model(n_classes, seq_len, n_chan):
    """Build and compile the 1D-CNN model.
    
    Architecture:
    - Input: (seq_len, n_chan)
    - Conv/SeparableConv blocks + pooling
    - Dilated conv blocks
    - Dropout + GlobalAveragePooling
    - Dense + softmax
    
    Returns:
        Compiled tf.keras.Model.
    """
    inp = layers.Input(shape=(seq_len, n_chan))
    x = layers.Conv1D(16, 1, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for f in [16, 32]:
        x = layers.SeparableConv1D(f, 5, padding='same', depthwise_regularizer=tf.keras.regularizers.l2(0.001), pointwise_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    for d in [1, 2]:
        x = layers.Conv1D(32, 5, padding='same', dilation_rate=d, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_curves(hist, out, tag):
    """Plot and save training curves (loss/accuracy) as PNG.
    """
    for metric in ['loss', 'accuracy']:
        data_train = hist.get(metric, [])
        data_val = hist.get(f'val_{metric}', [])
        if not data_train:
            print(f'[Plot] Geen data voor {metric} in {tag}')
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(data_train, label=f'Train {metric}')
        if data_val:
            plt.plot(data_val, label=f'Val {metric}')
        plt.title(f'{metric.title()} ({tag})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'{metric}_{tag}.png'))
        plt.close()

def _sanitize_binary_roc_inputs(y_true_bin: np.ndarray, y_score: np.ndarray, tag: str):
    """Sanitize inputs for binary ROC (filter non-finite, clip scores).
    """
    y_true_bin = np.asarray(y_true_bin, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    mask = np.isfinite(y_score)
    if not np.all(mask):
        n_bad = int(np.sum(~mask))
        print(f'[ROC] {tag}: {n_bad} non-finite scores gevonden -> gefilterd.')
        y_true_bin = y_true_bin[mask]
        y_score = y_score[mask]
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    y_score = np.clip(y_score, 0.0, 1.0)
    return (y_true_bin, y_score)

def plot_roc_binary(y_true_bin: np.ndarray, y_score: np.ndarray, outdir: str, tag: str) -> Optional[float]:
    """Compute and plot ROC curve + AUC for binary classification.
    
    Also writes a text file containing the AUC.
    
    Returns:
        AUC (float) or None when ROC cannot be computed.
    """
    y_true_bin, y_score = _sanitize_binary_roc_inputs(y_true_bin, y_score, tag)
    if y_true_bin.size == 0:
        print(f'[ROC] {tag}: geen geldige samples na filtering -> overgeslagen.')
        return None
    if len(np.unique(y_true_bin)) < 2:
        print(f'[ROC] {tag}: maar 1 class in y_true -> ROC-AUC overgeslagen.')
        return None
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({tag})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'roc_{tag}.png'))
    plt.close()
    with open(os.path.join(outdir, f'roc_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'AUC: {roc_auc:.6f}\n')
    return float(roc_auc)

def consensus_eval_multifile(raw_list_test, test_files, fs, model, classes_train, seq_len_ms, stride_ms, outdir, round_tag) -> Optional[Dict[str, Any]]:
    """Run well-level consensus evaluation across multiple test files.
    
    Per well:
    - Sliding-window segments are generated using seq_len_ms and stride_ms.
    - Model predicts per chunk (cfg.eval_pred_chunk) to limit memory usage.
    - Majority vote yields the hard label; p(pos_label) = fraction of segments voting pos_label.
    
    Outputs:
    - classification_report, confusion-matrix plot (Blues), per-well CSV, ROC curve.
    
    Returns:
        Dict with evaluation statistics and per-well outputs, or None if there is insufficient data.
    """
    seq_len = int(round(seq_len_ms * fs / 1000.0))
    stride = max(1, int(round(stride_ms * fs / 1000.0)))
    col2label = build_label_map_for_columns(cfg.cols)
    allowed = set(classes_train)
    if cfg.pos_label not in classes_train:
        print(f"[Evaluatie] {round_tag}: pos_label '{cfg.pos_label}' niet in classes_train {classes_train} -> overgeslagen.")
        return None
    pos_idx = classes_train.index(cfg.pos_label)
    y_true, y_pred = ([], [])
    score_pos: List[float] = []
    rows_out = [('filename', 'row', 'col', 'true_label', 'pred_label', 'score_pos')]
    print(f'\n[Evaluatie] Start consensus over {len(test_files)} testbestanden (FINAL split)...')
    n_skipped_nonfinite = 0
    for i, raw in enumerate(raw_list_test):
        fname = os.path.basename(test_files[i].get('filename', f'file_{i}'))
        R, C, E, T = raw.shape
        for r in range(R):
            for c in range(C):
                lab = col2label.get(c, 'UNK')
                if lab == 'UNK' or lab not in allowed:
                    continue
                sig = raw[r, c]
                if np.all(sig == 0):
                    continue
                if sig.shape[1] < seq_len:
                    continue
                starts = list(range(0, sig.shape[1] - seq_len + 1, stride))
                if not starts:
                    continue
                counts = np.zeros(len(classes_train), dtype=np.int64)
                total = 0
                bad = False
                for s0 in range(0, len(starts), cfg.eval_pred_chunk):
                    chunk_starts = starts[s0:s0 + cfg.eval_pred_chunk]
                    batch = np.stack([sig[:, s:s + seq_len].T for s in chunk_starts]).astype(np.float32)
                    batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                    probs = model.predict(batch, verbose=0)
                    if probs is None or not np.isfinite(probs).all():
                        bad = True
                        break
                    seg_argmax = probs.argmax(axis=1)
                    bc = np.bincount(seg_argmax, minlength=len(classes_train))
                    counts[:len(bc)] += bc
                    total += int(len(seg_argmax))
                if bad or total == 0:
                    n_skipped_nonfinite += 1
                    continue
                hard_vote_idx = int(np.argmax(counts))
                pred_lab = classes_train[hard_vote_idx]
                vote_pos = float(counts[pos_idx] / total)
                if not np.isfinite(vote_pos):
                    n_skipped_nonfinite += 1
                    continue
                vote_pos = float(np.clip(vote_pos, 0.0, 1.0))
                y_true.append(lab)
                y_pred.append(pred_lab)
                score_pos.append(vote_pos)
                rows_out.append((fname, r, c, lab, pred_lab, vote_pos))
    if n_skipped_nonfinite > 0:
        print(f'[Evaluatie] {round_tag}: {n_skipped_nonfinite} wells overgeslagen wegens non-finite model scores.')
    if not y_true:
        print(f'[{round_tag}] Geen data voor evaluatie.')
        return None
    cls_sorted = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=cls_sorted, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=cls_sorted)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f'\n=== RESULTS ({round_tag}) ===')
    print(f'Acc: {acc:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}')
    print(report)
    with open(os.path.join(outdir, f'report_{round_tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Acc: {acc:.6f}\nPrec: {prec:.6f}\nF1: {f1:.6f}\n\n{report}\n\nCM:\n{np.array2string(cm)}')
    with open(os.path.join(outdir, f'consensus_{round_tag}.csv'), 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows_out)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cls_sorted, yticklabels=cls_sorted)
    plt.title(f'Confusion Matrix - {round_tag}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'cm_{round_tag}.png'))
    plt.close()
    y_true_bin = np.array([1 if t == cfg.pos_label else 0 for t in y_true], dtype=np.int32)
    score_arr = np.array(score_pos, dtype=np.float32)
    auc_score = plot_roc_binary(y_true_bin, score_arr, outdir, round_tag)
    return {'y_true': y_true, 'y_pred': y_pred, 'score_vote': score_pos, 'auc': auc_score, 'acc': float(acc), 'prec': float(prec), 'f1': float(f1), 'n_eval': int(len(y_true)), 'classes_train': classes_train}

def write_round_summaries_csv(outdir: str):
    """Write a per-round overview to round_summaries.csv.
    """
    path = os.path.join(outdir, 'round_summaries.csv')
    rows = [('round', 'n_wells_eval', 'acc', 'prec', 'f1', 'auc')]
    for r in AGG.get('round_summaries', []):
        rows.append((r.get('round'), r.get('n_wells_eval'), f"{r.get('acc'):.6f}" if r.get('acc') is not None else '', f"{r.get('prec'):.6f}" if r.get('prec') is not None else '', f"{r.get('f1'):.6f}" if r.get('f1') is not None else '', '' if r.get('auc') is None else f"{r.get('auc'):.6f}"))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f'[RoundSummaries] CSV geschreven: {path}')

def write_overall_metrics(outdir: str):
    """Aggregate evaluations across all rounds and write overall metrics/plots.
    """
    if not AGG['y_true']:
        print('[Overall] Geen geaggregeerde evaluaties aanwezig -> overall metrics overgeslagen.')
        return
    y_true = AGG['y_true']
    y_pred = AGG['y_pred']
    cls_sorted = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=cls_sorted, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=cls_sorted)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    y_true_bin = np.array([1 if t == cfg.pos_label else 0 for t in y_true], dtype=np.int32)
    score_arr = np.array(AGG['score_vote'], dtype=np.float32)
    overall_auc = plot_roc_binary(y_true_bin, score_arr, outdir, 'overall')
    with open(os.path.join(outdir, 'overall_report.txt'), 'w', encoding='utf-8') as f:
        f.write('OVERALL (all rounds)\n')
        f.write(f'N wells evaluated: {len(y_true)}\n')
        f.write(f'Acc: {acc:.6f}\nPrec(w): {prec:.6f}\nF1(w): {f1:.6f}\n')
        f.write(f"AUC: {('' if overall_auc is None else f'{overall_auc:.6f}')}\n\n")
        f.write(report)
        f.write('\n\nCM:\n')
        f.write(np.array2string(cm))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cls_sorted, yticklabels=cls_sorted)
    plt.title('Confusion Matrix - OVERALL')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'overall_cm.png'))
    plt.close()
    summary = {'n_wells_eval': int(len(y_true)), 'labels_in_report': cls_sorted, 'acc': float(acc), 'prec_weighted': float(prec), 'f1_weighted': float(f1), 'auc': None if overall_auc is None else float(overall_auc), 'n_rounds_with_eval': int(len(AGG['round_summaries'])), 'pos_label': cfg.pos_label, 'global_split_path': GLOBAL_SPLIT_PATH, 'used_trainval_log': USED_TRAINVAL_LOG, 'global_best_model_path': GLOBAL_BEST_PATH, 'min_trainval_files': cfg.min_trainval_files, 'use_memmap_cache': cfg.use_memmap_cache}
    with open(os.path.join(outdir, 'overall_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('[Overall] Overall metrics geschreven (zie output files).')

def _safe_remove(path: str):
    """Safely remove a file (without failing hard on errors).
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f'[Cleanup Warning] Kon niet verwijderen: {path} ({e})')

def train_one_round(trainval, test, idx, spike_db):
    """Train one round and evaluate on final test files.
    
    Steps:
    - Preprocess train/val files (pool) and test files (final).
    - Split wells into train/val, segment to memmap.
    - Train model (warm start from GLOBAL_BEST_PATH if present).
    - Evaluate with consensus_eval_multifile().
    - Log results and clean the memmap cache.
    """
    print(f'\n========== RONDE {idx + 1} ==========')
    if len(trainval) < cfg.min_trainval_files:
        print(f'[Stop] Ronde {idx + 1}: train/val files={len(trainval)} < minimum {cfg.min_trainval_files}. Ronde overgeslagen.')
        return
    round_tag = f'round{idx + 1:03d}'
    round_dir = os.path.join(cfg.outdir, round_tag)
    os.makedirs(round_dir, exist_ok=True)
    CKPT_PATH_ROUND = os.path.join(round_dir, 'best_model.keras')
    LAST_PATH_ROUND = os.path.join(round_dir, 'last_model.keras')
    print('\n--- SAMENVATTING BESTANDEN DEZE RONDE ---')
    print(f'Train/Val files ({len(trainval)}):')
    for f in trainval:
        print(f"  - {f.get('filename', 'Unknown')}")
    print(f'Test files ({len(test)}):')
    for f in test:
        print(f"  - {f.get('filename', 'Unknown')}")
    print('-----------------------------------------\n')
    raw_tr, fs = process_file_list(trainval, spike_db, 'Train/Val Set (POOL)')
    raw_te, _ = process_file_list(test, spike_db, 'Test Set (FINAL)')
    if not raw_tr:
        print('Geen trainingsdata geladen. Stop ronde.')
        return
    E = raw_tr[0].shape[2]
    seq_len = int(round(cfg.seq_len_ms * fs / 1000.0))
    wells_all = [w for i in range(len(raw_tr)) for w in wells_from_file(i, cfg.cols, cfg.rows)]
    tr_wells, va_wells = train_test_split(wells_all, test_size=0.2, stratify=[w[3] for w in wells_all], random_state=cfg.seed + idx)
    tr_wells = filter_wells(tr_wells, get_allowed_cols_per_label(cfg.cols))
    cache_dir = os.path.join(round_dir, cfg.cache_dir_name)
    _ensure_dir(cache_dir)
    Xtr_path = os.path.join(cache_dir, 'Xtr.dat')
    ytr_path = os.path.join(cache_dir, 'ytr.dat')
    Xva_path = os.path.join(cache_dir, 'Xva.dat')
    yva_path = os.path.join(cache_dir, 'yva.dat')
    cleanup_paths = [Xtr_path, ytr_path, Xva_path, yva_path]
    ds_tr = ds_va = None
    model = None
    try:
        print('[Training] Segmenten naar memmap schrijven...')
        Ntr, classes, n2i, xtr_shape = make_segments_to_memmap(raw_list=raw_tr, wells=tr_wells, n_seg=cfg.train_segments_per_well, seq_len=seq_len, out_x_path=Xtr_path, out_y_path=ytr_path, name2idx=None, seed=cfg.seed + idx, shuffle_channels=cfg.channel_shuffle)
        Nva, _, _, xva_shape = make_segments_to_memmap(raw_list=raw_tr, wells=va_wells, n_seg=cfg.val_segments_per_well, seq_len=seq_len, out_x_path=Xva_path, out_y_path=yva_path, name2idx=n2i, seed=cfg.seed + idx + 1, shuffle_channels=False)
        print(f'[Training] Classes: {classes}')
        print(f'[Training] Xtr memmap: {xtr_shape} | Xva memmap: {xva_shape}')
        del raw_tr
        gc.collect()
        with tf.device('/CPU:0'):
            ds_tr = dataset_from_memmap(x_path=Xtr_path, y_path=ytr_path, x_shape=xtr_shape, batch_size=cfg.batch_size, training=True, seed=cfg.seed + idx)
            ds_va = dataset_from_memmap(x_path=Xva_path, y_path=yva_path, x_shape=xva_shape, batch_size=cfg.batch_size, training=False, seed=cfg.seed + idx)
        if os.path.exists(GLOBAL_BEST_PATH):
            print(f'[WarmStart] Laden global best: {GLOBAL_BEST_PATH}')
            model = tf.keras.models.load_model(GLOBAL_BEST_PATH)
        else:
            model = build_model(len(classes), seq_len, E)
        cbs = [callbacks.ModelCheckpoint(CKPT_PATH_ROUND, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1), callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True, monitor='val_accuracy', mode='max'), callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)]
        ytr_mm = np.memmap(ytr_path, dtype=np.int32, mode='r', shape=(Ntr,))
        cw = compute_class_weight('balanced', classes=np.arange(len(classes)), y=np.asarray(ytr_mm))
        del ytr_mm
        hist = model.fit(ds_tr, validation_data=ds_va, epochs=cfg.epochs, callbacks=cbs, class_weight=dict(enumerate(cw)))
        model.save(LAST_PATH_ROUND)
        if os.path.exists(CKPT_PATH_ROUND):
            shutil.copy2(CKPT_PATH_ROUND, GLOBAL_BEST_PATH)
            print(f'[WarmStart] Global best geüpdatet: {GLOBAL_BEST_PATH}')
        h_dict = {k: [float(x) for x in v] for k, v in hist.history.items()}
        with open(os.path.join(round_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(h_dict, f, indent=2)
        plot_curves(h_dict, round_dir, round_tag)
        eval_result = None
        if test and raw_te and (model is not None):
            eval_result = consensus_eval_multifile(raw_te, test, fs, model, classes, cfg.seq_len_ms, cfg.stride_ms_eval, round_dir, round_tag)
        else:
            print(f'[{round_tag}] Geen testbestanden beschikbaar -> evaluatie overgeslagen.')
        if eval_result is not None:
            AGG['y_true'].extend(eval_result['y_true'])
            AGG['y_pred'].extend(eval_result['y_pred'])
            AGG['score_vote'].extend(eval_result['score_vote'])
            AGG['classes'] = eval_result['classes_train']
            AGG['round_summaries'].append({'round': int(idx + 1), 'n_wells_eval': eval_result['n_eval'], 'acc': eval_result['acc'], 'prec': eval_result['prec'], 'f1': eval_result['f1'], 'auc': eval_result['auc'], 'classes_train': eval_result['classes_train'], 'round_dir': round_dir, 'best_model_path': CKPT_PATH_ROUND, 'last_model_path': LAST_PATH_ROUND, 'memmap_cache_dir': cache_dir})
            with open(os.path.join(cfg.outdir, 'round_summaries.json'), 'w', encoding='utf-8') as f:
                json.dump(AGG['round_summaries'], f, indent=2)
            write_round_summaries_csv(cfg.outdir)
    finally:
        try:
            if ds_tr is not None and hasattr(ds_tr, '_memmap_loader'):
                ds_tr._memmap_loader.close()
            if ds_va is not None and hasattr(ds_va, '_memmap_loader'):
                ds_va._memmap_loader.close()
        except Exception:
            pass
        try:
            del ds_tr, ds_va, model
        except Exception:
            pass
        tf.keras.backend.clear_session()
        gc.collect()
        if cfg.use_memmap_cache:
            for p in cleanup_paths:
                _safe_remove(p)
            try:
                if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) == 0:
                    os.rmdir(cache_dir)
            except Exception:
                pass
        try:
            del raw_te
        except Exception:
            pass
    print(f'Ronde {idx + 1} klaar. (best: {CKPT_PATH_ROUND})')

def main():
    """Entry point.
    
    Loads manifest and spike_db, creates/loads the global split, and iterates rounds until the pool
    is exhausted (or cfg.max_rounds). Afterwards, writes overall metrics.
    """
    if cfg.n_trainval_files < cfg.min_trainval_files:
        print(f'[Warning] n_trainval_files ({cfg.n_trainval_files}) < min_trainval_files ({cfg.min_trainval_files}). Zet n_trainval_files >= min_trainval_files.')
    manifest = load_manifest(cfg.manifest_path)
    spike_db = load_spike_database(cfg.spike_csv_path)
    pool, final = build_or_load_global_split(manifest, cfg.final_test_fraction)
    print(f'Pool (trainval_ids): {len(pool)}, Final (final_test_ids): {len(final)}')
    print(f'[Info] Global split path: {GLOBAL_SPLIT_PATH}')
    print(f'[Info] Used trainval log: {USED_TRAINVAL_LOG}')
    print(f'[Info] Global best model: {GLOBAL_BEST_PATH}')
    print(f'[Info] Minimum train/val files per ronde: {cfg.min_trainval_files}')
    print(f'[Info] Memmap cache: {cfg.use_memmap_cache} | eval_pred_chunk={cfg.eval_pred_chunk}')
    i = 0
    while True:
        if cfg.max_rounds and i >= cfg.max_rounds:
            break
        tr, done = pick_next_train_files(pool, cfg.seed + i, cfg.n_trainval_files)
        if done:
            print('[Done] Stopconditie bereikt.')
            break
        te = pick_test_files_from_final(final, cfg.seed + 10000 + i, cfg.n_test_files)
        train_one_round(tr, te, i, spike_db)
        i += 1
    write_overall_metrics(cfg.outdir)
    write_round_summaries_csv(cfg.outdir)
if __name__ == '__main__':
    main()
