#!/usr/bin/env python3
"""
Full electrode/well activity analysis pipeline.

Inputs:
 - analyses/electrode_activity_over_time.csv

Outputs:
 - activity_analyzed/
     - inactive_electrodes.csv
     - well_inactive_counts_per_measurement.csv
     - wells_inactive_per_measurement.csv
     - well_plate_mappings.csv
     - spikes_per_electrode_plots/*.png
     - bursts_per_electrode_plots/*.png
     - spikes_per_electrode_grid_plots/*.png
     - bursts_per_electrode_grid_plots/*.png
 - activity_analyzed/measurements_spikes_bursts/
     - electrode_spikes_over_measurement.csv
     - electrode_bursts_over_measurements.csv
     - well_spikes_over_measurements.csv
     - well_bursts_over_measurements.csv
     - electrode_spike_stats.csv
     - electrode_burst_stats.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
INPUT_CSV = Path("analyses/electrode_activity_over_time.csv")  # change if needed
OUT_ROOT = Path.cwd() / "activity_analyzed"
MEAS_DIR = OUT_ROOT / "measurements_spikes_bursts"

# plot dirs
PLOTS_DIR_SPIKES = OUT_ROOT / "spikes_per_electrode_plots"
PLOTS_DIR_BURSTS = OUT_ROOT / "bursts_per_electrode_plots"
PLOTS_DIR_SPIKES_GRID = OUT_ROOT / "spikes_per_electrode_grid_plots"
PLOTS_DIR_BURSTS_GRID = OUT_ROOT / "bursts_per_electrode_grid_plots"

# ensure directories
for p in (OUT_ROOT, MEAS_DIR, PLOTS_DIR_SPIKES, PLOTS_DIR_BURSTS, PLOTS_DIR_SPIKES_GRID, PLOTS_DIR_BURSTS_GRID):
    p.mkdir(parents=True, exist_ok=True)

OUT_INACTIVE_ELECTRODES = OUT_ROOT / "inactive_electrodes.csv"
OUT_WELL_INACTIVITY = OUT_ROOT / "well_inactive_counts_per_measurement.csv"
OUT_WELL_INACTIVE_LISTS = OUT_ROOT / "wells_inactive_per_measurement.csv"
OUT_WELL_MAPPINGS = OUT_ROOT / "well_plate_mappings.csv"

# plate geometry (used to produce mapping CSV for row/col checks)
N_COLS = 8
N_ROWS = 6

# plotting options
MARKER_SIZE = 4
LINE_WIDTH = 1.0
XTICK_LABEL_EVERY = 2   # show numeric label every N ticks in grid subplots
GRID_COLS = 2
GRID_ROWS = 8

# -----------------------------
# Utility functions
# -----------------------------
def find_column(df, candidates):
    """Return first column name in df that contains any candidate substring (case-insensitive)."""
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def robust_bool_series(s):
    """Convert series to boolean robustly."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    mask_num = ~num.isna()
    out = pd.Series(False, index=s.index)
    out.loc[mask_num] = num.loc[mask_num].astype(float) > 0
    mask_rem = ~mask_num
    out.loc[mask_rem] = s.loc[mask_rem].astype(str).str.lower().str.strip().isin(["true","t","yes","y","1"])
    return out.fillna(False)

def row_major_coords(well_id, n_cols=N_COLS):
    zero = int(well_id) - 1
    row = (zero // n_cols) + 1
    col = (zero % n_cols) + 1
    return int(row), int(col)

def col_major_coords(well_id, n_rows=N_ROWS):
    zero = int(well_id) - 1
    col = (zero // n_rows) + 1
    row = (zero % n_rows) + 1
    return int(row), int(col)

def coords_to_A1_label(row, col):
    # row=1->A, 2->B, ...
    letter = chr(64 + int(row)) if 1 <= row <= 26 else f"R{row}"
    return f"{letter}{int(col)}"

def _safe_std(a: np.ndarray) -> float:
    """Sample std (ddof=1); returns 0.0 if <2 samples."""
    if a.size < 2:
        return 0.0
    return float(np.nanstd(a, ddof=1))

def _stats_row(values: np.ndarray) -> dict:
    """
    Compute stats over a 1D vector of values.
    NOTE: 'min' wordt berekend ZONDER 0-metingen (alleen over v > 0).
          Als er geen niet-nul waarden zijn, wordt min = NaN.
    """
    v = np.nan_to_num(values, nan=0.0)
    n = v.size
    nonzero_mask = v > 0
    n_nonzero = int(np.count_nonzero(nonzero_mask))
    zero_frac = float((n - n_nonzero) / n) if n > 0 else np.nan

    total = float(np.nansum(v))
    mean = float(np.nanmean(v)) if n > 0 else np.nan
    median = float(np.nanmedian(v)) if n > 0 else np.nan
    std = _safe_std(v)

    # min zonder 0-metingen
    if n_nonzero > 0:
        vmin = float(np.nanmin(v[nonzero_mask]))
    else:
        vmin = np.nan

    vmax = float(np.nanmax(v)) if n > 0 else np.nan

    return {
        "n_measurements": int(n),
        "n_nonzero": n_nonzero,
        "fraction_zero": zero_frac,
        "total": total,
        "mean": mean,
        "median": median,
        "std": std,
        "min": vmin,   # zonder 0-metingen
        "max": vmax,
    }

def _make_electrode_stats_from_pivot(pivot_df: pd.DataFrame, out_path: Path, value_name: str):
    """
    pivot_df: index = (well, electrode), columns = measurement ids, numeric values
    Writes stats CSV with columns:
      well, electrode, n_measurements, {value}_n_nonzero, {value}_fraction_zero,
      {value}_total, {value}_mean, {value}_median, {value}_std, {value}_min, {value}_max
    """
    if pivot_df is None or pivot_df.empty:
        cols = ["well","electrode","n_measurements",
                f"{value_name}_n_nonzero", f"{value_name}_fraction_zero",
                f"{value_name}_total", f"{value_name}_mean", f"{value_name}_median",
                f"{value_name}_std", f"{value_name}_min", f"{value_name}_max"]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        print(f"⚠️ No data for {out_path.name}; wrote empty file.")
        return

    vals = pivot_df.fillna(0).astype(float).to_numpy()
    wells = pivot_df.index.get_level_values(0).astype(int)
    electrodes = pivot_df.index.get_level_values(1).astype(int)

    rows = []
    for i in range(vals.shape[0]):
        stats = _stats_row(vals[i, :])
        rows.append({
            "well": int(wells[i]),
            "electrode": int(electrodes[i]),
            "n_measurements": stats["n_measurements"],
            f"{value_name}_n_nonzero": stats["n_nonzero"],
            f"{value_name}_fraction_zero": stats["fraction_zero"],
            f"{value_name}_total": stats["total"],
            f"{value_name}_mean": stats["mean"],
            f"{value_name}_median": stats["median"],
            f"{value_name}_std": stats["std"],
            f"{value_name}_min": stats["min"],   # min zonder 0-metingen
            f"{value_name}_max": stats["max"],
        })

    out_df = pd.DataFrame(rows).sort_values(["well","electrode"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path.name} -> {out_path.resolve()}")

# -----------------------------
# Load and normalize input
# -----------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)

# detect columns
col_id = find_column(df, ["id", "measurement", "measurementid"])
col_date = find_column(df, ["date"])
col_well = find_column(df, ["well"])
col_elec = find_column(df, ["electrode", "chan", "channel"])
col_spikes = find_column(df, ["spike"])
col_bursts = find_column(df, ["burst"])
col_present = find_column(df, ["present_in_file", "present", "in_file"])
col_active = find_column(df, ["is_active", "active"])

if col_id is None or col_well is None or col_elec is None:
    raise ValueError("Required columns not found in input CSV. Need columns identifying measurement id, well and electrode.")

# rename to canonical names
rename_map = {col_id: "id", col_well: "well", col_elec: "electrode"}
if col_date:
    rename_map[col_date] = "date"
if col_spikes:
    rename_map[col_spikes] = "Spikes"
if col_bursts:
    rename_map[col_bursts] = "Bursts"
if col_present:
    rename_map[col_present] = "present_in_file"
if col_active:
    rename_map[col_active] = "is_active"
df = df.rename(columns=rename_map)

# coerce types
df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
df["well"] = pd.to_numeric(df["well"], errors="coerce").astype("Int64")
df["electrode"] = pd.to_numeric(df["electrode"], errors="coerce").astype("Int64")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

# numeric spikes/bursts
if "Spikes" in df.columns:
    df["Spikes"] = pd.to_numeric(df["Spikes"], errors="coerce").fillna(0).astype(int)
else:
    df["Spikes"] = 0
if "Bursts" in df.columns:
    df["Bursts"] = pd.to_numeric(df["Bursts"], errors="coerce").fillna(0).astype(int)
else:
    df["Bursts"] = 0

# present_in_file
if "present_in_file" in df.columns:
    df["present_in_file"] = robust_bool_series(df["present_in_file"])
else:
    df["present_in_file"] = True

# is_active
if "is_active" in df.columns:
    df["is_active"] = robust_bool_series(df["is_active"])
else:
    df["is_active"] = (df["Spikes"] > 0) | (df["Bursts"] > 0)

# ensure no duplicate rows for same id/well/electrode by aggregating (sum)
df = df.groupby(["id", "well", "electrode"], as_index=False).agg({
    **({} if "date" not in df.columns else {"date": "first"}),
    "Spikes": "sum",
    "Bursts": "sum",
    "present_in_file": "max",
    "is_active": "max"
})

# measurement ids sorted
measurement_ids = sorted(df["id"].dropna().unique().astype(int).tolist())
if len(measurement_ids) == 0:
    raise ValueError("No measurement ids found after parsing input CSV.")

# -----------------------------
# Create measurement-overview CSVs (electrode & well)
# -----------------------------
# electrode spikes pivot
pivot_elec_spikes = df.pivot_table(index=["well", "electrode"], columns="id", values="Spikes", aggfunc="sum")
pivot_elec_spikes = pivot_elec_spikes.reindex(columns=measurement_ids, fill_value=0).sort_index()
elec_spikes_out = pivot_elec_spikes.reset_index()
elec_spikes_out.columns = [str(c) for c in elec_spikes_out.columns]
elec_spikes_file = MEAS_DIR / "electrode_spikes_over_measurement.csv"
elec_spikes_out.to_csv(elec_spikes_file, index=False)

# electrode bursts pivot
pivot_elec_bursts = df.pivot_table(index=["well", "electrode"], columns="id", values="Bursts", aggfunc="sum")
pivot_elec_bursts = pivot_elec_bursts.reindex(columns=measurement_ids, fill_value=0).sort_index()
elec_bursts_out = pivot_elec_bursts.reset_index()
elec_bursts_out.columns = [str(c) for c in elec_bursts_out.columns]
elec_bursts_file = MEAS_DIR / "electrode_bursts_over_measurements.csv"
elec_bursts_out.to_csv(elec_bursts_file, index=False)

# well-level spikes (sum across electrodes)
pivot_well_spikes = df.pivot_table(index=["well"], columns="id", values="Spikes", aggfunc="sum")
pivot_well_spikes = pivot_well_spikes.reindex(columns=measurement_ids, fill_value=0).sort_index()
well_spikes_out = pivot_well_spikes.reset_index()
well_spikes_out.columns = [str(c) for c in well_spikes_out.columns]
well_spikes_file = MEAS_DIR / "well_spikes_over_measurements.csv"
well_spikes_out.to_csv(well_spikes_file, index=False)

# well-level bursts
pivot_well_bursts = df.pivot_table(index=["well"], columns="id", values="Bursts", aggfunc="sum")
pivot_well_bursts = pivot_well_bursts.reindex(columns=measurement_ids, fill_value=0).sort_index()
well_bursts_out = pivot_well_bursts.reset_index()
well_bursts_out.columns = [str(c) for c in well_bursts_out.columns]
well_bursts_file = MEAS_DIR / "well_bursts_over_measurements.csv"
well_bursts_out.to_csv(well_bursts_file, index=False)

print("Measurement-overview CSVs written to:", MEAS_DIR.resolve())

# -----------------------------
# Per-electrode stats CSVs (min excl. zeros)
# -----------------------------
spike_stats_csv = MEAS_DIR / "electrode_spike_stats.csv"
burst_stats_csv = MEAS_DIR / "electrode_burst_stats.csv"
_make_electrode_stats_from_pivot(pivot_elec_spikes, spike_stats_csv, value_name="spikes")
_make_electrode_stats_from_pivot(pivot_elec_bursts, burst_stats_csv, value_name="bursts")

# -----------------------------
# Inactive electrode & well summaries + mapping CSV
# -----------------------------
# grouped by well/electrode
grouped = df.groupby(["well", "electrode"])

wells = sorted(df["well"].dropna().unique().astype(int).tolist())
well_electrode_sets = {w: sorted(df[df["well"]==w]["electrode"].dropna().unique().astype(int).tolist()) for w in wells}
# estimate electrodes per well (if available) else fallback to max observed
electrodes_total_by_well = {}
for w, eles in well_electrode_sets.items():
    electrodes_total_by_well[w] = max(max(eles), len(eles)) if eles else 0

# build maps for presence/active per (well, electrode)
active_map = {}
present_map = {}
for (w, e), g in grouped:
    id_to_active = dict(zip(g["id"].astype(int), g["is_active"].astype(bool)))
    id_to_present = dict(zip(g["id"].astype(int), g["present_in_file"].astype(bool)))
    active_map[(w,e)] = id_to_active
    present_map[(w,e)] = id_to_present

measurement_ids_sorted = measurement_ids
total_measurements = len(measurement_ids_sorted)

# inactive_electrodes.csv rows
rows = []
all_pairs = sorted(list(grouped.groups.keys()), key=lambda x: (int(x[0]), int(x[1])))
for (w, e) in all_pairs:
    id_to_active = active_map.get((w,e), {})
    id_to_present = present_map.get((w,e), {})

    inactive_ids = []
    inactive_count = 0
    for mid in measurement_ids_sorted:
        present = bool(id_to_present.get(mid, False))
        active = bool(id_to_active.get(mid, False))
        if (not present) or (not active):
            inactive_count += 1
            inactive_ids.append(mid)

    fraction_inactive = inactive_count / total_measurements if total_measurements > 0 else np.nan
    always_inactive = 1 if inactive_count == total_measurements else 0

    rows.append({
        "well": int(w),
        "electrode": int(e),
        "total_measurements": int(total_measurements),
        "inactive_count": int(inactive_count),
        "fraction_inactive": float(fraction_inactive),
        "inactive_measurements": ",".join(str(i) for i in inactive_ids),
        "always_inactive": int(always_inactive)
    })

inactive_df = pd.DataFrame(rows).sort_values(["well", "electrode"]).reset_index(drop=True)
inactive_df.to_csv(OUT_INACTIVE_ELECTRODES, index=False)

# per-well per-measurement counts + wells-inactive summary
well_rows = []
measurement_inactive_wells = {mid: [] for mid in measurement_ids_sorted}

for mid in measurement_ids_sorted:
    date_val = None
    if "date" in df.columns:
        sample = df[df["id"]==mid]
        if not sample.empty:
            date_val = sample["date"].dropna().iloc[0]
    for w in wells:
        electrodes_seen = well_electrode_sets.get(w, [])
        electrodes_total = electrodes_total_by_well.get(w, len(electrodes_seen))
        n_inactive = 0
        n_present = 0
        for e in electrodes_seen:
            id_to_active = active_map.get((w,e), {})
            id_to_present = present_map.get((w,e), {})
            present = bool(id_to_present.get(mid, False))
            active = bool(id_to_active.get(mid, False))
            if present:
                n_present += 1
            if (not present) or (not active):
                n_inactive += 1
        if not electrodes_seen:
            electrodes_total = 0
        fraction_inactive_in_well = (n_inactive / electrodes_total) if electrodes_total > 0 else np.nan

        if electrodes_total > 0 and n_inactive == electrodes_total:
            measurement_inactive_wells[mid].append(int(w))

        well_rows.append({
            "id": int(mid),
            "date": pd.NaT if date_val is None else pd.to_datetime(date_val).date(),
            "well": int(w),
            "n_electrodes_inactive": int(n_inactive),
            "n_electrodes_present": int(n_present),
            "n_electrodes_total": int(electrodes_total),
            "fraction_inactive_in_well": float(fraction_inactive_in_well) if not np.isnan(fraction_inactive_in_well) else np.nan
        })

well_inact_df = pd.DataFrame(well_rows).sort_values(["id","well"]).reset_index(drop=True)
well_inact_df.to_csv(OUT_WELL_INACTIVITY, index=False)

# per-measurement wells-inactive list CSV
well_inactive_list_rows = []
n_wells_total = len(wells)
for mid in measurement_ids_sorted:
    inactive_wells = measurement_inactive_wells.get(mid, [])
    n_inactive = len(inactive_wells)
    fraction_wells_inactive = n_inactive / n_wells_total if n_wells_total > 0 else np.nan
    date_val = None
    if "date" in df.columns:
        sample = df[df["id"]==mid]
        if not sample.empty:
            date_val = sample["date"].dropna().iloc[0]
    well_inactive_list_rows.append({
        "id": int(mid),
        "date": pd.NaT if date_val is None else pd.to_datetime(date_val).date(),
        "n_wells_inactive": int(n_inactive),
        "n_wells_total": int(n_wells_total),
        "fraction_wells_inactive": float(fraction_wells_inactive) if not np.isnan(fraction_wells_inactive) else np.nan,
        "inactive_wells": ",".join(str(w) for w in sorted(inactive_wells))
    })

wells_inactive_df = pd.DataFrame(well_inactive_list_rows).sort_values("id").reset_index(drop=True)
wells_inactive_df.to_csv(OUT_WELL_INACTIVE_LISTS, index=False)

# mapping CSV
mapping_rows = []
for w in wells:
    rm_row, rm_col = row_major_coords(w)
    cm_row, cm_col = col_major_coords(w)
    rm_label = coords_to_A1_label(rm_row, rm_col)
    cm_label = coords_to_A1_label(cm_row, cm_col)
    mapping_rows.append({
        "well": int(w),
        "row_major_row": int(rm_row),
        "row_major_col": int(rm_col),
        "row_major_label": rm_label,
        "column_major_row": int(cm_row),
        "column_major_col": int(cm_col),
        "column_major_label": cm_label
    })
mapping_df = pd.DataFrame(mapping_rows).sort_values("well")
mapping_df.to_csv(OUT_WELL_MAPPINGS, index=False)

print("Summary CSVs written to:", OUT_ROOT.resolve())

# -----------------------------
# Plotting functions
# -----------------------------
def _make_per_well_line_plots(df_local, measurement_ids_local, out_dir, value_col, title_prefix):
    wells_local = sorted(df_local["well"].dropna().unique().astype(int).tolist())
    n_measurements = len(measurement_ids_local)
    tick_label_every = XTICK_LABEL_EVERY if n_measurements > 20 else 1

    for w in wells_local:
        dfw = df_local[df_local["well"] == w].copy()
        if dfw.empty:
            continue
        pivot = dfw.pivot_table(index="electrode", columns="id", values=value_col, aggfunc="sum")
        pivot = pivot.reindex(columns=measurement_ids_local, fill_value=0).sort_index()
        electrodes = list(pivot.index)
        if not electrodes:
            continue

        # colors
        cmap = plt.get_cmap("tab20")
        cmap_size = 20
        colors = cmap(np.linspace(0,1,cmap_size)) if len(electrodes) <= cmap_size else plt.get_cmap("viridis")(np.linspace(0,1,len(electrodes)))
        color_map = {elec: colors[i % len(colors)] for i, elec in enumerate(electrodes)}

        fig, ax = plt.subplots(figsize=(max(8, 0.25 * n_measurements + 2), 6))
        for elec in electrodes:
            y = pivot.loc[elec].values
            ax.plot(measurement_ids_local, y, marker='o', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, color=color_map[elec], label=f"E{int(elec)}", alpha=0.9)

        ax.set_title(f"{title_prefix} — Well {w}")
        ax.set_xlabel("Measurement id")
        ax.set_ylabel(value_col)

        # ticks at every measurement but label every Nth for readability
        ax.set_xticks(measurement_ids_local)
        label_list = [str(m) if (i % tick_label_every == 0) else "" for i, m in enumerate(measurement_ids_local)]
        ax.set_xticklabels(label_list, rotation=0, fontsize=8)

        if len(electrodes) <= 20:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize="small")
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:12], labels[:12], loc='upper left', bbox_to_anchor=(1.02, 1), fontsize="small")

        plt.tight_layout()
        out_file = out_dir / f"well_{int(w):02d}_{value_col.lower()}_per_electrode.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

def _make_per_well_grid_plots(df_local, measurement_ids_local, out_dir, value_col, title_prefix):
    wells_local = sorted(df_local["well"].dropna().unique().astype(int).tolist())
    n_measurements = len(measurement_ids_local)
    tick_label_every = XTICK_LABEL_EVERY if n_measurements > 20 else 1
    # precompute tick labels for reuse
    tick_labels_common = [str(m) if (i % tick_label_every == 0) else "" for i, m in enumerate(measurement_ids_local)]

    for w in wells_local:
        dfw = df_local[df_local["well"] == w].copy()
        if dfw.empty:
            continue
        pivot_raw = dfw.pivot_table(index="electrode", columns="id", values=value_col, aggfunc="sum")  # raw to count presence
        pivot = pivot_raw.reindex(columns=measurement_ids_local, fill_value=np.nan)
        pivot_filled = pivot.fillna(0).astype(float).sort_index()
        electrodes = list(pivot_filled.index)
        if not electrodes:
            continue

        max_slots = GRID_COLS * GRID_ROWS
        fig_w = max(12, 0.25 * n_measurements + 4)
        fig_h = max(12, 2.2 * GRID_ROWS)
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(fig_w, fig_h), sharex=False, sharey=False)
        axes = axes.flatten()

        for slot in range(max_slots):
            ax = axes[slot]
            if slot < len(electrodes):
                elec = electrodes[slot]
                y = pivot_filled.loc[elec].values
                ax.plot(measurement_ids_local, y, marker='o', markersize=3.5, linewidth=1, alpha=0.9)
                ax.set_title(f"E{int(elec)}", fontsize=9)

                # total and annotate top-right
                total_val = int(np.nansum(y))
                ax.text(0.98, 0.95, f"total: {total_val}", transform=ax.transAxes,
                        ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

                # number of measurements present (non-NaN in raw pivot)
                # n_meas_present = int(pivot_raw.loc[elec].count()) if elec in pivot_raw.index else 0
                # ax.text(0.5, -0.18, f"measurements present: {n_meas_present}", transform=ax.transAxes,
                #         ha='center', va='top', fontsize=7, color='black', alpha=0.9)

                # set ticks at every measurement; label only every tick_label_every-th
                ax.set_xticks(measurement_ids_local)
                ax.set_xticklabels(tick_labels_common, rotation=0, fontsize=7)
                ax.tick_params(axis='x', which='major', labelbottom=True)
            else:
                ax.set_visible(False)

            if slot % GRID_COLS == 0 and slot < len(electrodes):
                ax.set_ylabel(value_col, fontsize=8)

        plt.suptitle(f"{title_prefix} — Well {w} — each subplot = electrode", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.subplots_adjust(bottom=0.08, hspace=0.6)

        out_file = out_dir / f"well_{int(w):02d}_{value_col.lower()}_grid_per_electrode.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

# create plots
_make_per_well_line_plots(df, measurement_ids, PLOTS_DIR_SPIKES, value_col="Spikes", title_prefix="Spikes")
_make_per_well_line_plots(df, measurement_ids, PLOTS_DIR_BURSTS, value_col="Bursts", title_prefix="Bursts")

_make_per_well_grid_plots(df, measurement_ids, PLOTS_DIR_SPIKES_GRID, value_col="Spikes", title_prefix="Spikes")
_make_per_well_grid_plots(df, measurement_ids, PLOTS_DIR_BURSTS_GRID, value_col="Bursts", title_prefix="Bursts")

print("Plots written to:")
print(" -", PLOTS_DIR_SPIKES.resolve())
print(" -", PLOTS_DIR_BURSTS.resolve())
print(" -", PLOTS_DIR_SPIKES_GRID.resolve())
print(" -", PLOTS_DIR_BURSTS_GRID.resolve())

print("\n✅ All done.")
