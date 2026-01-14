# plot_overall_electrode_activity_simple.py
"""
Single scatter plot of all electrode spike measurements and CSV stats (simplified).

Outputs (folder ./activity_analyzed/):
 - overall_electrode_scatter.png
 - electrode_spike_stats.csv   (per well+electrode; includes min & max spikes)
 - overall_spike_summary.csv   (one-row overall summary)

This version DOES NOT mark persistent low/high electrodes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
INPUT_CSV = Path("analyses/electrode_activity_over_time.csv")  # adjust if necessary
OUT_DIR = Path.cwd() / "activity_analyzed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR
OUTPUT_PLOT = PLOTS_DIR / "overall_electrode_scatter.png"

# visual params
MARKER_SIZE = 18
ALPHA = 0.35
FIG_DPI = 150
# ====================

def find_column(df, candidates):
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def robust_bool_from_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    mask = ~num.isna()
    out = pd.Series(False, index=s.index)
    out.loc[mask] = num.loc[mask].astype(float) > 0
    rem = ~mask
    out.loc[rem] = s.loc[rem].astype(str).str.lower().isin(["true","t","yes","y","1"])
    return out.fillna(False)

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)

    # detect columns
    col_id = find_column(df, ["id", "measurement", "measurementid"])
    col_well = find_column(df, ["well"])
    col_elec = find_column(df, ["electrode","chan","channel"])
    col_spikes = find_column(df, ["spike"])
    col_date = find_column(df, ["date"])
    col_present = find_column(df, ["present_in_file","present","in_file"])
    col_active = find_column(df, ["is_active","active"])

    if col_id is None or col_well is None or col_elec is None:
        raise ValueError("Required columns not found in input CSV. Need: id, well, electrode.")

    # normalize names
    rename_map = {col_id: "id", col_well: "well", col_elec: "electrode"}
    if col_spikes:
        rename_map[col_spikes] = "Spikes"
    if col_date:
        rename_map[col_date] = "date"
    if col_present:
        rename_map[col_present] = "present_in_file"
    if col_active:
        rename_map[col_active] = "is_active"
    df = df.rename(columns=rename_map)

    # standardize types
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["well"] = pd.to_numeric(df["well"], errors="coerce").astype("Int64")
    df["electrode"] = pd.to_numeric(df["electrode"], errors="coerce").astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # spikes numeric
    if "Spikes" in df.columns:
        df["Spikes"] = pd.to_numeric(df["Spikes"], errors="coerce").fillna(0).astype(int)
    else:
        df["Spikes"] = 0

    # present / active inference (not used for stats here but kept robust)
    if "present_in_file" in df.columns:
        df["present_in_file"] = robust_bool_from_series(df["present_in_file"])
    else:
        df["present_in_file"] = True

    if "is_active" in df.columns:
        df["is_active"] = robust_bool_from_series(df["is_active"])
    else:
        df["is_active"] = df["Spikes"] > 0

    # measurement list
    measurement_ids = sorted(df["id"].dropna().unique().astype(int).tolist())
    if len(measurement_ids) == 0:
        raise ValueError("No measurement ids in input CSV.")
    n_measurements = len(measurement_ids)

    # ---------- per-electrode stats (mean, median, std, min, max, total, frac_nonzero) ----------
    grouped = df.groupby(["well","electrode"])
    rows = []
    for (w,e), g in grouped:
        g_sorted = g.sort_values("id")
        spikes = g_sorted["Spikes"].astype(float).values
        mean_spikes = float(np.mean(spikes)) if len(spikes)>0 else 0.0
        median_spikes = float(np.median(spikes)) if len(spikes)>0 else 0.0
        std_spikes = float(np.std(spikes, ddof=0)) if len(spikes)>0 else 0.0
        min_spikes = float(np.min(spikes)) if len(spikes)>0 else 0.0
        max_spikes = float(np.max(spikes)) if len(spikes)>0 else 0.0
        total_spikes = int(np.sum(spikes)) if len(spikes)>0 else 0
        # fraction nonzero: how often spikes>0 relative to global measurement count (so missing measured as 0)
        frac_nonzero = float(np.sum(spikes>0)/n_measurements) if n_measurements>0 else 0.0
        rows.append({
            "well": int(w),
            "electrode": int(e),
            "mean_spikes": mean_spikes,
            "median_spikes": median_spikes,
            "std_spikes": std_spikes,
            "min_spikes": min_spikes,
            "max_spikes": max_spikes,
            "total_spikes": total_spikes,
            "frac_nonzero": frac_nonzero,
            "n_measurements": int(n_measurements)
        })

    stats_df = pd.DataFrame(rows).sort_values(["well","electrode"]).reset_index(drop=True)

    electrode_out = OUT_DIR / "electrode_spike_stats.csv"
    stats_df.to_csv(electrode_out, index=False)

    # ---------- overall summary CSV ----------
    grand_mean_spikes = float(df["Spikes"].mean())
    mean_of_electrode_means = float(stats_df["mean_spikes"].mean()) if not stats_df.empty else float("nan")
    median_of_electrode_means = float(stats_df["mean_spikes"].median()) if not stats_df.empty else float("nan")
    overall_summary = {
        "grand_mean_spikes": grand_mean_spikes,
        "mean_of_electrode_means": mean_of_electrode_means,
        "median_of_electrode_means": median_of_electrode_means,
        "n_electrodes": int(len(stats_df)),
        "n_measurements": int(n_measurements)
    }
    summary_df = pd.DataFrame([overall_summary])
    summary_out = OUT_DIR / "overall_spike_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    # ---------- single scatter plot (all electrodes & measurements) ----------
    wells = sorted(df["well"].dropna().unique().astype(int).tolist())
    cmap = plt.get_cmap("tab20")
    colors = {w: cmap(i % 20) for i, w in enumerate(wells)}

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    for w in wells:
        sub = df[df["well"] == w]
        if sub.empty:
            continue
        ax.scatter(sub["id"], sub["Spikes"], s=MARKER_SIZE, alpha=ALPHA, color=colors[w], label=f"Well {w}", edgecolors="none")

    ax.set_xlabel("Measurement id")
    ax.set_ylabel("Spikes")
    ax.set_title("All electrodes: spikes per measurement (points colored by well)")
    if len(wells) <= 20:
        ax.legend(title="Well", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=FIG_DPI)
    plt.close()

    print("✅ Done.")
    print("Plot saved to:", OUTPUT_PLOT.resolve())
    print("Electrode stats saved to:", electrode_out.resolve())
    print("Overall summary saved to:", summary_out.resolve())

if __name__ == "__main__":
    main()
