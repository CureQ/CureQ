# analyze_mea_timeseries_manifest.py
# ------------------------------------------------------------
# MEA analysis script (manifest uses filenames only).
# - Shows every 2nd tick on the x-axis and no rotation of labels.
# - Automatic figure width (EXTRA_PER_POINT = 0.1 inch).
# - Outputs under ./analyses/
# ------------------------------------------------------------

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ CONFIG ============
WELL_DIR       = Path(r"C:\Users\chenp\Documents\SCA1_Features\Well")          # <-- zet dit
ELECTRODE_DIR  = Path(r"C:\Users\chenp\Documents\SCA1_Features\Electrode")     # <-- zet dit

SCRIPT_DIR     = Path(__file__).resolve().parent
MANIFEST_PATH  = SCRIPT_DIR / "./manifest/manifest.json"

DEFAULT_WELLS  = 48
DEFAULT_ELECS  = 16

DOT_SIZE       = 28
ALPHA          = 0.9
LINE_CONNECT   = True
LINE_WIDTH     = 1.2

# Automatic figure sizing parameters (0.1 inch per datapunt)
BASE_WIDTH_INCHES = 6.0
EXTRA_PER_POINT = 0.1
FIG_HEIGHT = 5.0

# Tick show step: show every Nth tick (user requested every 2nd)
TICK_STEP = 2
# ===============================


# ---------- Helpers: read CSVs ----------
def read_well_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read well CSV '{path}': {e}")
        return None

    c_well = next((c for c in df.columns if c.lower().startswith("well")), None)
    if c_well is None:
        print(f"[WARN] No 'Well' column found in well CSV '{path.name}'.")
        return None

    def pick_total(keyword: str):
        for c in df.columns:
            cl = c.lower()
            if keyword in cl and ("total" in cl or "sum" in cl):
                return c
        for c in df.columns:
            if keyword in c.lower():
                return c
        return None

    c_spikes_tot = pick_total("spike")
    c_bursts_tot = pick_total("burst")

    out = pd.DataFrame()
    out["Well"] = pd.to_numeric(df[c_well], errors="coerce").astype("Int64")
    if c_spikes_tot is not None:
        out["Spikes_total_well"] = pd.to_numeric(df[c_spikes_tot], errors="coerce")
    if c_bursts_tot is not None:
        out["Bursts_total_well"] = pd.to_numeric(df[c_bursts_tot], errors="coerce")
    return out


def read_electrode_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    def pick_col(keywords):
        for c in df.columns:
            cl = c.lower()
            for kw in keywords:
                if kw in cl:
                    return c
        return None

    c_well = pick_col(["well"])
    c_elec = pick_col(["electrode", "channel", "chan", "electrodeindex", "electrode_id"])
    if c_well is None or c_elec is None:
        raise ValueError(f"Could not find 'Well' or 'Electrode' column in electrode CSV '{path.name}'")

    c_spikes = pick_col(["spike"])
    c_bursts = pick_col(["burst"])

    out = pd.DataFrame()
    out["Well"] = pd.to_numeric(df[c_well], errors="coerce").astype("Int64")
    out["Electrode"] = pd.to_numeric(df[c_elec], errors="coerce").astype("Int64")
    out["Spikes"] = pd.to_numeric(df[c_spikes], errors="coerce").fillna(0).astype(int) if c_spikes is not None else 0
    out["Bursts"] = pd.to_numeric(df[c_bursts], errors="coerce").fillna(0).astype(int) if c_bursts is not None else 0

    return out


# ---------- Helpers: grid / summarise ----------
def ensure_full_grid_with_presence(df_elec: pd.DataFrame, wells: int, elecs: int) -> pd.DataFrame:
    wells_list = list(range(1, wells + 1))
    elecs_list = list(range(1, elecs + 1))
    grid = pd.MultiIndex.from_product([wells_list, elecs_list], names=["Well", "Electrode"]).to_frame(index=False)

    merged = grid.merge(df_elec, on=["Well", "Electrode"], how="left", indicator=True)
    merged["present_in_file"] = merged["_merge"].eq("both")
    merged.drop(columns="_merge", inplace=True)

    for col in ["Spikes", "Bursts"]:
        if col not in merged:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    merged["is_active"] = (merged["Spikes"] > 0) | (merged["Bursts"] > 0)
    return merged


def summarize_well(df_well: pd.DataFrame | None, df_elec_full: pd.DataFrame) -> pd.DataFrame:
    agg = df_elec_full.groupby("Well", as_index=False).agg(
        Spikes_total=("Spikes", "sum"), Bursts_total=("Bursts", "sum")
    )
    if df_well is None:
        return agg

    dfw = df_well.copy()
    out = agg.copy()
    if "Spikes_total_well" in dfw:
        out = out.drop(columns=["Spikes_total"], errors="ignore").merge(
            dfw[["Well", "Spikes_total_well"]], on="Well", how="left"
        ).rename(columns={"Spikes_total_well": "Spikes_total"})
        out["Spikes_total"] = out["Spikes_total"].fillna(agg["Spikes_total"])
    if "Bursts_total_well" in dfw:
        out = out.drop(columns=["Bursts_total"], errors="ignore").merge(
            dfw[["Well", "Bursts_total_well"]], on="Well", how="left"
        ).rename(columns={"Bursts_total_well": "Bursts_total"})
        out["Bursts_total"] = out["Bursts_total"].fillna(agg["Bursts_total"])
    return out


# ---------- Plot helpers ----------
def _auto_figsize_by_npoints(n_points: int) -> tuple[float, float]:
    max_extra = 80.0
    extra = min(EXTRA_PER_POINT * max(0, n_points), max_extra)
    width = max(BASE_WIDTH_INCHES, BASE_WIDTH_INCHES + extra)
    return (width, FIG_HEIGHT)


def set_xticks_subset(ax, positions: list[float], step: int = 2):
    """Set xticks using only every 'step'-th position from positions (positions is ordered)."""
    if not positions:
        return
    # ensure positions sorted
    pos = list(positions)
    ticks = pos[::step]
    labels = [str(int(v)) for v in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, ha="center")


def plot_per_well_timeseries(well_df: pd.DataFrame, out_dir: Path, metric: str, ids_sorted: list[int], dates_by_id: dict[int, str], line_or_scatter: str):
    sub = metric.lower()
    plot_dir = out_dir / "plots" / line_or_scatter / sub
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_points_total = len(ids_sorted)
    fig_size = _auto_figsize_by_npoints(n_points_total)

    for well, dfw in well_df.groupby("well"):
        dfw = dfw.sort_values("id")
        x = dfw["id"].to_numpy(dtype=float)
        y = dfw[metric].to_numpy()

        if x.size == 0:
            continue

        fig, ax = plt.subplots(figsize=fig_size)
        if line_or_scatter == "line":
            if len(x) > 1:
                ax.plot(x, y, linewidth=LINE_WIDTH)
            ax.scatter(x, y, s=max(4, DOT_SIZE / 1.6), alpha=ALPHA)
        else:
            ax.scatter(x, y, s=DOT_SIZE, alpha=ALPHA)

        ax.set_xlabel("Meting (id)")
        ax.set_ylabel(metric)
        ax.set_title(f"Well {int(well)} – {metric}")

        # Use only the positions that exist for this well, but show every TICK_STEP-th tick
        tick_positions = list(x)
        set_xticks_subset(ax, tick_positions, step=TICK_STEP)

        # Give small padding so markers not right at edge
        min_x, max_x = float(x.min()), float(x.max())
        pad = 0.4
        ax.set_xlim(min_x - pad, max_x + pad)

        # No rotation per your request; horizontal labels (default)
        # Make sure there's space under the plot
        plt.subplots_adjust(bottom=0.18)

        fig.tight_layout()
        out_file = plot_dir / f"well_{int(well)}_{metric}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)


def plot_totals_over_measurements(well_df: pd.DataFrame, out_dir: Path, ids_sorted: list[int]):
    summary_dir = out_dir / "plots" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    totals = (
        well_df.groupby(["id", "date"], as_index=False)
               .agg(Total_Spikes=("Spikes_total", "sum"),
                    Total_Bursts=("Bursts_total", "sum"))
               .sort_values("id")
    )

    if totals.empty:
        return

    n_points = len(ids_sorted)
    fig_size = _auto_figsize_by_npoints(n_points)

    xs = totals["id"].to_numpy(dtype=float)

    # Totale Spikes
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(xs, totals["Total_Spikes"].to_numpy(), linewidth=LINE_WIDTH)
    ax.scatter(xs, totals["Total_Spikes"].to_numpy(), s=max(4, DOT_SIZE/1.6), alpha=ALPHA)
    ax.set_xlabel("Meting (id)")
    ax.set_ylabel("Totale Spikes (alle wells)")
    ax.set_title("Totale Spikes per meting")
    set_xticks_subset(ax, xs.tolist(), step=TICK_STEP)
    ax.set_xlim(xs.min() - 0.4, xs.max() + 0.4)
    plt.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(summary_dir / "total_spikes_over_measurements.png", dpi=150)
    plt.close(fig)

    # Totale Bursts
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(xs, totals["Total_Bursts"].to_numpy(), linewidth=LINE_WIDTH)
    ax.scatter(xs, totals["Total_Bursts"].to_numpy(), s=max(4, DOT_SIZE/1.6), alpha=ALPHA)
    ax.set_xlabel("Meting (id)")
    ax.set_ylabel("Totale Bursts (alle wells)")
    ax.set_title("Totale Bursts per meting")
    set_xticks_subset(ax, xs.tolist(), step=TICK_STEP)
    ax.set_xlim(xs.min() - 0.4, xs.max() + 0.4)
    plt.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(summary_dir / "total_bursts_over_measurements.png", dpi=150)
    plt.close(fig)


# ---------------- Main pipeline ----------------
def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest.json not found at {MANIFEST_PATH}. Place manifest (filenames only) next to the script.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    meas = manifest.get("measurements", [])
    grid = manifest.get("grid", {}) or {}
    wells = int(grid.get("wells", DEFAULT_WELLS))
    elecs = int(grid.get("electrodes_per_well", DEFAULT_ELECS))

    if not meas:
        raise ValueError("No measurements found in manifest.json (manifest['measurements'] is empty).")

    df_meas = pd.DataFrame(meas).sort_values("id")
    ids_sorted = df_meas["id"].astype(int).tolist()
    id_to_date = dict(zip(df_meas["id"].astype(int), df_meas["date"]))

    out_root = SCRIPT_DIR / "analyses"
    out_root.mkdir(parents=True, exist_ok=True)

    all_elec_rows = []
    all_well_rows = []

    for _, row in df_meas.iterrows():
        mid = int(row["id"])
        date_str = row.get("date")
        w_name = row.get("well_csv")
        e_name = row.get("electrode_csv")
        if not w_name or not e_name:
            print(f"[INFO] Skipping measurement id={mid} because filenames missing in manifest (well_csv/electrode_csv).")
            continue

        well_path = WELL_DIR / w_name
        elec_path = ELECTRODE_DIR / e_name

        if not well_path.exists():
            print(f"[WARN] Well CSV not found: {well_path} (skipping measurement id={mid})")
            continue
        if not elec_path.exists():
            print(f"[WARN] Electrode CSV not found: {elec_path} (skipping measurement id={mid})")
            continue

        df_well = read_well_csv(well_path)
        try:
            df_elec = read_electrode_csv(elec_path)
        except Exception as e:
            print(f"[WARN] Failed to parse electrode CSV '{elec_path}': {e}. Skipping measurement id={mid}.")
            continue

        df_elec_full = ensure_full_grid_with_presence(df_elec, wells, elecs)
        df_elec_full.insert(0, "id", mid)
        df_elec_full.insert(1, "date", date_str)

        all_elec_rows.append(
            df_elec_full[["id", "date", "Well", "Electrode", "Spikes", "Bursts", "present_in_file", "is_active"]].copy()
        )

        df_wsum = summarize_well(df_well, df_elec_full)
        df_wsum.rename(columns={"Well": "well"}, inplace=True)
        df_wsum.insert(0, "id", mid)
        df_wsum.insert(1, "date", date_str)
        if "Spikes_total" not in df_wsum.columns:
            df_wsum["Spikes_total"] = 0
        if "Bursts_total" not in df_wsum.columns:
            df_wsum["Bursts_total"] = 0
        all_well_rows.append(df_wsum[["id", "date", "well", "Spikes_total", "Bursts_total"]].copy())

    if not all_elec_rows or not all_well_rows:
        raise ValueError("No valid measurement data was processed. Check manifest filenames and folder paths.")

    elec_all = pd.concat(all_elec_rows, ignore_index=True).sort_values(["id", "Well", "Electrode"])
    elec_all.rename(columns={"Well": "well", "Electrode": "electrode"}, inplace=True)

    well_all = pd.concat(all_well_rows, ignore_index=True).sort_values(["id", "well"])

    elec_out = out_root / "electrode_activity_over_time.csv"
    well_out = out_root / "well_activity_over_time.csv"
    elec_all.to_csv(elec_out, index=False)
    well_all.to_csv(well_out, index=False)

    print(f"Saved electrode CSV: {elec_out}")
    print(f"Saved well CSV:      {well_out}")

    well_for_plots = well_all.rename(columns={"Spikes_total": "Spikes", "Bursts_total": "Bursts"})

    # Generate plots with every-2nd-tick and no rotation
    plot_per_well_timeseries(well_for_plots, out_root, "Spikes", ids_sorted, id_to_date, line_or_scatter="line")
    plot_per_well_timeseries(well_for_plots, out_root, "Spikes", ids_sorted, id_to_date, line_or_scatter="scatter")
    plot_per_well_timeseries(well_for_plots, out_root, "Bursts", ids_sorted, id_to_date, line_or_scatter="line")
    plot_per_well_timeseries(well_for_plots, out_root, "Bursts", ids_sorted, id_to_date, line_or_scatter="scatter")

    plot_totals_over_measurements(well_all, out_root, ids_sorted)

    print("✅ Analysis complete.")
    print(f"All outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
