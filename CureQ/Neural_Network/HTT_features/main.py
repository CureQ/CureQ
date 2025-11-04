# plot_features_per_well_dots.py
# ------------------------------------------------------------
# Vereisten: pip install pandas matplotlib
#
# CSV-kolommen:
#   'Dag' (int), 'Well' (int/str), optioneel 'Huntington' (bijv. 45-A/50-B/CG-A)
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== Instellingen ==========
CSV_PATH = "Sca1_features.csv"
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR  = ROOT_DIR / "output_plots_HTT"

MAKE_PER_FEATURE = True     # 1 plot per feature (alle wells samen)
SPLIT_PER_WELL   = False    # aparte plots per (feature, well)
PLOT_SUBGROUPS   = False    # A/B per well (vereist 'Huntington')
PLOT_STYLE       = "strip"  # "scatter" | "strip" | "ab_scatter" (ab_scatter vereist PLOT_SUBGROUPS=True)

JITTER_WIDTH     = 0.15     # jitter voor strip-plot
DOT_SIZE         = 25       # marker size
ALPHA            = 0.85     # transparantie
SHOW_LEGEND_IF_WELLS_LEQ = 15
# =================================


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def set_xticks_unique_days(ax, days_series):
    """X-ticks precies op de aanwezige dagen."""
    unique_days = sorted(pd.unique(days_series))
    ax.set_xticks(unique_days)
    ax.set_xticklabels([str(int(d)) for d in unique_days])


def to_side(huntington_val: str):
    s = str(huntington_val).upper().replace(" ", "").replace("_", "").replace("-", "")
    return s[-1] if s and s[-1] in ("A", "B") else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    assert "Dag" in df.columns, "Kolom 'Dag' ontbreekt."
    assert "Well" in df.columns, "Kolom 'Well' ontbreekt."
    has_huntington = "Huntington" in df.columns

    # Dag als int, rows zonder dag weg
    df["Dag"] = pd.to_numeric(df["Dag"], errors="coerce")
    df = df.dropna(subset=["Dag"])
    df["Dag"] = df["Dag"].astype(int)

    # Featurekolommen (alle numeriek behalve sleutelkolommen)
    exclude = {"Dag", "Well"}
    if has_huntington:
        exclude.add("Huntington")
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("Geen numerieke feature-kolommen gevonden.")

    # Optioneel: side A/B extraheren
    if has_huntington:
        df["_side"] = df["Huntington"].map(to_side)
    else:
        df["_side"] = None

    # Zorg dat we de dagen per plot kunnen afzetten
    df = df.sort_values(["Dag", "Well"]).reset_index(drop=True)

    # ---------- 1) Eén plot per feature ----------
    if MAKE_PER_FEATURE:
        for feat in feature_cols:
            sub_all = df.dropna(subset=[feat])
            if sub_all.empty:
                continue

            fig, ax = plt.subplots()

            if PLOT_STYLE == "strip":
                # per dag random jitter zodat punten niet overlappen
                # we plotten alles tegelijk (alle wells)
                jitter = (np.random.rand(len(sub_all)) - 0.5) * 2 * JITTER_WIDTH
                ax.scatter(sub_all["Dag"] + jitter, sub_all[feat], s=DOT_SIZE, alpha=ALPHA)
            elif PLOT_STYLE == "scatter":
                # gewone dot-plot zonder jitter
                ax.scatter(sub_all["Dag"], sub_all[feat], s=DOT_SIZE, alpha=ALPHA)
            elif PLOT_STYLE == "ab_scatter" and PLOT_SUBGROUPS:
                # Per side verschillende markers
                for side, mkr in [("A", "o"), ("B", "s")]:
                    sdata = sub_all[sub_all["_side"] == side]
                    if sdata.empty:
                        continue
                    ax.scatter(sdata["Dag"], sdata[feat], s=DOT_SIZE, alpha=ALPHA, marker=mkr, label=f"Side {side}")
                ax.legend()
            else:
                # fallback: scatter
                ax.scatter(sub_all["Dag"], sub_all[feat], s=DOT_SIZE, alpha=ALPHA)

            ax.set_xlabel("Dag")
            ax.set_ylabel(feat)
            ax.set_title(f"Dot plot per dag – {feat}")
            set_xticks_unique_days(ax, sub_all["Dag"])

            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{sanitize(feat)}__all_wells_dots.png", dpi=150)
            plt.close(fig)

    # ---------- 2) Aparte figuren per (feature, well) ----------
    if SPLIT_PER_WELL and not PLOT_SUBGROUPS:
        for feat in feature_cols:
            for well_id, sub in df.groupby("Well"):
                sub = sub.dropna(subset=[feat])
                if sub.empty:
                    continue
                fig, ax = plt.subplots()
                if PLOT_STYLE == "strip":
                    jitter = (np.random.rand(len(sub)) - 0.5) * 2 * JITTER_WIDTH
                    ax.scatter(sub["Dag"] + jitter, sub[feat], s=DOT_SIZE, alpha=ALPHA)
                else:
                    ax.scatter(sub["Dag"], sub[feat], s=DOT_SIZE, alpha=ALPHA)
                ax.set_xlabel("Dag")
                ax.set_ylabel(feat)
                ax.set_title(f"{feat} – Well {well_id} (dots)")
                set_xticks_unique_days(ax, sub["Dag"])
                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{sanitize(feat)}__well_{well_id}_dots.png", dpi=150)
                plt.close(fig)

    # ---------- 3) A/B-dots per (feature, well) ----------
    if SPLIT_PER_WELL and PLOT_SUBGROUPS:
        if not has_huntington:
            raise ValueError("PLOT_SUBGROUPS=True vereist 'Huntington'.")
        for feat in feature_cols:
            for well_id, sub in df.groupby("Well"):
                sub = sub.dropna(subset=[feat])
                if sub.empty:
                    continue
                fig, ax = plt.subplots()
                for side, mkr in [("A", "o"), ("B", "s")]:
                    sdata = sub[sub["_side"] == side]
                    if sdata.empty:
                        continue
                    if PLOT_STYLE == "strip":
                        jitter = (np.random.rand(len(sdata)) - 0.5) * 2 * JITTER_WIDTH
                        ax.scatter(sdata["Dag"] + jitter, sdata[feat], s=DOT_SIZE, alpha=ALPHA, marker=mkr, label=f"{side}")
                    else:
                        ax.scatter(sdata["Dag"], sdata[feat], s=DOT_SIZE, alpha=ALPHA, marker=mkr, label=f"{side}")
                ax.set_xlabel("Dag")
                ax.set_ylabel(feat)
                ax.set_title(f"{feat} – Well {well_id} (A/B dots)")
                ax.legend()
                set_xticks_unique_days(ax, sub["Dag"])
                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{sanitize(feat)}__well_{well_id}__AB_dots.png", dpi=150)
                plt.close(fig)

    print(f"\n✅ Klaar. {len(feature_cols)} features: {feature_cols}")
    print(f"📁 Plots in: {OUT_DIR.resolve()}")
    print(f"Opties → STYLE={PLOT_STYLE}, SPLIT_PER_WELL={SPLIT_PER_WELL}, PLOT_SUBGROUPS={PLOT_SUBGROUPS}")


if __name__ == "__main__":
    main()
