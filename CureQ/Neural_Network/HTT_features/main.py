# plot_features_per_well.py
# ------------------------------------------------------------
# Vereisten:
#   pip install pandas matplotlib
#
# CSV verwacht kolommen:
#   'Dag' (int)          – tijd / meetdag
#   'Well' (int/str)     – well-ID
#   'Huntington' (str)   – bijv. 45-A, 45-B, 50-A, 50-B, CG-A, CG-B
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ========== Instellingen ==========
CSV_PATH = "Well_features_HTT.csv"   # pad naar CSV
ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR  = ROOT_DIR / "output_plots_HTT"

MAKE_PER_FEATURE = True     # 1 plot per feature (alle wells)
SPLIT_PER_WELL   = False    # aparte plots per (feature, well)
PLOT_SUBGROUPS   = False    # A/B-lijnen per well (vereist kolom 'Huntington')
SHOW_LEGEND_IF_WELLS_LEQ = 15
# =================================


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def set_xticks_unique_days(ax, days_series):
    """Zet x-ticks exact op de aanwezige dagen (uniek & gesorteerd)."""
    unique_days = sorted(pd.unique(days_series))
    ax.set_xticks(unique_days)
    # optioneel: alle labels als int tonen
    ax.set_xticklabels([str(int(d)) for d in unique_days])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Verplichte kolommen
    assert "Dag" in df.columns, "Kolom 'Dag' ontbreekt."
    assert "Well" in df.columns, "Kolom 'Well' ontbreekt."
    has_huntington = "Huntington" in df.columns

    # Dag naar int en rijen zonder dag weg
    df["Dag"] = pd.to_numeric(df["Dag"], errors="coerce")
    df = df.dropna(subset=["Dag"])
    df["Dag"] = df["Dag"].astype(int)

    # Feature-kolommen
    exclude = {"Dag", "Well"}
    if has_huntington:
        exclude.add("Huntington")
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("Geen numerieke feature-kolommen gevonden.")

    df = df.sort_values(["Well", "Dag"]).reset_index(drop=True)

    # ---------- 1) Eén plot per feature ----------
    if MAKE_PER_FEATURE:
        for feat in feature_cols:
            fig, ax = plt.subplots()
            grouped = df.groupby(["Well", "Dag"], as_index=False)[feat].mean()

            # Plot per well; alleen echte data
            for well_id, sub in grouped.groupby("Well"):
                sub = sub.dropna(subset=[feat])
                if sub.empty:
                    continue
                ax.plot(sub["Dag"], sub[feat], marker="o", label=f"Well {well_id}")

            ax.set_xlabel("Dag")
            ax.set_ylabel(feat)
            ax.set_title(f"Tijdreeks per well – {feat}")

            # X-ticks: uitsluitend dagen met data in deze plot
            days_with_data = grouped.dropna(subset=[feat])["Dag"]
            set_xticks_unique_days(ax, days_with_data)

            if grouped["Well"].nunique() <= SHOW_LEGEND_IF_WELLS_LEQ:
                ax.legend(ncol=2)

            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{sanitize(feat)}__all_wells.png", dpi=150)
            plt.close(fig)

    # ---------- 2) Aparte figuren per (feature, well) ----------
    if SPLIT_PER_WELL and not PLOT_SUBGROUPS:
        for feat in feature_cols:
            for well_id, sub in df.groupby("Well"):
                sub = sub.dropna(subset=[feat])
                if sub.empty:
                    continue
                sub = sub.groupby("Dag", as_index=False)[feat].mean()

                fig, ax = plt.subplots()
                ax.plot(sub["Dag"], sub[feat], marker="o")
                ax.set_xlabel("Dag")
                ax.set_ylabel(feat)
                ax.set_title(f"{feat} – Well {well_id}")

                # X-ticks: uitsluitend dagen met data voor deze well/feature
                set_xticks_unique_days(ax, sub["Dag"])

                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{sanitize(feat)}__well_{well_id}.png", dpi=150)
                plt.close(fig)

    # ---------- 3) A/B-lijnen per (feature, well) ----------
    if SPLIT_PER_WELL and PLOT_SUBGROUPS:
        if not has_huntington:
            raise ValueError("PLOT_SUBGROUPS=True vereist kolom 'Huntington'.")

        def to_side(x: str):
            s = str(x).upper().replace(" ", "").replace("_", "").replace("-", "")
            return s[-1] if s and s[-1] in ("A", "B") else None

        df["_side"] = df["Huntington"].map(to_side)

        for feat in feature_cols:
            for well_id, sub in df.groupby("Well"):
                sub = sub.dropna(subset=[feat])
                if sub.empty:
                    continue
                sub = sub.groupby(["Dag", "_side"], as_index=False)[feat].mean()

                fig, ax = plt.subplots()
                for side, sdata in sub.groupby("_side"):
                    if side not in ("A", "B"):
                        continue
                    ax.plot(sdata["Dag"], sdata[feat], marker="o", label=f"Side {side}")

                ax.set_xlabel("Dag")
                ax.set_ylabel(feat)
                ax.set_title(f"{feat} – Well {well_id} (A/B)")
                ax.legend()

                # X-ticks: uitsluitend dagen met data in deze well/feature (A/B samengenomen)
                set_xticks_unique_days(ax, sub["Dag"])

                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{sanitize(feet)}__well_{well_id}__AB.png", dpi=150)
                plt.close(fig)

    print(f"\n✅ Klaar. {len(feature_cols)} features gevonden: {feature_cols}")
    print(f"📁 Plots opgeslagen in: {OUT_DIR.resolve()}")
    print(f"Instellingen → MAKE_PER_FEATURE={MAKE_PER_FEATURE}, SPLIT_PER_WELL={SPLIT_PER_WELL}, PLOT_SUBGROUPS={PLOT_SUBGROUPS}")


if __name__ == "__main__":
    main()
