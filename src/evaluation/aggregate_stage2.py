"""Master Aggregator for Stage 2 Experimental Matrix.

Compiles:
1. Table 2: 12-Run Factorial Matrix (Arch x Loss x Seed) with Mean +/- Std across seeds.
2. Master TDI Table: Point slope and 95% Patient-Cluster Bootstrap CIs.
3. Recalibration Table: Temperature Scaling impact across future deployment bins.
4. Selective Abstention Table: Risk-coverage curves and class-conditional deferral.
5. Publication Figures (Fig 2, Fig 3, Fig 4).
"""

import os
import sys
import glob
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compile_metrics_matrix(reports_dir: str) -> pd.DataFrame:
    """Aggregates all individual model metrics files into a tidy master table."""
    table_dir = os.path.join(reports_dir, "tables")
    files = sorted(glob.glob(os.path.join(table_dir, "metrics_*.csv")))
    if not files:
        print(f"No metrics files found in {table_dir}")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    master_df = pd.concat(dfs, ignore_index=True)

    out_csv = os.path.join(table_dir, "master_metrics_all_runs.csv")
    master_df.to_csv(out_csv, index=False)
    print(f"✅ Saved master metrics archive ({len(master_df)} rows) to: {out_csv}")

    # Summary table: Mean +/- Std across seeds
    group_cols = ["architecture", "loss_type", "temporal_bin", "target"]
    agg_dict = {
        "auroc": ["mean", "std"],
        "brier_score": ["mean", "std"],
        "ece_equal_mass": ["mean", "std"],
    }
    summary = master_df.groupby(group_cols).agg(agg_dict).reset_index()
    # Flatten MultiIndex columns
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary.columns]
    
    summary_csv = os.path.join(table_dir, "table2_matrix_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved Table 2 matrix summary to: {summary_csv}")
    return summary


def compile_tdi_inference(reports_dir: str) -> pd.DataFrame:
    """Aggregates all 2,000-replicate patient-cluster bootstrap TDI tables."""
    table_dir = os.path.join(reports_dir, "tables")
    files = sorted(glob.glob(os.path.join(table_dir, "tdi_*.csv")))
    # Exclude summary files if re-running
    files = [f for f in files if not os.path.basename(f).startswith("master_") and not os.path.basename(f).startswith("table")]
    if not files:
        print(f"No TDI inference files found in {table_dir}")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    master_tdi = pd.concat(dfs, ignore_index=True)

    out_csv = os.path.join(table_dir, "master_tdi_all_runs.csv")
    master_tdi.to_csv(out_csv, index=False)
    print(f"✅ Saved master TDI inference archive ({len(master_tdi)} rows) to: {out_csv}")

    # Format human-readable summary table with 95% CIs
    master_tdi["tdi_ci_formatted"] = master_tdi.apply(
        lambda r: f"{r['point_tdi']:.4f} [{r['bootstrap_ci_lower']:.4f}, {r['bootstrap_ci_upper']:.4f}]" if not np.isnan(r['point_tdi']) else "N/A",
        axis=1
    )
    
    summary_tdi_csv = os.path.join(table_dir, "table_tdi_summary.csv")
    master_tdi.to_csv(summary_tdi_csv, index=False)
    print(f"✅ Saved TDI summary table to: {summary_tdi_csv}")
    return master_tdi


def compile_mitigation_results(reports_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates recalibration and selective prediction results."""
    mit_dir = os.path.join(reports_dir, "mitigation")
    table_dir = os.path.join(reports_dir, "tables")

    recalib_files = sorted(glob.glob(os.path.join(mit_dir, "recalibration_results_*.csv")))
    sel_files = sorted(glob.glob(os.path.join(mit_dir, "selective_prediction_*.csv")))

    recalib_df = pd.DataFrame()
    sel_df = pd.DataFrame()

    if recalib_files:
        recalib_df = pd.concat([pd.read_csv(f) for f in recalib_files], ignore_index=True)
        out_recalib = os.path.join(table_dir, "master_recalibration_summary.csv")
        recalib_df.to_csv(out_recalib, index=False)
        print(f"✅ Saved master recalibration table to: {out_recalib}")

    if sel_files:
        sel_df = pd.concat([pd.read_csv(f) for f in sel_files], ignore_index=True)
        out_sel = os.path.join(table_dir, "master_selective_prediction_summary.csv")
        sel_df.to_csv(out_sel, index=False)
        print(f"✅ Saved master selective prediction table to: {out_sel}")

    return recalib_df, sel_df


def generate_publication_figures(reports_dir: str):
    """Generates Figure 2, Figure 3, and Figure 4 with high-resolution formatting."""
    fig_dir = os.path.join(reports_dir, "figures")
    table_dir = os.path.join(reports_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)

    metrics_csv = os.path.join(table_dir, "master_metrics_all_runs.csv")
    recalib_csv = os.path.join(table_dir, "master_recalibration_summary.csv")
    sel_csv = os.path.join(table_dir, "master_selective_prediction_summary.csv")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.dpi": 300})

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Drift Trajectories (AUROC & ECE) across Future Bins
    # ──────────────────────────────────────────────────────────────────────────
    if os.path.isfile(metrics_csv):
        df = pd.read_csv(metrics_csv)
        effusion_df = df[df["target"] == "Pleural Effusion"].copy()

        if not effusion_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

            # Left: Discrimination (AUROC)
            sns.lineplot(
                data=effusion_df,
                x="temporal_bin",
                y="auroc",
                hue="architecture",
                style="loss_type",
                markers=True,
                dashes=False,
                err_style="bars",
                errorbar="sd",
                ax=axes[0],
                palette=["#1f77b4", "#ff7f0e"]
            )
            axes[0].set_title("(a) Discrimination Drift: Pleural Effusion AUROC", fontweight="bold")
            axes[0].set_ylabel("Test AUROC (Mean ± SD)")
            axes[0].set_xlabel("Chronological Deployment Strata")
            axes[0].set_ylim(0.70, 0.98)

            # Right: Calibration (Equal-Mass ECE)
            sns.lineplot(
                data=effusion_df,
                x="temporal_bin",
                y="ece_equal_mass",
                hue="architecture",
                style="loss_type",
                markers=True,
                dashes=False,
                err_style="bars",
                errorbar="sd",
                ax=axes[1],
                palette=["#1f77b4", "#ff7f0e"]
            )
            axes[1].set_title("(b) Calibration Degradation: Equal-Mass ECE", fontweight="bold")
            axes[1].set_ylabel("Expected Calibration Error (ECE)")
            axes[1].set_xlabel("Chronological Deployment Strata")

            for ax in axes:
                ax.legend(frameon=True, framealpha=0.9)

            fig.tight_layout()
            fig2_path = os.path.join(fig_dir, "fig2_drift_trajectories.png")
            fig.savefig(fig2_path, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Saved Figure 2 to: {fig2_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Recalibration Comparison (Raw vs Temperature Scaling)
    # ──────────────────────────────────────────────────────────────────────────
    if os.path.isfile(recalib_csv):
        r_df = pd.read_csv(recalib_csv)
        eff_r = r_df[r_df["target"] == "Pleural Effusion"].copy()

        if not eff_r.empty:
            melted = pd.melt(
                eff_r,
                id_vars=["temporal_bin", "architecture"],
                value_vars=["ece_raw_mass", "ece_calib_mass"],
                var_name="Calibration",
                value_name="ECE"
            )
            melted["Calibration"] = melted["Calibration"].replace({
                "ece_raw_mass": "Uncalibrated Raw",
                "ece_calib_mass": "Temperature Scaled (T*)"
            })

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=melted,
                x="temporal_bin",
                y="ECE",
                hue="Calibration",
                errorbar="sd",
                palette=["#d62728", "#2ca02c"],
                ax=ax
            )
            ax.set_title("Post-Hoc Recalibration Failure Across Temporal Shift", fontweight="bold")
            ax.set_ylabel("Equal-Mass ECE (Lower is Better)")
            ax.set_xlabel("Chronological Deployment Strata")
            ax.legend(frameon=True, framealpha=0.9)

            fig.tight_layout()
            fig3_path = os.path.join(fig_dir, "fig3_recalibration_comparison.png")
            fig.savefig(fig3_path, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Saved Figure 3 to: {fig3_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 4: Selective Prediction / Abstention Curves (Risk-Coverage)
    # ──────────────────────────────────────────────────────────────────────────
    if os.path.isfile(sel_csv):
        s_df = pd.read_csv(sel_csv)
        eff_s = s_df[s_df["target"] == "Pleural Effusion"].copy()

        if not eff_s.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            # Left: AUROC vs Coverage
            sns.lineplot(
                data=eff_s,
                x="coverage",
                y="auroc",
                hue="temporal_bin",
                marker="o",
                errorbar="sd",
                palette="viridis",
                ax=axes[0]
            )
            axes[0].set_title("(a) Retained Discrimination vs Coverage", fontweight="bold")
            axes[0].set_xlabel("Retention Coverage (1.0 = Full Cohort, 0.7 = Defer 30%)")
            axes[0].set_ylabel("Retained AUROC")
            axes[0].set_xlim(1.02, 0.48)  # Invert x-axis to show increasing deferral

            # Right: Brier Score vs Coverage
            sns.lineplot(
                data=eff_s,
                x="coverage",
                y="brier_score",
                hue="temporal_bin",
                marker="s",
                errorbar="sd",
                palette="viridis",
                ax=axes[1]
            )
            axes[1].set_title("(b) Probability Error Reduction vs Coverage", fontweight="bold")
            axes[1].set_xlabel("Retention Coverage (1.0 = Full Cohort, 0.7 = Defer 30%)")
            axes[1].set_ylabel("Retained Brier Score (Lower is Better)")
            axes[1].set_xlim(1.02, 0.48)

            for ax in axes:
                ax.legend(frameon=True, framealpha=0.9)

            fig.tight_layout()
            fig4_path = os.path.join(fig_dir, "fig4_selective_risk_coverage.png")
            fig.savefig(fig4_path, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Saved Figure 4 to: {fig4_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage 2 Matrix and Generate Manuscript Artifacts")
    parser.add_argument("--reports-dir", type=str, default="reports/stage2", help="Root reports directory")
    args = parser.parse_args()

    print("==============================================================================")
    print("  STAGE 2 MASTER REPORT AGGREGATOR & FIGURE GENERATOR")
    print("==============================================================================")
    compile_metrics_matrix(args.reports_dir)
    compile_tdi_inference(args.reports_dir)
    compile_mitigation_results(args.reports_dir)
    generate_publication_figures(args.reports_dir)
    print("\n🎉 ALL MASTER TABLES AND PUBLICATION FIGURES SUCCESSFULLY COMPILED!")


if __name__ == "__main__":
    main()
