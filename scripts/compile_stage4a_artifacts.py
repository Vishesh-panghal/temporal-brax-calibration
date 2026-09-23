#!/usr/bin/env python3
"""
Compilation script for MIMIC-CXR Stage 4A publication artifacts.
Generates:
1. LaTeX and Markdown tables for external validation results
2. View-specific stratification tables (AP vs PA)
3. Calibration reliability plots (Raw vs Corrected)
4. Comprehensive executive summary report (STAGE_4A_REPORT.md)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGETS = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]


def generate_summary_tables(table_dir: Path):
    metrics_file = table_dir / "mimic_stage4a_metrics_all.csv"
    if not metrics_file.exists():
        print(f"⚠️ Metrics file {metrics_file} not found. Skipping table generation.")
        return

    df = pd.read_csv(metrics_file)

    # Filter to all_frontal stratum for main table
    main_df = df[df["stratum"] == "all_frontal"].copy()

    # Aggregate across 3 seeds per architecture/loss/eval_mode/target
    group_cols = ["architecture", "loss_type", "eval_mode", "target"]
    num_cols = ["auroc", "auprc", "brier_score", "calibration_intercept", "calibration_slope", "ece_equal_mass"]

    agg_df = main_df.groupby(group_cols)[num_cols].agg(["mean", "std"]).reset_index()

    # Format LaTeX & Markdown table
    table_rows = []
    for _, row in agg_df.iterrows():
        arch = row["architecture"].iloc[0] if isinstance(row["architecture"], pd.Series) else row["architecture"]
        loss = row["loss_type"].iloc[0] if isinstance(row["loss_type"], pd.Series) else row["loss_type"]
        mode = row["eval_mode"].iloc[0] if isinstance(row["eval_mode"], pd.Series) else row["eval_mode"]
        tgt = row["target"].iloc[0] if isinstance(row["target"], pd.Series) else row["target"]

        auroc_m = row[("auroc", "mean")]
        auroc_s = row[("auroc", "std")]
        brier_m = row[("brier_score", "mean")]
        brier_s = row[("brier_score", "std")]
        cal_int_m = row[("calibration_intercept", "mean")]
        cal_int_s = row[("calibration_intercept", "std")]
        cal_slp_m = row[("calibration_slope", "mean")]
        cal_slp_s = row[("calibration_slope", "std")]

        table_rows.append({
            "Architecture": arch,
            "Loss": loss,
            "Mode": mode,
            "Target": tgt,
            "AUROC": f"{auroc_m:.3f} ± {auroc_s:.3f}",
            "Brier Score": f"{brier_m:.4f} ± {brier_s:.4f}",
            "Calib Intercept": f"{cal_int_m:+.3f} ± {cal_int_s:.3f}",
            "Calib Slope": f"{cal_slp_m:.3f} ± {cal_slp_s:.3f}",
        })

    summary_table_df = pd.DataFrame(table_rows)
    csv_out = table_dir / "mimic_external_validation_summary.csv"
    summary_table_df.to_csv(csv_out, index=False)
    print(f"✅ Generated summary table: {csv_out}")

    # Generate LaTeX version
    tex_out = table_dir / "mimic_external_validation_table.tex"
    with open(tex_out, "w") as f:
        f.write("% Table: External Validation of BRAX Models on MIMIC-CXR-JPG (Frontal Test Cohort)\n")
        f.write(summary_table_df.to_latex(index=False))
    print(f"✅ Generated LaTeX table: {tex_out}")


def generate_reliability_plots(pred_dir: Path, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_files = list(pred_dir.glob("preds_mimic_*.csv"))
    if not pred_files:
        print("⚠️ No prediction files found. Skipping plot generation.")
        return

    # Choose DenseNet-121 weighted model prediction file
    weighted_preds = [f for f in pred_files if "weighted" in f.name and "unweighted" not in f.name]
    if not weighted_preds:
        return

    pred_df = pd.read_csv(weighted_preds[0])
    print(f"Generating reliability diagrams from {weighted_preds[0].name}...")

    plt.figure(figsize=(14, 10))
    for idx, tgt in enumerate(TARGETS):
        sub = pred_df[pred_df["target"] == tgt]
        if len(sub) == 0:
            continue

        plt.subplot(2, 2, idx + 1)
        y_true = sub["y_true"].values
        p_raw = sub["prob_raw"].values
        p_corr = sub["prob_corrected"].values

        bins = np.linspace(0, 1, 11)
        # Compute bin accuracies
        bin_raw_acc, bin_raw_conf = [], []
        bin_corr_acc, bin_corr_conf = [], []

        for b in range(10):
            lo, hi = bins[b], bins[b+1]
            m_raw = (p_raw >= lo) & (p_raw < hi)
            m_corr = (p_corr >= lo) & (p_corr < hi)

            if np.sum(m_raw) > 0:
                bin_raw_acc.append(np.mean(y_true[m_raw]))
                bin_raw_conf.append(np.mean(p_raw[m_raw]))
            if np.sum(m_corr) > 0:
                bin_corr_acc.append(np.mean(y_true[m_corr]))
                bin_corr_conf.append(np.mean(p_corr[m_corr]))

        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        if bin_raw_conf:
            plt.plot(bin_raw_conf, bin_raw_acc, "s-", color="#d62728", label="Weighted Raw", alpha=0.8)
        if bin_corr_conf:
            plt.plot(bin_corr_conf, bin_corr_acc, "o-", color="#2ca02c", label="Analytic Corrected", alpha=0.8)

        plt.title(f"{tgt} (MIMIC External)", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Predicted Probability", fontsize=10)
        plt.ylabel("Observed Empirical Proportion", fontsize=10)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_out = fig_dir / "mimic_stage4a_reliability_curves.png"
    plt.savefig(fig_out, dpi=300)
    plt.close()
    print(f"✅ Generated calibration figure: {fig_out}")


def main():
    base_dir = Path("reports/stage4a")
    table_dir = base_dir / "tables"
    fig_dir = base_dir / "figures"
    pred_dir = base_dir / "predictions"

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 Compiling Stage 4A Publication Artifacts...")
    generate_summary_tables(table_dir)
    generate_reliability_plots(pred_dir, fig_dir)


if __name__ == "__main__":
    main()
