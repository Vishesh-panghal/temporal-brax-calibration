"""
Patient-Clustered Bootstrap Confidence Intervals for MIMIC-CXR Stage 4A External Validation.
Resamples at the unique patient level (subject_id) to preserve intra-patient correlation,
generating 2,000-replicate 95% percentile confidence intervals for all discrimination and
calibration metrics, plus paired differences (Raw vs Analytic Corrected).
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_all_metrics


def bootstrap_patient_clusters(
    df: pd.DataFrame,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Performs patient-clustered bootstrap resampling.
    df must contain columns: patient_id, y_true, prob_raw, prob_corrected (optional).
    """
    rng = np.random.default_rng(seed)
    unique_patients = df["patient_id"].unique()
    n_patients = len(unique_patients)

    # Pre-index row indices by patient
    patient_to_rows = defaultdict(list)
    for idx, p in enumerate(df["patient_id"].values):
        patient_to_rows[p].append(idx)
    patient_row_lists = [np.array(patient_to_rows[p], dtype=int) for p in unique_patients]

    y_true_arr = df["y_true"].values.astype(float)
    p_raw_arr = df["prob_raw"].values.astype(float)
    has_corrected = "prob_corrected" in df.columns and not df["prob_corrected"].isna().all()
    p_corr_arr = df["prob_corrected"].values.astype(float) if has_corrected else None

    # Storage for replicate metrics
    boot_metrics_raw = defaultdict(list)
    boot_metrics_corr = defaultdict(list)
    delta_brier = []
    delta_intercept = []

    for _ in range(n_bootstraps):
        sampled_pt_idx = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_rows = np.concatenate([patient_row_lists[i] for i in sampled_pt_idx])

        y_boot = y_true_arr[sampled_rows]
        if len(np.unique(y_boot)) < 2:
            continue

        # Raw metrics
        praw_boot = p_raw_arr[sampled_rows]
        m_raw = compute_all_metrics(y_boot, praw_boot)
        for k, v in m_raw.items():
            if not np.isnan(v):
                boot_metrics_raw[k].append(v)

        # Corrected metrics & paired deltas
        if has_corrected:
            pcorr_boot = p_corr_arr[sampled_rows]
            m_corr = compute_all_metrics(y_boot, pcorr_boot)
            for k, v in m_corr.items():
                if not np.isnan(v):
                    boot_metrics_corr[k].append(v)

            if "brier_score" in m_raw and "brier_score" in m_corr:
                delta_brier.append(m_corr["brier_score"] - m_raw["brier_score"])
            if "calibration_intercept" in m_raw and "calibration_intercept" in m_corr:
                delta_intercept.append(m_corr["calibration_intercept"] - m_raw["calibration_intercept"])

    # Compute percentiles
    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    summary = {}
    for k, vals in boot_metrics_raw.items():
        if len(vals) > 0:
            summary[f"raw_{k}_mean"] = float(np.mean(vals))
            summary[f"raw_{k}_ci_lower"] = float(np.percentile(vals, lower_pct))
            summary[f"raw_{k}_ci_upper"] = float(np.percentile(vals, upper_pct))

    if has_corrected:
        for k, vals in boot_metrics_corr.items():
            if len(vals) > 0:
                summary[f"corr_{k}_mean"] = float(np.mean(vals))
                summary[f"corr_{k}_ci_lower"] = float(np.percentile(vals, lower_pct))
                summary[f"corr_{k}_ci_upper"] = float(np.percentile(vals, upper_pct))

        if len(delta_brier) > 0:
            summary["delta_brier_mean"] = float(np.mean(delta_brier))
            summary["delta_brier_ci_lower"] = float(np.percentile(delta_brier, lower_pct))
            summary["delta_brier_ci_upper"] = float(np.percentile(delta_brier, upper_pct))
        if len(delta_intercept) > 0:
            summary["delta_intercept_mean"] = float(np.mean(delta_intercept))
            summary["delta_intercept_ci_lower"] = float(np.percentile(delta_intercept, lower_pct))
            summary["delta_intercept_ci_upper"] = float(np.percentile(delta_intercept, upper_pct))

    summary["n_valid_bootstraps"] = len(boot_metrics_raw.get("auroc", []))
    return summary


def run_bootstrap_analysis(
    pred_dir: str = "reports/stage4a/predictions",
    output_dir: str = "reports/stage4a/tables",
    n_bootstraps: int = 2000,
):
    pred_path = Path(pred_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_path.glob("preds_mimic_*.csv"))
    if not pred_files:
        print(f"❌ No prediction files found in '{pred_dir}'. Run evaluation first.")
        return

    print(f"Running {n_bootstraps}-replicate patient-clustered bootstrap across {len(pred_files)} prediction files...")

    all_bootstrap_records = []

    for f in tqdm(pred_files, desc="Bootstrap files"):
        df = pd.read_csv(f)
        targets = df["target"].unique()
        arch = df["architecture"].iloc[0]
        loss_type = df["loss_type"].iloc[0]
        seed = df["seed"].iloc[0]
        run_id = df["run_id"].iloc[0]

        strata = [
            ("all_frontal", None),
            ("ap_view", "AP"),
            ("pa_view", "PA"),
        ]

        for stratum_name, view_filter in strata:
            if view_filter is None:
                sub_df = df
            else:
                sub_df = df[df["view_position"] == view_filter]

            for tgt in targets:
                tgt_df = sub_df[sub_df["target"] == tgt]
                if len(tgt_df) == 0:
                    continue

                res = bootstrap_patient_clusters(tgt_df, n_bootstraps=n_bootstraps)
                res.update({
                    "run_id": run_id,
                    "architecture": arch,
                    "loss_type": loss_type,
                    "seed": seed,
                    "target": tgt,
                    "stratum": stratum_name,
                    "sample_count": len(tgt_df),
                    "unique_patients": tgt_df["patient_id"].nunique(),
                })
                all_bootstrap_records.append(res)

    boot_df = pd.DataFrame(all_bootstrap_records)
    out_file = out_path / "mimic_stage4a_bootstrap_ci.csv"
    boot_df.to_csv(out_file, index=False)
    print(f"\n✅ Saved Stage 4A Patient-Cluster Bootstrap CIs to: {out_file} ({len(boot_df):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default="reports/stage4a/predictions")
    parser.add_argument("--output-dir", type=str, default="reports/stage4a/tables")
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    run_bootstrap_analysis(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
    )
