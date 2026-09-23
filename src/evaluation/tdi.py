"""Temporal Decay Index (TDI) formulation and patient-level bootstrap inference."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from src.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_all_metrics,
)


LOSS_METRICS = {"brier_score", "log_loss", "ece"}
BENEFIT_METRICS = {"auroc", "auprc"}


def fit_temporal_decay_index(
    time_points: Union[List[float], np.ndarray],
    metric_values: Union[List[float], np.ndarray],
    metric_name: str,
) -> Dict[str, float]:
    """Fits linear drift trajectory S(D_t) = alpha + beta * time_t + error.
    
    Standardized TDI sign convention:
        TDI = beta for loss-like metrics (Brier, Log Loss, ECE)
        TDI = -beta for benefit-like metrics (AUROC, AUPRC)
    Positive TDI always indicates performance degradation.
    """
    t = np.asarray(time_points, dtype=float)
    y = np.asarray(metric_values, dtype=float)

    # Filter out NaNs
    valid = ~np.isnan(t) & ~np.isnan(y)
    t = t[valid]
    y = y[valid]

    if len(t) < 2:
        return {
            "alpha": float("nan"),
            "beta": float("nan"),
            "tdi": float("nan"),
            "r_squared": float("nan"),
            "p_value": float("nan"),
        }

    slope, intercept, r_val, p_val, std_err = stats.linregress(t, y)

    # Determine TDI sign
    name_lower = metric_name.lower()
    if name_lower in LOSS_METRICS or "brier" in name_lower or "ece" in name_lower or "loss" in name_lower:
        tdi = float(slope)
    else:
        # Benefit-like metric (AUROC, AUPRC, Sensitivity, etc.)
        tdi = float(-slope)

    return {
        "alpha": float(intercept),
        "beta": float(slope),
        "tdi": tdi,
        "r_squared": float(r_val ** 2),
        "p_value": float(p_val),
        "std_err": float(std_err),
    }


# Metrics that require both classes present
DISCRIMINATION_METRICS = {"auroc", "auprc"}
# Metrics that require at least one positive or one negative
CLASS_CONDITIONAL_METRICS = {"sensitivity", "specificity"}
# Metrics computable with any class distribution (including single-class)
PROPER_SCORE_METRICS = {"brier", "brier_score", "ece", "ece_equal_mass", "ece_mass", "ece_equal_width", "log_loss"}


def requires_both_classes(metric_name: str) -> bool:
    """Returns True if the metric requires both positive and negative examples."""
    name_lower = metric_name.lower()
    # AUROC, AUPRC need both classes
    if any(d in name_lower for d in ['auroc', 'auprc']):
        return True
    # Sensitivity needs positives, specificity needs negatives — but these
    # are handled inline. For the TDI bootstrap, treat them as class-conditional.
    if 'sensitivity' in name_lower or 'specificity' in name_lower:
        return True
    return False


def compute_single_metric_fast(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
) -> float:
    """High-speed single metric computation avoiding unneeded joint model fitting.
    
    Brier, ECE, and log-loss are computable in single-class bins.
    AUROC, AUPRC, sensitivity, and specificity require appropriate class presence.
    """
    if len(y_true) == 0:
        return float("nan")

    name_lower = metric_name.lower()
    if "brier" in name_lower:
        return float(np.mean((y_prob - y_true) ** 2))
    elif "log_loss" in name_lower:
        eps = 1e-7
        p = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
    elif "ece_equal_mass" in name_lower or "ece_mass" in name_lower:
        return compute_expected_calibration_error(y_true, y_prob, strategy="equal_mass")
    elif "ece" in name_lower:
        return compute_expected_calibration_error(y_true, y_prob, strategy="equal_width")
    elif "sensitivity" in name_lower:
        pos_mask = (y_true == 1)
        n_pos = int(np.sum(pos_mask))
        if n_pos == 0:
            return float("nan")
        return float(np.sum(y_prob[pos_mask] >= 0.5) / n_pos)
    elif "specificity" in name_lower:
        neg_mask = (y_true == 0)
        n_neg = int(np.sum(neg_mask))
        if n_neg == 0:
            return float("nan")
        return float(np.sum(y_prob[neg_mask] < 0.5) / n_neg)
    elif "auprc" in name_lower:
        if len(np.unique(y_true)) > 1:
            return float(average_precision_score(y_true, y_prob))
        return float("nan")
    elif "auroc" in name_lower:
        if len(np.unique(y_true)) > 1:
            return float(roc_auc_score(y_true, y_prob))
        return float("nan")

    return float("nan")


def bootstrap_tdi_confidence_interval(
    predictions_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "temporal_bin_idx",
    target_col: str = "y_true",
    pred_col: str = "prob_raw",
    metric_name: str = "auroc",
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Computes cluster bootstrap confidence intervals for TDI at the patient level.
    
    Resampling is strictly performed on unique PatientIDs to preserve intra-patient dependencies.
    """
    rng = np.random.default_rng(random_state)
    unique_patients = predictions_df[patient_col].unique()
    n_patients = len(unique_patients)

    unique_times = sorted(predictions_df[time_col].unique())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    
    # Pre-extract numpy arrays for high-speed resampling
    patients_arr = predictions_df[patient_col].values
    time_indices = np.array([time_to_idx[t] for t in predictions_df[time_col].values], dtype=int)
    y_true_arr = predictions_df[target_col].values.astype(float)
    y_prob_arr = predictions_df[pred_col].values.astype(float)

    # Pre-group row indices by unique patient
    from collections import defaultdict
    patient_to_indices = defaultdict(list)
    for row_idx, p in enumerate(patients_arr):
        patient_to_indices[p].append(row_idx)
    patient_indices_list = [np.array(patient_to_indices[p], dtype=int) for p in unique_patients]

    bootstrap_tdis = []
    unique_t_indices = list(range(len(unique_times)))

    for _ in range(n_bootstraps):
        sampled_pt_idx = rng.choice(len(unique_patients), size=n_patients, replace=True)
        sampled_rows = np.concatenate([patient_indices_list[i] for i in sampled_pt_idx])

        sampled_t = time_indices[sampled_rows]
        sampled_y = y_true_arr[sampled_rows]
        sampled_p = y_prob_arr[sampled_rows]

        metric_vals = []
        needs_both = requires_both_classes(metric_name)
        for t_idx in unique_t_indices:
            mask = (sampled_t == t_idx)
            yt_bin = sampled_y[mask]
            yp_bin = sampled_p[mask]

            if len(yt_bin) == 0:
                metric_vals.append(float("nan"))
                continue

            # Only require both classes for discrimination metrics (AUROC, AUPRC, etc.)
            # Brier, ECE, log-loss are computable even in single-class bins
            if needs_both and len(np.unique(yt_bin)) < 2:
                metric_vals.append(float("nan"))
                continue

            val = compute_single_metric_fast(yt_bin, yp_bin, metric_name)
            metric_vals.append(val)

        tdi_result = fit_temporal_decay_index(unique_times, metric_vals, metric_name)
        if not np.isnan(tdi_result["tdi"]):
            bootstrap_tdis.append(tdi_result["tdi"])

    if len(bootstrap_tdis) == 0:
        return {"tdi_mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_valid_bootstraps": 0}

    alpha_ci = (1.0 - ci_level) / 2.0
    lower_pct = alpha_ci * 100
    upper_pct = (1.0 - alpha_ci) * 100

    ci_lower = float(np.percentile(bootstrap_tdis, lower_pct))
    ci_upper = float(np.percentile(bootstrap_tdis, upper_pct))
    tdi_mean = float(np.mean(bootstrap_tdis))

    return {
        "tdi_mean": tdi_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_valid_bootstraps": len(bootstrap_tdis),
    }


def analyze_prediction_archive(
    pred_path: str,
    output_dir: str = "reports/stage2",
    n_bootstraps: int = 2000,
) -> pd.DataFrame:
    """Computes TDI with patient-cluster bootstrap for all targets in a prediction archive."""
    print(f"\n📊 Analyzing TDI on prediction archive: {pred_path}")
    df = pd.read_csv(pred_path)

    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # Check columns
    required = {"patient_id", "temporal_bin", "target", "y_true", "prob_raw"}
    if not required.issubset(df.columns):
        raise ValueError(f"Archive missing required columns. Required: {required}, found: {df.columns.tolist()}")

    # Map temporal bins to numeric indices if needed
    bins = sorted(df["temporal_bin"].unique())
    bin_map = {b: idx for idx, b in enumerate(bins)}
    df["temporal_bin_idx"] = df["temporal_bin"].map(bin_map)

    run_id = df["run_id"].iloc[0] if "run_id" in df.columns else os.path.basename(pred_path).replace("preds_", "").replace(".csv", "")
    arch = df["architecture"].iloc[0] if "architecture" in df.columns else "unknown"
    loss_type = df["loss_type"].iloc[0] if "loss_type" in df.columns else "unknown"
    seed = df["seed"].iloc[0] if "seed" in df.columns else "unknown"

    metrics_to_evaluate = ["auroc", "auprc", "brier_score", "log_loss", "ece_equal_mass", "sensitivity_th0.50", "specificity_th0.50"]
    targets = sorted(df["target"].unique())

    records = []
    for tgt in targets:
        sub_df = df[df["target"] == tgt].copy()
        # Check if target has enough cases across bins
        valid_bins = sub_df.groupby("temporal_bin")["y_true"].nunique()
        has_positives = (valid_bins > 1).sum()

        for metric in metrics_to_evaluate:
            print(f"  Target: {tgt:20s} | Metric: {metric:20s} | Bootstrapping (B={n_bootstraps})...", flush=True)
            # Point estimate (full cohort)
            bin_metrics = []
            needs_both = requires_both_classes(metric)
            for b in bins:
                bin_sub = sub_df[sub_df["temporal_bin"] == b]
                if len(bin_sub) == 0:
                    bin_metrics.append(float("nan"))
                elif needs_both and bin_sub["y_true"].nunique() < 2:
                    bin_metrics.append(float("nan"))
                else:
                    m_val = compute_single_metric_fast(bin_sub["y_true"].values, bin_sub["prob_raw"].values, metric)
                    bin_metrics.append(m_val)

            point_fit = fit_temporal_decay_index(list(range(len(bins))), bin_metrics, metric)

            # Patient-cluster bootstrap
            boot_res = bootstrap_tdi_confidence_interval(
                sub_df,
                patient_col="patient_id",
                time_col="temporal_bin_idx",
                target_col="y_true",
                pred_col="prob_raw",
                metric_name=metric,
                n_bootstraps=n_bootstraps,
                ci_level=0.95,
            )

            records.append({
                "run_id": run_id,
                "architecture": arch,
                "loss_type": loss_type,
                "seed": seed,
                "target": tgt,
                "metric": metric,
                "point_tdi": point_fit["tdi"],
                "point_slope": point_fit["beta"],
                "point_r2": point_fit["r_squared"],
                "point_p_val": point_fit["p_value"],
                "bootstrap_tdi_mean": boot_res["tdi_mean"],
                "bootstrap_ci_lower": boot_res["ci_lower"],
                "bootstrap_ci_upper": boot_res["ci_upper"],
                "n_valid_bootstraps": boot_res["n_valid_bootstraps"],
            })

    res_df = pd.DataFrame(records)
    out_file = os.path.join(table_dir, f"tdi_{run_id}.csv")
    res_df.to_csv(out_file, index=False)
    print(f"✅ Saved TDI inference table to: {out_file}")
    return res_df


if __name__ == "__main__":
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(description="Patient-Cluster Bootstrap TDI Analysis")
    parser.add_argument("--preds", type=str, default="", help="Path to single prediction archive CSV")
    parser.add_argument("--preds-dir", type=str, default="reports/stage2/predictions", help="Directory of prediction archives")
    parser.add_argument("--output-dir", type=str, default="reports/stage2", help="Output directory")
    parser.add_argument("--bootstraps", type=int, default=2000, help="Number of bootstrap replicates")
    args = parser.parse_args()

    if args.preds and os.path.isfile(args.preds):
        analyze_prediction_archive(args.preds, args.output_dir, args.bootstraps)
    elif os.path.isdir(args.preds_dir):
        files = sorted(glob.glob(os.path.join(args.preds_dir, "preds_*.csv")))
        if not files:
            print(f"No prediction files found in {args.preds_dir}")
        for f in files:
            analyze_prediction_archive(f, args.output_dir, args.bootstraps)
    else:
        print("Please provide --preds <file> or ensure --preds-dir contains prediction CSVs.")

