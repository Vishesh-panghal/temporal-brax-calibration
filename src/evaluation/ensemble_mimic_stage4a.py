#!/usr/bin/env python3
"""
3-Seed Probability Ensemble & Rigorous Patient-Clustered Bootstrapping for Stage 4A MIMIC-CXR.

Implements:
1. True 3-seed probability ensembling: p_ensemble = (p_42 + p_123 + p_2026) / 3
2. Prevalence-only null model baseline: Brier_null = pi * (1 - pi)
3. Brier Skill Score: BSS = 1 - (Brier / Brier_null)
4. Paired analytic correction delta: Delta_Brier = Brier_corr - Brier_raw with 95% CIs
5. Calibration intercept shift: a_raw -> a_corr
6. Secondary reporting of individual seed metrics (Mean ± SD across seeds)
7. Exports publication-grade Markdown and LaTeX booktabs tables.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGETS = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]


def fit_calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6):
    """Computes calibration slope and intercept via logistic regression."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    log_odds = logit(y_prob).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500)
    try:
        lr.fit(log_odds, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")
    return slope, intercept


def bootstrap_ensemble_cohort(
    df: pd.DataFrame,
    is_weighted: bool,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Runs patient-clustered bootstrap on probability ensemble."""
    rng = np.random.default_rng(seed)
    unique_patients = df["patient_id"].unique()
    n_patients = len(unique_patients)

    # Pre-index rows by patient
    patient_to_rows = defaultdict(list)
    for idx, p in enumerate(df["patient_id"].values):
        patient_to_rows[p].append(idx)
    patient_row_lists = [np.array(patient_to_rows[p], dtype=int) for p in unique_patients]

    y_arr = df["y_true"].values.astype(float)
    p_raw_arr = df["prob_raw_ensemble"].values.astype(float)
    p_corr_arr = df["prob_corr_ensemble"].values.astype(float) if is_weighted else None

    # Null baseline
    pi = float(np.mean(y_arr))
    brier_null = pi * (1.0 - pi)

    # Storage
    boot_aurocs_raw = []
    boot_auprcs_raw = []
    boot_brier_raw = []
    boot_bss_raw = []
    boot_slope_raw = []
    boot_int_raw = []

    boot_aurocs_corr = []
    boot_brier_corr = []
    boot_bss_corr = []
    boot_slope_corr = []
    boot_int_corr = []
    boot_delta_brier = []

    for _ in range(n_bootstraps):
        sampled_pt_idx = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_rows = np.concatenate([patient_row_lists[i] for i in sampled_pt_idx])

        y_b = y_arr[sampled_rows]
        if len(np.unique(y_b)) < 2:
            continue

        p_raw_b = p_raw_arr[sampled_rows]
        auc_r = roc_auc_score(y_b, p_raw_b)
        auprc_r = average_precision_score(y_b, p_raw_b)
        br_r = brier_score_loss(y_b, p_raw_b)
        bss_r = 1.0 - (br_r / brier_null) if brier_null > 0 else float("nan")
        sl_r, int_r = fit_calibration_intercept_slope(y_b, p_raw_b)

        boot_aurocs_raw.append(auc_r)
        boot_auprcs_raw.append(auprc_r)
        boot_brier_raw.append(br_r)
        boot_bss_raw.append(bss_r)
        boot_slope_raw.append(sl_r)
        boot_int_raw.append(int_r)

        if is_weighted and p_corr_arr is not None:
            p_corr_b = p_corr_arr[sampled_rows]
            auc_c = roc_auc_score(y_b, p_corr_b)
            br_c = brier_score_loss(y_b, p_corr_b)
            bss_c = 1.0 - (br_c / brier_null) if brier_null > 0 else float("nan")
            sl_c, int_c = fit_calibration_intercept_slope(y_b, p_corr_b)

            boot_aurocs_corr.append(auc_c)
            boot_brier_corr.append(br_c)
            boot_bss_corr.append(bss_c)
            boot_slope_corr.append(sl_c)
            boot_int_corr.append(int_c)
            boot_delta_brier.append(br_c - br_r)

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = alpha * 100.0, (1.0 - alpha) * 100.0

    res = {
        "prevalence": pi,
        "brier_null": brier_null,
        "auroc_raw_mean": float(np.mean(boot_aurocs_raw)),
        "auroc_raw_ci_lower": float(np.percentile(boot_aurocs_raw, lo)),
        "auroc_raw_ci_upper": float(np.percentile(boot_aurocs_raw, hi)),
        "auprc_raw_mean": float(np.mean(boot_auprcs_raw)),
        "auprc_raw_ci_lower": float(np.percentile(boot_auprcs_raw, lo)),
        "auprc_raw_ci_upper": float(np.percentile(boot_auprcs_raw, hi)),
        "brier_raw_mean": float(np.mean(boot_brier_raw)),
        "brier_raw_ci_lower": float(np.percentile(boot_brier_raw, lo)),
        "brier_raw_ci_upper": float(np.percentile(boot_brier_raw, hi)),
        "bss_raw_mean": float(np.mean(boot_bss_raw)),
        "bss_raw_ci_lower": float(np.percentile(boot_bss_raw, lo)),
        "bss_raw_ci_upper": float(np.percentile(boot_bss_raw, hi)),
        "intercept_raw_mean": float(np.mean(boot_int_raw)),
        "intercept_raw_ci_lower": float(np.percentile(boot_int_raw, lo)),
        "intercept_raw_ci_upper": float(np.percentile(boot_int_raw, hi)),
        "slope_raw_mean": float(np.mean(boot_slope_raw)),
        "slope_raw_ci_lower": float(np.percentile(boot_slope_raw, lo)),
        "slope_raw_ci_upper": float(np.percentile(boot_slope_raw, hi)),
    }

    if is_weighted and boot_brier_corr:
        res.update({
            "auroc_corr_mean": float(np.mean(boot_aurocs_corr)),
            "brier_corr_mean": float(np.mean(boot_brier_corr)),
            "brier_corr_ci_lower": float(np.percentile(boot_brier_corr, lo)),
            "brier_corr_ci_upper": float(np.percentile(boot_brier_corr, hi)),
            "bss_corr_mean": float(np.mean(boot_bss_corr)),
            "bss_corr_ci_lower": float(np.percentile(boot_bss_corr, lo)),
            "bss_corr_ci_upper": float(np.percentile(boot_bss_corr, hi)),
            "intercept_corr_mean": float(np.mean(boot_int_corr)),
            "intercept_corr_ci_lower": float(np.percentile(boot_int_corr, lo)),
            "intercept_corr_ci_upper": float(np.percentile(boot_int_corr, hi)),
            "slope_corr_mean": float(np.mean(boot_slope_corr)),
            "slope_corr_ci_lower": float(np.percentile(boot_slope_corr, lo)),
            "slope_corr_ci_upper": float(np.percentile(boot_slope_corr, hi)),
            "delta_brier_mean": float(np.mean(boot_delta_brier)),
            "delta_brier_ci_lower": float(np.percentile(boot_delta_brier, lo)),
            "delta_brier_ci_upper": float(np.percentile(boot_delta_brier, hi)),
        })

    return res


def run_ensemble_evaluation(pred_dir_str: str, out_dir_str: str, n_bootstraps: int = 1000):
    pred_dir = Path(pred_dir_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = list(pred_dir.glob("preds_mimic_*.csv"))
    if not pred_files:
        print(f"❌ No prediction files found in {pred_dir}")
        sys.exit(1)

    print("=" * 70)
    print("STAGE 4A: 3-SEED PROBABILITY ENSEMBLE & BOOTSTRAP PIPELINE")
    print("=" * 70)
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")

    configs = [
        ("densenet121", "unweighted_bce"),
        ("densenet121", "weighted_bce"),
        ("resnet50", "unweighted_bce"),
        ("resnet50", "weighted_bce"),
    ]

    all_ensemble_records = []
    all_seed_records = []

    for arch, loss_type in configs:
        is_weighted = (loss_type == "weighted_bce")
        # Match the 3 seed files
        matching_files = [
            f for f in pred_files
            if arch in f.name and loss_type in f.name
        ]
        seeds = sorted([int(f.stem.split("seed")[-1]) for f in matching_files if "seed" in f.stem])
        print(f"\nProcessing {arch.upper()} | {loss_type} (Found seeds: {seeds})...")

        if len(matching_files) < 3:
            print(f"  ⚠️ Warning: Found {len(matching_files)} seed files, expected 3.")

        # Load all seed files
        seed_dfs = []
        for mf in matching_files:
            s_df = pd.read_csv(mf)
            seed_dfs.append(s_df)

        combined_df = pd.concat(seed_dfs, ignore_index=True)

        # 1. Compute Individual Seed Metrics (for Secondary Variance Reporting)
        for s in seeds:
            s_sub = combined_df[combined_df["seed"] == s]
            s_all_front = s_sub[s_sub["view_position"].isin(["AP", "PA"])]
            for tgt in TARGETS:
                tgt_s = s_all_front[s_all_front["target"] == tgt]
                if len(tgt_s) == 0:
                    continue
                y_s = tgt_s["y_true"].values
                p_s = tgt_s["prob_raw"].values
                auc_s = roc_auc_score(y_s, p_s)
                br_s = brier_score_loss(y_s, p_s)
                all_seed_records.append({
                    "architecture": arch,
                    "loss_type": loss_type,
                    "seed": s,
                    "target": tgt,
                    "auroc": auc_s,
                    "brier_score": br_s,
                })

        # 2. Build 3-Seed Probability Ensemble
        # Group by unique radiograph key
        key_cols = ["patient_id", "study_id", "view_position", "target"]
        ensemble_agg = combined_df.groupby(key_cols).agg({
            "y_true": "first",
            "prob_raw": "mean",
            "prob_corrected": "mean" if is_weighted else "first",
        }).reset_index().rename(columns={
            "prob_raw": "prob_raw_ensemble",
            "prob_corrected": "prob_corr_ensemble",
        })

        # Stratifications: All Frontal, AP View, PA View
        strata = [
            ("all_frontal", ["AP", "PA"]),
            ("ap_view", ["AP"]),
            ("pa_view", ["PA"]),
        ]

        for stratum_name, allowed_views in strata:
            stratum_df = ensemble_agg[ensemble_agg["view_position"].isin(allowed_views)]

            for tgt in TARGETS:
                sub_tgt = stratum_df[stratum_df["target"] == tgt]
                if len(sub_tgt) == 0:
                    continue

                res = bootstrap_ensemble_cohort(
                    sub_tgt,
                    is_weighted=is_weighted,
                    n_bootstraps=n_bootstraps,
                )
                res.update({
                    "architecture": arch,
                    "loss_type": loss_type,
                    "target": tgt,
                    "stratum": stratum_name,
                    "n_images": len(sub_tgt),
                    "n_patients": sub_tgt["patient_id"].nunique(),
                    "seeds_used": "-".join(map(str, seeds)),
                })
                all_ensemble_records.append(res)

    ensemble_df = pd.DataFrame(all_ensemble_records)
    out_csv = out_dir / "mimic_stage4a_ensemble_metrics.csv"
    ensemble_df.to_csv(out_csv, index=False)
    print(f"\n✅ Successfully saved 3-seed ensemble metrics to: {out_csv}")

    # Compute Individual Seed Mean ± SD
    seed_df = pd.DataFrame(all_seed_records)
    seed_summary = seed_df.groupby(["architecture", "loss_type", "target"])[["auroc", "brier_score"]].agg(["mean", "std"]).reset_index()

    # Generate Markdown Table for All Frontal
    main_ens = ensemble_df[ensemble_df["stratum"] == "all_frontal"].copy()
    print("\n" + "=" * 95)
    print("STAGE 4A THREE-SEED PROBABILITY ENSEMBLE RESULTS (ALL FRONTAL)")
    print("=" * 95)
    fmt_cols = [
        "architecture", "loss_type", "target", "prevalence", "brier_null",
        "auroc_raw_mean", "auroc_raw_ci_lower", "auroc_raw_ci_upper",
        "brier_raw_mean", "brier_raw_ci_lower", "brier_raw_ci_upper",
        "bss_raw_mean", "intercept_raw_mean"
    ]
    display_df = main_ens.copy()
    display_df["AUROC [95% CI]"] = display_df.apply(
        lambda r: f"{r['auroc_raw_mean']:.3f} [{r['auroc_raw_ci_lower']:.3f}, {r['auroc_raw_ci_upper']:.3f}]", axis=1
    )
    display_df["Brier [95% CI]"] = display_df.apply(
        lambda r: f"{r['brier_raw_mean']:.4f} [{r['brier_raw_ci_lower']:.4f}, {r['brier_raw_ci_upper']:.4f}]", axis=1
    )
    display_df["BSS"] = display_df.apply(lambda r: f"{r['bss_raw_mean']:+.3f}", axis=1)
    display_df["Calib Intercept"] = display_df.apply(lambda r: f"{r['intercept_raw_mean']:+.2f}", axis=1)
    display_df["Brier_null"] = display_df["brier_null"].apply(lambda v: f"{v:.4f}")

    disp_cols = ["architecture", "loss_type", "target", "Brier_null", "AUROC [95% CI]", "Brier [95% CI]", "BSS", "Calib Intercept"]
    print(display_df[disp_cols].to_string(index=False))

    # Print Paired Analytic Correction Table for Weighted Models
    weighted_ens = main_ens[main_ens["loss_type"] == "weighted_bce"].copy()
    print("\n" + "=" * 95)
    print("STAGE 4A: PAIRED ANALYTIC CORRECTION RESOLUTION (WEIGHTED BCE)")
    print("=" * 95)
    weighted_ens["ΔBrier [95% CI]"] = weighted_ens.apply(
        lambda r: f"{r['delta_brier_mean']:.4f} [{r['delta_brier_ci_lower']:.4f}, {r['delta_brier_ci_upper']:.4f}]", axis=1
    )
    weighted_ens["Brier (Raw -> Corr)"] = weighted_ens.apply(
        lambda r: f"{r['brier_raw_mean']:.4f} -> {r['brier_corr_mean']:.4f}", axis=1
    )
    weighted_ens["Intercept (Raw -> Corr)"] = weighted_ens.apply(
        lambda r: f"{r['intercept_raw_mean']:+.2f} -> {r['intercept_corr_mean']:+.2f}", axis=1
    )
    w_cols = ["architecture", "target", "Brier (Raw -> Corr)", "ΔBrier [95% CI]", "Intercept (Raw -> Corr)"]
    print(weighted_ens[w_cols].to_string(index=False))
    print("=" * 95)

    # Save LaTeX Table
    tex_path = out_dir / "mimic_stage4a_ensemble_table.tex"
    with open(tex_path, "w") as f:
        f.write("% Table: 3-Seed Probability Ensemble External Validation on MIMIC-CXR-JPG (Frontal Test Cohort)\n")
        f.write(display_df[disp_cols].to_latex(index=False))
    print(f"✅ Generated LaTeX Table: {tex_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default="reports/stage4a/predictions")
    parser.add_argument("--output-dir", type=str, default="reports/stage4a/tables")
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    run_ensemble_evaluation(args.pred_dir, args.output_dir, args.bootstraps)
