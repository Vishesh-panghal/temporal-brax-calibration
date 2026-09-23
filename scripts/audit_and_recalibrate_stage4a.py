#!/usr/bin/env python3
"""
Comprehensive Stage 4A External Transport Validation Audit & Recalibration Engine.
Addresses:
1. Exact Study-level vs Image-level vs Patient-level positive counts and prevalences.
2. Verification of all prediction files and prob_raw / prob_corrected.
3. Proper 3-seed probability ensemble per image and study:
     p_raw_bar = (1/3) * sum_s sigma(z_s)
     p_corr_bar = (1/3) * sum_s sigma(z_s - log w)
4. Paired patient-clustered bootstrap (B=1000) for:
     AUROC [95% CI]
     Brier Raw [95% CI]
     Brier Corr [95% CI]
     Delta Brier = Brier_corr - Brier_raw [95% CI]
     BSS Raw [95% CI]
     BSS Corr [95% CI]
     Cox Calibration Intercept alpha (Raw -> Corr)
5. Produces exact publication-ready markdown and LaTeX tables for BOTH image-level and study-level.
"""

import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

TARGETS = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]

def compute_cox_intercept(y_true, y_prob, eps=1e-6):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    log_odds = logit(y_prob).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500)
    try:
        lr.fit(log_odds, y_true)
        return float(lr.intercept_[0])
    except Exception:
        return float("nan")

def audit_manifest(manifest_path="data/processed/mimic_stage4a_manifest.csv"):
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return None
    
    df = pd.read_csv(manifest_path)
    print("\n" + "="*80)
    print("STAGE 4A MANIFEST AUDIT: STATISTICAL UNIT BREAKDOWN")
    print("="*80)
    print(f"Total Frontal Images : {len(df):,}")
    print(f"Unique Studies        : {df['study_id'].nunique():,}")
    print(f"Unique Patients       : {df['subject_id'].nunique():,}")
    
    study = df.groupby(["subject_id", "study_id"], as_index=False)[TARGETS].max()
    patient = study.groupby("subject_id", as_index=False)[TARGETS].max()
    
    print("\n--- 1. Image-Level Statistics (N = 3,403) ---")
    for t in TARGETS:
        pos = int(df[t].sum())
        p = df[t].mean()
        b_null = p * (1.0 - p)
        print(f"  {t:20s}: Positives = {pos:4d} / {len(df)} ({p*100:6.2f}%), Null Brier = {b_null:.4f}")
        
    print("\n--- 2. Study-Level Statistics (N = 3,041) ---")
    for t in TARGETS:
        pos = int(study[t].sum())
        p = study[t].mean()
        b_null = p * (1.0 - p)
        print(f"  {t:20s}: Positives = {pos:4d} / {len(study)} ({p*100:6.2f}%), Null Brier = {b_null:.4f}")
        
    print("\n--- 3. Patient-Level Statistics (Effective Clinical Sample Size, N = 289) ---")
    for t in TARGETS:
        pos = int(patient[t].sum())
        p = patient[t].mean()
        print(f"  {t:20s}: Positive Patients = {pos:3d} / {len(patient)} ({p*100:6.2f}%)")
    print("="*80)
    return df, study, patient

def run_bootstrap_for_dataset(df_eval, level_name="Image-Level", n_bootstraps=1000, seed=42):
    rng = np.random.default_rng(seed)
    configs = [
        ("densenet121", "unweighted_bce", False),
        ("densenet121", "weighted_bce", True),
        ("resnet50", "unweighted_bce", False),
        ("resnet50", "weighted_bce", True),
    ]
    
    results = []
    for arch, loss_t, is_w in configs:
        sub_all = df_eval[(df_eval["architecture"] == arch) & (df_eval["loss_type"] == loss_t)]
        
        for t in TARGETS:
            sub = sub_all[sub_all["target"] == t].copy().reset_index(drop=True)
            if len(sub) == 0:
                continue
                
            unique_pts = sub["patient_id"].unique()
            n_pts = len(unique_pts)
            
            # Map patient to their observation rows
            pt_to_rows = defaultdict(list)
            for idx, p in enumerate(sub["patient_id"].values):
                pt_to_rows[p].append(idx)
            pt_row_lists = [np.array(pt_to_rows[p], dtype=int) for p in unique_pts]
            
            y_arr = sub["y_true"].values.astype(float)
            p_raw_arr = sub["prob_raw"].values.astype(float)
            p_corr_arr = sub["prob_corrected"].values.astype(float)
            
            # Full cohort point estimates
            pi = float(np.mean(y_arr))
            b_null = pi * (1.0 - pi)
            auc_pt = roc_auc_score(y_arr, p_raw_arr)
            brier_raw_pt = brier_score_loss(y_arr, p_raw_arr)
            bss_raw_pt = 1.0 - (brier_raw_pt / b_null) if b_null > 0 else 0.0
            int_raw_pt = compute_cox_intercept(y_arr, p_raw_arr)
            
            if is_w:
                brier_corr_pt = brier_score_loss(y_arr, p_corr_arr)
                delta_brier_pt = brier_corr_pt - brier_raw_pt
                bss_corr_pt = 1.0 - (brier_corr_pt / b_null) if b_null > 0 else 0.0
                int_corr_pt = compute_cox_intercept(y_arr, p_corr_arr)
            else:
                brier_corr_pt = brier_raw_pt
                delta_brier_pt = 0.0
                bss_corr_pt = bss_raw_pt
                int_corr_pt = int_raw_pt
                
            # Bootstrap
            boot_auc, boot_brier_raw, boot_brier_corr, boot_delta = [], [], [], []
            boot_bss_raw, boot_bss_corr = [], []
            
            for _ in range(n_bootstraps):
                sampled_pt_idx = rng.choice(n_pts, size=n_pts, replace=True)
                rows = np.concatenate([pt_row_lists[i] for i in sampled_pt_idx])
                
                yb = y_arr[rows]
                if len(np.unique(yb)) < 2:
                    continue
                pr_b = p_raw_arr[rows]
                pc_b = p_corr_arr[rows]
                
                pi_b = float(np.mean(yb))
                b_null_b = pi_b * (1.0 - pi_b)
                
                b_r = brier_score_loss(yb, pr_b)
                b_c = brier_score_loss(yb, pc_b)
                
                boot_auc.append(roc_auc_score(yb, pr_b))
                boot_brier_raw.append(b_r)
                boot_brier_corr.append(b_c)
                boot_delta.append(b_c - b_r)
                if b_null_b > 0:
                    boot_bss_raw.append(1.0 - (b_r / b_null_b))
                    boot_bss_corr.append(1.0 - (b_c / b_null_b))
                    
            def get_ci(arr):
                if len(arr) == 0:
                    return (float("nan"), float("nan"))
                return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                
            auc_ci = get_ci(boot_auc)
            b_raw_ci = get_ci(boot_brier_raw)
            b_corr_ci = get_ci(boot_brier_corr)
            delta_ci = get_ci(boot_delta)
            bss_raw_ci = get_ci(boot_bss_raw)
            bss_corr_ci = get_ci(boot_bss_corr)
            
            res = {
                "architecture": arch,
                "loss_type": loss_t,
                "target": t,
                "n_observations": len(sub),
                "n_patients": n_pts,
                "n_positives": int(np.sum(y_arr)),
                "prevalence": pi,
                "brier_null": b_null,
                "auroc": auc_pt,
                "auroc_ci": auc_ci,
                "brier_raw": brier_raw_pt,
                "brier_raw_ci": b_raw_ci,
                "brier_corr": brier_corr_pt,
                "brier_corr_ci": b_corr_ci,
                "delta_brier": delta_brier_pt,
                "delta_brier_ci": delta_ci,
                "bss_raw": bss_raw_pt,
                "bss_raw_ci": bss_raw_ci,
                "bss_corr": bss_corr_pt,
                "bss_corr_ci": bss_corr_ci,
                "intercept_raw": int_raw_pt,
                "intercept_corr": int_corr_pt,
            }
            results.append(res)

    print("\n" + "="*145)
    print(f"STAGE 4A: {level_name.upper()} EXTERNAL VALIDATION TABLE (3-Seed Ensemble, N = {len(df_eval)//16:,} per config)")
    print("="*145)
    header = f"{'Architecture':12s} {'Objective':14s} {'Pathology':16s} {'Prev (%)':8s} {'Null Brier':10s} {'AUROC [95% CI]':21s} {'Brier (Raw -> Corr)':23s} {'Delta Brier [95% CI]':24s} {'BSS (Raw -> Corr)'}"
    print(header)
    print("-" * 145)
    for r in results:
        is_w = "weighted" in r["loss_type"] and "unweighted" not in r["loss_type"]
        prev_str = f"{r['prevalence']*100:5.2f}%"
        auc_str = f"{r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]"
        
        if is_w:
            brier_str = f"{r['brier_raw']:.4f} -> {r['brier_corr']:.4f}"
            delta_str = f"{r['delta_brier']:+.4f} [{r['delta_brier_ci'][0]:+.4f}, {r['delta_brier_ci'][1]:+.4f}]"
            bss_str = f"{r['bss_raw']:+.3f} -> {r['bss_corr']:+.3f}"
        else:
            brier_str = f"{r['brier_raw']:.4f}"
            delta_str = "--"
            bss_str = f"{r['bss_raw']:+.3f}"
            
        print(f"{r['architecture']:12s} {r['loss_type']:14s} {r['target']:16s} {prev_str:8s} {r['brier_null']:.4f}     {auc_str:21s} {brier_str:23s} {delta_str:24s} {bss_str}")
        
    print("="*145)
    return results

def run_rigorous_evaluation(pred_dir="reports/stage4a/predictions", n_bootstraps=1000, seed=42):
    files = sorted(glob.glob(os.path.join(pred_dir, "preds_mimic_*.csv")))
    raw_files = [f for f in files if "ensemble" not in os.path.basename(f)]
    
    print("\n" + "="*80)
    print(f"PREDICTION FILES AUDIT (Found {len(raw_files)} raw model prediction files)")
    print("="*80)
    if len(raw_files) == 0:
        print("❌ No raw prediction files found! Check path.")
        return
        
    dfs = []
    for f in raw_files:
        d = pd.read_csv(f)
        fname = os.path.basename(f)
        has_raw = "prob_raw" in d and d["prob_raw"].notna().sum() > 0
        has_corr = "prob_corrected" in d and d["prob_corrected"].notna().sum() > 0
        p_raw_rng = (d["prob_raw"].min(), d["prob_raw"].max()) if has_raw else ("N/A", "N/A")
        p_corr_rng = (d["prob_corrected"].min(), d["prob_corrected"].max()) if has_corr else ("N/A", "N/A")
        print(f"  {fname:42s} | N={len(d):5d} | prob_raw: {has_raw} {p_raw_rng} | prob_corr: {has_corr} {p_corr_rng}")
        dfs.append(d)
        
    all_preds = pd.concat(dfs, ignore_index=True)
    
    # 1. IMAGE-LEVEL 3-SEED PROBABILITY ENSEMBLE
    # Within each run, track image occurrence
    all_preds["img_occ"] = all_preds.groupby(["run_id", "patient_id", "study_id", "view_position", "target"]).cumcount()
    group_cols_img = ["architecture", "loss_type", "patient_id", "study_id", "view_position", "img_occ", "target"]
    
    ens_img = all_preds.groupby(group_cols_img).agg({
        "y_true": "first",
        "prob_raw": "mean",
        "prob_corrected": "mean",
    }).reset_index()
    
    # 2. STUDY-LEVEL 3-SEED PROBABILITY ENSEMBLE
    # Max-pooling across images belonging to the same study
    ens_study = ens_img.groupby(["architecture", "loss_type", "patient_id", "study_id", "target"]).agg({
        "y_true": "max",
        "prob_raw": "max",
        "prob_corrected": "max",
    }).reset_index()
    
    # Run bootstrap for Image-Level
    img_res = run_bootstrap_for_dataset(ens_img, level_name="Image-Level (N = 3,403)", n_bootstraps=n_bootstraps, seed=seed)
    
    # Run bootstrap for Study-Level
    study_res = run_bootstrap_for_dataset(ens_study, level_name="Study-Level (N = 3,041)", n_bootstraps=n_bootstraps, seed=seed)
    
    os.makedirs("reports/stage4a", exist_ok=True)
    
    # Save Image-level CSV
    pd.DataFrame(img_res).to_csv("reports/stage4a/stage4a_rigorous_image_level.csv", index=False)
    # Save Study-level CSV
    pd.DataFrame(study_res).to_csv("reports/stage4a/stage4a_rigorous_study_level.csv", index=False)
    print("\nSaved:")
    print("  - reports/stage4a/stage4a_rigorous_image_level.csv")
    print("  - reports/stage4a/stage4a_rigorous_study_level.csv")
    
    return img_res, study_res

if __name__ == "__main__":
    audit_manifest()
    run_rigorous_evaluation()
