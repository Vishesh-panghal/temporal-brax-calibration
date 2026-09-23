#!/usr/bin/env python3
"""Stage 3 Recalibration Engine with Standardized Probability Ensembles.

Standardizes on Probability Averaging:
  p_raw_bar = (1/3) * sum_s sigma(z_s)
  p_analytic_bar = (1/3) * sum_s sigma(z_s - log w)
  p_calib_bar = (1/3) * sum_s p_calib_s
Rebuilds preds_ensemble_3seeds.csv on ALL 8,302 prospective images (dropna=False).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.mitigation import (
    TemperatureScaler,
    OffsetCalibrator,
    PlattCalibrator,
    analytic_weight_correction,
    evaluate_selective_prediction,
    compute_uncertainty,
)
from src.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_calibration_slope_and_intercept,
    compute_discrimination_metrics,
)


def load_prediction_archives(preds_dir: str) -> pd.DataFrame:
    """Loads all prediction archives from the specified directory, strictly excluding ensemble files."""
    files = sorted(glob.glob(os.path.join(preds_dir, "preds_anchor_*.csv")))
    files = [f for f in files if "ensemble" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No anchor prediction files found in {preds_dir}")
    
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def get_exact_training_pos_weight(manifest_path: str, target: str) -> float:
    """Computes exact positive class weight matching training loss: w = (total - pos) / pos."""
    df = pd.read_csv(manifest_path, low_memory=False)
    train = df[df['split'] == 'train'] if 'split' in df.columns else df
    pos = (train[target] == 1.0).sum()
    total = len(train)
    neg = total - pos
    return float(neg / pos) if pos > 0 else 1.0


def cross_fit_t1_calibrators(
    t1_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Fits calibrators out-of-fold on T1 using patient clustering."""
    patients = t1_df['patient_id'].unique()
    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(patients)
    folds = np.array_split(shuffled_patients, n_splits)
    
    patient_to_fold = {}
    for fold_idx, fold_pts in enumerate(folds):
        for pt in fold_pts:
            patient_to_fold[pt] = fold_idx
            
    t1_fold_assignments = t1_df['patient_id'].map(patient_to_fold).values
    z = t1_df['logit'].values.astype(np.float64)
    y = t1_df['y_true'].values.astype(np.float64)
    
    oof_probs = {
        'temperature': np.zeros_like(z),
        'offset': np.zeros_like(z),
        'platt': np.zeros_like(z),
    }
    
    for f in range(n_splits):
        val_mask = (t1_fold_assignments == f)
        train_mask = ~val_mask
        
        z_train, y_train = z[train_mask], y[train_mask]
        z_val = z[val_mask]
        
        # Temperature
        ts = TemperatureScaler()
        ts.fit(z_train, y_train)
        oof_probs['temperature'][val_mask] = ts.calibrate(z_val)
        
        # Offset
        oc = OffsetCalibrator()
        oc.fit(z_train, y_train)
        oof_probs['offset'][val_mask] = oc.calibrate(z_val)
        
        # Platt
        pc = PlattCalibrator()
        pc.fit(z_train, y_train)
        oof_probs['platt'][val_mask] = pc.calibrate(z_val)
        
    return oof_probs


def compute_patient_clustered_bootstrap_ci(
    df: pd.DataFrame,
    p1: np.ndarray,
    p2: np.ndarray,
    n_replications: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes 95% patient-clustered bootstrap CI for difference (method 1 - method 2)."""
    patient_groups = df.groupby('patient_id').indices
    index_lists = list(patient_groups.values())
    n_patients = len(index_lists)
    
    y = df['y_true'].values.astype(np.float64)
    eps = 1e-7
    p1_c = np.clip(p1, eps, 1.0 - eps)
    p2_c = np.clip(p2, eps, 1.0 - eps)
    
    sq_err_diff = ((p1_c - y) ** 2) - ((p2_c - y) ** 2)
    nll1 = -(y * np.log(p1_c) + (1.0 - y) * np.log(1.0 - p1_c))
    nll2 = -(y * np.log(p2_c) + (1.0 - y) * np.log(1.0 - p2_c))
    nll_diff = nll1 - nll2
    
    cluster_counts = np.array([len(idx) for idx in index_lists])
    cluster_sq_diff = np.array([sq_err_diff[idx].sum() for idx in index_lists])
    cluster_nll_diff = np.array([nll_diff[idx].sum() for idx in index_lists])
    
    try:
        seed_int = int(float(seed))
    except (ValueError, TypeError):
        seed_int = 42
    rng = np.random.default_rng(seed_int)
    sample_indices = rng.integers(0, n_patients, size=(n_replications, n_patients))
    
    boot_counts = cluster_counts[sample_indices].sum(axis=1)
    boot_brier_diff = cluster_sq_diff[sample_indices].sum(axis=1) / boot_counts
    boot_nll_diff = cluster_nll_diff[sample_indices].sum(axis=1) / boot_counts
    
    return {
        'brier_diff_ci_lower': float(np.percentile(boot_brier_diff, 2.5)),
        'brier_diff_ci_upper': float(np.percentile(boot_brier_diff, 97.5)),
        'log_loss_diff_ci_lower': float(np.percentile(boot_nll_diff, 2.5)),
        'log_loss_diff_ci_upper': float(np.percentile(boot_nll_diff, 97.5)),
    }


def run_full_recalibration(
    preds_dir: str,
    manifest_path: str,
    output_dir: str,
):
    print("=" * 80)
    print("  STAGE 3: RECALIBRATION ENGINE WITH PROBABILITY ENSEMBLES (8,302 IMAGES)")
    print("=" * 80)
    
    all_preds = load_prediction_archives(preds_dir)
    targets = sorted(all_preds['target'].unique())
    bins = sorted(all_preds['temporal_bin'].unique())
    architectures = sorted(all_preds['architecture'].unique())
    loss_types = sorted(all_preds['loss_type'].unique())
    seeds = sorted(all_preds['seed'].unique())
    
    print(f"Targets: {targets}")
    print(f"Temporal strata: {bins}")
    print(f"Architectures: {architectures}")
    print(f"Loss types: {loss_types}")
    print(f"Seeds: {seeds}")
    print(f"Manifest: {manifest_path}")
    print(f"Total rows loaded: {len(all_preds):,}")
    print(f"Unique images in loaded data: {all_preds['image_id'].nunique():,}")
    
    recalib_records = []
    deferral_records = []
    calibrator_ci_records = []
    ensemble_prediction_records = []
    
    for arch in architectures:
        for loss in loss_types:
            is_weighted = 'weighted' in str(loss).lower() and 'unweighted' not in str(loss).lower()
            print(f"\n=======================================================")
            print(f"  PROCESSING: {arch} | {loss} (is_weighted={is_weighted})")
            print(f"=======================================================")
            
            for target in targets:
                pos_weight = get_exact_training_pos_weight(manifest_path, target) if is_weighted else None
                
                # Dictionary to collect predictions per seed:
                # seed_preds[seed][t_bin] = DataFrame with columns [image_id, y_true, logit, raw, temp, offset, platt, analytic]
                seed_preds = {}
                seed_thresholds = {}
                
                for seed in seeds:
                    sub = all_preds[
                        (all_preds['architecture'] == arch) &
                        (all_preds['loss_type'] == loss) &
                        (all_preds['seed'] == seed) &
                        (all_preds['target'] == target)
                    ].copy().reset_index(drop=True)
                    
                    t1_data = sub[sub['temporal_bin'] == bins[0]].copy().reset_index(drop=True)
                    z_t1 = t1_data['logit'].values.astype(np.float64)
                    y_t1 = t1_data['y_true'].values.astype(np.float64)
                    
                    # 1. Fit full calibrators on T1 for prospective evaluation
                    ts_full = TemperatureScaler()
                    ts_full.fit(z_t1, y_t1)
                    
                    oc_full = OffsetCalibrator()
                    oc_full.fit(z_t1, y_t1)
                    
                    pc_full = PlattCalibrator()
                    pc_full.fit(z_t1, y_t1)
                    
                    # 2. 5-fold cross-fitting on T1
                    oof_t1_probs = cross_fit_t1_calibrators(t1_data, n_splits=5, seed=42)
                    
                    # Store seed predictions across all bins
                    seed_preds[seed] = {}
                    
                    for t_bin in bins:
                        bin_df = sub[sub['temporal_bin'] == t_bin].copy().reset_index(drop=True)
                        z_bin = bin_df['logit'].values.astype(np.float64)
                        y_bin = bin_df['y_true'].values.astype(np.float64)
                        
                        m_dict = {}
                        m_dict['raw'] = 1.0 / (1.0 + np.exp(-z_bin))
                        if t_bin == bins[0]:
                            m_dict['temperature'] = oof_t1_probs['temperature']
                            m_dict['offset'] = oof_t1_probs['offset']
                            m_dict['platt'] = oof_t1_probs['platt']
                            eval_mode = 'out_of_sample_cross_fitted'
                        else:
                            m_dict['temperature'] = ts_full.calibrate(z_bin)
                            m_dict['offset'] = oc_full.calibrate(z_bin)
                            m_dict['platt'] = pc_full.calibrate(z_bin)
                            eval_mode = 'out_of_sample_prospective'
                            
                        if is_weighted and pos_weight is not None:
                            m_dict['analytic_weight'] = analytic_weight_correction(z_bin, pos_weight)
                            
                        # Store in seed_preds
                        bin_pred_df = bin_df[['patient_id', 'study_date', 'view_position', 'image_id', 'y_true', 'logit']].copy()
                        for m_name, m_p in m_dict.items():
                            bin_pred_df[f'prob_{m_name}'] = m_p
                        seed_preds[seed][t_bin] = bin_pred_df
                        
                        # Evaluate proper scoring rules & calibration metrics for individual seed
                        for method_name, probs in m_dict.items():
                            probs_c = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
                            brier = float(np.mean((probs_c - y_bin) ** 2))
                            log_loss_val = float(-np.mean(y_bin * np.log(probs_c) + (1.0 - y_bin) * np.log(1.0 - probs_c)))
                            ece_mass = compute_expected_calibration_error(y_bin, probs_c, strategy="equal_mass")
                            ece_width = compute_expected_calibration_error(y_bin, probs_c, strategy="equal_width")
                            disc = compute_discrimination_metrics(y_bin, probs_c)
                            
                            slope, intercept = np.nan, np.nan
                            if len(np.unique(y_bin)) > 1:
                                logit_p = logit(probs_c).reshape(-1, 1)
                                lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=500)
                                try:
                                    lr.fit(logit_p, y_bin)
                                    intercept = float(lr.intercept_[0])
                                    slope = float(lr.coef_[0][0])
                                except Exception:
                                    pass
                                    
                            recalib_records.append({
                                'architecture': arch,
                                'loss_type': loss,
                                'seed': str(seed),
                                'target': target,
                                'temporal_bin': t_bin,
                                'eval_mode': eval_mode,
                                'calibration_method': method_name,
                                'brier_score': brier,
                                'log_loss': log_loss_val,
                                'ece_equal_mass': ece_mass,
                                'ece_equal_width': ece_width,
                                'auroc': disc['auroc'],
                                'auprc': disc['auprc'],
                                'calibration_slope': slope,
                                'calibration_intercept': intercept,
                                'n_total': len(y_bin),
                                'n_pos': int(np.sum(y_bin == 1)),
                                'prevalence': float(np.mean(y_bin)),
                                'pos_weight': pos_weight if is_weighted else 1.0,
                            })
                
                # -------------------------------------------------------------
                # 3. BUILD 3-SEED PROBABILITY ENSEMBLE
                # -------------------------------------------------------------
                ens_methods_t1 = {}
                ens_thresholds = {}
                
                for t_bin in bins:
                    # Align images across all 3 seeds by image_id
                    s_dfs = [seed_preds[s][t_bin].sort_values('image_id').reset_index(drop=True) for s in seeds]
                    # Verify alignment
                    assert (s_dfs[0]['image_id'] == s_dfs[1]['image_id']).all()
                    assert (s_dfs[0]['image_id'] == s_dfs[2]['image_id']).all()
                    
                    bin_ref = s_dfs[0][['patient_id', 'study_date', 'view_position', 'image_id', 'y_true']].copy()
                    y_ens = bin_ref['y_true'].values.astype(np.float64)
                    z_ens_mean = (s_dfs[0]['logit'].values + s_dfs[1]['logit'].values + s_dfs[2]['logit'].values) / 3.0
                    
                    method_names = ['raw', 'temperature', 'offset', 'platt']
                    if is_weighted:
                        method_names.append('analytic_weight')
                        
                    ens_probs_dict = {}
                    for m_name in method_names:
                        # PROBABILITY ENSEMBLE: Average of probabilities across seeds
                        p_bar = (s_dfs[0][f'prob_{m_name}'].values + 
                                 s_dfs[1][f'prob_{m_name}'].values + 
                                 s_dfs[2][f'prob_{m_name}'].values) / 3.0
                        ens_probs_dict[m_name] = p_bar
                        
                    if t_bin == bins[0]:
                        ens_methods_t1 = ens_probs_dict.copy()
                        # Derive Youden operating thresholds strictly on T1 ensemble
                        for m_name, m_p_t1 in ens_methods_t1.items():
                            if (y_ens == 1).sum() > 0 and (y_ens == 0).sum() > 0:
                                from sklearn.metrics import roc_curve
                                fpr_t1, tpr_t1, ths_t1 = roc_curve(y_ens, m_p_t1)
                                j_scores = tpr_t1 - fpr_t1
                                best_idx = np.argmax(j_scores)
                                t_youden = float(ths_t1[best_idx])
                            else:
                                t_youden = 0.5
                            ens_thresholds[m_name] = {
                                'youden_validation': t_youden,
                                'fixed_0.5': 0.5,
                            }
                            
                    # Record ensemble predictions for saving to preds_ensemble_3seeds.csv
                    bin_ref['architecture'] = arch
                    bin_ref['loss_type'] = loss
                    bin_ref['target'] = target
                    bin_ref['temporal_bin'] = t_bin
                    bin_ref['logit'] = z_ens_mean
                    bin_ref['prob_raw'] = ens_probs_dict['raw']
                    if is_weighted:
                        bin_ref['prob_corrected'] = ens_probs_dict['analytic_weight']
                    ensemble_prediction_records.append(bin_ref)
                    
                    # Evaluate metrics on the Probability Ensemble
                    eval_mode = 'out_of_sample_cross_fitted' if t_bin == bins[0] else 'out_of_sample_prospective'
                    
                    for m_name, probs in ens_probs_dict.items():
                        probs_c = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
                        brier = float(np.mean((probs_c - y_ens) ** 2))
                        log_loss_val = float(-np.mean(y_ens * np.log(probs_c) + (1.0 - y_ens) * np.log(1.0 - probs_c)))
                        ece_mass = compute_expected_calibration_error(y_ens, probs_c, strategy="equal_mass")
                        ece_width = compute_expected_calibration_error(y_ens, probs_c, strategy="equal_width")
                        disc = compute_discrimination_metrics(y_ens, probs_c)
                        
                        slope, intercept = np.nan, np.nan
                        if len(np.unique(y_ens)) > 1:
                            logit_p = logit(probs_c).reshape(-1, 1)
                            lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=500)
                            try:
                                lr.fit(logit_p, y_ens)
                                intercept = float(lr.intercept_[0])
                                slope = float(lr.coef_[0][0])
                            except Exception:
                                pass
                                
                        recalib_records.append({
                            'architecture': arch,
                            'loss_type': loss,
                            'seed': 'ensemble',
                            'target': target,
                            'temporal_bin': t_bin,
                            'eval_mode': eval_mode,
                            'calibration_method': m_name,
                            'brier_score': brier,
                            'log_loss': log_loss_val,
                            'ece_equal_mass': ece_mass,
                            'ece_equal_width': ece_width,
                            'auroc': disc['auroc'],
                            'auprc': disc['auprc'],
                            'calibration_slope': slope,
                            'calibration_intercept': intercept,
                            'n_total': len(y_ens),
                            'n_pos': int(np.sum(y_ens == 1)),
                            'prevalence': float(np.mean(y_ens)),
                            'pos_weight': pos_weight if is_weighted else 1.0,
                        })
                        
                        # 2,000-replicate patient-clustered bootstrap CI for improvement over raw
                        if m_name != 'raw':
                            ci_res = compute_patient_clustered_bootstrap_ci(
                                df=bin_ref,
                                p1=probs,
                                p2=ens_probs_dict['raw'],
                                n_replications=2000,
                                seed=42,
                            )
                            calibrator_ci_records.append({
                                'architecture': arch,
                                'loss_type': loss,
                                'seed': 'ensemble',
                                'target': target,
                                'temporal_bin': t_bin,
                                'calibration_method': m_name,
                                'comparison': f'{m_name}_vs_raw',
                                'brier_raw': float(np.mean((ens_probs_dict['raw'] - y_ens) ** 2)),
                                'brier_calibrated': brier,
                                'delta_brier': brier - float(np.mean((ens_probs_dict['raw'] - y_ens) ** 2)),
                                'brier_diff_ci_lower': ci_res['brier_diff_ci_lower'],
                                'brier_diff_ci_upper': ci_res['brier_diff_ci_upper'],
                                'log_loss_raw': float(-np.mean(y_ens * np.log(np.clip(ens_probs_dict['raw'], 1e-7, 1-1e-7)) + (1-y_ens)*np.log(np.clip(1-ens_probs_dict['raw'], 1e-7, 1-1e-7)))),
                                'log_loss_calibrated': log_loss_val,
                                'delta_log_loss': log_loss_val - float(-np.mean(y_ens * np.log(np.clip(ens_probs_dict['raw'], 1e-7, 1-1e-7)) + (1-y_ens)*np.log(np.clip(1-ens_probs_dict['raw'], 1e-7, 1-1e-7)))),
                                'log_loss_diff_ci_lower': ci_res['log_loss_diff_ci_lower'],
                                'log_loss_diff_ci_upper': ci_res['log_loss_diff_ci_upper'],
                            })
                            
                    # Selective Prediction / Deferral simulation on the ensemble
                    if target in ['Pleural Effusion', 'Pneumonia']:
                        coverage_levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                        for eval_method in method_names:
                            m_probs = ens_probs_dict[eval_method]
                            th_dict = ens_thresholds.get(eval_method, {'youden_validation': 0.5, 'fixed_0.5': 0.5})
                            for thresh_name, thresh_val in th_dict.items():
                                sel_df = evaluate_selective_prediction(
                                    y_ens, m_probs,
                                    coverage_levels=coverage_levels,
                                    operating_threshold=thresh_val,
                                )
                                sel_df['architecture'] = arch
                                sel_df['loss_type'] = loss
                                sel_df['seed'] = 'ensemble'
                                sel_df['target'] = target
                                sel_df['temporal_bin'] = t_bin
                                sel_df['calibration_method'] = eval_method
                                sel_df['threshold_method'] = thresh_name
                                sel_df['threshold_value'] = thresh_val
                                deferral_records.append(sel_df)
                                
                        # Random baseline (100 Monte Carlo draws)
                        raw_th = ens_thresholds.get('raw', {}).get('youden_validation', 0.5)
                        rng = np.random.default_rng(42)
                        n_total = len(y_ens)
                        for cov in coverage_levels:
                            k = int(np.ceil(cov * n_total))
                            k = max(1, min(k, n_total))
                            rand_briers, rand_fn = [], []
                            for _ in range(100):
                                idx_sel = rng.choice(n_total, size=k, replace=False)
                                y_sel = y_ens[idx_sel]
                                p_sel = ens_probs_dict['raw'][idx_sel]
                                rand_briers.append(float(np.mean((p_sel - y_sel) ** 2)))
                                y_pred = (p_sel >= raw_th).astype(int)
                                rand_fn.append(int(np.sum((y_pred == 0) & (y_sel == 1))))
                                
                            deferral_records.append(pd.DataFrame([{
                                'coverage': float(cov),
                                'retained_samples': k,
                                'deferred_samples': n_total - k,
                                'automated_false_negatives': float(np.mean(rand_fn)),
                                'brier_score': float(np.mean(rand_briers)),
                                'architecture': arch,
                                'loss_type': loss,
                                'seed': 'ensemble',
                                'target': target,
                                'temporal_bin': t_bin,
                                'calibration_method': 'random_baseline',
                                'threshold_method': 'youden_validation',
                                'threshold_value': raw_th,
                            }]))

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Full Recalibration Results
    recalib_df = pd.DataFrame(recalib_records)
    recalib_path = os.path.join(output_dir, "stage3_recalibration_full.csv")
    recalib_df.to_csv(recalib_path, index=False)
    print(f"\n✅ Saved full recalibration results to: {recalib_path} ({len(recalib_df)} rows)")
    
    # 2. Save Calibrator Bootstrap CIs
    calib_ci_df = pd.DataFrame(calibrator_ci_records)
    calib_ci_path = os.path.join(output_dir, "stage3_bootstrap_ci.csv")
    calib_ci_df.to_csv(calib_ci_path, index=False)
    # Also save with alternative name for backwards-compatibility
    calib_ci_df.to_csv(os.path.join(output_dir, "stage3_calibrator_improvements_ci.csv"), index=False)
    print(f"✅ Saved calibrator bootstrap CIs to: {calib_ci_path} ({len(calib_ci_df)} rows)")
    
    # 3. Save Deferral Results
    deferral_df = pd.concat(deferral_records, ignore_index=True)
    deferral_path = os.path.join(output_dir, "stage3_deferral_results.csv")
    deferral_df.to_csv(deferral_path, index=False)
    print(f"✅ Saved deferral results to: {deferral_path} ({len(deferral_df)} rows)")
    
    # 4. Save Rebuilt 3-Seed Probability Ensemble CSV
    ens_df = pd.concat(ensemble_prediction_records, ignore_index=True)
    ens_path = os.path.join(preds_dir, "preds_ensemble_3seeds.csv")
    ens_df.to_csv(ens_path, index=False)
    print(f"✅ Saved rebuilt 3-seed ensemble to: {ens_path} ({len(ens_df)} rows, {ens_df['image_id'].nunique()} unique images)")
    
    # Print Summary on Pleural Effusion for Ensemble
    print("\n" + "=" * 95)
    print("  PLEURAL EFFUSION: 3-SEED PROBABILITY ENSEMBLE BRIER SCORES")
    print("=" * 95)
    eff = recalib_df[(recalib_df['target'] == 'Pleural Effusion') & (recalib_df['seed'] == 'ensemble')]
    summary = eff.groupby(['architecture', 'loss_type', 'calibration_method', 'temporal_bin'])['brier_score'].mean().unstack().round(4)
    print(summary.to_string())
    print("\n" + "=" * 95)
    print("  PLEURAL EFFUSION: COX CALIBRATION INTERCEPT (alpha)")
    print("=" * 95)
    summary_alpha = eff.groupby(['architecture', 'loss_type', 'calibration_method', 'temporal_bin'])['calibration_intercept'].mean().unstack().round(4)
    print(summary_alpha.to_string())
    print("\n" + "=" * 95)
    print("  PLEURAL EFFUSION: CALIBRATOR BOOTSTRAP CIS ON T3")
    print("=" * 95)
    t3_ci = calib_ci_df[(calib_ci_df['target'] == 'Pleural Effusion') & (calib_ci_df['temporal_bin'] == 'T3_2017')]
    print(t3_ci[['architecture', 'loss_type', 'calibration_method', 'delta_brier', 'brier_diff_ci_lower', 'brier_diff_ci_upper']].to_string(index=False))
    
    return recalib_df, calib_ci_df, deferral_df, ens_df


if __name__ == "__main__":
    run_full_recalibration(
        preds_dir="reports/stage2/predictions",
        manifest_path="data/processed/stage2_manifest.csv",
        output_dir="reports/stage3",
    )
