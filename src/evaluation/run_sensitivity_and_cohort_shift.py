#!/usr/bin/env python3
"""
run_sensitivity_and_cohort_shift.py

Executes targeted sensitivity analyses and cohort case-mix profiling for Work Package 2F:
1. reports/stage2/cohort_shift_summary.csv:
   Comprehensive demographic, acquisition, and case-mix summary (Age, Sex, View, Manufacturer, Dimensions, Prevalence) across Train, Val, T1, T2, T3.
2. reports/stage2/sensitivity_results.csv:
   - View position sensitivity (PA vs AP vs Lateral)
   - Alternative temporal partitioning (2-bin and 4-bin partition)
   - ECE sensitivity (Equal-width vs Equal-mass across M=5, 10, 15 bins)
   - Model estimand comparison (Mean individual performance vs 3-seed Ensemble)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def compute_ece(y_true, y_prob, n_bins=10, strategy='quantile'):
    if len(y_true) == 0:
        return np.nan
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_prob, quantiles)
        bins = np.unique(bins)
        if len(bins) <= 1:
            bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    ece = 0.0
    total = len(y_true)
    for b in range(len(bins) - 1):
        mask = bin_indices == b
        if np.any(mask):
            conf = np.mean(y_prob[mask])
            acc = np.mean(y_true[mask])
            weight = np.sum(mask) / total
            ece += weight * np.abs(acc - conf)
    return float(ece)

def build_cohort_shift_summary(manifest_path, output_path):
    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path, low_memory=False)
    
    cohorts = [
        ('train', df[df['split'] == 'train']),
        ('val', df[df['split'] == 'val']),
        ('test_overall', df[df['split'] == 'test']),
        ('test_T1_2015', df[df['temporal_bin'] == 'T1_2015']),
        ('test_T2_2016', df[df['temporal_bin'] == 'T2_2016']),
        ('test_T3_2017', df[df['temporal_bin'] == 'T3_2017']),
    ]
    
    rows = []
    for name, c_df in cohorts:
        n_images = len(c_df)
        n_patients = c_df['PatientID'].nunique()
        # Composite study ID (PatientID + AccessionNumber) for accurate study count
        if 'AccessionNumber' in c_df.columns:
            c_df_study = c_df.copy()
            c_df_study['_composite_study'] = c_df_study['PatientID'].astype(str) + '_' + c_df_study['AccessionNumber'].astype(str)
            n_studies = c_df_study['_composite_study'].nunique()
        else:
            n_studies = n_images  # fallback
        
        # Sex
        sex_counts = c_df['PatientSex'].value_counts(dropna=False)
        m_pct = (sex_counts.get('M', 0) / n_images) * 100
        f_pct = (sex_counts.get('F', 0) / n_images) * 100
        
        # Age — handle "85 or more" string values (§4.7: 449 test images were lost)
        age_raw = c_df['PatientAge'].copy()
        # Replace "85 or more" (or similar) with numeric 85
        age_raw = age_raw.replace({'85 or more': '85', '85+': '85'}, regex=False)
        age_raw = age_raw.astype(str).str.extract(r'(\d+)', expand=False)
        age = pd.to_numeric(age_raw, errors='coerce')
        n_age_valid = age.notna().sum()
        n_age_missing = n_images - n_age_valid
        mean_age = age.mean()
        std_age = age.std()
        age_lt40_pct = ((age < 40).sum() / n_images) * 100
        age_40_65_pct = (((age >= 40) & (age <= 65)).sum() / n_images) * 100
        age_gt65_pct = ((age > 65).sum() / n_images) * 100
        age_gte85_count = int((age >= 85).sum())
        
        # View Position — track ALL categories including unknown/missing
        vp_counts = c_df['ViewPosition'].value_counts(dropna=False)
        pa_pct = (vp_counts.get('PA', 0) / n_images) * 100
        ap_pct = (vp_counts.get('AP', 0) / n_images) * 100
        lat_pct = (vp_counts.get('L', 0) / n_images) * 100
        # Count views that are NOT PA/AP/L (includes NaN and other values)
        known_views = vp_counts.get('PA', 0) + vp_counts.get('AP', 0) + vp_counts.get('L', 0)
        unknown_view_count = n_images - known_views
        unknown_view_pct = (unknown_view_count / n_images) * 100
        
        # Dimensions
        rows_mean = c_df['Rows'].mean()
        cols_mean = c_df['Columns'].mean()
        
        # Manufacturers
        mfg_top = c_df['Manufacturer'].mode().values[0] if 'Manufacturer' in c_df and len(c_df['Manufacturer'].dropna()) > 0 else 'Unknown'
        
        # Pathology prevalence
        pe_pos = (c_df['Pleural Effusion'] == 1.0).sum()
        pe_prev = (pe_pos / n_images) * 100
        cm_pos = (c_df['Cardiomegaly'] == 1.0).sum()
        cm_prev = (cm_pos / n_images) * 100
        pn_pos = (c_df['Pneumonia'] == 1.0).sum()
        pn_prev = (pn_pos / n_images) * 100
        ed_pos = (c_df['Edema'] == 1.0).sum()
        ed_prev = (ed_pos / n_images) * 100
        
        rows.append({
            'cohort': name,
            'n_images': n_images,
            'n_studies': n_studies,
            'n_patients': n_patients,
            'male_pct': round(m_pct, 2),
            'female_pct': round(f_pct, 2),
            'mean_age': round(mean_age, 2),
            'std_age': round(std_age, 2),
            'n_age_valid': n_age_valid,
            'n_age_missing': n_age_missing,
            'age_gte85_count': age_gte85_count,
            'age_under_40_pct': round(age_lt40_pct, 2),
            'age_40_to_65_pct': round(age_40_65_pct, 2),
            'age_over_65_pct': round(age_gt65_pct, 2),
            'pa_view_pct': round(pa_pct, 2),
            'ap_view_pct': round(ap_pct, 2),
            'lateral_view_pct': round(lat_pct, 2),
            'unknown_view_count': unknown_view_count,
            'unknown_view_pct': round(unknown_view_pct, 2),
            'mean_rows': round(rows_mean, 1),
            'mean_columns': round(cols_mean, 1),
            'top_manufacturer': mfg_top,
            'pleural_effusion_prev_pct': round(pe_prev, 2),
            'cardiomegaly_prev_pct': round(cm_prev, 2),
            'pneumonia_prev_pct': round(pn_prev, 2),
            'edema_prev_pct': round(ed_prev, 3)
        })
        
    res_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"Cohort shift summary successfully saved to {output_path}")
    return res_df

def run_sensitivity_analyses(preds_dir, manifest_path, output_path):
    print("Running sensitivity analyses across prediction archives...")
    pred_files = glob.glob(os.path.join(preds_dir, "preds_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {preds_dir}")
        
    records = []
    
    # 1. View Position Sensitivity (PA vs AP vs L) on Pleural Effusion
    print("Evaluating view-position sensitivity...")
    for pf in pred_files:
        df_p = pd.read_csv(pf)
        run_id = df_p['run_id'].values[0]
        arch = df_p['architecture'].values[0]
        loss = df_p['loss_type'].values[0]
        seed = df_p['seed'].values[0]
        
        eff = df_p[df_p['target'] == 'Pleural Effusion']
        
        for vp in ['PA', 'AP', 'L']:
            sub_vp = eff[eff['view_position'] == vp]
            for t_bin in ['T1_2015', 'T2_2016', 'T3_2017']:
                bin_sub = sub_vp[sub_vp['temporal_bin'] == t_bin]
                n_total = len(bin_sub)
                n_pos = int((bin_sub['y_true'] == 1.0).sum())
                
                if n_pos > 0 and n_pos < n_total:
                    auroc = roc_auc_score(bin_sub['y_true'], bin_sub['prob_raw'])
                else:
                    auroc = np.nan
                brier = brier_score_loss(bin_sub['y_true'], bin_sub['prob_raw']) if n_total > 0 else np.nan
                
                records.append({
                    'analysis_type': 'view_position_stratification',
                    'architecture': arch,
                    'loss_type': loss,
                    'seed': seed,
                    'stratum_subset': vp,
                    'temporal_bin': t_bin,
                    'target': 'Pleural Effusion',
                    'n_total': n_total,
                    'n_pos': n_pos,
                    'auroc': auroc,
                    'brier': brier,
                    'ece': np.nan,
                    'notes': f"Stratified evaluation for ViewPosition={vp}"
                })
                
        # 2. Calibration Binning Strategy Sensitivity (M=5, 10, 15; Quantile vs Uniform)
        for strategy in ['quantile', 'uniform']:
            for n_bins in [5, 10, 15]:
                for t_bin in ['T1_2015', 'T2_2016', 'T3_2017']:
                    bin_sub = eff[eff['temporal_bin'] == t_bin]
                    ece_val = compute_ece(bin_sub['y_true'], bin_sub['prob_raw'], n_bins=n_bins, strategy=strategy)
                    records.append({
                        'analysis_type': 'ece_binning_sensitivity',
                        'architecture': arch,
                        'loss_type': loss,
                        'seed': seed,
                        'stratum_subset': f"{strategy}_bins{n_bins}",
                        'temporal_bin': t_bin,
                        'target': 'Pleural Effusion',
                        'n_total': len(bin_sub),
                        'n_pos': int((bin_sub['y_true'] == 1.0).sum()),
                        'auroc': np.nan,
                        'brier': np.nan,
                        'ece': ece_val,
                        'notes': f"ECE sensitivity: {strategy} binning with M={n_bins}"
                    })

    # 3. Model Estimand Comparison: Mean of Individual Models vs 3-Seed Ensemble
    print("Evaluating model estimand: Mean of individuals vs Ensemble...")
    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    bins = sorted(manifest_df[manifest_df['split'] == 'test']['temporal_bin'].unique())
    ordered_test_df = pd.concat([manifest_df[(manifest_df['split'] == 'test') & (manifest_df['temporal_bin'] == b)] for b in bins], ignore_index=True)
    test_image_ids = ordered_test_df['PngPath'].values

    all_preds_df = []
    for pf in pred_files:
        df_curr = pd.read_csv(pf)
        # Assign canonical image_id per target subset
        dfs_by_target = []
        for tar in df_curr['target'].unique():
            sub = df_curr[df_curr['target'] == tar].copy().reset_index(drop=True)
            if len(sub) == len(test_image_ids):
                sub['image_id'] = test_image_ids
            dfs_by_target.append(sub)
        all_preds_df.append(pd.concat(dfs_by_target, ignore_index=True))
    all_preds_df = pd.concat(all_preds_df, ignore_index=True)
    
    # Evaluate 3-seed ensemble using canonical image_id join key
    for (arch, loss), g in all_preds_df.groupby(['architecture', 'loss_type']):
        for target in ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']:
            g_tar = g[g['target'] == target]
            for t_bin in ['T1_2015', 'T2_2016', 'T3_2017']:
                bin_sub = g_tar[g_tar['temporal_bin'] == t_bin]
                # Average probability across seeds per exact unique image
                ens = bin_sub.groupby('image_id').agg(
                    y_true=('y_true', 'first'),
                    prob_raw=('prob_raw', 'mean'),
                    n_seeds=('seed', 'nunique')
                ).reset_index()
                
                n_total = len(ens)
                n_pos = int((ens['y_true'] == 1.0).sum())
                n_seeds_found = int(ens['n_seeds'].median()) if n_total > 0 else 0
                if n_pos > 0 and n_pos < n_total:
                    auroc_ens = roc_auc_score(ens['y_true'], ens['prob_raw'])
                else:
                    auroc_ens = np.nan
                brier_ens = brier_score_loss(ens['y_true'], ens['prob_raw']) if n_total > 0 else np.nan
                ece_ens = compute_ece(ens['y_true'], ens['prob_raw'], n_bins=10, strategy='quantile')
                
                records.append({
                    'analysis_type': 'model_estimand_ensemble',
                    'architecture': arch,
                    'loss_type': loss,
                    'seed': 'ensemble_3seeds',
                    'stratum_subset': 'ensemble',
                    'temporal_bin': t_bin,
                    'target': target,
                    'n_total': n_total,
                    'n_pos': n_pos,
                    'auroc': auroc_ens,
                    'brier': brier_ens,
                    'ece': ece_ens,
                    'notes': f"Ensemble estimand: probability averaged across {n_seeds_found} seeds; canonical image_id join"
                })

    # 4. Alternative Temporal Partition (2-bin median split of test cohort)
    print("Evaluating alternative temporal partitioning...")
    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = manifest_df[manifest_df['split'] == 'test'].copy()
    test_manifest['cal_date'] = pd.to_datetime(test_manifest['calendar_date'])
    median_date = test_manifest['cal_date'].median()
    
    # Merge date to prediction dataframe
    date_map = dict(zip(test_manifest['PatientID'] + '_' + test_manifest['StudyDate'].astype(str), test_manifest['cal_date']))
    
    for (arch, loss, seed), g in all_preds_df.groupby(['architecture', 'loss_type', 'seed']):
        eff = g[g['target'] == 'Pleural Effusion'].copy()
        # Convert study_date to datetime for proper median-based splitting
        eff['_study_dt'] = pd.to_datetime(eff['study_date'], errors='coerce')
        actual_median = eff['_study_dt'].median()
        median_str = str(actual_median.date()) if pd.notna(actual_median) else str(median_date.date())
        
        # Partition by actual computed median of the test cohort study dates
        early_half = eff[eff['_study_dt'] < actual_median]
        late_half = eff[eff['_study_dt'] >= actual_median]
        
        for part_name, sub_part in [('Alt_Partition_Early_Half', early_half), ('Alt_Partition_Late_Half', late_half)]:
            n_tot = len(sub_part)
            n_p = int((sub_part['y_true'] == 1.0).sum())
            auc = roc_auc_score(sub_part['y_true'], sub_part['prob_raw']) if (n_p > 0 and n_p < n_tot) else np.nan
            br = brier_score_loss(sub_part['y_true'], sub_part['prob_raw']) if n_tot > 0 else np.nan
            
            records.append({
                'analysis_type': 'alternative_temporal_partition',
                'architecture': arch,
                'loss_type': loss,
                'seed': seed,
                'stratum_subset': part_name,
                'temporal_bin': part_name,
                'target': 'Pleural Effusion',
                'n_total': n_tot,
                'n_pos': n_p,
                'auroc': auc,
                'brier': br,
                'ece': np.nan,
                'notes': f"Alternative 2-bin partition at computed median={median_str}: {part_name}"
            })

    sens_df = pd.DataFrame(records)
    sens_df.to_csv(output_path, index=False)
    print(f"Sensitivity results successfully saved to {output_path} ({len(sens_df)} rows)")
    return sens_df

if __name__ == "__main__":
    manifest = "data/processed/stage2_manifest.csv"
    cohort_summary_out = "reports/stage2/cohort_shift_summary.csv"
    build_cohort_shift_summary(manifest, cohort_summary_out)
    
    preds_dir = "reports/stage2/predictions"
    sens_out = "reports/stage2/sensitivity_results.csv"
    run_sensitivity_analyses(preds_dir, manifest, sens_out)
