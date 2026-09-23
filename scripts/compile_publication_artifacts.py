#!/usr/bin/env python3
"""Stage 3 Manuscript Artifacts Generator.

Compiles publication-grade tables (Markdown + LaTeX) and publication figures (300 DPI):
  - Table 1: Demographic & Cohort Summary (with corrected age & study counts)
  - Table 2: 12-Run Factorial Experimental Matrix (AUROC, Brier, ECE, Log-Loss)
  - Table 3: Multi-Calibrator Performance & Patient-Clustered Bootstrap CIs
  - Table 4: Clinical Selective Prediction & Radiologist Deferral Workload
  - Figure 2: Temporal Trajectories (AUROC & Brier Score)
  - Figure 3: Multi-Calibrator Recalibration Comparison (Resolving the Loss-Weighting Offset)
  - Figure 4: Selective Prediction Risk-Coverage & Missed Positives vs Workload
  - Figure 5: Reliability Diagrams Across Temporal Strata (T1, T2, T3)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def export_latex_table(df: pd.DataFrame, tex_path: str):
    """Exports a pandas DataFrame as a publication-quality booktabs LaTeX tabular."""
    df_tex = df.copy()
    clean_cols = []
    for c in df_tex.columns:
        c_str = str(c).replace('%', '\\%').replace('_', '\\_')
        c_str = c_str.replace('α', '$\\alpha$').replace('β', '$\\beta$').replace('Δ', '$\\Delta$')
        clean_cols.append(c_str)
    df_tex.columns = clean_cols
    for col in df_tex.columns:
        if df_tex[col].dtype == object:
            df_tex[col] = df_tex[col].astype(str).str.replace('%', '\\%', regex=False).str.replace('_', '\\_', regex=False)
            df_tex[col] = df_tex[col].str.replace('α', '$\\alpha$', regex=False).str.replace('β', '$\\beta$', regex=False).str.replace('Δ', '$\\Delta$', regex=False)
    tex_str = df_tex.to_latex(index=False, escape=False)
    with open(tex_path, 'w') as f:
        f.write(tex_str)
    print(f"   LaTeX table written to: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TABLE 1: Cohort Demographics Summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_table1(manifest_path: str, out_dir: str):
    print("Generating Table 1 (Cohort Demographics)...")
    df = pd.read_csv(manifest_path, low_memory=False)
    
    # Preprocess age: handle '85 or more'
    df['clean_age'] = df['PatientAge'].astype(str).str.replace('85 or more', '85', regex=False)
    df['clean_age'] = pd.to_numeric(df['clean_age'], errors='coerce')
    
    # Composite study ID
    df['study_id'] = df['PatientID'].astype(str) + '_' + df['AccessionNumber'].astype(str)
    
    strata = [
        ('Train (Pre-deployment)', df[df['split'] == 'train']),
        ('Val (Pre-deployment)', df[df['split'] == 'val']),
        ('Test T1 (2015)', df[(df['split'] == 'test') & (df['temporal_bin'] == 'T1_2015')]),
        ('Test T2 (2016)', df[(df['split'] == 'test') & (df['temporal_bin'] == 'T2_2016')]),
        ('Test T3 (2017)', df[(df['split'] == 'test') & (df['temporal_bin'] == 'T3_2017')]),
        ('Test All (Pooled)', df[df['split'] == 'test']),
    ]
    
    targets = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']
    records = []
    
    for name, sub in strata:
        n_img = len(sub)
        n_pts = sub['PatientID'].nunique()
        n_std = sub['study_id'].nunique()
        
        # Sex
        sex_vc = sub['PatientSex'].value_counts(dropna=False)
        m_pct = (sex_vc.get('M', 0) / n_img) * 100.0 if n_img > 0 else 0.0
        f_pct = (sex_vc.get('F', 0) / n_img) * 100.0 if n_img > 0 else 0.0
        
        # Age
        age_mean = sub['clean_age'].mean()
        age_std = sub['clean_age'].std()
        
        # Views
        vp_vc = sub['ViewPosition'].value_counts(dropna=False)
        pa_pct = (vp_vc.get('PA', 0) / n_img) * 100.0 if n_img > 0 else 0.0
        ap_pct = (vp_vc.get('AP', 0) / n_img) * 100.0 if n_img > 0 else 0.0
        lat_pct = (vp_vc.get('LATERAL', 0) / n_img) * 100.0 if n_img > 0 else 0.0
        unk_pct = (sub['ViewPosition'].isna().sum() / n_img) * 100.0 if n_img > 0 else 0.0
        
        rec = {
            'Cohort Stratum': name,
            'Images (N)': f"{n_img:,}",
            'Patients (N)': f"{n_pts:,}",
            'Studies (N)': f"{n_std:,}",
            'Sex (% M / F)': f"{m_pct:.1f}% / {f_pct:.1f}%",
            'Age (Mean ± SD)': f"{age_mean:.1f} ± {age_std:.1f}",
            'Views (% PA / AP / Lat / Unk)': f"{pa_pct:.1f}% / {ap_pct:.1f}% / {lat_pct:.1f}% / {unk_pct:.1f}%",
        }
        for t in targets:
            pos = int((sub[t] == 1.0).sum())
            pct = (pos / n_img) * 100.0 if n_img > 0 else 0.0
            rec[f'{t} N (%)'] = f"{pos:,} ({pct:.2f}%)"
            
        records.append(rec)
        
    t1_df = pd.DataFrame(records)
    
    # Save CSV, Markdown, LaTeX
    csv_path = os.path.join(out_dir, "table1_cohort_demographics.csv")
    md_path = os.path.join(out_dir, "table1_cohort_demographics.md")
    tex_path = os.path.join(out_dir, "table1_cohort_demographics.tex")
    t1_df.to_csv(csv_path, index=False)
    export_latex_table(t1_df, tex_path)
    
    md_content = "# Table 1: Patient Cohort Demographics and Clinical Pathology Prevalence Across Deployment Strata\n\n"
    md_content += t1_df.to_markdown(index=False)
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✅ Saved Table 1 to: {csv_path}, {md_path}, and {tex_path}")
    return t1_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TABLE 2: 12-Run Factorial Matrix (AUROC, Brier, ECE, Log-Loss)
# ─────────────────────────────────────────────────────────────────────────────

def generate_table2(recalib_csv_path: str, out_dir: str):
    print("Generating Table 2 (Factorial Experimental Matrix)...")
    df = pd.read_csv(recalib_csv_path)
    # Filter for raw (uncalibrated) individual seed performance (excluding ensemble) to report Mean ± SD across seeds
    raw_df = df[(df['calibration_method'] == 'raw') & (df['seed'] != 'ensemble')].copy()
    
    arch_order = ['densenet121', 'resnet50']
    arch_display = {'densenet121': 'DenseNet-121', 'resnet50': 'ResNet-50'}
    loss_order = ['unweighted_bce', 'weighted_bce']
    loss_display = {'unweighted_bce': 'Unweighted BCE', 'weighted_bce': 'Weighted BCE'}
    target_order = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']
    
    rows = []
    for arch in arch_order:
        for loss in loss_order:
            for target in target_order:
                sub = raw_df[
                    (raw_df['architecture'] == arch) &
                    (raw_df['loss_type'] == loss) &
                    (raw_df['target'] == target)
                ]
                
                t1 = sub[sub['temporal_bin'] == 'T1_2015']
                t2 = sub[sub['temporal_bin'] == 'T2_2016']
                t3 = sub[sub['temporal_bin'] == 'T3_2017']
                
                # AUROC
                auc_t1_m, auc_t1_s = t1['auroc'].mean(), t1['auroc'].std()
                auc_t2_m, auc_t2_s = t2['auroc'].mean(), t2['auroc'].std()
                auc_t3_m, auc_t3_s = t3['auroc'].mean(), t3['auroc'].std()
                auc_delta = auc_t3_m - auc_t1_m if pd.notna(auc_t3_m) and pd.notna(auc_t1_m) else np.nan
                
                # Brier
                br_t1_m, br_t1_s = t1['brier_score'].mean(), t1['brier_score'].std()
                br_t2_m, br_t2_s = t2['brier_score'].mean(), t2['brier_score'].std()
                br_t3_m, br_t3_s = t3['brier_score'].mean(), t3['brier_score'].std()
                br_delta = br_t3_m - br_t1_m
                
                # Equal-Mass ECE
                ece_t1_m, ece_t1_s = t1['ece_equal_mass'].mean(), t1['ece_equal_mass'].std()
                ece_t2_m, ece_t2_s = t2['ece_equal_mass'].mean(), t2['ece_equal_mass'].std()
                ece_t3_m, ece_t3_s = t3['ece_equal_mass'].mean(), t3['ece_equal_mass'].std()
                ece_delta = ece_t3_m - ece_t1_m
                
                # Log-Loss
                nll_t1_m, nll_t1_s = t1['log_loss'].mean(), t1['log_loss'].std()
                nll_t2_m, nll_t2_s = t2['log_loss'].mean(), t2['log_loss'].std()
                nll_t3_m, nll_t3_s = t3['log_loss'].mean(), t3['log_loss'].std()
                nll_delta = nll_t3_m - nll_t1_m
                
                fmt_fn = lambda m, s: f"{m:.4f} ± {s:.4f}" if pd.notna(m) and pd.notna(s) else (f"{m:.4f}" if pd.notna(m) else "--")
                fmt_d = lambda d: f"{d:+.4f}" if pd.notna(d) else "--"
                
                rows.append({
                    'Architecture': arch_display[arch],
                    'Objective': loss_display[loss],
                    'Target': target + ("*" if target == "Edema" else ""),
                    'AUROC T1': fmt_fn(auc_t1_m, auc_t1_s),
                    'AUROC T2': fmt_fn(auc_t2_m, auc_t2_s),
                    'AUROC T3': fmt_fn(auc_t3_m, auc_t3_s),
                    'AUROC Δ': fmt_d(auc_delta),
                    'Brier T1': fmt_fn(br_t1_m, br_t1_s),
                    'Brier T2': fmt_fn(br_t2_m, br_t2_s),
                    'Brier T3': fmt_fn(br_t3_m, br_t3_s),
                    'Brier Δ': fmt_d(br_delta),
                    'ECE T1': fmt_fn(ece_t1_m, ece_t1_s),
                    'ECE T3': fmt_fn(ece_t3_m, ece_t3_s),
                    'ECE Δ': fmt_d(ece_delta),
                    'LogLoss T1': fmt_fn(nll_t1_m, nll_t1_s),
                    'LogLoss T3': fmt_fn(nll_t3_m, nll_t3_s),
                    'LogLoss Δ': fmt_d(nll_delta),
                })
                
    t2_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table2_factorial_matrix.csv")
    md_path = os.path.join(out_dir, "table2_factorial_matrix.md")
    tex_path = os.path.join(out_dir, "table2_factorial_matrix.tex")
    t2_df.to_csv(csv_path, index=False)
    export_latex_table(t2_df, tex_path)
    
    md_content = "# Table 2: Factorial Experimental Matrix Across Deployment Strata (AUROC, Brier, ECE, Log-Loss)\n\n"
    md_content += "*Reported as Mean ± Standard Deviation across three independent training seeds (Seed 42, 123, 2026).*\n"
    md_content += "> **Note on Edema (*):** Positive cases in stratum T3 are zero (N=0), rendering T3 discrimination non-computable. Scoring rules (Brier, ECE, Log-Loss) remain computable.\n\n"
    md_content += t2_df.to_markdown(index=False)
    with open(md_path, 'w') as f:
        f.write(md_content)
        
    print(f"✅ Saved Table 2 to: {csv_path}, {md_path}, and {tex_path}")
    return t2_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. TABLE 3: Multi-Calibrator Mitigation Comparison & Bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────

def generate_table3(recalib_csv_path: str, ci_csv_path: str, out_dir: str):
    print("Generating Table 3 (Multi-Calibrator Comparison & Bootstrap CIs)...")
    df = pd.read_csv(recalib_csv_path)
    ci_df = pd.read_csv(ci_csv_path) if os.path.exists(ci_csv_path) else pd.DataFrame()
    
    eff = df[df['target'] == 'Pleural Effusion'].copy()
    # Use 3-seed ensemble for unified metrics and exact patient-clustered bootstrap CIs
    eff_ens = eff[eff['seed'] == 'ensemble'].copy()
    
    methods = ['raw', 'temperature', 'analytic_weight', 'offset', 'platt']
    method_display = {
        'raw': 'Raw (Uncalibrated)',
        'temperature': 'Temperature Scaling (T*)',
        'analytic_weight': 'Analytic Offset (z - log w)',
        'offset': 'Fitted Offset (z + a*)',
        'platt': 'Platt Scaling (a + bz)',
    }
    
    rows = []
    for arch in ['densenet121', 'resnet50']:
        for loss in ['weighted_bce', 'unweighted_bce']:
            sub = eff_ens[(eff_ens['architecture'] == arch) & (eff_ens['loss_type'] == loss)]
            for m in methods:
                if m == 'analytic_weight' and 'unweighted' in loss:
                    continue
                m_sub = sub[sub['calibration_method'] == m]
                if len(m_sub) == 0:
                    continue
                    
                t1_brier = m_sub[m_sub['temporal_bin'] == 'T1_2015']['brier_score'].values[0]
                t2_brier = m_sub[m_sub['temporal_bin'] == 'T2_2016']['brier_score'].values[0]
                t3_brier = m_sub[m_sub['temporal_bin'] == 'T3_2017']['brier_score'].values[0]
                delta_brier = t3_brier - t1_brier
                
                t3_ece = m_sub[m_sub['temporal_bin'] == 'T3_2017']['ece_equal_mass'].values[0]
                t3_nll = m_sub[m_sub['temporal_bin'] == 'T3_2017']['log_loss'].values[0]
                t3_alpha = m_sub[m_sub['temporal_bin'] == 'T3_2017']['calibration_intercept'].values[0]
                t3_beta = m_sub[m_sub['temporal_bin'] == 'T3_2017']['calibration_slope'].values[0]
                
                # CI for T3 improvement relative to raw on ensemble
                ci_str = "--"
                if not ci_df.empty and m != 'raw':
                    m_col = 'calibration_method' if 'calibration_method' in ci_df.columns else 'method'
                    ci_sub = ci_df[
                        (ci_df['architecture'] == arch) &
                        (ci_df['loss_type'] == loss) &
                        (ci_df['target'] == 'Pleural Effusion') &
                        (ci_df['temporal_bin'] == 'T3_2017') &
                        (ci_df[m_col] == m) &
                        (ci_df['seed'] == 'ensemble')
                    ]
                    if len(ci_sub) > 0:
                        r_ci = ci_sub.iloc[0]
                        d_br = r_ci['delta_brier']
                        l = r_ci['brier_diff_ci_lower']
                        u = r_ci['brier_diff_ci_upper']
                        ci_str = f"{d_br:+.4f} [{l:+.4f}, {u:+.4f}]"
                        
                rows.append({
                    'Architecture': 'DenseNet-121' if arch == 'densenet121' else 'ResNet-50',
                    'Loss Objective': 'Weighted BCE' if 'weighted' in loss and 'unweighted' not in loss else 'Unweighted BCE',
                    'Calibrator': method_display[m],
                    'T1 Brier': f"{t1_brier:.4f}",
                    'T2 Brier': f"{t2_brier:.4f}",
                    'T3 Brier': f"{t3_brier:.4f}",
                    'Δ Brier (T3 - T1)': f"{delta_brier:+.4f}",
                    'T3 Intercept (α)': f"{t3_alpha:+.4f}",
                    'T3 Slope (β)': f"{t3_beta:.4f}",
                    'T3 ECE': f"{t3_ece:.4f}",
                    'T3 Log-Loss': f"{t3_nll:.4f}",
                    'T3 Δ Brier vs Raw [95% Bootstrap CI]': ci_str,
                })
                
    t3_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table3_calibrator_comparison.csv")
    md_path = os.path.join(out_dir, "table3_calibrator_comparison.md")
    tex_path = os.path.join(out_dir, "table3_calibrator_comparison.tex")
    t3_df.to_csv(csv_path, index=False)
    export_latex_table(t3_df, tex_path)
    
    md_content = "# Table 3: Comprehensive Multi-Calibrator Comparison on Pleural Effusion\n\n"
    md_content += "*Evaluation of all 5 calibrators across deployment strata on the 3-seed ensemble with 2,000-replicate patient-clustered bootstrap 95% confidence intervals, including Cox calibration regression intercept (α) and slope (β).*\n\n"
    md_content += t3_df.to_markdown(index=False)
    with open(md_path, 'w') as f:
        f.write(md_content)
        
    print(f"✅ Saved Table 3 to: {csv_path}, {md_path}, and {tex_path}")
    return t3_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLE 4: Clinical Selective Prediction & Workload Table
# ─────────────────────────────────────────────────────────────────────────────

def generate_table4(deferral_csv_path: str, out_dir: str):
    print("Generating Table 4 (Selective Prediction & Workload)...")
    df = pd.read_csv(deferral_csv_path)
    
    # Filter for DenseNet-121, Pleural Effusion, Stratum T3 (Late Deployment), Ensemble, Youden validation threshold
    sub = df[
        (df['architecture'] == 'densenet121') &
        (df['target'] == 'Pleural Effusion') &
        (df['temporal_bin'] == 'T3_2017') &
        (df['seed'] == 'ensemble') &
        (df['threshold_method'] == 'youden_validation')
    ].copy()
    
    # Compare: Weighted Raw, Weighted Analytic Correction, Unweighted Raw, Random Baseline
    cases = [
        ('Weighted BCE (Raw)', sub[(sub['loss_type'] == 'weighted_bce') & (sub['calibration_method'] == 'raw')]),
        ('Weighted BCE (Analytic Corrected)', sub[(sub['loss_type'] == 'weighted_bce') & (sub['calibration_method'] == 'analytic_weight')]),
        ('Unweighted BCE (Raw)', sub[(sub['loss_type'] == 'unweighted_bce') & (sub['calibration_method'] == 'raw')]),
        ('Random Deferral Baseline', sub[(sub['calibration_method'] == 'random_baseline') & (sub['loss_type'] == 'weighted_bce')]),
    ]
    
    rows = []
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for case_name, c_df in cases:
            c_cov = c_df[c_df['coverage'] == cov]
            if len(c_cov) == 0:
                continue
                
            r = c_cov.iloc[0]
            ret_samples = int(r['retained_samples'])
            def_samples = int(r['deferred_samples'])
            fn = float(r['automated_false_negatives'])
            brier = float(r['brier_score'])
            
            if 'Random' in case_name:
                ret_pos = 85.0 * cov
                tp = ret_pos - fn
                sens_ret = tp / ret_pos if ret_pos > 0 else np.nan
                auc = np.nan
            else:
                ret_pos = float(r['retained_positives'])
                tp = ret_pos - fn
                sens_ret = float(r['sensitivity_retained']) if 'sensitivity_retained' in r and pd.notna(r['sensitivity_retained']) else (tp / ret_pos if ret_pos > 0 else np.nan)
                auc = float(r['auroc']) if 'auroc' in r and pd.notna(r['auroc']) else np.nan
            
            # Full-cohort system sensitivity: assumes referred cases are identified by the reader
            # Total positive cases in T3 is 85. Missed by autonomous pipeline = fn. System sensitivity = (85 - fn) / 85.
            sys_sens = (85.0 - fn) / 85.0
            
            # Programmatic assertions for mathematical consistency
            assert ret_samples + def_samples == 2436, f"Sample split mismatch: {ret_samples} + {def_samples} != 2436"
            if 'Random' not in case_name:
                assert abs(tp + fn - ret_pos) < 1e-4, f"Positives accounting mismatch: {tp} + {fn} != {ret_pos}"
                if ret_pos > 0:
                    assert abs(sens_ret - (tp / ret_pos)) < 1e-3, f"Retained sensitivity mismatch: {sens_ret} vs {tp/ret_pos}"
            assert 0.0 <= sys_sens <= 1.0, f"Invalid system sensitivity: {sys_sens}"
            
            rows.append({
                'Coverage': f"{int(cov*100)}%",
                'Strategy': case_name,
                'Automated Decisions (N)': f"{ret_samples:,}",
                'Referred to Reader (N)': f"{def_samples:,} ({int((1-cov)*100)}%)",
                'Retained Positives (N)': f"{ret_pos:.0f}" if 'Random' not in case_name else f"{ret_pos:.1f}",
                'Automated True Positives (N)': f"{tp:.0f}" if 'Random' not in case_name else f"{tp:.1f}",
                'Automated False Negatives (Missed Cases)': f"{fn:.1f}",
                'Retained Sensitivity': f"{sens_ret:.4f}" if pd.notna(sens_ret) else "--",
                'Full-Cohort Triage Sensitivity': f"{sys_sens:.4f}",
                'Retained AUROC': f"{auc:.4f}" if pd.notna(auc) else "--",
                'Retained Brier': f"{brier:.4f}",
            })
            
    t4_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table4_selective_prediction_workload.csv")
    md_path = os.path.join(out_dir, "table4_selective_prediction_workload.md")
    tex_path = os.path.join(out_dir, "table4_selective_prediction_workload.tex")
    t4_df.to_csv(csv_path, index=False)
    export_latex_table(t4_df, tex_path)
    
    md_content = "# Table 4: Clinical Selective Prediction and Radiologist Deferral Workload (Stratum T3: Pleural Effusion)\n\n"
    md_content += "*Comprehensive triage accounting under increasing radiologist referral workload. Evaluated on 3-seed ensemble with matched Youden decision thresholds determined on development cohort T1. Compares raw weighted, analytic-corrected, unweighted, and random deferral. Shows both Retained Sensitivity (TP / Retained Positives) and Full-Cohort Triage Sensitivity (1 - Missed Cases / 85).*\n\n"
    md_content += t4_df.to_markdown(index=False)
    with open(md_path, 'w') as f:
        f.write(md_content)
        
    print(f"✅ Saved Table 4 to: {csv_path}, {md_path}, and {tex_path}")
    return t4_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 2: Temporal Trajectories (AUROC & Brier Score)
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure2(recalib_csv_path: str, out_dir: str):
    print("Generating Figure 2 (Temporal Trajectories)...")
    df = pd.read_csv(recalib_csv_path)
    # Raw individual models (excluding ensemble) so error bands accurately represent inter-seed SD across the 3 seeds, matching Table 2
    raw = df[(df['calibration_method'] == 'raw') & (df['seed'] != 'ensemble')].copy()
    eff = raw[raw['target'] == 'Pleural Effusion'].copy()
    
    eff['Architecture'] = eff['architecture'].map({'densenet121': 'DenseNet-121', 'resnet50': 'ResNet-50'})
    eff['Objective'] = eff['loss_type'].map({'unweighted_bce': 'Unweighted BCE', 'weighted_bce': 'Weighted BCE'})
    eff['Stratum'] = eff['temporal_bin'].map({'T1_2015': 'T1 (2015)', 'T2_2016': 'T2 (2016)', 'T3_2017': 'T3 (2017)'})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), dpi=300)
    
    palette = {'DenseNet-121': '#1f77b4', 'ResNet-50': '#ff7f0e'}
    
    # Left: AUROC
    sns.lineplot(
        data=eff, x='Stratum', y='auroc', hue='Architecture', style='Objective',
        markers=True, dashes=True, err_style='band', errorbar='sd',
        linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
    )
    axes[0].set_title("(a) Discrimination Trajectory (AUROC ↑)", fontweight="bold", pad=12)
    axes[0].set_ylabel("Area Under the ROC Curve")
    axes[0].set_xlabel("Deployment Strata")
    axes[0].set_ylim(0.85, 0.96)
    axes[0].legend(frameon=True, framealpha=0.9, loc="lower left")
    
    # Right: Brier Score
    sns.lineplot(
        data=eff, x='Stratum', y='brier_score', hue='Architecture', style='Objective',
        markers=True, dashes=True, err_style='band', errorbar='sd',
        linewidth=2.5, markersize=8, palette=palette, ax=axes[1]
    )
    axes[1].set_title("(b) Probability Error Trajectory (Brier Score ↓)", fontweight="bold", pad=12)
    axes[1].set_ylabel("Brier Score (Mean Squared Error)")
    axes[1].set_xlabel("Deployment Strata")
    axes[1].legend(frameon=True, framealpha=0.9, loc="upper left")
    
    fig.tight_layout()
    fig2_path = os.path.join(out_dir, "fig2_drift_trajectories.png")
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 2 to: {fig2_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 3: Multi-Calibrator Mitigation Comparison
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure3(recalib_csv_path: str, out_dir: str):
    print("Generating Figure 3 (Multi-Calibrator Comparison)...")
    df = pd.read_csv(recalib_csv_path)
    eff = df[df['target'] == 'Pleural Effusion'].copy()
    
    # Focus on DenseNet-121 3-seed ensemble to match Table 3 exactly
    sub = eff[(eff['architecture'] == 'densenet121') & (eff['seed'] == 'ensemble')].copy()
    
    plot_methods = [
        ('unweighted_bce', 'raw', 'Unweighted Raw', '#2ca02c'),
        ('weighted_bce', 'raw', 'Weighted Raw (Uncalibrated)', '#d62728'),
        ('weighted_bce', 'temperature', 'Weighted + Temp Scaling (T*)', '#ff7f0e'),
        ('weighted_bce', 'analytic_weight', 'Weighted + Analytic Offset (z - log w)', '#1f77b4'),
        ('weighted_bce', 'platt', 'Weighted + Platt Scaling (a + bz)', '#9467bd'),
    ]
    
    plot_rows = []
    for loss, method, label, color in plot_methods:
        s = sub[(sub['loss_type'] == loss) & (sub['calibration_method'] == method)]
        for _, r in s.iterrows():
            plot_rows.append({
                'Stratum': {'T1_2015': 'T1 (2015)', 'T2_2016': 'T2 (2016)', 'T3_2017': 'T3 (2017)'}[r['temporal_bin']],
                'Brier Score': r['brier_score'],
                'ECE': r['ece_equal_mass'],
                'Log-Loss': r['log_loss'],
                'Method': label,
            })
            
    plot_df = pd.DataFrame(plot_rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), dpi=300)
    palette = {label: color for _, _, label, color in plot_methods}
    
    # Left: Brier Score across strata
    sns.barplot(
        data=plot_df, x='Stratum', y='Brier Score', hue='Method',
        palette=palette, ax=axes[0]
    )
    axes[0].set_title("(a) Ensemble Brier Score Across Deployment Strata (Lower is Better)", fontweight="bold", pad=12)
    axes[0].set_ylabel("Brier Score")
    axes[0].set_xlabel("Deployment Strata")
    axes[0].legend(frameon=True, framealpha=0.9, fontsize=9.5)
    
    # Right: Log-Loss across strata
    sns.barplot(
        data=plot_df, x='Stratum', y='Log-Loss', hue='Method',
        palette=palette, ax=axes[1]
    )
    axes[1].set_title("(b) Ensemble Cross-Entropy / Log-Loss (Lower is Better)", fontweight="bold", pad=12)
    axes[1].set_ylabel("Log-Loss (NLL)")
    axes[1].set_xlabel("Deployment Strata")
    axes[1].legend(frameon=True, framealpha=0.9, fontsize=9.5)
    
    fig.tight_layout()
    fig3_path = os.path.join(out_dir, "fig3_recalibration_comparison.png")
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 3 to: {fig3_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 4: Selective Prediction Risk-Coverage Curves
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure4(deferral_csv_path: str, out_dir: str):
    print("Generating Figure 4 (Selective Prediction Curves)...")
    df = pd.read_csv(deferral_csv_path)
    
    sub = df[
        (df['architecture'] == 'densenet121') &
        (df['target'] == 'Pleural Effusion') &
        (df['temporal_bin'] == 'T3_2017') &
        (df['seed'] == 'ensemble') &
        (df['threshold_method'] == 'youden_validation')
    ].copy()
    
    # Filter methods of interest
    cases = [
        ('Unweighted BCE', sub[(sub['loss_type'] == 'unweighted_bce') & (sub['calibration_method'] == 'raw')]),
        ('Weighted BCE (Raw)', sub[(sub['loss_type'] == 'weighted_bce') & (sub['calibration_method'] == 'raw')]),
        ('Weighted BCE (Analytic)', sub[(sub['loss_type'] == 'weighted_bce') & (sub['calibration_method'] == 'analytic_weight')]),
        ('Random Triage Baseline', sub[(sub['calibration_method'] == 'random_baseline') & (sub['loss_type'] == 'weighted_bce')]),
    ]
    
    plot_rows = []
    for label, c_df in cases:
        for _, r in c_df.iterrows():
            plot_rows.append({
                'Coverage': r['coverage'],
                'Deferred_Pct': (1.0 - r['coverage']) * 100.0,
                'Brier Score': r['brier_score'],
                'False Negatives': r['automated_false_negatives'],
                'Strategy': label,
            })
    plot_df = pd.DataFrame(plot_rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), dpi=300)
    palette = {
        'Unweighted BCE': '#2ca02c',
        'Weighted BCE (Raw)': '#d62728',
        'Weighted BCE (Analytic)': '#1f77b4',
        'Random Triage Baseline': '#7f7f7f',
    }
    
    # Left: Brier Score vs Coverage (Coverage decreasing = increasing deferral)
    sns.lineplot(
        data=plot_df, x='Coverage', y='Brier Score', hue='Strategy',
        marker='o', linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
    )
    axes[0].set_title("(a) Retained Error vs Coverage", fontweight="bold", pad=12)
    axes[0].set_xlabel("Retention Coverage (1.0 = Full Cohort, 0.7 = Defer 30%)")
    axes[0].set_ylabel("Retained Brier Score")
    axes[0].set_xlim(1.02, 0.48)
    axes[0].legend(frameon=True, framealpha=0.9)
    
    # Right: Automated False Negatives (Missed Cases) vs Deferral Percentage
    sns.lineplot(
        data=plot_df, x='Deferred_Pct', y='False Negatives', hue='Strategy',
        marker='s', linewidth=2.5, markersize=8, palette=palette, ax=axes[1]
    )
    axes[1].set_title("(b) Missed Pathology Cases vs Radiologist Workload", fontweight="bold", pad=12)
    axes[1].set_xlabel("Percentage of Difficult Cases Referred to Radiologist (%)")
    axes[1].set_ylabel("Automated False Negatives (Missed Cases)")
    axes[1].legend(frameon=True, framealpha=0.9)
    
    fig.tight_layout()
    fig4_path = os.path.join(out_dir, "fig4_selective_risk_coverage.png")
    fig.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 4 to: {fig4_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 5: Reliability Diagrams Across Strata
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure5(preds_dir: str, manifest_path: str, out_dir: str):
    print("Generating Figure 5 (Reliability Diagrams)...")
    # Load 3-seed ensemble predictions for DenseNet-121 on Pleural Effusion
    ens_path = os.path.join(preds_dir, "preds_ensemble_3seeds.csv")
    ens_df = pd.read_csv(ens_path)
    
    sub = ens_df[
        (ens_df['architecture'] == 'densenet121') &
        (ens_df['target'] == 'Pleural Effusion')
    ].copy()
    
    uw_eff = sub[sub['loss_type'] == 'unweighted_bce'].copy()
    wt_eff = sub[sub['loss_type'] == 'weighted_bce'].copy()
    
    # Apply analytic correction to wt_eff ensemble: p_analytic = sigmoid(logit - log w)
    from src.training.loss import compute_pos_weights
    m_df = pd.read_csv(manifest_path, low_memory=False)
    w_eff = float(compute_pos_weights(m_df, ['Pleural Effusion'], split='train')[0].item())
    wt_eff['prob_analytic'] = (1.0 / (1.0 + np.exp(-(wt_eff['logit'] - np.log(w_eff))))).astype(np.float32)
    
    bins = ['T1_2015', 'T2_2016', 'T3_2017']
    bin_titles = ['Stratum T1 (2015 Baseline)', 'Stratum T2 (2016 Intermediate)', 'Stratum T3 (2017 Late Deployment)']
    
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), dpi=300, sharey=True)
    
    for idx, (t_bin, title) in enumerate(zip(bins, bin_titles)):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
        
        # 1. Unweighted Raw
        sub_uw = uw_eff[uw_eff['temporal_bin'] == t_bin]
        p_u = sub_uw['prob_raw'].values
        y_u = sub_uw['y_true'].values
        
        # 2. Weighted Raw
        sub_wt = wt_eff[wt_eff['temporal_bin'] == t_bin]
        p_w = sub_wt['prob_raw'].values
        y_w = sub_wt['y_true'].values
        
        # 3. Weighted Analytic
        p_a = sub_wt['prob_analytic'].values
        
        for p_vals, y_vals, label, color, marker in [
            (p_u, y_u, 'Unweighted Raw', '#2ca02c', 'o'),
            (p_w, y_w, 'Weighted Raw (Severe Bias)', '#d62728', '^'),
            (p_a, y_w, 'Weighted Analytic (z - log w)', '#1f77b4', 's'),
        ]:
            # Equal-mass 10 bins
            quantiles = np.linspace(0, 1, 11)
            bin_edges = np.quantile(p_vals, quantiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
            
            bin_true, bin_pred = [], []
            for b in range(10):
                mask = (p_vals >= bin_edges[b]) & (p_vals <= bin_edges[b+1])
                if np.sum(mask) > 0:
                    bin_true.append(float(np.mean(y_vals[mask])))
                    bin_pred.append(float(np.mean(p_vals[mask])))
            ax.plot(bin_pred, bin_true, marker=marker, linewidth=2.0, markersize=7, color=color, label=label)
            
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Predicted Probability")
        if idx == 0:
            ax.set_ylabel("Observed Empirical Frequency")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc='upper left')
        
    fig.tight_layout()
    fig5_path = os.path.join(out_dir, "fig5_reliability_diagrams.png")
    fig.savefig(fig5_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 5 to: {fig5_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 6: Empirical Logit Offset Verification (z_w - z_u vs log w)
# ─────────────────────────────────────────────────────────────────────────────

def generate_figure6(preds_dir: str, recalib_csv_path: str, out_dir: str):
    print("Generating Figure 6 (Empirical Logit Shift Verification)...")
    ens_path = os.path.join(preds_dir, "preds_ensemble_3seeds.csv")
    if not os.path.exists(ens_path):
        print(f"Warning: {ens_path} not found, skipping Figure 6.")
        return
        
    preds = pd.read_csv(ens_path)
    recalib = pd.read_csv(recalib_csv_path)
    
    targets = ['Pleural Effusion', 'Pneumonia', 'Cardiomegaly', 'Edema']
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)
    
    for idx, target in enumerate(targets):
        p_w = preds[(preds['loss_type'] == 'weighted_bce') & (preds['target'] == target)].set_index(['architecture', 'temporal_bin', 'image_id'])['logit']
        p_u = preds[(preds['loss_type'] == 'unweighted_bce') & (preds['target'] == target)].set_index(['architecture', 'temporal_bin', 'image_id'])['logit']
        diff = (p_w - p_u).dropna()
        
        pw_sub = recalib[(recalib['target'] == target) & (recalib['loss_type'] == 'weighted_bce')]
        pw = pw_sub['pos_weight'].dropna().iloc[0] if len(pw_sub) > 0 else 1.0
        log_w = float(np.log(pw))
        
        ax = axes[idx]
        sns.kdeplot(diff, ax=ax, fill=True, color='#1f77b4', alpha=0.3, linewidth=2)
        ax.axvline(log_w, color='#d62728', linestyle='--', linewidth=2, label=f'Theoretical $\\log w = {log_w:.2f}$')
        ax.axvline(diff.median(), color='#2ca02c', linestyle='-', linewidth=2, label=f'Observed Median $= {diff.median():.2f}$')
        
        ax.set_title(f'({chr(97+idx)}) {target}', fontweight='bold', fontsize=11)
        ax.set_xlabel(r'Logit Shift $\Delta z = z_w - z_u$', fontsize=10)
        ax.set_ylabel('Density' if idx == 0 else '', fontsize=10)
        ax.legend(fontsize=8, loc='upper right' if idx != 3 else 'upper left', framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    fig.tight_layout()
    fig6_path = os.path.join(out_dir, "fig6_logit_offset_verification.png")
    fig.savefig(fig6_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 6 to: {fig6_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  COMPILING ALL PUBLICATION-GRADE TABLES AND FIGURES")
    print("=" * 80)
    
    manifest_path = "data/processed/stage2_manifest.csv"
    recalib_csv = "reports/stage3/stage3_recalibration_full.csv"
    ci_csv = "reports/stage3/stage3_calibrator_improvements_ci.csv"
    deferral_csv = "reports/stage3/stage3_deferral_results.csv"
    preds_dir = "reports/stage2/predictions"
    
    tables_out = "reports/manuscript_tables"
    figures_out = "reports/manuscript_figures"
    os.makedirs(tables_out, exist_ok=True)
    os.makedirs(figures_out, exist_ok=True)
    
    # 1. Tables
    generate_table1(manifest_path, tables_out)
    generate_table2(recalib_csv, tables_out)
    generate_table3(recalib_csv, ci_csv, tables_out)
    generate_table4(deferral_csv, tables_out)
    
    # 2. Figures
    generate_figure2(recalib_csv, figures_out)
    generate_figure3(recalib_csv, figures_out)
    generate_figure4(deferral_csv, figures_out)
    generate_figure5(preds_dir, manifest_path, figures_out)
    generate_figure6(preds_dir, recalib_csv, figures_out)
    
    # Synchronize figures directly to manuscript/figures and manuscript_cibm/figures
    import shutil
    fig_names = [
        "fig2_drift_trajectories.png",
        "fig3_recalibration_comparison.png",
        "fig4_selective_risk_coverage.png",
        "fig5_reliability_diagrams.png",
        "fig6_logit_offset_verification.png",
    ]
    for target_dir in ["manuscript/figures", "manuscript_cibm/figures"]:
        os.makedirs(target_dir, exist_ok=True)
        for fig_name in fig_names:
            src = os.path.join(figures_out, fig_name)
            dst = os.path.join(target_dir, fig_name)
            if os.path.exists(src):
                shutil.copy(src, dst)
        print(f"✅ Synchronized figures to: {target_dir}")
        
    print("\n✅ All manuscript tables and figures generated successfully!")


if __name__ == "__main__":
    main()
