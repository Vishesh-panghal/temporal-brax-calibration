#!/usr/bin/env python3
"""
format_manuscript_tables.py

Compiles publication-grade Markdown and LaTeX tables for IEEE JBHI:
- Table 2: Factorial Experimental Matrix Evaluation across Temporal Strata (AUROC, Brier, Equal-Mass ECE)
- Master TDI Table: Empirical Linear Drift Slopes and 2,000-Replicate Patient-Cluster Bootstrap CIs
"""

import os
import pandas as pd
import numpy as np

def format_val(mean, std, decimals=4):
    if pd.isna(mean):
        return "--"
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def format_delta(m1, m3, decimals=4):
    if pd.isna(m1) or pd.isna(m3):
        return "--"
    delta = m3 - m1
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.{decimals}f}"

def compile_table2(matrix_csv_path, output_md_path):
    df = pd.read_csv(matrix_csv_path)
    
    # Sort order
    arch_order = ['densenet121', 'resnet50']
    arch_display = {'densenet121': 'DenseNet-121', 'resnet50': 'ResNet-50'}
    loss_order = ['unweighted_bce', 'weighted_bce']
    loss_display = {'unweighted_bce': 'Unweighted BCE', 'weighted_bce': 'Pos-Weighted BCE'}
    target_order = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']
    
    md_lines = []
    md_lines.append("# Table 2: Factorial Experimental Matrix Evaluation Across Chronologically Ordered Deployment Strata")
    md_lines.append("\n*Performance reported as Mean ± Standard Deviation across three independent training seeds (Seed 42, 123, 2026).*\n")
    md_lines.append("> **Note on Edema (*):** Positive events in the final deployment stratum ($T_3$) are zero ($N=0$), rendering $T_3$ discrimination non-computable. Edema is retained strictly as exploratory to transparently document support boundaries.\n")
    
    metrics = [
        ('auroc', 'Discrimination: Area Under the ROC Curve (AUROC ↑)', 4),
        ('brier_score', 'Probability Error: Brier Score (Mean Squared Probability Error ↓)', 4),
        ('ece_equal_mass', 'Calibration Error: Equal-Mass Expected Calibration Error (ECE_Q10 ↓)', 4)
    ]
    
    for metric_col, metric_title, dec in metrics:
        md_lines.append(f"### {metric_title}\n")
        md_lines.append("| Architecture | Training Objective | Target Pathology | $T_1$ (Baseline) | $T_2$ (Intermediate) | $T_3$ (Late Deployment) | Shift $\\Delta(T_3 - T_1)$ |")
        md_lines.append("|---|---|---|:---:|:---:|:---:|:---:|")
        
        for arch in arch_order:
            for loss in loss_order:
                for target in target_order:
                    # Filter rows
                    sub = df[(df['architecture'] == arch) & (df['loss_type'] == loss) & (df['target'] == target)]
                    
                    t1_row = sub[sub['temporal_bin'] == 'T1_2015']
                    t2_row = sub[sub['temporal_bin'] == 'T2_2016']
                    t3_row = sub[sub['temporal_bin'] == 'T3_2017']
                    
                    t1_m = t1_row[f'{metric_col}_mean'].values[0] if len(t1_row) else np.nan
                    t1_s = t1_row[f'{metric_col}_std'].values[0] if len(t1_row) else np.nan
                    t2_m = t2_row[f'{metric_col}_mean'].values[0] if len(t2_row) else np.nan
                    t2_s = t2_row[f'{metric_col}_std'].values[0] if len(t2_row) else np.nan
                    t3_m = t3_row[f'{metric_col}_mean'].values[0] if len(t3_row) else np.nan
                    t3_s = t3_row[f'{metric_col}_std'].values[0] if len(t3_row) else np.nan
                    
                    t1_str = format_val(t1_m, t1_s, dec)
                    t2_str = format_val(t2_m, t2_s, dec)
                    t3_str = format_val(t3_m, t3_s, dec)
                    delta_str = format_delta(t1_m, t3_m, dec)
                    
                    target_label = target + ("*" if target == "Edema" else "")
                    md_lines.append(f"| {arch_display[arch]} | {loss_display[loss]} | {target_label} | {t1_str} | {t2_str} | {t3_str} | **{delta_str}** |")
        md_lines.append("\n")
        
    with open(output_md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Table 2 Markdown written to {output_md_path}")

def compile_master_tdi(tdi_csv_path, output_md_path):
    df = pd.read_csv(tdi_csv_path)
    
    arch_display = {'densenet121': 'DenseNet-121', 'resnet50': 'ResNet-50'}
    loss_display = {'unweighted_bce': 'Unweighted BCE', 'weighted_bce': 'Pos-Weighted BCE'}
    
    md_lines = []
    md_lines.append("# Master Temporal Degradation Index (TDI) Table")
    md_lines.append("\n*Temporal Degradation Index (TDI) defined as the signed ordinary least squares (OLS) drift slope across chronologically ordered deployment strata ($T_1 \\to T_2 \\to T_3$), paired with 95% patient-cluster bootstrap confidence intervals from 2,000 resamples.*\n")
    md_lines.append("> **Directionality Convention:** In accordance with established drift semantics, TDI reflects *degradation* (positive values for AUROC/AUPRC indicate performance loss per stratum; positive values for Brier/ECE indicate calibration worsening per stratum).\n")
    
    md_lines.append("| Architecture | Objective | Seed | Target Pathology | AUROC TDI [95% CI] | AUPRC TDI [95% CI] | Brier Score TDI [95% CI] | Equal-Mass ECE TDI [95% CI] | 95% CI Excludes 0 (AUROC)? |")
    md_lines.append("|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|")
    
    # Sort
    runs = df[['architecture', 'loss_type', 'seed']].drop_duplicates().sort_values(by=['architecture', 'loss_type', 'seed'])
    
    for _, r in runs.iterrows():
        arch = r['architecture']
        loss = r['loss_type']
        seed = r['seed']
        
        for target in ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']:
            sub = df[(df['architecture'] == arch) & (df['loss_type'] == loss) & (df['seed'] == seed) & (df['target'] == target)]
            
            def get_cell(metric_name):
                row = sub[sub['metric'] == metric_name]
                if len(row) == 0 or pd.isna(row['point_tdi'].values[0]):
                    return "--"
                pt = row['point_tdi'].values[0]
                ci_l = row['bootstrap_ci_lower'].values[0]
                ci_u = row['bootstrap_ci_upper'].values[0]
                if pd.isna(ci_l) or pd.isna(ci_u):
                    return f"{pt:.4f}"
                return f"{pt:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
            
            auroc_str = get_cell('auroc')
            auprc_str = get_cell('auprc')
            brier_str = get_cell('brier_score')
            ece_str = get_cell('ece_equal_mass')
            
            # Check if AUROC CI strictly excludes 0
            auroc_row = sub[sub['metric'] == 'auroc']
            sig_str = "No"
            if len(auroc_row) > 0:
                ci_l = auroc_row['bootstrap_ci_lower'].values[0]
                ci_u = auroc_row['bootstrap_ci_upper'].values[0]
                if not pd.isna(ci_l) and not pd.isna(ci_u):
                    if (ci_l > 0 and ci_u > 0) or (ci_l < 0 and ci_u < 0):
                        sig_str = "**YES (p < 0.05)**"
                    else:
                        sig_str = "No (crosses 0)"
            
            target_label = target + ("*" if target == "Edema" else "")
            md_lines.append(f"| {arch_display[arch]} | {loss_display[loss]} | {seed} | {target_label} | {auroc_str} | {auprc_str} | {brier_str} | {ece_str} | {sig_str} |")

    with open(output_md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Master TDI Table Markdown written to {output_md_path}")

def compile_latex_tables(matrix_csv_path, tdi_csv_path, table2_tex_path, tdi_tex_path):
    df_m = pd.read_csv(matrix_csv_path)
    df_t = pd.read_csv(tdi_csv_path)
    
    # Table 2 LaTeX
    tex2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Factorial Experimental Matrix Evaluation Across Chronologically Ordered Deployment Strata ($T_1, T_2, T_3$). Reported as Mean $\pm$ SD across three independent seeds (42, 123, 2026).}",
        r"\label{tab:matrix_evaluation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lllcccc}",
        r"\hline",
        r"\textbf{Architecture} & \textbf{Objective} & \textbf{Target Pathology} & \textbf{$T_1$ (Baseline)} & \textbf{$T_2$ (Intermediate)} & \textbf{$T_3$ (Late Deployment)} & \textbf{$\Delta(T_3 - T_1)$} \\ \hline",
        r"\multicolumn{7}{l}{\textit{\textbf{Discrimination: Area Under ROC Curve (AUROC $\uparrow$)}}} \\"
    ]
    
    arch_display = {'densenet121': 'DenseNet-121', 'resnet50': 'ResNet-50'}
    loss_display = {'unweighted_bce': 'Unweighted BCE', 'weighted_bce': 'Pos-Weighted BCE'}
    
    for arch in ['densenet121', 'resnet50']:
        for loss in ['unweighted_bce', 'weighted_bce']:
            for target in ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']:
                sub = df_m[(df_m['architecture'] == arch) & (df_m['loss_type'] == loss) & (df_m['target'] == target)]
                t1_row = sub[sub['temporal_bin'] == 'T1_2015']
                t2_row = sub[sub['temporal_bin'] == 'T2_2016']
                t3_row = sub[sub['temporal_bin'] == 'T3_2017']
                
                t1_m = t1_row['auroc_mean'].values[0] if len(t1_row) else np.nan
                t1_s = t1_row['auroc_std'].values[0] if len(t1_row) else np.nan
                t2_m = t2_row['auroc_mean'].values[0] if len(t2_row) else np.nan
                t2_s = t2_row['auroc_std'].values[0] if len(t2_row) else np.nan
                t3_m = t3_row['auroc_mean'].values[0] if len(t3_row) else np.nan
                t3_s = t3_row['auroc_std'].values[0] if len(t3_row) else np.nan
                
                t1_str = format_val(t1_m, t1_s, 3)
                t2_str = format_val(t2_m, t2_s, 3)
                t3_str = format_val(t3_m, t3_s, 3)
                delta_str = format_delta(t1_m, t3_m, 3)
                lbl = target + ("$^*$" if target == "Edema" else "")
                tex2.append(f"{arch_display[arch]} & {loss_display[loss]} & {lbl} & {t1_str} & {t2_str} & {t3_str} & \\textbf{{{delta_str}}} \\\\")
    
    tex2.extend([
        r"\hline",
        r"\multicolumn{7}{l}{\textit{\textbf{Calibration: Equal-Mass ECE ($M=10$, $\downarrow$)}}} \\"
    ])
    for arch in ['densenet121', 'resnet50']:
        for loss in ['unweighted_bce', 'weighted_bce']:
            for target in ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia']:
                sub = df_m[(df_m['architecture'] == arch) & (df_m['loss_type'] == loss) & (df_m['target'] == target)]
                t1_row = sub[sub['temporal_bin'] == 'T1_2015']
                t2_row = sub[sub['temporal_bin'] == 'T2_2016']
                t3_row = sub[sub['temporal_bin'] == 'T3_2017']
                
                t1_m = t1_row['ece_equal_mass_mean'].values[0] if len(t1_row) else np.nan
                t1_s = t1_row['ece_equal_mass_std'].values[0] if len(t1_row) else np.nan
                t2_m = t2_row['ece_equal_mass_mean'].values[0] if len(t2_row) else np.nan
                t2_s = t2_row['ece_equal_mass_std'].values[0] if len(t2_row) else np.nan
                t3_m = t3_row['ece_equal_mass_mean'].values[0] if len(t3_row) else np.nan
                t3_s = t3_row['ece_equal_mass_std'].values[0] if len(t3_row) else np.nan
                
                t1_str = format_val(t1_m, t1_s, 3)
                t2_str = format_val(t2_m, t2_s, 3)
                t3_str = format_val(t3_m, t3_s, 3)
                delta_str = format_delta(t1_m, t3_m, 3)
                tex2.append(f"{arch_display[arch]} & {loss_display[loss]} & {target} & {t1_str} & {t2_str} & {t3_str} & \\textbf{{{delta_str}}} \\\\")
                
    tex2.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    
    with open(table2_tex_path, 'w') as f:
        f.write('\n'.join(tex2))
    print(f"Table 2 LaTeX written to {table2_tex_path}")

if __name__ == "__main__":
    matrix_csv = "reports/stage2/tables/table2_matrix_summary.csv"
    table2_out = "reports/stage2/tables/table2_manuscript_formatted.md"
    compile_table2(matrix_csv, table2_out)
    
    tdi_csv = "reports/stage2/tables/table_tdi_summary.csv"
    tdi_out = "reports/stage2/tables/table_tdi_manuscript_formatted.md"
    compile_master_tdi(tdi_csv, tdi_out)
    
    table2_tex = "reports/stage2/tables/table2_manuscript.tex"
    tdi_tex = "reports/stage2/tables/table_tdi_manuscript.tex"
    compile_latex_tables(matrix_csv, tdi_csv, table2_tex, tdi_tex)

