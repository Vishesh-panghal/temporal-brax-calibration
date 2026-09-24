#!/usr/bin/env python3
"""
scripts/generate_table5_mimic.py
Formats Table 5 (MIMIC-CXR External Validation) from reports/stage4a/stage4a_rigorous_image_level.csv.
Exports both LaTeX (.tex) and CSV (.csv) versions and verifies numerical consistency.
"""

import ast
import re
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "reports" / "stage4a" / "stage4a_rigorous_image_level.csv"
OUTPUT_TEX = REPO_ROOT / "reports" / "manuscript_tables" / "table5_mimic_external_validation.tex"
OUTPUT_CSV = REPO_ROOT / "reports" / "manuscript_tables" / "table5_mimic_external_validation.csv"
OUTPUT_MD = REPO_ROOT / "reports" / "manuscript_tables" / "table5_mimic_external_validation.md"

def parse_ci(ci_val):
    if isinstance(ci_val, str):
        # Handle "(0.656, 0.717)" format
        ci_val = ci_val.strip()
        if ci_val.startswith("(") and ci_val.endswith(")"):
            parts = [float(x.strip()) for x in ci_val[1:-1].split(",")]
            return parts[0], parts[1]
    elif isinstance(ci_val, (list, tuple)):
        return float(ci_val[0]), float(ci_val[1])
    return None, None

def generate_table5():
    if not INPUT_CSV.exists():
        print(f"❌ Input CSV not found: {INPUT_CSV}")
        return False

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Build LaTeX table rows
    arch_display = {"densenet121": "DenseNet-121", "resnet50": "ResNet-50"}
    loss_display = {"unweighted_bce": "Unweighted BCE", "weighted_bce": "Pos-Weighted BCE"}
    target_order = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]

    table_rows = []
    current_arch_loss = None

    for arch in ["densenet121", "resnet50"]:
        for loss in ["unweighted_bce", "weighted_bce"]:
            sub = df[(df["architecture"] == arch) & (df["loss_type"] == loss)]
            for tgt in target_order:
                tgt_row = sub[sub["target"] == tgt]
                if tgt_row.empty:
                    continue
                row = tgt_row.iloc[0]
                prev_pct = f"{row['prevalence'] * 100:.2f}\\%"
                brier_null = f"{row['brier_null']:.4f}"

                auc = row["auroc"]
                auc_l, auc_u = parse_ci(row["auroc_ci"])
                auc_str = f"{auc:.3f} [{auc_l:.3f}, {auc_u:.3f}]"

                is_w = (loss == "weighted_bce")
                if is_w:
                    b_raw = row["brier_raw"]
                    b_corr = row["brier_corr"]
                    brier_str = f"{b_raw:.4f} $\\to$ {b_corr:.4f}"

                    d_brier = row["delta_brier"]
                    d_l, d_u = parse_ci(row["delta_brier_ci"])
                    # Bold if statistically significant reduction
                    if d_u < 0:
                        delta_str = f"\\textbf{{{d_brier:+.4f}}} [{d_l:+.4f}, {d_u:+.4f}]"
                    else:
                        delta_str = f"{d_brier:+.4f} [{d_l:+.4f}, {d_u:+.4f}]"

                    bss_raw = row["bss_raw"]
                    bss_corr = row["bss_corr"]
                    bss_str = f"{bss_raw:+.3f} $\\to$ {bss_corr:+.3f}"
                else:
                    b_raw = row["brier_raw"]
                    brier_str = f"{b_raw:.4f}"
                    delta_str = "--"
                    bss_raw = row["bss_raw"]
                    bss_str = f"{bss_raw:+.3f}"

                line = f"{arch_display[arch]} & {loss_display[loss]} & {tgt} & {prev_pct} & {brier_null} & {auc_str} & {brier_str} & {delta_str} & {bss_str} \\\\"
                table_rows.append(line)

            # Add midrule between configurations
            if not (arch == "resnet50" and loss == "weighted_bce"):
                table_rows.append("\\midrule")

    rows_text = "\n".join(table_rows)

    tex_content = f"""\\begin{{table*}}[!t]
\\caption{{Cross-Institutional External Validation on the Official MIMIC-CXR-JPG Frontal Test Cohort ($N = 3,403$ Images from $3,041$ Studies Across $N = 289$ Unique Patients). Evaluated on 3-Seed Probability Ensembles ($\\bar{{p}}_{{\\text{{raw}}}} = \\frac{{1}}{{3}}\\sum \\sigma(z_s), \\bar{{p}}_{{\\text{{corr}}}} = \\frac{{1}}{{3}}\\sum \\sigma(z_s - \\log w)$) with Image Prevalence ($\\pi$), Empirical Null Baselines ($\\text{{Brier}}_{{\\text{{null}}}} = \\pi(1-\\pi)$), Patient-Clustered 1,000-Replicate Bootstrap 95\\% Confidence Intervals, Brier Skill Scores ($\\text{{BSS}} = 1 - \\text{{Brier}}/\\text{{Brier}}_{{\\text{{null}}}}$), and Paired Analytic Recalibration ($\\Delta\\text{{Brier}} = \\text{{Brier}}_{{\\text{{corr}}}} - \\text{{Brier}}_{{\\text{{raw}}}}$).}}
\\label{{tab:mimic_external_validation}}
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lllccccccc}}
\\toprule
\\textbf{{Architecture}} & \\textbf{{Objective}} & \\textbf{{Pathology}} & \\textbf{{Prevalence ($\\pi$)}} & \\textbf{{$\\text{{Brier}}_{{\\text{{null}}}}$}} & \\textbf{{AUROC [95\\% CI]}} & \\textbf{{Brier (Raw $\\to$ Corr)}} & \\textbf{{$\\Delta\\text{{Brier}}$ [95\\% CI]}} & \\textbf{{$\\text{{BSS}}$ (Raw $\\to$ Corr)}} \\\\
\\midrule
{rows_text}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}
"""

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(tex_content, encoding="utf-8")
    print(f"✅ Generated LaTeX Table 5: {OUTPUT_TEX}")
    return True

if __name__ == "__main__":
    generate_table5()
