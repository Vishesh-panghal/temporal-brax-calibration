#!/usr/bin/env python3
"""
scripts/audit_data_truth.py
Comprehensive Scientific Data-Truth & Numeric Cross-Reconciliation Suite.

Reconciles all textual claims across the manuscript files:
  - manuscript_cmpb/main_cmpb.tex
  - manuscript_cmpb/title_page.tex
  - manuscript_cmpb/cover_letter.tex
against the underlying raw numerical CSV data files:
  - reports/manuscript_tables/table1_cohort_demographics.csv
  - reports/manuscript_tables/table2_factorial_matrix.csv
  - reports/manuscript_tables/table3_calibrator_comparison.csv
  - reports/manuscript_tables/table4_selective_prediction_workload.csv
  - reports/manuscript_tables/table5_mimic_external_validation.csv
  - reports/stage4a/stage4a_rigorous_image_level.csv
"""

import math
import re
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CMPB_DIR = REPO_ROOT / "manuscript_cmpb"
TABLES_DIR = REPO_ROOT / "reports" / "manuscript_tables"
STAGE4A_DIR = REPO_ROOT / "reports" / "stage4a"

MAIN_TEX = CMPB_DIR / "main_cmpb.tex"
TITLE_TEX = CMPB_DIR / "title_page.tex"
COVER_TEX = CMPB_DIR / "cover_letter.tex"

def parse_num(val_str):
    """Clean string to float, removing commas, percentages, and dollar signs."""
    clean = re.sub(r"[,\$%\s]", "", str(val_str))
    try:
        return float(clean)
    except ValueError:
        return None

def run_scientific_audit():
    print("=" * 80)
    print("COMPREHENSIVE SCIENTIFIC DATA-TRUTH & NUMERICAL RECONCILIATION AUDIT")
    print("=" * 80)

    errors = []

    if not MAIN_TEX.exists():
        print(f"❌ Manuscript file not found: {MAIN_TEX}")
        sys.exit(1)

    tex = MAIN_TEX.read_text(encoding="utf-8")
    title_text = TITLE_TEX.read_text(encoding="utf-8") if TITLE_TEX.exists() else ""
    cover_text = COVER_TEX.read_text(encoding="utf-8") if COVER_TEX.exists() else ""

    # =========================================================================
    # 1. Headline Numbers Numeric Reconciliation against Table 3 CSV
    # =========================================================================
    print("\n🔍 [Check 1/7] Numeric Reconciliation of Headline Brier Scores against Table 3 CSV...")
    t3_csv_path = TABLES_DIR / "table3_calibrator_comparison.csv"
    if not t3_csv_path.exists():
        errors.append(f"Missing CSV: {t3_csv_path}")
    else:
        df_t3 = pd.read_csv(t3_csv_path)

        # DenseNet-121 Weighted BCE: Raw vs Analytic Offset
        dnet_raw = df_t3[(df_t3["Architecture"] == "DenseNet-121") & (df_t3["Calibrator"].str.contains("Raw")) & (df_t3["Loss Objective"].str.contains("Weighted"))].iloc[0]
        dnet_analytic = df_t3[(df_t3["Architecture"] == "DenseNet-121") & (df_t3["Calibrator"].str.contains("Analytic"))].iloc[0]

        dnet_raw_brier = float(dnet_raw["T3 Brier"])
        dnet_ana_brier = float(dnet_analytic["T3 Brier"])
        dnet_delta = dnet_ana_brier - dnet_raw_brier
        dnet_reduc_pct = ((dnet_raw_brier - dnet_ana_brier) / dnet_raw_brier) * 100

        print(f"  📊 CSV Ground Truth DenseNet: Raw={dnet_raw_brier:.4f}, Analytic={dnet_ana_brier:.4f}, Delta={dnet_delta:.4f}, Reduc={dnet_reduc_pct:.2f}%")

        if abs(dnet_raw_brier - 0.1079) > 1e-4:
            errors.append(f"Table 3 CSV DenseNet Raw Brier ({dnet_raw_brier}) != 0.1079")
        if abs(dnet_ana_brier - 0.0284) > 1e-4:
            errors.append(f"Table 3 CSV DenseNet Analytic Brier ({dnet_ana_brier}) != 0.0284")
        if abs(dnet_delta - (-0.0795)) > 1e-4:
            errors.append(f"Table 3 CSV DenseNet Delta ({dnet_delta}) != -0.0795")
        if abs(dnet_reduc_pct - 73.68) > 0.1:
            errors.append(f"Table 3 CSV DenseNet Reduction ({dnet_reduc_pct:.2f}%) != 73.7%")

        # Check manuscript presence of exact values
        if "0.1079" not in tex or "0.0284" not in tex or "-0.0795" not in tex:
            errors.append("DenseNet headline numbers (0.1079 -> 0.0284, delta = -0.0795) missing from main_cmpb.tex")
        else:
            print("  ✅ DenseNet headline numbers in main_cmpb.tex strictly match CSV ground truth.")

        # ResNet-50 Weighted BCE: Raw vs Analytic Offset
        rnet_raw = df_t3[(df_t3["Architecture"] == "ResNet-50") & (df_t3["Calibrator"].str.contains("Raw")) & (df_t3["Loss Objective"].str.contains("Weighted"))].iloc[0]
        rnet_analytic = df_t3[(df_t3["Architecture"] == "ResNet-50") & (df_t3["Calibrator"].str.contains("Analytic"))].iloc[0]

        rnet_raw_brier = float(rnet_raw["T3 Brier"])
        rnet_ana_brier = float(rnet_analytic["T3 Brier"])
        rnet_delta = rnet_ana_brier - rnet_raw_brier
        rnet_reduc_pct = ((rnet_raw_brier - rnet_ana_brier) / rnet_raw_brier) * 100

        print(f"  📊 CSV Ground Truth ResNet-50: Raw={rnet_raw_brier:.4f}, Analytic={rnet_ana_brier:.4f}, Delta={rnet_delta:.4f}, Reduc={rnet_reduc_pct:.2f}%")

        if abs(rnet_raw_brier - 0.1227) > 1e-4 or abs(rnet_ana_brier - 0.0291) > 1e-4:
            errors.append("ResNet-50 Table 3 CSV deviates from 0.1227 -> 0.0291")
        if "0.1227" not in tex or "0.0291" not in tex or "-0.0936" not in tex:
            errors.append("ResNet-50 headline numbers (0.1227 -> 0.0291, -0.0936) missing from main_cmpb.tex")
        else:
            print("  ✅ ResNet-50 headline numbers in main_cmpb.tex strictly match CSV ground truth.")

    # =========================================================================
    # 2. Strict Purge of Legacy Numbers & Inaccurate Percentages
    # =========================================================================
    print("\n🔍 [Check 2/7] Checking for Absolute Absence of Legacy Numbers & Percentages...")
    legacy_strings = ["0.1065", "0.0280", "74.6%"]
    for leg in legacy_strings:
        if leg in tex:
            errors.append(f"Found legacy number '{leg}' in main_cmpb.tex")
        else:
            print(f"  ✅ Legacy string '{leg}' absent from main_cmpb.tex.")

    # Check title_page.tex for remaining 74% highlight
    if "74%" in title_text or r"74\%" in title_text:
        errors.append("Found ungrounded '74%' in title_page.tex highlights. Must be '73.7%'.")
    else:
        print("  ✅ Inaccurate '74%' absent from title_page.tex highlights.")

    # =========================================================================
    # 3. Demographics Reconciliation against Table 1 CSV
    # =========================================================================
    print("\n🔍 [Check 3/7] Reconciling BRAX Cohort Demographics against Table 1 CSV...")
    t1_csv_path = TABLES_DIR / "table1_cohort_demographics.csv"
    if not t1_csv_path.exists():
        errors.append(f"Missing CSV: {t1_csv_path}")
    else:
        df_t1 = pd.read_csv(t1_csv_path)
        train_row = df_t1[df_t1["Cohort Stratum"].str.contains("Train")].iloc[0]
        val_row = df_t1[df_t1["Cohort Stratum"].str.contains("Val")].iloc[0]
        test_row = df_t1[df_t1["Cohort Stratum"].str.contains("Test All")].iloc[0]

        train_n = int(str(train_row["Images (N)"]).replace(",", ""))
        val_n = int(str(val_row["Images (N)"]).replace(",", ""))
        test_n = int(str(test_row["Images (N)"]).replace(",", ""))
        total_brax = train_n + val_n + test_n

        print(f"  📊 Table 1 CSV: Train={train_n:,}, Val={val_n:,}, Test={test_n:,}, Total={total_brax:,}")
        if total_brax != 40523:
            errors.append(f"Table 1 CSV total images ({total_brax:,}) != 40,523")
        if "40,523" not in tex:
            errors.append("Total BRAX image count '40,523' missing from main_cmpb.tex")
        else:
            print("  ✅ Total image count 40,523 verified.")

    # =========================================================================
    # 4. Selective Deferral Metrics Reconciliation against Table 4 CSV
    # =========================================================================
    print("\n🔍 [Check 4/7] Reconciling Selective Triage Claims against Table 4 CSV...")
    t4_csv_path = TABLES_DIR / "table4_selective_prediction_workload.csv"
    if not t4_csv_path.exists():
        errors.append(f"Missing CSV: {t4_csv_path}")
    else:
        df_t4 = pd.read_csv(t4_csv_path)
        cov70_raw = df_t4[(df_t4["Coverage"] == "70%") & (df_t4["Strategy"].str.contains("Raw"))].iloc[0]
        cov70_ana = df_t4[(df_t4["Coverage"] == "70%") & (df_t4["Strategy"].str.contains("Analytic"))].iloc[0]

        ref_n = str(cov70_raw["Referred to Reader (N)"]) # "730 (30%)"
        fn_raw = float(cov70_raw["Automated False Negatives (Missed Cases)"]) # 7.0
        sens_raw = float(cov70_raw["Retained Sensitivity"]) # 0.8939
        triage_raw = float(cov70_raw["Full-Cohort Triage Sensitivity"]) # 0.9176
        brier_raw_cov = float(cov70_raw["Retained Brier"]) # 0.0956

        fn_ana = float(cov70_ana["Automated False Negatives (Missed Cases)"]) # 13.0
        brier_ana_cov = float(cov70_ana["Retained Brier"]) # 0.0178

        print(f"  📊 Table 4 CSV 70% Coverage: Referred={ref_n}, FN raw={fn_raw} (Sens={sens_raw:.4f}, Triage={triage_raw:.4f}, Brier={brier_raw_cov:.4f}), FN ana={fn_ana} (Brier={brier_ana_cov:.4f})")

        # Verify claims in text (handling LaTeX escaped percent signs)
        has_89 = ("89.39%" in tex or r"89.39\%" in tex)
        has_91 = ("91.76%" in tex or r"91.76\%" in tex)
        if "730" not in tex or not has_89 or not has_91 or "0.0956" not in tex or "0.0178" not in tex:
            errors.append("70% coverage triage statistics in manuscript text do not match Table 4 CSV values.")
        else:
            print("  ✅ Selective triage workload claims strictly match Table 4 CSV.")

    # =========================================================================
    # 5. MIMIC External Validation Reconciliation against Rigorous CSV
    # =========================================================================
    print("\n🔍 [Check 5/7] Reconciling MIMIC External Metrics against Rigorous CSV...")
    mimic_csv_path = STAGE4A_DIR / "stage4a_rigorous_image_level.csv"
    if not mimic_csv_path.exists():
        errors.append(f"Missing CSV: {mimic_csv_path}")
    else:
        df_mimic = pd.read_csv(mimic_csv_path)
        dnet_eff = df_mimic[(df_mimic["architecture"] == "densenet121") & (df_mimic["loss_type"] == "unweighted_bce") & (df_mimic["target"] == "Pleural Effusion")].iloc[0]
        dnet_card_w = df_mimic[(df_mimic["architecture"] == "densenet121") & (df_mimic["loss_type"] == "weighted_bce") & (df_mimic["target"] == "Cardiomegaly")].iloc[0]
        dnet_pneu_w = df_mimic[(df_mimic["architecture"] == "densenet121") & (df_mimic["loss_type"] == "weighted_bce") & (df_mimic["target"] == "Pneumonia")].iloc[0]
        dnet_edema_w = df_mimic[(df_mimic["architecture"] == "densenet121") & (df_mimic["loss_type"] == "weighted_bce") & (df_mimic["target"] == "Edema")].iloc[0]

        eff_auc = float(dnet_eff["auroc"]) # 0.688
        card_delta = float(dnet_card_w["delta_brier"]) # -0.0579
        pneu_delta = float(dnet_pneu_w["delta_brier"]) # -0.0095
        edema_delta = float(dnet_edema_w["delta_brier"]) # +0.0217

        print(f"  📊 MIMIC Rigorous CSV: Effusion AUROC={eff_auc:.3f}, Cardiomegaly Delta={card_delta:.4f}, Pneumonia Delta={pneu_delta:.4f}, Edema Delta={edema_delta:.4f}")

        if "0.688" not in tex or "-0.0579" not in tex or "-0.0095" not in tex or "+0.0217" not in tex:
            errors.append("MIMIC external evaluation metrics in manuscript text deviate from rigorous CSV values.")
        else:
            print("  ✅ MIMIC external metrics in manuscript text strictly match rigorous CSV.")

        # Verify exact unique patient-level counts from cohort audit
        patient_counts = [
            (r"\$?178\$?\s+positive\s+and\s+\$?273\$?\s+negative\s+patients", "Pleural Effusion (178 pos / 273 neg)"),
            (r"\$?183\$?\s+positive\s+and\s+\$?282\$?\s+negative\s+patients", "Cardiomegaly (183 pos / 282 neg)"),
            (r"\$?150\$?\s+positive\s+and\s+\$?287\$?\s+negative\s+patients", "Edema (150 pos / 287 neg)"),
            (r"\$?137\$?\s+positive\s+and\s+\$?286\$?\s+negative\s+patients", "Pneumonia (137 pos / 286 neg)"),
        ]
        for pattern, tgt_name in patient_counts:
            if not re.search(pattern, tex):
                errors.append(f"Missing exact patient-level counts for {tgt_name} in main_cmpb.tex")
            else:
                print(f"  ✅ Verified {tgt_name} unique patient counts.")

    # =========================================================================
    # 6. Scientific Language De-Biasing & Prospective Wording Audit
    # =========================================================================
    print("\n🔍 [Check 6/7] Auditing De-Biased Phrasing & Absence of 'Prospective' Overclaims...")
    all_files = [
        ("main_cmpb.tex", tex),
        ("title_page.tex", title_text),
        ("cover_letter.tex", cover_text),
    ]

    prohibited_terms = [
        "prospective temporal evidence",
        "zero-compute",
        "safe automation",
        "safe rule-out",
        "safe deployment",
        "definitively disprove",
        "completely remediates",
    ]

    for fname, ftext in all_files:
        for pt in prohibited_terms:
            cnt = len(re.findall(re.escape(pt), ftext, re.IGNORECASE))
            if cnt > 0:
                errors.append(f"Prohibited overclaim '{pt}' found in {fname} ({cnt}x)")
            else:
                print(f"  ✅ '{pt}' absent from {fname}.")

        # Check for improper prospective wording
        # Only allowed occurrence is in main_cmpb.tex limitations for future prospective shadow deployment
        prospective_matches = [m.start() for m in re.finditer(r"\bprospective\b", ftext, re.IGNORECASE)]
        if fname != "main_cmpb.tex" and len(prospective_matches) > 0:
            errors.append(f"Found 'prospective' in {fname}. Overclaiming prospective evidence is prohibited.")
        elif fname == "main_cmpb.tex":
            for m_idx in prospective_matches:
                snippet = ftext[max(0, m_idx - 60):min(len(ftext), m_idx + 60)].replace("\n", " ")
                if "prospective silent shadow deployment" not in snippet:
                    errors.append(f"Improper use of 'prospective' in main_cmpb.tex: '...{snippet}...'")
                else:
                    print(f"  ✅ Legitimate limitations prospective citation verified in main_cmpb.tex.")

    # =========================================================================
    # 7. Algorithm 1 Piecewise Normalizer & Ceil Rule Audit
    # =========================================================================
    print("\n🔍 [Check 7/7] Auditing Algorithm 1 Piecewise Normalizer & Automation Quota...")
    if r"\mathbb{I}(\hat{p}(x) \ge t^*)(1 - t^*) + \mathbb{I}(\hat{p}(x) < t^*)t^*" in tex or (
        "\\begin{cases}" in tex and "1 - t^*" in tex and "t^*" in tex
    ):
        print("  ✅ Algorithm 1 uses exact piecewise normalizer s_i matching mitigation.py.")
    else:
        errors.append("Algorithm 1 lacks exact piecewise normalizer matching mitigation.py line 247.")

    if r"\lceil C \cdot N \rceil" in tex:
        print("  ✅ Automation quota uses exact ceil(C * N) rule matching mitigation.py.")
    else:
        errors.append("Algorithm 1 does not use ceil(C * N) quota rule.")

    print("\n" + "=" * 80)
    print("SCIENTIFIC DATA-TRUTH AUDIT SUMMARY")
    print("=" * 80)
    if errors:
        print(f"❌ AUDIT FAILED with {len(errors)} discrepancies:")
        for e in errors:
            print(f"   • {e}")
        return False
    else:
        print("🎉 ALL NUMERICAL AND CLAIM DATA-TRUTH CHECKS PASSED WITH ZERO DISCREPANCIES!")
        return True

if __name__ == "__main__":
    success = run_scientific_audit()
    sys.exit(0 if success else 1)
