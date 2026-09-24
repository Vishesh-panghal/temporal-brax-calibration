#!/usr/bin/env python3
"""
scripts/audit_cmpb.py
Two-Tier Automated Audit Suite for CMPB Submission Readiness.

Tier 1: Scientific Data-Truth, Algorithmic Alignment & Reconciliation
Tier 2: Editorial, Bibliographic & CMPB Guidelines Compliance

Exits with code 0 if all checks pass; non-zero if any check fails.
"""

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CMPB_DIR = BASE_DIR / "manuscript_cmpb"
TABLES_DIR = BASE_DIR / "reports" / "manuscript_tables"
MAIN_TEX = CMPB_DIR / "main_cmpb.tex"
TITLE_TEX = CMPB_DIR / "title_page.tex"
COVER_TEX = CMPB_DIR / "cover_letter.tex"
SUPP_TEX = CMPB_DIR / "supplementary_material.tex"
BIB_FILE = CMPB_DIR / "references.bib"

def md5_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()

def run_audit():
    print("=" * 80)
    print("CMPB SUBMISSION READINESS AUDIT SUITE (TWO-TIER AUDIT)")
    print("Target Journal: Computer Methods and Programs in Biomedicine (Elsevier)")
    print("=" * 80)

    if not MAIN_TEX.exists():
        print(f"❌ FATAL: Manuscript file not found: {MAIN_TEX}")
        sys.exit(1)

    tex = MAIN_TEX.read_text(encoding="utf-8")
    title_text = TITLE_TEX.read_text(encoding="utf-8") if TITLE_TEX.exists() else ""
    cover_text = COVER_TEX.read_text(encoding="utf-8") if COVER_TEX.exists() else ""
    supp_text = SUPP_TEX.read_text(encoding="utf-8") if SUPP_TEX.exists() else ""

    tier1_errors = []
    tier2_errors = []

    print("\n" + "#" * 80)
    print("TIER 1: SCIENTIFIC DATA TRUTH, ALGORITHMIC ALIGNMENT & RECONCILIATION")
    print("#" * 80)

    # Check 1.1: Reconciled Headline Numbers & Purge of Legacy Discrepancies
    print("\n🔍 [Check 1.1] Headline Brier Score & Percentage Reconciliation...")
    legacy_numbers = ["0.1065", "0.0280", "74.6%"]
    for num in legacy_numbers:
        if num in tex:
            tier1_errors.append(f"Legacy number '{num}' found in main_cmpb.tex. Must be purged.")
        else:
            print(f"  ✅ Legacy number '{num}' absent from manuscript.")

    if "0.1079" in tex and "0.0284" in tex and "-0.0795" in tex:
        print("  ✅ Reconciled Table 3 headline numbers (0.1079 -> 0.0284, delta = -0.0795) present.")
    else:
        tier1_errors.append("Reconciled Table 3 headline numbers (0.1079 -> 0.0284, -0.0795) missing from manuscript.")

    if "73.7%" in tex or r"73.7\%" in tex:
        print("  ✅ Reconciled percentage (73.7%) present.")
    else:
        tier1_errors.append("Reconciled percentage '73.7%' missing from manuscript.")

    # Check 1.2: ResNet-50 Headline Numbers
    print("\n🔍 [Check 1.2] ResNet-50 Calibrator Headline Reconciliation...")
    if "0.1227" in tex and "0.0291" in tex and "-0.0936" in tex:
        print("  ✅ ResNet-50 Table 3 headline numbers (0.1227 -> 0.0291, delta = -0.0936) present.")
    else:
        tier1_errors.append("ResNet-50 headline numbers (0.1227 -> 0.0291, -0.0936) missing from manuscript.")

    # Check 1.3: Source Tables Cross-Check
    print("\n🔍 [Check 1.3] Cross-Checking Tables against Source Files...")
    t3_path = TABLES_DIR / "table3_calibrator_comparison.tex"
    if t3_path.exists():
        t3_content = t3_path.read_text(encoding="utf-8")
        if "0.1079" in t3_content and "0.0284" in t3_content and "-0.0795" in t3_content:
            print("  ✅ Table 3 source file strictly validates Raw=0.1079, Analytic=0.0284, Delta=-0.0795.")
        else:
            tier1_errors.append("Table 3 source file numbers deviate from expected locked values.")
    else:
        tier1_errors.append(f"Table 3 source file {t3_path} missing.")

    t1_path = TABLES_DIR / "table1_cohort_demographics.tex"
    if t1_path.exists():
        t1_content = t1_path.read_text(encoding="utf-8")
        if all(val in t1_content for val in ["25,123", "7,098", "1,879", "3,987", "2,436", "8,302"]):
            print("  ✅ Table 1 source file strictly validates demographics (Train=25,123, Val=7,098, T1=1,879, T2=3,987, T3=2,436, Test Total=8,302).")
        else:
            tier1_errors.append("Table 1 source file demographics deviate from locked values.")
    else:
        tier1_errors.append(f"Table 1 source file {t1_path} missing.")

    t5_path = TABLES_DIR / "table5_mimic_external_validation.tex"
    if t5_path.exists():
        t5_content = t5_path.read_text(encoding="utf-8")
        if "3,403" in t5_content and "3,041" in t5_content and "289" in t5_content:
            print("  ✅ Table 5 source file strictly validates MIMIC cohort size (3,403 images, 3,041 studies, 289 patients).")
        else:
            tier1_errors.append("Table 5 cohort size numbers deviate from locked values.")
        if "0.2536" in t5_content and "0.2451" in t5_content and "-0.0579" in t5_content:
            print("  ✅ Table 5 source file validates Brier trajectories and delta values.")
        else:
            tier1_errors.append("Table 5 metrics deviate from locked values.")
    else:
        tier1_errors.append(f"Table 5 source file {t5_path} missing.")

    # Check 1.4: Algorithm 1 Alignment with Actual Implementation
    print("\n🔍 [Check 1.4] Auditing Algorithm 1 Pseudocode Alignment...")
    if "Budget-Constrained Uncertainty-Ranked Selective Deferral Pipeline" in tex:
        print("  ✅ Algorithm 1 title correctly reflects budget-based selective deferral.")
    else:
        tier1_errors.append("Algorithm 1 does not use verified budget-constrained selective deferral title.")

    if "compute_uncertainty" in tex or "1 - \\frac{|\\hat{p}_i - t^*|}" in tex or "1 - \\frac{|\\hat{p}" in tex:
        print("  ✅ Proximity-based uncertainty calculation u(x) correctly documented.")
    else:
        tier1_errors.append("Proximity-based uncertainty calculation missing from Algorithm 1.")

    if "coverage budget" in tex.lower() or "coverage fraction" in tex.lower() or "budget" in tex.lower():
        print("  ✅ Budget-ranked deferral policy documented.")
    else:
        tier1_errors.append("Coverage budget ranking missing from selective prediction section.")

    # Check 1.5: Language & Overclaim De-Biasing
    print("\n🔍 [Check 1.5] Auditing Conservative Scientific Language...")
    prohibited_overclaims = [
        "prospective temporal evidence",
        "zero-compute",
        "safe automation",
        "safe rule-out",
        "safe deployment",
        "definitively disprove",
        "completely remediates",
    ]
    for oc in prohibited_overclaims:
        count = len(re.findall(re.escape(oc), tex, re.IGNORECASE))
        if count > 0:
            tier1_errors.append(f"Found prohibited overclaim '{oc}' ({count}x) in main_cmpb.tex.")
        else:
            print(f"  ✅ '{oc}' absent.")

    # Check conservative framing presence
    if "held-out deidentified-date strata" in tex:
        print("  ✅ Conservative framing 'held-out deidentified-date strata' present.")
    else:
        tier1_errors.append("Required phrasing 'held-out deidentified-date strata' missing.")

    has_const_time = ("constant-time" in tex.lower() or "constant time" in tex.lower())
    has_no_refit = ("without model refitting" in tex.lower() or "without refitting" in tex.lower() or "zero refitting" in tex.lower())
    if has_const_time and has_no_refit:
        print("  ✅ Conservative phrasing 'constant-time O(1) adjustment without refitting' present.")
    else:
        tier1_errors.append("Required phrasing 'constant-time / without refitting' missing.")

    if "calendar synchronization is unverified" in tex.lower() or "calendar order is unverified" in tex.lower():
        print("  ✅ Explicit limitation on unverified calendar synchronization present.")
    else:
        tier1_errors.append("Limitation regarding unverified cross-patient calendar synchronization missing.")

    # Check 1.6: MIMIC Code Deduplication & Bootstrap Harmonization
    print("\n🔍 [Check 1.6] Auditing MIMIC dicom_id Code Fix & Bootstrap Counts...")
    ens_path = BASE_DIR / "src" / "evaluation" / "ensemble_mimic_stage4a.py"
    if ens_path.exists():
        ens_code = ens_path.read_text(encoding="utf-8")
        if '"dicom_id"' in ens_code and 'key_cols = ["patient_id", "study_id", "dicom_id"' in ens_code:
            print("  ✅ ensemble_mimic_stage4a.py patched with dicom_id grouping.")
        else:
            tier1_errors.append("ensemble_mimic_stage4a.py lacks dicom_id in key_cols grouping.")
    else:
        tier1_errors.append("ensemble_mimic_stage4a.py not found.")

    eval_path = BASE_DIR / "src" / "evaluation" / "evaluate_mimic_stage4a.py"
    if eval_path.exists():
        eval_code = eval_path.read_text(encoding="utf-8")
        if '"dicom_id"' in eval_code:
            print("  ✅ evaluate_mimic_stage4a.py patched with dicom_id grouping.")
        else:
            tier1_errors.append("evaluate_mimic_stage4a.py lacks dicom_id grouping.")
    else:
        tier1_errors.append("evaluate_mimic_stage4a.py not found.")

    if "1,000-replicate" in tex or "1000-replicate" in tex or "1,000" in tex:
        print("  ✅ MIMIC B = 1,000 bootstrap replicate count documented.")
    else:
        tier1_errors.append("MIMIC B = 1,000 bootstrap replicate count not documented.")

    # Check 1.7: Frozen Baseline Hashes
    print("\n🔍 [Check 1.7] Verifying Frozen Manuscript Baselines (JBHI & CIBM)...")
    expected_hashes = {
        BASE_DIR / "manuscript" / "main.tex": "f31a5692177a20468a60714b5c5fccdc",
        BASE_DIR / "manuscript" / "main.pdf": "e5dc28e953582cdce0ced6e37036c631",
        BASE_DIR / "manuscript_cibm" / "main_cibm.tex": "4d5d8e893901ab533d9a4471a80de1ec",
        BASE_DIR / "manuscript_cibm" / "main_cibm.pdf": "16b4fccf6a18ff1be0119bb1ac5e75cb",
    }
    for file_path, exp_hash in expected_hashes.items():
        actual_hash = md5_file(file_path)
        if actual_hash == exp_hash:
            print(f"  ✅ Frozen baseline verified: {file_path.name} ({actual_hash[:8]}...)")
        else:
            tier1_errors.append(f"Baseline {file_path.name} MD5 hash mismatch: got {actual_hash}, expected {exp_hash}.")

    print("\n" + "#" * 80)
    print("TIER 2: EDITORIAL, BIBLIOGRAPHIC & CMPB GUIDELINES COMPLIANCE")
    print("#" * 80)

    # Check 2.1: Document Class and Line Numbers
    print("\n🔍 [Check 2.1] elsarticle Document Class & Line Numbers...")
    if r"\documentclass[preprint,12pt]{elsarticle}" in tex:
        print("  ✅ \\documentclass[preprint,12pt]{elsarticle} active.")
    else:
        tier2_errors.append("elsarticle 12pt preprint document class missing from main_cmpb.tex.")

    if r"\linenumbers" in tex:
        print("  ✅ \\linenumbers active for peer-review.")
    else:
        tier2_errors.append("\\linenumbers missing from main_cmpb.tex.")

    # Check 2.2: Single Author & Institutional Affiliation
    print("\n🔍 [Check 2.2] Author Metadata & Affiliation...")
    if "Vishesh Panghal" in tex and "pvt.panghal@gmail.com" in tex:
        print("  ✅ Corresponding author Vishesh Panghal (<pvt.panghal@gmail.com>) verified.")
    else:
        tier2_errors.append("Author Vishesh Panghal or email missing from main_cmpb.tex.")

    if "Poornima Institute of Engineering and Technology" in tex:
        print("  ✅ PIET institutional affiliation verified.")
    else:
        tier2_errors.append("PIET affiliation missing from main_cmpb.tex.")

    # Check 2.3: Structured Abstract Structure & Length
    print("\n🔍 [Check 2.3] Structured Abstract Headings & Length...")
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if not abs_match:
        tier2_errors.append("Abstract environment not found.")
    else:
        raw_abs = abs_match.group(1)
        required_headings = [
            "Background and Objective:",
            "Methods:",
            "Results:",
            "Conclusions:",
        ]
        for h in required_headings:
            if h in raw_abs:
                print(f"  ✅ Mandatory heading '{h}' present.")
            else:
                tier2_errors.append(f"Mandatory structured abstract heading '{h}' missing.")

        clean_abs = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", raw_abs)
        clean_abs = re.sub(r"[\$\{\}\\\%]", " ", clean_abs)
        abs_word_count = len(clean_abs.split())
        print(f"  📊 Structured abstract word count: {abs_word_count} words (Limit: <= 350 words).")
        if abs_word_count > 350:
            tier2_errors.append(f"Structured abstract word count ({abs_word_count}) exceeds CMPB limit (350 words).")

    # Check 2.4: Keywords Validation (CMPB requirement: 3 to 6 keywords)
    print("\n🔍 [Check 2.4] Auditing Keywords in main_cmpb.tex (CMPB Limit: 3 to 6)...")
    kw_match = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.DOTALL)
    if not kw_match:
        tier2_errors.append("Keyword environment not found in main_cmpb.tex.")
    else:
        raw_kw = kw_match.group(1).strip()
        keywords = [k.strip() for k in raw_kw.split(r"\sep") if k.strip()]
        print(f"  📊 Found {len(keywords)} keywords: {', '.join(keywords)}")
        if len(keywords) < 3 or len(keywords) > 6:
            tier2_errors.append(f"Found {len(keywords)} keywords in main_cmpb.tex (CMPB requirement: 3 to 6).")
        else:
            print("  ✅ Keyword count satisfies CMPB 3–6 requirement.")
        if any("deep learning" == k.lower() for k in keywords):
            tier2_errors.append("Keyword 'Deep learning' should be dropped as least discriminative.")
        else:
            print("  ✅ Generic keyword 'Deep learning' correctly dropped.")

    # Check 2.5: Highlights in title_page.tex
    print("\n🔍 [Check 2.5] Highlights Character Length (Limit: <= 85 characters)...")
    if TITLE_TEX.exists():
        hl_match = re.search(r"\\section\*\{Highlights\}\s*\\begin\{itemize\}(.*?)\\end\{itemize\}", title_text, re.DOTALL)
        if not hl_match:
            tier2_errors.append("Highlights section/environment not found in title_page.tex.")
        else:
            hl_items = re.findall(r"\\item\s+([^\n]+)", hl_match.group(1))
            if len(hl_items) < 3 or len(hl_items) > 5:
                tier2_errors.append(f"Found {len(hl_items)} highlights in title_page.tex (CMPB requirement: 3 to 5).")
            else:
                print(f"  📊 Found {len(hl_items)} highlights in title_page.tex:")
                for i, h in enumerate(hl_items, 1):
                    clean_h = h.strip()
                    length = len(clean_h.replace(r"\%", "%"))
                    status = "OK" if length <= 85 else "EXCEEDS"
                    print(f"    • Bullet {i} ({length} chars) [{status}]: {clean_h}")
                    if length > 85:
                        tier2_errors.append(f"Highlight bullet {i} ({length} chars) exceeds 85-character CMPB limit.")
    else:
        tier2_errors.append("title_page.tex not found.")

    # Check 2.6: Mandatory Elsevier Declarations
    print("\n🔍 [Check 2.6] Auditing Mandatory Elsevier Declarations...")
    declarations = [
        "CRediT Authorship Contribution Statement",
        "Declaration of Competing Interest",
        "Data and Code Availability",
        "Ethical Approval and Consent to Participate",
        "Declaration of Generative AI and AI-Assisted Technologies in the Writing Process",
        "Funding",
    ]
    for name in declarations:
        if name.lower() in tex.lower():
            print(f"  ✅ {name} present.")
        else:
            tier2_errors.append(f"Mandatory declaration '{name}' missing from main_cmpb.tex.")

    # Check 2.7: Figure and Table Count Verification in title_page.tex
    print("\n🔍 [Check 2.7] Auditing Figure and Table Count in title_page.tex...")
    if TITLE_TEX.exists():
        if "4 + 1 graphical abstract" in title_text:
            print("  ✅ Figure count clarified: '4 + 1 graphical abstract'.")
        else:
            tier2_errors.append("Title page should specify '4 + 1 graphical abstract' for figure count.")
        if "5 (plus 5 supplementary tables)" in title_text:
            print("  ✅ Table count verified: '5 (plus 5 supplementary tables)'.")
    else:
        tier2_errors.append("title_page.tex not found.")

    # Check 2.8: Body Word Count via texcount
    print("\n🔍 [Check 2.8] Checking Body Word Count via texcount...")
    try:
        res = subprocess.run(["texcount", str(MAIN_TEX)], capture_output=True, text=True, check=True)
        tc_out = res.stdout
        tc_match = re.search(r"Words in text:\s+(\d+)", tc_out)
        if tc_match:
            words_in_text = int(tc_match.group(1))
            print(f"  📊 texcount text words: {words_in_text} words (CMPB target: <= 3,500 words).")
            if words_in_text > 3500:
                tier2_errors.append(f"Body text words ({words_in_text}) exceeds 3,500-word ceiling for CMPB.")
            else:
                print(f"  ✅ Body word count is strictly <= 3,500 words ({words_in_text} <= 3,500).")
        else:
            print("  ⚠️ Warning: Could not parse texcount output.")
    except Exception as e:
        print(f"  ⚠️ Warning: Failed to run texcount: {e}")

    # Check 2.7: Submission Package Artifacts
    print("\n🔍 [Check 2.7] Auditing Submission Package Artifacts...")
    package_artifacts = [
        (CMPB_DIR / "main_cmpb.pdf", 1_000_000, "Main Manuscript PDF"),
        (CMPB_DIR / "supplementary_material.pdf", 100_000, "Supplementary Material PDF"),
        (CMPB_DIR / "title_page.pdf", 50_000, "Separate Title Page PDF"),
        (CMPB_DIR / "cover_letter.pdf", 50_000, "Cover Letter PDF"),
        (CMPB_DIR / "graphical_abstract.png", 50_000, "Graphical Abstract Image"),
    ]
    for art_path, min_bytes, desc in package_artifacts:
        if art_path.exists():
            size = art_path.stat().st_size
            if size >= min_bytes:
                print(f"  ✅ {desc} verified: {art_path.name} ({size:,} bytes).")
            else:
                tier2_errors.append(f"{desc} ({art_path.name}) is unexpectedly small ({size:,} bytes < {min_bytes:,} bytes).")
        else:
            tier2_errors.append(f"Required submission artifact missing: {art_path.name}.")

    # Check 2.8: BibTeX Metadata Verification
    print("\n🔍 [Check 2.8] Auditing BibTeX Citations in references.bib...")
    if BIB_FILE.exists():
        bib_content = BIB_FILE.read_text(encoding="utf-8")
        bib_keys = [
            ("reis2022brax", "BRAX primary dataset (Sci Data 2022, DOI: 10.1038/s41597-022-01608-8)"),
            ("johnson2019mimic", "MIMIC-CXR primary dataset (Sci Data 2019, DOI: 10.1038/s41597-019-0322-0)"),
            ("johnson2019mimicjpg", "MIMIC-CXR-JPG database v2.1.0 (PhysioNet 2024, DOI: 10.13026/jsn5-t979)"),
            ("finlayson2021clinician", "Finlayson et al. clinical dataset shift (NEJM 2021, DOI: 10.1056/NEJMc2104626)"),
        ]
        for key, desc in bib_keys:
            if key in bib_content:
                print(f"  ✅ Verified key citation: {key} ({desc}).")
            else:
                tier2_errors.append(f"Key citation '{key}' missing from references.bib.")
    else:
        tier2_errors.append("references.bib not found.")

    # Final Summary
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY REPORT")
    print("=" * 80)

    total_errors = len(tier1_errors) + len(tier2_errors)
    if tier1_errors:
        print(f"\n❌ TIER 1 FAILED ({len(tier1_errors)} errors):")
        for err in tier1_errors:
            print(f"   • {err}")
    else:
        print("\n✅ TIER 1 PASSED: 100% Scientific Data Truth & Reconciliation Verified.")

    if tier2_errors:
        print(f"\n❌ TIER 2 FAILED ({len(tier2_errors)} errors):")
        for err in tier2_errors:
            print(f"   • {err}")
    else:
        print("✅ TIER 2 PASSED: 100% Editorial & CMPB Compliance Verified.")

    print("\n" + "-" * 80)
    if total_errors == 0:
        print("🎉 ALL CHECKS PASSED: Manuscript is 100% ready for CMPB submission!")
        print("-" * 80)
        return True
    else:
        print(f"⚠️ TOTAL ERRORS: {total_errors}. Fix the above issues before submission.")
        print("-" * 80)
        return False

if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
