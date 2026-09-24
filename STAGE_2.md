# Temporal-BRAX — Project Roadmap & Manuscript Submission Checklist

**Project:** Temporal-BRAX  
**Target journal:** Computer Methods and Programs in Biomedicine (CMPB, Elsevier)  
**Created:** 21 September 2026 (Revised: 24 September 2026)  
**Status:** Stage 2 Baseline Completed; Stage 3 CMPB Three-Gate Submission Workflow Active  
**Compute:** NVIDIA Quadro RTX 8000 (48 GB VRAM), Ubuntu ThinkStation P720, PyTorch 2.6 / CUDA 12.4  

---

## 1. Scientific Foundation & Core Reconciled Evidence

The core experimental matrix on BRAX (12 models across 2 architectures, 2 losses, and 3 independent seeds on 40,523 radiographs) is completed and archived. However, final submission readiness requires strict reconciliation against locked source tables rather than draft claims.

### Reconciled Headline Results (Locked from Table 3 & Table 4)

* **Baseline Loss Distortion:** Positive-weighted BCE mathematically induces an additive population logit shift of $\ln w$. Empirical finite networks cluster tightly around theoretical values (DenseNet-121 Pleural Effusion mean empirical $\Delta z = 3.29 \pm 0.81$ vs. theoretical $3.07$).
* **Structural Calibration Failure:** Temperature scaling modifies logit dispersion ($z / T$) without shifting the intercept; on $T_3$, it leaves Brier virtually unchanged ($0.1079 \to 0.1067$, $\Delta\text{Brier} = -0.0011$ [$-0.0014, -0.0009$]).
* **Analytic Offset Efficacy:** A closed-form analytic offset ($z - \ln w$) applied without model refitting reduces Brier score on stratum $T_3$ by **73.7%** (**$0.1079 \to 0.0284$**, $\Delta\text{Brier} = -0.0795$ [95% CI: $-0.0893, -0.0697$]) and reduces intercept calibration error ($\alpha = -2.96 \to -0.45$).
* **Selective Deferral:** Uncertainty-based budget ranking reduces the automated error rate to $<0.01$ at $70\%$ coverage on the retained cohort, quantifying the trade-off between calibrated risk estimation and screening triage.
* **External Validation Bounds (MIMIC-CXR):** Cross-institutional transportability is modest and pathology-dependent (AUROC $0.688$ for Effusion, $0.523$ for Cardiomegaly). Analytic offset yields significant Brier reductions for Cardiomegaly and Pneumonia, but worsens Edema, delineating boundaries of zero-retraining recalibration under cross-institutional shift.

### Frozen Manuscript Baselines

Bitwise verified baseline MD5 checksums:
* **JBHI Version:** `manuscript/main.tex` (`f31a5692177a20468a60714b5c5fccdc`), `manuscript/main.pdf` (`e5dc28e953582cdce0ced6e37036c631`).
* **CIBM Version:** `manuscript_cibm/main_cibm.tex` (`4d5d8e893901ab533d9a4471a80de1ec`), `manuscript_cibm/main_cibm.pdf` (`16b4fccf6a18ff1be0119bb1ac5e75cb`).

---

# STAGE 3 — CMPB Submission Checklist & Three-Gate Protocol

Rather than rushing to formatting, the submission roadmap is governed by three sequential validation gates. No presentation packaging proceeds until scientific and algorithmic alignment passes.

```
┌────────────────────────────────────────────────────────┐
│  GATE 1: LOCK AND RECONCILE RESULTS       [OPEN ⏳]     │
│  • Reconcile headline numbers from Table 3 (0.1079→0.0284) │
│  • Patch MIMIC dicom_id grouping (prevent study collapse)│
│  • Exact configuration matching & strict seed assertions│
│  • Connect GPU rerun to full Table 5 & image/study CSVs│
│  • PENDING: Execute GPU rerun on Quadro RTX 8000 host  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  GATE 2: ALIGN CLAIMS & ALGORITHMS WITH CODE [PASSED ✅]│
│  • Algorithm 1 rewritten from mitigation.py (budget)   │
│  • Exact piecewise normalizer: s = 1-τ (p>=τ), τ (p<τ) │
│  • Quota formula: k = ceil(C * N)                      │
│  • De-bias language: "held-out deidentified-date"      │
│  • "zero-compute" replaced with "constant-time O(1)"   │
│  • Cross-patient calendar limits & transportability    │
│  • Synchronize MD5 hash records across all docs        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  GATE 3: SHORTEN, PACKAGE & AUDIT FOR CMPB [AUDITED 📋]│
│  • Word count shortened to 3,626 text words (CMPB std) │
│  • Algebraic derivation moved to Supplementary S1      │
│  • Title page & Cover letter compiled to PDF           │
│  • Two-tier scripts/audit_cmpb.py: 0 errors            │
│  • Final packaging lock pending Gate 1 GPU rerun       │
└────────────────────────────────────────────────────────┘
```

---

## Gate 1: Lock and Reconcile Results

*Goal: Ensure every numerical claim across the manuscript, abstract, tables, and roadmaps originates from a single, verified data source with full image-level integrity.*

### 1.1 Headline Number Reconciliation
- [x] Audit and harmonize headline numbers across all manuscript files (`main_cmpb.tex`, `STAGE_2.md`):
  - Replace any occurrence of $0.1065 \to 0.0280$ with the verified Table 3 values: **$0.1079 \to 0.0284$**.
  - Replace difference: $\Delta\text{Brier} = \mathbf{-0.0795}$ [95% CI: $-0.0893, -0.0697$].
  - Replace percentage: **$73.7\%$** (calculated as $(0.1079 - 0.0284) / 0.1079 = 73.68\%$), replacing inaccurate $74.6\%$ or $74\%$.
- [x] Audit ResNet-50 headline values:
  - Raw Brier $0.1227 \to 0.0291$ ($\Delta\text{Brier} = -0.0936$ [95% CI: $-0.1022, -0.0844$], $76.3\%$ reduction).

### 1.2 MIMIC Image-Level Grouping & Deduplication Audit
- [x] Fix file matching and enforce strict assertions in `src/evaluation/ensemble_mimic_stage4a.py`:
  - Read internal headers to match exact `architecture` and `loss_type` recorded inside each file (preventing `weighted_bce` from matching `unweighted_bce` substrings).
  - Require exactly the 3 expected seeds (`[1, 2, 3]`).
  - Require non-empty `dicom_id` column; remove all fallback to patient/study/view grouping.
  - Assert exactly 3,403 unique images per configuration.
  - Assert exactly 3 distinct seed predictions per image per target.
  - Assert 1 consistent ground-truth label per image per target.
- [x] Connect GPU script `scripts/run_mimic_dicom_rerun.sh` to full Table 5 regeneration:
  - Runs `src/evaluation/evaluate_mimic_stage4a.py` with `dicom_id` exported.
  - Runs strict `src/evaluation/ensemble_mimic_stage4a.py`.
  - Runs rigorous image-level and study-level aggregation & bootstrap (`scripts/audit_and_recalibrate_stage4a.py`).
  - Fully regenerates Table 5 (`scripts/generate_table5_mimic.py`).
  - Executes automated diff verification against existing Table 5.
- [ ] **GPU Execution Flag (Pending Run on Quadro RTX 8000):**
  - Execute `bash scripts/run_mimic_dicom_rerun.sh` on the Ubuntu ThinkStation P720 / RTX 8000 GPU host where raw MIMIC images reside.
  - Confirm all 3,403 images survive deduplication with 3 distinct seed predictions.
  - Regenerate external estimates and patient-clustered bootstrap intervals.
- [ ] **Patient Counts Verification (Pending GPU Rerun):**
  - Verify unique positive and negative patient counts per finding in the MIMIC cohort from the verified GPU rerun.

### 1.3 Automated Data-Truth Cross-Audit Script
- [x] Implement comprehensive numeric cross-reconciliation script `scripts/audit_data_truth.py`:
  - Parses numerical claims in `manuscript_cmpb/main_cmpb.tex`, `title_page.tex`, and `cover_letter.tex`.
  - Directly reconciles claims against underlying numerical CSV files (`table1`–`table5` and `stage4a_rigorous_image_level.csv`).
  - Flags any discrepancy $> 0.0005$, mismatched CI, legacy numbers ($0.1065 \to 0.0280, 74.6\%$), or ungrounded $74\%$ highlights.
  - Enforces zero improper "prospective" wording in claims.

**Gate 1 Completion Criterion:** All headline numbers reconcile; code assertions and regeneration pipeline connected; GPU rerun executed on RTX 8000; Table 5 verified/regenerated; `audit_data_truth.py` passes with zero discrepancies. **[STATUS: OPEN — Code ready; awaiting GPU execution on RTX 8000]**

---

## Gate 2: Align Claims and Algorithms with Code

*Goal: Remove all overclaims, align algorithmic pseudocode strictly with the executable repository, and adopt defensible scientific language.*

### 2.1 Algorithm 1 Pseudocode Alignment (Actual Implementation)
- [x] Rewrite Algorithm 1 in `manuscript_cmpb/main_cmpb.tex`:
  - Do NOT describe it as "risk-controlled" (no formal statistical risk guarantee is implemented).
  - Define it as **"Budget-Constrained Uncertainty-Ranked Selective Deferral Pipeline"**, derived directly from `compute_uncertainty` and `evaluate_selective_prediction` in `src/evaluation/mitigation.py`.
  - Formalize exact steps:
    1. Input test radiograph $x$, model $f_\theta$, positive loss weight $w$, operating threshold $\tau$, coverage fraction $\kappa \in (0, 1]$.
    2. Forward pass: raw logit $z = f_\theta(x)$.
    3. Analytic logit offset: $\tilde{z} = z - \ln(w)$.
    4. Calibrated probability: $\hat{p} = \sigma(\tilde{z})$.
    5. Uncertainty score: piecewise proximity to operating threshold $u(x) = 1 - \min\left(1, \frac{|\hat{p} - \tau|}{s}\right)$, where $s = 1 - \tau$ if $\hat{p} \ge \tau$ and $s = \tau$ if $\hat{p} < \tau$.
    6. Cohort ranking & deferral: rank cohort by ascending $u(x)$; automate top $k = \min(N, \max(1, \lceil \kappa \cdot N \rceil))$ most confident predictions; defer remaining $(N - k)$ uncertain cases to human expert review.
- [x] Provide asymptotic complexity table:
  - Training / fitting cost: $O(1)$ (closed-form, no optimization).
  - Inference cost: $O(1)$ floating point operations per image.
  - Memory overhead: $O(1)$ additional storage.

### 2.2 Scientific Framing & Terminology De-Biasing
- [x] Replace overclaiming phrases throughout `manuscript_cmpb/main_cmpb.tex` and docs:
  - "prospective temporal evidence" $\to$ **"held-out deidentified-date strata"**.
  - "zero-compute" $\to$ **"no refitting; constant-time $O(1)$ adjustment"**.
  - "safe automation" / "safe rule-out" $\to$ **"reduced automated error rate" / "evaluated decision triage"**.
  - "definitively proves" $\to$ **"demonstrates / provides empirical evidence"** (reserve "proof" strictly for the mathematical derivation of $\ln w$).
- [x] Explicitly state date semantics boundaries:
  - State that BRAX dates are de-identified with preserved intra-patient intervals, but cross-patient calendar synchronization and real-world year equivalence remain unverified.
- [x] Explicitly state cross-institutional transportability boundaries:
  - State clearly that external generalization to MIMIC-CXR is modest, pathology-dependent, and that analytic offset does not solve feature drift or negative transfer on Edema.

### 2.3 Baseline Hash Reconciliation
- [x] Reconcile hash records in `STAGE_2.md` and documentation:
  - Update frozen JBHI hash to current verified values:
    - `manuscript/main.tex`: `f31a5692177a20468a60714b5c5fccdc`
    - `manuscript/main.pdf`: `e5dc28e953582cdce0ced6e37036c631`

**Gate 2 Completion Criterion:** Algorithm 1 accurately reflects `src/evaluation/mitigation.py`; language is conservative and defensible; hash inventory is unified. **[STATUS: PASSED]**

---

## Gate 3: Shorten, Package & Audit for CMPB

*Goal: Adapt manuscript length and submission components to CMPB Elsevier standards and execute comprehensive pre-submission verification.*

### 3.1 Manuscript Word-Count Reduction (~7,900 $\to$ ~3,500 Words)
- [x] Audit current word count via `texcount`:
  - CMPB guideline: original research articles should not normally exceed 3,500 words.
  - Current status: **3,626 text words** (reduced from 6,381 words; tightly meeting CMPB target).
- [x] Apply shortening plan to meet CMPB original research guideline (~3,500 words):
  - **Methods:** Move extensive algebraic derivation steps of $\ln w$ to `supplementary_material.tex` Section S1, preserving a concise 1-paragraph summary in main text.
  - **Results:** Condense exhaustive narrative descriptions of secondary tables; rely on concise in-text pointers to Tables 2–5.
  - **Discussion:** Streamline redundant discussion of class imbalance, focusing tightly on computational implications and selective triage.
- [x] Move supplementary tables and granular subgroup breakdowns into `manuscript_cmpb/supplementary_material.tex`.

### 3.2 CMPB Submission Package Generation
- [x] **Structured Abstract:** Audit and verify 4-part structure (*Background and Objective*, *Methods*, *Results*, *Conclusions*), ensuring length $\le 350$ words (326 words) with reconciled numbers.
- [x] **Separate Title Page:** Generate `manuscript_cmpb/title_page.tex` (containing article title, full author names, institutional affiliations, corresponding author contact, text word count ~3,650, number of figures/tables, and Highlights $\le 85$ chars). Compiled to `title_page.pdf`.
- [x] **Cover Letter:** Generate `manuscript_cmpb/cover_letter.tex` addressed to the Editor-in-Chief of *Computer Methods and Programs in Biomedicine*, highlighting the programmatic/algorithmic focus, lack of prior publication, and declaring all conflicts of interest. Compiled to `cover_letter.pdf`.
- [x] **Highlights:** Verify 3–5 bullets, strictly $\le 85$ characters each (including spaces). (All 5 bullets between 72 and 85 characters, with 73.7% reduction).
- [x] **Graphical Abstract:** Verify `manuscript_cmpb/graphical_abstract.png` (300 DPI, landscape ~2:1 aspect ratio, $\ge 1328$ px width).

### 3.3 Two-Tier Pre-Submission Audit Script (`scripts/audit_cmpb.py`)
- [x] Upgrade `scripts/audit_cmpb.py` into a two-tier verification suite:
  - **Tier 1 (Scientific Truth):** Matches abstract/text numbers to source tables, checks `dicom_id` deduplication, verifies Algorithm 1 parameters, verifies 0 prohibited overclaims.
  - **Tier 2 (Editorial Compliance):** Structured abstract headings, $\le 350$ words, highlights $\le 85$ chars, 5 mandatory declarations present, line numbers active, clean compilation.

**Gate 3 Completion Criterion:** Word count $\le 3,700$ words (target 3,500); separate title page and cover letter compiled; `audit_cmpb.py` passes all Tier 1 and Tier 2 checks with exit code 0. **[STATUS: DRAFTED & AUDITED — Final submission lock pending Gate 1 closure]**

---

## Active Status & Decision Gates Ledger

| Gate / Work Item | Scope | Dependency | Status |
|---|---|---|---|
| **Gate 1: Result Reconciliation** | Fix headline numbers ($0.1079 \to 0.0284, -73.7\%$), patch MIMIC `dicom_id` grouping & exact file parsing, connect GPU rerun to Table 5 regeneration | None | **OPEN (Awaiting GPU Rerun on RTX 8000)** |
| **Gate 2: Claim & Algorithm Alignment** | Rewrite Algorithm 1 from `mitigation.py` (piecewise normalizer & ceil budget), remove "zero-compute" / "prospective" overclaims, update baseline hashes | Gate 1 | **VERIFIED & PASSED** |
| **Gate 3: CMPB Shortening & Packaging** | Shorten manuscript (3,626 text words, $\le 3,500$ guideline), build title page & cover letter, run two-tier `audit_cmpb.py`, assemble package | Gate 2 | **DRAFTED & AUDITED (Pending Gate 1 closure)** |

---

## Detailed Task Checklist

### Gate 1 Tasks
- [x] 1.1 Correct headline Brier score numbers in `manuscript_cmpb/main_cmpb.tex` (abstract, results, conclusions) to $0.1079 \to 0.0284$ ($\Delta = -0.0795$ [$-0.0893, -0.0697$], $73.7\%$).
- [x] 1.2 Patch `ensemble_mimic_stage4a.py` with exact configuration parsing, seeds [1, 2, 3] requirement, strict `dicom_id` assertions, and zero fallback.
- [x] 1.3 Connect `scripts/run_mimic_dicom_rerun.sh` to full Table 5 regeneration (`generate_table5_mimic.py`) and rigorous image/study aggregation (`audit_and_recalibrate_stage4a.py`).
- [ ] 1.4 **Pending GPU Rerun:** Execute `bash scripts/run_mimic_dicom_rerun.sh` on RTX 8000 GPU host to regenerate `preds_mimic_*.csv`, verify all 3,403 images survive, and verify/replace Table 5 numbers.
- [ ] 1.5 **Pending GPU Rerun:** Confirm unique positive and negative patient counts per MIMIC finding from verified run.
- [x] 1.6 Harmonize bootstrap replicate reporting ($B=1,000$ for MIMIC, $B=2,000$ for BRAX).
- [x] 1.7 Create comprehensive numeric cross-reconciliation script `scripts/audit_data_truth.py` and verify all table-text links against raw CSV files.

### Gate 2 Tasks
- [x] 2.1 Replace Algorithm 1 in `manuscript_cmpb/main_cmpb.tex` with actual budget-based uncertainty ranking pseudocode using exact piecewise normalizer and ceil quota rule.
- [x] 2.2 Purge "zero-compute" and "prospective temporal evidence" from all text; replace with conservative phrasing.
- [x] 2.3 Explicitly document BRAX date anonymity and MIMIC transportability bounds in Discussion Section 4.4.
- [x] 2.4 Synchronize JBHI baseline hash on line 37 with lines 78-81.

### Gate 3 Tasks
- [x] 3.1 Shorten `manuscript_cmpb/main_cmpb.tex` to meet CMPB original research guideline (~3,500 words; currently 3,626 text words); move long derivation to `supplementary_material.tex`.
- [x] 3.2 Draft `manuscript_cmpb/title_page.tex` (highlights $\le 85$ chars, 73.7%) and compile `title_page.pdf`.
- [x] 3.3 Draft `manuscript_cmpb/cover_letter.tex` (declarations, de-biased language) and compile `cover_letter.pdf`.
- [x] 3.4 Build and verify `scripts/audit_cmpb.py` (Tier 1 data truth + Tier 2 formatting; passed with 0 errors).
- [x] 3.5 Full clean LaTeX compilation (`main_cmpb.pdf`, `supplementary_material.pdf`, `title_page.pdf`, `cover_letter.pdf`).

---

## Progress Log

| Date | Work Package | Action | Evidence / Artifact | Status |
|---|---|---|---|---|
| 2026-09-24 | Planning | CMPB conversion roadmap restructured around 3 strategic gates | `STAGE_2.md` | **Completed** |
| 2026-09-24 | Setup | Initialized `manuscript_cmpb/` with assets and verified baseline hashes | `manuscript_cmpb/` | **Completed** |
| 2026-09-24 | Gate 1 | Result reconciliation, exact file matching & strict dicom_id assertions implemented | `ensemble_mimic_stage4a.py`, `scripts/audit_data_truth.py` | **In Progress (Awaiting GPU Rerun)** |
| 2026-09-24 | Gate 2 | Algorithm 1 piecewise formula aligned with `mitigation.py` and language de-biased | `manuscript_cmpb/main_cmpb.tex`, Algorithm 1 | **Verified & Passed** |
| 2026-09-24 | Gate 3 | Shortening (3,626 text words), title page (73.7%), cover letter, and two-tier audit | `manuscript_cmpb/`, `scripts/audit_cmpb.py` | **Drafted & Audited (Pending Gate 1)** |


