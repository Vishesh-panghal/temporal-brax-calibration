# Quantifying Temporal Calibration Decay in Chest X-ray Classification Models Using BRAX
## Visual Roadmap for Building the Paper

### Purpose
Use this document as a visual build map for the BRAX temporal decay paper. Start with the flowcharts, then convert each visual into the corresponding manuscript section after the experiments produce real results.

---

### Locked Parameters
- **Dataset**: BRAX v1.1.0
- **Models**: DenseNet-121 and ResNet-50
- **Primary target**: Pleural Effusion
- **Secondary targets**: Cardiomegaly, Pneumonia, Edema
- **Core objective**: Temporal robustness, calibration, and reliability under broken deployment assumptions
- **Working question**: *How can we quantify and predict temporal degradation in medical vision models over a multi-year deployment horizon?*

---

### 1. Master Research Pipeline
Follow this flow when building the codebase and Methods section:
1. Parse `master_spreadsheet.csv` into a deterministic cohort manifest.
2. Filter the image cohort and define target labels before any split.
3. Build the patient-safe chronological split.
4. Train DenseNet-121 and ResNet-50 only on the anchor block.
5. Freeze model weights and evaluate future bins without retraining.
6. Estimate temporal decay and convert it into paper tables, figures, and policy rules.

---

### 2. Patient-Safe Temporal Split and Leakage Control
Use this figure to write the dataset split subsection:
- Sort patients by their first `StudyDate`.
- Keep every image and study from the same patient inside one split.
- Treat the 60/20/20 ratio as approximate by image count because patient safety is more important than exact row proportions.
- Divide the locked test block into ordered future bins for decay analysis.

---

### 3. Metric and Temporal Decay Logic
Use this figure to write the evaluation and formalization sections:
- **Calibration metrics**: Brier score, log loss, ECE, calibration slope, calibration intercept.
- **Discrimination metrics**: AUROC and AUPRC, interpreted with prevalence.
- **Temporal change**: $\Delta S(t) = S(M; D_t) - S(M; D_{\text{anchor}})$.
- **Temporal Decay Index**: Fit $S(D_t) = \alpha + \beta \cdot \text{time}_t + \text{error}$.
- Use $\text{TDI} = \beta$ for loss-like metrics and $\text{TDI} = -\beta$ for benefit-like metrics, so positive TDI always means degradation.
- Fit TDI slopes with patient-level bootstrap intervals.

---

### 4. Paper-Writing Blueprint
Use this figure to assemble the manuscript after experiments are complete:
- **Introduction**: Why random splits miss deployment decay.
- **Methods**: BRAX cohort, patient-safe split, model training, frozen future evaluation.
- **Results**: Cohort table, metric trajectories, reliability diagrams, TDI estimates.
- **Discussion**: What decayed, why it matters clinically, recalibration or abstention policy, limitations.

---

### Quick Build Checklist
- [x] Get PhysioNet access and download BRAX v1.1.0.
- [x] Create the cohort manifest from `master_spreadsheet.csv`.
- [x] Verify patient leakage is zero across train, validation, and locked test.
- [ ] Generate split summary table before model training.
- [ ] Train DenseNet-121 and ResNet-50 on the anchor block only.
- [ ] Save frozen predictions for every temporal bin.
- [ ] Compute calibration and discrimination metrics from saved predictions.
- [ ] Fit Temporal Decay Index slopes with patient-level bootstrap intervals.
- [ ] Replace this roadmap with real result tables and final figures when experiments finish.
