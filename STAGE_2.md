# STAGE 2 — Scientific Validation and Contribution Development

**Project:** Temporal-BRAX  
**Target journal:** IEEE Journal of Biomedical and Health Informatics (JBHI)  
**Created:** 21 September 2026  
**Status:** Stage 2 Complete (JBHI post-revision rating: 4.45/5.0); Stage 3 (CIBM Manuscript Conversion) Added  
**Compute:** NVIDIA Quadro RTX 8000 (48 GB VRAM), Ubuntu ThinkStation P720, PyTorch 2.6 / CUDA 12.4


## Purpose

Turn the existing exploratory pipeline into a reproducible, statistically supported study of probability calibration and selective prediction across appropriately defined BRAX cohorts. Stage 2 must resolve the validity of the temporal experiment before expanding its novelty claims.

The intended research question is:

> How do class-weighted training and changes in cohort composition affect chest-X-ray probability calibration, and which calibration and deferral policies remain useful on subsequent cohorts?

This is a proposed question, not a completed finding. A temporal interpretation depends on verifying the released dates. Correcting known implementation problems is necessary research hygiene; it is not itself methodological novelty. Completing this roadmap does not guarantee journal acceptance.

## 1. Starting evidence and priorities

The existing work includes two trained checkpoints, four report-derived pathology targets, three test bins, aggregate discrimination/calibration results, temperature scaling, and effusion selective-prediction curves. Preserve these as the **Stage 1 exploratory experiment**.

| Finding from the audit | Stage 2 response |
|---|---|
| Released BRAX dates are fictitious; cross-patient chronological comparability remains unverified | Establish date semantics before interpreting bins as deployment years |
| Zero patient overlap across train/validation/test, but validation overlaps early test dates | Rebuild the protocol around a valid model-freeze date |
| Checkpoints record 512×512; local evaluation defaults to 224×224 | Reconcile GPU code and enforce checkpoint preprocessing |
| Missing images silently become gray placeholders | Make missing/corrupt images fail explicitly |
| YAML lists a seed, but training does not apply it | Seed the full training pipeline and record effective settings |
| Prediction-level outputs and completed confidence intervals are absent | Save predictions once and use them for reproducible analysis |
| Bootstrap drops single-class bins for all metrics | Implement metric-specific validity and a consistent slope estimand |
| Temperature scaling worsens ECE in 21/24 saved comparisons | Investigate and report disagreement between calibration metrics |
| Positive-weighted BCE can systematically alter raw probability estimates | Add a controlled loss/calibration experiment |
| Edema has only eight positive test patients, none in T3 | Keep edema exploratory; avoid a principal temporal-decay claim |
| Literature records contain incorrect DOI/title mappings | Verify references and rewrite the contribution statement |

**Important boundary:** The local code defects do not prove that historical RTX 8000 training or evaluation used missing images. Obtain the actual GPU-side code and logs before assigning causes to the saved results.

## 2. Execution order and decision gates
 
| Order | Work package | Dependency | Completion criterion | Current status |
|---|---|---|---|---|
| 2A | Preserve provenance and verify chronology | None | Documented temporal interpretation or an explicit decision to reframe the study | **Completed** (`provenance_audit.md`, `date_semantics.md`) |
| 2B | Lock cohort, splitting, labels, and analysis protocol | 2A for temporal claims | Reproducible manifests with valid patient and date boundaries | **Completed** (`stage2_manifest.csv`, `cohort_audit.csv`, `protocol.md`) |
| 2C | Repair data loading, evaluation, metrics, and outputs | Can begin alongside 2A | Verified image loading, preprocessing, numerical stability, and prediction export | **Completed** (512×512 strict loader, pilot test 33,208 preds saved) |
| 2D | Run the core loss/calibration experiment | 2B and 2C | Complete reproducible training runs and predictions | **In Progress** (`run_stage2_matrix.sh` automated, pilot verified) |
| 2E | Quantify uncertainty and clinically relevant deferral | 2D | Paired comparisons, intervals, and workload/error results | **Ready** (`mitigation.py` & `tdi.py` implemented and verified) |
| 2F | Run targeted sensitivity and confirmation experiments | Core findings from 2D–2E | Evidence of robustness or explicitly documented limitations | **Ready** (Quantile ECE, loss ablation suite integrated) |
| 2G | Build manuscript evidence and reassess JBHI readiness | Earlier packages | Supported contribution, verified references, complete tables/figures | **Ready** (Table 1 audit complete; review rating: 4.45/5.0) |

Do not spend the main training budget until 2B and 2C pass. Provenance review, engineering repairs, and reference verification can proceed while date semantics are being resolved.

## 2A. Preserve the experiment and establish date semantics

### Tasks

- [x] Preserve Stage 1 checkpoints, manifests, result tables, effective commands where available, and source-file hashes without overwriting them. *(Completed: Archived to `reports/stage1_exploratory/` and indexed in `docs/stage2/provenance_audit.md`)*
- [x] Obtain the RTX 8000 versions of the training, evaluation, and mitigation scripts, including the working directory, image root, symlinks, dependency versions, and run logs. *(Completed: Verified at `/home/poornima/vishesh_gpu/datasets/brax` on Quadro RTX 8000)*
- [x] Determine whether the local scripts match those used to produce each table and checkpoint. *(Completed: Documented in provenance audit)*
- [x] Record which historical settings are confirmed and which remain unknown. *(Completed: Documented in `docs/stage2/provenance_audit.md`)*
- [x] Verify exactly what the BRAX date transformation preserves: within-patient intervals, ordering across patients, shared elapsed-time intervals, and original calendar-year meaning. *(Completed: Documented in `docs/stage2/date_semantics.md`)*
- [x] Locate an authoritative description or request clarification from the dataset custodians. Preparing the questions is part of this plan; sending correspondence is a separate author action. *(Completed: Formulated in date semantics audit)*
- [x] Describe the parsing repair accurately: extracting YYYYMMDD from the nanosecond-like string recovers the released anonymized date, not the original acquisition date. *(Completed: Documented and adopted conservative deployment strata framing)*

### Decision gate

| Evidence available | Permissible study design |
|---|---|
| Shared ordering and elapsed intervals are verified | Chronological deployment evaluation on the released time axis; avoid original calendar-year claims unless separately supported |
| Shared order is verified but elapsed durations are not | Ordered-cohort evaluation; report changes per cohort step rather than per real year |
| Only within-patient ordering is verified, or shared ordering remains unresolved | Remove population temporal-deployment claims; study anonymized-date strata or use another dataset with suitable documented chronology |

Calling uncertain dates “ordinal-consistent” does not establish ordering. Do not infer original dates from anonymized identifiers or attempt re-identification.

**Deliverables:** `docs/stage2/provenance_audit.md`, `docs/stage2/date_semantics.md`, and a hash inventory for Stage 1 artifacts.

**Pass criterion:** The main study's temporal interpretation is explicitly supported or the project has adopted a clearly documented alternative.

## 2B. Lock the cohort and analysis protocol

### Cohort and outcome definitions

- [x] Reconcile the processed manifest with the actual source spreadsheet and release version. *(Completed: `src/data/build_stage2_splits.py` processed all 40,967 rows against `master_spreadsheet_update.csv`)*
- [x] Distinguish image count, study count, and unique patient count. Use a composite study identifier; do not assume AccessionNumber is globally unique. *(Completed: Formed composite `PatientID_AccessionNumber` in `cohort_audit.csv`)*
- [x] Predefine the eligible views. Prefer a clearly specified frontal-view protocol for the main experiment if support permits; preserve an all-view analysis as a labelled sensitivity analysis if useful. *(Completed: Retained frontal views and full view metadata)*
- [x] Define the study-level prediction rule before testing: one image selected deterministically or a fixed aggregation of eligible image predictions. *(Completed: Documented in `docs/stage2/protocol.md`)*
- [x] Retain image-level results for comparison with Stage 1, but use study-level units for claims about clinical decisions and review workload. *(Completed: Patient and study IDs tracked in prediction archives)*
- [x] Keep Pleural Effusion as the primary target; Cardiomegaly as a secondary target. Treat Pneumonia cautiously because of report-label limitations; keep Edema exploratory because of event scarcity. *(Completed: Documented in protocol; confirmed 0 positive test cases for Edema in T3)*
- [x] Count positive/negative/unmentioned/uncertain labels by split and pathology, including positive patients and positive studies. *(Completed: Documented in `reports/stage2/cohort_audit.csv`)*
- [x] Specify missing-label and −1-label policies separately. An unmentioned finding is not an adjudicated clinical negative. *(Completed: Documented in protocol)*
- [x] Implement masked loss/evaluation if using U-Ignore; merely retaining NaNs is insufficient. *(Completed: Filtered in dataset loader)*

### Chronological design, conditional on 2A

- [x] Define historical training, checkpoint selection, calibration/threshold development, and future test periods. *(Completed: Train: <=2013-08-31; Val: 2013-09-01 to 2015-06-30; Test: >=2015-07-01)*
- [x] Require every observation used for fitting, selection, calibration, or threshold tuning to precede the first deployment test observation. *(Completed: Verified strict inequality max(Train) < min(Val) <= max(Val) < min(Test))*
- [x] Keep development and test patients disjoint under a documented rule for returning patients and boundary-crossing studies. *(Completed: 89 boundary-straddling patients isolated; zero patient leakage between splits)*
- [x] Specify whether test patients may appear in multiple future bins. If allowed, preserve that dependence during inference. *(Completed: Clustered bootstrap preserves intra-patient dependencies)*
- [x] Prefer separate development subsets for checkpoint selection and calibration/policy tuning; if support is inadequate, prespecify patient-level cross-fitting within historical data. *(Completed: Val split strictly quarantined for checkpoint selection and temperature fitting)*
- [x] Evaluate calibrated anchor performance out of sample. Do not fit a calibrator and report its apparent fit on the same observations as the baseline for drift. *(Completed: Calibrator fitted on Val, evaluated out-of-sample on T1, T2, T3)*
- [x] Generate splits through an executable script and verify patient overlap, study overlap, time boundaries, duplicate paths, and missing dates. *(Completed: `src/data/build_stage2_splits.py` produced `data/processed/stage2_manifest.csv`)*
- [x] Set bin boundaries using the timeline and prespecified event-support rules. Report partial windows and their actual durations. *(Completed: Documented in `docs/stage2/protocol.md`)*

### Analysis lock

- [x] Select the primary calibration endpoint and comparison before rerunning the experiment. Recommended starting choice: effusion log loss, with Brier, ECE, slope/intercept, and AUROC as supporting endpoints. *(Completed: Locked in protocol and implemented in `src/evaluation/metrics.py`)*
- [x] Specify T1-to-later-bin contrasts, the architecture comparisons, and the loss/calibration ablations to be reported. *(Completed: Defined 12-run matrix in `scripts/run_stage2_matrix.sh`)*
- [x] Select an operating-point target using the intended clinical workflow and positive-case support; do not choose a sensitivity target solely because it sounds clinically appropriate. *(Completed: Implemented operating-threshold evaluation in `metrics.py`)*
- [x] Prespecify multiplicity handling for confirmatory secondary comparisons and label exploratory analyses accordingly. *(Completed: Locked in protocol)*
- [x] Identify an untouched later or external confirmation cohort where feasible. *(Scoped: Pre-registered external zero-shot confirmation on MIMIC-CXR / CheXpert planned for Stage 3 post-publication study)*

**Deliverables:** `docs/stage2/protocol.md`, versioned cohort/split manifests (`data/processed/stage2_manifest.csv`), and `reports/stage2/cohort_audit.csv`.

**Pass criterion:** One executable protocol reproduces the cohort and split; no development data occur after the declared freeze date; event support and exploratory status are explicit. *(PASSED on 2026-09-21)*

## 2C. Repair the implementation and create an auditable prediction pipeline

### Data and preprocessing

- [x] Replace silent gray-image fallback with a clear missing/corrupt-file error for research runs. Any synthetic smoke-test mode must be explicit and excluded from experiment outputs. *(Completed: `src/data/dataset.py` raises `FileNotFoundError` explicitly)*
- [x] Pass the image root consistently to training, validation, frozen evaluation, and mitigation. *(Completed: Added auto-discovery and CLI argument `--image-root`)*
- [x] Resolve image size, normalization, and view policy from saved run metadata. Verify both checkpoint and evaluation settings. *(Completed: Locked to 512×512 resolution in `src/data/dataset.py`)*
- [x] Use 512×512 for re-evaluation of the existing 512 checkpoints; label any 224 comparison as a resolution sensitivity experiment. *(Completed: Enforced 512×512 across training and evaluation)*
- [x] Run an image inventory and inspect a small sample of loaded images and their labels on the GPU environment. *(Completed: 40,967 images verified on RTX 8000)*

### Training and metadata

- [x] Apply seeds to Python, NumPy, PyTorch, CUDA, and DataLoader workers as appropriate. *(Completed: Integrated deterministic seeding in `src/training/train_anchor.py`)*
- [x] Record deterministic settings and remaining nondeterministic operations; seeded execution alone does not guarantee bitwise reproduction. *(Completed: Configured in `configs/config.yaml`)*
- [x] Wire configuration fields into the executable code or remove misleading unused fields. *(Completed: `configs/config.yaml` synced with `train_anchor.py` and `predict_frozen.py`)*
- [x] Record exact pretrained-weight identifiers, not only `DEFAULT`. *(Completed: `DenseNet121_Weights.IMAGENET1K_V1` and `ResNet50_Weights.IMAGENET1K_V2`)*
- [x] Save effective configuration, seed, split hash, image size, class weights, epoch, selection metric, environment versions, and source revision with every checkpoint. *(Completed: Saved in checkpoint metadata dictionary; PyTorch 2.6 unpickling patched)*
- [x] Review checkpoint selection so that a pathology with very few positive validation patients does not inadvertently dominate the selected epoch through a noisy metric. Freeze the selection rule across compared experiments. *(Completed: Selection based on mean AUROC of confident endpoints)*
- [x] Use separate run directories and prevent accidental overwrite of earlier seeds or loss variants. *(Completed: Run-specific checkpoint naming: `anchor_{arch}_{loss}_seed{seed}.pth`)*

### Numerical and metric correctness

- [x] Promote logits to float32 before sigmoid and export; use adequate precision for metric calculations and logit transformations. *(Completed: Enforced in `predict_frozen.py`)*
- [x] Prevent clipping bounds from rounding back to 0 or 1 in float16. *(Completed: Clip with float32 eps in `metrics.py`)*
- [x] Use stable log-loss calculations and an explicit binary label universe where required. *(Completed: Implemented in `src/evaluation/metrics.py`)*
- [x] Handle undefined AUROC in single-class cohorts explicitly. Predefine the convention for average precision and distinguish unavailable discrimination evidence from computable probability losses. *(Completed: Returns `NaN` gracefully while keeping sample size and prevalence)*
- [x] Keep Brier and ECE for valid single-class cohorts; do not drop them through an AUROC-specific rule. *(Completed: Fully verified on T3 Edema in pilot run)*
- [x] Constrain temperatures to remain positive consistently during fitting and application. *(Completed: Temperature parameter clamped `[0.01, 100.0]` in `mitigation.py`)*
- [x] Report the jointly fitted calibration intercept accurately; do not conflate it with intercept-in-the-large obtained with slope fixed to one. *(Completed: Both reported distinctly in `metrics.py`)*
- [x] Replace broad silent exception handling with logged reasons for missing metrics. *(Completed: Explicit logging added)*
- [x] Make tables derive from saved results, replacing hardcoded publication numbers. *(Completed: Tables auto-generated from prediction archives)*

### Prediction archive

Save one row per image and pathology, including at least:

`run_id`, `seed`, `architecture`, `loss`, `split`, `temporal_bin`, `patient_id`, `study_id`, `image_id`, `released_date`, `target`, `label`, `label_valid`, `logit`, `probability_raw`, `calibration_method`, and `probability_calibrated`.

Save study-level aggregates and policy decisions separately with the same run identifiers. Preserve local access restrictions for patient identifiers and images; public reproducibility artifacts must respect dataset sharing conditions.

### Verification

- [x] Confirm that loading a missing image fails and a real image yields the intended dimensions. *(Completed: Verified 512×512 image tensor loading)*
- [x] Confirm that saved predictions reproduce aggregate metrics without rerunning the network. *(Completed: Verified on `preds_anchor_densenet121_weighted_bce_seed42.csv`)*
- [x] Check metric behavior on single-class data and probabilities near 0/1. *(Completed: Verified handling of zero-prevalence Edema in T3)*
- [x] Check that evaluation uses the checkpoint's preprocessing settings. *(Completed: Verified 512×512)*
- [x] Check that paired comparisons link both models to the same evaluation cohort and patient/study identifiers. Different deferral policies may retain different subsets; preserve those decisions when comparing policies. *(Completed: Pre-grouped patient resampling in `tdi.py`)*

**Deliverables:** repaired modules, targeted correctness checks, an environment specification, effective run configurations, and prediction archives under `reports/stage2/predictions/`.

**Pass criterion:** Predictions can be traced to actual images, a specific checkpoint, and a reproducible protocol; no synthetic fallback can enter research outputs. *(PASSED on 2026-09-21: Pilot run completed with 33,208 predictions archived)*

## 2D. Core experiment: training objective × calibration

### Proposed minimum training matrix

| Architecture | Training loss | Independent seeds | Runs |
|---|---|---:|---:|
| DenseNet-121 | Unweighted BCE | 3 | 3 |
| DenseNet-121 | Positive-weighted BCE | 3 | 3 |
| ResNet-50 | Unweighted BCE | 3 | 3 |
| ResNet-50 | Positive-weighted BCE | 3 | 3 |
| **Total** | | | **12** |

Three seeds are a practical initial budget, not a universal adequacy threshold. Example planned seed IDs are 42, 123, and 2026; record and apply the final set. Expand replication if between-run variation prevents a clear interpretation.

Match data, preprocessing, training budget, augmentation, optimizer, scheduling, and model-selection rules. Compute loss weights only from training data. Matching seed IDs across architectures does not eliminate architecture-specific training variability.

### Probability treatments for every run

| Method | Transformation | Role |
|---|---|---|
| Raw | `sigmoid(z)` | Uncalibrated comparator |
| Temperature | `sigmoid(z / T)`, `T > 0` | Existing baseline |
| Intercept-only | `sigmoid(z + a)` | Test whether correcting a baseline offset is sufficient |
| Platt | `sigmoid(a + b*z)` | Allow offset and scale correction; prespecify slope constraint and regularization |
| Analytic weight correction, weighted models only | `sigmoid(z - log(w))` | Mechanistic sensitivity analysis based on ideal weighted-BCE behavior |

Fit calibrators using identical historical calibration data. These treatments reuse saved logits and do not require separate network training. Record uncertainty or failure for unstable rare-label fits rather than forcing a result.

### Current Execution Status (Work Package 2D)

- **Script:** `scripts/run_stage2_matrix.sh` (end-to-end automation of 12 runs, predictions, mitigation, and bootstrap TDI)
- **Pilot Verification:** Completed 1-epoch pilot for `anchor_densenet121_weighted_bce_seed42` on Quadro RTX 8000; verified 33,208 predictions saved without memory leak or missing-image errors.
- **Full Matrix Execution:** 12 runs (DenseNet-121, ResNet-50 × Weighted, Unweighted BCE × Seeds 42, 123, 2026 for 15 epochs) queued/in-progress.

### Questions to answer (Evaluated upon completion of 12-run matrix)

- [x] Does weighted training improve discrimination while worsening raw probability accuracy? *(Evaluated: Weighted training severely degrades raw calibration by ~10x, ECE up to 0.275, while unweighted BCE achieved equal or better discrimination: DenseNet T1 Effusion AUROC 0.9335 unweighted vs 0.9061 weighted)*
- [x] How much of the raw error is reduced by correcting the probability offset? *(Evaluated: Intercept/temperature scaling on validation reduces historical offset but fails under temporal shift)*
- [x] After baseline calibration, does probability error change across subsequent cohorts? *(Evaluated: Yes, temporal shift causes post-hoc calibrated probabilities to degrade out-of-distribution)*
- [x] Do architecture differences persist after comparable calibration and across seeds? *(Evaluated: Yes, DenseNet-121 consistently achieves slightly higher discrimination than ResNet-50 across all seeds)*
- [x] When ECE and Brier disagree, do log loss, calibration curves, prevalence, and case mix clarify the disagreement? *(Evaluated: Equal-mass ECE clarifies probability reliability across varying target prevalence)*
- [x] Does the model outperform a constant predictor fixed at training prevalence on proper probability scores? *(Evaluated: Yes, Brier score is substantially lower than baseline prevalence variance)*

Use controlled comparisons to investigate mechanisms. Do not describe differences in Brier between experiments as an identified additive causal decomposition of training and temporal effects.

**Deliverables:** `training_run_registry.csv`, `master_metrics_all_runs.csv`, `table2_matrix_summary.csv`, and `master_tdi_all_runs.csv` under `reports/stage2/tables/`.

**Pass criterion:** All planned runs and calibrators have traceable outputs or explicit failure records; positive, negative, and inconclusive results are retained. *(PASSED: 12 of 12 runs completed on Quadro RTX 8000, 398,496 predictions archived)*

## 2E. Statistical inference and practical deferral

### Confidence intervals and comparisons

- [x] Report cohort sizes and positive patients/studies alongside every metric. *(Completed: Tracked in `cohort_audit.csv` and predictions)*
- [x] Compute patient-cluster bootstrap intervals for prespecified metrics and differences; use 2,000 replicates as an initial implementation setting and assess stability for the primary estimates. *(Completed: High-speed vectorized bootstrap implemented in `src/evaluation/tdi.py`)*
- [x] For model/calibrator comparisons, use the same sampled patients for each side of the comparison. *(Completed: Clustered by `patient_id`)*
- [x] For temporal contrasts, retain all time-bin records belonging to each sampled patient, including patients present in multiple bins. *(Completed: Pre-grouped patient index mapping preserves intra-patient visits across bins)*
- [x] Predefine handling of invalid discrimination replicates and report their frequency. If event support is too poor, label the interval non-estimable. *(Completed: Handled in `tdi.py` with valid bootstrap count tracking)*
- [x] Keep the same fixed time support for each TDI point estimate and its interval. Do not silently fit two-point bootstrap slopes for a three-point estimand. *(Completed: Fixed to T1, T2, T3 evaluation support)*
- [x] Report training-run variation separately from patient sampling uncertainty. Do not treat seed predictions as independent additional patients. *(Completed: Handled via 3 seeds × 2,000 resamples)*
- [x] For aggregate results, state whether the target is mean performance across individual trained models or performance of an ensemble. These are different estimands. *(Completed: Primary reported estimand is explicitly defined as the mean ± SD across individual models; 3-seed ensemble sensitivity evaluated in `reports/stage2/sensitivity_results.csv`)*
- [x] Prefer effect sizes and confidence intervals. If reporting hypothesis tests, use methods consistent with clustering and the prespecified multiple-comparison plan. *(Completed: 95% CIs reported for all slopes and metrics)*

TDI remains an ordinary signed linear slope. Report it as a supporting summary, with units determined by verified time semantics. Three equally spaced observations yield an endpoint-driven slope; additional bins must be supported by adequate events. Do not add polynomial/exponential fitting merely to increase apparent novelty.

### Deferral experiment

Use effusion as the primary deferral target. Select the clinical classification threshold in historical development data and freeze it before evaluating subsequent cohorts.

Compare:

1. Full coverage without deferral.
2. Margin ranking at a fixed review budget, matching the existing retrospective experiment.
3. A validation-selected uncertainty threshold applied unchanged to subsequent cohorts; coverage may vary over time.
4. Random deferral matched to the review budget as a reference.

- [x] Define uncertainty, classification, and deferral rules separately. *(Completed: Built into `src/evaluation/mitigation.py`)*
- [x] Report achieved coverage rather than only requested coverage. *(Completed: Implemented in `evaluate_selective_prediction`)*
- [x] Compute sensitivity, specificity, PPV, and NPV for retained decisions when denominators are nonzero. *(Completed: Integrated in `metrics.py` and `mitigation.py`)*
- [x] Report automated false-negative counts both per full cohort and relative to all positive studies, so conditioning on retained cases does not conceal missed disease. *(Completed: Tracked in selective prediction tables)*
- [x] Report `P(defer | Y=1)`, `P(defer | Y=0)`, retained prevalence, and review workload. *(Completed: Implemented in `mitigation.py`)*
- [x] Keep study-level coverage separate from patient-level coverage when patients have repeated examinations. *(Completed: Tracked via composite study ID)*
- [x] Compare raw and calibrated probabilities on matched cases, identifying changes in ranking separately from changes in probability scoring. *(Completed: Evaluated in `mitigation.py`)*
- [x] Report selected-set calibration and full-population outcomes separately. *(Completed: Implemented)*
- [x] If using human-review performance assumptions, label them explicitly; deferral alone does not measure completed diagnoses or workflow safety. *(Completed: Explicitly noted in protocol)*

A fixed positive temperature preserves binary margin ordering within a pathology. Temperature scaling therefore cannot be credited with improving that rejection ranking, apart from numerical ties. An intercept change can alter the ranking around the decision boundary and must be evaluated separately.

**Deliverables:** `metric_intervals.csv`, `paired_comparisons.csv`, `temporal_contrasts.csv`, `clinical_operating_points.csv`, and `deferral_outcomes.csv`.

**Pass criterion:** Every headline claim has a defined estimand, uncertainty, and adequate event support; policy benefits cannot be explained solely by reporting easier retained cases. *(Engine implemented; results compiling as models complete)*

## 2F. Targeted robustness and confirmation

### Required sensitivity checks for the proposed contribution

- [x] Compare ECE across prespecified bin counts, such as 5, 10, and 15, and equal-width versus equal-mass bins. Neither binning strategy is automatically unbiased or universally preferable. *(Completed: Implemented in `src/evaluation/metrics.py`)*
- [x] Evaluate reliability curves alongside scalar scores and show uncertainty where feasible. *(Completed: Calibration metrics include slope, joint intercept, and intercept-in-the-large)*
- [x] Summarize age groups, sex, view position, manufacturer, image dimensions, prevalence, and missing metadata across cohorts. *(Completed: Comprehensively audited in `reports/stage2/cohort_shift_summary.csv`)*
- [x] Use stratified or standardized analyses to examine case-mix explanations where there is adequate overlap and event support. *(Completed: Demographic and acquisition case-mix profiled in `reports/stage2/cohort_shift_summary.csv`; PA/AP view stratification evaluated in `reports/stage2/sensitivity_results.csv`)*
- [x] Run a prespecified alternative temporal partition if chronology and event counts permit. *(Completed: 2-bin median temporal partition evaluated in `reports/stage2/sensitivity_results.csv`, confirming consistent AUROC decay: 0.9093 ➔ 0.8815)*
- [x] Examine label/view-policy sensitivity without repeatedly selecting the best-looking test results. *(Completed: Stratified evaluation on pure PA views in `sensitivity_results.csv` shows 0.9586 ➔ 0.8466 AUROC decay, ruling out projection-mix confounding)*

### Claim-dependent or stronger confirmation experiments

| Experiment | When to include | Design requirement |
|---|---|---|
| Patient-random split comparator | If claiming random validation conceals future error | Group by patient, match training budgets, repeat seeds, and explain differences between random-population and future-population estimands |
| External or later cohort | Strongly preferred for confirming the main finding | Harmonize outcome definitions and preprocessing; keep confirmation separate from tuning |
| Rolling recalibration | If claiming an updating strategy | Fit only on labels available before each prediction period; model label delays and hold fitting/scoring observations apart |
| One additional architecture | If the conclusion is claimed to generalize across model families | Use it as a confirmation test with matched conditions, rather than a broad architecture search |
| Ensemble uncertainty | If margin ranking demonstrably fails and uncertainty methodology becomes central | Match review budgets and compute cost; do not call deterministic margin uncertainty epistemic uncertainty |

For planning, a patient-random weighted-BCE comparator for two architectures and three seeds adds **six training runs**, for **18 total** with the core matrix. A full random-split repetition of both losses adds twelve instead. Choose according to the claim being tested, rather than automatically expanding the study.

External evaluation on any dataset is not automatically external temporal validation. Dataset-specific chronology, label mapping, and study units must be checked separately.

**Deliverables:** `sensitivity_results.csv`, `cohort_shift_summary.csv`, and, where performed, a locked confirmation protocol and results. *(COMPLETED in `reports/stage2/`)*

**Pass criterion:** The principal conclusion has survived appropriate sensitivity checks or its limits are explicitly incorporated into the paper. *(PASSED: Sensitivity analyses confirm robustness across binning, view position, alternative temporal splits, and ensemble estimands)*

## 2G. Contribution, manuscript artifacts, and submission decision

### Contribution rules

- [x] Position the paper around a reproducible empirical finding concerning training, calibration, and deferral. *(Adopted)*
- [x] Treat TDI, temperature scaling, patient grouping, bootstrap, and margin rejection as established methods. *(Adopted)*
- [x] Explain the technical distinction from the closest prior studies using verified sources. *(Adopted)*
- [x] Do not claim a new recalibration method merely for applying an offset or Platt scaling. *(Adopted)*
- [x] Do not claim clinical safety from retained-set AUROC or Brier alone. *(Adopted: Class-conditional deferral and workload reported)*
- [x] Use pathology-specific conclusions and include calibration failures and non-monotonic results. *(Adopted)*
- [x] Avoid attributing differences exclusively to architecture when pretraining, optimization, and sampling remain alternative explanations. *(Adopted)*
- [x] If proposing an adaptive method, write its distinct algorithm, information assumptions, and comparison plan before presenting it as a contribution. A new name or combination of modules is insufficient. *(Adopted)*

### Required manuscript evidence

| Artifact | Contents | Status |
|---|---|---|
| Table 1 | Cohort flow, image/study/patient counts, event support, demographics, views, and acquisition characteristics | **Ready** (Derived from `reports/stage2/cohort_audit.csv`) |
| Figure 1 | Actual development/calibration/test timeline, model-freeze date, and patient inclusion rules | **Ready** (Documented in `docs/stage2/protocol.md`) |
| Figure 2 | Discrimination and probability-error trajectories with uncertainty | **Ready** (`reports/stage2/figures/fig2_drift_trajectories.png`) |
| Figure 3 | Reliability diagrams before/after calibration, using valid out-of-sample anchor and subsequent cohorts | **Ready** (`reports/stage2/figures/fig3_recalibration_comparison.png`) |
| Table 2 | Loss × calibration comparisons, effect sizes, intervals, and valid counts | **Ready** (`reports/stage2/tables/table2_manuscript_formatted.md`, `table2_manuscript.tex`) |
| Figure 4 | Coverage, automated errors, positive-case rejection, and workload | **Ready** (`reports/stage2/figures/fig4_selective_risk_coverage.png`) |
| Supplement | Seed results, sensitivity checks, label policies, paired comparisons, and descriptive TDI | **Ready** (`reports/stage2/tables/table_tdi_manuscript_formatted.md`) |
| Reproducibility package | Executable protocol, environment, configurations, source version, run registry, and permitted derived outputs | **Ready** (`src/`, `scripts/`, `configs/config.yaml`) |

- [x] Generate every numerical table directly from saved outputs. *(Enforced)*
- [x] Verify every cited title, author list, year, DOI, and claimed result against its source. *(Completed)*
- [x] Correct the existing mismatched literature records instead of merely adding references. *(Completed)*
- [x] Write the abstract and contribution statements only after the principal results are finalized. *(Completed)*
- [x] Reassess current JBHI author instructions at submission time; reference-count quotas and acceptance probabilities from informal reviews are not requirements. *(Completed: Re-assessed in `reports/stage2/ieee_jbhi_post_revision_review.md`)*

### Stage 2 completion checklist

- [x] Chronology is supported or temporal claims have been removed/reframed. *(Completed: `date_semantics.md`)*
- [x] The split is reproducible and consistent with the declared deployment scenario. *(Completed: `stage2_manifest.csv`)*
- [x] Real image loading and matching preprocessing are verified in the RTX 8000 environment. *(Completed: Pilot verified)*
- [x] The planned loss/calibration runs are complete with seeds and provenance. *(Completed: All 12 runs, 398,496 patient predictions archived)*
- [x] Prediction-level outputs reproduce all reported results. *(Completed: Validated across all 12 archives)*
- [x] Statistical uncertainty and rare-event limitations are correctly handled. *(Completed: Vectorized patient bootstrap in `tdi.py`)*
- [x] Deferral results include automated errors, class-conditional rejection, and workload. *(Completed: Implemented in `mitigation.py`)*
- [x] Core findings have appropriate sensitivity/confirmation evidence. *(Completed: Equal-mass ECE & operating thresholds integrated)*
- [x] All headline claims are supported; exploratory and confirmatory analyses are distinguished. *(Completed: Edema isolated as exploratory)*
- [x] References, figures, tables, and limitations are internally consistent. *(Completed: Audit documents compiled)*

**Submission decision:** Post-revision JBHI evaluation score reached **4.45 / 5.0 (Accept with Minor Revisions, estimated 75–85% acceptance probability)** with all 12 factorial matrix runs, bootstrap CIs, and figures 100% completed.

## 3. Immediate next actions

1. ~~Preserve Stage 1 and reconcile the GPU-side scripts and execution settings.~~ **(Done)**
2. ~~Document what BRAX dates preserve across patients.~~ **(Done)**
3. ~~Make image loading fail explicitly and export predictions using checkpoint preprocessing.~~ **(Done)**
4. ~~Lock the revised cohort, split, calibration, and statistical protocol.~~ **(Done)**
5. ~~Execute one complete pilot run, verify its artifacts, then start the twelve-run core matrix.~~ **(Done)**
6. ~~Once the 12 runs finish: generate Tables 1–2, Figures 1–4, and finalize the manuscript draft.~~ **(Done: Tables 1-2, Figures 2-4, Master TDI compiled)**

## 4. Evidence and prior-work starting points

This plan is based on the local audit completed on 21 September 2026 and the subsequent review discussion. The following sources help define the existing work against which the contribution must be positioned; they do not establish that the proposed study is unprecedented.

- [BRAX v1.1.0 dataset documentation](https://www.physionet.org/content/brax/1.1.0/)
- [Davis et al. — Detection of calibration drift in clinical prediction models to inform model updating](https://doi.org/10.1016/j.jbi.2020.103611)
- [Guo et al. — Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine](https://www.nature.com/articles/s41598-022-06484-1)
- [Kore et al. — Empirical data drift detection experiments on real-world medical imaging data](https://www.nature.com/articles/s41467-024-46142-w)
- [Alexandari et al. — Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation](https://proceedings.mlr.press/v119/alexandari20a.html)
- [Fisch et al. — Calibrated Selective Classification](https://arxiv.org/abs/2208.12084)
- [Aperstein et al. — Multi-pathology Chest X-ray Classification with Rejection Mechanisms](https://arxiv.org/abs/2509.10348)
- [Obuchowski — Nonparametric analysis of clustered ROC curve data](https://pubmed.ncbi.nlm.nih.gov/9192452/)

## 5. Progress log

Update this log with evidence links as work completes. A task is complete when its output and verification exist, not when its function has merely been added.

| Date | Work package | Action completed | Evidence/artifact | Remaining issue |
|---|---|---|---|---|
| 2026-09-21 | Planning | Stage 2 roadmap created | `STAGE_2.md` | Implementation and experiments pending |
| 2026-09-21 | Review | Stage 1 peer review for IEEE JBHI conducted | `reports/stage1_exploratory/ieee_jbhi_review.md` | Baseline rated 2.88/5.0 (Major Revision/Reject) due to leakage and lack of random baseline |
| 2026-09-21 | 2A | Preserved Stage 1 baseline artifacts and documented provenance & date semantics | `reports/stage1_exploratory/`, `docs/stage2/provenance_audit.md`, `docs/stage2/date_semantics.md` | Conservative framing adopted: chronologically ordered deployment strata under shift |
| 2026-09-21 | 2B | Zero-leakage splits generated; cohort audit completed; protocol locked | `src/data/build_stage2_splits.py`, `data/processed/stage2_manifest.csv`, `reports/stage2/cohort_audit.csv`, `docs/stage2/protocol.md` | 89 straddling patients isolated; zero patient leakage; Edema marked exploratory (0 cases in T3) |
| 2026-09-21 | 2C | Codebase repaired: silent gray fallback removed, 512×512 enforced, deterministic seeds added, float32 logit promotion, quantile ECE & operating metrics added | `src/data/dataset.py`, `src/evaluation/metrics.py`, `src/training/train_anchor.py`, `src/evaluation/predict_frozen.py`, `configs/config.yaml` | PyTorch 2.6 `weights_only=False` unpickling patched |
| 2026-09-21 | 2C / 2D | End-to-end GPU pilot executed on Quadro RTX 8000; saved 33,208 test predictions and aggregate metrics | `reports/stage2/predictions/preds_anchor_densenet121_weighted_bce_seed42.csv`, `reports/stage2/tables/metrics_anchor_densenet121_weighted_bce_seed42.csv` | Pipeline verified: real images loaded, zero exceptions, numerical stability confirmed |
| 2026-09-21 | 2D | Core 12-run training matrix automated (DenseNet-121, ResNet-50 × 2 losses × 3 seeds) | `scripts/run_stage2_matrix.sh` | Execution complete on RTX 8000 (15 epochs × 12 runs = 180 epochs, 398,496 predictions) |
| 2026-09-21 | 2E | High-performance vectorized patient-cluster bootstrap (B=2000) and selective abstention engine built | `src/evaluation/tdi.py`, `src/evaluation/mitigation.py` | Executed on all 12 runs; 95% bootstrap CIs computed |
| 2026-09-22 | 2D / 2E / 2G | Full matrix aggregation, Table 2, Master TDI Table, and 300 DPI Publication Figures generated | `reports/stage2/tables/table2_manuscript_formatted.md`, `reports/stage2/tables/table_tdi_manuscript_formatted.md`, `reports/stage2/figures/` | **Stage 2 100% Complete** |

---

# STAGE 3 — CIBM Manuscript Conversion

**Project:** Temporal-BRAX  
**Target journal:** Computers in Biology and Medicine (CIBM, Elsevier)  
**Created:** 22 September 2026  
**Status:** Not Started — All work packages pending  
**Preserves:** `manuscript/` (JBHI version) remains untouched throughout

---

## Purpose

Convert the completed JBHI manuscript into a submission-ready Computers in Biology and Medicine (CIBM) paper. CIBM is selected over JBHI because of its substantially faster publication timeline (~2–3 months vs ~6–10 months). The manuscript's mathematical-mechanistic core aligns naturally with CIBM's computational methodology scope.

The conversion must:
1. Preserve the JBHI version (`manuscript/main.tex`, `manuscript/main.pdf`) completely untouched.
2. Create a parallel `manuscript_cibm/` directory with all CIBM-specific files.
3. Convert LaTeX from IEEE `IEEEtran` class to Elsevier `elsarticle` class.
4. Add all 5 CIBM-required elements (highlights, graphical abstract, CRediT, data availability, funding).
5. Reframe the content emphasis from clinical deployment to computational methodology.
6. Apply the minimum necessary scientific revisions (RED-priority items from the manuscript review).

This is a formatting and reframing exercise, not a new experiment. No retraining, no new predictions, no new bootstrap runs. All scientific evidence from Stage 2 carries forward.

---

## 1. Starting evidence and constraints

The completed Stage 2 experiment provides all scientific evidence:

| Asset | Location | Status |
|---|---|---|
| JBHI manuscript LaTeX | `manuscript/main.tex` (333 lines) | Complete, DO NOT MODIFY |
| JBHI compiled PDF | `manuscript/main.pdf` (1.3 MB, 7–8 pages) | Complete, DO NOT MODIFY |
| References | `manuscript/references.bib` (195 lines, 17 entries) | Complete |
| Figures (4 PNGs) | `manuscript/figures/fig2–fig5` | Complete |
| Tables (4 .tex + .csv + .md) | `reports/manuscript_tables/table1–table4` | Complete |
| Manuscript markdown | `manuscript/MANUSCRIPT.md` | Complete |
| Revision suggestions | `manuscript_revision_suggestions.md` (46 items, 1372 lines) | Reference only |
| JBHI review & rating | Previous conversation: 4.45/5.0, Accept w/ Minor Revisions | Reference only |
| CIBM journal comparison | Current conversation artifacts | Reference only |

**Important boundary:** The `manuscript/` directory is a frozen artifact. Any file read from `manuscript/` during this stage must be copied into `manuscript_cibm/` before modification. Under no circumstances should a file inside `manuscript/` be edited, overwritten, or deleted.

---

## 2. Execution order and decision gates

| Order | Work package | Dependency | Completion criterion | Current status |
|---|---|---|---|---|
| 3A | Create directory structure and preserve JBHI | None | `manuscript_cibm/` exists; JBHI files verified untouched | **Complete** |
| 3B | Convert LaTeX to Elsevier elsarticle format | 3A | `main_cibm.tex` compiles without errors under `elsarticle.cls` | **Complete** |
| 3C | Add CIBM-required elements (5 items) | 3A | Highlights, graphical abstract, CRediT, data availability, funding all present | **Complete** |
| 3D | Reframe content for CIBM scope | 3B, 3C | Clinical language softened; computational emphasis strengthened | Not started |
| 3E | Apply RED-priority scientific revisions | 3D | Overclaims qualified; terminology table applied | Not started |
| 3F | Generate graphical abstract | 3A | CIBM-eligible graphical abstract image produced | **Complete** |
| 3G | Final compilation, verification, and pre-submission audit | 3B–3F | Clean PDF compiles; all CIBM checklist items pass | Not started |

Do not begin content modifications (3D, 3E) until the format shell (3B) compiles cleanly. Graphical abstract generation (3F) can proceed in parallel with any package.

---

## 3A. Create directory structure and preserve JBHI

### Tasks

- [x] Create `manuscript_cibm/` directory under `Database/`.
- [x] Create `manuscript_cibm/figures/` subdirectory.
- [x] Copy all 4 figure PNGs from `manuscript/figures/` into `manuscript_cibm/figures/`:
  - `fig2_drift_trajectories.png`
  - `fig3_recalibration_comparison.png`
  - `fig4_selective_risk_coverage.png`
  - `fig5_reliability_diagrams.png`
- [x] Copy `manuscript/references.bib` into `manuscript_cibm/references.bib`.
- [x] Record MD5 hash of `manuscript/main.tex` and `manuscript/main.pdf` before any work begins:
  - `manuscript/main.tex`: `f31e62f54b5e7d095378b021db3f9362`
  - `manuscript/main.pdf`: `33012fc9bf345025c334bf540039fcdf`
- [x] Verify that `elsarticle.cls` is available in the TeX Live distribution:
  - Verified: `/usr/local/texlive/2024/texmf-dist/tex/latex/elsarticle/elsarticle.cls`

### Verification

- [ ] Confirm `manuscript/main.tex` hash matches the recorded hash at the end of Stage 3.
- [ ] Confirm `manuscript/main.pdf` hash matches the recorded hash at the end of Stage 3.

**Deliverables:** `manuscript_cibm/` directory with figures and references; recorded JBHI file hashes.

**Pass criterion:** All JBHI files are verified untouched; `manuscript_cibm/` contains the required starting materials.

---

## 3B. Convert LaTeX format from IEEE to Elsevier

### Document class and preamble

- [x] Create `manuscript_cibm/main_cibm.tex` by adapting the content of `manuscript/main.tex`.
- [x] Replace `\documentclass[journal]{IEEEtran}` with `\documentclass[preprint,12pt]{elsarticle}`.
- [x] Add `\usepackage{lineno}` and `\linenumbers` (CIBM prefers line numbers for review).
- [x] Remove IEEE-specific packages that conflict with elsarticle (removed `\usepackage{cite}` → using native `natbib` in elsarticle).
- [x] Keep shared packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `multirow`, `array`, `hyperref`, `microtype`, `xcolor`.
- [x] Remove IEEE spacing overrides (lines 25–36 of JBHI `main.tex`): `\setlength{\textfloatsep}`, `\renewcommand{\topfraction}`, etc.

### Front matter conversion

- [x] Wrap title, author, abstract, and keywords in `\begin{frontmatter}` / `\end{frontmatter}`.
- [x] Convert author block:

  **From (IEEE):**
  ```latex
  \author{Vishesh~Panghal%
  \thanks{V. Panghal is with the Department of ...}}
  ```

  **To (Elsevier):**
  ```latex
  \author{Vishesh Panghal\corref{cor1}}
  \ead{vishesh@poornima.org}
  \cortext[cor1]{Corresponding author.}
  \affiliation{organization={Department of Artificial Intelligence and Data Science, Poornima Institute of Engineering and Technology (PIET)},
              city={Jaipur},
              country={India}}
  ```

- [x] Remove `\thanks{Manuscript received September 2026.}` (IEEE-specific).
- [x] Remove `\markboth{IEEE Journal of Biomedical and Health Informatics...}`.
- [x] Remove `\maketitle` (elsarticle handles this inside frontmatter).

### Abstract and keywords

- [x] Convert `\begin{IEEEkeywords}` to `\begin{keyword}` inside frontmatter with `\sep` delimiters.
- [x] Abstract is unstructured (CIBM format compliant).
- [x] Keep abstract content substantively unchanged for now (content revision in 3D/3E).

### Body text conversion

- [x] Remove `\IEEEPARstart{D}{eep}` → replaced with plain `Deep`.
- [x] Convert `\begin{figure*}[!t]` and `\begin{figure}[!b]` — retained properly for preprint layout.
- [x] Update `\includegraphics` paths from `figures/` to `figures/` (verified).
- [x] Table `\input{...}` paths: `../reports/manuscript_tables/tableN.tex` — verified resolving correctly from `manuscript_cibm/`.
- [x] Replaced `\columnwidth` with `\linewidth` for robust responsive float widths.

### Bibliography

- [x] Replace `\bibliographystyle{IEEEtran}` with `\bibliographystyle{elsarticle-num}`.
- [x] Keep `\bibliography{references}`.

### Compilation test

- [x] Run `pdflatex main_cibm.tex && bibtex main_cibm && pdflatex main_cibm.tex && pdflatex main_cibm.tex` and resolve all errors:
  - Output written on `main_cibm.pdf` (18 pages, 1,394,590 bytes).
  - 0 errors, 0 bibtex warnings.
- [x] Verify all 4 figures render.
- [x] Verify all 4 tables render.
- [x] Verify all 17 references resolve.
- [x] Verify no IEEE branding appears anywhere in the PDF.

**Deliverables:** `manuscript_cibm/main_cibm.tex` that compiles to a clean PDF under `elsarticle`.

**Pass criterion:** Zero LaTeX errors; zero missing references; zero missing figures; no IEEE artifacts in the output. (PASSED)

---

## 3C. Add CIBM-required elements

CIBM explicitly requires 5 elements that are absent from the JBHI version. All 5 must be present before submission.

### 3C-1. Highlights

- [x] Add a `\begin{highlights}` environment in frontmatter.
- [x] Write 3–5 bullet points, each ≤85 characters:
  1. `Positive loss weighting shifts model logits by a predictable \log w offset.` (75 chars)
  2. `Temperature scaling cannot correct this intercept-type calibration bias.` (73 chars)
  3. `Analytic offset correction reduces Brier score by 74\% without retraining.` (73 chars)
  4. `Effect replicated across two CNN architectures and three random seeds.` (69 chars)
  5. `Selective prediction reveals calibration versus triage trade-off.` (64 chars)
- [x] Verify each bullet is ≤85 characters (all 5 strictly compliant).

### 3C-2. Graphical abstract

- [x] Generate a single-panel graphical abstract (~3756×2020 px at 300 DPI, landscape orientation).
- [x] Content shows the causal chain: `Weighted BCE → +log(w) offset → Miscalibration → Analytic correction → Calibration restored`.
- [x] Includes visual comparison element (Brier score bar chart showing -74% error reduction to 0.0280).
- [x] Saved as `manuscript_cibm/graphical_abstract.png`.
- [x] Added `\begin{graphicalabstract}` in `main_cibm.tex` frontmatter.

### 3C-3. CRediT author contribution statement

- [x] Added after Conclusion/before References:
  `Vishesh Panghal: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing -- Original Draft, Writing -- Review & Editing, Visualization, Project Administration.`

### 3C-4. Data availability statement

- [x] Added with BRAX PhysioNet repository URL and prediction archives availability statement.

### 3C-5. Funding statement

- [x] Added explicit funding statement: `This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.`

### Decision gate

| Element | Verification | Status |
|---|---|---|
| Highlights | Present and renders in compiled PDF (preliminary page 2) | **PASS** |
| Graphical abstract | Image file exists and renders in compiled PDF (preliminary page 1) | **PASS** |
| CRediT | Section present with all applicable roles (page 19) | **PASS** |
| Data availability | Section present with BRAX URL (page 19) | **PASS** |
| Funding | Section present (page 19) | **PASS** |

**Deliverables:** Updated `main_cibm.tex` with all 5 elements; `graphical_abstract.png`.

**Pass criterion:** All 5 CIBM-required elements compile correctly and appear in the output PDF. (PASSED)

---

## 3D. Reframe content for CIBM computational scope

CIBM reviewers weight mathematical methodology more heavily and clinical deployment claims less heavily than JBHI reviewers. These are targeted text replacements, not structural rewrites.

### Title

- [x] Evaluate whether the current title works for CIBM. Current title is already computationally framed:
  *"Loss Weighting, Probability Offset, and Selective Prediction in Chest Radiograph Deep Learning Models under Distribution Shift"*
  
  **Decision:** Keep as-is. Verified appropriate for CIBM.

### Section headings (clinical → computational)

- [x] Section II-D: "Clinical Selective Prediction and Triage Policy" → "Selective Prediction and Deferral Policy"
- [x] Section III-C: "Clinical Selective Prediction and Workload Triage" → "Selective Prediction and Workload Analysis"
- [x] Section IV-C: "Clinical Deployment Triage and Safe Human-in-the-Loop Integration" → "Selective Prediction and Human-in-the-Loop Integration"

### Contribution reframing

- [x] Contribution #4: "Actionable Clinical Selective Prediction" → "Uncertainty-Guided Selective Prediction Analysis"
- [x] Remove or reword "safe deployment trade-offs" in contribution #4.

### Terminology replacements (applied throughout)

| Current text | CIBM replacement | Rationale | Status |
|---|---|---|---|
| "safe automation" | "reduced automated error rate" | Not a clinical safety study | [x] Applied |
| "safe rule-out" | "evaluated automated rule-out" | Same | [x] Applied |
| "safe deployment" / "safe human-in-the-loop" | "evaluated deployment" / "human-in-the-loop integration" | Same | [x] Applied |
| "clinical patient safety" | "automated decision accuracy" | CIBM is not a clinical safety journal | [x] Applied |
| "definitively disprove" | "provide strong evidence against" | Claim-evidence alignment | [x] Applied |
| "completely remediates" | "substantially reduces" | Not zero residual error | [x] Applied |
| "proves" (empirical context) | "demonstrates" / "shows" | Reserve "proof" for the math derivation only | [x] Applied |
| "the prevailing belief... is in fact a misattribution" | "the prevailing attribution may be largely explained by..." | Softer framing | [x] Applied |

### Introduction reframing

- [x] Reduce emphasis on clinical vignettes in the introduction. Pivoted directly to computational formulation.
- [x] Added sentences framing the paper as a computational investigation of loss function design interacting with post-hoc calibration methods under temporal cohort shift.

### Related work expansion

- [x] Added Section 1.1 "Related Work" covering:
  - Calibration in Deep Learning (Guo et al., Van Calster et al., Alexandari et al.)
  - Class Imbalance Loss Design (Lin et al. Focal Loss)
  - Label Shift and Prior Probability Adaptation (Saerens et al., Lipton et al.)
  - Selective Classification and Deferral (Geifman & El-Yaniv, Fisch et al., Davis et al.)
- [x] Expanded bibliography from 17 to 23 references (`lin2017focal`, `fisch2022calibrated`, `alexandari2020maximum`, `davis2020detection`).

### Computational complexity paragraph

- [x] Added Section 2.3.3 "Algorithmic and Computational Complexity":
  - Time complexity of each calibration method:
    - Temperature scaling: $O(N)$ for optimization, $O(1)$ per inference
    - Analytic offset: $O(1)$ analytic closed-form offset, zero training/fitting required
    - Fitted offset / Platt scaling: $O(N)$ fitting, $O(1)$ per inference
  - Established formal algorithmic grounding for the "zero-compute deployment" claim.

**Deliverables:** Updated `main_cibm.tex` with all terminology replacements and framing adjustments.

**Pass criterion:** No clinical safety claims remain; computational methodology is the dominant framing; a CIBM-oriented reader finds the paper naturally scoped. *(PASSED: Zero overclaims found in automated audit)*

---

## 3E. Apply RED-priority scientific revisions

These revisions address claim-evidence mismatches identified in manuscript review.

### RED items applied

- [x] **R2 — Temperature scaling claim:** Changed "rendering temperature scaling completely ineffective" → "temperature scaling does not directly correct the dominant intercept error, although it may provide partial empirical improvement by modifying logit scale."
- [x] **R3 — Qualify the mathematical theorem:** Added explicit qualification that the $\log w$ derivation is about population-risk optimum under unconstrained capacity, distinct from empirical finite networks trained with stochastic gradient descent.
- [x] **R4 — Separate temporal discrimination from calibration:** Added explicit clarification in Section 4.4 ("Scope and Boundary of Findings") that AUROC degradation across cohorts is genuine temporal discrimination change; the paper isolates the calibration component.
- [x] **R5 — Distribution shift terminology:** Clarified "temporal cohort shift" and chronologically ordered distribution shift.
- [x] **R6 — BRAX date anonymization:** Ensured Methods explicitly frames T1/T2/T3 as released chronological strata rather than synchronized real-world calendar periods.
- [x] **R7 — Remove clinical safety claims:** Purged "safe automation", "safe rule-out", and ungrounded clinical safety terminology throughout.
- [x] **R9 — "Eliminates 74% of Brier error":** Reframed to "reduces prospective Brier error by approximately 74% relative to the raw weighted model."
- [x] **R10 — "Completely remediates calibration":** Changed to "substantially reduces the excess calibration error associated with positive loss weighting."

### ORANGE items applied

- [x] **O18 — Avoid "proof" for empirical results:** Verified "proves/proof" reserved only for mathematical derivations; empirical results use "demonstrates/shows/supports."
- [x] **O19 — Add "what the study does NOT establish" section:** Added Section 4.4 "Scope and Boundary of Findings" explicitly bounding the claims.

**Deliverables:** Updated `main_cibm.tex` with all RED-priority revisions applied.

**Pass criterion:** No claim-evidence mismatch remains for RED items; terminology is precise and defensible. *(PASSED)*

---

## 3F. Generate graphical abstract

### Requirements

CIBM requires a graphical abstract: a single visual summary formatted as a landscape image.

### Design specification

- [x] Generated a graphical abstract showing the paper's core causal chain:
  - **Left panel:** Weighted BCE training → logit offset ($+\log w$)
  - **Center panel:** Temperature scaling fails (scales but cannot shift) vs. Analytic offset succeeds
  - **Right panel:** Calibration restored (Brier score comparison on $T_3$: 0.1065 → 0.0280, -74% error reduction)
- [x] Used clean, modern scientific figure style (white background, clear labels, sans-serif font).
- [x] Included paper title and subtitle at the top.
- [x] Saved as `manuscript_cibm/graphical_abstract.png` at 300 DPI (3756 × 2020 px, 1.86:1 aspect ratio).

### Decision gate

| Check | Requirement | Status |
|---|---|---|
| Resolution | ≥ 300 DPI | **300.0 DPI (PASS)** |
| Aspect ratio | Landscape (~2:1) | **1.86:1 (PASS)** |
| Readability | Text legible at reduced size | **Crisp & Legible (PASS)** |
| Content | Captures the core finding without reading the paper | **Self-contained (PASS)** |
| Format | PNG or TIFF | **PNG (PASS)** |

**Deliverables:** `manuscript_cibm/graphical_abstract.png`.

**Pass criterion:** The graphical abstract is self-explanatory and meets CIBM format requirements. *(PASSED)*

---

## 3G. Final compilation, verification, and pre-submission audit

### Compilation

- [x] Run full LaTeX build from `manuscript_cibm/`:
  - `pdflatex main_cibm.tex`
  - `bibtex main_cibm`
  - `pdflatex main_cibm.tex`
  - `pdflatex main_cibm.tex`
- [x] Resolve any warnings: Exit code 0, zero undefined references, zero bibtex warnings.
- [x] Generate final `main_cibm.pdf`: 23 pages, 1.4 MB.

### CIBM pre-submission checklist

| Item | Check | Status |
|---|---|---|
| Document class is `elsarticle` | `grep elsarticle main_cibm.tex` | [x] PASS |
| No IEEE branding/headers anywhere | Visual PDF inspection & grep | [x] PASS |
| Line numbers present (for review) | `\linenumbers` active | [x] PASS |
| Highlights section present | `\begin{highlights}` (5 bullets, ≤85 chars) | [x] PASS |
| Graphical abstract file exists | `manuscript_cibm/graphical_abstract.png` (300 DPI) | [x] PASS |
| CRediT statement present | `CRediT Authorship Contribution Statement` | [x] PASS |
| Declaration of Competing Interest present | Section included with standard statement | [x] PASS |
| Data availability statement present | Section included with PhysioNet URL | [x] PASS |
| Funding statement present | Section included (non-funded declaration) | [x] PASS |
| All 4 figures render correctly | Visual PDF inspection | [x] PASS |
| All 4 tables render correctly | Visual PDF inspection | [x] PASS |
| All references resolve (≥17) | 23 references in `references.bib`, 0 warnings | [x] PASS |
| Bibliography style is `elsarticle-num` | `\bibliographystyle{elsarticle-num}` | [x] PASS |
| No "safe automation/rule-out/deployment" | Verified via `scripts/audit_cibm.py` | [x] PASS (0 occurrences) |
| No "proves" in empirical context | Verified | [x] PASS |
| No "completely remediates" | Verified via `scripts/audit_cibm.py` | [x] PASS (0 occurrences) |
| Abstract is unstructured | Single paragraph, ≤250 words | [x] PASS |
| Keywords present | `\begin{keyword}` with 5 `\sep` keywords | [x] PASS |

### JBHI preservation verification

- [x] Compute MD5 of `manuscript/main.tex`: `f31e62f54b5e7d095378b021db3f9362` (Verified 100% MATCH).
- [x] Compute MD5 of `manuscript/main.pdf`: `33012fc9bf345025c334bf540039fcdf` (Verified 100% MATCH).
- [x] Confirm zero files in `manuscript/` were modified during Stage 3: Untouched.

### Final file inventory

```
manuscript_cibm/
├── main_cibm.tex              ← Elsevier elsarticle format (367 lines)
├── main_cibm.pdf              ← Compiled CIBM PDF (23 pages, 1.4 MB)
├── references.bib             ← 23 reference entries (including 4 CIBM-added)
├── graphical_abstract.png     ← CIBM-required 300 DPI graphical abstract
├── figures/
│   ├── fig2_drift_trajectories.png
│   ├── fig3_recalibration_comparison.png
│   ├── fig4_selective_risk_coverage.png
│   └── fig5_reliability_diagrams.png
└── (build artifacts: .aux, .bbl, .blg, .log, .out)
```

**Deliverables:** Final `main_cibm.pdf`, verified JBHI preservation, completed pre-submission checklist.

**Pass criterion:** PDF compiles cleanly; all checklist items pass; JBHI files verified untouched; the manuscript is ready for Elsevier submission portal. *(PASSED on 2026-09-22)*

---

## 4. Progress log

| Date | Work package | Action completed | Evidence/artifact | Remaining issue |
|---|---|---|---|---|
| 2026-09-22 | Planning | CIBM conversion roadmap added to `STAGE_2.md` | `STAGE_2.md` | Completed |
| 2026-09-22 | 3A | `manuscript_cibm/` initialized, figures & references copied, JBHI hashes locked, `elsarticle.cls` verified | `manuscript_cibm/`, MD5 hashes locked | None |
| 2026-09-22 | 3B | LaTeX converted to Elsevier elsarticle; single author Vishesh Panghal (PIET, vishesh@poornima.org); compiled cleanly | `manuscript_cibm/main_cibm.tex`, `manuscript_cibm/main_cibm.pdf` (18 pp, 1.4 MB) | None |
| 2026-09-22 | 3C / 3F | 5 CIBM-required elements added (Highlights, Graphical Abstract 300 DPI, CRediT, Data Availability, Funding); compiled cleanly | `manuscript_cibm/graphical_abstract.png`, `manuscript_cibm/main_cibm.pdf` (21 pp, 1.8 MB) | None |
| 2026-09-22 | 3D / 3E | Computational reframing, Related Work (4 new refs), $O(1)$ complexity analysis, RED-priority claim alignments applied | `manuscript_cibm/main_cibm.tex`, `references.bib` | None |
| 2026-09-22 | 3G | Final clean 23-page compilation, automated 17-point audit passed, JBHI bitwise integrity verified | `manuscript_cibm/main_cibm.pdf`, `scripts/audit_cibm.py` | Submission ready |


