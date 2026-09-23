# Stage 2 Scientific Protocol & Statistical Analysis Plan

**Target Journal:** IEEE Journal of Biomedical and Health Informatics (JBHI)  
**Status:** Locked Protocol  
**Date:** 21 September 2026

---

## 1. Study Objectives & Hypotheses

The primary research question is:
> *How do training loss formulations (unweighted vs positive-weighted BCE) and cohort composition shifts interact to degrade probability calibration in chest radiograph models, and which post-hoc recalibration and clinical abstention policies provide verifiable risk mitigation?*

### Pre-Registered Endpoints

| Category | Endpoint | Target Condition | Role |
|:---|:---|:---|:---|
| **Primary Calibration** | Log Loss & Brier Score | Pleural Effusion | Confirmatory primary |
| **Supporting Calibration** | Equal-Mass ECE & Intercept-in-the-large | Pleural Effusion | Confirmatory supporting |
| **Primary Discrimination** | AUROC & AUPRC | Pleural Effusion | Confirmatory supporting |
| **Secondary Target** | All calibration + AUROC | Cardiomegaly | Confirmatory secondary |
| **Exploratory Targets** | All metrics | Pneumonia, Edema | Exploratory (event scarcity) |
| **Clinical Decision-Point** | Sensitivity, Specificity, PPV, NPV at fixed operating points | Pleural Effusion | Clinical utility |

---

## 2. Cohort Definition & Temporal Model-Freeze Boundary

- **Dataset:** BRAX v1.1.0 (`data/processed/stage2_manifest.csv`)
- **Strict Model Freeze Cutoff ($T_{\text{freeze}}$):** `2015-07-01`
- **Validation Split Cutoff ($T_{\text{val}}$):** `2013-09-01`

$$\max(\text{StudyDate}_{\text{train}}) < \min(\text{StudyDate}_{\text{val}}) \le \max(\text{StudyDate}_{\text{val}}) < \min(\text{StudyDate}_{\text{test}})$$

### Verified Cohort Distribution

| Cohort | Time Window | Studies | Unique Patients | Primary Target Prevalence (Effusion) |
|:---|:---|---:|---:|---:|
| **Anchor Train** | 2008-03-25 to 2013-08-31 | 25,123 (61.3%) | 11,265 | 4.44% (1,116 cases) |
| **Anchor Val** | 2013-09-01 to 2015-06-30 | 7,098 (17.3%) | 3,281 | 4.16% (295 cases) |
| **Prospective Test** | 2015-07-01 to 2017-12-17 | 8,302 (20.3%) | 3,807 | 4.35% (361 cases) |
| → $T_1$ (2015) | 2015-07-01 to 2015-12-31 | 1,879 | 913 | 5.27% (99 cases) |
| → $T_2$ (2016) | 2016-01-01 to 2016-12-31 | 3,987 | 1,832 | 4.44% (177 cases) |
| → $T_3$ (2017) | 2017-01-01 to 2017-12-17 | 2,436 | 1,164 | 3.49% (85 cases) |

*Patient Isolation Guarantee:* $\text{PatientID}_{\text{train}} \cap \text{PatientID}_{\text{val}} \cap \text{PatientID}_{\text{test}} = \emptyset$.  
89 boundary-straddling patients (0.48%) whose studies crossed cutoffs are isolated to prevent longitudinal leakage.

---

## 3. Core Experimental Matrix (Work Package 2D)

A controlled $2 \times 2 \times 3$ factorial design (12 training runs total):

| Run ID | Architecture | Training Objective | Seed | Epochs | Selection Metric |
|:---|:---|:---|:---:|:---:|:---|
| `anchor_densenet121_unweighted_bce_seed42` | DenseNet-121 | Standard BCE | 42 | 15 | Effusion AUROC |
| `anchor_densenet121_unweighted_bce_seed123` | DenseNet-121 | Standard BCE | 123 | 15 | Effusion AUROC |
| `anchor_densenet121_unweighted_bce_seed2026` | DenseNet-121 | Standard BCE | 2026 | 15 | Effusion AUROC |
| `anchor_densenet121_weighted_bce_seed42` | DenseNet-121 | Pos-Weighted BCE | 42 | 15 | Effusion AUROC |
| `anchor_densenet121_weighted_bce_seed123` | DenseNet-121 | Pos-Weighted BCE | 123 | 15 | Effusion AUROC |
| `anchor_densenet121_weighted_bce_seed2026` | DenseNet-121 | Pos-Weighted BCE | 2026 | 15 | Effusion AUROC |
| `anchor_resnet50_unweighted_bce_seed42` | ResNet-50 | Standard BCE | 42 | 15 | Effusion AUROC |
| `anchor_resnet50_unweighted_bce_seed123` | ResNet-50 | Standard BCE | 123 | 15 | Effusion AUROC |
| `anchor_resnet50_unweighted_bce_seed2026` | ResNet-50 | Standard BCE | 2026 | 15 | Effusion AUROC |
| `anchor_resnet50_weighted_bce_seed42` | ResNet-50 | Pos-Weighted BCE | 42 | 15 | Effusion AUROC |
| `anchor_resnet50_weighted_bce_seed123` | ResNet-50 | Pos-Weighted BCE | 123 | 15 | Effusion AUROC |
| `anchor_resnet50_weighted_bce_seed2026` | ResNet-50 | Pos-Weighted BCE | 2026 | 15 | Effusion AUROC |

---

## 4. Post-Hoc Calibration & Mitigation Policies

For every trained model, raw validation logits will be used to fit post-hoc calibrators without altering feature representations:
1. **Raw Logits:** $p = \sigma(z)$
2. **Temperature Scaling:** $p = \sigma(z / T)$ ($T > 0$ fit by minimizing BCE)
3. **Intercept-in-the-large Offset:** $p = \sigma(z + a)$ (offset correction for loss-weighting bias)
4. **Platt Scaling:** $p = \sigma(a + b \cdot z)$
5. **Analytic Weight Correction:** $p = \sigma(z - \log w)$ (exact mathematical inverse of positive loss weighting)

### Selective Prediction / Deferral
- Uncertainty: Binary margin $u(x) = 1 - |2p - 1|$
- Coverage levels evaluated: 100%, 95%, 90%, 85%, 80%, 75%, 70%
- Key safety constraint: Report $P(\text{defer} \mid Y=1)$ vs $P(\text{defer} \mid Y=0)$ to guarantee diseased patients are not disproportionately deferred.

---

## 5. Statistical Inference (Work Package 2E)

- **Patient-Cluster Bootstrapping:** $B = 2,000$ resamples with replacement at the unique `PatientID` level.
- **Paired Comparisons:** Contrast differences (e.g. DenseNet vs ResNet, or Unweighted vs Weighted BCE) on identical bootstrap samples.
- **95% Confidence Intervals:** 2.5th and 97.5th percentiles of bootstrap distribution.
