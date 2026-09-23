# Loss Weighting, Probability Offset, and Selective Prediction in Chest Radiograph Deep Learning Models under Distribution Shift

**Author:** Vishesh Panghal  
**Affiliation:** Department of Biomedical Engineering and Computer Science  
**Target Journal:** *IEEE Journal of Biomedical and Health Informatics (JBHI)*  
**Date:** September 2026  

---

## Abstract

**Objective:** Deep learning models deployed in clinical radiology frequently encounter longitudinal distribution shifts. Under extreme class imbalance, training with positive-weighted binary cross-entropy (BCE) is standard practice. However, its downstream impact on probability calibration under temporal shift has been widely conflated with intrinsic feature drift, leading to claims of severe and unavoidable "temporal calibration decay." In this study, we resolve this conflation through a controlled $2 \times 2 \times 3$ factorial experiment across 12 deep learning models (DenseNet-121 and ResNet-50 trained under unweighted and positive-weighted BCE across 3 random seeds) on 40,523 chest radiographs from the BRAX dataset.

**Methods:** Models are frozen at an anchor cutoff ($<$\,2015-07) and evaluated prospectively across chronologically stratified cohorts ($T_1$: 2015-H2, $T_2$: 2016, $T_3$: 2017) with strict patient-level isolation. We prove that positive loss weighting induces a constant logit offset ($\log w$), which standard temperature scaling structurally fails to correct. We benchmark post-hoc calibrators (temperature scaling, analytic offset, fitted offset, and non-negative Platt scaling) using 5-fold patient-clustered cross-fitting on $T_1$ with 2,000-replicate cluster bootstrap inference. Furthermore, we design a multi-tier clinical selective prediction policy tracking radiologist referral workloads and automated false negatives.

**Results:** Raw weighted DenseNet-121 models exhibit apparent severe calibration failure on Pleural Effusion (ensemble Brier score $0.0898 \to 0.1065$ from $T_1$ to $T_3$). Temperature scaling yields negligible improvement ($T_3$ Brier: 0.1059, $\Delta\text{Brier} = -0.0007$ [95% CI: $-0.0023, +0.0010$]). In contrast, an analytic offset correction ($z - \log w$) eliminates 74% of the Brier error ($T_3$ Brier: 0.0280, $\Delta\text{Brier} = -0.0786$ [95% CI: $-0.0894, -0.0677$]), matching unweighted models (0.0269) and proving that calibration actually improves over time as prevalence decreases. Finally, we evaluate a clinical selective prediction policy tracking radiologist referral workloads and automated false negatives. Evaluated with matched decision thresholds determined on development data, selective triage at 70% coverage automates 1,706 radiographs in $T_3$: referring 30% of ambiguous examinations under the raw weighted model halves automated false negatives from 16.0 to 7.0 (88.9% sensitivity) but leaves probabilities uncalibrated (Brier 0.0959); under analytic correction, probability accuracy is dramatically superior (Brier 0.0170), though 13 false negatives remain automated due to compressed low-probability margins.

**Conclusion:** Apparent "temporal calibration decay" in weighted chest radiograph models is predominantly an uncorrected artifact of positive loss weighting rather than irreversible feature drift. A single-parameter logit offset ($z - \log w$) or Platt scaling restores prospective calibration without model retraining. Selective prediction with matched decision thresholds reveals a key clinical dichotomy: raw weighted models provide superior recall-oriented triage by halving automated false negatives under a 30% referral budget, whereas analytic calibration restores epidemiologically faithful probabilities for downstream clinical risk assessment. Neither system eliminates missed disease entirely, underscoring the necessity of transparent, workload-aware deployment policies.

**Keywords:** Chest Radiography, Probability Calibration, Class Imbalance, Distribution Shift, Selective Prediction, Deep Learning, Clinical Decision Support.

---

## 1. Introduction

Deep learning models for automated chest radiograph interpretation have achieved radiologist-level discriminatory performance across thoracic abnormalities such as pleural effusion, cardiomegaly, and pneumonia. However, the transition from stationary retrospective validation to prospective clinical deployment introduces substantial distribution shifts. In healthcare environments, populations evolve, imaging protocols shift, scanner hardware is replaced, and seasonal or epidemiological patterns alter disease incidence over time.

While discrimination (measured via AUROC) often exhibits gradual degradation under distribution shift, clinical deployment decisions rely heavily on *calibrated probabilities*. A physician deciding whether to perform thoracentesis or refer an outpatient for urgent computed tomography requires an accurate estimate of posterior disease probability: $P(Y = 1 \mid X = x)$. When predicted probabilities deviate from empirical event rates, clinical decision thresholds fail, risk scores become misleading, and automated triage pipelines can cause catastrophic undertreatment or excessive diagnostic referrals.

Recent clinical AI literature has reported rapid "temporal calibration decay," asserting that frozen deep neural networks suffer progressive miscalibration over deployment years. Simultaneously, because medical imaging datasets are characterized by extreme class imbalance (pathology prevalence often below 5%), researchers routinely train vision models using positive-weighted binary cross-entropy (BCE) to penalize missed positive cases.

In this work, we demonstrate that these two phenomena are deeply entangled, and that the prevailing belief in unavoidable temporal calibration decay is in fact a misattribution. When models are trained with positive loss weighting $w > 1$, the asymptotic Bayes-optimal logit is shifted by $+\log w$. Standard post-hoc calibration methods—most notably temperature scaling—only rescale the logit variance ($z / T$) and cannot shift log-odds. Consequently, weighted models continuously over-predict positive probabilities, producing severe Brier score and Expected Calibration Error (ECE) degradation that worsens over time as disease prevalence fluctuates.

```
+-----------------------------------------------------------------------------+
|                               CONTRIBUTIONS                                 |
+-----------------------------------------------------------------------------+
| 1. Controlled Factorial Matrix: 12 models (DenseNet-121 & ResNet-50 x       |
|    Standard & Weighted BCE x 3 seeds) on 40,523 BRAX chest radiographs.     |
| 2. Mathematical Disproof of Inherent Decay: Proved positive BCE shifts      |
|    logits by log(w); demonstrated temperature scaling fails while analytic   |
|    offset (z - log w) eliminates 74% of Brier error.                        |
| 3. Leak-Free Cross-Fitting Protocol: 5-fold patient-clustered cross-fitting  |
|    on T1 with 2,000-replicate patient-cluster bootstrap inference.          |
| 4. Actionable Clinical Selective Triage: Evaluated referral workloads and   |
|    demonstrated a 56% reduction in automated false negatives at 70% retention|
+-----------------------------------------------------------------------------+
```

---

## 2. Materials and Methods

### 2.1 Dataset and Longitudinal Cohort Stratification
We utilize the Brazilian Chest X-ray (BRAX) v1.1.0 benchmark, containing 40,523 chest radiographs acquired across multiple inpatient and outpatient emergency facilities at Hospital Israelita Albert Einstein in São Paulo, Brazil. Radiographs are labeled with 14 thoracic findings extracted from Portuguese radiological reports using a validated NLP pipeline.

In compliance with HIPAA de-identification standards, direct identifiers and original study timestamps are protected. We extract chronological study dates from the released 8-digit calendar integers (`YYYYMMDD`), spanning 2008 through 2017. As is standard under HIPAA Safe Harbor de-identification, date anonymization may include patient-specific random date shifting that preserves relative within-patient temporal intervals while shifting absolute calendar offsets across patients. Consequently, our prospective partitions $T_1$ (2015-H2), $T_2$ (2016), and $T_3$ (2017) are interpreted as chronologically ordered cohort strata under longitudinal distribution shift rather than synchronized real-world calendar years.
1. All studies are partitioned chronologically around a strict model freeze boundary: $T_{\text{freeze}} = \text{2015-07-01}$.
2. Training studies occur before 2013-09-01 ($N = 25,123$ images from 11,265 patients across 14,315 studies).
3. Validation studies occur between 2013-09-01 and 2015-06-30 ($N = 7,098$ images from 3,281 patients across 4,012 studies).
4. Prospective test studies occur between 2015-07-01 and 2017-12-17 ($N = 8,302$ images from 3,807 patients across 4,670 studies), partitioned into three consecutive deployment strata: $T_1$ (2015-H2, $N=1,879$), $T_2$ (2016, $N=3,987$), and $T_3$ (2017, $N=2,436$).
5. Strict patient isolation is enforced: $\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} \cap \text{Patients}_{\text{test}} = \emptyset$. A small subset of 89 patients whose visits crossed split boundaries were excluded from validation and testing to guarantee zero within-patient contamination.

### Table 1: Cohort Demographics and Clinical Characteristics
| Cohort Stratum | Images ($N$) | Patients ($N$) | Studies ($N$) | Sex (% M / F) | Age (Mean ± SD) | Views (% PA / AP / Lat / Unk) | Pleural Effusion $N$ (%) | Cardiomegaly $N$ (%) | Pneumonia $N$ (%) | Edema $N$ (%) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| **Train (Pre-deployment)** | 25,123 | 11,265 | 14,315 | 49.3% / 50.7% | 46.0 ± 24.3 | 34.0% / 13.5% / 0.0% / 27.2% | 1,116 (4.44%) | 2,398 (9.55%) | 456 (1.82%) | 23 (0.09%) |
| **Val (Pre-deployment)** | 7,098 | 3,281 | 4,012 | 49.8% / 50.2% | 46.2 ± 24.1 | 34.7% / 12.2% / 0.0% / 27.7% | 295 (4.16%) | 681 (9.59%) | 128 (1.80%) | 8 (0.11%) |
| **Test $T_1$ (2015)** | 1,879 | 913 | 1,039 | 47.8% / 52.2% | 44.5 ± 23.9 | 34.4% / 10.2% / 0.0% / 29.0% | 99 (5.27%) | 147 (7.82%) | 50 (2.66%) | 4 (0.21%) |
| **Test $T_2$ (2016)** | 3,987 | 1,832 | 2,249 | 48.5% / 51.5% | 46.1 ± 23.8 | 34.6% / 12.0% / 0.0% / 27.6% | 177 (4.44%) | 422 (10.58%) | 79 (1.98%) | 13 (0.33%) |
| **Test $T_3$ (2017)** | 2,436 | 1,164 | 1,382 | 50.9% / 49.1% | 46.0 ± 24.6 | 34.4% / 12.4% / 0.0% / 28.1% | 85 (3.49%) | 258 (10.59%) | 50 (2.05%) | 0 (0.00%) |
| **Test All (Pooled)** | 8,302 | 3,807 | 4,670 | 49.0% / 51.0% | 45.7 ± 24.1 | 34.5% / 11.7% / 0.0% / 28.1% | 361 (4.35%) | 827 (9.96%) | 179 (2.16%) | 17 (0.20%) |

---

### 2.2 Model Architectures, Training, and Ensembling
We evaluate two standard clinical vision backbones: DenseNet-121 and ResNet-50, both initialized with ImageNet pre-trained weights. Input chest radiographs are resized to $224 \times 224$ pixels and normalized using ImageNet channel statistics.

- **Optimization:** Models are trained using the Adam optimizer with an initial learning rate of $1 \times 10^{-4}$ and weight decay of $1 \times 10^{-5}$, using a batch size of 32 for exactly 15 epochs with a Cosine Annealing learning rate schedule.
- **Checkpoint Selection:** Rather than using early stopping with arbitrary patience cutoffs, models are trained for the complete 15 epochs, and validation AUROC across all 14 findings is logged at each epoch. The checkpoint achieving the highest mean validation AUROC (`selection_metric_val`) across the 15 epochs is saved as the final model.
- **Random Seeds & Ensembling:** For each architecture and loss function, models are trained across three independent random seeds (seed42, seed43, seed44), yielding 12 base neural networks. Prospective ensemble predictions are computed by averaging uncalibrated logits across the three seeds:
$$\bar{z}(x) = \frac{1}{3} \sum_{s=1}^3 z_s(x)$$
with calibrated probabilities computed directly on ensemble logits.

---

### 2.3 Mathematical Mechanics of Loss Weighting and Calibration

Under positive-weighted binary cross-entropy:
$$\mathcal{L}_{\text{pos}}(y, \hat{p}) = - [w \cdot y \log \hat{p} + (1 - y) \log(1 - \hat{p})]$$
where $w = (N_{\text{total}} - N_{\text{pos}}) / N_{\text{pos}}$.

Setting the derivative of expected loss with respect to logit $z$ to zero yields the Bayes-optimal logit:
$$z^*(X) = \log\left(\frac{P(Y=1 \mid X)}{P(Y=0 \mid X)}\right) + \log w$$

This proves that positive loss weighting introduces an exact additive bias $+\log w$ into model log-odds.

#### Why Temperature Scaling Fails
Temperature scaling transforms probabilities via $p = \sigma(z / T)$. Because $T$ scales the entire logit multiplicatively, it cannot shift the logit intercept. In low-prevalence clinical settings, scaling $z$ either flattens all predictions toward 0.5 or exacerbates over-confidence. Consequently, numerical optimization routinely selects $T^* \approx 1.0$, completely failing to address the underlying $\log w$ offset.

#### Recalibration Suite
We benchmark four calibrator formulations:
1. **Temperature Scaling ($T^*$):** $p = \sigma(z / T^*)$, bounded Brent minimization with monotonicity safeguard.
2. **Analytic Offset ($z - \log w$):** Subtracts the exact theoretical offset without requiring optimization or tuning data.
3. **Fitted Offset ($z + a^*$):** Optimizes scalar intercept $a^*$ via Brent minimization, slope fixed at 1.
4. **Platt Scaling ($a + bz$):** Logistic regression on logits with non-negative slope constraint ($b \ge 0$).

---

### 2.4 Clinical Selective Prediction Protocol
To evaluate safe deployment in human-in-the-loop triage, we implement a selective classification rule. To prevent evaluation leakage across different probability scales, we determine an operating decision cutoff $t^*$ for each method strictly on development stratum $T_1$ using Youden's $J$-index ($J = \text{Sensitivity} + \text{Specificity} - 1$), without using prospective evaluation labels. For unweighted models, $t^* \approx 0.014$; for analytic offset calibrated models, $t^* \approx 0.022$; and for raw weighted models, $t^* \approx 0.33$. We define a threshold-relative uncertainty metric based on piecewise distance from the operational cutoff $t^*$:

$$u(x) = 1 - \frac{|\hat{p}(x) - t^*|}{\mathbb{I}(\hat{p}(x) \ge t^*)(1 - t^*) + \mathbb{I}(\hat{p}(x) < t^*)t^*} \in [0, 1]$$

where $u(x) \to 1$ indicates maximal ambiguity ($\hat{p}(x) \approx t^*$) and $u(x) \to 0$ indicates decisive confidence. Crucially, because $t^*$ scales with baseline disease prevalence (e.g., $t^* \approx 0.022$ for analytic calibrated models), predictions below $t^*$ are normalized by $t^*$. Consequently, low-probability false negatives ($\hat{p} \ll t^*$) yield low uncertainty and are retained in the automated pool rather than referred. At coverage level $C \in [0.5, 1.0]$, the automated system retains the fraction $C$ of cases with lowest uncertainty and refers the remaining fraction $1 - C$ to radiologist review.

---

## 3. Experimental Results

### 3.1 Factorial Matrix: Longitudinal Trajectories
Table 2 and Figure 2 display the prospective performance trajectories across DenseNet-121 and ResNet-50.

### Table 2: Factorial Matrix Evaluation across Architectures, Objectives, and Conditions
| Architecture | Objective | Target | AUROC $T_1$ | AUROC $T_2$ | AUROC $T_3$ | AUROC $\Delta$ | Brier $T_1$ | Brier $T_2$ | Brier $T_3$ | Brier $\Delta$ | ECE $T_1$ | ECE $T_3$ | ECE $\Delta$ | LogLoss $T_1$ | LogLoss $T_3$ | LogLoss $\Delta$ |
|---|---|---|---|---|---|---:|---|---|---|---:|---|---|---:|---|---|---:|
| **DenseNet-121** | Unweighted BCE | Pleural Effusion | 0.9335 ± 0.0052 | 0.8895 ± 0.0115 | 0.8731 ± 0.0120 | -0.0604 | 0.0369 ± 0.0029 | 0.0356 ± 0.0009 | 0.0279 ± 0.0009 | -0.0090 | 0.0232 ± 0.0078 | 0.0090 ± 0.0022 | -0.0143 | 0.1318 ± 0.0076 | 0.1114 ± 0.0027 | -0.0204 |
| DenseNet-121 | Unweighted BCE | Cardiomegaly | 0.9075 ± 0.0116 | 0.8965 ± 0.0140 | 0.9017 ± 0.0170 | -0.0057 | 0.0532 ± 0.0012 | 0.0677 ± 0.0021 | 0.0677 ± 0.0028 | +0.0145 | 0.0140 ± 0.0009 | 0.0210 ± 0.0074 | +0.0071 | 0.1798 ± 0.0072 | 0.2231 ± 0.0102 | +0.0433 |
| DenseNet-121 | Unweighted BCE | Pneumonia | 0.8473 ± 0.0129 | 0.8818 ± 0.0166 | 0.8647 ± 0.0172 | +0.0174 | 0.0235 ± 0.0008 | 0.0182 ± 0.0003 | 0.0187 ± 0.0006 | -0.0048 | 0.0142 ± 0.0011 | 0.0095 ± 0.0016 | -0.0046 | 0.0996 ± 0.0035 | 0.0808 ± 0.0039 | -0.0187 |
| DenseNet-121 | Weighted BCE | Pleural Effusion | 0.9061 ± 0.0537 | 0.8676 ± 0.0457 | 0.8391 ± 0.0657 | -0.0670 | 0.1081 ± 0.0394 | 0.1287 ± 0.0432 | 0.1254 ± 0.0456 | +0.0173 | 0.1964 ± 0.0702 | 0.2227 ± 0.0765 | +0.0264 | 0.3646 ± 0.1179 | 0.4100 ± 0.1279 | +0.0454 |
| DenseNet-121 | Weighted BCE | Cardiomegaly | 0.8739 ± 0.0540 | 0.8599 ± 0.0600 | 0.8575 ± 0.0699 | -0.0164 | 0.1262 ± 0.0247 | 0.1358 ± 0.0316 | 0.1360 ± 0.0306 | +0.0098 | 0.2031 ± 0.0507 | 0.1969 ± 0.0486 | -0.0062 | 0.3982 ± 0.0807 | 0.4241 ± 0.0906 | +0.0259 |
| DenseNet-121 | Weighted BCE | Pneumonia | 0.8181 ± 0.0206 | 0.8494 ± 0.0313 | 0.8424 ± 0.0496 | +0.0243 | 0.1313 ± 0.0533 | 0.1308 ± 0.0555 | 0.1298 ± 0.0513 | -0.0015 | 0.2393 ± 0.0894 | 0.2425 ± 0.0883 | +0.0032 | 0.4226 ± 0.1511 | 0.4166 ± 0.1471 | -0.0060 |
| **ResNet-50** | Unweighted BCE | Pleural Effusion | 0.9295 ± 0.0113 | 0.8755 ± 0.0293 | 0.8608 ± 0.0190 | -0.0687 | 0.0372 ± 0.0022 | 0.0364 ± 0.0013 | 0.0284 ± 0.0010 | -0.0088 | 0.0162 ± 0.0050 | 0.0132 ± 0.0030 | -0.0029 | 0.1294 ± 0.0099 | 0.1145 ± 0.0067 | -0.0149 |
| ResNet-50 | Unweighted BCE | Cardiomegaly | 0.8898 ± 0.0177 | 0.8756 ± 0.0317 | 0.8785 ± 0.0196 | -0.0112 | 0.0572 ± 0.0037 | 0.0709 ± 0.0059 | 0.0712 ± 0.0051 | +0.0140 | 0.0233 ± 0.0190 | 0.0175 ± 0.0081 | -0.0058 | 0.1954 ± 0.0193 | 0.2379 ± 0.0174 | +0.0425 |
| ResNet-50 | Unweighted BCE | Pneumonia | 0.8432 ± 0.0178 | 0.8659 ± 0.0130 | 0.8798 ± 0.0169 | +0.0366 | 0.0242 ± 0.0007 | 0.0187 ± 0.0003 | 0.0191 ± 0.0003 | -0.0051 | 0.0128 ± 0.0050 | 0.0106 ± 0.0015 | -0.0023 | 0.1024 ± 0.0059 | 0.0809 ± 0.0035 | -0.0215 |
| ResNet-50 | Weighted BCE | Pleural Effusion | 0.8789 ± 0.0508 | 0.8467 ± 0.0458 | 0.8191 ± 0.0496 | -0.0598 | 0.1256 ± 0.0325 | 0.1417 ± 0.0277 | 0.1392 ± 0.0310 | +0.0137 | 0.2481 ± 0.0733 | 0.2751 ± 0.0769 | +0.0270 | 0.4003 ± 0.0895 | 0.4365 ± 0.0750 | +0.0361 |
| ResNet-50 | Weighted BCE | Cardiomegaly | 0.8298 ± 0.0550 | 0.8046 ± 0.0701 | 0.7990 ± 0.0788 | -0.0308 | 0.1415 ± 0.0124 | 0.1526 ± 0.0083 | 0.1522 ± 0.0085 | +0.0107 | 0.2450 ± 0.0348 | 0.2338 ± 0.0258 | -0.0112 | 0.4427 ± 0.0389 | 0.4678 ± 0.0305 | +0.0251 |
| ResNet-50 | Weighted BCE | Pneumonia | 0.8349 ± 0.0199 | 0.8359 ± 0.0345 | 0.8233 ± 0.0533 | -0.0116 | 0.1667 ± 0.0584 | 0.1681 ± 0.0583 | 0.1691 ± 0.0613 | +0.0024 | 0.3189 ± 0.1175 | 0.3253 ± 0.1208 | +0.0064 | 0.5056 ± 0.1443 | 0.5096 ± 0.1527 | +0.0040 |

---

### 3.2 Multi-Calibrator Comparison on Pleural Effusion
Table 3 summarizes calibration performance across 2,000 patient-clustered bootstrap replicates on the 3-seed ensemble.

### Table 3: Multi-Calibrator Evaluation on Pleural Effusion (Ensemble Bootstrap 95% CIs)
| Architecture | Loss Objective | Calibrator | $T_1$ Brier | $T_2$ Brier | $T_3$ Brier | $\Delta$ Brier ($T_3 - T_1$) | $T_3$ ECE | $T_3$ Log-Loss | $T_3$ $\Delta$ Brier vs Raw [95% Bootstrap CI] |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **DenseNet-121** | **Weighted BCE** | Raw (Uncalibrated) | 0.0898 | 0.1103 | 0.1065 | +0.0167 | 0.2074 | 0.3524 | Reference |
| DenseNet-121 | Weighted BCE | Temperature Scaling ($T^*$) | 0.0874 | 0.1098 | 0.1059 | +0.0185 | 0.1805 | 0.3513 | -0.0007 [-0.0023, +0.0010] |
| DenseNet-121 | Weighted BCE | **Analytic Offset ($z - \log w$)** | **0.0365** | **0.0345** | **0.0280** | **-0.0085** | **0.0060** | **0.1099** | **-0.0786 [-0.0894, -0.0677]** |
| DenseNet-121 | Weighted BCE | Fitted Offset ($z + a^*$) | 0.0351 | 0.0354 | 0.0292 | -0.0059 | 0.0180 | 0.1137 | -0.0773 [-0.0873, -0.0674] |
| DenseNet-121 | Weighted BCE | Platt Scaling ($a + bz$) | 0.0354 | 0.0361 | 0.0297 | -0.0056 | 0.0186 | 0.1148 | -0.0768 [-0.0867, -0.0670] |
| **DenseNet-121** | **Unweighted BCE**| Raw (Uncalibrated) | 0.0360 | 0.0343 | 0.0269 | -0.0091 | 0.0079 | 0.1053 | Reference |
| DenseNet-121 | Unweighted BCE| Temperature Scaling ($T^*$) | 0.0352 | 0.0339 | 0.0268 | -0.0084 | 0.0081 | 0.1054 | -0.0001 [-0.0005, +0.0003] |
| DenseNet-121 | Unweighted BCE| Fitted Offset ($z + a^*$) | 0.0331 | 0.0346 | 0.0277 | -0.0054 | 0.0165 | 0.1083 | +0.0008 [-0.0008, +0.0023] |
| DenseNet-121 | Unweighted BCE| Platt Scaling ($a + bz$) | 0.0331 | 0.0361 | 0.0289 | -0.0042 | 0.0173 | 0.1103 | +0.0020 [-0.0003, +0.0042] |
| **ResNet-50** | **Weighted BCE** | Raw (Uncalibrated) | 0.1018 | 0.1189 | 0.1156 | +0.0138 | 0.2525 | 0.3783 | Reference |
| ResNet-50 | Weighted BCE | Temperature Scaling ($T^*$) | 0.0887 | 0.1137 | 0.1069 | +0.0182 | 0.1794 | 0.3424 | -0.0087 [-0.0131, -0.0043] |
| ResNet-50 | Weighted BCE | **Analytic Offset ($z - \log w$)** | **0.0428** | **0.0377** | **0.0299** | **-0.0128** | **0.0155** | **0.1165** | **-0.0856 [-0.0952, -0.0757]** |
| ResNet-50 | Weighted BCE | Fitted Offset ($z + a^*$) | 0.0382 | 0.0359 | 0.0292 | -0.0090 | 0.0213 | 0.1197 | -0.0864 [-0.0950, -0.0776] |
| ResNet-50 | Weighted BCE | Platt Scaling ($a + bz$) | 0.0344 | 0.0380 | 0.0313 | -0.0031 | 0.0225 | 0.1205 | -0.0843 [-0.0920, -0.0765] |
| **ResNet-50** | **Unweighted BCE**| Raw (Uncalibrated) | 0.0362 | 0.0351 | 0.0271 | -0.0091 | 0.0067 | 0.1070 | Reference |

Key empirical findings:
1. Temperature scaling offers virtually zero calibration remediation in weighted DenseNet-121 ($\Delta\text{Brier} = -0.0007$ [95% CI: $-0.0023, +0.0010$]).
2. The Analytic Offset ($z - \log w$) eliminates 74% of the probability error ($\Delta\text{Brier} = -0.0786$ [95% CI: $-0.0894, -0.0677$]), reducing $T_3$ Brier from $0.1065$ down to $0.0280$, in complete parity with unweighted BCE ($0.0269$).

---

### 3.3 Clinical Selective Prediction and Referral Workload
Table 4 displays the clinical workload and patient safety profile on cohort $T_3$ ($N = 2,436$ radiographs from 1,382 studies across 1,164 patients) evaluated with matched decision thresholds determined on development stratum $T_1$.

### Table 4: Clinical Selective Prediction Workload and Risk Profile on Prospective Cohort $T_3$ (Matched Youden Thresholds)
| Coverage (Retention) | Model Strategy | Automated Decisions ($N$) | Referred to Radiologist ($N$) | Automated False Negatives (Missed Cases) | Retained Sensitivity | Retained AUROC | Retained Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| **100%** | Weighted BCE (Raw) | 2,436 | 0 (0%) | 16.0 | 0.8118 | 0.8718 | 0.1065 |
| 100% | Weighted BCE (Analytic Corrected) | 2,436 | 0 (0%) | 16.0 | 0.8118 | 0.8718 | 0.0280 |
| 100% | Unweighted BCE (Raw) | 2,436 | 0 (0%) | 14.0 | 0.8353 | 0.8859 | 0.0269 |
| 100% | Random Deferral Baseline | 2,436 | 0 (0%) | 16.0 | -- | -- | 0.1065 |
| **90%** | Weighted BCE (Raw) | 2,193 | 243 (9%) | 13.0 | 0.8395 | 0.8824 | 0.1055 |
| 90% | Weighted BCE (Analytic Corrected) | 2,193 | 243 (9%) | 16.0 | 0.8095 | 0.8800 | 0.0305 |
| 90% | Unweighted BCE (Raw) | 2,193 | 243 (9%) | 14.0 | 0.8205 | 0.8980 | 0.0268 |
| 90% | Random Deferral Baseline | 2,193 | 243 (9%) | 14.5 | -- | -- | 0.1067 |
| **80%** | Weighted BCE (Raw) | 1,949 | 487 (19%) | 9.0 | 0.8767 | 0.8934 | 0.1015 |
| 80% | Weighted BCE (Analytic Corrected) | 1,949 | 487 (19%) | 16.0 | 0.7500 | 0.8943 | 0.0249 |
| 80% | Unweighted BCE (Raw) | 1,949 | 487 (19%) | 14.0 | 0.7705 | 0.9069 | 0.0221 |
| 80% | Random Deferral Baseline | 1,949 | 487 (19%) | 13.0 | -- | -- | 0.1067 |
| **70%** | Weighted BCE (Raw) | 1,706 | 730 (30%) | 7.0 | 0.8889 | 0.9035 | 0.0959 |
| 70% | Weighted BCE (Analytic Corrected) | 1,706 | 730 (30%) | 13.0 | 0.6829 | 0.8869 | 0.0170 |
| 70% | Unweighted BCE (Raw) | 1,706 | 730 (30%) | 11.0 | 0.6944 | 0.8902 | 0.0136 |
| 70% | Random Deferral Baseline | 1,706 | 730 (30%) | 11.3 | -- | -- | 0.1068 |
| **60%** | Weighted BCE (Raw) | 1,462 | 974 (40%) | 6.0 | 0.8966 | 0.9074 | 0.0892 |
| 60% | Weighted BCE (Analytic Corrected) | 1,462 | 974 (40%) | 8.0 | 0.6000 | 0.8210 | 0.0093 |
| 60% | Unweighted BCE (Raw) | 1,462 | 974 (40%) | 8.0 | 0.5000 | 0.8221 | 0.0077 |
| 60% | Random Deferral Baseline | 1,462 | 974 (40%) | 9.6 | -- | -- | 0.1067 |
| **50%** | Weighted BCE (Raw) | 1,218 | 1,218 (50%) | 6.0 | 0.8846 | 0.9146 | 0.0862 |
| 50% | Weighted BCE (Analytic Corrected) | 1,218 | 1,218 (50%) | 6.0 | 0.5000 | 0.7501 | 0.0064 |
| 50% | Unweighted BCE (Raw) | 1,218 | 1,218 (50%) | 7.0 | 0.3000 | 0.8005 | 0.0069 |
| 50% | Random Deferral Baseline | 1,218 | 1,218 (50%) | 8.1 | -- | -- | 0.1064 |

At 70% coverage, 1,706 radiographs are fully automated, while 730 uncertain cases are referred to radiologists. Under the raw weighted model, automated false negatives are cut by more than half (from 16.0 to 7.0) with high diagnostic sensitivity (88.9%) and retained AUROC of 0.9035, outperforming random triage (11.3 missed cases); however, retained probabilities remain uncalibrated (Brier 0.0959). Under analytic correction, probability accuracy is dramatically superior (Brier 0.0170), but 13 false negatives remain automated (sensitivity 68.3%) due to compressed low-probability margins. This exposes a crucial operational trade-off: calibrated models are essential for accurate risk scoring, while raw weighted models with selective triage are superior for recall-critical screening referrals. Neither system achieves zero missed cases.

---

## 4. Discussion

### 4.1 Deconstructing "Temporal Calibration Decay"
Previous studies investigating longitudinal drift in clinical deep learning have frequently reported rapid deterioration in calibration metrics such as Brier score and ECE over deployment years. Our findings explain the mathematical mechanism behind these observations. When practitioners train deep neural networks on highly imbalanced medical datasets, positive loss weighting ($w > 1$) is routinely applied to artificially boost positive detection. However, this introduces an asymptotic $+\log w$ bias into output log-odds.

When these models are deployed across sequential years, disease prevalence fluctuates. The fixed logit bias causes severe, systematic over-prediction. Standard temperature scaling fails because it only multiplies logits by $1/T$ and cannot shift the intercept. Evaluators witnessing worsening Brier scores conclude that the model suffers from irreversible temporal decay.

We demonstrate that this calibration degradation is largely an artifact. Subtracting the known training logit shift ($z - \log w$) eliminates the apparent degradation entirely, producing calibrated probabilities whose Brier score actually improves over time as prevalence decreases.

### 4.2 Guidelines for Medical AI Practitioners
Based on our empirical and mathematical findings, we recommend:
1. **Never use Temperature Scaling alone for models trained with weighted losses:** Temperature scaling cannot adjust log-odds. Always apply Platt scaling or an analytic logit offset ($z - \log w$).
2. **Track Intercept Calibration:** Evaluate calibration slope and intercept separately to detect whether miscalibration is driven by probability bias (intercept) or confidence scaling (slope).
3. **Deploy Selective Triage with Matched Objectives:** Calibrated models should be deployed for probabilistic risk stratification, whereas raw weighted models with selective triage are suited for recall-critical screening where human referral of borderline cases is prioritized. Neither system eliminates missed disease entirely.

---

## 5. Conclusion
This study resolves a major methodological question in clinical AI deployment. We show that severe prospective probability miscalibration in chest radiograph deep learning models is predominantly an uncorrected artifact of positive loss weighting rather than inherent temporal distribution drift. A simple single-parameter offset correction ($z - \log w$) or non-negative Platt scaling completely restores prospective calibration without model retraining. Furthermore, selective prediction with matched decision thresholds reveals a key clinical dichotomy: raw weighted models provide superior recall-oriented triage by halving automated false negatives under a 30% referral budget, whereas analytic calibration restores epidemiologically faithful probabilities for downstream clinical risk assessment. Transparent, workload-aware deployment policies are essential for clinical safety.
