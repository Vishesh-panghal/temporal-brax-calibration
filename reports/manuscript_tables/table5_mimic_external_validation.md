# Table 5: Cross-Institutional External Transport Validation (Official MIMIC-CXR-JPG Frontal Test Cohort)

*External evaluation of 12 frozen BRAX models (DenseNet-121 and ResNet-50 $\times$ unweighted and positive-weighted BCE $\times$ 3 seeds) evaluated on true 3-seed probability ensembles ($\bar{p}_{\text{raw}} = \frac{1}{3}\sum \sigma(z_s), \bar{p}_{\text{corr}} = \frac{1}{3}\sum \sigma(z_s - \log w)$) across all eligible frontal radiographs ($N = 3,403$ images from $3,041$ studies across $289$ patients) in the official patient-partitioned MIMIC-CXR test split. Includes image-level prevalence, empirical null Brier baselines ($\text{Brier}_{\text{null}} = \pi(1-\pi)$), patient-clustered 1,000-replicate bootstrap 95% confidence intervals, Brier Skill Scores ($\text{BSS} = 1 - \text{Brier}/\text{Brier}_{\text{null}}$), and paired analytic recalibration ($\Delta\text{Brier} = \text{Brier}_{\text{corr}} - \text{Brier}_{\text{raw}}$).*

| Architecture | Objective | Pathology | Prevalence ($\pi$) | $\text{Brier}_{\text{null}}$ | AUROC [95% CI] | Brier (Raw $\to$ Corr) | $\Delta\text{Brier}$ [95% CI] | $\text{BSS}$ (Raw $\to$ Corr) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| DenseNet-121 | Unweighted BCE | Pleural Effusion | 32.18% | 0.2182 | 0.688 [0.656, 0.717] | 0.2488 | -- | -0.140 |
| DenseNet-121 | Unweighted BCE | Cardiomegaly | 26.33% | 0.1940 | 0.523 [0.491, 0.556] | 0.2375 | -- | -0.224 |
| DenseNet-121 | Unweighted BCE | Pneumonia | 10.05% | 0.0904 | 0.607 [0.560, 0.650] | 0.0986 | -- | -0.091 |
| DenseNet-121 | Unweighted BCE | Edema | 21.33% | 0.1678 | 0.562 [0.524, 0.596] | 0.2124 | -- | -0.266 |
| DenseNet-121 | Pos-Weighted BCE | Pleural Effusion | 32.18% | 0.2182 | 0.668 [0.636, 0.700] | 0.2536 $\to$ 0.2451 | -0.0084 [-0.0401, +0.0240] | -0.162 $\to$ -0.123 |
| DenseNet-121 | Pos-Weighted BCE | Cardiomegaly | 26.33% | 0.1940 | 0.525 [0.492, 0.558] | 0.2907 $\to$ 0.2328 | **-0.0579** [-0.0790, -0.0349] | -0.499 $\to$ -0.200 |
| DenseNet-121 | Pos-Weighted BCE | Pneumonia | 10.05% | 0.0904 | 0.598 [0.553, 0.639] | 0.1078 $\to$ 0.0983 | **-0.0095** [-0.0170, -0.0016] | -0.192 $\to$ -0.087 |
| DenseNet-121 | Pos-Weighted BCE | Edema | 21.33% | 0.1678 | 0.581 [0.549, 0.611] | 0.1914 $\to$ 0.2131 | +0.0217 [+0.0118, +0.0307] | -0.141 $\to$ -0.270 |
| ResNet-50 | Unweighted BCE | Pleural Effusion | 32.18% | 0.2182 | 0.677 [0.645, 0.709] | 0.2430 | -- | -0.113 |
| ResNet-50 | Unweighted BCE | Cardiomegaly | 26.33% | 0.1940 | 0.519 [0.489, 0.548] | 0.2333 | -- | -0.203 |
| ResNet-50 | Unweighted BCE | Pneumonia | 10.05% | 0.0904 | 0.583 [0.540, 0.623] | 0.0982 | -- | -0.086 |
| ResNet-50 | Unweighted BCE | Edema | 21.33% | 0.1678 | 0.516 [0.480, 0.552] | 0.2126 | -- | -0.266 |
| ResNet-50 | Pos-Weighted BCE | Pleural Effusion | 32.18% | 0.2182 | 0.620 [0.588, 0.653] | 0.2665 $\to$ 0.2557 | -0.0109 [-0.0417, +0.0253] | -0.221 $\to$ -0.172 |
| ResNet-50 | Pos-Weighted BCE | Cardiomegaly | 26.33% | 0.1940 | 0.522 [0.491, 0.552] | 0.2822 $\to$ 0.2211 | **-0.0611** [-0.0818, -0.0415] | -0.455 $\to$ -0.140 |
| ResNet-50 | Pos-Weighted BCE | Pneumonia | 10.05% | 0.0904 | 0.589 [0.545, 0.629] | 0.1137 $\to$ 0.0982 | **-0.0155** [-0.0254, -0.0059] | -0.257 $\to$ -0.086 |
| ResNet-50 | Pos-Weighted BCE | Edema | 21.33% | 0.1678 | 0.570 [0.538, 0.602] | 0.1929 $\to$ 0.2130 | +0.0201 [+0.0092, +0.0313] | -0.149 $\to$ -0.269 |
