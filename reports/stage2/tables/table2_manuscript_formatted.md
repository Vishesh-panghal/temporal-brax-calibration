# Table 2: Factorial Experimental Matrix Evaluation Across Chronologically Ordered Deployment Strata

*Performance reported as Mean ± Standard Deviation across three independent training seeds (Seed 42, 123, 2026).*

> **Note on Edema (*):** Positive events in the final deployment stratum ($T_3$) are zero ($N=0$), rendering $T_3$ discrimination non-computable. Edema is retained strictly as exploratory to transparently document support boundaries.

### Discrimination: Area Under the ROC Curve (AUROC ↑)

| Architecture | Training Objective | Target Pathology | $T_1$ (Baseline) | $T_2$ (Intermediate) | $T_3$ (Late Deployment) | Shift $\Delta(T_3 - T_1)$ |
|---|---|---|:---:|:---:|:---:|:---:|
| DenseNet-121 | Unweighted BCE | Pleural Effusion | 0.9335 ± 0.0052 | 0.8895 ± 0.0115 | 0.8731 ± 0.0120 | **-0.0604** |
| DenseNet-121 | Unweighted BCE | Cardiomegaly | 0.9075 ± 0.0116 | 0.8965 ± 0.0140 | 0.9017 ± 0.0170 | **-0.0057** |
| DenseNet-121 | Unweighted BCE | Pneumonia | 0.8473 ± 0.0129 | 0.8818 ± 0.0166 | 0.8647 ± 0.0172 | **+0.0174** |
| DenseNet-121 | Unweighted BCE | Edema* | 0.5402 ± 0.0467 | 0.8640 ± 0.0686 | -- | **--** |
| DenseNet-121 | Pos-Weighted BCE | Pleural Effusion | 0.9061 ± 0.0537 | 0.8676 ± 0.0457 | 0.8391 ± 0.0657 | **-0.0670** |
| DenseNet-121 | Pos-Weighted BCE | Cardiomegaly | 0.8739 ± 0.0540 | 0.8599 ± 0.0600 | 0.8575 ± 0.0699 | **-0.0164** |
| DenseNet-121 | Pos-Weighted BCE | Pneumonia | 0.8181 ± 0.0206 | 0.8494 ± 0.0313 | 0.8424 ± 0.0496 | **+0.0243** |
| DenseNet-121 | Pos-Weighted BCE | Edema* | 0.6122 ± 0.0268 | 0.8753 ± 0.0129 | -- | **--** |
| ResNet-50 | Unweighted BCE | Pleural Effusion | 0.9295 ± 0.0113 | 0.8755 ± 0.0293 | 0.8608 ± 0.0190 | **-0.0687** |
| ResNet-50 | Unweighted BCE | Cardiomegaly | 0.8898 ± 0.0177 | 0.8756 ± 0.0317 | 0.8785 ± 0.0196 | **-0.0112** |
| ResNet-50 | Unweighted BCE | Pneumonia | 0.8432 ± 0.0178 | 0.8659 ± 0.0130 | 0.8798 ± 0.0169 | **+0.0366** |
| ResNet-50 | Unweighted BCE | Edema* | 0.5564 ± 0.0368 | 0.8966 ± 0.0216 | -- | **--** |
| ResNet-50 | Pos-Weighted BCE | Pleural Effusion | 0.8789 ± 0.0508 | 0.8467 ± 0.0458 | 0.8191 ± 0.0496 | **-0.0598** |
| ResNet-50 | Pos-Weighted BCE | Cardiomegaly | 0.8298 ± 0.0550 | 0.8046 ± 0.0701 | 0.7990 ± 0.0788 | **-0.0308** |
| ResNet-50 | Pos-Weighted BCE | Pneumonia | 0.8349 ± 0.0199 | 0.8359 ± 0.0345 | 0.8233 ± 0.0533 | **-0.0116** |
| ResNet-50 | Pos-Weighted BCE | Edema* | 0.5490 ± 0.0852 | 0.8310 ± 0.0437 | -- | **--** |


### Probability Error: Brier Score (Mean Squared Probability Error ↓)

| Architecture | Training Objective | Target Pathology | $T_1$ (Baseline) | $T_2$ (Intermediate) | $T_3$ (Late Deployment) | Shift $\Delta(T_3 - T_1)$ |
|---|---|---|:---:|:---:|:---:|:---:|
| DenseNet-121 | Unweighted BCE | Pleural Effusion | 0.0369 ± 0.0029 | 0.0356 ± 0.0009 | 0.0279 ± 0.0009 | **-0.0090** |
| DenseNet-121 | Unweighted BCE | Cardiomegaly | 0.0532 ± 0.0012 | 0.0677 ± 0.0021 | 0.0677 ± 0.0028 | **+0.0145** |
| DenseNet-121 | Unweighted BCE | Pneumonia | 0.0235 ± 0.0008 | 0.0182 ± 0.0003 | 0.0187 ± 0.0006 | **-0.0048** |
| DenseNet-121 | Unweighted BCE | Edema* | 0.0021 ± 0.0000 | 0.0032 ± 0.0000 | 0.0000 ± 0.0000 | **-0.0021** |
| DenseNet-121 | Pos-Weighted BCE | Pleural Effusion | 0.1081 ± 0.0394 | 0.1287 ± 0.0432 | 0.1254 ± 0.0456 | **+0.0173** |
| DenseNet-121 | Pos-Weighted BCE | Cardiomegaly | 0.1262 ± 0.0247 | 0.1358 ± 0.0316 | 0.1360 ± 0.0306 | **+0.0098** |
| DenseNet-121 | Pos-Weighted BCE | Pneumonia | 0.1313 ± 0.0533 | 0.1308 ± 0.0555 | 0.1298 ± 0.0513 | **-0.0015** |
| DenseNet-121 | Pos-Weighted BCE | Edema* | 0.0614 ± 0.0637 | 0.0672 ± 0.0763 | 0.0636 ± 0.0734 | **+0.0022** |
| ResNet-50 | Unweighted BCE | Pleural Effusion | 0.0372 ± 0.0022 | 0.0364 ± 0.0013 | 0.0284 ± 0.0010 | **-0.0088** |
| ResNet-50 | Unweighted BCE | Cardiomegaly | 0.0572 ± 0.0037 | 0.0709 ± 0.0059 | 0.0712 ± 0.0051 | **+0.0140** |
| ResNet-50 | Unweighted BCE | Pneumonia | 0.0242 ± 0.0007 | 0.0187 ± 0.0003 | 0.0191 ± 0.0003 | **-0.0051** |
| ResNet-50 | Unweighted BCE | Edema* | 0.0021 ± 0.0000 | 0.0032 ± 0.0000 | 0.0000 ± 0.0000 | **-0.0021** |
| ResNet-50 | Pos-Weighted BCE | Pleural Effusion | 0.1256 ± 0.0325 | 0.1417 ± 0.0277 | 0.1392 ± 0.0310 | **+0.0137** |
| ResNet-50 | Pos-Weighted BCE | Cardiomegaly | 0.1415 ± 0.0124 | 0.1526 ± 0.0083 | 0.1522 ± 0.0085 | **+0.0107** |
| ResNet-50 | Pos-Weighted BCE | Pneumonia | 0.1667 ± 0.0584 | 0.1681 ± 0.0583 | 0.1691 ± 0.0613 | **+0.0024** |
| ResNet-50 | Pos-Weighted BCE | Edema* | 0.0693 ± 0.0461 | 0.0749 ± 0.0509 | 0.0746 ± 0.0494 | **+0.0052** |


### Calibration Error: Equal-Mass Expected Calibration Error (ECE_Q10 ↓)

| Architecture | Training Objective | Target Pathology | $T_1$ (Baseline) | $T_2$ (Intermediate) | $T_3$ (Late Deployment) | Shift $\Delta(T_3 - T_1)$ |
|---|---|---|:---:|:---:|:---:|:---:|
| DenseNet-121 | Unweighted BCE | Pleural Effusion | 0.0232 ± 0.0078 | 0.0105 ± 0.0037 | 0.0090 ± 0.0022 | **-0.0143** |
| DenseNet-121 | Unweighted BCE | Cardiomegaly | 0.0140 ± 0.0009 | 0.0236 ± 0.0085 | 0.0210 ± 0.0074 | **+0.0071** |
| DenseNet-121 | Unweighted BCE | Pneumonia | 0.0142 ± 0.0011 | 0.0085 ± 0.0015 | 0.0095 ± 0.0016 | **-0.0046** |
| DenseNet-121 | Unweighted BCE | Edema* | 0.0029 ± 0.0013 | 0.0029 ± 0.0002 | 0.0029 ± 0.0029 | **-0.0000** |
| DenseNet-121 | Pos-Weighted BCE | Pleural Effusion | 0.1964 ± 0.0702 | 0.2224 ± 0.0773 | 0.2227 ± 0.0765 | **+0.0264** |
| DenseNet-121 | Pos-Weighted BCE | Cardiomegaly | 0.2031 ± 0.0507 | 0.1984 ± 0.0530 | 0.1969 ± 0.0486 | **-0.0062** |
| DenseNet-121 | Pos-Weighted BCE | Pneumonia | 0.2393 ± 0.0894 | 0.2443 ± 0.0921 | 0.2425 ± 0.0883 | **+0.0032** |
| DenseNet-121 | Pos-Weighted BCE | Edema* | 0.1191 ± 0.1228 | 0.1278 ± 0.1379 | 0.1274 ± 0.1327 | **+0.0082** |
| ResNet-50 | Unweighted BCE | Pleural Effusion | 0.0162 ± 0.0050 | 0.0101 ± 0.0003 | 0.0132 ± 0.0030 | **-0.0029** |
| ResNet-50 | Unweighted BCE | Cardiomegaly | 0.0233 ± 0.0190 | 0.0181 ± 0.0107 | 0.0175 ± 0.0081 | **-0.0058** |
| ResNet-50 | Unweighted BCE | Pneumonia | 0.0128 ± 0.0050 | 0.0100 ± 0.0008 | 0.0106 ± 0.0015 | **-0.0023** |
| ResNet-50 | Unweighted BCE | Edema* | 0.0023 ± 0.0005 | 0.0023 ± 0.0003 | 0.0017 ± 0.0009 | **-0.0006** |
| ResNet-50 | Pos-Weighted BCE | Pleural Effusion | 0.2481 ± 0.0733 | 0.2712 ± 0.0710 | 0.2751 ± 0.0769 | **+0.0270** |
| ResNet-50 | Pos-Weighted BCE | Cardiomegaly | 0.2450 ± 0.0348 | 0.2357 ± 0.0266 | 0.2338 ± 0.0258 | **-0.0112** |
| ResNet-50 | Pos-Weighted BCE | Pneumonia | 0.3189 ± 0.1175 | 0.3255 ± 0.1178 | 0.3253 ± 0.1208 | **+0.0064** |
| ResNet-50 | Pos-Weighted BCE | Edema* | 0.1545 ± 0.1121 | 0.1639 ± 0.1187 | 0.1670 ± 0.1160 | **+0.0126** |

