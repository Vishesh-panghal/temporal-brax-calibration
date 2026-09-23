# Stage 4A Model Prediction Artifacts (MIMIC-CXR-JPG External Validation)

This directory contains the 12 model prediction archives evaluated on the official frontal MIMIC-CXR-JPG test cohort ($N = 3,403$ images from $3,041$ studies across $289$ patients, evaluating 4 target pathologies = 13,612 predictions per run):

1. `preds_mimic_anchor_densenet121_unweighted_bce_seed123.csv`
2. `preds_mimic_anchor_densenet121_unweighted_bce_seed2026.csv`
3. `preds_mimic_anchor_densenet121_unweighted_bce_seed42.csv`
4. `preds_mimic_anchor_densenet121_weighted_bce_seed123.csv`
5. `preds_mimic_anchor_densenet121_weighted_bce_seed2026.csv`
6. `preds_mimic_anchor_densenet121_weighted_bce_seed42.csv`
7. `preds_mimic_anchor_resnet50_unweighted_bce_seed123.csv`
8. `preds_mimic_anchor_resnet50_unweighted_bce_seed2026.csv`
9. `preds_mimic_anchor_resnet50_unweighted_bce_seed42.csv`
10. `preds_mimic_anchor_resnet50_weighted_bce_seed123.csv`
11. `preds_mimic_anchor_resnet50_weighted_bce_seed2026.csv`
12. `preds_mimic_anchor_resnet50_weighted_bce_seed42.csv`

## PhysioNet Credentialed Data Notice
In compliance with the PhysioNet Credentialed Data Use Agreement for MIMIC-CXR-JPG (v2.1.0), individual patient identifiers and study associations must remain credentialed and are not hosted in the public git repository.

To sync prediction files directly from the compute host:
```bash
rsync -avzP poornima-gpu:~/vishesh_gpu/temporal-BRAX/reports/stage4a/predictions/ reports/stage4a/predictions/
```
