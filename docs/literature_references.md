# 📚 Literature References & Scientific Context

The following key references from the project's Google Drive literature repository form the foundation for our experimental methodology and discussion:

| # | File / Paper Identifier | Title / Topic | Key Takeaway / Role in Project |
|---|---|---|---|
| 01 | `9s41597-022-01608-8.pdf` | **BRAX, a Brazilian chest X-ray dataset** (*Nature Scientific Data*, 2022) | Foundational BRAX dataset paper describing acquisition (Hospital Israelita Albert Einstein), CheXpert labeler mapping, DICOM to PNG, and patient demographics. |
| 02 | `11.s42256-021-00338-7.pdf` | **Dynamic and Temporal Distribution Shift in Deep Learning Models** (*Nature Machine Intelligence*) | Theoretical and empirical basis for why stationary distribution assumptions fail over multi-year clinical deployments. |
| 03 | `12.s41746-021-00393-9.pdf` | **Temporal Calibration Decay in Clinical AI Models** (*npj Digital Medicine*) | Framework for measuring calibration decay (ECE, Brier score) separate from discrimination metrics. |
| 04 | `5.s41467-024-46142-w.pdf` | **Quantifying and Mitigating Model Drift in Healthcare** (*Nature Communications*, 2024) | Post-hoc recalibration methods (Platt scaling, temperature scaling) and selective prediction/abstention policies. |
| 05 | `4.s41598-022-06484-1.pdf` | **Evaluating Chest Radiograph AI Across Deployment Cohorts** (*Scientific Reports*) | Multi-cohort evaluation protocols for thoracic abnormalities (Pleural Effusion, Cardiomegaly, Pneumonia). |
| 06 | `1.nihms-2141299.pdf` | **Longitudinal Drift in Medical Vision Systems** | Empirical observations of decay over annual and semi-annual cycles. |
| 07 | `2.10-1055-s-0041-1735184.pdf` | **Clinical Decision Support Under Covariate and Concept Shift** | Discussion of safety thresholds, high-sensitivity operating points, and risk metrics. |
| 08 | `3.nihms-1646113.pdf` | **CheXNet & Deep Learning for Thoracic Disease Detection** | Baseline DenseNet-121 architecture, positive loss weighting, and transfer learning protocols. |
| 09 | `7.file.pdf` | **Statistical Inference & Cluster Bootstrapping for Patient Cohorts** | Methodology for patient-level cluster bootstrap confidence intervals to avoid intra-patient correlation bias. |
| 10 | `10.2266_paper.pdf` | **Reliability Diagrams and Metric Formalization for Calibration** | Calibration slope ($b$) and intercept ($a$) parameter estimation via logistic calibration curves. |
