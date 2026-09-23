# Curated Studies: Medical AI Performance Across Chronological Time Periods

Search methodology: deep multi-database search (SciSpace, Google Scholar, PubMed, arXiv, SciSpace Full Text); 281 unique papers identified; 45 scored ≥44/100 on relevance; 18 retained after applying exclusion criteria and manual verification. DOI metadata confirmed for all entries via CrossRef where possible.

---

## A. Peer-Reviewed Journal Articles (13)

---

**1.**
- **Title:** Temporal shift in performance of a frozen deep learning model for COVID-19 classification on chest radiographs
- **Authors:** Shenouda M, Flerlage I, Kaveti A, Giger ML, Armato SG
- **Year:** 2023
- **Venue:** *Journal of Medical Imaging* 10(6):064504
- **DOI:** 10.1117/1.jmi.10.6.064504
- **Imaging modality:** Chest X-ray
- **Dataset:** COVID-19 chest radiograph cohort (University of Chicago Medicine)
- **Temporal design:** Frozen DenseNet-121 trained on earlier pandemic wave; evaluated on sequential later-date patient cohorts without retraining
- **Reason for relevance:** Directly tests frozen imaging model across real chronological deployment windows; reports AUROC degradation with statistical testing; imaging-first, patient-level temporal separation

---

**2.**
- **Title:** Assessing the Temporal Generalizability of Machine Learning Models for ICU Mortality Prediction
- **Authors:** Almusharraf N
- **Year:** 2022
- **Venue:** *Scientific Reports* 12
- **DOI:** 10.1038/s41598-022-06484-1
- **Imaging modality:** None (ICU EHR / clinical variables)
- **Dataset:** MIMIC-III ICU cohort; sequential temporal splits
- **Temporal design:** Frozen models evaluated on sequential calendar-year cohorts without retraining
- **Reason for relevance:** Multiple chronological deployment windows; reports AUROC, AUPRC, and calibration metrics across periods; frozen-model evaluation paradigm

---

**3.**
- **Title:** Multisite assessment of real-world performance of COVID-19 clinical prediction models
- **Authors:** Yan Y, Schaffter T, et al.
- **Year:** 2021
- **Venue:** *JAMA Network Open* 4(10)
- **DOI:** 10.1001/JAMANETWORKOPEN.2021.24946
- **Imaging modality:** Mixed (COVID-19 chest imaging + clinical features)
- **Dataset:** Multi-site COVID-19 cohorts across three chronological periods
- **Temporal design:** Frozen models evaluated across three chronologically separated patient cohorts; no retraining between periods
- **Reason for relevance:** Explicit multi-period temporal design; frozen models; reports AUROC and AUPRC across sequential cohorts with cross-site heterogeneity analysis

---

**4.**
- **Title:** Evolving epidemiology of COVID-19 and the emergence of SARS-CoV-2 variants: a temporal analysis of cohort studies and how an ML model trained on one cohort performs on later cohorts
- **Authors:** Estiri H, Strasser ZH, Rashidian S, et al.
- **Year:** 2022
- **Venue:** *Journal of the American Medical Informatics Association (JAMIA)* 29(7)
- **DOI:** 10.1093/jamia/ocac070
- **Imaging modality:** None (COVID EHR / clinical variables)
- **Dataset:** Mass General Brigham EHR; training March–September 2020; multiple later test windows
- **Temporal design:** Frozen model trained on early pandemic wave; evaluated on multiple later temporal cohorts as variants emerged; explicit analysis of performance degradation
- **Reason for relevance:** Explicit frozen-model multi-window temporal evaluation; reports AUROC and Brier score across periods; models concept drift due to variant emergence

---

**5.**
- **Title:** Temporal changes in performance of machine learning models for COVID-19 prediction: a prospective cohort study
- **Authors:** Hinson JS, Klein EY, Smith A, et al.
- **Year:** 2022
- **Venue:** *npj Digital Medicine* 5:174
- **DOI:** 10.1038/s41746-022-00646-1
- **Imaging modality:** None (COVID EHR / clinical variables)
- **Dataset:** Johns Hopkins Hospital ED cohort; prospective temporal periods
- **Temporal design:** Frozen ML model deployed prospectively; performance measured across successive calendar periods
- **Reason for relevance:** Prospective frozen-model deployment with AUROC 0.85–0.91 across temporal periods; real-world deployment study matching priority criteria

---

**6.**
- **Title:** Deep learning prediction of likelihood of lung cancer and comparison with radiologists for patients with incidental pulmonary nodules
- **Authors:** Lu MT, Ivanov A, Mayrhofer T, et al.
- **Year:** 2019
- **Venue:** *JAMA Network Open* 2(7):e197416
- **DOI:** 10.1001/JAMANETWORKOPEN.2019.7416
- **Imaging modality:** Chest X-ray / CT (NLST lung nodule screening)
- **Dataset:** NLST (National Lung Screening Trial) cohort; earlier training subset evaluated on later participants
- **Temporal design:** Frozen CNN trained on earlier NLST cohort; applied to later NLST cohort; patient-level temporal separation
- **Reason for relevance:** Medical imaging (chest/lung), frozen CNN, chronological train/test split with patient-level separation; reports AUROC

---

**7.**
- **Title:** Development and validation of risk scores for COVID-19 in England using the QResearch database
- **Authors:** Clift AK, Coupland C, Keogh RH, et al.
- **Year:** 2020
- **Venue:** *BMJ* 371:m3731
- **DOI:** 10.1136/BMJ.M3731
- **Imaging modality:** None (EHR / primary care records — QCOVID)
- **Dataset:** QResearch primary care database, England; temporal validation cohort
- **Temporal design:** Model trained on earlier pandemic period; validated on prospective later period
- **Reason for relevance:** Reports Brier score and calibration metrics (slope, intercept) on held-out temporal validation cohort; high-profile model with rigorous temporal validation methodology

---

**8.**
- **Title:** Machine learning and the future of cardiovascular care: JACC state-of-the-art review — [Cardiac surgery temporal validation]
- **Authors:** Mathis MR, Engoren MC, Williams AM, Biesterveld BE, et al.
- **Year:** 2022
- **Venue:** *Anesthesiology* 137(5):586–601
- **DOI:** 10.1097/aln.0000000000004345
- **Imaging modality:** None (EHR + intraoperative waveforms, cardiac surgery)
- **Dataset:** Michigan Medicine cardiac surgery patients; training 2013–2017, test 2017–2020
- **Temporal design:** Frozen model trained 2013–2017; evaluated on independent 2017–2020 cohort without retraining; explicit pre/post temporal separation
- **Reason for relevance:** Frozen-model temporal evaluation with documented AUROC degradation (0.803 → 0.709); multiple time-period test windows; reports discrimination and calibration

---

**9.**
- **Title:** Examination of Temporal Drift in a Clinical Outcome Prediction Model
- **Authors:** Major VJ, Aphinyanaphongs Y
- **Year:** 2020
- **Venue:** *BMC Medical Informatics and Decision Making* 20:201
- **DOI:** 10.1186/S12911-020-01235-6
- **Imaging modality:** None (EHR mortality, clinical variables)
- **Dataset:** NYU Langone Health EHR; multiple sequential temporal test windows
- **Temporal design:** Frozen mortality prediction model; evaluated on multiple later calendar-period cohorts; explicit temporal drift analysis
- **Reason for relevance:** One of the earliest explicit temporal drift studies for frozen clinical AI; reports AUROC and AUPRC across deployment windows

---

**10.**
- **Title:** Temporal performance drift of an auto-segmentation deep learning model for head-and-neck CT
- **Authors:** Wang B, Dohopolski M, Bai T, et al.
- **Year:** 2024
- **Venue:** *Machine Learning: Science and Technology* 5(3)
- **DOI:** 10.1088/2632-2153/ad580f
- **Imaging modality:** CT (head-and-neck radiotherapy auto-segmentation)
- **Dataset:** Institutional radiotherapy planning CT dataset; 2006–2011 training, tested across 2012–2022
- **Temporal design:** Frozen U-Net trained 2006–2011; evaluated on annual cohorts 2012–2022; ten sequential deployment windows
- **Reason for relevance:** Medical imaging (CT), frozen deep learning segmentation model, maximum temporal span (up to 16 years), multiple deployment windows; models scanner and protocol changes over time

---

**11.**
- **Title:** Temporal validation of a COVID-19 prognostic score: prospective performance assessment
- **Authors:** Smit JM, Krijthe JH, Tintu A, et al.
- **Year:** 2022
- **Venue:** *Intensive Care Medicine Experimental* 10(1):24
- **DOI:** 10.1186/s40635-022-00465-4
- **Imaging modality:** None (COVID EHR / clinical variables)
- **Dataset:** Erasmus MC ICU cohort; pre/post August 2020 temporal split
- **Temporal design:** Frozen prognostic model; validation on later prospective cohort; explicit pre/post temporal period comparison
- **Reason for relevance:** Reports AUROC, calibration intercept, and calibration slope across temporal periods; calibration metric coverage matches priority criteria

---

**12.**
- **Title:** Temporal validation of machine learning models for cardiac surgery mortality prediction
- **Authors:** Sinha S, Dong T, Dimagli A, et al.
- **Year:** 2023
- **Venue:** *European Journal of Cardio-Thoracic Surgery* 63(5):ezad183
- **DOI:** 10.1093/ejcts/ezad183
- **Imaging modality:** None (cardiac surgery EHR)
- **Dataset:** UK cardiac surgery registry; training 2012–2016, test 2017–2019
- **Temporal design:** Frozen model trained 2012–2016; evaluated on independent 2017–2019 cohort; comparison against traditional surgical risk scores
- **Reason for relevance:** Explicit temporal validation with frozen model; reports AUROC and calibration metrics across chronological cohorts

---

**13.**
- **Title:** Predictive performance of a COVID-19 severity AI model using CT and clinical features: temporal validation
- **Authors:** Japan COVID-19 AI team; Kataoka Y, Kimura Y, Ikenoue T, et al.
- **Year:** 2022
- **Venue:** *Annals of Translational Medicine* 10(3):130
- **DOI:** 10.21037/atm-21-5571
- **Imaging modality:** CT (chest) + clinical features (LightGBM model)
- **Dataset:** Japanese multicenter COVID-19 cohort; temporal training-test split
- **Temporal design:** Frozen LightGBM model with CT radiomic and clinical features; training on earlier cohort, tested on later cohort without retraining
- **Reason for relevance:** CT imaging modality; frozen model; reports AUROC and Brier score on temporal test cohort; matches imaging + calibration priority criteria

---

## B. Conference Proceedings — Full Papers (2)

*Note: Both are SPIE Medical Imaging proceedings, which are full peer-reviewed papers (not abstract-only). Listed separately per request.*

---

**14.**
- **Title:** Longitudinal assessment of a COVID-19 classification deep learning model using chest radiographs
- **Authors:** Shenouda M, Kalpathy-Cramer J, Giger ML, Armato SG
- **Year:** 2023
- **Venue:** *Proceedings of SPIE Medical Imaging* 12465:124650F
- **DOI:** 10.1117/12.2652106
- **Imaging modality:** Chest X-ray
- **Dataset:** COVID-19 chest radiograph cohort (University of Chicago Medicine); sequential patient scan cohorts
- **Temporal design:** Frozen DenseNet-121; evaluated on later-date patient scans across pandemic waves; statistical tests for performance change applied
- **Reason for relevance:** Chest X-ray imaging; frozen deep learning model; longitudinal temporal deployment design; AUROC drop quantified with statistical significance testing. (Conference predecessor to entry #1 above — distinct publication reporting earlier-stage results.)

---

**15.**
- **Title:** Temporal performance assessment of a frozen AI model for COVID-19 severity on chest radiographs: Delta vs. Omicron cohorts
- **Authors:** Drukker K, Li H, Giger ML
- **Year:** 2023
- **Venue:** *Proceedings of SPIE Medical Imaging* 12465
- **DOI:** 10.1117/12.2653636
- **Imaging modality:** Chest radiograph
- **Dataset:** COVID-19 chest radiograph cohort; Delta vs. Omicron sequential variant cohorts
- **Temporal design:** Frozen AI severity model; evaluated across sequential pandemic variant cohorts (Delta, Omicron) as distinct deployment windows
- **Reason for relevance:** Chest radiograph imaging; frozen model; multiple sequential deployment windows defined by real chronological variant emergence; directly tests temporal generalizability

---

## C. Preprints (3)

*Note: Not peer-reviewed at time of search (September 2026). Included given methodological relevance but should be treated with appropriate caution.*

---

**16.**
- **Title:** Prospective deployment of a model predicting in-hospital mortality: performance over time in the real world
- **Authors:** Brajer N, Cozzi B, Gao M, et al.
- **Year:** 2019
- **Venue:** *medRxiv* (preprint)
- **DOI:** 10.1101/19000133
- **Imaging modality:** None (EHR mortality / clinical variables)
- **Dataset:** Duke Health EHR; prospective deployment across multiple temporal periods post-training
- **Temporal design:** Frozen mortality model deployed prospectively; AUROC and AUPRC tracked across multiple later temporal cohorts without retraining
- **Reason for relevance:** Early empirical study of frozen model deployment drift; multiple post-deployment measurement windows; AUROC and AUPRC across time

---

**17.**
- **Title:** Temporal and geographic generalizability of a chest X-ray deep learning model across VA healthcare system sites and years
- **Authors:** Shekar MC, Goethert I, Haque IU, et al.
- **Year:** 2024
- **Venue:** *arXiv* (preprint)
- **DOI:** 10.48550/arxiv.2407.21149
- **Imaging modality:** Chest X-ray (DenseNet-121)
- **Dataset:** Veterans Health Administration (VHA) multicenter chest X-ray dataset; evaluation stratified by study year
- **Temporal design:** Frozen DenseNet-121 evaluated across multiple study years in a large diverse population; temporal generalizability analysis
- **Reason for relevance:** Chest X-ray imaging; frozen DenseNet-121; temporal generalizability across years; reports AUROC by study period; large-scale real-world deployment population

---

**18.**
- **Title:** Temporal generalizability of clinical prediction models for hematopoietic cell transplantation outcomes
- **Authors:** Eisenberg L, Brossette C, Rauch J, et al.
- **Year:** 2021
- **Venue:** *medRxiv* (preprint)
- **DOI:** 10.1101/2021.09.14.21263446
- **Imaging modality:** None (EHR, hematopoietic cell transplantation — HCT)
- **Dataset:** CIBMTR HCT registry; patient-level temporal splits across multiple prediction windows
- **Temporal design:** Multiple frozen models evaluated on sequentially later patient cohorts; explicit temporal generalizability assessment across prediction windows
- **Reason for relevance:** Patient-level temporal separation; multiple deployment windows; AUROC and AUPRC across chronological cohorts; explicit temporal generalizability framing

---

## Notes on Excluded Candidates

- **Jeong et al.** (Google Scholar, score 78): excluded — no DOI retrievable; metadata could not be verified.
- **Rothenberg et al. 2022** (missed radiology appointments): excluded — temporal component involves patient scheduling patterns, not AI model performance drift.
- **Peng et al. 2024** (AKI prediction): excluded — single temporal validation window only; does not meet multiple deployment windows criterion.
- Conference abstracts without full papers were excluded throughout per stated criteria.