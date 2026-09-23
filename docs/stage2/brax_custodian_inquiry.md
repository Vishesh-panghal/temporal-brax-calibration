# Formal Inquiry: Date Anonymization Semantics in the BRAX Dataset

**Recipient:** PhysioNet / BRAX Dataset Custodians (`brax-dataset@googlegroups.com` or primary correspondence authors of *Nature Scientific Data* 9, 577, 2022)  
**Subject:** Technical Inquiry regarding `StudyDate` de-identification protocol in BRAX v1.1.0  
**Date Drafted:** 22 September 2026  
**Purpose:** Clarify cross-patient chronological comparability for longitudinal clinical evaluation.

---

### Email Draft

**Dear BRAX Dataset Team and Authors,**

First, thank you for curating and openly sharing the BRAX dataset (*Nature Scientific Data*, 2022). It is a vital and exemplary benchmark for chest radiograph analysis in Latin America and global medical AI.

We are preparing a longitudinal evaluation studying calibration drift and selective prediction over time using the BRAX v1.1.0 release (`master_spreadsheet_update.csv`). In our study, we recover the 8-digit calendar dates in `StudyDate` (`YYYYMMDD`, spanning 2008 to 2017) to partition studies into temporally ordered anchor training ($<2015-07$) and prospective deployment cohorts ($T_1$: 2015-H2, $T_2$: 2016, $T_3$: 2017).

To ensure that our statistical reporting and methodological framing adhere to the highest scientific rigor in peer review, we would appreciate clarification on the specific date-shifting mechanism implemented during the de-identification pipeline:

1. **Global vs. Per-Patient Shifting:** Was `StudyDate` shifted by a single **global constant offset** across the entire hospital database, or was an **independent random time offset** sampled for each unique `PatientID` (e.g., within a $\pm 365$-day window)?
2. **Intra-Patient Longitudinal Intervals:** We observe that longitudinal study intervals for individual patients appear preserved. Can you confirm that elapsed days between sequential visits for the same patient are preserved?
3. **Cohort Strata Comparability:** If independent random patient shifts were applied, do the calendar years (e.g., 2015 vs. 2017) still preserve macro-level cohort ordering, or should multi-year groupings be interpreted strictly as anonymized cohort strata rather than literal calendar years?

In our current manuscript, we have adopted the conservative and safe framing: describing $T_1, T_2, T_3$ as *“chronologically stratified deployment cohorts under distribution shift”* rather than attributing shifts to specific external calendar events, and we utilize strict patient-clustered bootstrap inference.

Your clarification will allow us to document the exact data provenance in our methods section.

Thank you very much for your time and continued support of open biomedical research.

Sincerely,  
**Vishesh Panghal and Research Team**  
Temporal Medical AI Research  
