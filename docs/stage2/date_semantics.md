# BRAX Date Semantics & Chronological Framing Analysis

**Date:** 21 September 2026  
**Reference:** Work Package 2A / Decision Gate 2A

---

## 1. Context & Metadata Representation

In BRAX v1.1.0 (`master_spreadsheet_update.csv`), the primary temporal field is `StudyDate`, represented as an 8-digit integer:
- Format: `YYYYMMDD` (e.g., `20101129`).
- Derived timeline in Stage 1: 2008 to 2017.

### Recovery Mechanism
Naively parsing `StudyDate` with `pd.to_datetime(df['StudyDate'])` erroneously interprets integer timestamps as nanoseconds since epoch, placing all records in January 1970. The string regex extraction in `src/data/bins.py`:
```python
raw = df['StudyDate'].astype(str)
extracted = raw.str.extract(r'(\d{8})$')[0]
return pd.to_datetime(extracted, format='%Y%m%d', errors='coerce')
```
correctly recovers the released integer calendar strings.

---

## 2. Dataset De-identification Policies

PhysioNet de-identification guidelines (HIPAA Safe Harbor) require that direct identifiers and specific dates be modified. Typically:
1. **Intra-Patient Intervals:** The elapsed time between repeated studies for a given patient is strictly preserved (e.g., Study 2 occurring 45 days after Study 1).
2. **Cross-Patient Anchors:** Depending on the anonymization pipeline, dates may be:
   - Shifted by a single global constant across the entire database, OR
   - Shifted by an independent random offset per patient (e.g., random $\Delta t \in [-365, +365]$ days).

---

## 3. Methodological Decision Gate (2A Resolution)

To maintain unimpeachable peer-review standards for **IEEE-JBHI**:

| Scenario | Interpretation | Manuscript Framing Policy |
|:---|:---|:---|
| **Conservative / Standard Clinical AI Position** | Exact calendar years may reflect anonymized cohort strata rather than verified external calendar years. | Frame $T_1, T_2, T_3$ as **"ordered cohort strata under longitudinal shift"** rather than claiming specific macro-environmental effects (e.g., external epidemics or specific scanner replacement dates). |
| **Within-Patient Validity** | Longitudinal intervals are preserved. | Cluster bootstrapping must strictly resample at the patient level to account for clustered study intervals. |

### Conclusion for Stage 2 Manuscript
We will avoid overclaiming "calendar year drift" and instead explicitly describe the evaluation as:
> *"An ordered longitudinal evaluation across chronologically stratified deployment cohorts ($T_1, T_2, T_3$), isolating the effects of distribution shift and model calibration erosion over time."*

This framing is scientifically airtight and prevents any reviewer from rejecting the paper based on date anonymization nuances.
