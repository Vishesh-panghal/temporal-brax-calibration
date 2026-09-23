"""Deterministic Stage 2 cohort splitting protocol with zero patient and zero temporal leakage.

Guarantees:
1. Patient Isolation: PatientID(train) ∩ PatientID(val) ∩ PatientID(test) = ∅
2. Temporal Isolation (Strict Model-Freeze):
   max(StudyDate_train) < min(StudyDate_val) <= max(StudyDate_val) < min(StudyDate_test)
   - Train: Earliest to 2013-08-31 (~61.3% of cohort)
   - Val:   2013-09-01 to 2015-06-30 (~17.3% of cohort)
   - Test:  2015-07-01 to 2017-12-17 (~20.3% of cohort)
3. Prospective Test Stratification:
   Test is partitioned into ordered temporal bins:
   - T1_2015: 2015 (H2 2015, immediate deployment window)
   - T2_2016: 2016 (Year 1 deployment window)
   - T3_2017: 2017 (Year 2 deployment window)
4. Boundary Straddlers:
   Patients whose studies cross the temporal boundaries are isolated and excluded from test
   to prevent longitudinal leakage.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
from src.data.bins import recover_calendar_dates

TARGETS = ['Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema']


def build_stage2_splits(
    raw_manifest_path: str = "data/processed/brax_temporal_manifest_updated.csv",
    output_manifest_path: str = "data/processed/stage2_manifest.csv",
    output_audit_path: str = "reports/stage2/cohort_audit.csv",
    t_freeze_str: str = "2015-07-01",
    t_val_split_str: str = "2013-09-01",
):
    print(f"Loading master manifest: {raw_manifest_path}...")
    df = pd.read_csv(raw_manifest_path)
    df['calendar_date'] = recover_calendar_dates(df)
    df['year'] = df['calendar_date'].dt.year

    t_freeze = pd.Timestamp(t_freeze_str)
    t_val = pd.Timestamp(t_val_split_str)

    # Compute patient-level study date spans
    p_spans = df.groupby('PatientID')['calendar_date'].agg(min_date='min', max_date='max')

    # Test patients: all studies strictly on or after t_freeze
    test_patients = set(p_spans[p_spans['min_date'] >= t_freeze].index)

    # Dev patients: all studies strictly before t_freeze
    dev_patients = set(p_spans[p_spans['max_date'] < t_freeze].index)
    dev_spans = p_spans.loc[list(dev_patients)]

    # Split Dev into Train and Val using t_val
    train_patients = set(dev_spans[dev_spans['max_date'] < t_val].index)
    val_patients = set(dev_spans[dev_spans['min_date'] >= t_val].index)

    # Straddle patients (span across t_val or t_freeze)
    straddle_freeze = set(p_spans.index) - test_patients - dev_patients
    straddle_val = dev_patients - train_patients - val_patients
    all_straddle = straddle_freeze | straddle_val

    print(f"Clean Train patients: {len(train_patients):,}")
    print(f"Clean Val patients:   {len(val_patients):,}")
    print(f"Clean Test patients:  {len(test_patients):,}")
    print(f"Boundary straddling patients: {len(all_straddle):,}")

    # Verify zero patient overlap
    assert len(train_patients & val_patients) == 0, "Leakage: train & val overlap!"
    assert len(train_patients & test_patients) == 0, "Leakage: train & test overlap!"
    assert len(val_patients & test_patients) == 0, "Leakage: val & test overlap!"

    # Assign split column
    def assign_split(pid):
        if pid in train_patients:
            return "train"
        elif pid in val_patients:
            return "val"
        elif pid in test_patients:
            return "test"
        else:
            return "excluded_straddle"

    df['split'] = df['PatientID'].map(assign_split)

    # Assign temporal bins within the prospective test cohort
    def assign_temporal_bin(row):
        if row['split'] != 'test':
            return 'anchor_dev'
        yr = row['year']
        if yr <= 2015:
            return 'T1_2015'
        elif yr == 2016:
            return 'T2_2016'
        else:
            return 'T3_2017'

    df['temporal_bin'] = df.apply(assign_temporal_bin, axis=1)

    # Filter to active research splits
    active_df = df[df['split'].isin(['train', 'val', 'test'])].copy()

    # Verify strict temporal ordering
    train_max = active_df[active_df['split'] == 'train']['calendar_date'].max()
    val_min = active_df[active_df['split'] == 'val']['calendar_date'].min()
    val_max = active_df[active_df['split'] == 'val']['calendar_date'].max()
    test_min = active_df[active_df['split'] == 'test']['calendar_date'].min()

    print("\n--- Strict Temporal Isolation Verification ---")
    print(f"Train date span: {active_df[active_df['split'] == 'train']['calendar_date'].min().date()} to {train_max.date()}")
    print(f"Val date span:   {val_min.date()} to {val_max.date()}")
    print(f"Test date span:  {test_min.date()} to {active_df[active_df['split'] == 'test']['calendar_date'].max().date()}")

    assert train_max < val_min, f"Train-Val temporal leakage: {train_max} >= {val_min}"
    assert val_max < test_min, f"Val-Test temporal leakage: {val_max} >= {test_min}"
    print("✅ Temporal isolation verified: max(Train) < min(Val) <= max(Val) < min(Test) strictly holds!")

    # Summary table
    records = []
    for s in ['train', 'val', 'test']:
        sub = active_df[active_df['split'] == s]
        row_dict = {
            "split": s,
            "studies": len(sub),
            "study_pct": f"{len(sub) / len(df) * 100:.1f}%",
            "unique_patients": sub['PatientID'].nunique(),
            "min_date": str(sub['calendar_date'].min().date()),
            "max_date": str(sub['calendar_date'].max().date()),
        }
        for t in TARGETS:
            pos = int((sub[t] == 1.0).sum())
            row_dict[f"{t}_pos"] = pos
            row_dict[f"{t}_prev_pct"] = f"{pos / len(sub) * 100:.2f}%"
        records.append(row_dict)

    # Test bins summary
    for b in ['T1_2015', 'T2_2016', 'T3_2017']:
        sub = active_df[active_df['temporal_bin'] == b]
        row_dict = {
            "split": f"test_{b}",
            "studies": len(sub),
            "study_pct": f"{len(sub) / len(df) * 100:.1f}%",
            "unique_patients": sub['PatientID'].nunique(),
            "min_date": str(sub['calendar_date'].min().date()),
            "max_date": str(sub['calendar_date'].max().date()),
        }
        for t in TARGETS:
            pos = int((sub[t] == 1.0).sum())
            row_dict[f"{t}_pos"] = pos
            row_dict[f"{t}_prev_pct"] = f"{pos / len(sub) * 100:.2f}%"
        records.append(row_dict)

    audit_df = pd.DataFrame(records)
    print("\n--- Cohort Distribution & Prevalence Table ---")
    cols_to_print = ["split", "studies", "study_pct", "unique_patients", "min_date", "max_date", "Pleural Effusion_prev_pct", "Cardiomegaly_prev_pct"]
    print(audit_df[cols_to_print])

    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_audit_path), exist_ok=True)

    df.to_csv(output_manifest_path, index=False)
    audit_df.to_csv(output_audit_path, index=False)
    print(f"\nSaved stage2 manifest to: {output_manifest_path}")
    print(f"Saved cohort audit to: {output_audit_path}")


if __name__ == "__main__":
    build_stage2_splits()
