#!/usr/bin/env python3
"""
MIMIC-CXR-JPG Stage 4A Test Cohort Formal Audit.
Verifies complete physical integrity and extracts exact cohort statistics
(images, studies, patients, AP/PA projection counts, disease prevalence, null Brier).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

TARGETS = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]


def audit_cohort(manifest_path: str, image_root_path: str = None):
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        print(f"❌ Manifest not found at: {manifest_p}")
        sys.exit(1)

    df = pd.read_csv(manifest_p)

    # Resolve image root
    possible_roots = [
        Path(image_root_path) if image_root_path else None,
        Path("/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg/images"),
        Path("data/mimic/images"),
        Path("../data/mimic/images"),
    ]
    resolved_root = None
    for r in possible_roots:
        if r and r.exists():
            resolved_root = r
            break

    print("=" * 68)
    print("STAGE 4A EXTERNAL VALIDATION COHORT FORMAL AUDIT")
    print("=" * 68)
    n_images = len(df)
    n_studies = df["study_id"].nunique()
    n_patients = df["subject_id"].nunique()
    print(f"Total Frontal Radiographs : {n_images:,}")
    print(f"Total Diagnostic Studies   : {n_studies:,}")
    print(f"Total Unique Patients      : {n_patients:,}")

    # Projections
    print("\nView Position Breakdown:")
    view_counts = df["ViewPosition"].value_counts()
    for view, cnt in view_counts.items():
        print(f"  {view:4s}: {cnt:5d} ({cnt/n_images:6.2%})")

    # Pathologies & Null Brier
    print("\nPathology Prevalence & Null Baselines (u_zero policy):")
    print(f"  {'Pathology':20s} {'Positives':>10s} {'Prevalence (π)':>16s} {'Brier_null [π(1-π)]':>22s}")
    print("  " + "-" * 72)
    stats_records = []
    for tgt in TARGETS:
        if tgt in df.columns:
            pos = int(df[tgt].sum())
            prev = pos / n_images
            b_null = prev * (1.0 - prev)
            print(f"  {tgt:20s} {pos:10d} {prev:16.2%} {b_null:22.4f}")
            stats_records.append({
                "Target": tgt,
                "Positives": pos,
                "Prevalence": prev,
                "Brier_null": b_null,
            })

    # Physical file verification
    print("\nPhysical Disk Verification:")
    if resolved_root:
        print(f"  Image root directory: {resolved_root.resolve()}")
        missing = [p for p in df["relative_path"] if not (resolved_root / p).is_file()]
        print(f"  Missing files       : {len(missing)}")
        completeness = 1.0 - (len(missing) / n_images)
        print(f"  Completeness        : {completeness:6.2%}")
        if len(missing) == 0:
            print("  ✅ 100.00% physical integrity verified! Zero missing images.")
        else:
            print(f"  ⚠️ Warning: {len(missing)} images could not be located.")
    else:
        print("  ⚠️ Image root directory not found on local path. Skipping physical check.")

    print("=" * 68)

    # Generate Methods Text Snippet
    print("\n📝 Manuscript Text Snippet (for Methods Section):")
    print("-" * 68)
    ap_cnt = view_counts.get("AP", 0)
    pa_cnt = view_counts.get("PA", 0)
    methods_text = (
        f"\"To assess cross-institutional transportability under realistic clinical domain shift, "
        f"frozen BRAX models were externally validated on all eligible frontal radiographs "
        f"from the official patient-partitioned MIMIC-CXR-JPG test split (N = {n_images:,} images, "
        f"{n_studies:,} studies, across {n_patients:,} unique patients; "
        f"AP: {ap_cnt:,} [{ap_cnt/n_images:.1%}], PA: {pa_cnt:,} [{pa_cnt/n_images:.1%}]). "
        f"Diagnostic labels were extracted using the official CheXpert labeler with an uncertainty zero "
        f"(U-zero) policy. Prevalence varied by finding: Pleural Effusion ({stats_records[0]['Prevalence']:.1%}, N={stats_records[0]['Positives']}), "
        f"Cardiomegaly ({stats_records[1]['Prevalence']:.1%}, N={stats_records[1]['Positives']}), "
        f"Pneumonia ({stats_records[2]['Prevalence']:.1%}, N={stats_records[2]['Positives']}), "
        f"and Edema ({stats_records[3]['Prevalence']:.1%}, N={stats_records[3]['Positives']}).\""
    )
    print(methods_text)
    print("-" * 68 + "\n")


if __name__ == "__main__":
    manifest_arg = sys.argv[1] if len(sys.argv) > 1 else "data/processed/mimic_stage4a_manifest.csv"
    img_arg = sys.argv[2] if len(sys.argv) > 2 else None
    audit_cohort(manifest_arg, img_arg)
