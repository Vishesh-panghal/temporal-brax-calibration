#!/usr/bin/env python3
"""
MIMIC-CXR-JPG Stage 4A Manifest Builder.
Filters official test set to frontal projections (AP/PA), directly maps the 4 target
pathologies ('Pleural Effusion', 'Cardiomegaly', 'Pneumonia', 'Edema') with u_zero policy,
and produces:
1. data/processed/mimic_stage4a_manifest.csv (primary CheXpert-labeled test set)
2. data/processed/mimic_stage4a_radiologist_manifest.csv (sensitivity radiologist ground truth)
3. data/processed/selected_IMAGE_FILENAMES.txt (exact list of test images for targeted download)
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TARGET_PATHOLOGIES = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]


def load_csv_or_gz(path: Path) -> pd.DataFrame:
    """Loads a CSV file that may be uncompressed or gzipped."""
    if not path.exists():
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(f"Could not find {path} or {gz_path}")

    print(f"Loading {path.name}...")
    return pd.read_csv(path)


def construct_relative_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """Constructs official MIMIC-CXR-JPG relative path: files/p{sub[:2]}/p{sub}/s{study}/{dicom}.jpg"""
    sub_str = str(int(subject_id))
    p_prefix = f"p{sub_str[:2]}"
    return f"files/{p_prefix}/p{sub_str}/s{int(study_id)}/{dicom_id}.jpg"


def process_chexpert_labels(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """Processes CheXpert labels under the u_zero policy: 1.0 -> 1, 0.0 -> 0, -1.0/NaN -> 0."""
    out_df = df.copy()
    for col in targets:
        if col in out_df.columns:
            # Map 1.0 to 1, everything else (0, -1, NaN) to 0
            out_df[col] = (out_df[col] == 1.0).astype(float)
        else:
            print(f"⚠️ Warning: Target column '{col}' missing from labels dataframe. Filling with 0.0.")
            out_df[col] = 0.0
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Build MIMIC-CXR Stage 4A Test Cohort Manifest.")
    parser.add_argument("--meta-dir", type=str, default="data/mimic/metadata", help="Directory containing downloaded metadata.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save manifests.")
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata and split files
    meta_df = load_csv_or_gz(meta_dir / "mimic-cxr-2.0.0-metadata.csv")
    split_df = load_csv_or_gz(meta_dir / "mimic-cxr-2.0.0-split.csv")
    chexpert_df = load_csv_or_gz(meta_dir / "mimic-cxr-2.0.0-chexpert.csv")

    # Standardize column types
    for df in [meta_df, split_df, chexpert_df]:
        df["subject_id"] = df["subject_id"].astype(int)
        df["study_id"] = df["study_id"].astype(int)
        if "dicom_id" in df.columns:
            df["dicom_id"] = df["dicom_id"].astype(str)

    # 2. Filter split == "test"
    test_split = split_df[split_df["split"] == "test"].copy()
    print(f"\nInitial test split: {len(test_split):,} images across {test_split['subject_id'].nunique():,} patients")

    # 3. Join with metadata to get ViewPosition
    merged_test = pd.merge(
        test_split[["dicom_id", "study_id", "subject_id", "split"]],
        meta_df[["dicom_id", "ViewPosition", "StudyDate", "StudyTime", "Rows", "Columns"]],
        on="dicom_id",
        how="inner",
    )

    # 4. Filter ViewPosition in {"AP", "PA"} (Frontal views only)
    frontal_test = merged_test[merged_test["ViewPosition"].isin(["AP", "PA"])].copy().reset_index(drop=True)
    print(f"Frontal (AP/PA) test images: {len(frontal_test):,} images across {frontal_test['subject_id'].nunique():,} patients")
    print(f"  - AP view: {(frontal_test['ViewPosition'] == 'AP').sum():,}")
    print(f"  - PA view: {(frontal_test['ViewPosition'] == 'PA').sum():,}")

    # 5. Join CheXpert labels (labels are at study_id level)
    chexpert_clean = process_chexpert_labels(chexpert_df, TARGET_PATHOLOGIES)
    keep_cols = ["subject_id", "study_id"] + TARGET_PATHOLOGIES
    primary_manifest = pd.merge(
        frontal_test,
        chexpert_clean[keep_cols],
        on=["subject_id", "study_id"],
        how="left",
    )

    # Construct relative image path
    primary_manifest["relative_path"] = primary_manifest.apply(
        lambda r: construct_relative_path(r["subject_id"], r["study_id"], r["dicom_id"]),
        axis=1,
    )
    primary_manifest["PngPath"] = primary_manifest["relative_path"]  # Compatibility with BRAX loader

    # Save primary manifest
    primary_manifest_path = output_dir / "mimic_stage4a_manifest.csv"
    primary_manifest.to_csv(primary_manifest_path, index=False)
    print(f"\n✅ Saved primary manifest to: {primary_manifest_path} ({len(primary_manifest):,} rows)")

    # 6. Save selected_IMAGE_FILENAMES.txt
    filenames_path = output_dir / "selected_IMAGE_FILENAMES.txt"
    with open(filenames_path, "w") as f:
        for path in sorted(primary_manifest["relative_path"].unique()):
            f.write(path + "\n")
    print(f"✅ Saved selected image list to: {filenames_path} ({len(primary_manifest):,} images)")

    # 7. Check and build radiologist ground truth sensitivity manifest if available
    rad_path = meta_dir / "mimic-cxr-2.1.0-test-set-labeled.csv"
    if rad_path.exists():
        print(f"\nLoading radiologist sensitivity test set: {rad_path.name}...")
        rad_df = pd.read_csv(rad_path)
        print(f"  Radiologist CSV columns: {list(rad_df.columns)}")
        print(f"  Radiologist CSV shape: {rad_df.shape}")

        # This file contains only study_id + 14 CheXpert pathology columns
        # (no subject_id or dicom_id). Join via study_id to frontal_test.
        rad_df["study_id"] = rad_df["study_id"].astype(int)

        # Apply u_zero label policy to radiologist labels
        rad_clean = process_chexpert_labels(rad_df, TARGET_PATHOLOGIES)

        # Suffix the radiologist labels to distinguish from CheXpert labels
        rad_label_cols = {t: f"{t}_rad" for t in TARGET_PATHOLOGIES}
        rad_for_merge = rad_clean[["study_id"] + TARGET_PATHOLOGIES].rename(columns=rad_label_cols)

        rad_manifest = pd.merge(
            frontal_test[["dicom_id", "study_id", "subject_id", "split", "ViewPosition"]],
            rad_for_merge,
            on="study_id",
            how="inner",
        )

        # Rename radiologist columns back to standard target names for evaluation
        rad_manifest = rad_manifest.rename(columns={v: k for k, v in rad_label_cols.items()})

        rad_manifest["relative_path"] = rad_manifest.apply(
            lambda r: construct_relative_path(r["subject_id"], r["study_id"], r["dicom_id"]),
            axis=1,
        )
        rad_manifest["PngPath"] = rad_manifest["relative_path"]

        rad_manifest_path = output_dir / "mimic_stage4a_radiologist_manifest.csv"
        rad_manifest.to_csv(rad_manifest_path, index=False)
        print(f"✅ Saved radiologist sensitivity manifest to: {rad_manifest_path} ({len(rad_manifest):,} rows)")

    # 8. Report cohort statistics & class prevalence
    print("\n" + "=" * 65)
    print("STAGE 4A COHORT PREVALENCE & METRICS SUMMARY")
    print("=" * 65)
    print(f"Total Frontal Test Studies : {primary_manifest['study_id'].nunique():,}")
    print(f"Total Unique Patients      : {primary_manifest['subject_id'].nunique():,}")
    print(f"Total Test Images          : {len(primary_manifest):,}")
    print(f"  - AP Projections         : {(primary_manifest['ViewPosition'] == 'AP').sum():,} ({(primary_manifest['ViewPosition'] == 'AP').mean():.1%})")
    print(f"  - PA Projections         : {(primary_manifest['ViewPosition'] == 'PA').sum():,} ({(primary_manifest['ViewPosition'] == 'PA').mean():.1%})")
    print("\nPathology Prevalence (u_zero policy):")
    for t in TARGET_PATHOLOGIES:
        pos_cnt = int(primary_manifest[t].sum())
        prev = pos_cnt / len(primary_manifest)
        print(f"  {t:20s}: {pos_cnt:5d} / {len(primary_manifest):5d} ({prev:6.2%})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
