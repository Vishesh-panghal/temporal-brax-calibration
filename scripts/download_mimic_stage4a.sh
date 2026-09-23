#!/usr/bin/env bash
# ==============================================================================
# MIMIC-CXR-JPG Stage 4A Test Cohort One-Step Downloader
# ==============================================================================
# Downloads metadata, filters the official test set (AP/PA frontal only, ~5,000 imgs),
# and parallel downloads ONLY those ~5,000 images (~1 GB) instead of the 567 GB ZIP.
#
# Usage:
#   bash scripts/download_mimic_stage4a.sh [DEST_DIR] [PHYSIONET_USER]
#
# Examples:
#   # On GPU server:
#   bash scripts/download_mimic_stage4a.sh /home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg panghal
#
#   # Locally on Mac:
#   bash scripts/download_mimic_stage4a.sh data/mimic panghal
# ==============================================================================

set -euo pipefail

DEST_DIR="${1:-data/mimic}"
PHYSIONET_USER="${2:-panghal}"

echo "======================================================================"
echo "  MIMIC-CXR-JPG Stage 4A Targeted Test Cohort Downloader"
echo "======================================================================"
echo "  Destination Directory : ${DEST_DIR}"
echo "  PhysioNet User        : ${PHYSIONET_USER}"
echo "======================================================================"

META_DIR="${DEST_DIR}/metadata"
IMG_DIR="${DEST_DIR}/images"
mkdir -p "${META_DIR}" "${IMG_DIR}" "data/processed"

if [ -z "${PHYSIONET_PASSWORD:-}" ]; then
    read -rsp "Enter PhysioNet password for '${PHYSIONET_USER}': " PHYSIONET_PASSWORD
    echo
    export PHYSIONET_PASSWORD
fi

# Step 1: Download Metadata Files (~60 MB)
echo -e "\n📥 [Step 1/3] Downloading metadata files from PhysioNet..."
python3 scripts/download_mimic_metadata.py \
    --output-dir "${META_DIR}" \
    --user "${PHYSIONET_USER}"

# Step 2: Build Stage 4A Manifest and selected_IMAGE_FILENAMES.txt
echo -e "\n📊 [Step 2/3] Building Stage 4A manifest & filtering official AP/PA test set..."
python3 scripts/build_mimic_stage4a_manifest.py \
    --meta-dir "${META_DIR}" \
    --output-dir "data/processed"

# Step 3: Parallel Download Selected Test Images (~1 GB, ~5,000 images)
echo -e "\n🚀 [Step 3/3] Downloading selected test images into ${IMG_DIR}..."
python3 scripts/download_mimic_images.py \
    --file-list "data/processed/selected_IMAGE_FILENAMES.txt" \
    --output-dir "${IMG_DIR}" \
    --user "${PHYSIONET_USER}" \
    --workers 16

echo -e "\n======================================================================"
echo "  ✅ STAGE 4A MIMIC DATASET READY FOR EVALUATION!"
echo "  Images saved at: ${IMG_DIR}"
echo "  Manifest saved at: data/processed/mimic_stage4a_manifest.csv"
echo ""
echo "  To run evaluation across all 12 frozen BRAX models:"
echo "    python3 src/evaluation/evaluate_mimic_stage4a.py --image-root ${IMG_DIR}"
echo "======================================================================"
