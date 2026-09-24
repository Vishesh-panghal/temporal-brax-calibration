#!/usr/bin/env bash
# ==============================================================================
# MIMIC-CXR-JPG DICOM-ID Level Re-evaluation Pipeline (RTX 8000 Execution)
# ==============================================================================
# This script re-evaluates all 12 frozen BRAX models on MIMIC-CXR-JPG,
# preserving DICOM-ID level granularity to prevent collapsing multiple images
# of the same view position within a study during ensemble aggregation.
#
# It executes the complete end-to-end chain:
# 1. Inference across all 12 model checkpoints with explicit dicom_id export.
# 2. Strict 3-seed probability ensemble with exact configuration matching and
#    assertions (3,403 unique images, exactly 3 distinct seeds per image, 1 label).
# 3. Rigorous paired patient-clustered bootstrap (B=1,000) for both image- and
#    study-level datasets (audit_and_recalibrate_stage4a.py).
# 4. Full regeneration of manuscript Table 5 (reports/manuscript_tables/table5_mimic_external_validation.tex).
# 5. Compilation of MIMIC reliability curves and publication artifacts.
# 6. Automated diff verification against published manuscript Table 5 numbers.
#
# Usage (on ThinkStation P720 / RTX 8000 GPU machine):
#   bash scripts/run_mimic_dicom_rerun.sh [IMAGE_ROOT]
# ==============================================================================

set -euo pipefail

IMAGE_ARG="${1:-}"
if [ -z "$IMAGE_ARG" ] || [ "$IMAGE_ARG" = "/path/to/mimic-cxr-jpg/images" ]; then
    if [ -d "/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg/images" ]; then
        IMAGE_ROOT="/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg/images"
    elif [ -d "/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg" ]; then
        IMAGE_ROOT="/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg"
    elif [ -d "data/mimic-cxr-jpg/images" ]; then
        IMAGE_ROOT="data/mimic-cxr-jpg/images"
    else
        IMAGE_ROOT="/home/poornima/vishesh_gpu/datasets/mimic-cxr-jpg/images"
    fi
else
    IMAGE_ROOT="$IMAGE_ARG"
fi

MANIFEST="data/processed/mimic_stage4a_manifest.csv"

echo "======================================================================"
echo "  MIMIC-CXR-JPG DICOM-ID Level Evaluation & Full Table 5 Regeneration"
echo "======================================================================"
echo "  Image Root : ${IMAGE_ROOT}"
echo "  Manifest   : ${MANIFEST}"
echo "======================================================================"

if [ ! -d "${IMAGE_ROOT}" ]; then
    echo "❌ ERROR: Image directory '${IMAGE_ROOT}' does not exist!"
    echo "   Please specify the valid directory containing MIMIC-CXR images, e.g.:"
    echo "   bash scripts/run_mimic_dicom_rerun.sh /path/to/actual/images"
    exit 1
fi

# Backup existing Table 5 for automated diff comparison
if [ -f "reports/manuscript_tables/table5_mimic_external_validation.tex" ]; then
    cp reports/manuscript_tables/table5_mimic_external_validation.tex reports/manuscript_tables/table5_mimic_external_validation.tex.bak
    echo "  📁 Backed up existing Table 5 to table5_mimic_external_validation.tex.bak"
fi

# Step 1: Run inference across all 12 checkpoints with dicom_id exported
echo -e "\n🚀 [Step 1/5] Running inference across 12 frozen models..."
python3 src/evaluation/evaluate_mimic_stage4a.py \
    --image-root "${IMAGE_ROOT}" \
    --manifest "${MANIFEST}" \
    --batch-size 32 \
    --workers 8

# Step 2: Build 3-seed probability ensemble with strict dicom_id grouping & assertions
echo -e "\n📊 [Step 2/5] Running strict 3-seed probability ensemble (dicom_id-keyed, B=1000)..."
python3 src/evaluation/ensemble_mimic_stage4a.py \
    --pred-dir reports/stage4a/predictions \
    --out-dir reports/stage4a/tables \
    --bootstraps 1000

# Step 3: Run rigorous paired patient-clustered bootstrap for image-level and study-level cohorts
echo -e "\n🔬 [Step 3/5] Running rigorous image-level and study-level aggregation & B=1000 bootstrap..."
python3 scripts/audit_and_recalibrate_stage4a.py

# Step 4: Full regeneration of manuscript Table 5
echo -e "\n📝 [Step 4/5] Regenerating manuscript Table 5 from rigorous image-level evaluation..."
python3 scripts/generate_table5_mimic.py

# Step 5: Recompile publication plots and artifacts
echo -e "\n📦 [Step 5/5] Recompiling Stage 4A reliability curves and summary tables..."
python3 scripts/compile_stage4a_artifacts.py

# Step 6: Automated verification against previous Table 5
echo -e "\n🔍 [Verification] Comparing newly generated Table 5 against baseline..."
if [ -f "reports/manuscript_tables/table5_mimic_external_validation.tex.bak" ]; then
    if diff -u reports/manuscript_tables/table5_mimic_external_validation.tex.bak reports/manuscript_tables/table5_mimic_external_validation.tex; then
        echo "  ✅ PERFECT CONCORDANCE: Newly generated Table 5 matches published numbers bitwise!"
    else
        echo "  ⚠️ NOTE: Numerical differences detected between previous and newly generated Table 5."
        echo "     Review the diff above and update manuscript text if needed."
    fi
    rm -f reports/manuscript_tables/table5_mimic_external_validation.tex.bak
fi

echo -e "\n======================================================================"
echo "  ✅ DICOM-ID RE-EVALUATION & TABLE 5 REGENERATION COMPLETE!"
echo "======================================================================"
