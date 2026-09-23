#!/bin/bash
# ==============================================================================
# STAGE 2 — 12-Run Core Training Matrix on Quadro RTX 8000
#
# Design:
#   2 Architectures: DenseNet-121, ResNet-50
#   2 Objectives:    Positive-Weighted BCE, Standard Unweighted BCE
#   3 Seeds:         42, 123, 2026
#
# Batch size: 64 (512x512 resolution) with mixed-precision AMP fp16
# Expected compute: ~1.5 hours per run on Quadro RTX 8000 (~18 hours total)
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/stage2"
mkdir -p "$LOG_DIR" checkpoints reports/stage2/predictions reports/stage2/tables

ARCHITECTURES=("densenet121" "resnet50")
LOSS_TYPES=("weighted_bce" "unweighted_bce")
SEEDS=(42 123 2026)

TOTAL_RUNS=$(( ${#ARCHITECTURES[@]} * ${#LOSS_TYPES[@]} * ${#SEEDS[@]} ))
CURRENT_RUN=0

echo "=============================================================================="
echo "  LAUNCHING STAGE 2 CORE EXPERIMENTAL MATRIX ($TOTAL_RUNS RUNS TOTAL)"
echo "=============================================================================="
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo ""

for ARCH in "${ARCHITECTURES[@]}"; do
  for LOSS in "${LOSS_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      CURRENT_RUN=$((CURRENT_RUN + 1))
      RUN_ID="anchor_${ARCH}_${LOSS}_seed${SEED}"
      LOG_FILE="$LOG_DIR/${RUN_ID}.log"
      CHECKPOINT="checkpoints/${RUN_ID}.pth"

      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "  [$CURRENT_RUN/$TOTAL_RUNS] Architecture: $ARCH | Loss: $LOSS | Seed: $SEED"
      echo "  Run ID: $RUN_ID"
      echo "  Log:    $LOG_FILE"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

      # 1. Train Anchor Model
      if [ -f "$CHECKPOINT" ]; then
        echo "  ℹ Checkpoint $CHECKPOINT already exists. Skipping training."
      else
        echo "  🚀 Starting training at $(date)..."
        PYTHONPATH=. python3 src/training/train_anchor.py \
          --arch "$ARCH" \
          --loss "$LOSS" \
          --seed "$SEED" \
          --epochs 15 \
          --batch-size 64 \
          --image-size 512 \
          --workers 8 \
          2>&1 | tee "$LOG_FILE"
        echo "  ✓ Training completed at $(date)"
      fi

      # 2. Run Frozen Evaluation & Export Prediction Archive
      PRED_ARCHIVE="reports/stage2/predictions/preds_${RUN_ID}.csv"
      if [ -f "$PRED_ARCHIVE" ]; then
        echo "  ℹ Predictions for $RUN_ID already exported."
      else
        echo "  📊 Generating frozen predictions and metric suite..."
        PYTHONPATH=. python3 src/evaluation/predict_frozen.py \
          --checkpoint "$CHECKPOINT" \
          --manifest "data/processed/stage2_manifest.csv" \
          --output-dir "reports/stage2" \
          --batch-size 64 \
          --workers 8 \
          2>&1 | tee -a "$LOG_FILE"
        echo "  ✓ Evaluation & archiving completed."
      fi

      # 3. Post-Hoc Mitigation & Selective Abstention
      RECALIB_FILE="reports/stage2/mitigation/recalibration_results_${RUN_ID}.csv"
      if [ -f "$RECALIB_FILE" ]; then
        echo "  ℹ Mitigation for $RUN_ID already evaluated."
      else
        echo "  🛡 Running post-hoc temperature scaling & selective abstention..."
        PYTHONPATH=. python3 src/evaluation/mitigation.py \
          --checkpoint "$CHECKPOINT" \
          --manifest "data/processed/stage2_manifest.csv" \
          --output-dir "reports/stage2/mitigation" \
          --batch-size 64 \
          --workers 8 \
          2>&1 | tee -a "$LOG_FILE"
        echo "  ✓ Mitigation evaluation completed."
      fi

      # 4. Patient-Cluster Bootstrap TDI Analysis
      TDI_FILE="reports/stage2/tables/tdi_${RUN_ID}.csv"
      if [ -f "$TDI_FILE" ]; then
        echo "  ℹ TDI inference for $RUN_ID already completed."
      else
        echo "  📐 Running 2,000-replicate patient-cluster bootstrap TDI..."
        PYTHONPATH=. python3 src/evaluation/tdi.py \
          --preds "$PRED_ARCHIVE" \
          --output-dir "reports/stage2" \
          --bootstraps 2000 \
          2>&1 | tee -a "$LOG_FILE"
        echo "  ✓ TDI inference completed."
      fi

      echo ""
    done
  done
done

echo "=============================================================================="
echo "  ✅ ALL $TOTAL_RUNS STAGE 2 RUNS AND PREDICTION ARCHIVES COMPLETE!"
echo "  End time: $(date)"
echo "=============================================================================="
