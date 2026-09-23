"""Frozen model evaluation engine across future temporal test bins."""

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BraxDataset
from src.models.architectures import build_model
from src.evaluation.metrics import compute_all_metrics


@torch.no_grad()
def evaluate_frozen_bins(
    checkpoint_path: str,
    manifest_path: str = "data/processed/stage2_manifest.csv",
    output_dir: str = "reports/stage2",
    image_root: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    allow_synthetic: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load image_root from config.yaml if not provided
    if not image_root and os.path.exists("configs/config.yaml"):
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
            image_root = cfg.get("data", {}).get("image_root", "")

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "densenet121")
    targets = checkpoint.get("targets", ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"])
    image_size = checkpoint.get("image_size", 512)
    run_id = checkpoint.get("run_id", f"eval_{architecture}")
    loss_type = checkpoint.get("loss_type", "unknown")
    seed = checkpoint.get("seed", 42)

    print(f"Model: {architecture} | Run ID: {run_id} | Resolution: {image_size}x{image_size} | Loss: {loss_type}")
    print(f"Image Root: '{image_root}'")

    model = build_model(architecture=architecture, num_classes=len(targets), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    full_df = pd.read_csv(manifest_path)
    bins = sorted(full_df[full_df["split"] == "test"]["temporal_bin"].unique())
    print(f"Evaluating across {len(bins)} ordered future bins: {bins}")

    all_metrics_list = []
    all_predictions_list = []

    pred_dir = os.path.join(output_dir, "predictions")
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    for bin_name in bins:
        dataset = BraxDataset(
            manifest_path,
            split="test",
            temporal_bin=bin_name,
            targets=targets,
            image_root=image_root,
            image_size=image_size,
            allow_synthetic=allow_synthetic,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

        bin_logits = []
        bin_probs = []
        bin_targets = []
        bin_patients = []
        bin_dates = []
        bin_views = []

        for images, labels, meta in tqdm(loader, desc=f"Inference {bin_name}", leave=False):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(images)
            else:
                logits = model(images)

            logits_f32 = logits.float()
            probs = torch.sigmoid(logits_f32)

            bin_logits.append(logits_f32.cpu().numpy())
            bin_probs.append(probs.cpu().numpy())
            bin_targets.append(labels.cpu().numpy())
            bin_patients.extend(meta["patient_id"])
            bin_dates.extend(meta["study_date"])
            bin_views.extend(meta["view_position"])

        if len(bin_logits) == 0:
            continue

        bin_logits = np.vstack(bin_logits)
        bin_probs = np.vstack(bin_probs)
        bin_targets = np.vstack(bin_targets)
        n_samples = len(bin_targets)

        # Assemble prediction-level records
        for idx, target_name in enumerate(targets):
            y_t = bin_targets[:, idx]
            y_p = bin_probs[:, idx]
            z = bin_logits[:, idx]

            for i in range(n_samples):
                all_predictions_list.append({
                    "run_id": run_id,
                    "architecture": architecture,
                    "loss_type": loss_type,
                    "seed": seed,
                    "temporal_bin": bin_name,
                    "patient_id": bin_patients[i],
                    "study_date": bin_dates[i],
                    "view_position": bin_views[i],
                    "target": target_name,
                    "y_true": float(y_t[i]),
                    "logit": float(z[i]),
                    "prob_raw": float(y_p[i]),
                })

            metrics = compute_all_metrics(y_t, y_p)
            metrics["run_id"] = run_id
            metrics["architecture"] = architecture
            metrics["loss_type"] = loss_type
            metrics["seed"] = seed
            metrics["temporal_bin"] = bin_name
            metrics["target"] = target_name
            metrics["sample_count"] = n_samples
            all_metrics_list.append(metrics)

    # Save prediction archive
    pred_df = pd.DataFrame(all_predictions_list)
    pred_path = os.path.join(pred_dir, f"preds_{run_id}.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n📦 Saved prediction-level archive ({len(pred_df):,} rows) to: {pred_path}")

    # Save aggregate metrics
    results_df = pd.DataFrame(all_metrics_list)
    out_csv = os.path.join(table_dir, f"metrics_{run_id}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"📊 Saved aggregate metrics to: {out_csv}")

    summary_cols = [c for c in ["temporal_bin", "target", "auroc", "brier_score", "ece_equal_width", "ece_equal_mass"] if c in results_df.columns]
    print(results_df[summary_cols].head(12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint .pth")
    parser.add_argument("--manifest", type=str, default="data/processed/stage2_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="reports/stage2")
    parser.add_argument("--image-root", type=str, default="", help="Root path to images (defaults to config.yaml if empty)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    evaluate_frozen_bins(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        allow_synthetic=args.allow_synthetic,
    )
