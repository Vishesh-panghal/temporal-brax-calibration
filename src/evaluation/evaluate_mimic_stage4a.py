"""
Evaluation engine for frozen BRAX models on MIMIC-CXR-JPG Stage 4A test cohort.
Evaluates 12 checkpoints across:
- Unweighted raw
- Weighted raw
- Weighted analytic correction: sigmoid(z - log w_BRAX)
Computes discrimination (AUROC, AUPRC) and calibration (Brier score, log-loss, slope, intercept, ECE).
Stratifies by All Frontal, AP view, and PA view.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mimic_dataset import MimicCxrDataset
from src.models.architectures import build_model
from src.evaluation.metrics import compute_all_metrics

DEFAULT_TARGETS = ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"]


@torch.no_grad()
def evaluate_model_on_mimic(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    targets: list,
    pos_weights: np.ndarray = None,
) -> tuple:
    """
    Runs inference on a DataLoader.
    Returns raw logits, raw probs, corrected probs (if pos_weights given), targets, and metadata.
    """
    model.eval()
    all_logits = []
    all_targets = []
    meta_records = []

    for images, labels, meta in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
        else:
            logits = model(images)

        logits_f32 = logits.float().cpu().numpy()
        all_logits.append(logits_f32)
        all_targets.append(labels.numpy())

        # Unpack batch metadata
        batch_size = len(labels)
        for i in range(batch_size):
            meta_records.append({
                "patient_id": meta["patient_id"][i],
                "study_id": meta["study_id"][i],
                "dicom_id": meta["dicom_id"][i],
                "view_position": meta["view_position"][i],
                "relative_path": meta["relative_path"][i],
            })

    all_logits = np.vstack(all_logits)
    all_targets = np.vstack(all_targets)
    raw_probs = 1.0 / (1.0 + np.exp(-all_logits))

    # Compute analytic correction if class weights exist
    if pos_weights is not None:
        log_w = np.log(np.maximum(pos_weights, 1e-7))
        corrected_logits = all_logits - log_w.reshape(1, -1)
        corrected_probs = 1.0 / (1.0 + np.exp(-corrected_logits))
    else:
        corrected_logits = None
        corrected_probs = None

    return all_logits, raw_probs, corrected_logits, corrected_probs, all_targets, meta_records


def run_stage4a_evaluation(
    checkpoint_dir: str = "checkpoints",
    manifest_path: str = "data/processed/mimic_stage4a_manifest.csv",
    image_root: str = "data/mimic/images",
    output_dir: str = "reports/stage4a",
    batch_size: int = 64,
    num_workers: int = 4,
    allow_synthetic: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Initializing Stage 4A MIMIC External Validation on device: {device}")

    out_path = Path(output_dir)
    pred_dir = out_path / "predictions"
    table_dir = out_path / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    print(f"Loaded MIMIC test manifest: {len(manifest_df):,} frontal images.")

    # Find all anchor checkpoints
    ckpt_files = sorted(Path(checkpoint_dir).glob("anchor_*.pth"))
    if not ckpt_files:
        print(f"❌ No anchor checkpoints found in '{checkpoint_dir}'. Check path.")
        return

    print(f"Found {len(ckpt_files)} BRAX checkpoints to evaluate.")

    all_metrics_list = []

    for ckpt_path in ckpt_files:
        print(f"\n=======================================================")
        print(f"Loading checkpoint: {ckpt_path.name}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        arch = ckpt.get("architecture", "densenet121")
        run_id = ckpt.get("run_id", ckpt_path.stem)
        loss_type = ckpt.get("loss_type", "unweighted" if "unweighted" in ckpt_path.name else "weighted")
        seed = ckpt.get("seed", 42)
        targets = ckpt.get("targets", DEFAULT_TARGETS)
        pos_weights = ckpt.get("pos_weights", None)

        is_weighted = "weighted" in loss_type.lower() and "unweighted" not in loss_type.lower()
        if is_weighted and pos_weights is None:
            # Standard BRAX class weights if not in checkpoint dict
            pos_weights = np.array([21.511648, 9.476647, 54.0943, 1091.3043], dtype=np.float32)

        print(f"  Architecture : {arch}")
        print(f"  Loss Type    : {loss_type} (is_weighted={is_weighted})")
        print(f"  Seed         : {seed}")
        if is_weighted:
            print(f"  Weights w    : {pos_weights.tolist()}")
            print(f"  log(w)       : {np.log(pos_weights).tolist()}")

        model = build_model(architecture=arch, num_classes=len(targets), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        # Evaluate across strata: All, AP, PA
        strata = [
            ("all_frontal", None),
            ("ap_view", "AP"),
            ("pa_view", "PA"),
        ]

        # Assemble predictions file once for the full cohort
        full_dataset = MimicCxrDataset(
            manifest_df,
            image_root=image_root,
            targets=targets,
            allow_synthetic=allow_synthetic,
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        logits, probs_raw, corr_logits, probs_corr, targets_arr, meta = evaluate_model_on_mimic(
            model, full_loader, device, targets, pos_weights if is_weighted else None
        )

        # Build prediction records for archive
        pred_records = []
        n_samples = len(targets_arr)
        views_arr = np.array([m["view_position"] for m in meta])
        pts_arr = np.array([m["patient_id"] for m in meta])
        studies_arr = np.array([m["study_id"] for m in meta])

        for t_idx, t_name in enumerate(targets):
            yt = targets_arr[:, t_idx]
            z_raw = logits[:, t_idx]
            p_raw = probs_raw[:, t_idx]
            z_corr = corr_logits[:, t_idx] if corr_logits is not None else z_raw
            p_corr = probs_corr[:, t_idx] if probs_corr is not None else p_raw

            for i in range(n_samples):
                pred_records.append({
                    "run_id": run_id,
                    "architecture": arch,
                    "loss_type": loss_type,
                    "seed": seed,
                    "patient_id": pts_arr[i],
                    "study_id": studies_arr[i],
                    "view_position": views_arr[i],
                    "target": t_name,
                    "y_true": float(yt[i]),
                    "logit_raw": float(z_raw[i]),
                    "prob_raw": float(p_raw[i]),
                    "logit_corrected": float(z_corr[i]),
                    "prob_corrected": float(p_corr[i]),
                })

            # Calculate metrics across strata
            for stratum_name, view_filter in strata:
                if view_filter is None:
                    mask = np.ones(n_samples, dtype=bool)
                else:
                    mask = (views_arr == view_filter)

                yt_sub = yt[mask]
                praw_sub = p_raw[mask]
                pcorr_sub = p_corr[mask]

                # 1. Raw metrics
                m_raw = compute_all_metrics(yt_sub, praw_sub)
                m_raw.update({
                    "run_id": run_id,
                    "architecture": arch,
                    "loss_type": loss_type,
                    "seed": seed,
                    "target": t_name,
                    "stratum": stratum_name,
                    "eval_mode": "raw",
                    "sample_count": int(np.sum(mask)),
                })
                all_metrics_list.append(m_raw)

                # 2. Corrected metrics (if model is weighted)
                if is_weighted:
                    m_corr = compute_all_metrics(yt_sub, pcorr_sub)
                    m_corr.update({
                        "run_id": run_id,
                        "architecture": arch,
                        "loss_type": loss_type,
                        "seed": seed,
                        "target": t_name,
                        "stratum": stratum_name,
                        "eval_mode": "analytic_corrected",
                        "sample_count": int(np.sum(mask)),
                    })
                    all_metrics_list.append(m_corr)

        # Save prediction archive for this model
        pred_df = pd.DataFrame(pred_records)
        pred_file = pred_dir / f"preds_mimic_{run_id}.csv"
        pred_df.to_csv(pred_file, index=False)
        print(f"  📦 Saved prediction archive: {pred_file} ({len(pred_df):,} rows)")

    # Save summary metrics table
    metrics_df = pd.DataFrame(all_metrics_list)
    metrics_file = table_dir / "mimic_stage4a_metrics_all.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n📊 Successfully saved full Stage 4A metrics table: {metrics_file} ({len(metrics_df):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frozen BRAX models on MIMIC-CXR-JPG Stage 4A test cohort.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory with .pth checkpoints.")
    parser.add_argument("--manifest", type=str, default="data/processed/mimic_stage4a_manifest.csv")
    parser.add_argument("--image-root", type=str, default="data/mimic/images")
    parser.add_argument("--output-dir", type=str, default="reports/stage4a")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic grey images for dry-run testing.")
    args = parser.parse_args()

    run_stage4a_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        allow_synthetic=args.allow_synthetic,
    )
