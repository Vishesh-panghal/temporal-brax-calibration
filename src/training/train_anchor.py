"""Anchor block training pipeline optimized for NVIDIA Quadro RTX 8000 (48GB VRAM).

Supports:
- Automatic Mixed Precision (AMP fp16) via Turing Tensor Cores
- High-resolution (512x512) medical chest radiograph training
- High-throughput batch size (64 - 128) with fast pin_memory DMA transfers
- VRAM usage monitoring across 48GB capacity
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BraxDataset, get_default_transforms
from src.models.architectures import build_model
from src.training.loss import compute_pos_weights, WeightedBCEWithLogitsLoss
from src.evaluation.metrics import compute_discrimination_metrics


def set_seed(seed: int = 42):
    """Enforces deterministic seeding across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True):
    model.train()
    running_loss = 0.0

    for images, targets, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, target_names, device, use_amp=True):
    model.eval()
    all_targets = []
    all_probs = []

    for images, targets, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images)
        else:
            logits = model(images)

        probs = torch.sigmoid(logits.float())

        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    metrics = {}
    aurocs = []
    for idx, target_name in enumerate(target_names):
        y_true = all_targets[:, idx]
        y_prob = all_probs[:, idx]
        disc = compute_discrimination_metrics(y_true, y_prob)
        metrics[f"{target_name}_auroc"] = disc["auroc"]
        metrics[f"{target_name}_auprc"] = disc["auprc"]
        if not np.isnan(disc["auroc"]):
            aurocs.append(disc["auroc"])

    metrics["mean_auroc"] = float(np.mean(aurocs)) if aurocs else 0.0
    return metrics, all_targets, all_probs


def run_training(
    config_path: str = "configs/config.yaml",
    architecture: str = "densenet121",
    loss_type: str = "weighted_bce",
    seed: int = 42,
    max_epochs: int = 15,
    batch_size: int = 64,
    image_size: int = 512,
    num_workers: int = 8,
    use_amp: bool = True,
    selection_target: str = "Pleural Effusion",
):
    set_seed(seed)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Detect computing device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🚀 Detected CUDA GPU: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
        print(f"⚡ Mixed Precision (AMP): {use_amp} | Seed: {seed} | Loss: {loss_type}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
        use_amp = False
    else:
        device = torch.device("cpu")
        print("Using CPU device")
        use_amp = False

    manifest_path = config["data"]["manifest_path"]
    targets = [config["data"]["primary_target"]] + config["data"]["secondary_targets"]
    print(f"Target conditions: {targets}")
    print(f"Image resolution: {image_size}x{image_size} | Batch size: {batch_size} | Workers: {num_workers}")

    manifest_df = pd.read_csv(manifest_path)

    # Setup loss criterion
    if loss_type == "weighted_bce":
        pos_weights = compute_pos_weights(manifest_df, targets, split="train").to(device)
        criterion = WeightedBCEWithLogitsLoss(pos_weights)
        print(f"Computed positive class weights: {pos_weights.cpu().numpy().tolist()}")
    else:
        pos_weights = None
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard unweighted BCEWithLogitsLoss")

    train_transforms = get_default_transforms(image_size=image_size, is_training=True)
    val_transforms = get_default_transforms(image_size=image_size, is_training=False)

    image_root = config["data"].get("image_root", "")
    train_dataset = BraxDataset(
        manifest_path, split="train", targets=targets, transform=train_transforms,
        image_root=image_root, image_size=image_size
    )
    val_dataset = BraxDataset(
        manifest_path, split="val", targets=targets, transform=val_transforms,
        image_root=image_root, image_size=image_size
    )

    # Hardware DataLoader configuration with worker seed generator
    g = torch.Generator()
    g.manual_seed(seed)
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = build_model(
        architecture=architecture,
        num_classes=len(targets),
        pretrained=config["models"].get("pretrained", True),
        dropout=config["models"].get("dropout", 0.2),
    ).to(device)

    lr = config["training"].get("learning_rate", 1e-4)
    weight_decay = config["training"].get("weight_decay", 1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    os.makedirs("checkpoints", exist_ok=True)
    run_id = f"anchor_{architecture}_{loss_type}_seed{seed}"
    best_checkpoint_path = f"checkpoints/{run_id}.pth"
    best_selection_metric = 0.0

    print(f"\n=======================================================")
    print(f"   Starting Anchor Training: {run_id}")
    print(f"=======================================================\n")

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp=use_amp)
        val_metrics, _, _ = evaluate_epoch(model, val_loader, targets, device, use_amp=use_amp)
        scheduler.step()

        mean_auroc = val_metrics["mean_auroc"]
        effusion_auroc = val_metrics.get("Pleural Effusion_auroc", 0.0)
        current_metric = effusion_auroc if selection_target == "Pleural Effusion" else mean_auroc

        vram_info = ""
        if device.type == "cuda":
            alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_info = f" | VRAM Peak: {alloc_gb:.2f}/{total_vram_gb:.0f}GB"

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {loss:.4f} | Val Mean AUROC: {mean_auroc:.4f} | Effusion AUROC: {effusion_auroc:.4f}{vram_info}")

        if current_metric > best_selection_metric:
            best_selection_metric = current_metric
            torch.save({
                "epoch": epoch,
                "run_id": run_id,
                "architecture": architecture,
                "loss_type": loss_type,
                "seed": seed,
                "image_size": image_size,
                "state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "targets": targets,
                "selection_target": selection_target,
                "selection_metric_val": best_selection_metric,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "pos_weights": pos_weights.cpu().numpy() if pos_weights is not None else None,
            }, best_checkpoint_path)
            print(f"  ⭐ Checkpointed new best weights to {best_checkpoint_path} ({selection_target} AUROC: {best_selection_metric:.4f})")

    print(f"\n✅ Training Complete: {run_id}! Best Selection Metric: {best_selection_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BRAX Anchor Model on Quadro RTX 8000")
    parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "resnet50"])
    parser.add_argument("--loss", type=str, default="weighted_bce", choices=["weighted_bce", "unweighted_bce"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    run_training(
        architecture=args.arch,
        loss_type=args.loss,
        seed=args.seed,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.workers,
        use_amp=not args.no_amp,
    )
