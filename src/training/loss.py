"""Loss functions for class-imbalanced multi-label thoracic pathology classification."""

from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weights(
    manifest_df: pd.DataFrame,
    target_columns: List[str],
    split: str = "train",
    multiplier: float = 1.0,
) -> torch.Tensor:
    """Calculates inverse class prevalence positive weights for BCEWithLogitsLoss.
    
    Formula: pos_weight = (num_negatives / num_positives) * multiplier
    """
    df_split = manifest_df[manifest_df["split"] == split] if "split" in manifest_df.columns else manifest_df
    weights = []
    
    for col in target_columns:
        pos = (df_split[col] == 1.0).sum()
        total = len(df_split)
        neg = total - pos
        if pos > 0:
            w = (neg / pos) * multiplier
        else:
            w = 1.0
        weights.append(w)
        
    return torch.tensor(weights, dtype=torch.float32)


class WeightedBCEWithLogitsLoss(nn.Module):
    """Multi-label BCE loss with class-specific positive weights."""

    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance in medical vision."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
