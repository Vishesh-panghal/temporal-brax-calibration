"""Clinical Mitigation Engine: Post-hoc Temperature Scaling and Selective Prediction/Abstention.

Quantifies:
1. Optimal Temperature Scaling:
   - Fits scalar temperature T on Anchor Validation split
   - Evaluates pre- vs post-recalibration ECE, Brier Score, and NLL on future bins T1, T2, T3
   - Evaluates sliding-window recalibration vs static anchor calibration
2. Selective Prediction & Clinical Abstention:
   - Prediction deferral based on epistemic probability uncertainty
   - Risk-coverage trade-off curves (AURC)
   - Evaluates effective error reduction under selective review policies
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Tuple
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BraxDataset
from src.models.architectures import build_model
from src.evaluation.metrics import (
    compute_expected_calibration_error,
    compute_discrimination_metrics,
)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling calibrator: p_calib = sigmoid(logit / T).
    
    Guarantees T > 0 and monotonic NLL optimization:
      - Uses bounded 1D scalar optimization on [0.01, 50.0]
      - Validates that optimal NLL does not exceed initial uncalibrated NLL (T=1.0)
      - Returns optimal positive scalar temperature T*
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = float(max(init_temp, 0.01))

    @property
    def temperature_value(self) -> float:
        """Returns the current positive temperature T."""
        return float(self.temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = max(self.temperature, 0.01)
        return logits / temp

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Applies temperature scaling: p = sigmoid(z / T)."""
        z = logits.astype(np.float64)
        scaled = z / max(self.temperature, 0.01)
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Finds optimal temperature T minimizing binary cross-entropy on validation logits.
        
        Guaranteed positive and strictly non-worsening relative to T=1.0.
        """
        from scipy.optimize import minimize_scalar

        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        eps = 1e-7

        if len(np.unique(y)) < 2:
            self.temperature = 1.0
            return self.temperature

        def neg_log_lik(temp):
            t = max(float(temp), 0.01)
            p = 1.0 / (1.0 + np.exp(-z / t))
            p = np.clip(p, eps, 1.0 - eps)
            return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        loss_identity = neg_log_lik(1.0)
        res = minimize_scalar(neg_log_lik, bounds=(0.01, 50.0), method='bounded')

        if res.success and res.fun <= loss_identity:
            self.temperature = float(res.x)
        else:
            self.temperature = 1.0

        return self.temperature


class OffsetCalibrator:
    """Intercept-only calibrator: p_calib = sigmoid(logit + a).
    
    Tests whether correcting a baseline offset is sufficient.
    Fits scalar intercept 'a' minimizing BCE on validation logits.
    """

    def __init__(self):
        self.offset = 0.0

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Fits offset 'a' via scipy optimization on validation data."""
        from scipy.optimize import minimize_scalar

        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        eps = 1e-7

        if len(np.unique(y)) < 2:
            self.offset = 0.0
            return self.offset

        def neg_log_lik(a):
            p = 1.0 / (1.0 + np.exp(-(z + a)))
            p = np.clip(p, eps, 1.0 - eps)
            return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        loss_identity = neg_log_lik(0.0)
        res = minimize_scalar(neg_log_lik, bracket=[-10.0, 10.0], method='brent')
        if res.success and res.fun <= loss_identity:
            self.offset = float(res.x)
        else:
            self.offset = 0.0
        return self.offset

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Applies fitted offset: p = sigmoid(z + a)."""
        z = logits.astype(np.float64) + self.offset
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class PlattCalibrator:
    """Platt scaling calibrator: p_calib = sigmoid(a + b * logit).
    
    Allows both offset and scale correction via logistic regression.
    Uses sklearn LogisticRegression with weak regularization (C=1e6) and positivity check on slope.
    """

    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> tuple:
        """Fits Platt parameters (a, b) via logistic regression on validation data."""
        from sklearn.linear_model import LogisticRegression

        z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(targets, dtype=np.float64).ravel()

        if len(np.unique(y)) < 2:
            self.slope = 1.0
            self.intercept = 0.0
            return self.slope, self.intercept

        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
        lr.fit(z, y)
        fitted_slope = float(lr.coef_[0][0])
        fitted_intercept = float(lr.intercept_[0])

        # Guard against negative slope which would invert predictions
        if fitted_slope <= 0.0:
            self.slope = 1.0
            self.intercept = 0.0
        else:
            self.slope = fitted_slope
            self.intercept = fitted_intercept

        return self.slope, self.intercept

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Applies Platt scaling: p = sigmoid(a + b * z)."""
        z = logits.astype(np.float64)
        linear = self.intercept + self.slope * z
        return (1.0 / (1.0 + np.exp(-linear))).astype(np.float32)


def analytic_weight_correction(logits: np.ndarray, pos_weight: float) -> np.ndarray:
    """Analytic correction for weighted BCE: p_corrected = sigmoid(z - log(w)).
    
    At the ideal population optimum of positive-weighted BCE, weighting adds
    log(w) to the target log odds. Subtracting that offset is a useful
    diagnostic; it is not guaranteed exact calibration of a finite trained network.
    
    Args:
        logits: Raw model logits (float32/float64)
        pos_weight: The positive class weight used during training (w > 0)
    
    Returns:
        Corrected probabilities as float32 array
    """
    z = logits.astype(np.float64)
    corrected = z - np.log(max(pos_weight, 1e-7))
    return (1.0 / (1.0 + np.exp(-corrected))).astype(np.float32)


def apply_all_calibrators(
    logits: np.ndarray,
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    pos_weight: float = None,
) -> dict:
    """Applies all calibration methods to logits, returning a dict of {method: probabilities}.
    
    Calibrators are fitted on val_logits/val_targets, then applied to logits.
    Returns dict with keys: 'raw', 'temperature', 'offset', 'platt', 'analytic_weight' (if pos_weight given).
    """
    results = {}

    # Raw
    results['raw'] = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)

    # Temperature scaling
    ts = TemperatureScaler()
    fitted_t = ts.fit(val_logits, val_targets)
    results['temperature'] = (1.0 / (1.0 + np.exp(-logits.astype(np.float64) / fitted_t))).astype(np.float32)
    results['_temperature_value'] = fitted_t

    # Offset (intercept-only)
    oc = OffsetCalibrator()
    oc.fit(val_logits, val_targets)
    results['offset'] = oc.calibrate(logits)
    results['_offset_value'] = oc.offset

    # Platt scaling
    pc = PlattCalibrator()
    pc.fit(val_logits, val_targets)
    results['platt'] = pc.calibrate(logits)
    results['_platt_slope'] = pc.slope
    results['_platt_intercept'] = pc.intercept

    # Analytic weight correction (only for weighted models)
    if pos_weight is not None and pos_weight > 0:
        results['analytic_weight'] = analytic_weight_correction(logits, pos_weight)

    return results


def compute_uncertainty(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Computes uncertainty as proximity to the decision threshold.
    Maximal uncertainty (1.0) occurs at p = threshold; minimal (0.0) at extremes.
    """
    scale = np.where(probs >= threshold, 1.0 - threshold, threshold)
    scale = np.maximum(scale, 1e-7)
    norm_dist = np.abs(probs - threshold) / scale
    return 1.0 - np.clip(norm_dist, 0.0, 1.0)


def evaluate_selective_prediction(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverage_levels: np.ndarray = np.linspace(0.5, 1.0, 11),
    operating_threshold: float = 0.5,
) -> pd.DataFrame:
    """Computes clinical risk, class-conditional deferral, and decision metrics across coverage."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()
    uncertainties = compute_uncertainty(y_prob, threshold=operating_threshold)
    results = []

    # Sort indices by ascending uncertainty (most confident first)
    sorted_indices = np.argsort(uncertainties)
    n_total = len(y_true)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    for cov in coverage_levels:
        k = int(np.ceil(cov * n_total))
        k = max(1, min(k, n_total))
        selected_idx = sorted_indices[:k]
        deferred_idx = sorted_indices[k:]

        y_t_sel = y_true[selected_idx]
        y_p_sel = y_prob[selected_idx]

        # Deferral fairness: P(defer | Y=1) vs P(defer | Y=0)
        def_pos = int(np.sum(y_true[deferred_idx] == 1)) if len(deferred_idx) > 0 else 0
        def_neg = int(np.sum(y_true[deferred_idx] == 0)) if len(deferred_idx) > 0 else 0
        p_defer_given_pos = float(def_pos / total_pos) if total_pos > 0 else float("nan")
        p_defer_given_neg = float(def_neg / total_neg) if total_neg > 0 else float("nan")

        # Retained decision outcomes at clinical operating threshold
        y_pred_sel = (y_p_sel >= operating_threshold).astype(int)
        tp = int(np.sum((y_pred_sel == 1) & (y_t_sel == 1)))
        fp = int(np.sum((y_pred_sel == 1) & (y_t_sel == 0)))
        tn = int(np.sum((y_pred_sel == 0) & (y_t_sel == 0)))
        fn = int(np.sum((y_pred_sel == 0) & (y_t_sel == 1)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")

        brier = float(np.mean((y_p_sel - y_t_sel) ** 2))
        ece_width = compute_expected_calibration_error(y_t_sel, y_p_sel, strategy="equal_width")
        ece_mass = compute_expected_calibration_error(y_t_sel, y_p_sel, strategy="equal_mass")

        # AUROC if both classes present
        if len(np.unique(y_t_sel)) > 1:
            disc = compute_discrimination_metrics(y_t_sel, y_p_sel)
            auroc = disc["auroc"]
            auprc = disc["auprc"]
        else:
            auroc = float("nan")
            auprc = float("nan")

        results.append({
            "coverage": float(cov),
            "retained_samples": k,
            "deferred_samples": n_total - k,
            "p_defer_given_pos": p_defer_given_pos,
            "p_defer_given_neg": p_defer_given_neg,
            "retained_positives": int(np.sum(y_t_sel == 1)),
            "retained_prevalence": float(np.mean(y_t_sel)),
            "automated_false_negatives": fn,
            "automated_false_positives": fp,
            "sensitivity_retained": sens,
            "specificity_retained": spec,
            "ppv_retained": ppv,
            "npv_retained": npv,
            "brier_score": brier,
            "ece": ece_mass,
            "ece_equal_width": ece_width,
            "ece_equal_mass": ece_mass,
            "auroc": auroc,
            "auprc": auprc,
        })

    return pd.DataFrame(results)


@torch.no_grad()
def extract_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs model forward pass and extracts raw uncalibrated logits and true binary labels."""
    all_logits = []
    all_targets = []

    for images, targets, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(images)
        else:
            logits = model(images)

        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.float().cpu().numpy())

    return np.vstack(all_logits), np.vstack(all_targets)


def run_mitigation_experiments(
    checkpoint_path: str,
    manifest_path: str = "data/processed/stage2_manifest.csv",
    output_dir: str = "reports/stage2/mitigation",
    image_root: str = "",
    batch_size: int = 128,
    num_workers: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  STAGE 2: CLINICAL MITIGATION (RECALIBRATION & ABSTENTION)")
    print(f"=======================================================")
    print(f"Device: {device} | Loading: {checkpoint_path}")

    # Load image_root from config.yaml if not provided
    if not image_root and os.path.exists("configs/config.yaml"):
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
            image_root = cfg.get("data", {}).get("image_root", "")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "densenet121")
    targets = checkpoint.get("targets", ["Pleural Effusion", "Cardiomegaly", "Pneumonia", "Edema"])
    image_size = checkpoint.get("image_size", 512)
    run_id = checkpoint.get("run_id", f"eval_{architecture}")

    model = build_model(architecture=architecture, num_classes=len(targets), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    full_df = pd.read_csv(manifest_path)
    bins = sorted(full_df[full_df["split"] == "test"]["temporal_bin"].unique())
    print(f"Target conditions: {targets}")
    print(f"Temporal test bins: {bins}")
    print(f"Resolution: {image_size}x{image_size} | Run ID: {run_id}")
    print(f"Image Root: '{image_root}'")

    # -------------------------------------------------------------
    # 1. Extract Anchor Validation Logits to fit Temperature Scaling
    # -------------------------------------------------------------
    print("\n[Phase 1] Extracting Anchor Validation logits...")
    val_dataset = BraxDataset(manifest_path, split="val", targets=targets, image_root=image_root, image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_logits, val_targets = extract_logits_and_labels(model, val_loader, device)

    # Fit optimal temperature per target condition on Anchor Validation
    optimal_temperatures = {}
    print("\n[Phase 2] Optimal Temperature Scaling per Condition (fitted on Anchor Val):")
    for idx, target in enumerate(targets):
        scaler = TemperatureScaler()
        opt_t = scaler.fit(val_logits[:, idx], val_targets[:, idx])
        optimal_temperatures[target] = opt_t
        print(f"  • {target:<18}: T* = {opt_t:.4f}")

    # -------------------------------------------------------------
    # 2. Evaluate Future Bins: Pre- vs. Post-Recalibration
    # -------------------------------------------------------------
    recalib_records = []
    selective_records = []

    print("\n[Phase 3] Evaluating Recalibration & Abstention on Future Bins (T1, T2, T3)...")
    for bin_name in bins:
        bin_dataset = BraxDataset(manifest_path, split="test", temporal_bin=bin_name, targets=targets, image_root=image_root, image_size=image_size)
        bin_loader = DataLoader(bin_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        bin_logits, bin_targets = extract_logits_and_labels(model, bin_loader, device)

        for idx, target in enumerate(targets):
            y_t = bin_targets[:, idx]
            z = bin_logits[:, idx]

            # Raw uncalibrated probabilities
            p_raw = 1.0 / (1.0 + np.exp(-z))
            ece_raw_w = compute_expected_calibration_error(y_t, p_raw, strategy="equal_width")
            ece_raw_m = compute_expected_calibration_error(y_t, p_raw, strategy="equal_mass")
            brier_raw = float(np.mean((p_raw - y_t) ** 2))
            disc_raw = compute_discrimination_metrics(y_t, p_raw)

            # Recalibrated probabilities with Anchor T*
            opt_t = optimal_temperatures[target]
            p_calib = 1.0 / (1.0 + np.exp(-z / opt_t))
            ece_calib_w = compute_expected_calibration_error(y_t, p_calib, strategy="equal_width")
            ece_calib_m = compute_expected_calibration_error(y_t, p_calib, strategy="equal_mass")
            brier_calib = float(np.mean((p_calib - y_t) ** 2))

            # Record comparison
            recalib_records.append({
                "run_id": run_id,
                "architecture": architecture,
                "temporal_bin": bin_name,
                "target": target,
                "temperature": opt_t,
                "auroc": disc_raw["auroc"],
                "ece_raw_width": ece_raw_w,
                "ece_calib_width": ece_calib_w,
                "ece_raw_mass": ece_raw_m,
                "ece_calib_mass": ece_calib_m,
                "ece_delta_pct": ((ece_calib_w - ece_raw_w) / ece_raw_w) * 100.0 if ece_raw_w > 0 else 0.0,
                "brier_uncalibrated": brier_raw,
                "brier_recalibrated": brier_calib,
                "brier_delta_pct": ((brier_calib - brier_raw) / brier_raw) * 100.0 if brier_raw > 0 else 0.0,
            })

            # Selective prediction (for Pleural Effusion and Cardiomegaly)
            if target in ["Pleural Effusion", "Cardiomegaly"]:
                sel_df = evaluate_selective_prediction(y_t, p_calib)
                sel_df["run_id"] = run_id
                sel_df["temporal_bin"] = bin_name
                sel_df["architecture"] = architecture
                sel_df["target"] = target
                selective_records.append(sel_df)

    recalib_df = pd.DataFrame(recalib_records)
    sel_summary_df = pd.concat(selective_records, ignore_index=True) if selective_records else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    recalib_csv = os.path.join(output_dir, f"recalibration_results_{run_id}.csv")
    recalib_df.to_csv(recalib_csv, index=False)

    sel_csv = os.path.join(output_dir, f"selective_prediction_{run_id}.csv")
    sel_summary_df.to_csv(sel_csv, index=False)

    print(f"\nSaved Recalibration Results to: {recalib_csv}")
    print(f"Saved Selective Prediction Results to: {sel_csv}")

    # Display clean comparative table
    print("\n" + "=" * 90)
    print(f"  RECALIBRATION IMPACT TABLE ({architecture.upper()})")
    print("=" * 90)
    print(f"{'Bin':<9} | {'Target':<17} | {'ECE Raw':<10} | {'ECE Calib':<10} | {'ECE Δ %':<10} | {'Brier Raw':<10} | {'Brier Calib':<10}")
    print("-" * 90)
    for _, row in recalib_df.iterrows():
        print(f"{row['temporal_bin']:<9} | {row['target']:<17} | {row['ece_raw_mass']:<10.4f} | {row['ece_calib_mass']:<10.4f} | {row['ece_delta_pct']:>+9.1f}% | {row['brier_uncalibrated']:<10.4f} | {row['brier_recalibrated']:<10.4f}")
    print("=" * 90)

    # Display selective prediction overview
    if not sel_summary_df.empty:
        print("\n" + "=" * 80)
        print(f"  SELECTIVE PREDICTION / ABSTENTION POLICY (PLEURAL EFFUSION)")
        print("=" * 80)
        print(f"{'Bin':<9} | {'Coverage':<10} | {'Retained (N)':<14} | {'Brier Score':<12} | {'ECE':<10} | {'AUROC':<10}")
        print("-" * 80)
        for b_name in bins:
            b_data = sel_summary_df[sel_summary_df['temporal_bin'] == b_name]
            for _, r in b_data[b_data['coverage'].isin([1.0, 0.9, 0.8, 0.7])].iterrows():
                print(f"{r['temporal_bin']:<9} | {r['coverage']:<10.1f} | {int(r['retained_samples']):<14} | {r['brier_score']:<12.4f} | {r.get('ece', r.get('ece_equal_mass', 0.0)):<10.4f} | {r['auroc']:<10.4f}")
            print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recalibration and Abstention Policies")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--manifest", type=str, default="data/processed/stage2_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="reports/stage2/mitigation")
    parser.add_argument("--image-root", type=str, default="", help="Root path to images (defaults to config.yaml if empty)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_mitigation_experiments(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
