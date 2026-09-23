"""Clinical discrimination and calibration metrics for chest radiography models."""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Computes discrimination metrics: AUROC, AUPRC, and positive prevalence."""
    # Ensure 1D arrays
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    prevalence = float(np.mean(y_true))
    
    # AUROC requires at least one positive and one negative sample
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": prevalence,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width or equal-mass (quantile) bins."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_prob)
    if n == 0:
        return float("nan")

    ece = 0.0

    if strategy == "equal_mass":
        # Quantile binning: each bin contains approximately equal sample count
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_boundaries = np.percentile(y_prob, quantiles * 100.0)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        # Remove duplicate bin boundaries if predictions are concentrated
        bin_boundaries = np.unique(bin_boundaries)
        effective_bins = len(bin_boundaries) - 1
        if effective_bins <= 0:
            return float(np.abs(np.mean(y_prob) - np.mean(y_true)))
    else:
        # Standard equal-width probability bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        effective_bins = n_bins

    for i in range(effective_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_prob = np.mean(y_prob[in_bin])
            acc = np.mean(y_true[in_bin])
            ece += prop_in_bin * np.abs(avg_prob - acc)

    return float(ece)


def compute_calibration_slope_and_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """Computes calibration slope, joint intercept, and intercept-in-the-large.
    
    1. Joint regression: logit(P(Y=1)) = a + b * logit(p_hat)
       (ideal: slope b = 1.0, joint intercept a = 0.0)
    2. Intercept-in-the-large: logit(P(Y=1)) = a_large + 1.0 * logit(p_hat)
       (ideal: a_large = 0.0; measures overall over/under-estimation)
    
    Returns:
        (slope, joint_intercept, intercept_in_the_large)
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    log_odds = logit(y_prob).reshape(-1, 1)

    # 1. Joint logistic regression (slope + intercept)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(log_odds, y_true)
        slope = float(lr.coef_[0][0])
        joint_intercept = float(lr.intercept_[0])
    except Exception:
        slope, joint_intercept = float("nan"), float("nan")

    # 2. Intercept-in-the-large (offset model with slope fixed at 1.0)
    # logit(P) = a_large + log_odds <=> offset optimization via logistic loss with fixed log_odds
    try:
        # Logistic loss with log_odds as offset: optimize a single scalar a_large
        from scipy.optimize import minimize_scalar
        def neg_log_lik(a):
            p = 1.0 / (1.0 + np.exp(-(a + log_odds.ravel())))
            p = np.clip(p, eps, 1.0 - eps)
            return -np.sum(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        res = minimize_scalar(neg_log_lik, bracket=[-5.0, 5.0])
        intercept_in_the_large = float(res.x) if res.success else float("nan")
    except Exception:
        intercept_in_the_large = float("nan")

    return slope, joint_intercept, intercept_in_the_large


def compute_clinical_operating_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes decision-point clinical metrics: Sensitivity, Specificity, PPV, NPV, and Accuracy."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else float("nan")

    return {
        f"sensitivity_th{threshold:.2f}": sens,
        f"specificity_th{threshold:.2f}": spec,
        f"ppv_th{threshold:.2f}": ppv,
        f"npv_th{threshold:.2f}": npv,
        f"accuracy_th{threshold:.2f}": acc,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes comprehensive calibration metrics including equal-width and quantile ECE."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), eps, 1.0 - eps)

    brier = float(brier_score_loss(y_true, y_prob))

    try:
        ll = float(log_loss(y_true, y_prob))
    except ValueError:
        ll = float("nan")

    ece_width = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="equal_width")
    ece_mass = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="equal_mass")
    slope, joint_intercept, intercept_large = compute_calibration_slope_and_intercept(y_true, y_prob, eps=eps)

    return {
        "brier_score": brier,
        "log_loss": ll,
        "ece": ece_width,
        "ece_equal_width": ece_width,
        "ece_equal_mass": ece_mass,
        "calibration_slope": slope,
        "calibration_intercept": joint_intercept,
        "intercept_in_the_large": intercept_large,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    operating_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes joint discrimination, calibration, and clinical decision-point metrics."""
    disc = compute_discrimination_metrics(y_true, y_prob)
    calib = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins)
    clin = compute_clinical_operating_metrics(y_true, y_prob, threshold=operating_threshold)
    return {**disc, **calib, **clin}
