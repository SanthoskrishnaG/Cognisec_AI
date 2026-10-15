from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Computes non-parametric empirical bootstrap confidence interval for a single metric.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = metric_func(y_true[idx], y_pred[idx])
            bootstrapped_scores.append(score)
        except Exception:
            continue

    if not bootstrapped_scores:
        point_est = float(metric_func(y_true, y_pred))
        return {"point_estimate": point_est, "ci_lower": point_est, "ci_upper": point_est}

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(bootstrapped_scores, alpha * 100))
    upper = float(np.percentile(bootstrapped_scores, (1.0 - alpha) * 100))
    point_est = float(metric_func(y_true, y_pred))

    return {
        "point_estimate": round(point_est, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std_error": round(float(np.std(bootstrapped_scores)), 4)
    }


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired bootstrap differences: Delta = Metric(Model1) - Metric(Model2).
    Evaluates whether the difference is statistically significant.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s1 = metric_func(y_true[idx], y_pred_model1[idx])
            s2 = metric_func(y_true[idx], y_pred_model2[idx])
            diffs.append(s1 - s2)
        except Exception:
            continue

    point_diff = float(metric_func(y_true, y_pred_model1) - metric_func(y_true, y_pred_model2))
    if not diffs:
        return {"delta_point": round(point_diff, 4), "ci_lower": point_diff, "ci_upper": point_diff, "p_value_approx": 1.0}

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(diffs, alpha * 100))
    upper = float(np.percentile(diffs, (1.0 - alpha) * 100))
    # Two-sided empirical p-value test against 0
    p_val = float(2.0 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))

    return {
        "delta_point": round(point_diff, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "p_value_approx": round(min(1.0, p_val), 4),
        "significant_at_95": bool(lower > 0 or upper < 0)
    }
