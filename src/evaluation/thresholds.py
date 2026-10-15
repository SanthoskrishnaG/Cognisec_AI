from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def select_threshold_from_benign_validation(
    benign_val_scores: np.ndarray,
    target_fpr_percentile: float = 1.0
) -> float:
    """
    Selects the decision threshold strictly from benign validation scores.
    Enforces that false alarms on validation benign traffic do not exceed target_fpr_percentile.
    """
    percentile = 100.0 - target_fpr_percentile
    threshold = float(np.percentile(benign_val_scores, percentile))
    return threshold


def sweep_threshold_sensitivity(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    benign_val_scores: np.ndarray,
    target_fpr_sweep: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
) -> List[Dict[str, Any]]:
    """
    Sweeps pre-declared benign false-alarm targets, calibrates thresholds on validation benign,
    and evaluates out-of-sample operational metrics on test set.
    """
    results = []
    for target_fpr in target_fpr_sweep:
        thresh = select_threshold_from_benign_validation(benign_val_scores, target_fpr_percentile=target_fpr)
        preds = (test_scores >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        test_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(precision_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        results.append({
            "target_fpr_percent": target_fpr,
            "calibrated_threshold": round(thresh, 6),
            "test_fpr": round(test_fpr, 4),
            "test_recall": round(recall, 4),
            "test_precision": round(precision, 4),
            "test_f1": round(f1, 4),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        })

    return results
