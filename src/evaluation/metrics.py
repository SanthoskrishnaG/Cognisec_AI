import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics including PR-AUC and FPR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "roc_auc": None,
        "pr_auc": None
    }

    if y_score is not None:
        try:
            # Handle single class edge case in slice
            if len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

    return metrics


def calculate_per_family_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_labels: np.ndarray,
    benign_label: str = "BENIGN"
) -> Dict[str, Dict[str, Any]]:
    """
    Disaggregates detection metrics by specific attack family.
    Crucial for identifying which families are detected vs missed.
    """
    unique_labels = np.unique(raw_labels)
    family_results = {}

    for label in unique_labels:
        mask = (raw_labels == label)
        count = int(np.sum(mask))
        if count == 0:
            continue

        if label == benign_label:
            # For benign, compute False Alarm Rate (FPR)
            false_alarms = int(np.sum(y_pred[mask] == 1))
            family_results[label] = {
                "total_samples": count,
                "false_alarms": false_alarms,
                "false_alarm_rate": round(false_alarms / count, 4)
            }
        else:
            # For attacks, compute Detection Recall
            detected = int(np.sum(y_pred[mask] == 1))
            family_results[label] = {
                "total_samples": count,
                "detected": detected,
                "missed": count - detected,
                "recall": round(detected / count, 4)
            }

    return family_results


def benchmark_inference_latency(
    predict_fn: Any,
    X: np.ndarray,
    warmup_runs: int = 2,
    benchmark_runs: int = 5
) -> Dict[str, float]:
    """
    Measures mean inference latency and throughput.
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = predict_fn(X[: min(100, len(X))])

    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = predict_fn(X)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    mean_total_time = float(np.mean(latencies))
    n_samples = len(X)
    ms_per_sample = (mean_total_time / n_samples) * 1000.0 if n_samples > 0 else 0.0
    throughput = (n_samples / mean_total_time) if mean_total_time > 0 else 0.0

    return {
        "mean_latency_seconds": round(mean_total_time, 4),
        "ms_per_sample": round(ms_per_sample, 5),
        "throughput_samples_per_sec": round(throughput, 1)
    }
