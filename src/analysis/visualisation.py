from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Use high-aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

class Visualiser:
    """
    Produces publication-grade figures from empirical evaluation results.
    """
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_dataset_distribution(self, family_counts: Dict[str, int], save_name: str = "fig1_dataset_distribution.png"):
        plt.figure(figsize=(10, 6))
        sorted_items = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, v in sorted_items]
        counts = [v for k, v in sorted_items]
        colors = ["#2b5c8f" if lbl == "BENIGN" else "#d95f02" for lbl in labels]

        bars = plt.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        plt.xlabel("Number of Flow Records (Log Scale)")
        plt.title("Authentic CICIDS2017 Curated Flow Distribution by Traffic Family")
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_roc_and_pr_curves(
        self,
        y_true: np.ndarray,
        xgb_scores: np.ndarray,
        vae_scores: np.ndarray,
        save_name: str = "fig2_roc_pr_curves.png"
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. ROC Curves
        fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_scores)
        fpr_vae, tpr_vae, _ = roc_curve(y_true, vae_scores)

        axes[0].plot(fpr_xgb, tpr_xgb, label="XGBoost (Supervised)", color="#1f77b4", lw=2)
        axes[0].plot(fpr_vae, tpr_vae, label="VAE (Reconstruction Score)", color="#e377c2", lw=2)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
        axes[0].set_xlabel("False Positive Rate (FPR)")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("ROC Curves Comparison")
        axes[0].legend(loc="lower right")

        # 2. PR Curves
        p_xgb, r_xgb, _ = precision_recall_curve(y_true, xgb_scores)
        p_vae, r_vae, _ = precision_recall_curve(y_true, vae_scores)

        axes[1].plot(r_xgb, p_xgb, label="XGBoost", color="#1f77b4", lw=2)
        axes[1].plot(r_vae, p_vae, label="VAE", color="#e377c2", lw=2)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curves Comparison")
        axes[1].legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        xgb_preds: np.ndarray,
        vae_preds: np.ndarray,
        save_name: str = "fig3_confusion_matrices.png"
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cm_xgb = confusion_matrix(y_true, xgb_preds, labels=[0, 1])
        cm_vae = confusion_matrix(y_true, vae_preds, labels=[0, 1])

        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                    xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
        axes[0].set_title("XGBoost Confusion Matrix")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")

        sns.heatmap(cm_vae, annot=True, fmt="d", cmap="Oranges", ax=axes[1], cbar=False,
                    xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
        axes[1].set_title("VAE (Anomaly Thresholded) Confusion Matrix")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_anomaly_score_distribution(
        self,
        benign_scores: np.ndarray,
        known_attack_scores: np.ndarray,
        held_out_attack_scores: np.ndarray,
        threshold: float,
        save_name: str = "fig4_anomaly_score_distributions.png"
    ):
        plt.figure(figsize=(11, 6))

        # Clip high extreme outliers for aesthetic density plotting
        p99 = np.percentile(held_out_attack_scores, 98) if len(held_out_attack_scores) > 0 else 10.0
        max_clip = max(p99 * 1.5, threshold * 2.0)

        sns.kdeplot(np.clip(benign_scores, 0, max_clip), label="Benign Traffic", color="#2ca02c", fill=True, alpha=0.3, lw=2)
        if len(known_attack_scores) > 0:
            sns.kdeplot(np.clip(known_attack_scores, 0, max_clip), label="Known Attacks (Seen Families)", color="#1f77b4", fill=True, alpha=0.3, lw=2)
        if len(held_out_attack_scores) > 0:
            sns.kdeplot(np.clip(held_out_attack_scores, 0, max_clip), label="Held-Out Attacks (Zero-Day)", color="#d62728", fill=True, alpha=0.3, lw=2)

        plt.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Calibrated Threshold ({round(threshold, 4)})")
        plt.xlabel("Reconstruction Anomaly Score (Mean Squared Error)")
        plt.ylabel("Density")
        plt.title("VAE Latent Reconstruction Anomaly Score Distributions")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_per_family_recall(
        self,
        family_results_xgb: Dict[str, Dict[str, Any]],
        family_results_vae: Dict[str, Dict[str, Any]],
        save_name: str = "fig5_per_family_recall.png"
    ):
        families = [f for f in family_results_xgb.keys() if f != "BENIGN"]
        if not families:
            return

        xgb_recalls = [family_results_xgb[f]["recall"] for f in families]
        vae_recalls = [family_results_vae.get(f, {}).get("recall", 0.0) for f in families]

        x = np.arange(len(families))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, xgb_recalls, width, label="XGBoost Recall", color="#1f77b4", edgecolor="black", alpha=0.85)
        plt.bar(x + width/2, vae_recalls, width, label="VAE Recall", color="#e377c2", edgecolor="black", alpha=0.85)

        plt.xlabel("Attack Family")
        plt.ylabel("Detection Recall Rate")
        plt.title("Per-Attack-Family Detection Recall: Supervised vs Benign Anomaly Modelling")
        plt.xticks(x, families, rotation=25, ha="right")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_threshold_sensitivity(
        self,
        sensitivity_records: List[Dict[str, Any]],
        save_name: str = "fig6_threshold_sensitivity.png"
    ):
        targets = [r["target_fpr_percent"] for r in sensitivity_records]
        test_fprs = [r["test_fpr"] * 100 for r in sensitivity_records]
        recalls = [r["test_recall"] * 100 for r in sensitivity_records]
        f1s = [r["test_f1"] * 100 for r in sensitivity_records]

        plt.figure(figsize=(10, 6))
        plt.plot(targets, recalls, marker="o", label="Attack Recall (%)", color="#2ca02c", lw=2)
        plt.plot(targets, f1s, marker="s", label="F1-Score (%)", color="#1f77b4", lw=2)
        plt.plot(targets, test_fprs, marker="^", label="Empirical Test FPR (%)", color="#d62728", lw=2)

        plt.xlabel("Pre-Declared Benign Validation False-Alarm Target (%)")
        plt.ylabel("Performance Metric Percentage (%)")
        plt.title("VAE Anomaly Threshold Sensitivity & Operational Trade-Offs")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_distribution_shift_degradation(
        self,
        severities: List[float],
        xgb_degradations: List[float],
        vae_degradations: List[float],
        shift_label: str = "Packet Size Scaling Factor",
        save_name: str = "fig7_distribution_shift_degradation.png"
    ):
        plt.figure(figsize=(10, 6))
        plt.plot(severities, xgb_degradations, marker="o", lw=2, label="XGBoost PR-AUC", color="#1f77b4")
        plt.plot(severities, vae_degradations, marker="s", lw=2, label="VAE PR-AUC", color="#e377c2")

        plt.xlabel(shift_label)
        plt.ylabel("PR-AUC Score")
        plt.title(f"Model Generalisation Degradation under Distribution Shift ({shift_label})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()

    def plot_model_disagreement(
        self,
        both_detected: int,
        xgb_only: int,
        vae_only: int,
        both_missed: int,
        save_name: str = "fig8_model_disagreement.png"
    ):
        plt.figure(figsize=(9, 6))
        categories = ["Both Detected", "XGBoost Only", "VAE Only", "Both Missed"]
        counts = [both_detected, xgb_only, vae_only, both_missed]
        colors = ["#2ca02c", "#1f77b4", "#e377c2", "#d62728"]

        bars = plt.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 10, f"{height:,}", ha="center", va="bottom", fontsize=10, weight="bold")

        plt.ylabel("Number of Attack Instances")
        plt.title("Complementarity Analysis: Attack Detection Overlap on Held-Out Families")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()
