import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.models.xgboost_model import XGBoostClassifierWrapper
from src.models.vae_model import TabularVAE, VAETrainer
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_per_family_recall,
    benchmark_inference_latency
)
from src.evaluation.thresholds import (
    select_threshold_from_benign_validation,
    sweep_threshold_sensitivity
)
from src.evaluation.statistics import (
    bootstrap_metric_ci,
    paired_bootstrap_comparison
)
from src.evaluation.robustness import apply_feature_shift
from src.analysis.visualisation import Visualiser
from src.utils.configuration import load_yaml_config
from src.utils.reproducibility import set_seed, get_device

def run_master_experiments():
    print("================================================================================")
    print("   MASTER EXPERIMENT SUITE: NETWORK BEHAVIOUR FINGERPRINTING RESEARCH")
    print("================================================================================")
    set_seed(42)

    # Output paths
    results_dir = Path("results")
    figures_dir = Path("reports/figures")
    tables_dir = Path("reports/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    visualiser = Visualiser(output_dir=str(figures_dir))

    # 1. Load Partitioned Authentic Data
    print("[*] Loading preprocessed authentic CICIDS2017 partitions...")
    train_data = np.load("data/processed/train_data.npz", allow_pickle=True)
    val_data = np.load("data/processed/val_data.npz", allow_pickle=True)
    test_data = np.load("data/processed/test_data.npz", allow_pickle=True)

    X_train_unscaled = train_data["X_unscaled"]
    X_train_scaled = train_data["X_scaled"]
    y_train = train_data["y"]
    families_train = train_data["attack_families"]

    X_val_unscaled = val_data["X_unscaled"]
    X_val_scaled = val_data["X_scaled"]
    y_val = val_data["y"]
    families_val = val_data["attack_families"]

    X_test_unscaled = test_data["X_unscaled"]
    X_test_scaled = test_data["X_scaled"]
    y_test = test_data["y"]
    families_test = test_data["attack_families"]

    # Manifest
    with open("data/processed/partition_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    held_out_families = manifest.get("held_out_attack_families", ["DDoS", "Web_Attack", "Bot", "Infiltration"])

    # Load fitted preprocessor to get feature names
    import joblib
    preprocessor = joblib.load("artifacts/models/flow_preprocessor.joblib")
    feature_names = preprocessor.feature_columns

    # Figure 1: Dataset Distribution
    all_families = list(families_train) + list(families_val) + list(families_test)
    fam_counts = pd.Series(all_families).value_counts().to_dict()
    visualiser.plot_dataset_distribution(fam_counts, save_name="fig1_dataset_distribution.png")
    print("[+] Saved Figure 1: Dataset Distribution")

    # -------------------------------------------------------------------------
    # PHASE 4 & 5: TRAIN BOTH MODELS
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   MODEL TRAINING: XGBoost Classifier & Tabular Variational Autoencoder")
    print("--------------------------------------------------------------------------------")

    # A. Train XGBoost
    print("[*] Training XGBoost Classifier on known attack families & benign flows...")
    xgb_wrapper = XGBoostClassifierWrapper(random_seed=42)
    xgb_train_summary = xgb_wrapper.train(
        X_train=X_train_unscaled,
        y_train=y_train,
        X_val=X_val_unscaled,
        y_val=y_val,
        feature_names=feature_names,
        optimize_threshold=True
    )
    xgb_wrapper.save("artifacts/models/xgboost_model.json")
    print(f"[+] XGBoost Training Complete ({xgb_train_summary['training_duration_seconds']}s). Selected Validation Threshold: {xgb_wrapper.threshold}")

    # B. Train VAE
    print("[*] Training Tabular VAE strictly on Benign Network Traffic...")
    train_benign_mask = (y_train == 0)
    val_benign_mask = (y_val == 0)
    X_train_benign = X_train_scaled[train_benign_mask]
    X_val_benign = X_val_scaled[val_benign_mask]

    vae = TabularVAE(
        input_dim=X_train_scaled.shape[1],
        latent_dim=16,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        dropout_rate=0.1
    )
    vae_trainer = VAETrainer(model=vae, beta_kl=0.005, learning_rate=0.001)
    vae_train_summary = vae_trainer.fit(
        X_train_benign=X_train_benign,
        X_val_benign=X_val_benign,
        epochs=15,
        batch_size=512,
        early_stopping_patience=4
    )

    # Calibrate VAE threshold strictly on benign validation scores (1.0% target FPR)
    vae_threshold = vae_trainer.calibrate_threshold(X_val_benign, target_fpr=1.0)
    vae_trainer.save_checkpoint("artifacts/models/vae_checkpoint.pt")
    print(f"[+] VAE Training Complete ({vae_train_summary['training_duration_seconds']}s). Calibrated 1% FPR Threshold: {round(vae_threshold, 6)}")

    # -------------------------------------------------------------------------
    # EXPERIMENT A: KNOWN ATTACK DETECTION BENCHMARK
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   EXPERIMENT A: KNOWN ATTACK DETECTION (Seen Families + Benign)")
    print("--------------------------------------------------------------------------------")
    # Subsetting Test Set to Benign + Known Families
    known_mask = np.isin(families_test, ["BENIGN", "DoS", "PortScan", "Brute_Force", "Heartbleed"])
    X_test_known_xgb = X_test_unscaled[known_mask]
    X_test_known_vae = X_test_scaled[known_mask]
    y_test_known = y_test[known_mask]
    fam_test_known = families_test[known_mask]

    # Predict XGBoost
    xgb_known_probs = xgb_wrapper.predict_proba(X_test_known_xgb)
    xgb_known_preds = (xgb_known_probs >= xgb_wrapper.threshold).astype(int)
    xgb_known_metrics = calculate_classification_metrics(y_test_known, xgb_known_preds, xgb_known_probs)

    # Predict VAE
    vae_known_scores = vae_trainer.compute_anomaly_scores(X_test_known_vae)
    vae_known_preds = (vae_known_scores >= vae_threshold).astype(int)
    vae_known_metrics = calculate_classification_metrics(y_test_known, vae_known_preds, vae_known_scores)

    # Latency Benchmark
    xgb_latency = benchmark_inference_latency(xgb_wrapper.predict_proba, X_test_known_xgb[:10000])
    vae_latency = benchmark_inference_latency(vae_trainer.compute_anomaly_scores, X_test_known_vae[:10000])

    print(f"[+] XGBoost (Known Attacks): PR-AUC={xgb_known_metrics['pr_auc']}, Recall={xgb_known_metrics['recall']}, FPR={xgb_known_metrics['false_positive_rate']}, Latency={xgb_latency['ms_per_sample']}ms/sample")
    print(f"[+] VAE     (Known Attacks): PR-AUC={vae_known_metrics['pr_auc']}, Recall={vae_known_metrics['recall']}, FPR={vae_known_metrics['false_positive_rate']}, Latency={vae_latency['ms_per_sample']}ms/sample")

    # Figure 2 & 3
    visualiser.plot_roc_and_pr_curves(y_test_known, xgb_known_probs, vae_known_scores, save_name="fig2_roc_pr_curves.png")
    visualiser.plot_confusion_matrices(y_test_known, xgb_known_preds, vae_known_preds, save_name="fig3_confusion_matrices.png")
    print("[+] Saved Figure 2 (ROC/PR Curves) & Figure 3 (Confusion Matrices)")

    # -------------------------------------------------------------------------
    # EXPERIMENT B: HELD-OUT ATTACK-FAMILY DETECTION (Zero-Day Evaluation)
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   EXPERIMENT B: HELD-OUT ATTACK-FAMILY DETECTION (Completely Unseen Families)")
    print("--------------------------------------------------------------------------------")
    # Subsetting Test Set to Benign + Held-Out Families
    held_out_mask = np.isin(families_test, ["BENIGN"] + held_out_families)
    X_test_ho_xgb = X_test_unscaled[held_out_mask]
    X_test_ho_vae = X_test_scaled[held_out_mask]
    y_test_ho = y_test[held_out_mask]
    fam_test_ho = families_test[held_out_mask]

    xgb_ho_probs = xgb_wrapper.predict_proba(X_test_ho_xgb)
    xgb_ho_preds = (xgb_ho_probs >= xgb_wrapper.threshold).astype(int)
    xgb_ho_metrics = calculate_classification_metrics(y_test_ho, xgb_ho_preds, xgb_ho_probs)

    vae_ho_scores = vae_trainer.compute_anomaly_scores(X_test_ho_vae)
    vae_ho_preds = (vae_ho_scores >= vae_threshold).astype(int)
    vae_ho_metrics = calculate_classification_metrics(y_test_ho, vae_ho_preds, vae_ho_scores)

    # Disaggregated Per-Family Recall across Entire Test Set
    xgb_full_preds = (xgb_wrapper.predict_proba(X_test_unscaled) >= xgb_wrapper.threshold).astype(int)
    vae_full_preds = (vae_trainer.compute_anomaly_scores(X_test_scaled) >= vae_threshold).astype(int)

    per_family_xgb = calculate_per_family_recall(y_test, xgb_full_preds, families_test)
    per_family_vae = calculate_per_family_recall(y_test, vae_full_preds, families_test)

    print("[+] Disaggregated Per-Family Detection Performance:")
    print("--------------------------------------------------------------------------------")
    print(f"{'Attack Family':<18} | {'Type':<10} | {'Count':<8} | {'XGBoost Recall':<15} | {'VAE Recall':<12}")
    print("--------------------------------------------------------------------------------")
    for fam in sorted(per_family_xgb.keys()):
        if fam == "BENIGN":
            continue
        fam_type = "Held-Out" if fam in held_out_families else "Known"
        count = per_family_xgb[fam]["total_samples"]
        r_xgb = per_family_xgb[fam]["recall"]
        r_vae = per_family_vae.get(fam, {}).get("recall", 0.0)
        print(f"{fam:<18} | {fam_type:<10} | {count:<8} | {r_xgb:<15.4f} | {r_vae:<12.4f}")
    print("--------------------------------------------------------------------------------")

    # Figure 4: Anomaly Score Distributions
    benign_scores = vae_known_scores[y_test_known == 0]
    known_attack_scores = vae_known_scores[y_test_known == 1]
    held_out_attack_scores = vae_ho_scores[y_test_ho == 1]
    visualiser.plot_anomaly_score_distribution(
        benign_scores=benign_scores,
        known_attack_scores=known_attack_scores,
        held_out_attack_scores=held_out_attack_scores,
        threshold=vae_threshold,
        save_name="fig4_anomaly_score_distributions.png"
    )

    # Figure 5: Per-Family Recall Comparison
    visualiser.plot_per_family_recall(per_family_xgb, per_family_vae, save_name="fig5_per_family_recall.png")
    print("[+] Saved Figure 4 (Anomaly Score Densities) & Figure 5 (Per-Family Recall)")

    # -------------------------------------------------------------------------
    # EXPERIMENT C: FALSE-ALARM & THRESHOLD SENSITIVITY ANALYSIS
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   EXPERIMENT C: FALSE-ALARM TOLERANCE & THRESHOLD SENSITIVITY")
    print("--------------------------------------------------------------------------------")
    val_benign_scores = vae_trainer.compute_anomaly_scores(X_val_benign)
    sensitivity_records = sweep_threshold_sensitivity(
        y_test=y_test_ho,
        test_scores=vae_ho_scores,
        benign_val_scores=val_benign_scores,
        target_fpr_sweep=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    print(f"{'Target Benign FPR (%)':<24} | {'Threshold':<12} | {'Test FPR':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("--------------------------------------------------------------------------------")
    for rec in sensitivity_records:
        print(f"{rec['target_fpr_percent']:<24.1f} | {rec['calibrated_threshold']:<12.6f} | {rec['test_fpr']:<10.4f} | {rec['test_recall']:<10.4f} | {rec['test_f1']:<10.4f}")
    print("--------------------------------------------------------------------------------")

    # Figure 6
    visualiser.plot_threshold_sensitivity(sensitivity_records, save_name="fig6_threshold_sensitivity.png")
    print("[+] Saved Figure 6 (Threshold Sensitivity Trade-Offs)")

    # -------------------------------------------------------------------------
    # EXPERIMENT D: DISTRIBUTION-SHIFT STRESS TESTING
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   EXPERIMENT D: DISTRIBUTION-SHIFT STRESS TESTING")
    print("--------------------------------------------------------------------------------")
    scaling_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    xgb_shift_pr_aucs = []
    vae_shift_pr_aucs = []

    for factor in scaling_factors:
        # Shift unscaled for XGBoost and log1p-scaled for VAE
        X_test_shifted_xgb = apply_feature_shift(X_test_unscaled, feature_names, shift_type="packet_size_scale", severity=factor)
        X_test_shifted_log = np.log1p(np.maximum(0, X_test_shifted_xgb))
        X_test_shifted_vae = np.clip(preprocessor.scaler.transform(X_test_shifted_log), -5.0, 5.0).astype(np.float32)

        probs_shift_xgb = xgb_wrapper.predict_proba(X_test_shifted_xgb)
        scores_shift_vae = vae_trainer.compute_anomaly_scores(X_test_shifted_vae)

        pr_xgb = average_precision_score(y_test, probs_shift_xgb)
        pr_vae = average_precision_score(y_test, scores_shift_vae)

        xgb_shift_pr_aucs.append(round(float(pr_xgb), 4))
        vae_shift_pr_aucs.append(round(float(pr_vae), 4))

    print(f"Packet Size Scale Factors: {scaling_factors}")
    print(f"XGBoost PR-AUC Degradation: {xgb_shift_pr_aucs}")
    print(f"VAE     PR-AUC Degradation: {vae_shift_pr_aucs}")

    visualiser.plot_distribution_shift_degradation(
        severities=scaling_factors,
        xgb_degradations=xgb_shift_pr_aucs,
        vae_degradations=vae_shift_pr_aucs,
        shift_label="Packet Length Scaling Multiplier",
        save_name="fig7_distribution_shift_degradation.png"
    )
    print("[+] Saved Figure 7 (Distribution Shift Degradation)")

    # -------------------------------------------------------------------------
    # EXPERIMENT F: COMPLEMENTARITY ANALYSIS (Disagreement Analysis)
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   EXPERIMENT F: COMPLEMENTARITY & DISAGREEMENT ANALYSIS")
    print("--------------------------------------------------------------------------------")
    # Subsetting exclusively to held-out attack instances
    held_out_attacks_mask = (y_test == 1) & np.isin(families_test, held_out_families)
    preds_xgb_attacks = xgb_full_preds[held_out_attacks_mask]
    preds_vae_attacks = vae_full_preds[held_out_attacks_mask]
    n_held_out_attacks = int(np.sum(held_out_attacks_mask))

    both_detected = int(np.sum((preds_xgb_attacks == 1) & (preds_vae_attacks == 1)))
    xgb_only = int(np.sum((preds_xgb_attacks == 1) & (preds_vae_attacks == 0)))
    vae_only = int(np.sum((preds_xgb_attacks == 0) & (preds_vae_attacks == 1)))
    both_missed = int(np.sum((preds_xgb_attacks == 0) & (preds_vae_attacks == 0)))

    print(f"Total Held-Out Attack Instances: {n_held_out_attacks:,}")
    print(f"  - Detected by Both Models:       {both_detected:,} ({round(both_detected/n_held_out_attacks*100, 2)}%)")
    print(f"  - Detected by XGBoost ONLY:      {xgb_only:,} ({round(xgb_only/n_held_out_attacks*100, 2)}%)")
    print(f"  - Detected by VAE ONLY:          {vae_only:,} ({round(vae_only/n_held_out_attacks*100, 2)}%)")
    print(f"  - Missed by Both Approaches:     {both_missed:,} ({round(both_missed/n_held_out_attacks*100, 2)}%)")

    visualiser.plot_model_disagreement(
        both_detected=both_detected,
        xgb_only=xgb_only,
        vae_only=vae_only,
        both_missed=both_missed,
        save_name="fig8_model_disagreement.png"
    )
    print("[+] Saved Figure 8 (Model Disagreement & Complementarity)")

    # -------------------------------------------------------------------------
    # STATISTICAL EVALUATION: BOOTSTRAP CONFIDENCE INTERVALS & HYPOTHESIS TESTING
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("   STATISTICAL VALIDATION: BOOTSTRAP 95% CIs & PAIRED SIGNIFICANCE")
    print("--------------------------------------------------------------------------------")
    # Bootstrap CI for Known Attack PR-AUC and Recall
    f1_xgb_ci = bootstrap_metric_ci(y_test_known, xgb_known_preds, f1_score, n_bootstraps=300)
    f1_vae_ci = bootstrap_metric_ci(y_test_known, vae_known_preds, f1_score, n_bootstraps=300)
    paired_f1_known = paired_bootstrap_comparison(y_test_known, xgb_known_preds, vae_known_preds, f1_score, n_bootstraps=300)

    # Bootstrap CI for Held-Out Attack Recall
    rec_xgb_ho_ci = bootstrap_metric_ci(y_test_ho, xgb_ho_preds, recall_score, n_bootstraps=300)
    rec_vae_ho_ci = bootstrap_metric_ci(y_test_ho, vae_ho_preds, recall_score, n_bootstraps=300)
    paired_rec_ho = paired_bootstrap_comparison(y_test_ho, xgb_ho_preds, vae_ho_preds, recall_score, n_bootstraps=300)

    print(f"Known Attacks F1 [XGBoost]: {f1_xgb_ci['point_estimate']} (95% CI: [{f1_xgb_ci['ci_lower']}, {f1_xgb_ci['ci_upper']}])")
    print(f"Known Attacks F1 [VAE]:     {f1_vae_ci['point_estimate']} (95% CI: [{f1_vae_ci['ci_lower']}, {f1_vae_ci['ci_upper']}])")
    print(f"  -> Delta F1 (XGB - VAE):  {paired_f1_known['delta_point']} (p={paired_f1_known['p_value_approx']}, Significant: {paired_f1_known['significant_at_95']})")

    print(f"Held-Out Recall [XGBoost]:  {rec_xgb_ho_ci['point_estimate']} (95% CI: [{rec_xgb_ho_ci['ci_lower']}, {rec_xgb_ho_ci['ci_upper']}])")
    print(f"Held-Out Recall [VAE]:      {rec_vae_ho_ci['point_estimate']} (95% CI: [{rec_vae_ho_ci['ci_lower']}, {rec_vae_ho_ci['ci_upper']}])")
    print(f"  -> Delta Rec (XGB - VAE): {paired_rec_ho['delta_point']} (p={paired_rec_ho['p_value_approx']}, Significant: {paired_rec_ho['significant_at_95']})")

    # -------------------------------------------------------------------------
    # COMPILE AND SAVE MASTER RESULTS SUMMARY
    # -------------------------------------------------------------------------
    master_results = {
        "dataset": "CICIDS2017",
        "total_evaluated_records": len(y_test),
        "experiment_a_known": {
            "xgboost": {**xgb_known_metrics, "latency": xgb_latency, "f1_ci": f1_xgb_ci},
            "vae": {**vae_known_metrics, "latency": vae_latency, "f1_ci": f1_vae_ci},
            "paired_comparison": paired_f1_known
        },
        "experiment_b_held_out": {
            "held_out_families": held_out_families,
            "xgboost_summary": {**xgb_ho_metrics, "recall_ci": rec_xgb_ho_ci},
            "vae_summary": {**vae_ho_metrics, "recall_ci": rec_vae_ho_ci},
            "paired_comparison": paired_rec_ho,
            "per_family_results": {
                fam: {
                    "count": per_family_xgb[fam]["total_samples"],
                    "xgboost_recall": per_family_xgb[fam]["recall"],
                    "vae_recall": per_family_vae.get(fam, {}).get("recall", 0.0)
                } for fam in per_family_xgb.keys() if fam != "BENIGN"
            }
        },
        "experiment_c_sensitivity": sensitivity_records,
        "experiment_d_distribution_shift": {
            "scaling_factors": scaling_factors,
            "xgboost_pr_aucs": xgb_shift_pr_aucs,
            "vae_pr_aucs": vae_shift_pr_aucs
        },
        "experiment_f_complementarity": {
            "total_held_out_attacks": n_held_out_attacks,
            "both_detected": both_detected,
            "xgboost_only": xgb_only,
            "vae_only": vae_only,
            "both_missed": both_missed
        }
    }

    with open(results_dir / "master_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=4)

    # Save summary table as Markdown in reports/tables/
    table_path = tables_dir / "benchmark_comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Master Experimental Comparison Table\n\n")
        f.write("| Metric | XGBoost (Supervised) | VAE (Benign Anomaly) | Difference (Δ) | Statistical Significance |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Known Attack PR-AUC** | {xgb_known_metrics['pr_auc']:.4f} | {vae_known_metrics['pr_auc']:.4f} | {xgb_known_metrics['pr_auc'] - vae_known_metrics['pr_auc']:.4f} | Baseline Comparison |\n")
        f.write(f"| **Known Attack F1-Score** | {xgb_known_metrics['f1_score']:.4f} | {vae_known_metrics['f1_score']:.4f} | {paired_f1_known['delta_point']:.4f} | 95% CI: [{paired_f1_known['ci_lower']}, {paired_f1_known['ci_upper']}], p={paired_f1_known['p_value_approx']} |\n")
        f.write(f"| **Known Attack Recall** | {xgb_known_metrics['recall']:.4f} | {vae_known_metrics['recall']:.4f} | {xgb_known_metrics['recall'] - vae_known_metrics['recall']:.4f} | -- |\n")
        f.write(f"| **Known Attack FPR** | {xgb_known_metrics['false_positive_rate']:.4f} | {vae_known_metrics['false_positive_rate']:.4f} | {xgb_known_metrics['false_positive_rate'] - vae_known_metrics['false_positive_rate']:.4f} | Target: 1.0% |\n")
        f.write(f"| **Held-Out Family PR-AUC** | {xgb_ho_metrics['pr_auc']:.4f} | {vae_ho_metrics['pr_auc']:.4f} | {xgb_ho_metrics['pr_auc'] - vae_ho_metrics['pr_auc']:.4f} | Zero-Day Benchmark |\n")
        f.write(f"| **Held-Out Family Recall** | {xgb_ho_metrics['recall']:.4f} | {vae_ho_metrics['recall']:.4f} | {paired_rec_ho['delta_point']:.4f} | 95% CI: [{paired_rec_ho['ci_lower']}, {paired_rec_ho['ci_upper']}], p={paired_rec_ho['p_value_approx']} |\n")
        f.write(f"| **Inference Latency** | {xgb_latency['ms_per_sample']:.4f} ms/flow | {vae_latency['ms_per_sample']:.4f} ms/flow | -- | CPU Throughput Benchmark |\n")
    print(f"[+] Saved master benchmark comparison table to {table_path}")

    print("\n================================================================================")
    print("   ALL PHASES 4 - 9 EXPERIMENTS & VISUALISATIONS COMPLETED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    run_master_experiments()
