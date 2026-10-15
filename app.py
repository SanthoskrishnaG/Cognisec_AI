import json
import os
import sys
from pathlib import Path
import io
import streamlit as st

# Check for required ML dependencies before loading models
try:
    import torch
    import xgboost
    import joblib
    import numpy as np
    import pandas as pd
    from src.models.xgboost_model import XGBoostClassifierWrapper
    from src.models.vae_model import TabularVAE, VAETrainer
except ImportError as e:
    st.set_page_config(page_title="Cognisec AI - Environment Error", page_icon="⚠️")
    st.error(
        f"### ⚠️ Cognisec AI Environment Configuration Error\n\n"
        f"**Missing Dependency:** `{e}`\n\n"
        f"**Active Interpreter:** `{sys.executable}` (Python {sys.version.split()[0]})\n\n"
        f"PyTorch and XGBoost are installed in the project virtual environment (`.venv`).\n\n"
        f"**Recommended Launch Command:**\n"
        f"```bash\n"
        f"python main.py --step streamlit\n"
        f"# or\n"
        f".\\.venv\\Scripts\\python.exe -m streamlit run app.py\n"
        f"```"
    )
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Network Behaviour Fingerprinting Research",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Subtle, professional, light theme)
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 16px;
        color: #4a5568;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_models():
    preprocessor = joblib.load("artifacts/models/flow_preprocessor.joblib")
    xgb_wrapper = XGBoostClassifierWrapper().load("artifacts/models/xgboost_model.json")
    vae_trainer = VAETrainer.load_checkpoint("artifacts/models/vae_checkpoint.pt", device="cpu")
    with open("results/master_experiment_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return preprocessor, xgb_wrapper, vae_trainer, results

preprocessor, xgb_wrapper, vae_trainer, master_results = load_saved_models()

# Sidebar Navigation
st.sidebar.title("🛡️ Research Navigator")
st.sidebar.markdown("**Network Behaviour Fingerprinting**")
st.sidebar.caption("XGBoost vs. Variational Autoencoders for Unseen Cyberattack Detection")

menu = st.sidebar.radio(
    "Navigation Sections:",
    [
        "1. Executive Overview",
        "2. Dataset & Quality Audit",
        "3. Exp A: Known Attack Benchmark",
        "4. Exp B: Held-Out (Zero-Day) Detection",
        "5. Exp C: False-Alarm & Thresholds",
        "6. Exp D: Distribution Shift",
        "7. Exp F: Complementarity Analysis",
        "8. Interactive Flow Inference"
    ]
)

st.sidebar.divider()
st.sidebar.info("💡 **Dataset:** Authentic CICIDS2017 flow logs\n**Models:** Supervised XGBoost & Tabular PyTorch VAE\n**Hardware:** 16GB RAM + NVIDIA RTX 3050")

# -----------------------------------------------------------------------------
# 1. Executive Overview
# -----------------------------------------------------------------------------
if menu == "1. Executive Overview":
    st.markdown('<div class="main-title">Network Behaviour Fingerprinting for Unseen Attack Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An Experimental Comparison of Supervised Classification (XGBoost) and Benign Normal-Behaviour Modelling (PyTorch VAE)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Benchmark Dataset", "CICIDS2017", "757,646 flows")
    with col2:
        st.metric("Supervised Model", "XGBoost", "PR-AUC: 0.9999")
    with col3:
        st.metric("Anomaly Model", "Tabular PyTorch VAE", "Benign-Only Training")
    with col4:
        st.metric("Zero-Day Evaluation", "4 Held-Out Families", "132,209 unseen attacks")

    st.markdown("### Research Aim & Problem Statement")
    st.write("""
    Network intrusion detection systems (NIDS) routinely face two conflicting operational paradigms:
    1. **Supervised Classifiers (XGBoost):** Highly sensitive to known attack signatures with near-zero false alarms, but theoretically vulnerable to novel, unseen cyberattack families.
    2. **Normal-Behaviour Anomaly Models (Variational Autoencoders):** Trained exclusively on normal benign network flows to detect statistical deviations, offering potential zero-day detection at the cost of higher false-alarm rates.
    
    This research framework conducts a strict, leakage-controlled comparative benchmark using authentic **CICIDS2017** flow records.
    """)

    st.markdown("### Key Empirical Discoveries")
    st.info("""
    - **Known Attacks (Exp A):** XGBoost achieved **99.99% PR-AUC** with 0.09% false alarms. The VAE achieved **59.74% PR-AUC** when constrained to an operational 1% false alarm rate ($p < 0.001$).
    - **Held-Out Generalisation (Exp B):** XGBoost retained **62.5% overall recall** on unseen attack families because volumetric attacks (DDoS) shared statistical flow signatures with trained DoS attacks.
    - **Stealthy Infiltration Complementarity (Exp F):** For subtle attacks (Infiltration), XGBoost failed completely (0.0% recall), whereas the VAE detected **16.67% of infiltration instances**, proving targeted complementarity for low-footprint anomalies.
    """)

# -----------------------------------------------------------------------------
# 2. Dataset & Quality Audit
# -----------------------------------------------------------------------------
elif menu == "2. Dataset & Quality Audit":
    st.header("📊 Dataset Integrity & Leakage Prevention Audit")
    st.write("Authentic network flow records acquired from the official CICIDS2017 benchmark.")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig1_path = Path("reports/figures/fig1_dataset_distribution.png")
        if fig1_path.exists():
            st.image(str(fig1_path), caption="Figure 1: Authentic CICIDS2017 Flow Distribution by Family")
    with col2:
        with open("reports/data_audit_report.json", "r") as f:
            audit = json.load(f)
        st.markdown("#### Quality Audit Metrics")
        st.write(f"- **Total Ingested Records:** {audit['total_records']:,}")
        st.write(f"- **Total Flow Features:** {audit['total_features']}")
        st.write(f"- **Exact Duplicate Flows:** {audit['duplicate_records']:,} ({audit['duplicate_percentage']}%)")
        st.write(f"- **Columns with Infinite Values:** {len(audit['infinite_values'])} (Cleaned with median imputation)")
        st.write(f"- **Socket Leakage Columns Removed:** {', '.join(audit['leakage_columns_detected'])}")
        st.write(f"- **Retained Clean Features:** 70 flow behavioral metrics")

    st.subheader("Partitioning Manifest Summary")
    with open("data/processed/partition_manifest.json", "r") as f:
        manifest = json.load(f)
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Train Split (60%)", f"{manifest['train_summary']['total']:,}", "Benign + Known Attacks Only")
    p_col2.metric("Validation Split (20%)", f"{manifest['val_summary']['total']:,}", "Threshold Tuning")
    p_col3.metric("Test Split (20%)", f"{manifest['test_summary']['total']:,}", "Held-Out Families Reserved")

# -----------------------------------------------------------------------------
# 3. Exp A: Known Attack Benchmark
# -----------------------------------------------------------------------------
elif menu == "3. Exp A: Known Attack Benchmark":
    st.header("⚔️ Experiment A: Known Attack Detection Benchmark")
    st.write("Evaluation on held-out test traffic containing benign traffic and attack families represented during training (DoS, PortScan, Brute Force).")

    res_a = master_results["experiment_a_known"]
    col1, col2 = st.columns(2)
    with col1:
        fig2_path = Path("reports/figures/fig2_roc_pr_curves.png")
        if fig2_path.exists():
            st.image(str(fig2_path), caption="Figure 2: ROC and Precision-Recall Curves")
    with col2:
        fig3_path = Path("reports/figures/fig3_confusion_matrices.png")
        if fig3_path.exists():
            st.image(str(fig3_path), caption="Figure 3: Confusion Matrices (XGBoost vs VAE)")

    st.subheader("Statistical Metric Comparison")
    comp_df = pd.DataFrame([
        {
            "Metric": "PR-AUC",
            "XGBoost": f"{res_a['xgboost']['pr_auc']:.4f}",
            "VAE": f"{res_a['vae']['pr_auc']:.4f}",
            "Advantage": "XGBoost (+0.4026)"
        },
        {
            "Metric": "F1-Score (95% CI)",
            "XGBoost": f"{res_a['xgboost']['f1_score']:.4f} [{res_a['xgboost']['f1_ci']['ci_lower']}, {res_a['xgboost']['f1_ci']['ci_upper']}]",
            "VAE": f"{res_a['vae']['f1_score']:.4f} [{res_a['vae']['f1_ci']['ci_lower']}, {res_a['vae']['f1_ci']['ci_upper']}]",
            "Advantage": f"Δ = {res_a['paired_comparison']['delta_point']:.4f} (p < 0.001)"
        },
        {
            "Metric": "Detection Recall",
            "XGBoost": f"{res_a['xgboost']['recall']:.4f}",
            "VAE": f"{res_a['vae']['recall']:.4f}",
            "Advantage": "XGBoost (+0.9906)"
        },
        {
            "Metric": "False Positive Rate (FPR)",
            "XGBoost": f"{res_a['xgboost']['false_positive_rate']:.4f} (0.09%)",
            "VAE": f"{res_a['vae']['false_positive_rate']:.4f} (1.16%)",
            "Advantage": "XGBoost (Lower FPR)"
        },
        {
            "Metric": "Inference Latency",
            "XGBoost": f"{res_a['xgboost']['latency']['ms_per_sample']:.4f} ms/flow",
            "VAE": f"{res_a['vae']['latency']['ms_per_sample']:.4f} ms/flow",
            "Advantage": f"{res_a['xgboost']['latency']['throughput_samples_per_sec']:,} flows/sec"
        }
    ])
    st.table(comp_df)

# -----------------------------------------------------------------------------
# 4. Exp B: Held-Out (Zero-Day) Detection
# -----------------------------------------------------------------------------
elif menu == "4. Exp B: Held-Out (Zero-Day) Detection":
    st.header("🛡️ Experiment B: Unseen Held-Out Attack-Family Generalisation")
    st.write("Primary research evaluation: Models evaluated on attack families excluded entirely from training and threshold selection.")

    col1, col2 = st.columns(2)
    with col1:
        fig4_path = Path("reports/figures/fig4_anomaly_score_distributions.png")
        if fig4_path.exists():
            st.image(str(fig4_path), caption="Figure 4: VAE Latent Reconstruction Score Densities")
    with col2:
        fig5_path = Path("reports/figures/fig5_per_family_recall.png")
        if fig5_path.exists():
            st.image(str(fig5_path), caption="Figure 5: Disaggregated Recall by Attack Family")

    st.subheader("Per-Family Detection Disaggregation")
    fam_data = []
    ho_results = master_results["experiment_b_held_out"]["per_family_results"]
    for fam, d in ho_results.items():
        fam_type = "Held-Out (Unseen)" if fam in ["DDoS", "Web_Attack", "Bot", "Infiltration"] else "Known (Seen)"
        fam_data.append({
            "Attack Family": fam,
            "Type": fam_type,
            "Test Sample Count": f"{d['count']:,}",
            "XGBoost Recall": f"{d['xgboost_recall'] * 100:.2f}%",
            "VAE Recall": f"{d['vae_recall'] * 100:.2f}%"
        })
    st.dataframe(pd.DataFrame(fam_data), use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Exp C: False-Alarm & Thresholds
# -----------------------------------------------------------------------------
elif menu == "5. Exp C: False-Alarm & Thresholds":
    st.header("🎯 Experiment C: False-Alarm Tolerance & Threshold Sensitivity")
    st.write("Evaluating VAE operational trade-offs across pre-declared benign validation false-alarm thresholds.")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig6_path = Path("reports/figures/fig6_threshold_sensitivity.png")
        if fig6_path.exists():
            st.image(str(fig6_path), caption="Figure 6: Threshold Sensitivity Curves")
    with col2:
        sens_df = pd.DataFrame(master_results["experiment_c_sensitivity"])
        sens_df.columns = ["Target FPR (%)", "Calibrated Threshold", "Test FPR", "Recall", "Precision", "F1", "FP", "TP", "FN", "TN"]
        st.dataframe(sens_df, use_container_width=True)
        st.caption("Notice that increasing false-alarm budget from 1% to 10% only increases attack recall from 9.09% to 9.26%, demonstrating severe anomaly score overlap on subtle network flows.")

# -----------------------------------------------------------------------------
# 6. Exp D: Distribution Shift
# -----------------------------------------------------------------------------
elif menu == "6. Exp D: Distribution Shift":
    st.header("🌪️ Experiment D: Distribution-Shift Robustness")
    st.write("Model degradation analysis under feature distribution perturbations (packet size scaling multipliers).")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig7_path = Path("reports/figures/fig7_distribution_shift_degradation.png")
        if fig7_path.exists():
            st.image(str(fig7_path), caption="Figure 7: PR-AUC Degradation under Feature Perturbation")
    with col2:
        shift_data = master_results["experiment_d_distribution_shift"]
        df_shift = pd.DataFrame({
            "Packet Scaling Factor": shift_data["scaling_factors"],
            "XGBoost PR-AUC": shift_data["xgboost_pr_aucs"],
            "VAE PR-AUC": shift_data["vae_pr_aucs"]
        })
        st.dataframe(df_shift, use_container_width=True)
        st.caption("Supervised decision trees remain resilient to monotonic feature scalings, whereas the VAE reconstruction error shows progressive degradation.")

# -----------------------------------------------------------------------------
# 7. Exp F: Complementarity Analysis
# -----------------------------------------------------------------------------
elif menu == "7. Exp F: Complementarity Analysis":
    st.header("🔬 Experiment F: Complementarity & Disagreement Analysis")
    st.write("Disagreement breakdown on the 132,209 unseen held-out attack instances.")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig8_path = Path("reports/figures/fig8_model_disagreement.png")
        if fig8_path.exists():
            st.image(str(fig8_path), caption="Figure 8: Complementarity Breakdown on Held-Out Attacks")
    with col2:
        comp = master_results["experiment_f_complementarity"]
        total = comp["total_held_out_attacks"]
        st.markdown(f"### Disagreement Summary (Total = {total:,} attacks)")
        st.write(f"- **Both Models Detected:** {comp['both_detected']:,} ({comp['both_detected']/total*100:.2f}%)")
        st.write(f"- **XGBoost ONLY:** {comp['xgboost_only']:,} ({comp['xgboost_only']/total*100:.2f}%)")
        st.write(f"- **VAE ONLY:** {comp['vae_only']:,} ({comp['vae_only']/total*100:.2f}%)")
        st.write(f"- **Both Missed:** {comp['both_missed']:,} ({comp['both_missed']/total*100:.2f}%)")
        st.success("Key Finding: The VAE successfully detects stealthy Infiltration attacks that completely evade XGBoost.")

# -----------------------------------------------------------------------------
# 8. Interactive Flow Inference
# -----------------------------------------------------------------------------
elif menu == "8. Interactive Flow Inference":
    st.header("🧪 Interactive Offline Network-Flow Inference")
    st.write("Upload a compatible network-flow CSV file to execute real-time dual-model prediction using saved artifacts.")
    st.warning("⚠️ **Notice:** This interface performs offline batch inference on flow-record CSVs. It does not monitor live raw network interfaces.")

    uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, nrows=1000)
            st.write(f"Loaded {len(input_df)} flows. Validating schema...")

            # Validate schema
            missing_cols = [c for c in preprocessor.feature_columns if c not in input_df.columns]
            if len(missing_cols) > len(preprocessor.feature_columns) * 0.5:
                st.error(f"Incompatible CSV schema. Missing {len(missing_cols)} expected flow features.")
            else:
                X_unscaled, X_scaled, y_dummy, _ = preprocessor.transform(input_df)

                xgb_probs = xgb_wrapper.predict_proba(X_unscaled)
                xgb_preds = (xgb_probs >= xgb_wrapper.threshold).astype(int)

                vae_scores = vae_trainer.compute_anomaly_scores(X_scaled)
                vae_preds = (vae_scores >= vae_trainer.threshold).astype(int)

                results_df = pd.DataFrame({
                    "Flow_Index": np.arange(len(input_df)),
                    "XGBoost_Attack_Prob": np.round(xgb_probs, 4),
                    "XGBoost_Prediction": np.where(xgb_preds == 1, "🚨 Attack", "✅ Benign"),
                    "VAE_Anomaly_Score": np.round(vae_scores, 2),
                    "VAE_Threshold": round(vae_trainer.threshold, 2),
                    "VAE_Prediction": np.where(vae_preds == 1, "🚨 Anomaly", "✅ Normal")
                })

                st.subheader("Inference Results Preview")
                st.dataframe(results_df.head(20), use_container_width=True)

                col1, col2 = st.columns(2)
                col1.metric("XGBoost Flagged Attacks", int(np.sum(xgb_preds == 1)))
                col2.metric("VAE Flagged Anomalies", int(np.sum(vae_preds == 1)))

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Inference CSV",
                    data=csv_buffer.getvalue(),
                    file_name="inference_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error executing inference: {str(e)}")
