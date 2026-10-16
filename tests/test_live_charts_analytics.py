"""
Cognisec AI - Automated Verification Suite for Live Data-Driven Charts & Analytics.
Verifies:
1. Dynamic CSV upload analytics calculations (ground-truth vs predictions, VAE anomaly scores,
   prediction confidence histograms, cross-model disagreement, feature quality audit).
2. Unlabeled CSV handling (no fabricated performance metrics or invented attack families).
3. Differing CSV files produce distinct calculated values.
4. Live network monitoring chart analytics (volume over time, protocols, top suspicious flows,
   VAE threshold alignment at 0.330610, empty state when no traffic).
5. Historical research figures data provider (fig1 through fig8).
"""

import io
import pytest
import pandas as pd
import numpy as np
from src.api.server import create_app
from src.live.live_manager import get_live_manager


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def reset_live():
    mgr = get_live_manager()
    if mgr.is_running:
        mgr.stop()
    mgr.flow_buffer.clear()
    mgr.packets_seen = 0
    mgr.flows_processed = 0
    mgr.benign_count = 0
    mgr.suspicious_count = 0
    mgr.validation_errors.clear()
    yield mgr
    if mgr.is_running:
        mgr.stop()
    mgr.flow_buffer.clear()


# =============================================================================
# 1. Historical Research Figures Endpoint
# =============================================================================

def test_research_figures_endpoint(client):
    resp = client.get("/api/results/research-figures")
    assert resp.status_code == 200
    data = resp.get_json()

    for fig_key in [
        "fig1_dataset_distribution",
        "fig2_roc_pr_curves",
        "fig3_confusion_matrices",
        "fig4_anomaly_score_distribution",
        "fig5_per_family_recall",
        "fig6_threshold_sensitivity",
        "fig7_distribution_shift",
        "fig8_model_disagreement"
    ]:
        assert fig_key in data
        assert "artifact_url" in data[fig_key]
        assert data[fig_key]["artifact_url"].startswith("/figures/")

    # Verify Figure 1 counts
    assert "BENIGN" in data["fig1_dataset_distribution"]["family_counts"]
    # Verify Figure 2 ROC / PR metrics
    assert data["fig2_roc_pr_curves"]["metrics"]["xgboost"]["pr_auc"] >= 0.99
    assert data["fig2_roc_pr_curves"]["metrics"]["vae"]["pr_auc"] >= 0.90
    # Verify Figure 3 Confusion Matrices
    known_cm = data["fig3_confusion_matrices"]["known_benchmark"]
    assert known_cm["xgboost"]["tp"] == 85080
    assert known_cm["xgboost"]["tn"] == 39961
    assert known_cm["vae"]["tp"] == 22030
    assert known_cm["vae"]["tn"] == 39628


# =============================================================================
# 2. CSV Upload Live Analytics Tests
# =============================================================================

def test_csv_analytics_labeled_benign(client):
    """Test uploading labeled benign sample CSV."""
    df_benign = pd.read_csv("data/samples/sample_benign_flows.csv")
    csv_text = df_benign.to_csv(index=False)

    resp = client.post("/api/analyze/csv-analytics", json={"csv_text": csv_text})
    assert resp.status_code == 200
    res = resp.get_json()["data"]

    assert res["total_flows"] == len(df_benign)
    assert res["has_ground_truth"] is True
    assert "BENIGN" in res["traffic_family_distribution"]
    assert res["traffic_family_distribution"]["BENIGN"] == len(df_benign)

    # Anomaly score distribution
    vae_dist = res["anomaly_score_distribution"]
    assert vae_dist["calibrated_threshold"] == 0.330610
    assert len(vae_dist["histogram"]) == 15
    assert vae_dist["min"] >= 0.0

    # Predictions
    pred = res["prediction_distribution"]
    assert pred["xgboost"]["benign"] == len(df_benign)
    assert pred["xgboost"]["attack"] == 0

    # Disagreement
    disag = res["model_disagreement"]
    assert "both_normal" in disag
    assert disag["both_normal"] + disag["xgboost_only"] + disag["vae_only"] + disag["both_flagged"] == len(df_benign)

    # Ground truth evaluation metrics
    eval_m = res["ground_truth_evaluation"]
    assert eval_m is not None
    assert eval_m["xgboost"]["accuracy"] == 1.0
    assert eval_m["xgboost"]["false_positive_rate"] == 0.0

    # Test retrieval by analysis_id
    aid = res["analysis_id"]
    get_resp = client.get(f"/api/analytics/upload/{aid}")
    assert get_resp.status_code == 200
    assert get_resp.get_json()["data"]["analysis_id"] == aid


def test_csv_analytics_differing_files(client):
    """Verify that uploading two different CSV files produces different chart numbers."""
    df_benign = pd.read_csv("data/samples/sample_benign_flows.csv")
    df_attacks = pd.read_csv("data/samples/sample_known_attacks.csv")

    res_b = client.post("/api/analyze/csv-analytics", json={"csv_text": df_benign.to_csv(index=False)}).get_json()["data"]
    res_a = client.post("/api/analyze/csv-analytics", json={"csv_text": df_attacks.to_csv(index=False)}).get_json()["data"]

    # Compare predictions
    assert res_b["prediction_distribution"]["xgboost"]["attack"] != res_a["prediction_distribution"]["xgboost"]["attack"]
    assert res_b["prediction_distribution"]["xgboost"]["attack"] == 0
    assert res_a["prediction_distribution"]["xgboost"]["attack"] > 0

    # Compare VAE mean anomaly score
    assert res_a["anomaly_score_distribution"]["mean"] > res_b["anomaly_score_distribution"]["mean"]


def test_csv_analytics_unlabeled_csv(client):
    """Verify that an unlabeled CSV shows 'Ground truth unavailable' and does not fabricate metrics."""
    df_benign = pd.read_csv("data/samples/sample_benign_flows.csv")
    df_unlabeled = df_benign.drop(columns=["Label"])
    assert "Label" not in df_unlabeled.columns

    resp = client.post("/api/analyze/csv-analytics", json={"csv_text": df_unlabeled.to_csv(index=False)})
    assert resp.status_code == 200
    res = resp.get_json()["data"]

    assert res["has_ground_truth"] is False
    assert "Ground truth unavailable" in res["ground_truth_status"]
    assert res["traffic_family_distribution"] is None
    assert res["ground_truth_evaluation"] is None

    # Predictions and anomaly scores must still be authentically computed
    assert res["prediction_distribution"]["xgboost"]["threshold"] == 0.52
    assert res["anomaly_score_distribution"]["calibrated_threshold"] == 0.330610
    assert len(res["anomaly_score_distribution"]["histogram"]) == 15


# =============================================================================
# 3. Live Monitoring Analytics Tests
# =============================================================================

def test_live_analytics_empty_state(client, reset_live):
    """Test live analytics when no traffic has arrived (must not show fake data)."""
    resp = client.get("/api/analytics/live")
    assert resp.status_code == 200
    res = resp.get_json()["data"]

    assert res["has_traffic"] is False
    assert res["total_flows"] == 0
    assert res["total_packets"] == 0
    assert len(res["traffic_volume_over_time"]) == 0
    assert res["normal_vs_suspicious"]["normal"] == 0
    assert res["normal_vs_suspicious"]["suspicious"] == 0
    assert res["calibrated_thresholds"]["vae"] == 0.330610
    assert res["calibrated_thresholds"]["xgboost"] == 0.52


def test_live_analytics_with_actual_flows(client, reset_live):
    """Test live analytics with actual completed flows in the manager buffer."""
    mgr = reset_live

    # Inject simulated flows with real inference results into the buffer
    t_now = "2026-10-16 10:30:00"
    mgr.flow_buffer.append({
        "flow_id": "test-flow-1",
        "timestamp": t_now,
        "src_ip": "192.168.1.10",
        "src_port": 49152,
        "dst_ip": "172.217.16.206",
        "dst_port": 443,
        "protocol": "TCP",
        "total_packets": 12,
        "total_bytes": 1450,
        "xgboost": {"probability": 0.02, "is_attack": False, "prediction": "Benign"},
        "vae": {"anomaly_score": 0.085, "calibrated_threshold": 0.330610, "is_anomaly": False, "prediction": "Normal"},
        "joint_decision": "Normal Network Behavior",
        "is_suspicious": False
    })
    mgr.flow_buffer.append({
        "flow_id": "test-flow-2",
        "timestamp": t_now,
        "src_ip": "192.168.1.55",
        "src_port": 52100,
        "dst_ip": "10.0.0.5",
        "dst_port": 80,
        "protocol": "TCP",
        "total_packets": 240,
        "total_bytes": 28000,
        "xgboost": {"probability": 0.94, "is_attack": True, "prediction": "Attack"},
        "vae": {"anomaly_score": 0.620, "calibrated_threshold": 0.330610, "is_anomaly": True, "prediction": "Anomaly"},
        "joint_decision": "High-Confidence Attack (Both Flagged)",
        "is_suspicious": True
    })
    mgr.flows_processed = 2
    mgr.packets_seen = 252
    mgr.benign_count = 1
    mgr.suspicious_count = 1

    resp = client.get("/api/analytics/live")
    assert resp.status_code == 200
    res = resp.get_json()["data"]

    assert res["has_traffic"] is True
    assert res["total_flows"] == 2
    assert res["total_packets"] == 252
    assert res["normal_vs_suspicious"]["normal"] == 1
    assert res["normal_vs_suspicious"]["suspicious"] == 1
    assert res["normal_vs_suspicious"]["suspicious_pct"] == 50.0
    assert res["protocol_distribution"]["TCP"] == 2

    # Top suspicious flow ranking
    assert len(res["top_suspicious_flows"]) == 2
    top = res["top_suspicious_flows"][0]
    assert top["flow_id"] == "test-flow-2"
    assert top["is_suspicious"] is True
    assert top["risk_score"] > 0.9

    # VAE anomaly score stream with threshold
    assert len(res["vae_scores_over_time"]) == 2
    assert res["calibrated_thresholds"]["vae"] == 0.330610
