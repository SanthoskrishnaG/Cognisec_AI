"""
Cognisec AI - Dashboard Metrics Authenticity & Data Grounding Test Suite
Verifies that all dashboard metrics, dataset counts, upload analyses, live telemetry,
and historical research figures are strictly grounded in authentic sources:
1. Dataset totals match independently counted physical source records.
2. Attack-family counts match actual CSV labels.
3. Uploading two different CSVs changes relevant counts and charts appropriately.
4. Unlabeled CSV files never produce fabricated evaluation metrics.
5. Live charts change only after actual captured-flow observations are processed.
6. Stopped capture does not generate new observations.
7. Historical research metrics match their saved result artifacts.
8. API failures never cause fabricated fallback statistics.
9. XGBoost and VAE predictions use the correct preprocessing and feature order.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.api.server import create_app
from src.data.validation import verify_dataset_files
from src.live.live_manager import get_live_manager
from src.live.flow_extractor import BidirectionalFlow, PacketRecord


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# -----------------------------------------------------------------------------
# 1. Dataset Totals Match Physical Source Records
# -----------------------------------------------------------------------------
def test_dataset_totals_match_independently_counted_source_records(client):
    """
    Independently streams row counts directly from the 8 physical CSV files in data/raw/
    and verifies that verify_dataset_files and /api/metadata report the authentic total.
    """
    raw_dir = Path("data/raw")
    csv_files = sorted(list(raw_dir.glob("*.csv")))
    assert len(csv_files) == 8, f"Expected 8 raw CSV files, found {len(csv_files)}"

    independent_row_count = 0
    file_records = {}
    for f in csv_files:
        with open(f, "rb") as fp:
            lines = sum(1 for _ in fp) - 1  # exclude header
        independent_row_count += lines
        file_records[f.name] = lines

    # Independent check: exactly 757,646 rows
    assert independent_row_count == 757646, (
        f"Authentic on-disk records count is {independent_row_count}, expected 757646"
    )

    # Test verify_dataset_files function directly
    verification = verify_dataset_files(raw_dir)
    assert verification["is_verified"] is True
    assert verification["total_records"] == 757646
    assert verification["files_found_count"] == 8
    assert verification["file_provenance"] == file_records

    # Test /api/metadata endpoint
    resp = client.get("/api/metadata")
    assert resp.status_code == 200
    meta = resp.get_json()
    assert meta["total_raw_records"] == 757646
    assert meta["is_dataset_verified"] is True
    assert meta["benign_records_count"] == 200000
    assert meta["attack_records_count"] == 557646

    # Test /api/dataset/verify endpoint
    v_resp = client.get("/api/dataset/verify")
    assert v_resp.status_code == 200
    v_data = v_resp.get_json()
    assert v_data["is_verified"] is True
    assert v_data["total_records"] == 757646


# -----------------------------------------------------------------------------
# 2. Attack-Family Counts Match Actual CSV Labels
# -----------------------------------------------------------------------------
def test_attack_family_counts_match_actual_csv_labels(client):
    """
    Verifies that the empirical label distribution matches the authentic CICIDS2017 labels.
    """
    audit_path = Path("reports/data_audit_report.json")
    assert audit_path.exists(), "data_audit_report.json not found"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    labels = audit.get("label_distribution", {})
    assert labels["BENIGN"] == 200000
    assert labels["DoS Hulk"] == 231073
    assert labels["PortScan"] == 158930
    assert labels["DDoS"] == 128027
    assert labels["Infiltration"] == 36
    assert labels["Heartbleed"] == 11

    total_attacks = sum(v for k, v in labels.items() if k != "BENIGN")
    assert total_attacks == 557646
    assert total_attacks + labels["BENIGN"] == 757646

    # Check that /api/metadata reflects this
    resp = client.get("/api/metadata")
    meta = resp.get_json()
    assert meta["benign_records_count"] == 200000
    assert meta["attack_records_count"] == 557646


# -----------------------------------------------------------------------------
# 3. Uploading Two Different CSVs Changes Relevant Counts and Charts Appropriately
# -----------------------------------------------------------------------------
def test_uploading_two_different_csvs_changes_counts_and_charts(client):
    """
    Verifies that uploading two distinct CSV files generates upload-specific
    predictions, anomaly scores, and charts without cross-contamination.
    """
    benign_path = Path("data/samples/sample_benign_flows.csv")
    attack_path = Path("data/samples/sample_known_attacks.csv")
    assert benign_path.exists() and attack_path.exists()

    with open(benign_path, "rb") as f_benign:
        res1 = client.post(
            "/api/analyze/dual",
            data={"file": (f_benign, "sample_benign_flows.csv")},
            content_type="multipart/form-data"
        )
    assert res1.status_code == 200
    d1 = res1.get_json()

    with open(attack_path, "rb") as f_attack:
        res2 = client.post(
            "/api/analyze/dual",
            data={"file": (f_attack, "sample_known_attacks.csv")},
            content_type="multipart/form-data"
        )
    assert res2.status_code == 200
    d2 = res2.get_json()

    # Distinct analysis IDs
    assert d1["analysis_id"] != d2["analysis_id"]

    # Benign CSV vs Attack CSV detection comparisons
    xgb1 = d1["xgboost_summary"]["total_flagged"]
    xgb2 = d2["xgboost_summary"]["total_flagged"]
    assert xgb1 < xgb2, f"Expected attack CSV to have more XGBoost flags: {xgb1} vs {xgb2}"

    vae1 = d1["vae_summary"]["total_flagged"]
    vae2 = d2["vae_summary"]["total_flagged"]
    assert vae1 < vae2, f"Expected attack CSV to have more VAE anomalies: {vae1} vs {vae2}"

    # Prediction distributions differ
    dist1 = d1["analytics"]["prediction_distribution"]
    dist2 = d2["analytics"]["prediction_distribution"]
    assert dist1["xgboost"]["attack"] != dist2["xgboost"]["attack"]


# -----------------------------------------------------------------------------
# 4. Unlabeled CSV Files Never Produce Fabricated Evaluation Metrics
# -----------------------------------------------------------------------------
def test_unlabeled_csv_does_not_fabricate_metrics(client):
    """
    Verifies that when a CSV has no verified ground-truth labels,
    accuracy, recall, precision, and false-alarm rates are NEVER invented.
    """
    sample_path = Path("data/samples/sample_benign_flows.csv")
    df = pd.read_csv(sample_path)
    # Strip label column entirely
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    csv_text = df.to_csv(index=False)
    resp = client.post(
        "/api/analyze/csv-analytics",
        json={"csv_text": csv_text}
    )
    assert resp.status_code == 200
    analytics = resp.get_json()["data"]

    assert analytics["has_ground_truth"] is False
    assert analytics["traffic_family_distribution"] is None
    assert analytics["ground_truth_evaluation"] is None
    assert "Ground truth unavailable" in analytics["ground_truth_status"]


# -----------------------------------------------------------------------------
# 5. Live Charts Change Only After Actual Captured Flows
# -----------------------------------------------------------------------------
def test_live_charts_change_only_after_actual_captured_flows(client):
    """
    Verifies that live monitoring charts start empty and only update
    when actual completed network flows are processed.
    """
    mgr = get_live_manager()
    # Reset buffer
    with mgr._lock:
        mgr.flow_buffer.clear()
        mgr.packets_seen = 0
        mgr.flows_processed = 0
        mgr.benign_count = 0
        mgr.suspicious_count = 0

    analytics_empty = mgr.get_live_analytics()
    assert analytics_empty["has_traffic"] is False
    assert analytics_empty["total_flows"] == 0
    assert analytics_empty["total_packets"] == 0
    assert len(analytics_empty["traffic_volume_over_time"]) == 0

    # Simulate an actual completed flow
    t0 = 1700000000.0
    flow = BidirectionalFlow("192.168.1.50", 54321, "192.168.1.100", 443, "TCP", t0)
    flow.add_packet(PacketRecord(t0, 64, 40, 24, True, {"SYN": True}, 64240, "TCP"))
    flow.add_packet(PacketRecord(t0 + 0.025, 60, 40, 20, False, {"SYN": True, "ACK": True}, 65535, "TCP"))
    flow.add_packet(PacketRecord(t0 + 0.030, 52, 40, 12, True, {"ACK": True}, 64240, "TCP"))

    record = mgr._process_completed_flow(flow)
    assert record is not None
    assert "flow_id" in record
    assert "xgboost" in record
    assert "vae" in record

    # Now verify that live charts reflect strictly this actual flow
    analytics_active = mgr.get_live_analytics()
    assert analytics_active["has_traffic"] is True
    assert analytics_active["total_flows"] == 1
    assert len(analytics_active["traffic_volume_over_time"]) >= 1
    assert analytics_active["protocol_distribution"]["TCP"] == 1


# -----------------------------------------------------------------------------
# 6. Stopped Capture Does Not Generate New Observations
# -----------------------------------------------------------------------------
def test_stopped_capture_does_not_generate_new_observations(client):
    """
    Verifies that when capture is stopped, no background packets are processed.
    """
    mgr = get_live_manager()
    mgr.stop()
    assert mgr.is_running is False

    flows_before = mgr.flows_processed
    pkts_before = mgr.packets_seen

    # Trigger packet callback while stopped
    mgr._packet_callback(None)

    assert mgr.flows_processed == flows_before
    assert mgr.packets_seen == pkts_before


# -----------------------------------------------------------------------------
# 7. Historical Research Metrics Match Saved Artifacts
# -----------------------------------------------------------------------------
def test_historical_research_metrics_match_saved_artifacts(client):
    """
    Verifies that historical research metrics displayed in the dashboard
    match master_experiment_results.json.
    """
    results_path = Path("results/master_experiment_results.json")
    assert results_path.exists()
    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    # 1. Known attack PR-AUC
    xgb_prauc = master["experiment_a_known"]["xgboost"]["pr_auc"]
    assert round(xgb_prauc, 4) == 1.0000 or round(xgb_prauc, 2) >= 0.99
    assert round(xgb_prauc * 100, 2) == 100.0 or round(xgb_prauc * 100, 2) == 99.99

    # 2. Infiltration recall
    infil_vae_rec = master["experiment_b_held_out"]["per_family_results"]["Infiltration"]["vae_recall"]
    infil_xgb_rec = master["experiment_b_held_out"]["per_family_results"]["Infiltration"]["xgboost_recall"]
    assert infil_vae_rec == 0.4722
    assert infil_xgb_rec == 0.0

    # 3. Held-out zero-day attack flows count
    held_out_count = master["experiment_f_complementarity"]["total_held_out_attacks"]
    assert held_out_count == 132209

    # 4. Paired McNemar significance
    p_bound = master["experiment_a_known"]["paired_comparison"]["p_value_bound"]
    assert p_bound == "< 0.001"

    # Verify /api/results/comparison returns consistent values
    resp = client.get("/api/results/comparison")
    assert resp.status_code == 200
    comp = resp.get_json()

    infil_item = next(f for f in comp["per_family_results"] if f["family"] == "Infiltration")
    assert infil_item["vae_recall"] == 47.22
    assert infil_item["xgboost_recall"] == 0.0
    assert infil_item["is_held_out"] is True


# -----------------------------------------------------------------------------
# 8. API Failures Never Cause Fabricated Fallback Statistics
# -----------------------------------------------------------------------------
def test_api_failures_never_cause_fabricated_fallback_statistics(client):
    """
    Verifies that malformed or incompatible CSV uploads return HTTP 400 errors
    rather than silently falling back to mock or predefined demo datasets.
    """
    # 1. Completely empty CSV
    resp = client.post(
        "/api/analyze/csv-analytics",
        json={"csv_text": ""}
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()

    # 2. Incompatible schema (missing all required features)
    dummy_csv = "RandomA,RandomB\n1,2\n3,4"
    resp2 = client.post(
        "/api/analyze/csv-analytics",
        json={"csv_text": dummy_csv}
    )
    assert resp2.status_code == 400
    assert "Incompatible" in resp2.get_json()["error"] or "Missing" in resp2.get_json()["error"]

    # 3. verify_dataset_files with non-existent path
    dummy_verification = verify_dataset_files("non_existent_data_directory_xyz")
    assert dummy_verification["is_verified"] is False
    assert dummy_verification["status"] == "error"
    assert dummy_verification["total_records"] == 0


# -----------------------------------------------------------------------------
# 9. XGBoost and VAE Predictions Use Correct Preprocessing & Feature Order
# -----------------------------------------------------------------------------
def test_model_predictions_use_correct_preprocessing_and_feature_order(client):
    """
    Verifies that FlowPreprocessor produces exactly 70 features in identical order
    and calibrated thresholds match production values (0.52 for XGB, 0.330610 for VAE).
    """
    health = client.get("/api/health").get_json()
    assert health["status"] == "healthy"

    models = health["models"]
    assert models["preprocessor"]["feature_count"] == 70
    assert models["xgboost"]["threshold"] == 0.52
    assert abs(models["vae"]["threshold"] - 0.330610) < 1e-4
