"""Cognisec AI - Automated Test Suite for Live Network Monitoring and Inference.
Tests packet flow extraction, 70-feature schema conformity, dual model inference,
session lifecycle management, error handling, and Flask REST API endpoints.
Uses synthetic mock packets and flows; requires NO administrative privileges or live network traffic.
"""

import math
import time
import pytest
from scapy.layers.inet import IP, TCP, UDP

from src.api.server import create_app
from src.live.flow_extractor import BidirectionalFlow, FlowTracker, PacketRecord
from src.live.live_manager import get_live_manager


@pytest.fixture
def client():
    """Flask test client fixture."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def live_manager():
    """LiveCaptureManager fixture reset before each test."""
    mgr = get_live_manager()
    if mgr.is_running:
        mgr.stop()
    mgr.flow_buffer.clear()
    mgr.packets_seen = 0
    mgr.flows_processed = 0
    mgr.benign_count = 0
    mgr.suspicious_count = 0
    mgr.validation_errors.clear()
    return mgr


# =============================================================================
# 1. Flow Extractor & 70-Feature Schema Tests
# =============================================================================

def test_bidirectional_flow_feature_schema(live_manager):
    """Verify that BidirectionalFlow generates all 70 exact features expected by preprocessor."""
    prep = live_manager.preprocessor
    expected_cols = prep.feature_columns
    assert len(expected_cols) == 70, f"Expected 70 features, got {len(expected_cols)}"

    t0 = time.time()
    flow = BidirectionalFlow("192.168.1.100", 49152, "172.217.16.206", 443, "TCP", t0)

    # Add forward SYN
    flow.add_packet(PacketRecord(t0, 64, 40, 24, True, {"SYN": True}, 64240, "TCP"))
    # Add backward SYN-ACK
    flow.add_packet(PacketRecord(t0 + 0.025, 60, 40, 20, False, {"SYN": True, "ACK": True}, 65535, "TCP"))
    # Add forward ACK
    flow.add_packet(PacketRecord(t0 + 0.030, 52, 40, 12, True, {"ACK": True}, 64240, "TCP"))

    features = flow.to_features_dict()

    # Verify all 70 columns are present
    missing = [c for c in expected_cols if c not in features]
    assert len(missing) == 0, f"Features missing from extractor: {missing}"

    # Verify duplicate mirror column
    assert "Fwd Header Length_duplicated_0" in features
    assert features["Fwd Header Length_duplicated_0"] == features["Fwd Header Length"]

    # Verify units and statistical values
    assert features["Destination Port"] == 443.0
    assert features["Total Fwd Packets"] == 2.0
    assert features["Total Backward Packets"] == 1.0
    assert features["Flow Duration"] >= 29000.0  # ~30ms in microseconds
    assert features["SYN Flag Count"] == 2.0
    assert features["ACK Flag Count"] == 2.0
    assert features["Init_Win_bytes_forward"] == 64240.0
    assert features["Init_Win_bytes_backward"] == 65535.0


def test_flow_tracker_tcp_termination():
    """Verify FlowTracker automatically terminates and returns completed flow on TCP FIN/RST."""
    tracker = FlowTracker(idle_timeout_sec=10.0, active_timeout_sec=60.0)

    # Synthetic Scapy TCP SYN packet
    pkt_syn = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="S", seq=1000)
    res1 = tracker.process_scapy_packet(pkt_syn)
    assert res1 is None  # Flow remains active
    assert len(tracker.flows) == 1

    # Synthetic TCP ACK
    pkt_ack = IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=80, dport=12345, flags="A", ack=1001)
    res2 = tracker.process_scapy_packet(pkt_ack)
    assert res2 is None
    assert len(tracker.flows) == 1

    # Synthetic TCP RST -> triggers immediate flow completion
    pkt_rst = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=12345, dport=80, flags="R")
    completed_flow = tracker.process_scapy_packet(pkt_rst)
    assert completed_flow is not None
    assert completed_flow.is_terminated is True
    assert completed_flow.termination_reason == "TCP_RST"
    assert len(tracker.flows) == 0  # Cleaned up from active tracker


def test_flow_tracker_timeouts():
    """Verify FlowTracker expires flows that exceed idle and active timeout thresholds."""
    tracker = FlowTracker(idle_timeout_sec=5.0, active_timeout_sec=30.0)
    t0 = 1000.0

    pkt = IP(src="192.168.1.1", dst="8.8.8.8") / UDP(sport=53000, dport=53)
    pkt.time = t0
    tracker.process_scapy_packet(pkt)
    assert len(tracker.flows) == 1

    # Check at t0 + 2s -> not expired
    expired_early = tracker.check_timeouts(current_time=t0 + 2.0)
    assert len(expired_early) == 0
    assert len(tracker.flows) == 1

    # Check at t0 + 6s -> idle expired (> 5s)
    expired_late = tracker.check_timeouts(current_time=t0 + 6.0)
    assert len(expired_late) == 1
    assert expired_late[0].termination_reason == "IDLE_TIMEOUT"
    assert len(tracker.flows) == 0


# =============================================================================
# 2. Live Manager & Inference Execution Tests
# =============================================================================

def test_live_manager_inference(live_manager):
    """Test full dual-model inference on a simulated completed network flow."""
    t0 = time.time()
    flow = BidirectionalFlow("192.168.1.5", 55555, "1.1.1.1", 53, "UDP", t0)
    flow.add_packet(PacketRecord(t0, 68, 28, 40, True, {}, -1, "UDP"))
    flow.add_packet(PacketRecord(t0 + 0.015, 120, 28, 92, False, {}, -1, "UDP"))

    result = live_manager._process_completed_flow(flow)
    assert result is not None
    assert result["flow_id"] == "192.168.1.5:55555-1.1.1.1:53_UDP"
    assert "xgboost" in result
    assert "vae" in result
    assert "probability" in result["xgboost"]
    assert result["xgboost"]["threshold"] == 0.52
    assert "anomaly_score" in result["vae"]
    assert result["vae"]["calibrated_threshold"] == 0.330610
    assert "joint_decision" in result
    assert result["disclaimer"] == "Model Prediction Only (No Ground-Truth Label)"

    # Verify ring buffer updated
    assert len(live_manager.flow_buffer) == 1
    assert live_manager.flows_processed == 1
    assert (live_manager.benign_count + live_manager.suspicious_count) == 1


def test_live_manager_lifecycle(live_manager):
    """Test start, stop, and status querying lifecycle controls."""
    # Start on Loopback or Default
    success, msg = live_manager.start(interface="Default", clear_buffer=True)
    assert success is True
    assert live_manager.is_running is True

    # Attempting to start again while running should fail safely
    success2, msg2 = live_manager.start(interface="Default")
    assert success2 is False
    assert "already running" in msg2.lower()

    # Query status
    status = live_manager.get_status()
    assert status["is_running"] is True
    assert status["models_ready"] is True
    assert status["thresholds"]["xgboost"] == 0.52
    assert status["thresholds"]["vae"] == 0.330610

    # Stop capture
    stop_success, stop_msg = live_manager.stop()
    assert stop_success is True
    assert live_manager.is_running is False

    # Second stop when already stopped should return error
    stop_again, _ = live_manager.stop()
    assert stop_again is False


# =============================================================================
# 3. Flask REST API Live Monitoring Route Tests
# =============================================================================

def test_api_live_interfaces(client):
    """Test GET /api/live/interfaces endpoint returns valid interface list."""
    response = client.get("/api/live/interfaces")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "interfaces" in data
    assert isinstance(data["interfaces"], list)
    assert len(data["interfaces"]) > 0
    # Check structure of first interface
    first_iface = data["interfaces"][0]
    assert "id" in first_iface
    assert "name" in first_iface
    assert "ip" in first_iface
    assert "type" in first_iface


def test_api_live_status(client):
    """Test GET /api/live/status endpoint returns valid telemetry."""
    response = client.get("/api/live/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    telemetry = data["data"]
    assert "is_running" in telemetry
    assert "models_ready" in telemetry
    assert telemetry["models_ready"] is True
    assert "thresholds" in telemetry
    assert telemetry["thresholds"]["xgboost"] == 0.52
    assert telemetry["thresholds"]["vae"] == 0.330610


def test_api_live_flows_endpoint(client, live_manager):
    """Test GET /api/live/flows and POST /api/live/clear endpoints."""
    # Seed 3 flows into buffer
    t0 = time.time()
    for port in [80, 443, 8080]:
        f = BidirectionalFlow("10.0.0.5", 50000 + port, "93.184.216.34", port, "TCP", t0)
        f.add_packet(PacketRecord(t0, 60, 40, 20, True, {"SYN": True}, 64240, "TCP"))
        live_manager._process_completed_flow(f)

    # Query flows
    resp = client.get("/api/live/flows?limit=10")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["returned"] == 3
    assert len(data["flows"]) == 3
    assert data["flows"][0]["dst_port"] == 8080  # Most recent first

    # Clear buffer
    clear_resp = client.post("/api/live/clear")
    assert clear_resp.status_code == 200

    # Query flows again -> should be empty
    resp_empty = client.get("/api/live/flows")
    assert resp_empty.get_json()["returned"] == 0


# =============================================================================
# 3. Stage C Controlled Feature Semantics Regression Tests (Deterministic Fixtures)
# =============================================================================

def test_unit_control_vs_payload_packet_lengths():
    """Unit Test: Verify pure TCP control packets yield 0 payload length and packets
    with payload yield correct length statistics conforming to CICFlowMeter BasicFlow.java.
    """
    t0 = 1000.0
    # Pure TCP handshake (SYN, SYN-ACK, ACK with 0 payload)
    flow_ctrl = BidirectionalFlow("10.0.0.1", 12345, "10.0.0.2", 80, "TCP", t0)
    flow_ctrl.add_packet(PacketRecord(t0, 60, 20, 0, True, {"SYN": True}, 64240, "TCP"))
    flow_ctrl.add_packet(PacketRecord(t0 + 0.01, 60, 20, 0, False, {"SYN": True, "ACK": True}, 65535, "TCP"))
    flow_ctrl.add_packet(PacketRecord(t0 + 0.02, 52, 20, 0, True, {"ACK": True}, 64240, "TCP"))

    feat_ctrl = flow_ctrl.to_features_dict()
    assert feat_ctrl["Total Length of Fwd Packets"] == 0.0
    assert feat_ctrl["Total Length of Bwd Packets"] == 0.0
    assert feat_ctrl["Fwd Packet Length Min"] == 0.0
    assert feat_ctrl["Fwd Packet Length Max"] == 0.0
    assert feat_ctrl["Fwd Packet Length Mean"] == 0.0
    assert feat_ctrl["Bwd Packet Length Mean"] == 0.0
    assert feat_ctrl["Min Packet Length"] == 0.0
    assert feat_ctrl["Max Packet Length"] == 0.0
    assert feat_ctrl["Packet Length Mean"] == 0.0
    assert feat_ctrl["Average Packet Size"] == 0.0
    assert feat_ctrl["Subflow Fwd Bytes"] == 0.0
    assert feat_ctrl["Subflow Bwd Bytes"] == 0.0
    assert feat_ctrl["act_data_pkt_fwd"] == 0.0

    # Flow with data packets (50 bytes fwd, 150 bytes bwd)
    flow_data = BidirectionalFlow("10.0.0.1", 12345, "10.0.0.2", 80, "TCP", t0)
    flow_data.add_packet(PacketRecord(t0, 70, 20, 50, True, {"PSH": True, "ACK": True}, 64240, "TCP"))
    flow_data.add_packet(PacketRecord(t0 + 0.02, 170, 20, 150, False, {"PSH": True, "ACK": True}, 65535, "TCP"))

    feat_data = flow_data.to_features_dict()
    assert feat_data["Total Length of Fwd Packets"] == 50.0
    assert feat_data["Total Length of Bwd Packets"] == 150.0
    assert feat_data["Fwd Packet Length Min"] == 50.0
    assert feat_data["Fwd Packet Length Max"] == 50.0
    assert feat_data["Fwd Packet Length Mean"] == 50.0
    assert feat_data["Bwd Packet Length Mean"] == 150.0
    assert feat_data["Average Packet Size"] == 100.0  # (50 + 150) / 2
    assert feat_data["Avg Fwd Segment Size"] == 50.0
    assert feat_data["Avg Bwd Segment Size"] == 150.0
    assert feat_data["Subflow Fwd Bytes"] == 50.0
    assert feat_data["Subflow Bwd Bytes"] == 150.0
    assert feat_data["act_data_pkt_fwd"] == 1.0


def test_unit_header_length_semantics_tcp_and_udp():
    """Unit Test: Verify header lengths use transport layer header size (TCP=20, UDP=8)
    and do not accumulate the 20-byte IPv4 header in feature columns.
    """
    tracker = FlowTracker()
    # TCP packet with standard 20-byte TCP header and 20-byte IP header
    pkt_tcp = IP(src="192.168.1.10", dst="192.168.1.20") / TCP(sport=50000, dport=80, flags="S")
    pkt_tcp.time = 1000.0
    tracker.process_scapy_packet(pkt_tcp)

    # UDP packet with 8-byte UDP header
    pkt_udp = IP(src="192.168.1.10", dst="8.8.8.8") / UDP(sport=53000, dport=53)
    pkt_udp.time = 1000.0
    tracker.process_scapy_packet(pkt_udp)

    flows = tracker.flush_all()
    assert len(flows) == 2
    tcp_flow = next(f for f in flows if f.protocol == "TCP")
    udp_flow = next(f for f in flows if f.protocol == "UDP")

    feat_tcp = tcp_flow.to_features_dict()
    assert feat_tcp["Fwd Header Length"] == 20.0
    assert feat_tcp["Fwd Header Length_duplicated_0"] == 20.0
    assert feat_tcp["min_seg_size_forward"] == 20.0

    feat_udp = udp_flow.to_features_dict()
    assert feat_udp["Fwd Header Length"] == 8.0
    assert feat_udp["min_seg_size_forward"] == 8.0


def test_unit_flow_rates_and_zero_duration():
    """Unit Test: Verify single-packet and zero-duration flows produce rate=0.0
    rather than arbitrary extreme numbers (e.g. 1,000,000 pkts/s).
    """
    t0 = 1000.0
    # Single packet flow (duration = 0.0)
    flow_zero = BidirectionalFlow("10.0.0.1", 10000, "10.0.0.2", 80, "TCP", t0)
    flow_zero.add_packet(PacketRecord(t0, 60, 20, 0, True, {"SYN": True}, 64240, "TCP"))

    feat_zero = flow_zero.to_features_dict()
    assert feat_zero["Flow Duration"] == 0.0
    assert feat_zero["Flow Bytes/s"] == 0.0
    assert feat_zero["Flow Packets/s"] == 0.0
    assert feat_zero["Fwd Packets/s"] == 0.0
    assert feat_zero["Bwd Packets/s"] == 0.0

    # Flow with positive duration (50ms) and 100 bytes payload
    flow_pos = BidirectionalFlow("10.0.0.1", 10000, "10.0.0.2", 80, "TCP", t0)
    flow_pos.add_packet(PacketRecord(t0, 70, 20, 50, True, {"PA": True}, 64240, "TCP"))
    flow_pos.add_packet(PacketRecord(t0 + 0.05, 70, 20, 50, False, {"PA": True}, 64240, "TCP"))

    feat_pos = flow_pos.to_features_dict()
    assert feat_pos["Flow Duration"] == pytest.approx(50000.0, rel=1e-3)  # 50ms in microseconds
    assert feat_pos["Flow Bytes/s"] == pytest.approx(100.0 / 0.05, rel=1e-3)  # 2000 B/s
    assert feat_pos["Flow Packets/s"] == pytest.approx(2.0 / 0.05, rel=1e-3)   # 40 pkts/s


def test_unit_down_up_ratio_integer_division():
    """Unit Test: Verify Down/Up Ratio matches Java integer division (BasicFlow.java:243).
    E.g. 5 bwd / 8 fwd -> 0.0; 4 bwd / 3 fwd -> 1.0; 6 bwd / 3 fwd -> 2.0.
    """
    t0 = 1000.0
    # 8 fwd, 5 bwd -> ratio should be 0.0
    flow1 = BidirectionalFlow("10.0.0.1", 1000, "10.0.0.2", 80, "TCP", t0)
    for _ in range(8):
        flow1.add_packet(PacketRecord(t0, 60, 20, 0, True, {}, 0, "TCP"))
    for _ in range(5):
        flow1.add_packet(PacketRecord(t0, 60, 20, 0, False, {}, 0, "TCP"))
    assert flow1.to_features_dict()["Down/Up Ratio"] == 0.0

    # 3 fwd, 4 bwd -> ratio should be 1.0
    flow2 = BidirectionalFlow("10.0.0.1", 1000, "10.0.0.2", 80, "TCP", t0)
    for _ in range(3):
        flow2.add_packet(PacketRecord(t0, 60, 20, 0, True, {}, 0, "TCP"))
    for _ in range(4):
        flow2.add_packet(PacketRecord(t0, 60, 20, 0, False, {}, 0, "TCP"))
    assert flow2.to_features_dict()["Down/Up Ratio"] == 1.0

    # 3 fwd, 6 bwd -> ratio should be 2.0
    flow3 = BidirectionalFlow("10.0.0.1", 1000, "10.0.0.2", 80, "TCP", t0)
    for _ in range(3):
        flow3.add_packet(PacketRecord(t0, 60, 20, 0, True, {}, 0, "TCP"))
    for _ in range(6):
        flow3.add_packet(PacketRecord(t0, 60, 20, 0, False, {}, 0, "TCP"))
    assert flow3.to_features_dict()["Down/Up Ratio"] == 2.0


def test_unit_schema_ordering_and_inference_pipeline(live_manager):
    """Unit Test: Verify all 70 features are correctly accepted by FlowPreprocessor
    and produce valid XGBoost and VAE inference outputs without non-finite values.
    """
    t0 = time.time()
    flow = BidirectionalFlow("192.168.1.100", 54321, "1.1.1.1", 443, "TCP", t0)
    flow.add_packet(PacketRecord(t0, 60, 20, 0, True, {"SYN": True}, 64240, "TCP"))
    flow.add_packet(PacketRecord(t0 + 0.02, 60, 20, 0, False, {"SYN": True, "ACK": True}, 65535, "TCP"))
    flow.add_packet(PacketRecord(t0 + 0.03, 52, 20, 0, True, {"ACK": True}, 64240, "TCP"))

    result = live_manager._process_completed_flow(flow)
    assert result is not None
    assert "xgboost" in result
    assert "vae" in result
    assert not math.isnan(result["xgboost"]["probability"])
    assert not math.isnan(result["vae"]["anomaly_score"])
    assert result["vae"]["anomaly_score"] >= 0.0
    assert result["joint_decision"] in [
        "Normal Network Behavior",
        "Unseen / Behavioral Anomaly (VAE Only)",
        "Supervised Pattern Detected (XGBoost Only)",
        "High-Confidence Attack (Both Flagged)"
    ]


def test_regression_csv_upload_inference_intact(client):
    """Regression Test: Confirm existing CSV upload endpoints remain fully functional."""
    resp = client.get("/api/sample-csv")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "csv_text" in data
    assert data["rows_count"] > 0

