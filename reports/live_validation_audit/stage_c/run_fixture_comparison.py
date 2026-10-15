"""Cognisec AI - Stage C Controlled Feature Difference & Model Validation Script.
Executes both old extractor (flow_extractor.py.bak_stage_c) and new extractor
(flow_extractor.py) on identical deterministic network packet fixtures.
Calculates per-feature delta and evaluates PyTorch Tabular VAE and XGBoost inference.
"""

import importlib.util
import json
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scapy.layers.inet import IP, TCP, UDP

from importlib.machinery import SourceFileLoader

# Load old extractor module
old_fe = SourceFileLoader("old_fe", os.path.abspath("src/live/flow_extractor.py.bak_stage_c")).load_module()

# Load new extractor module
import src.live.flow_extractor as new_fe
from src.models.vae_model import TabularVAE

# Load models and preprocessor
prep = joblib.load("artifacts/models/flow_preprocessor.joblib")
vae_ckpt = torch.load("artifacts/models/vae_checkpoint.pt", map_location="cpu", weights_only=False)
vae = TabularVAE(input_dim=70, latent_dim=16)
vae.load_state_dict(vae_ckpt["model_state_dict"])
vae.eval()
vae_threshold = float(vae_ckpt.get("calibrated_threshold", 0.330610))

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("artifacts/models/xgboost_model.json")
xgb_threshold = 0.5200


def evaluate_features(feat_dict):
    df = pd.DataFrame([feat_dict])
    X_unscaled, X_scaled, _, _ = prep.transform(df)
    
    # XGBoost
    xgb_prob = float(xgb_model.predict_proba(X_unscaled)[0, 1])
    xgb_pred = "ATTACK" if xgb_prob >= xgb_threshold else "BENIGN"
    
    # VAE
    with torch.no_grad():
        x_t = torch.tensor(X_scaled, dtype=torch.float32)
        x_recon, _, _ = vae(x_t)
        vae_mse = float(torch.mean((x_t - x_recon) ** 2, dim=1).item())
    vae_pred = "ANOMALY" if vae_mse >= vae_threshold else "NORMAL"
    
    return {
        "xgb_prob": xgb_prob,
        "xgb_pred": xgb_pred,
        "vae_mse": vae_mse,
        "vae_pred": vae_pred
    }


def run_fixtures():
    fixtures = {}
    
    # Fixture 1: Clean 3-Packet TCP Handshake (SYN, SYN-ACK, ACK with 0 payload)
    t0 = 1700000000.0
    f1_pkts = [
        IP(src="192.168.1.50", dst="93.184.216.34") / TCP(sport=50000, dport=443, flags="S", seq=100),
        IP(src="93.184.216.34", dst="192.168.1.50") / TCP(sport=443, dport=50000, flags="SA", seq=200, ack=101),
        IP(src="192.168.1.50", dst="93.184.216.34") / TCP(sport=50000, dport=443, flags="A", seq=101, ack=201)
    ]
    f1_pkts[0].time = t0
    f1_pkts[1].time = t0 + 0.025
    f1_pkts[2].time = t0 + 0.030
    fixtures["TCP_3Way_Handshake_No_Payload"] = f1_pkts

    # Fixture 2: HTTP Transaction with Data Payload
    t0 = 1700000100.0
    f2_pkts = [
        IP(src="10.0.0.5", dst="10.0.0.1") / TCP(sport=45000, dport=80, flags="S", seq=1),
        IP(src="10.0.0.1", dst="10.0.0.5") / TCP(sport=80, dport=45000, flags="SA", seq=10, ack=2),
        IP(src="10.0.0.5", dst="10.0.0.1") / TCP(sport=45000, dport=80, flags="A", seq=2, ack=11),
        IP(src="10.0.0.5", dst="10.0.0.1") / TCP(sport=45000, dport=80, flags="PA", seq=2, ack=11) / b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
        IP(src="10.0.0.1", dst="10.0.0.5") / TCP(sport=80, dport=45000, flags="PA", seq=11, ack=52) / (b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n" + b"A" * 20),
        IP(src="10.0.0.5", dst="10.0.0.1") / TCP(sport=45000, dport=80, flags="FA", seq=52, ack=65),
        IP(src="10.0.0.1", dst="10.0.0.5") / TCP(sport=80, dport=45000, flags="FA", seq=65, ack=53),
        IP(src="10.0.0.5", dst="10.0.0.1") / TCP(sport=45000, dport=80, flags="A", seq=53, ack=66)
    ]
    for idx, p in enumerate(f2_pkts):
        p.time = t0 + (idx * 0.010)
    fixtures["HTTP_Transaction_With_Payload"] = f2_pkts

    # Fixture 3: Instantaneous Single-Packet Flow (0 us duration)
    t0 = 1700000200.0
    f3_pkts = [
        IP(src="192.168.1.75", dst="8.8.8.8") / UDP(sport=53000, dport=53) / b"\xaa\xbb\x01\x00\x00\x01\x00\x00"
    ]
    f3_pkts[0].time = t0
    fixtures["Single_Packet_Zero_Duration_UDP"] = f3_pkts

    # Fixture 4: Asymmetric Flow (Down/Up Ratio Test: 8 Fwd, 5 Bwd -> Java integer division = 0)
    t0 = 1700000300.0
    f4_pkts = []
    for i in range(8):
        p = IP(src="192.168.1.99", dst="1.1.1.1") / UDP(sport=60000, dport=53) / (b"fwd_" + bytes([i]))
        p.time = t0 + i * 0.005
        f4_pkts.append(p)
    for j in range(5):
        p = IP(src="1.1.1.1", dst="192.168.1.99") / UDP(sport=53, dport=60000) / (b"bwd_" + bytes([j]))
        p.time = t0 + 0.050 + j * 0.005
        f4_pkts.append(p)
    fixtures["Asymmetric_8Fwd_5Bwd_UDP"] = f4_pkts

    results = {}
    for name, pkts in fixtures.items():
        # Old
        tr_old = old_fe.FlowTracker()
        fl_old = None
        for p in pkts:
            res = tr_old.process_scapy_packet(p)
            if res is not None:
                fl_old = res
        if fl_old is None:
            fl_old = tr_old.flush_all()[0]
        feat_old = fl_old.to_features_dict()
        res_old = evaluate_features(feat_old)

        # New
        tr_new = new_fe.FlowTracker()
        fl_new = None
        for p in pkts:
            res = tr_new.process_scapy_packet(p)
            if res is not None:
                fl_new = res
        if fl_new is None:
            fl_new = tr_new.flush_all()[0]
        feat_new = fl_new.to_features_dict()
        res_new = evaluate_features(feat_new)

        diffs = {}
        for k in feat_old:
            val_old = feat_old[k]
            val_new = feat_new[k]
            if abs(val_old - val_new) > 1e-4:
                diffs[k] = {"old": val_old, "new": val_new, "delta": val_new - val_old}

        results[name] = {
            "num_packets": len(pkts),
            "num_diff_features": len(diffs),
            "differing_features": diffs,
            "inference_old": res_old,
            "inference_new": res_new
        }

    return results

if __name__ == "__main__":
    res = run_fixtures()
    print(json.dumps(res, indent=2))
    with open("reports/live_validation_audit/stage_c/fixture_comparison_results.json", "w") as f:
        json.dump(res, f, indent=2)
