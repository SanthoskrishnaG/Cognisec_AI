"""Cognisec AI - Live Capture and Inference Manager.
Coordinates background Scapy packet sniffing, flow extraction, schema adaptation,
and dual-model inference (XGBoost + Tabular VAE).
Implements safe session lifecycle controls, thread safety, bounded memory ring buffers,
and diagnostic interface discovery on Windows.
"""

import collections
import datetime
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.live.flow_extractor import BidirectionalFlow, FlowTracker

logger = logging.getLogger("cognisec.live")

# Workspace paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "models"


class LiveCaptureManager:
    """Thread-safe coordinator for live network capture and real-time inference."""

    def __init__(
        self,
        max_buffer_size: int = 2000,
        idle_timeout_sec: float = 15.0,
        active_timeout_sec: float = 120.0
    ):
        self.max_buffer_size = max_buffer_size
        self.idle_timeout_sec = idle_timeout_sec
        self.active_timeout_sec = active_timeout_sec

        # State lock
        self._lock = threading.Lock()

        # Session attributes
        self.is_running = False
        self.selected_interface: Optional[str] = None
        self.bpf_filter: Optional[str] = None
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None

        # Thread handles & cancellation flags
        self._capture_thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Flow tracker
        self.flow_tracker = FlowTracker(
            idle_timeout_sec=self.idle_timeout_sec,
            active_timeout_sec=self.active_timeout_sec
        )

        # Ring buffer for recent flow records
        self.flow_buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=self.max_buffer_size)

        # Statistics
        self.packets_seen = 0
        self.flows_processed = 0
        self.benign_count = 0
        self.suspicious_count = 0
        self.validation_errors: List[str] = []

        # Model cache
        self.preprocessor = None
        self.xgb_wrapper = None
        self.vae_trainer = None
        self.models_loaded = False
        self.load_error: Optional[str] = None

        # Preload models
        self._ensure_models_loaded()

    def _ensure_models_loaded(self) -> bool:
        """Load the preprocessor, XGBoost wrapper, and VAE checkpoint if not already loaded."""
        if self.models_loaded:
            return True

        try:
            from src.models.xgboost_model import XGBoostClassifierWrapper
            from src.models.vae_model import VAETrainer

            prep_path = ARTIFACTS_DIR / "flow_preprocessor.joblib"
            xgb_path = ARTIFACTS_DIR / "xgboost_model.json"
            vae_path = ARTIFACTS_DIR / "vae_checkpoint.pt"

            if not prep_path.exists():
                raise FileNotFoundError(f"Preprocessor artifact missing at {prep_path}")
            if not xgb_path.exists():
                raise FileNotFoundError(f"XGBoost artifact missing at {xgb_path}")
            if not vae_path.exists():
                raise FileNotFoundError(f"VAE checkpoint missing at {vae_path}")

            self.preprocessor = joblib.load(prep_path)
            self.xgb_wrapper = XGBoostClassifierWrapper().load(str(xgb_path))
            self.vae_trainer = VAETrainer.load_checkpoint(str(vae_path), device="cpu")

            self.models_loaded = True
            self.load_error = None
            logger.info("LiveCaptureManager: Models successfully loaded for live inference.")
            return True
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"LiveCaptureManager: Failed to load models: {e}")
            return False

    # -------------------------------------------------------------------------
    # Interface Discovery & Diagnostics
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_interfaces() -> List[Dict[str, Any]]:
        """Enumerate local network interfaces with friendly names, IPs, and Scapy identifiers."""
        interfaces: List[Dict[str, Any]] = []
        try:
            import scapy.arch.windows as scapy_win
            from scapy.all import get_if_list, conf

            # On Windows, scapy provides get_windows_if_list()
            win_ifaces = []
            if hasattr(scapy_win, "get_windows_if_list"):
                try:
                    win_ifaces = scapy_win.get_windows_if_list()
                except Exception:
                    win_ifaces = []

            seen_names = set()
            for w in win_ifaces:
                guid = w.get("guid", "")
                name = w.get("name", "")
                desc = w.get("description", "")
                ips = w.get("ips", [])
                mac = w.get("mac", "")

                # Prefer IPv4 if available
                ipv4 = next((ip for ip in ips if "." in ip and not ip.startswith("169.254")), None)
                if not ipv4 and ips:
                    ipv4 = ips[0]

                is_loopback = "loopback" in desc.lower() or "loopback" in name.lower() or ipv4 == "127.0.0.1"
                is_wifi = "wi-fi" in desc.lower() or "wireless" in desc.lower() or "wi-fi" in name.lower()
                is_ethernet = "ethernet" in desc.lower() or "realtek" in desc.lower() or "intel" in desc.lower()

                label = f"{name} ({desc})" if desc else name
                if ipv4:
                    label += f" - {ipv4}"

                interfaces.append({
                    "id": name,
                    "guid": guid,
                    "name": name,
                    "description": desc,
                    "ip": ipv4 or "No IP Assigned",
                    "mac": mac,
                    "is_loopback": is_loopback,
                    "is_active": bool(ipv4 and ipv4 != "127.0.0.1"),
                    "type": "Wi-Fi" if is_wifi else ("Loopback" if is_loopback else ("Ethernet" if is_ethernet else "Virtual/Other")),
                    "label": label
                })
                seen_names.add(name)

            # Fallback if get_windows_if_list returned nothing
            if not interfaces:
                for iface in get_if_list():
                    interfaces.append({
                        "id": str(iface),
                        "guid": str(iface),
                        "name": str(iface),
                        "description": "Standard Network Interface",
                        "ip": "Unknown",
                        "mac": "",
                        "is_loopback": False,
                        "is_active": True,
                        "type": "Generic",
                        "label": str(iface)
                    })

            # Sort active interfaces to the top
            interfaces.sort(key=lambda x: (not x["is_active"], not x["is_loopback"], x["name"]))
            return interfaces
        except Exception as e:
            logger.warning(f"Error enumerating interfaces: {e}")
            return [{
                "id": "default",
                "name": "Default System Interface",
                "description": f"Interface enumeration fallback ({str(e)})",
                "ip": "127.0.0.1",
                "mac": "",
                "is_loopback": True,
                "is_active": True,
                "type": "Default",
                "label": "Default System Interface (127.0.0.1)"
            }]

    # -------------------------------------------------------------------------
    # Inference Pipeline on Completed Flows
    # -------------------------------------------------------------------------
    def _process_completed_flow(self, flow: BidirectionalFlow) -> Optional[Dict[str, Any]]:
        """Transform flow features and execute dual XGBoost and VAE inference."""
        if not self._ensure_models_loaded():
            self.validation_errors.append("Models unavailable for live inference.")
            return None

        try:
            # 1. Extract 70 features dictionary
            features_dict = flow.to_features_dict()

            # 2. Convert to 1-row DataFrame preserving exact column order
            df_flow = pd.DataFrame([features_dict])

            # 3. Apply saved FlowPreprocessor transformations
            X_unscaled, X_scaled, _, _ = self.preprocessor.transform(df_flow)

            # 4. XGBoost Inference (raw unscaled imputed features)
            xgb_prob = float(self.xgb_wrapper.predict_proba(X_unscaled)[0])
            xgb_thresh = float(self.xgb_wrapper.threshold)
            xgb_is_attack = bool(xgb_prob >= xgb_thresh)

            # 5. PyTorch Tabular VAE Inference (scaled & clipped features)
            vae_score = float(self.vae_trainer.compute_anomaly_scores(X_scaled)[0])
            vae_thresh = float(self.vae_trainer.threshold)
            vae_is_anomaly = bool(vae_score >= vae_thresh)

            # 6. Joint Research Decision Tag
            if xgb_is_attack and vae_is_anomaly:
                joint_decision = "High-Confidence Attack (Both Flagged)"
                decision_level = "critical"
            elif xgb_is_attack and not vae_is_anomaly:
                joint_decision = "Supervised Pattern Detected (XGBoost Only)"
                decision_level = "warning"
            elif not xgb_is_attack and vae_is_anomaly:
                joint_decision = "Unseen / Behavioral Anomaly (VAE Only)"
                decision_level = "anomaly"
            else:
                joint_decision = "Normal Network Behavior"
                decision_level = "normal"

            is_suspicious = (xgb_is_attack or vae_is_anomaly)

            # Flow metadata
            meta = flow.compute_summary_metadata()

            record = {
                "flow_id": meta["flow_id"],
                "timestamp": datetime.datetime.fromtimestamp(meta["last_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": meta["src_ip"],
                "src_port": meta["src_port"],
                "dst_ip": meta["dst_ip"],
                "dst_port": meta["dst_port"],
                "protocol": meta["protocol"],
                "duration_sec": round(meta["duration_sec"], 4),
                "total_packets": meta["total_packets"],
                "total_bytes": meta["total_bytes"],
                "termination_reason": meta["termination_reason"],
                "xgboost": {
                    "probability": round(xgb_prob, 4),
                    "threshold": round(xgb_thresh, 4),
                    "is_attack": xgb_is_attack,
                    "prediction": "Attack" if xgb_is_attack else "Benign"
                },
                "vae": {
                    "anomaly_score": round(vae_score, 6),
                    "calibrated_threshold": round(vae_thresh, 6),
                    "is_anomaly": vae_is_anomaly,
                    "prediction": "Anomaly" if vae_is_anomaly else "Normal"
                },
                "joint_decision": joint_decision,
                "decision_level": decision_level,
                "is_suspicious": is_suspicious,
                "disclaimer": "Model Prediction Only (No Ground-Truth Label)"
            }

            with self._lock:
                self.flows_processed += 1
                if is_suspicious:
                    self.suspicious_count += 1
                else:
                    self.benign_count += 1
                self.flow_buffer.append(record)

            return record

        except Exception as e:
            err = f"Flow inference failed for {flow.flow_id}: {str(e)}"
            logger.error(err)
            with self._lock:
                if len(self.validation_errors) < 50:
                    self.validation_errors.append(err)
            return None

    # -------------------------------------------------------------------------
    # Background Workers
    # -------------------------------------------------------------------------
    def _packet_callback(self, packet: Any) -> None:
        """Callback invoked by Scapy for every observed packet."""
        if self._stop_event.is_set():
            return

        with self._lock:
            self.packets_seen += 1

        # Process packet through flow tracker
        completed_flow = self.flow_tracker.process_scapy_packet(packet)
        if completed_flow:
            self._process_completed_flow(completed_flow)

    def _timeout_worker(self) -> None:
        """Periodically scans active flows and expires those reaching timeouts."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            if self._stop_event.is_set():
                break

            expired_flows = self.flow_tracker.check_timeouts()
            for flow in expired_flows:
                self._process_completed_flow(flow)

    def _sniff_worker(self, interface: Optional[str], bpf_filter: Optional[str]) -> None:
        """Background sniffing thread invoking Scapy sniff."""
        try:
            from scapy.all import sniff

            logger.info(f"Live sniffing started on interface: {interface or 'DEFAULT'}")

            # Define stop filter condition
            def should_stop(pkt):
                return self._stop_event.is_set()

            sniff_kwargs: Dict[str, Any] = {
                "prn": self._packet_callback,
                "store": False,
                "stop_filter": should_stop,
            }
            if interface and interface.lower() != "default":
                sniff_kwargs["iface"] = interface
            if bpf_filter and bpf_filter.strip():
                sniff_kwargs["filter"] = bpf_filter.strip()

            sniff(**sniff_kwargs)

        except Exception as e:
            err = f"Sniffing thread error on {interface}: {str(e)}"
            logger.error(err)
            with self._lock:
                self.validation_errors.append(err)
        finally:
            with self._lock:
                self.is_running = False
                self.stopped_at = time.time()
            logger.info("Sniffing background worker exited.")

    # -------------------------------------------------------------------------
    # Lifecycle Controls: Start / Stop / Status
    # -------------------------------------------------------------------------
    def start(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        clear_buffer: bool = False
    ) -> Tuple[bool, str]:
        """Start a new live network capture session."""
        with self._lock:
            if self.is_running:
                return False, "Capture session is already running. Stop the current session first."

            if not self._ensure_models_loaded():
                return False, f"Cannot start capture: Models failed to load ({self.load_error})"

            if clear_buffer:
                self.flow_buffer.clear()
                self.packets_seen = 0
                self.flows_processed = 0
                self.benign_count = 0
                self.suspicious_count = 0
                self.validation_errors.clear()

            self._stop_event.clear()
            self.selected_interface = interface
            self.bpf_filter = bpf_filter
            self.started_at = time.time()
            self.stopped_at = None
            self.is_running = True

            # Launch sniffing thread
            self._capture_thread = threading.Thread(
                target=self._sniff_worker,
                args=(interface, bpf_filter),
                name="CognisecSniffThread",
                daemon=True
            )
            self._capture_thread.start()

            # Launch timeout worker thread
            self._timeout_thread = threading.Thread(
                target=self._timeout_worker,
                name="CognisecTimeoutThread",
                daemon=True
            )
            self._timeout_thread.start()

            return True, f"Capture successfully started on interface '{interface or 'Default'}'"

    def stop(self) -> Tuple[bool, str]:
        """Stop the currently active capture session and flush in-flight flows."""
        with self._lock:
            if not self.is_running:
                return False, "No capture session is currently active."

            self._stop_event.set()
            self.is_running = False
            self.stopped_at = time.time()

        # Flush any in-flight active flows
        flushed_flows = self.flow_tracker.flush_all()
        for flow in flushed_flows:
            self._process_completed_flow(flow)

        return True, f"Capture stopped. Flushed {len(flushed_flows)} remaining active flows."

    def get_status(self) -> Dict[str, Any]:
        """Return the current capture session status, metrics, and health diagnostics."""
        with self._lock:
            uptime = 0.0
            if self.is_running and self.started_at:
                uptime = time.time() - self.started_at
            elif self.started_at and self.stopped_at:
                uptime = self.stopped_at - self.started_at

            active_in_flight = len(self.flow_tracker.flows)

            return {
                "is_running": self.is_running,
                "interface": self.selected_interface or "Default System Interface",
                "bpf_filter": self.bpf_filter or "None (All IP/TCP/UDP)",
                "uptime_seconds": round(uptime, 1),
                "started_at": datetime.datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d %H:%M:%S") if self.started_at else None,
                "packets_seen": self.packets_seen,
                "active_in_flight_flows": active_in_flight,
                "flows_processed": self.flows_processed,
                "benign_flows": self.benign_count,
                "suspicious_flows": self.suspicious_count,
                "suspicious_percentage": round((self.suspicious_count / max(self.flows_processed, 1)) * 100, 2),
                "recent_buffer_size": len(self.flow_buffer),
                "max_buffer_capacity": self.max_buffer_size,
                "models_ready": self.models_loaded,
                "load_error": self.load_error,
                "recent_errors": list(self.validation_errors)[-5:],
                "thresholds": {
                    "xgboost": 0.52,
                    "vae": 0.330610
                }
            }

    def get_recent_flows(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Return a slice of recent flow predictions in reverse chronological order."""
        with self._lock:
            # Most recent flows first
            items = list(self.flow_buffer)[::-1]
            return items[offset: offset + limit]

    def get_live_analytics(self, max_points: int = 60) -> Dict[str, Any]:
        """
        Calculates dynamic, bounded live chart analytics from actual captured flows.
        Returns empty state structures when no traffic has arrived, with zero simulated data.
        """
        with self._lock:
            buffer_items = list(self.flow_buffer)
            running = self.is_running
            pkts = self.packets_seen
            flows_count = self.flows_processed
            benign = self.benign_count
            suspicious = self.suspicious_count
            uptime = 0.0
            if self.is_running and self.started_at:
                uptime = time.time() - self.started_at
            elif self.started_at and self.stopped_at:
                uptime = self.stopped_at - self.started_at

        xgb_thresh = 0.52
        vae_thresh = 0.330610

        if not buffer_items:
            return {
                "has_traffic": False,
                "is_running": running,
                "uptime_seconds": round(uptime, 1),
                "total_flows": flows_count,
                "total_packets": pkts,
                "traffic_volume_over_time": [],
                "normal_vs_suspicious": {
                    "normal": 0,
                    "suspicious": 0,
                    "normal_pct": 0.0,
                    "suspicious_pct": 0.0
                },
                "model_detections": {
                    "xgboost_attacks": 0,
                    "vae_anomalies": 0,
                    "both_flagged": 0
                },
                "vae_scores_over_time": [],
                "protocol_distribution": {},
                "top_suspicious_flows": [],
                "model_disagreement_over_time": [],
                "calibrated_thresholds": {
                    "xgboost": xgb_thresh,
                    "vae": vae_thresh
                },
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Analyze actual captured flows
        protocols = collections.Counter()
        xgb_attacks = 0
        vae_anomalies = 0
        both_flagged = 0

        time_buckets = collections.OrderedDict()

        for item in buffer_items:
            proto = str(item.get("protocol", "OTHER")).upper()
            protocols[proto] += 1

            is_xgb_atk = bool(item.get("xgboost", {}).get("is_attack", False))
            is_vae_anom = bool(item.get("vae", {}).get("is_anomaly", False))

            if is_xgb_atk:
                xgb_attacks += 1
            if is_vae_anom:
                vae_anomalies += 1
            if is_xgb_atk and is_vae_anom:
                both_flagged += 1

            ts_str = str(item.get("timestamp", ""))
            bucket_key = ts_str[-8:-1] + "0" if len(ts_str) >= 8 else "00:00:00"
            if bucket_key not in time_buckets:
                time_buckets[bucket_key] = {
                    "time": bucket_key,
                    "flows": 0,
                    "packets": 0,
                    "bytes": 0,
                    "suspicious": 0,
                    "normal": 0,
                    "xgb_attacks": 0,
                    "vae_anomalies": 0,
                    "agreed": 0,
                    "disagreed": 0
                }

            b = time_buckets[bucket_key]
            b["flows"] += 1
            b["packets"] += int(item.get("total_packets", 1))
            b["bytes"] += int(item.get("total_bytes", 0))

            if is_xgb_atk or is_vae_anom:
                b["suspicious"] += 1
            else:
                b["normal"] += 1

            if is_xgb_atk:
                b["xgb_attacks"] += 1
            if is_vae_anom:
                b["vae_anomalies"] += 1

            if (is_xgb_atk and is_vae_anom) or (not is_xgb_atk and not is_vae_anom):
                b["agreed"] += 1
            else:
                b["disagreed"] += 1

        recent_bucket_keys = list(time_buckets.keys())[-max_points:]
        traffic_volume = [time_buckets[k] for k in recent_bucket_keys]

        vae_scores_stream = []
        for flow in buffer_items[-30:]:
            vae_info = flow.get("vae", {})
            score = float(vae_info.get("anomaly_score", 0.0))
            vae_scores_stream.append({
                "time": str(flow.get("timestamp", ""))[-8:],
                "score": round(score, 6),
                "is_anomaly": bool(vae_info.get("is_anomaly", False)),
                "flow_id": flow.get("flow_id", "flow")
            })

        scored_flows = []
        for flow in buffer_items:
            xgb_p = float(flow.get("xgboost", {}).get("probability", 0.0))
            vae_s = float(flow.get("vae", {}).get("anomaly_score", 0.0))
            norm_vae = vae_s / max(vae_thresh, 1e-6)
            risk = max(xgb_p, min(1.0, norm_vae))
            scored_flows.append((risk, flow))

        scored_flows.sort(key=lambda x: x[0], reverse=True)
        top_suspicious = []
        for risk, flow in scored_flows[:10]:
            src = f"{flow.get('src_ip')}:{flow.get('src_port')}"
            dst = f"{flow.get('dst_ip')}:{flow.get('dst_port')}"
            top_suspicious.append({
                "flow_id": flow.get("flow_id"),
                "timestamp": flow.get("timestamp"),
                "source": src,
                "destination": dst,
                "protocol": flow.get("protocol"),
                "packets": flow.get("total_packets"),
                "bytes": flow.get("total_bytes"),
                "xgb_prob": flow.get("xgboost", {}).get("probability"),
                "vae_score": flow.get("vae", {}).get("anomaly_score"),
                "decision": flow.get("joint_decision"),
                "risk_score": round(risk, 4),
                "is_suspicious": flow.get("is_suspicious", False)
            })

        norm_pct = round((benign / max(flows_count, 1)) * 100, 2)
        susp_pct = round((suspicious / max(flows_count, 1)) * 100, 2)

        return {
            "has_traffic": True,
            "is_running": running,
            "uptime_seconds": round(uptime, 1),
            "total_flows": flows_count,
            "total_packets": pkts,
            "traffic_volume_over_time": traffic_volume,
            "normal_vs_suspicious": {
                "normal": benign,
                "suspicious": suspicious,
                "normal_pct": norm_pct,
                "suspicious_pct": susp_pct
            },
            "model_detections": {
                "xgboost_attacks": xgb_attacks,
                "vae_anomalies": vae_anomalies,
                "both_flagged": both_flagged
            },
            "vae_scores_over_time": vae_scores_stream,
            "protocol_distribution": dict(protocols),
            "top_suspicious_flows": top_suspicious,
            "model_disagreement_over_time": [
                {
                    "time": b["time"],
                    "agreed": b["agreed"],
                    "disagreed": b["disagreed"]
                }
                for b in traffic_volume
            ],
            "calibrated_thresholds": {
                "xgboost": xgb_thresh,
                "vae": vae_thresh
            },
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }



# Global singleton instance
_GLOBAL_LIVE_MANAGER: Optional[LiveCaptureManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_live_manager() -> LiveCaptureManager:
    """Retrieve the global singleton LiveCaptureManager instance."""
    global _GLOBAL_LIVE_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_LIVE_MANAGER is None:
            _GLOBAL_LIVE_MANAGER = LiveCaptureManager()
        return _GLOBAL_LIVE_MANAGER
