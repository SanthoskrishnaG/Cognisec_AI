"""Cognisec AI - Live Network Monitoring API Routes.
Exposes REST endpoints for interface enumeration, capture lifecycle (start/stop),
real-time status telemetry, and streaming flow inference records.
"""

from typing import Any, Dict
from flask import Blueprint, jsonify, request

from src.live.live_manager import get_live_manager

live_bp = Blueprint("live", __name__)


@live_bp.route("/interfaces", methods=["GET"])
def get_interfaces():
    """Enumerate host network interfaces available for packet capture."""
    mgr = get_live_manager()
    interfaces = mgr.get_available_interfaces()
    return jsonify({
        "status": "success",
        "count": len(interfaces),
        "interfaces": interfaces
    })


@live_bp.route("/start", methods=["POST"])
def start_capture():
    """Start an authorized network capture session."""
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    interface = data.get("interface")
    bpf_filter = data.get("bpf_filter")
    clear_buffer = bool(data.get("clear_buffer", False))

    mgr = get_live_manager()
    success, message = mgr.start(
        interface=interface,
        bpf_filter=bpf_filter,
        clear_buffer=clear_buffer
    )

    status_code = 200 if success else 400
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "is_running": mgr.is_running,
        "interface": mgr.selected_interface or "Default"
    }), status_code


@live_bp.route("/stop", methods=["POST"])
def stop_capture():
    """Stop the currently active network capture session."""
    mgr = get_live_manager()
    success, message = mgr.stop()
    status_code = 200 if success else 400
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "is_running": mgr.is_running
    }), status_code


@live_bp.route("/status", methods=["GET"])
def capture_status():
    """Query current capture metrics, flow counts, and system status."""
    mgr = get_live_manager()
    status_dict = mgr.get_status()
    return jsonify({
        "status": "success",
        "data": status_dict
    })


@live_bp.route("/flows", methods=["GET"])
def get_flows():
    """Retrieve recent network flow predictions from the in-memory ring buffer."""
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)

    # Bound limits
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    mgr = get_live_manager()
    flows = mgr.get_recent_flows(limit=limit, offset=offset)

    return jsonify({
        "status": "success",
        "total_buffer": len(mgr.flow_buffer),
        "returned": len(flows),
        "offset": offset,
        "limit": limit,
        "flows": flows
    })


@live_bp.route("/clear", methods=["POST"])
def clear_buffer():
    """Clear the flow buffer and reset capture counters (allowed when stopped)."""
    mgr = get_live_manager()
    with mgr._lock:
        if mgr.is_running:
            return jsonify({
                "status": "error",
                "message": "Cannot clear buffer while capture is actively running. Stop capture first."
            }), 400
        mgr.flow_buffer.clear()
        mgr.packets_seen = 0
        mgr.flows_processed = 0
        mgr.benign_count = 0
        mgr.suspicious_count = 0
        mgr.validation_errors.clear()

    return jsonify({
        "status": "success",
        "message": "Live monitoring buffer and statistics cleared successfully."
    })
