"""Cognisec AI - Flow Extractor Module.
Aggregates raw network packets into completed bidirectional flows and calculates
the exact 70 statistical features required by the trained Cognisec AI pipeline.
Strictly adheres to CICFlowMeter / CICIDS2017 definitions and units (microseconds).
Payload contents are discarded immediately after extracting header and length metadata.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple


class PacketRecord:
    """Lightweight metadata extracted from a single network packet. No payload is stored.
    header_len stores transport-layer header size (TCP dataofs*4 or UDP 8 bytes)
    conforming to CICFlowMeter BasicFlow.java getHeaderBytes().
    wire_len stores total physical wire frame length for traffic volume monitoring.
    payload_len stores transport payload bytes (0 for pure TCP control packets).
    """
    __slots__ = (
        "timestamp",
        "wire_len",
        "header_len",
        "payload_len",
        "is_forward",
        "tcp_flags",
        "window_size",
        "protocol"
    )

    def __init__(
        self,
        timestamp: float,
        wire_len: int,
        header_len: int,
        payload_len: int,
        is_forward: bool,
        tcp_flags: Dict[str, bool],
        window_size: int,
        protocol: str
    ):
        self.timestamp = timestamp
        self.wire_len = wire_len
        self.header_len = header_len
        self.payload_len = payload_len
        self.is_forward = is_forward
        self.tcp_flags = tcp_flags
        self.window_size = window_size
        self.protocol = protocol


class BidirectionalFlow:
    """Tracks a single bidirectional communication flow between two network endpoints."""

    def __init__(
        self,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        first_packet_time: float
    ):
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol
        self.first_packet_time = first_packet_time
        self.last_packet_time = first_packet_time

        # Packet records (no payloads stored)
        self.fwd_packets: List[PacketRecord] = []
        self.bwd_packets: List[PacketRecord] = []
        self.all_packets: List[PacketRecord] = []

        # Flow termination state
        self.is_terminated = False
        self.termination_reason: Optional[str] = None

    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}_{self.protocol}"

    def add_packet(self, packet: PacketRecord) -> None:
        """Add a packet record to the flow."""
        self.last_packet_time = packet.timestamp
        self.all_packets.append(packet)
        if packet.is_forward:
            self.fwd_packets.append(packet)
        else:
            self.bwd_packets.append(packet)

        # Immediate TCP connection termination detection
        if packet.protocol == "TCP":
            flags = packet.tcp_flags
            if flags.get("RST", False):
                self.is_terminated = True
                self.termination_reason = "TCP_RST"
            elif flags.get("FIN", False):
                # If FIN seen in both directions, or single FIN with multiple packets
                fin_fwd = any(p.tcp_flags.get("FIN", False) for p in self.fwd_packets)
                fin_bwd = any(p.tcp_flags.get("FIN", False) for p in self.bwd_packets)
                if fin_fwd and fin_bwd:
                    self.is_terminated = True
                    self.termination_reason = "TCP_FIN_BOTH"

    def compute_summary_metadata(self) -> Dict[str, Any]:
        """Return non-sensitive flow metadata for UI presentation."""
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_time": self.first_packet_time,
            "last_time": self.last_packet_time,
            "duration_sec": max(0.0, self.last_packet_time - self.first_packet_time),
            "total_packets": len(self.all_packets),
            "fwd_packets": len(self.fwd_packets),
            "bwd_packets": len(self.bwd_packets),
            "total_bytes": sum(p.wire_len for p in self.all_packets),
            "termination_reason": self.termination_reason or "TIMEOUT"
        }

    # -------------------------------------------------------------------------
    # Statistical Calculation Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        n = len(values)
        mean_val = sum(values) / n
        if n < 2:
            return mean_val, 0.0
        var_val = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        return mean_val, math.sqrt(max(0.0, var_val))

    @staticmethod
    def _iat_stats(packets: List[PacketRecord]) -> Tuple[float, float, float, float, float]:
        """Calculate IAT metrics in microseconds: (Total, Mean, Std, Max, Min)."""
        if len(packets) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        iats = []
        for i in range(1, len(packets)):
            delta_us = max(0.0, (packets[i].timestamp - packets[i - 1].timestamp) * 1_000_000.0)
            iats.append(delta_us)

        total_iat = sum(iats)
        n = len(iats)
        mean_iat = total_iat / n
        var_iat = sum((x - mean_iat) ** 2 for x in iats) / (n - 1) if n > 1 else 0.0
        std_iat = math.sqrt(max(0.0, var_iat))
        max_iat = max(iats)
        min_iat = min(iats)
        return total_iat, mean_iat, std_iat, max_iat, min_iat

    def _active_idle_stats(self, threshold_us: float = 1_000_000.0) -> Tuple[
        Tuple[float, float, float, float], Tuple[float, float, float, float]
    ]:
        """Calculate Active and Idle period statistics in microseconds.
        Threshold: 1,000,000 us (1 second) of inactivity signifies an idle period.
        """
        if len(self.all_packets) < 2:
            return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

        active_times: List[float] = []
        idle_times: List[float] = []

        cur_active_start = self.all_packets[0].timestamp
        prev_pkt_time = self.all_packets[0].timestamp

        for i in range(1, len(self.all_packets)):
            t = self.all_packets[i].timestamp
            diff_us = (t - prev_pkt_time) * 1_000_000.0

            if diff_us > threshold_us:
                # Flow was idle
                idle_times.append(diff_us)
                active_us = max(0.0, (prev_pkt_time - cur_active_start) * 1_000_000.0)
                if active_us > 0:
                    active_times.append(active_us)
                cur_active_start = t

            prev_pkt_time = t

        # Final active chunk
        final_active_us = max(0.0, (prev_pkt_time - cur_active_start) * 1_000_000.0)
        if final_active_us > 0:
            active_times.append(final_active_us)

        def _stats(arr: List[float]) -> Tuple[float, float, float, float]:
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            n = len(arr)
            m = sum(arr) / n
            var = sum((x - m) ** 2 for x in arr) / (n - 1) if n > 1 else 0.0
            return m, math.sqrt(max(0.0, var)), max(arr), min(arr)

        return _stats(active_times), _stats(idle_times)

    def to_features_dict(self) -> Dict[str, float]:
        """Extract all 70 continuous behavioral statistical features conforming
        exactly to CICFlowMeter / CICIDS2017 schema and units.
        """
        # Duration in microseconds
        duration_sec = max(0.0, self.last_packet_time - self.first_packet_time)
        duration_us = duration_sec * 1_000_000.0

        # Forward & Backward packet lists
        fwd = self.fwd_packets
        bwd = self.bwd_packets
        all_pkts = self.all_packets

        tot_fwd_pkts = len(fwd)
        tot_bwd_pkts = len(bwd)
        tot_pkts = len(all_pkts)

        # In CICFlowMeter (BasicFlow.java:137, 151, 168), all packet length statistics
        # and byte totals accumulate transport payload bytes, NOT raw wire length.
        # Pure control packets (SYN, ACK, FIN, RST with 0 payload) have length 0.
        fwd_lens = [float(p.payload_len) for p in fwd]
        bwd_lens = [float(p.payload_len) for p in bwd]
        all_lens = [float(p.payload_len) for p in all_pkts]

        tot_len_fwd = sum(fwd_lens)
        tot_len_bwd = sum(bwd_lens)
        tot_bytes = tot_len_fwd + tot_len_bwd

        # Packet Length Statistics (payload bytes)
        fwd_mean, fwd_std = self._mean_std(fwd_lens)
        bwd_mean, bwd_std = self._mean_std(bwd_lens)
        all_mean, all_std = self._mean_std(all_lens)

        fwd_max = max(fwd_lens) if fwd_lens else 0.0
        fwd_min = min(fwd_lens) if fwd_lens else 0.0
        bwd_max = max(bwd_lens) if bwd_lens else 0.0
        bwd_min = min(bwd_lens) if bwd_lens else 0.0

        all_max = max(all_lens) if all_lens else 0.0
        all_min = min(all_lens) if all_lens else 0.0
        pkt_len_var = all_std ** 2

        # Rates: In CICFlowMeter (BasicFlow.java:214-227), if duration <= 0,
        # getFlowBytesPerSecond() and getFlowPacketsPerSecond() return 0.0.
        if duration_sec > 0.0:
            flow_bytes_s = tot_bytes / duration_sec
            flow_pkts_s = tot_pkts / duration_sec
            fwd_pkts_s = tot_fwd_pkts / duration_sec
            bwd_pkts_s = tot_bwd_pkts / duration_sec
        else:
            flow_bytes_s = 0.0
            flow_pkts_s = 0.0
            fwd_pkts_s = 0.0
            bwd_pkts_s = 0.0

        # Inter-Arrival Times (microseconds)
        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._iat_stats(all_pkts)
        fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._iat_stats(fwd)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._iat_stats(bwd)

        # Header lengths: In CICFlowMeter (BasicFlow.java:148, 152), fHeaderBytes records
        # transport layer header only (TCP = 20 or 32 bytes; UDP = 8 bytes).
        fwd_header_len = sum(p.header_len for p in fwd)
        bwd_header_len = sum(p.header_len for p in bwd)

        # Flag counts
        fin_count = sum(1 for p in all_pkts if p.tcp_flags.get("FIN", False))
        syn_count = sum(1 for p in all_pkts if p.tcp_flags.get("SYN", False))
        rst_count = sum(1 for p in all_pkts if p.tcp_flags.get("RST", False))
        psh_count = sum(1 for p in all_pkts if p.tcp_flags.get("PSH", False))
        ack_count = sum(1 for p in all_pkts if p.tcp_flags.get("ACK", False))
        urg_count = sum(1 for p in all_pkts if p.tcp_flags.get("URG", False))
        cwe_count = sum(1 for p in all_pkts if p.tcp_flags.get("CWR", False))
        ece_count = sum(1 for p in all_pkts if p.tcp_flags.get("ECE", False))

        fwd_psh = sum(1 for p in fwd if p.tcp_flags.get("PSH", False))
        fwd_urg = sum(1 for p in fwd if p.tcp_flags.get("URG", False))

        # Down/Up Ratio: In CICFlowMeter (BasicFlow.java:243), Java integer division
        # (int)(backward.size() / forward.size()) truncates to integer.
        if tot_fwd_pkts > 0:
            down_up_ratio = float(tot_bwd_pkts // tot_fwd_pkts)
        else:
            down_up_ratio = 0.0

        # Segment sizes (based on payload statistics matching BasicFlow.java)
        avg_pkt_size = float(tot_bytes) / max(float(tot_pkts), 1.0)
        avg_fwd_seg = fwd_mean
        avg_bwd_seg = bwd_mean

        # TCP initial window sizes (bytes)
        init_win_fwd = -1.0
        for p in fwd:
            if p.protocol == "TCP" and p.window_size >= 0:
                init_win_fwd = float(p.window_size)
                break

        init_win_bwd = -1.0
        for p in bwd:
            if p.protocol == "TCP" and p.window_size >= 0:
                init_win_bwd = float(p.window_size)
                break

        # Forward packets with payload > 0 bytes
        act_data_fwd = sum(1 for p in fwd if p.payload_len > 0)

        # Minimum segment size forward (TCP header size of forward packet)
        min_seg_fwd = min((p.header_len for p in fwd), default=0.0)

        # Active & Idle periods
        (act_mean, act_std, act_max, act_min), (idl_mean, idl_std, idl_max, idl_min) = self._active_idle_stats()

        # Build feature map matching the exact 70 names and order
        features: Dict[str, float] = {
            "Destination Port": float(self.dst_port),
            "Flow Duration": float(duration_us),
            "Total Fwd Packets": float(tot_fwd_pkts),
            "Total Backward Packets": float(tot_bwd_pkts),
            "Total Length of Fwd Packets": float(tot_len_fwd),
            "Total Length of Bwd Packets": float(tot_len_bwd),
            "Fwd Packet Length Max": float(fwd_max),
            "Fwd Packet Length Min": float(fwd_min),
            "Fwd Packet Length Mean": float(fwd_mean),
            "Fwd Packet Length Std": float(fwd_std),
            "Bwd Packet Length Max": float(bwd_max),
            "Bwd Packet Length Min": float(bwd_min),
            "Bwd Packet Length Mean": float(bwd_mean),
            "Bwd Packet Length Std": float(bwd_std),
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Fwd IAT Total": float(fwd_iat_tot),
            "Fwd IAT Mean": float(fwd_iat_mean),
            "Fwd IAT Std": float(fwd_iat_std),
            "Fwd IAT Max": float(fwd_iat_max),
            "Fwd IAT Min": float(fwd_iat_min),
            "Bwd IAT Total": float(bwd_iat_tot),
            "Bwd IAT Mean": float(bwd_iat_mean),
            "Bwd IAT Std": float(bwd_iat_std),
            "Bwd IAT Max": float(bwd_iat_max),
            "Bwd IAT Min": float(bwd_iat_min),
            "Fwd PSH Flags": float(fwd_psh),
            "Fwd URG Flags": float(fwd_urg),
            "Fwd Header Length": float(fwd_header_len),
            "Bwd Header Length": float(bwd_header_len),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": float(all_min),
            "Max Packet Length": float(all_max),
            "Packet Length Mean": float(all_mean),
            "Packet Length Std": float(all_std),
            "Packet Length Variance": float(pkt_len_var),
            "FIN Flag Count": float(fin_count),
            "SYN Flag Count": float(syn_count),
            "RST Flag Count": float(rst_count),
            "PSH Flag Count": float(psh_count),
            "ACK Flag Count": float(ack_count),
            "URG Flag Count": float(urg_count),
            "CWE Flag Count": float(cwe_count),
            "ECE Flag Count": float(ece_count),
            "Down/Up Ratio": float(down_up_ratio),
            "Average Packet Size": float(avg_pkt_size),
            "Avg Fwd Segment Size": float(avg_fwd_seg),
            "Avg Bwd Segment Size": float(avg_bwd_seg),
            "Fwd Header Length_duplicated_0": float(fwd_header_len),  # Duplicate mirror required by schema
            "Subflow Fwd Packets": float(tot_fwd_pkts),
            "Subflow Fwd Bytes": float(tot_len_fwd),
            "Subflow Bwd Packets": float(tot_bwd_pkts),
            "Subflow Bwd Bytes": float(tot_len_bwd),
            "Init_Win_bytes_forward": float(init_win_fwd),
            "Init_Win_bytes_backward": float(init_win_bwd),
            "act_data_pkt_fwd": float(act_data_fwd),
            "min_seg_size_forward": float(min_seg_fwd),
            "Active Mean": float(act_mean),
            "Active Std": float(act_std),
            "Active Max": float(act_max),
            "Active Min": float(act_min),
            "Idle Mean": float(idl_mean),
            "Idle Std": float(idl_std),
            "Idle Max": float(idl_max),
            "Idle Min": float(idl_min),
        }

        # Sanitize any non-finite values (NaN / Inf) with zero
        for k, v in features.items():
            if math.isnan(v) or math.isinf(v):
                features[k] = 0.0

        return features


class FlowTracker:
    """Manages active network flows, associates bidirectional packets, handles timeouts,
    and returns completed flows for live inference.
    """

    def __init__(
        self,
        idle_timeout_sec: float = 15.0,
        active_timeout_sec: float = 120.0,
        max_active_flows: int = 5000
    ):
        self.idle_timeout_sec = idle_timeout_sec
        self.active_timeout_sec = active_timeout_sec
        self.max_active_flows = max_active_flows

        # Active flows dictionary: canonical_key -> BidirectionalFlow
        self.flows: Dict[Tuple[str, int, str, int, str], BidirectionalFlow] = {}

    @staticmethod
    def canonical_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> Tuple[
        Tuple[str, int, str, int, str], bool
    ]:
        """Generate a canonical key regardless of packet direction.
        Returns (canonical_tuple, is_forward_relative_to_canonical).
        """
        endpoint_a = (src_ip, src_port)
        endpoint_b = (dst_ip, dst_port)
        if endpoint_a <= endpoint_b:
            return (src_ip, src_port, dst_ip, dst_port, proto), True
        else:
            return (dst_ip, dst_port, src_ip, src_port, proto), False

    def process_scapy_packet(self, pkt: Any) -> Optional[BidirectionalFlow]:
        """Extract packet metadata and associate with a flow.
        If the packet triggers flow completion (FIN/RST), the completed flow is returned.
        """
        # We handle IP (IPv4) or IPv6 packets
        has_ip = hasattr(pkt, "haslayer") and (pkt.haslayer("IP") or pkt.haslayer("IPv6"))
        if not has_ip:
            return None

        # Extract IP layer
        if pkt.haslayer("IP"):
            ip_layer = pkt["IP"]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            ihl_val = getattr(ip_layer, "ihl", None)
            ip_header_len = int(ihl_val) * 4 if ihl_val is not None else 20
        elif pkt.haslayer("IPv6"):
            ip_layer = pkt["IPv6"]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            ip_header_len = 40
        else:
            return None

        # Extract Transport layer
        protocol = "OTHER"
        src_port = 0
        dst_port = 0
        transport_header_len = 0
        payload_len = 0
        window_size = -1
        tcp_flags = {}

        if pkt.haslayer("TCP"):
            protocol = "TCP"
            tcp = pkt["TCP"]
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            dataofs_val = getattr(tcp, "dataofs", None)
            transport_header_len = int(dataofs_val) * 4 if dataofs_val is not None else 20
            win_val = getattr(tcp, "window", None)
            window_size = int(win_val) if win_val is not None else 0

            # Decode TCP flags
            flags_int = int(tcp.flags)
            tcp_flags = {
                "FIN": bool(flags_int & 0x01),
                "SYN": bool(flags_int & 0x02),
                "RST": bool(flags_int & 0x04),
                "PSH": bool(flags_int & 0x08),
                "ACK": bool(flags_int & 0x10),
                "URG": bool(flags_int & 0x20),
                "ECE": bool(flags_int & 0x40),
                "CWR": bool(flags_int & 0x80),
            }
            if hasattr(tcp, "payload"):
                payload_len = len(bytes(tcp.payload))

        elif pkt.haslayer("UDP"):
            protocol = "UDP"
            udp = pkt["UDP"]
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            transport_header_len = 8
            if hasattr(udp, "payload"):
                payload_len = len(bytes(udp.payload))
        else:
            # We focus on TCP and UDP for network flow analysis
            return None

        pkt_time = float(getattr(pkt, "time", time.time()))
        wire_len = len(pkt)
        # In CICFlowMeter (BasicFlow.java:148, 152), getHeaderBytes() records the
        # transport layer header only (TCP dataofs*4 = 20 or 32 bytes; UDP = 8 bytes).
        # We record transport_header_len as header_len for feature extraction conformity.
        transport_hdr = transport_header_len

        # Canonical lookup
        canon_key, _ = self.canonical_key(src_ip, src_port, dst_ip, dst_port, protocol)

        if canon_key not in self.flows:
            # First packet observed sets the forward direction
            flow = BidirectionalFlow(
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=protocol,
                first_packet_time=pkt_time
            )
            # Memory bounded check
            if len(self.flows) >= self.max_active_flows:
                # Evict oldest flow
                oldest_key = min(self.flows.keys(), key=lambda k: self.flows[k].first_packet_time)
                del self.flows[oldest_key]

            self.flows[canon_key] = flow
            is_forward = True
        else:
            flow = self.flows[canon_key]
            # Match direction with the flow's designated forward direction
            is_forward = (src_ip == flow.src_ip and src_port == flow.src_port)

        record = PacketRecord(
            timestamp=pkt_time,
            wire_len=wire_len,
            header_len=transport_hdr,
            payload_len=payload_len,
            is_forward=is_forward,
            tcp_flags=tcp_flags,
            window_size=window_size,
            protocol=protocol
        )
        flow.add_packet(record)

        # Check if flow terminated immediately on TCP FIN or RST
        if flow.is_terminated:
            del self.flows[canon_key]
            return flow

        return None

    def check_timeouts(self, current_time: Optional[float] = None) -> List[BidirectionalFlow]:
        """Scan active flows and expire any that exceed the idle or active timeouts."""
        now = current_time if current_time is not None else time.time()
        expired: List[BidirectionalFlow] = []
        keys_to_remove = []

        for key, flow in self.flows.items():
            idle_duration = now - flow.last_packet_time
            total_duration = now - flow.first_packet_time

            if idle_duration >= self.idle_timeout_sec:
                flow.is_terminated = True
                flow.termination_reason = "IDLE_TIMEOUT"
                expired.append(flow)
                keys_to_remove.append(key)
            elif total_duration >= self.active_timeout_sec:
                flow.is_terminated = True
                flow.termination_reason = "ACTIVE_TIMEOUT"
                expired.append(flow)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.flows[key]

        return expired

    def flush_all(self) -> List[BidirectionalFlow]:
        """Flush and return all remaining active flows (e.g. on capture stop)."""
        flushed = list(self.flows.values())
        for f in flushed:
            f.is_terminated = True
            if not f.termination_reason:
                f.termination_reason = "CAPTURE_STOPPED"
        self.flows.clear()
        return flushed
