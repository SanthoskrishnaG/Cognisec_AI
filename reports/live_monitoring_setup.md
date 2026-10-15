# Cognisec AI: Live Network Monitoring Setup Guide

**Project Title:** Cognisec AI: Network Behaviour Fingerprinting for Unseen Attack Detection — An Experimental Comparison of XGBoost and Variational Autoencoders  
**Module:** Live Network Traffic Capture and Real-Time ML/DL Inference  
**OS Platform:** Windows 10 / 11 (x64)  
**Environment:** Python 3.12 (`.venv`)  
**Backend:** Flask 3.0+, Scapy 2.8.0+, Npcap Driver  

---

## 1. Prerequisites & Windows Dependencies

### 1.1 Npcap Packet Capture Library for Windows
To perform raw layer-2 packet sniffing on Windows, Scapy requires the **Npcap** packet capture architecture:
1. Download the official installer from [https://npcap.com/#download](https://npcap.com/#download).
2. Run `npcap-<version>.exe` with Administrator privileges.
3. **Critical Installation Option:** Check the box:
   - `[x] Install Npcap in WinPcap-API-compatible Mode` (allows standard packet capture libraries to locate `wpcap.dll` and `Packet.dll`).
4. Complete the installation. (Reboot is typically not required, but recommended if updating older drivers).

### 1.2 Python Environment & Scapy
The project runs within the dedicated Python 3.12 virtual environment at `.\.venv`.

To verify or install Scapy:
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Scapy installation
python -c "import scapy.all as scapy; print('Scapy version:', scapy.__version__); print('use_pcap:', scapy.conf.use_pcap)"
```
Expected output:
```text
Scapy version: 2.8.0.post5
use_pcap: True
```

---

## 2. Supported Interfaces & Windows Device Enumeration

Cognisec AI automatically enumerates available network adapters using Windows GUIDs and friendly adapter names:

| Interface Type | Typical Windows Name | Supported Scapy Identifier | Usage Scope |
| :--- | :--- | :--- | :--- |
| **Wi-Fi Adapter** | `Wi-Fi (Intel/Realtek/MediaTek)` | GUID / Device Name | Primary host monitoring on active wireless LAN |
| **Ethernet Adapter** | `Ethernet (Realtek PCIe GbE...)` | GUID / Device Name | Primary host monitoring on wired LAN |
| **Npcap Loopback** | `Npcap Loopback Adapter` | `\Device\NPF_Loopback` | Isolated testing of local IPC / loopback HTTP traffic |
| **Default System** | `Default` | `scapy.conf.iface` | Default system outbound interface |

### 2.1 Identifying Available Interfaces
Query the REST endpoint or run the command-line helper:
```powershell
.\.venv\Scripts\python.exe -c "from src.live.live_manager import LiveCaptureManager; [print(i['label']) for i in LiveCaptureManager.get_available_interfaces()]"
```

---

## 3. Starting the Cognisec AI Application

Run the server bound to localhost or 0.0.0.0:
```powershell
# In PowerShell (from project root C:\Users\Santhoskrishna\Documents\Cognisec_AI):
.\.venv\Scripts\python.exe server.py --host 127.0.0.1 --port 5000
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
Click **Live Monitoring** in the left sidebar navigation.

---

## 4. Operational Instructions: Capturing & Analyzing Live Flows

1. **Select Network Interface:**
   - From the **Network Interface** dropdown, select your active adapter (e.g. `Wi-Fi` or `Ethernet`).
   - If testing locally, select `Npcap Loopback Adapter` or `Default`.
2. **Optional BPF Filter:**
   - Use standard Berkeley Packet Filter (BPF) syntax if you wish to restrict capture scope:
     - `tcp or udp` (default recommendation)
     - `tcp port 80 or tcp port 443` (web traffic only)
     - `not port 5000` (exclude Cognisec dashboard traffic itself)
3. **Start Capture:**
   - Click the green **Start Capture** button.
   - The status indicator will switch to `MONITORING ACTIVE` with a pulsing dot.
   - The dashboard polls `/api/live/status` and `/api/live/flows` every 1.5 seconds.
4. **Inspect Live Predictions:**
   - As bidirectional flows complete (via TCP FIN/RST or 15s inactivity timeout), they appear in the real-time stream.
   - Each row displays:
     - Flow 5-Tuple: `Source IP:Port -> Destination IP:Port`
     - Transport Protocol (`TCP` / `UDP`)
     - Duration, Packet Count, and Total Wire Bytes
     - Supervised XGBoost Probability and Prediction (threshold `0.5200`)
     - PyTorch Tabular VAE Reconstruction Error MSE and Anomaly Decision (threshold `0.330610`)
     - Joint Research Synthesis Badge
5. **Stop Capture:**
   - Click **Stop Capture**.
   - The background sniffer halts cleanly and all remaining in-flight flows are immediately flushed and evaluated.

---

## 5. Security & Ethical Boundaries

1. **Authorized Targets Only:**
   - Capture traffic only on interfaces, machines, and networks you own or have explicit written authorization to monitor.
2. **Passive Observation Only:**
   - The Cognisec AI Live Monitoring engine is strictly passive. It listens for incoming/outgoing frames; it never injects packets, alters transmission headers, conducts port scans, or probes external systems.
3. **Immediate Payload Discard:**
   - In accordance with privacy and cybersecurity best practices, packet application payloads (HTTP bodies, credentials, tokens) are discarded instantly upon extracting wire and header length metrics. No payloads are ever stored in RAM or written to disk.
4. **Model Output Status:**
   - All flags are probabilistic **model predictions**, not confirmed security breaches.

---

## 6. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| `scapy.conf.use_pcap: False` | Npcap is not installed or not in WinPcap mode | Re-run Npcap installer and check `Install Npcap in WinPcap-API-compatible Mode`. |
| `Permission denied` / Interface access error | Non-administrator user profile on Windows | Open PowerShell as Administrator or ensure the Npcap driver permits non-admin packet capture (Npcap installer option: `[ ] Restrict Npcap driver's access to Administrators only`). |
| No flows appear in table | Capture is idle, or BPF filter is too restrictive | Verify packets observed counter is incrementing. Note that flows only complete after TCP FIN/RST or upon 15s inactivity timeout. Clicking "Stop Capture" will instantly flush all in-flight flows. |
| Duplicate capture error (`Capture session already running`) | A previous session was not terminated | Click **Stop Capture** or call `POST /api/live/stop`. Only one capture thread can run at a time to prevent race conditions. |
