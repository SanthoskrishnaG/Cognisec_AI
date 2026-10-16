"""Cognisec AI - Live Network Traffic Simulator.
Generates controlled, safe network activity (benign HTTP requests, rapid port probes,
and simulated burst flows) to feed the Live Monitoring packet sniffer on localhost or host Wi-Fi/Ethernet.

Usage:
  1. Open Cognisec AI Dashboard: http://127.0.0.1:5000
  2. Go to the "Live Monitoring" view in the sidebar.
  3. Select your active interface (e.g., Wi-Fi, Ethernet, or Loopback) and click "Start Live Capture".
  4. Run this script in a terminal:
       python simulate_live_traffic.py --mode all
"""

import argparse
import socket
import sys
import time
import urllib.request

def send_benign_http():
    """Generates standard benign HTTP traffic flows."""
    print("[*] Generating benign web traffic flows (HTTP/HTTPS)...")
    urls = [
        "http://httpbin.org/get",
        "http://httpbin.org/status/200",
        "http://httpbin.org/headers",
        "http://127.0.0.1:5000/api/health"
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Cognisec-Live-Monitor/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                status = resp.status
                print(f"    [+] Benign request to {url} -> HTTP {status}")
        except Exception as e:
            print(f"    [-] Request to {url}: {e}")
        time.sleep(0.5)

def send_simulated_portscan():
    """Generates rapid TCP probe packets across local ports (simulating a lightweight port scan flow)."""
    print("[*] Generating simulated port scan probe flows (rapid connection attempts)...")
    target_host = "127.0.0.1"
    target_ports = [80, 443, 8080, 21, 22, 25, 3306, 5000, 8443, 9000]

    for port in target_ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.15)
        try:
            s.connect((target_host, port))
            print(f"    [!] Probe connection to {target_host}:{port} -> OPEN")
        except socket.timeout:
            print(f"    [!] Probe connection to {target_host}:{port} -> TIMEOUT")
        except ConnectionRefusedError:
            print(f"    [!] Probe connection to {target_host}:{port} -> CLOSED (RST packet observed)")
        except Exception as e:
            print(f"    [!] Probe {port}: {e}")
        finally:
            s.close()
        time.sleep(0.1)

def send_simulated_burst():
    """Generates high-frequency burst requests mimicking DoS/DDoS flow rate spikes."""
    print("[*] Generating simulated traffic burst (high-rate HTTP queries)...")
    url = "http://127.0.0.1:5000/api/health"
    for i in range(25):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                pass
        except Exception:
            pass
        if (i + 1) % 5 == 0:
            print(f"    [+] Burst queries sent: {i + 1}/25")
        time.sleep(0.04)

def main():
    parser = argparse.ArgumentParser(description="Cognisec AI - Live Network Traffic Generator")
    parser.add_argument(
        "--mode",
        choices=["benign", "scan", "burst", "all"],
        default="all",
        help="Type of network activity to generate (default: all)"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=2,
        help="Number of iterations to execute (default: 2)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("   COGNISEC AI: LIVE NETWORK TRAFFIC GENERATION UTILITY")
    print("   Generates authentic packet flows to test Live Network Monitoring")
    print("=" * 70)
    print(f"[*] Target mode: {args.mode.upper()} | Cycles: {args.loops}")
    print("[*] Please ensure Live Capture is STARTED on http://127.0.0.1:5000")
    print("=" * 70)

    for cycle in range(1, args.loops + 1):
        print(f"\n>>> Starting Cycle {cycle}/{args.loops} <<<")
        if args.mode in ("benign", "all"):
            send_benign_http()
        if args.mode in ("scan", "all"):
            send_simulated_portscan()
        if args.mode in ("burst", "all"):
            send_simulated_burst()
        time.sleep(1.0)

    print("\n" + "=" * 70)
    print("[SUCCESS] Traffic generation complete!")
    print("Check the Live Monitoring flow table in your browser to inspect the detected flows.")
    print("=" * 70)

if __name__ == "__main__":
    main()
