"""Cognisec AI - Comprehensive Automated Responsiveness & Browser Zoom Test Suite.
Validates:
  1. All 9 Application Views
  2. All Required Browser Zoom Levels: 50%, 75%, 90%, 100%, 110%, 125%, 150%, 175%, 200%
  3. All Required Viewport Widths: 1920px, 1440px, 1280px, 1024px, 768px, 390px
  4. Fully Populated State: Loads sample data into Traffic Analysis (70-column table)
     and executes dual inference for Attack Detection.
  5. Verifies ZERO horizontal clipping / unintended overflow.
  6. Collects browser console errors.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9225
BASE_URL = "http://127.0.0.1:5000"
REPORTS_DIR = Path("reports/screenshots/zoom_audit")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_audit():
    print("=" * 80)
    print("  COGNISEC AI: RIGOROUS BROWSER ZOOM & VIEWPORT RESPONSIVENESS AUDIT")
    print("=" * 80)

    chrome_proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"{BASE_URL}/"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    pending_futures = {}
    console_errors = []

    async def cdp_reader(ws):
        try:
            async for msg in ws:
                data = json.loads(msg)
                if "id" in data and data["id"] in pending_futures:
                    fut = pending_futures.pop(data["id"])
                    if not fut.done():
                        fut.set_result(data)
                elif data.get("method") == "Runtime.consoleAPICalled":
                    params = data.get("params", {})
                    if params.get("type") == "error":
                        console_errors.append(params)
        except Exception:
            pass

    try:
        ws_url = None
        for _ in range(15):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2) as r:
                    tabs = json.loads(r.read().decode("utf-8"))
                    for t in tabs:
                        if t.get("type") == "page":
                            ws_url = t.get("webSocketDebuggerUrl")
                            break
                    if ws_url:
                        break
            except Exception:
                pass

        if not ws_url:
            raise RuntimeError("Could not connect to Chrome CDP.")

        print(f"[*] Connected to Chrome remote debugging on port {PORT}")

        async with websockets.connect(ws_url, max_size=25*1024*1024) as ws:
            reader_task = asyncio.create_task(cdp_reader(ws))

            async def send_cmd(mid, method, params=None):
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                pending_futures[mid] = fut
                await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
                return await fut

            async def eval_js(mid, expr):
                res = await send_cmd(mid, "Runtime.evaluate", {
                    "expression": expr,
                    "returnByValue": True,
                    "awaitPromise": True
                })
                return res.get("result", {}).get("result", {}).get("value")

            mid = 1
            await send_cmd(mid, "Page.enable"); mid += 1
            await send_cmd(mid, "Runtime.enable"); mid += 1

            # Wait for initial dashboard load
            await asyncio.sleep(2.0)

            # Ingest sample CSV data so Traffic Analysis and Attack Detection have full data loaded
            print("[*] Ingesting sample flow dataset to populate 70-column preview & inference tables...")
            await eval_js(mid, """
                (() => {
                    const btn = document.querySelector('.nav-item[data-view="traffic-analysis"]');
                    if (btn) btn.click();
                })()
            """); mid += 1
            await asyncio.sleep(0.5)

            await eval_js(mid, """
                (() => {
                    const btn = document.getElementById('loadSampleCsvBtn');
                    if (btn) btn.click();
                })()
            """); mid += 1
            await asyncio.sleep(1.5)

            # Run dual inference in Attack Detection so results table is populated
            print("[*] Running dual inference to populate attack detection table...")
            await eval_js(mid, """
                (() => {
                    const btn = document.querySelector('.nav-item[data-view="attack-detection"]');
                    if (btn) btn.click();
                })()
            """); mid += 1
            await asyncio.sleep(0.5)

            await eval_js(mid, """
                (() => {
                    const btn = document.getElementById('runDualBtn');
                    if (btn && !btn.disabled) btn.click();
                })()
            """); mid += 1
            await asyncio.sleep(2.0)

            views = [
                "overview",
                "live-monitoring",
                "traffic-analysis",
                "attack-detection",
                "model-comparison",
                "unseen-attacks",
                "dataset-experiments",
                "research-methodology",
                "system-info"
            ]

            viewports = [1920, 1440, 1280, 1024, 768, 390]
            zoom_scales = [0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00]

            audit_results = []

            # -------------------------------------------------------------
            # Part 1: Viewport Testing across all 9 views
            # -------------------------------------------------------------
            print("\n" + "-" * 80)
            print("1. TESTING VIEWPORT SIZES (1920, 1440, 1280, 1024, 768, 390 CSS px)")
            print("-" * 80)

            for vp_w in viewports:
                vp_h = 1080 if vp_w >= 1024 else (900 if vp_w >= 768 else 844)
                await send_cmd(mid, "Emulation.setDeviceMetricsOverride", {
                    "width": vp_w,
                    "height": vp_h,
                    "deviceScaleFactor": 1.0,
                    "mobile": vp_w < 768
                }); mid += 1
                await asyncio.sleep(0.2)

                for v in views:
                    # Switch view
                    await eval_js(mid, f"""
                        (() => {{
                            const btn = document.querySelector('.nav-item[data-view="{v}"]');
                            if (btn) btn.click();
                        }})()
                    """); mid += 1
                    await asyncio.sleep(0.1)

                    # Check clipping on key structural components
                    metrics = await eval_js(mid, """
                        (() => {
                            const winW = window.innerWidth;
                            const mainWrapper = document.querySelector('.main-wrapper');
                            const topNavbar = document.querySelector('.top-navbar');
                            const activeSection = document.querySelector('.view-section.active');
                            const cards = Array.from(activeSection ? activeSection.querySelectorAll('.card') : []);
                            
                            function getClipped(el, name) {
                                if (!el) return null;
                                const r = el.getBoundingClientRect();
                                const clipped = Math.round(r.right - winW);
                                return clipped > 2 ? { name: name, right: Math.round(r.right), clipped: clipped } : null;
                            }

                            const clippedItems = [
                                getClipped(mainWrapper, 'main-wrapper'),
                                getClipped(topNavbar, 'top-navbar'),
                                getClipped(activeSection, 'activeSection')
                            ];
                            cards.forEach((c, idx) => {
                                const cl = getClipped(c, 'card-' + idx);
                                if (cl) clippedItems.push(cl);
                            });

                            const allClipped = clippedItems.filter(Boolean);
                            return {
                                winW: winW,
                                pass: allClipped.length === 0,
                                clipped: allClipped
                            };
                        })()
                    """); mid += 1

                    status_str = "PASS" if metrics and metrics.get("pass") else "FAIL"
                    audit_results.append({
                        "test": f"Viewport {vp_w}px",
                        "view": v,
                        "status": status_str,
                        "details": metrics
                    })

                    if status_str == "PASS":
                        print(f"  [PASS] Width: {vp_w:>4d}px | View: {v:22s} | No clipped elements")
                    else:
                        print(f"  [FAIL] Width: {vp_w:>4d}px | View: {v:22s} | Clipped: {metrics.get('clipped')}")

            # -------------------------------------------------------------
            # Part 2: Browser Zoom Levels (50% to 200%) on standard 1280 base
            # -------------------------------------------------------------
            print("\n" + "-" * 80)
            print("2. TESTING BROWSER ZOOM LEVELS (50%, 75%, 90%, 100%, 110%, 125%, 150%, 175%, 200%)")
            print("-" * 80)

            for scale in zoom_scales:
                zoom_pct = f"{int(scale * 100)}%"
                eff_w = int(1280 / scale)
                eff_h = int(800 / scale)

                await send_cmd(mid, "Emulation.setDeviceMetricsOverride", {
                    "width": eff_w,
                    "height": eff_h,
                    "deviceScaleFactor": scale,
                    "mobile": eff_w < 768
                }); mid += 1
                await asyncio.sleep(0.2)

                for v in views:
                    await eval_js(mid, f"""
                        (() => {{
                            const btn = document.querySelector('.nav-item[data-view="{v}"]');
                            if (btn) btn.click();
                        }})()
                    """); mid += 1
                    await asyncio.sleep(0.1)

                    metrics = await eval_js(mid, """
                        (() => {
                            const winW = window.innerWidth;
                            const mainWrapper = document.querySelector('.main-wrapper');
                            const topNavbar = document.querySelector('.top-navbar');
                            const activeSection = document.querySelector('.view-section.active');
                            const cards = Array.from(activeSection ? activeSection.querySelectorAll('.card') : []);
                            
                            function getClipped(el, name) {
                                if (!el) return null;
                                const r = el.getBoundingClientRect();
                                const clipped = Math.round(r.right - winW);
                                return clipped > 2 ? { name: name, right: Math.round(r.right), clipped: clipped } : null;
                            }

                            const clippedItems = [
                                getClipped(mainWrapper, 'main-wrapper'),
                                getClipped(topNavbar, 'top-navbar'),
                                getClipped(activeSection, 'activeSection')
                            ];
                            cards.forEach((c, idx) => {
                                const cl = getClipped(c, 'card-' + idx);
                                if (cl) clippedItems.push(cl);
                            });

                            const allClipped = clippedItems.filter(Boolean);
                            return {
                                winW: winW,
                                pass: allClipped.length === 0,
                                clipped: allClipped
                            };
                        })()
                    """); mid += 1

                    status_str = "PASS" if metrics and metrics.get("pass") else "FAIL"
                    audit_results.append({
                        "test": f"Zoom {zoom_pct}",
                        "view": v,
                        "status": status_str,
                        "details": metrics
                    })

                    if status_str == "PASS":
                        print(f"  [PASS] Zoom: {zoom_pct:>4s} (eff {eff_w:>4d}px) | View: {v:22s} | No clipped elements")
                    else:
                        print(f"  [FAIL] Zoom: {zoom_pct:>4s} (eff {eff_w:>4d}px) | View: {v:22s} | Clipped: {metrics.get('clipped')}")

            # -------------------------------------------------------------
            # Part 3: Capture Full Page Proof Screenshots
            # -------------------------------------------------------------
            print("\n[*] Capturing audit verification screenshots at key zoom/viewport states...")
            proof_states = [
                ("traffic_100_1280.png", "traffic-analysis", 1280, 800, 1.0),
                ("traffic_150_zoom.png", "traffic-analysis", 853, 533, 1.5),
                ("traffic_200_zoom.png", "traffic-analysis", 640, 400, 2.0),
                ("traffic_390_mobile.png", "traffic-analysis", 390, 844, 1.0),
                ("attack_detection_100.png", "attack-detection", 1280, 800, 1.0),
                ("overview_100.png", "overview", 1280, 800, 1.0),
                ("model_comparison_100.png", "model-comparison", 1280, 800, 1.0)
            ]

            for fname, vname, w, h, scale in proof_states:
                await send_cmd(mid, "Emulation.setDeviceMetricsOverride", {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": scale,
                    "mobile": w < 768
                }); mid += 1
                await eval_js(mid, f"document.querySelector('.nav-item[data-view=\"{vname}\"]').click()"); mid += 1
                await asyncio.sleep(0.3)
                shot = await send_cmd(mid, "Page.captureScreenshot", {"format": "png"}); mid += 1
                if shot and "result" in shot and "data" in shot["result"]:
                    img_data = base64.b64decode(shot["result"]["data"])
                    out_path = REPORTS_DIR / fname
                    out_path.write_bytes(img_data)
                    print(f"    Saved screenshot: {out_path}")

            total_runs = len(audit_results)
            passed_runs = sum(1 for r in audit_results if r["status"] == "PASS")
            failed_runs = total_runs - passed_runs

            print("\n" + "=" * 80)
            print(f"AUDIT OUTCOME: {passed_runs} / {total_runs} PASSED ({passed_runs/total_runs*100:.1f}%)")
            print(f"Console Errors Encountered: {len(console_errors)}")
            print("=" * 80)

            reader_task.cancel()
            return failed_runs == 0 and len(console_errors) == 0

    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except Exception:
            chrome_proc.kill()

if __name__ == "__main__":
    success = asyncio.run(run_audit())
    sys.exit(0 if success else 1)
