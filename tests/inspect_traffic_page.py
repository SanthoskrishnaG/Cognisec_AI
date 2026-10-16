import asyncio
import json
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9223

async def send_cmd(ws, msg_id, method, params=None):
    payload = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("id") == msg_id:
            return data

async def eval_js(ws, msg_id, expression):
    res = await send_cmd(ws, msg_id, "Runtime.evaluate", {
        "expression": expression,
        "returnByValue": True,
        "awaitPromise": True
    })
    return res.get("result", {}).get("result", {}).get("value")

async def main():
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--disable-gpu",
        "http://127.0.0.1:5000/"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = None
        for _ in range(15):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as r:
                    for t in json.loads(r.read().decode("utf-8")):
                        if t.get("type") == "page":
                            ws_url = t.get("webSocketDebuggerUrl")
                            break
                    if ws_url:
                        break
            except Exception:
                pass

        if not ws_url:
            print("Failed to get websocket debugger URL")
            return

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            mid = 1
            await send_cmd(ws, mid, "Page.enable"); mid += 1
            await send_cmd(ws, mid, "Runtime.enable"); mid += 1

            await asyncio.sleep(2.0)

            # Switch to Traffic Analysis
            await eval_js(ws, mid, "document.querySelector('.nav-item[data-view=\"traffic-analysis\"]').click()"); mid += 1
            await asyncio.sleep(0.5)

            # Click Load Sample CSV button
            await eval_js(ws, mid, "document.getElementById('loadSampleCsvBtn').click()"); mid += 1
            await asyncio.sleep(1.5)

            # Check bounding boxes at 1280px (100%), 150%, 200%, 768px, 390px
            test_cases = [
                ("100% (1280x800)", 1280, 800, 1.0),
                ("125% (1024x640)", 1024, 640, 1.25),
                ("150% (853x533)", 853, 533, 1.50),
                ("175% (731x457)", 731, 457, 1.75),
                ("200% (640x400)", 640, 400, 2.00),
                ("Mobile 768px", 768, 1024, 1.0),
                ("Mobile 390px", 390, 844, 1.0),
            ]

            print("=" * 80)
            print("  AUDITING TRAFFIC ANALYSIS PAGE ELEMENTS WITH LOADED DATA")
            print("=" * 80)

            for name, w, h, scale in test_cases:
                await send_cmd(ws, mid, "Emulation.setDeviceMetricsOverride", {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": scale,
                    "mobile": w < 768
                }); mid += 1
                await asyncio.sleep(0.3)

                info = await eval_js(ws, mid, """
                    (() => {
                        const winW = window.innerWidth;
                        const mainWrapper = document.querySelector('.main-wrapper');
                        const previewCard = document.getElementById('previewCard');
                        const uploadMetaBar = document.getElementById('uploadMetaBar');
                        const topNavbar = document.querySelector('.top-navbar');
                        const sampleBtn = document.getElementById('loadSampleCsvBtn');
                        const kpiCards = Array.from(document.querySelectorAll('#previewCard .kpi-card'));
                        
                        function checkElem(el, name) {
                            if (!el || el.offsetParent === null) return null;
                            const r = el.getBoundingClientRect();
                            return {
                                name: name,
                                left: Math.round(r.left),
                                right: Math.round(r.right),
                                width: Math.round(r.width),
                                clippedRight: Math.round(r.right - winW)
                            };
                        }

                        const items = [
                            checkElem(mainWrapper, 'main-wrapper'),
                            checkElem(topNavbar, 'top-navbar'),
                            checkElem(sampleBtn, 'loadSampleCsvBtn'),
                            checkElem(uploadMetaBar, 'uploadMetaBar'),
                            checkElem(previewCard, 'previewCard')
                        ];

                        kpiCards.forEach((c, i) => items.push(checkElem(c, 'kpi-card-' + i)));

                        const clipped = items.filter(x => x && x.clippedRight > 1);

                        return {
                            winW: winW,
                            items: items.filter(Boolean),
                            clipped: clipped
                        };
                    })()
                """); mid += 1

                clipped = info.get("clipped", [])
                if clipped:
                    print(f"\n[FAIL] {name} - Window Width: {info['winW']}px")
                    for c in clipped:
                        print(f"       -> Element '{c['name']}' right is {c['right']}px (CLIPPED by +{c['clippedRight']}px!)")
                else:
                    print(f"[PASS] {name} - Window Width: {info['winW']}px - All elements fit within viewport.")

    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(main())
