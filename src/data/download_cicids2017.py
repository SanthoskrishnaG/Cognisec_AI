import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd

FILES_TO_DOWNLOAD = [
    {
        "name": "Friday-WorkingHours-Afternoon-PortScan.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
        "primary_attack": "PortScan"
    },
    {
        "name": "Friday-WorkingHours-Afternoon-DDos.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
        "primary_attack": "DDoS"
    },
    {
        "name": "Friday-WorkingHours-Morning.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
        "primary_attack": "Bot"
    },
    {
        "name": "Monday-WorkingHours.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Monday-WorkingHours.pcap_ISCX.csv.parquet",
        "primary_attack": "BENIGN"
    },
    {
        "name": "Tuesday-WorkingHours.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
        "primary_attack": "Brute_Force"
    },
    {
        "name": "Wednesday-workingHours.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Wednesday-workingHours.pcap_ISCX.csv.parquet",
        "primary_attack": "DoS"
    },
    {
        "name": "Thursday-WorkingHours-Morning-WebAttacks.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
        "primary_attack": "Web_Attack"
    },
    {
        "name": "Thursday-WorkingHours-Afternoon-Infilteration.csv",
        "url": "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
        "primary_attack": "Infiltration"
    }
]

def acquire_and_save_dataset(
    target_dir: str = "data/raw",
    max_benign_per_file: int = 25000
) -> Dict[str, Dict[str, int]]:
    """
    Downloads authentic CICIDS2017 flow files via curl, preserves 100% of attack records,
    and saves standardized authentic CSVs into data/raw/.
    """
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_report = {}
    temp_parquet = out_dir / "_temp_download.parquet"

    for item in FILES_TO_DOWNLOAD:
        csv_path = out_dir / item["name"]

        # Check if already present and valid
        if csv_path.exists() and csv_path.stat().st_size > 1000000:
            print(f"[+] Skipping {item['name']} (already present, size: {round(csv_path.stat().st_size / (1024*1024), 2)} MB)")
            df_existing = pd.read_csv(csv_path, nrows=5)
            summary_report[item["name"]] = {
                "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2),
                "status": "cached"
            }
            continue

        print(f"[*] Downloading {item['name']} via curl...")
        t0 = time.time()
        
        # Download with curl.exe following redirects
        curl_cmd = ["curl.exe", "-L", "-s", "-o", str(temp_parquet), item["url"]]
        res = subprocess.run(curl_cmd, capture_output=True)
        if res.returncode != 0 or not temp_parquet.exists() or temp_parquet.stat().st_size < 1000:
            print(f"[!] Warning: Failed downloading {item['name']}: {res.stderr.decode('utf-8', errors='ignore')}")
            continue

        # Load local parquet
        df = pd.read_parquet(temp_parquet)
        df.columns = [c.strip() for c in df.columns]

        # Standardise label column name
        if "Label" not in df.columns:
            for c in df.columns:
                if "label" in c.lower():
                    df.rename(columns={c: "Label"}, inplace=True)
                    break

        label_col = "Label"
        attack_mask = (df[label_col].astype(str).str.strip().str.upper() != "BENIGN")
        attack_df = df[attack_mask]
        benign_df = df[~attack_mask]

        if len(benign_df) > max_benign_per_file:
            benign_df = benign_df.sample(n=max_benign_per_file, random_state=42)

        curated_df = pd.concat([benign_df, attack_df], ignore_index=True)
        curated_df = curated_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        curated_df.to_csv(csv_path, index=False)
        duration = round(time.time() - t0, 2)

        summary_report[item["name"]] = {
            "total_records": len(curated_df),
            "benign_records": len(benign_df),
            "attack_records": len(attack_df),
            "attacks_preserved_percentage": 100.0,
            "seconds_to_acquire": duration,
            "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2)
        }
        print(f"    -> Saved {len(curated_df)} records ({len(attack_df)} attacks) to {csv_path.name} in {duration}s")

        if temp_parquet.exists():
            temp_parquet.unlink()

    return summary_report

if __name__ == "__main__":
    import json
    report = acquire_and_save_dataset()
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open("reports/dataset_acquisition_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print("[+] Dataset acquisition successfully finished.")
