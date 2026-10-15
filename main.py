import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd_args, desc):
    print(f"\n================================================================================")
    print(f"[*] Step: {desc}")
    print(f"[*] Executing: {' '.join(cmd_args)}")
    print(f"================================================================================")
    res = subprocess.run(cmd_args)
    if res.returncode != 0:
        print(f"[!] Error: Step '{desc}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[+] Completed: {desc}")

def main():
    parser = argparse.ArgumentParser(
        description="Network Behaviour Fingerprinting Research Suite: XGBoost vs. Variational Autoencoders"
    )
    parser.add_argument(
        "--step",
        choices=["download", "prepare", "train-xgb", "train-vae", "experiment", "test", "dashboard", "all"],
        default="all",
        help="Specify which pipeline step to execute (default: all)"
    )

    args = parser.parse_args()
    py_bin = sys.executable

    if args.step == "download":
        run_command([py_bin, "-m", "src.data.download_cicids2017"], "Dataset Acquisition")
    elif args.step == "prepare":
        run_command([py_bin, "-m", "src.data.prepare_data"], "Data Preprocessing & Partitioning")
    elif args.step == "train-xgb":
        from src.training.train_xgboost import train_xgboost_pipeline
        import numpy as np
        train_data = np.load("data/processed/train_data.npz", allow_pickle=True)
        val_data = np.load("data/processed/val_data.npz", allow_pickle=True)
        train_xgboost_pipeline(train_data["X_unscaled"], train_data["y"], val_data["X_unscaled"], val_data["y"])
        print("[+] XGBoost training finished successfully.")
    elif args.step == "train-vae":
        from src.training.train_vae import train_vae_pipeline
        import numpy as np
        train_data = np.load("data/processed/train_data.npz", allow_pickle=True)
        val_data = np.load("data/processed/val_data.npz", allow_pickle=True)
        train_vae_pipeline(train_data["X_scaled"], train_data["y"], val_data["X_scaled"], val_data["y"])
        print("[+] VAE training finished successfully.")
    elif args.step == "experiment":
        run_command([py_bin, "-m", "src.experiments.run_experiments"], "Master Experiments & Figures")
    elif args.step == "test":
        run_command([py_bin, "-m", "pytest", "tests/"], "Automated Test Suite")
    elif args.step == "dashboard":
        run_command([py_bin, "-m", "streamlit", "run", "app.py"], "Streamlit Research Dashboard")
    elif args.step == "all":
        # Check if raw files exist
        raw_files = list(Path("data/raw").glob("*.csv"))
        if len(raw_files) < 8:
            run_command([py_bin, "-m", "src.data.download_cicids2017"], "Dataset Acquisition")
        run_command([py_bin, "-m", "src.data.prepare_data"], "Data Preprocessing & Partitioning")
        run_command([py_bin, "-m", "src.experiments.run_experiments"], "Master Experiments & Figures")
        run_command([py_bin, "-m", "pytest", "tests/"], "Automated Test Suite")
        print("\n================================================================================")
        print("[+] Full end-to-end pipeline successfully executed and validated!")
        print("[+] Run 'streamlit run app.py' to launch the demonstration dashboard.")
        print("================================================================================\n")

if __name__ == "__main__":
    main()
