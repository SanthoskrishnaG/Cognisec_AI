import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def get_virtual_env_python():
    """
    Resolves the project virtual environment Python executable.
    Returns Path if it exists, or None.
    """
    win_venv = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    posix_venv = ROOT_DIR / ".venv" / "bin" / "python"
    if os.name == "nt" and win_venv.exists():
        return win_venv
    if posix_venv.exists():
        return posix_venv
    if win_venv.exists():
        return win_venv
    return None

def is_same_interpreter(path1, path2) -> bool:
    """
    Robustly checks whether two interpreter paths refer to the same executable,
    handling Windows case-insensitivity, 8.3 short names, and symbolic links.
    """
    if path1 is None or path2 is None:
        return False
    try:
        return os.path.samefile(path1, path2)
    except (FileNotFoundError, OSError, ValueError):
        return Path(path1).resolve() == Path(path2).resolve()

def _ensure_runtime_environment():
    """
    Ensures main.py executes under the local virtual environment (.venv)
    where PyTorch and all ML dependencies are installed.
    Stops with a clear setup message if .venv does not exist.
    """
    target_venv = get_virtual_env_python()

    if target_venv is None:
        print("=" * 80)
        print("  [!] COGNISEC AI: VIRTUAL ENVIRONMENT NOT FOUND")
        print("=" * 80)
        print(f"Current interpreter: {sys.executable} (Python {sys.version.split()[0]})")
        print("\nCognisec AI requires an isolated virtual environment with Python 3.10 - 3.12")
        print("to support PyTorch, XGBoost, and the scientific ML pipeline.")
        print("The global Python interpreter will NOT be used silently.\n")
        print("Setup Procedure:")
        print("  1. Verify Python 3.12 is installed (or use py launcher):")
        print("     py -3.12 --version")
        print("  2. Create the project virtual environment:")
        print("     py -3.12 -m venv .venv")
        print("  3. Activate and install verified dependencies:")
        print("     .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        print("  4. Re-run your command:")
        print(f"     python {' '.join(sys.argv)}")
        print("=" * 80)
        sys.exit(1)

    # Check if already running inside the target virtual environment
    if is_same_interpreter(sys.executable, target_venv):
        pass  # Already running under correct virtual environment; no relaunch needed
    else:
        # Prevent recursive relaunch loops
        if os.environ.get("_COGNISEC_RELAUNCH_GUARD") == "1":
            print("=" * 80)
            print("  [!] COGNISEC AI: RECURSIVE RELAUNCH DETECTED")
            print("=" * 80)
            print(f"Target interpreter:  {target_venv}")
            print(f"Current interpreter: {sys.executable}")
            print("Failed to switch to virtual environment. Aborting.")
            print("=" * 80)
            sys.exit(1)

        print(f"[*] Cognisec AI: Switching to project virtual environment ({target_venv})...")
        env = os.environ.copy()
        env["_COGNISEC_RELAUNCH_GUARD"] = "1"
        try:
            res = subprocess.run([str(target_venv)] + sys.argv, cwd=str(ROOT_DIR), env=env)
            sys.exit(res.returncode)
        except KeyboardInterrupt:
            # Clean exit on Ctrl+C without traceback
            sys.exit(0)


    # Verify that core ML dependencies are available in this environment
    missing = []
    for pkg in ["torch", "xgboost", "sklearn", "pandas", "numpy", "flask"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 80)
        print("  [!] COGNISEC AI: INCOMPLETE VIRTUAL ENVIRONMENT")
        print("=" * 80)
        print(f"Interpreter: {sys.executable} (Python {sys.version.split()[0]})")
        print(f"Missing packages: {', '.join(missing)}")
        print("\nPlease install the required dependencies into .venv:")
        print("  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        print("=" * 80)
        sys.exit(1)

_ensure_runtime_environment()

def run_command(cmd_args, desc):
    print(f"\n================================================================================")
    print(f"[*] Step: {desc}")
    print(f"[*] Executing: {' '.join(cmd_args)}")
    print(f"================================================================================")
    res = subprocess.run(cmd_args, cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print(f"[!] Error: Step '{desc}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[+] Completed: {desc}")

def main():
    parser = argparse.ArgumentParser(
        description="Cognisec AI: Network Behaviour Fingerprinting Research Suite"
    )
    parser.add_argument(
        "--step",
        choices=["download", "prepare", "train-xgb", "train-vae", "experiment", "test", "dashboard", "web", "streamlit", "all"],
        default="all",
        help="Specify which pipeline step to execute (default: all)"
    )

    args = parser.parse_args()
    
    # We are already executing inside target_venv
    py_bin = str(Path(sys.executable).resolve())
    print(f"[*] Cognisec AI: Using Python interpreter: {py_bin} (Python {sys.version.split()[0]})")

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
        run_command([py_bin, "-m", "pytest", "tests/", "-v"], "Automated Test Suite")
    elif args.step in ["dashboard", "web"]:
        run_command([py_bin, "server.py"], "Cognisec AI Web Dashboard (HTML5/CSS3/JS)")
    elif args.step == "streamlit":
        # Check streamlit availability for legacy dashboard
        try:
            import streamlit
        except ImportError:
            print("[!] Streamlit is not installed in .venv. Installing it:")
            print("    .\\.venv\\Scripts\\python.exe -m pip install streamlit")
            sys.exit(1)
        run_command([py_bin, "-m", "streamlit", "run", "app.py"], "Legacy Streamlit Dashboard")
    elif args.step == "all":
        # Check if raw files exist
        raw_files = list(Path("data/raw").glob("*.csv"))
        if len(raw_files) < 8:
            run_command([py_bin, "-m", "src.data.download_cicids2017"], "Dataset Acquisition")
        run_command([py_bin, "-m", "src.data.prepare_data"], "Data Preprocessing & Partitioning")
        run_command([py_bin, "-m", "src.experiments.run_experiments"], "Master Experiments & Figures")
        run_command([py_bin, "-m", "pytest", "tests/", "-v"], "Automated Test Suite")
        print("\n================================================================================")
        print("[+] Full end-to-end pipeline successfully executed and validated!")
        print("[+] Run 'python server.py' to launch the Cognisec AI web dashboard.")
        print("================================================================================\n")

if __name__ == "__main__":
    main()
