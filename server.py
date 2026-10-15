"""
Entrypoint for Cognisec AI - Network Behaviour Fingerprinting Web Application.
Launches the Flask HTTP server on http://localhost:5000 (or specified port).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    Ensure the script executes under the local virtual environment (.venv)
    where PyTorch and all ML dependencies are installed, avoiding Python 3.14
    or system interpreter conflicts.
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
        print("  4. Launch the application:")
        print("     python server.py")
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


    # Verify required dependencies
    missing = []
    for pkg in ["torch", "xgboost", "flask", "sklearn", "pandas", "numpy"]:
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

from src.api.server import create_app

def main():
    parser = argparse.ArgumentParser(
        description="Launch Cognisec AI Network Behaviour Fingerprinting Web Server"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind the server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the HTTP server on (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode"
    )

    args = parser.parse_args()

    app = create_app()
    print("=" * 80)
    print("   COGNISEC AI - NETWORK BEHAVIOUR FINGERPRINTING RESEARCH DASHBOARD")
    print("   Tagline: Intelligent Network Behaviour Analysis for Unseen Attack Detection")
    print("   Academic: An Experimental Comparison of XGBoost and Variational Autoencoders")
    print("=" * 80)
    print(f"[*] Server listening at: http://{args.host}:{args.port}")
    print(f"[*] API Health check:     http://{args.host}:{args.port}/api/health")
    print(f"[*] Press Ctrl+C to terminate.")
    print("=" * 80)

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
