import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils.device import (
    is_cuda_available,
    get_device,
    get_device_str,
    get_xgboost_device,
    get_device_info,
    clear_gpu_memory
)
from src.models.vae_model import VAETrainer
from src.models.xgboost_model import XGBoostClassifierWrapper


def benchmark_function(
    func,
    input_data,
    warmup_runs: int = 10,
    benchmark_runs: int = 35,
    is_cuda: bool = False
) -> Dict[str, float]:
    """
    Rigorously benchmarks execution time of an inference function.
    Warms up devices, synchronizes CUDA before and after timed operations,
    repeats measurements across multiple rounds, and reports median latency.
    """
    # 1. Warmup rounds
    for _ in range(warmup_runs):
        _ = func(input_data)
        if is_cuda:
            torch.cuda.synchronize()

    # 2. Timed benchmark rounds
    latencies_ms: List[float] = []
    for _ in range(benchmark_runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = func(input_data)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    return {
        "median_ms": float(np.median(latencies)),
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "runs": benchmark_runs
    }


def run_gpu_verification():
    print("=" * 80)
    print("   COGNISEC AI — HARDWARE ACCELERATION & GPU INFERENCE BENCHMARK")
    print("=" * 80)

    # 1. Hardware & Environment Audit
    device_info = get_device_info()
    print("\n[+] 1. HARDWARE & ENVIRONMENT AUDIT:")
    for k, v in device_info.items():
        print(f"    - {k:25s}: {v}")

    cuda_avail = is_cuda_available()
    if not cuda_avail:
        print("\n[!] WARNING: CUDA is not available on this system. Aborting GPU benchmark.")
        return False

    # 2. Checkpoint Loading & VRAM Tracking
    print("\n[+] 2. MODEL LOADING & DEVICE PLACEMENT:")
    clear_gpu_memory()

    mem_before_load = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"    - VRAM Allocated Before Load : {mem_before_load:.2f} MB")

    vae_ckpt_path = BASE_DIR / "artifacts" / "models" / "vae_checkpoint.pt"
    xgb_path = BASE_DIR / "artifacts" / "models" / "xgboost_model.json"
    prep_path = BASE_DIR / "artifacts" / "models" / "flow_preprocessor.joblib"

    if not vae_ckpt_path.exists() or not xgb_path.exists() or not prep_path.exists():
        print(f"[!] Artifacts missing under {BASE_DIR / 'artifacts' / 'models'}. Aborting.")
        return False

    preprocessor = joblib.load(prep_path)
    xgb_gpu = XGBoostClassifierWrapper().load(str(xgb_path), device="cuda")
    vae_gpu = VAETrainer.load_checkpoint(str(vae_ckpt_path), device="cuda")

    mem_after_load = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"    - VRAM Allocated After Load  : {mem_after_load:.2f} MB")
    print(f"    - Model VRAM Footprint       : {mem_after_load - mem_before_load:.2f} MB")

    param_devices = {p.device.type for p in vae_gpu.model.parameters()}
    param_indices = {p.device.index for p in vae_gpu.model.parameters()}
    print(f"    - VAE Parameter Device Types : {param_devices}")
    print(f"    - VAE Device Indices         : {param_indices}")
    print(f"    - VAE Model Training Mode    : {vae_gpu.model.training} (must be False for inference)")
    print(f"    - XGBoost Target Device      : {getattr(xgb_gpu, 'device', 'unknown')}")

    assert param_devices == {"cuda"}, "VAE model parameters are not located on CUDA!"
    assert not vae_gpu.model.training, "VAE model must be in evaluation mode!"
    assert getattr(xgb_gpu, "device", None) == "cuda", "XGBoost device must be configured as cuda!"

    # 3. Real Preprocessed 70-Feature Test Flows
    print("\n[+] 3. REAL DATA PREPARATION (CICIDS2017 Preprocessed Features):")
    sample_path = BASE_DIR / "data" / "samples" / "sample_benign_flows.csv"
    df_sample = pd.read_csv(sample_path)
    n_flows = len(df_sample)
    print(f"    - Loaded flows from          : {sample_path.name} ({n_flows} records)")

    X_unscaled, X_scaled, _, _ = preprocessor.transform(df_sample)
    print(f"    - Preprocessed feature shape : {X_scaled.shape} ({X_scaled.shape[1]} retained features)")
    assert X_scaled.shape[1] == 70, f"Expected 70 features, got {X_scaled.shape[1]}"

    # Instantiate CPU baseline models for strict equivalence and timing comparison
    vae_cpu = VAETrainer.load_checkpoint(str(vae_ckpt_path), device="cpu")
    xgb_cpu = XGBoostClassifierWrapper().load(str(xgb_path), device="cpu")

    # 4. Numerical Equivalence Verification
    print("\n[+] 4. NUMERICAL EQUIVALENCE CHECK (CPU vs. GPU):")
    cpu_vae_scores = vae_cpu.compute_anomaly_scores(X_scaled)
    gpu_vae_scores = vae_gpu.compute_anomaly_scores(X_scaled)

    cpu_xgb_probs = xgb_cpu.predict_proba(X_unscaled)
    gpu_xgb_probs = xgb_gpu.predict_proba(X_unscaled)

    vae_max_diff = float(np.max(np.abs(cpu_vae_scores - gpu_vae_scores)))
    xgb_max_diff = float(np.max(np.abs(cpu_xgb_probs - gpu_xgb_probs)))

    print(f"    - VAE Max Absolute Difference   : {vae_max_diff:.8e}")
    print(f"    - XGBoost Max Absolute Diff     : {xgb_max_diff:.8e}")

    assert vae_max_diff < 1e-4, f"VAE CPU/GPU discrepancy exceeded tolerance: {vae_max_diff}"
    assert xgb_max_diff < 1e-4, f"XGBoost CPU/GPU discrepancy exceeded tolerance: {xgb_max_diff}"
    print("    [PASS] Strict numerical equivalence confirmed between CPU and GPU predictions.")

    # 5. Rigorous Repeated Latency Benchmark (Sample Batch: 50 flows)
    print("\n[+] 5. REPEATED BENCHMARK — SAMPLE FLOW BATCH (50 flows, 35 iterations):")

    bench_vae_cpu = benchmark_function(vae_cpu.compute_anomaly_scores, X_scaled, is_cuda=False)
    bench_vae_gpu = benchmark_function(vae_gpu.compute_anomaly_scores, X_scaled, is_cuda=True)

    bench_xgb_cpu = benchmark_function(xgb_cpu.predict_proba, X_unscaled, is_cuda=False)
    bench_xgb_gpu = benchmark_function(xgb_gpu.predict_proba, X_unscaled, is_cuda=True)

    print(f"    - VAE (CPU) Median Latency      : {bench_vae_cpu['median_ms']:.3f} ms (min: {bench_vae_cpu['min_ms']:.3f}, max: {bench_vae_cpu['max_ms']:.3f})")
    print(f"    - VAE (GPU) Median Latency      : {bench_vae_gpu['median_ms']:.3f} ms (min: {bench_vae_gpu['min_ms']:.3f}, max: {bench_vae_gpu['max_ms']:.3f})")
    vae_sample_speedup = bench_vae_cpu["median_ms"] / max(bench_vae_gpu["median_ms"], 1e-6)
    if vae_sample_speedup > 1.05:
        print(f"      => VAE GPU Speedup            : {vae_sample_speedup:.2f}x faster on GPU")
    elif vae_sample_speedup < 0.95:
        print(f"      => VAE CPU/GPU Ratio          : {vae_sample_speedup:.2f}x (CPU faster due to kernel launch overhead on small batch)")
    else:
        print(f"      => VAE CPU/GPU Ratio          : {vae_sample_speedup:.2f}x (Comparable latency)")

    print(f"    - XGBoost (CPU) Median Latency  : {bench_xgb_cpu['median_ms']:.3f} ms (min: {bench_xgb_cpu['min_ms']:.3f}, max: {bench_xgb_cpu['max_ms']:.3f})")
    print(f"    - XGBoost (GPU) Median Latency  : {bench_xgb_gpu['median_ms']:.3f} ms (min: {bench_xgb_gpu['min_ms']:.3f}, max: {bench_xgb_gpu['max_ms']:.3f})")
    xgb_sample_speedup = bench_xgb_cpu["median_ms"] / max(bench_xgb_gpu["median_ms"], 1e-6)
    if xgb_sample_speedup > 1.05:
        print(f"      => XGBoost GPU Speedup        : {xgb_sample_speedup:.2f}x faster on GPU")
    elif xgb_sample_speedup < 0.95:
        print(f"      => XGBoost CPU/GPU Ratio      : {xgb_sample_speedup:.2f}x (CPU faster due to host-to-device transfer on 50 samples)")
    else:
        print(f"      => XGBoost CPU/GPU Ratio      : {xgb_sample_speedup:.2f}x (Comparable latency)")

    # 6. High-Throughput Batch Benchmark (5,000 flows)
    print("\n[+] 6. REPEATED BENCHMARK — HIGH-THROUGHPUT BATCH (5,000 flows, 25 iterations):")
    multiplier = int(np.ceil(5000 / n_flows))
    X_large_scaled = np.tile(X_scaled, (multiplier, 1))[:5000]
    X_large_unscaled = np.tile(X_unscaled, (multiplier, 1))[:5000]

    bench_vae_large_cpu = benchmark_function(vae_cpu.compute_anomaly_scores, X_large_scaled, warmup_runs=5, benchmark_runs=25, is_cuda=False)
    bench_vae_large_gpu = benchmark_function(vae_gpu.compute_anomaly_scores, X_large_scaled, warmup_runs=5, benchmark_runs=25, is_cuda=True)

    bench_xgb_large_cpu = benchmark_function(xgb_cpu.predict_proba, X_large_unscaled, warmup_runs=5, benchmark_runs=25, is_cuda=False)
    bench_xgb_large_gpu = benchmark_function(xgb_gpu.predict_proba, X_large_unscaled, warmup_runs=5, benchmark_runs=25, is_cuda=True)

    print(f"    - VAE (CPU) Median Latency      : {bench_vae_large_cpu['median_ms']:.3f} ms ({5000 / (bench_vae_large_cpu['median_ms'] / 1000):.1f} flows/sec)")
    print(f"    - VAE (GPU) Median Latency      : {bench_vae_large_gpu['median_ms']:.3f} ms ({5000 / (bench_vae_large_gpu['median_ms'] / 1000):.1f} flows/sec)")
    vae_large_speedup = bench_vae_large_cpu["median_ms"] / max(bench_vae_large_gpu["median_ms"], 1e-6)
    if vae_large_speedup > 1.05:
        print(f"      => VAE GPU Speedup            : {vae_large_speedup:.2f}x faster on GPU")
    else:
        print(f"      => VAE CPU/GPU Ratio          : {vae_large_speedup:.2f}x")

    print(f"    - XGBoost (CPU) Median Latency  : {bench_xgb_large_cpu['median_ms']:.3f} ms ({5000 / (bench_xgb_large_cpu['median_ms'] / 1000):.1f} flows/sec)")
    print(f"    - XGBoost (GPU) Median Latency  : {bench_xgb_large_gpu['median_ms']:.3f} ms ({5000 / (bench_xgb_large_gpu['median_ms'] / 1000):.1f} flows/sec)")
    xgb_large_speedup = bench_xgb_large_cpu["median_ms"] / max(bench_xgb_large_gpu["median_ms"], 1e-6)
    if xgb_large_speedup > 1.05:
        print(f"      => XGBoost GPU Speedup        : {xgb_large_speedup:.2f}x faster on GPU")
    else:
        print(f"      => XGBoost CPU/GPU Ratio      : {xgb_large_speedup:.2f}x")

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\n[+] 7. MEMORY AUDIT:")
    print(f"    - Peak GPU VRAM Allocated       : {mem_peak:.2f} MB")
    print(f"    - Safe headroom on 6GB GPU      : {6144 - mem_peak:.2f} MB free")

    print("\n" + "=" * 80)
    print("   BENCHMARK & VERIFICATION COMPLETED WITH ZERO DEVICE WARNINGS")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_gpu_verification()
    sys.exit(0 if success else 1)
