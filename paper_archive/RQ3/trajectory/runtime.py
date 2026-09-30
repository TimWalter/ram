import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import time
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from ram.metrics import bootstrap_mean_ci
from ram.dataset.morphology import sample_morph

from paper_archive.RQ3.trajectory.task import sample_target_trajectory
from paper_archive.RQ3.trajectory.ours import ours
from paper_archive.RQ3.trajectory.ik import ik

device = torch.device("cuda")

cache = Path(__file__).parent / "cache"
cache.mkdir(parents=True, exist_ok=True)
filepath = cache / "runtime.pkl"
inputs_path = cache / "runtime_inputs.pt"
# Config
seeds = 10
sizes = torch.logspace(0, 4, 10).int().tolist()
n_iter = 100
pairs = [("Ours", "CMA"), ("Ours", "GD"), ("IK", "CMA"), ("IK", "GD")]

if inputs_path.exists():
    inputs = torch.load(inputs_path)
    morph = inputs["morph"].to(device)
    target_trajectory = {size: t.to(device) for size, t in inputs["target_trajectory"].items()}
else:
    torch.manual_seed(0)
    morph = sample_morph(seeds, 6, False, device)
    target_trajectory = {
        size: torch.stack([sample_target_trajectory(morph[seed], size) for seed in range(seeds)])
        for size in sizes
    }
    torch.save({"morph": morph, "target_trajectory": target_trajectory}, inputs_path)

runtime = pickle.load(open(filepath, "rb")) if filepath.exists() else {}

for method, optimiser in pairs:
    if optimiser in runtime.get(method, {}):
        continue
    call = ours if method == "Ours" else ik
    result = torch.zeros(len(sizes), 3, device=device)
    for i, size in enumerate(sizes):
        # Warm up
        call(morph[0], target_trajectory[size][0], 1, optimiser)
        runtime_accumulator = torch.zeros(seeds, 1, device=device)
        for seed in tqdm(range(seeds), f"[{method}][{optimiser}|{size}] Seed"):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            call(morph[seed], target_trajectory[size][seed], n_iter, optimiser)
            torch.cuda.synchronize()
            runtime_accumulator[seed] = time.perf_counter_ns() - start
        result[i, :] = torch.cat(bootstrap_mean_ci(runtime_accumulator))
        torch.cuda.empty_cache()

    runtime.setdefault(method, {})[optimiser] = result
    with open(filepath, "wb") as file:
        pickle.dump(runtime, file)
