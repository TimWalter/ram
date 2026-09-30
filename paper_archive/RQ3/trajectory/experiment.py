import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float

import ram.dataset.se3 as se3
from ram.metrics import bootstrap_mean_ci
from ram.dataset.morphology import sample_morph
from ram.dataset.kinematics import numerical_inverse_kinematics, forward_kinematics
from ram.dataset.self_collision import collision_check, EPS

from paper_archive.RQ3.trajectory.task import sample_target_trajectory
from paper_archive.RQ3.trajectory.ours import ours
from paper_archive.RQ3.trajectory.ik import ik

import jax
torch.set_float32_matmul_precision("highest")
jax.config.update("jax_default_matmul_precision", "highest")

device = torch.device("cuda")

cache = Path(__file__).parent / "cache"
cache.mkdir(parents=True, exist_ok=True)
filepath = cache / "experiment.pkl"
inputs_path = cache / "experiment_inputs.pt"
# Config
seeds = 1000
num_samples = 100
n_iter = 100
pairs = [("Ours", "CMA"), ("Ours", "GD"), ("IK", "CMA"), ("IK", "GD")]
metrics = ["success_rate", "self_collisions", "pose_error", "deviation"]


def evaluate(morph: Float[Tensor, "dofp1 3"],
             target_trajectory: Float[Tensor, "num_samples 4 4"],
             trajectories: Float[Tensor, "n_traj num_samples 4 4"]) \
        -> Float[Tensor, "4 n_traj"]:
    """
    Scores each trajectory with the ground-truth numerical inverse kinematics.

    Returns:
        Per trajectory the success rate (fraction of waypoints reached within EPS without self-collision), the number
        of self-colliding waypoints, the mean pose error and the mean deviation from the target trajectory, in the
        order of `metrics`.
    """
    n_traj, n_samples = trajectories.shape[0], trajectories.shape[1]
    poses = trajectories.reshape(-1, 4, 4)
    bmorph = morph.repeat(poses.shape[0], 1, 1)

    joints = numerical_inverse_kinematics(bmorph, poses)[0]
    reached_pose = forward_kinematics(bmorph, joints)
    error = se3.distance(reached_pose[:, -1, :, :], poses).view(n_traj, n_samples)
    self_collision = collision_check(bmorph, reached_pose).view(n_traj, n_samples)
    deviation = se3.distance(target_trajectory.expand(n_traj, -1, -1, -1), trajectories).squeeze(-1)

    success = (error < EPS) & ~self_collision
    return torch.stack([success.float().mean(dim=1), self_collision.float().sum(dim=1), error.mean(dim=1),
                        deviation.mean(dim=1)])


if inputs_path.exists():
    inputs = torch.load(inputs_path)
    morph = inputs["morph"].to(device)
    target_trajectory = inputs["target_trajectory"].to(device)
else:
    torch.manual_seed(0)
    morph = sample_morph(seeds, 6, False, device)
    target_trajectory = torch.stack([sample_target_trajectory(morph[seed], num_samples) for seed in range(seeds)])
    torch.save({"morph": morph, "target_trajectory": target_trajectory}, inputs_path)

experiment = pickle.load(open(filepath, "rb")) if filepath.exists() else {}

for method, optimiser in pairs:
    if optimiser in experiment.get(method, {}):
        continue
    call = ours if method == "Ours" else ik
    metric_accumulator = torch.zeros(len(metrics), seeds, n_iter + 1, device=device)
    for seed in tqdm(range(seeds), f"[{method}][{optimiser}] Seed"):
        trajectories = call(morph[seed], target_trajectory[seed], n_iter, optimiser)
        torch.manual_seed(seed)
        metric_accumulator[:, seed] = evaluate(morph[seed], target_trajectory[seed], trajectories)
        torch.cuda.empty_cache()

    experiment.setdefault(method, {})[optimiser] = {
        metric: torch.stack(bootstrap_mean_ci(metric_accumulator[i]), dim=1)
        for i, metric in enumerate(metrics)
    }
    with open(filepath, "wb") as file:
        pickle.dump(experiment, file)
