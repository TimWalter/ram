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

from paper_archive.RQ3.morphology.ours import ours
from paper_archive.RQ3.morphology.ik import ik

device = torch.device("cuda")

cache = Path(__file__).parent / "cache"
cache.mkdir(parents=True, exist_ok=True)
filepath = cache / "experiment.pkl"
inputs_path = cache / "experiment_inputs.pt"
# Config
seeds = 1000
num_poses = 100
n_iter = 100
pairs = [("Ours", "CMA"), ("Ours", "GD"), ("IK", "CMA"), ("IK", "GD")]
metrics = ["success_rate", "self_collisions", "pose_error"]


def evaluate(morphs: Float[Tensor, "n_morphs dofp1 3"], task: Float[Tensor, "num_poses 4 4"]) \
        -> Float[Tensor, "3 n_morphs"]:
    """
    Scores each morphology on the task with the ground-truth numerical inverse kinematics.

    Returns:
        Per morphology the success rate (fraction of poses reached within EPS without self-collision), the number of
        self-colliding solutions and the mean pose error, in the order of `metrics`.
    """
    n_morphs, n_poses = morphs.shape[0], task.shape[0]
    bmorph = morphs.repeat_interleave(n_poses, dim=0)
    btask = task.repeat(n_morphs, 1, 1)

    joints = numerical_inverse_kinematics(bmorph, btask)[0]
    reached_pose = forward_kinematics(bmorph, joints)
    error = se3.distance(reached_pose[:, -1, :, :], btask).squeeze(-1).view(n_morphs, n_poses)
    self_collision = collision_check(bmorph, reached_pose).view(n_morphs, n_poses)

    success = (error < EPS) & ~self_collision
    return torch.stack([success.float().mean(dim=1), self_collision.float().sum(dim=1), error.mean(dim=1)])


if inputs_path.exists():
    inputs = torch.load(inputs_path)
    initial_morph = inputs["initial_morph"].to(device)
    task = inputs["task"].to(device)
else:
    initial_morph = sample_morph(seeds, 6, False, device)
    task = se3.random_ball(seeds * num_poses, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])) \
        .to(device).view(seeds, num_poses, 4, 4)
    torch.save({"initial_morph": initial_morph, "task": task}, inputs_path)

experiment = pickle.load(open(filepath, "rb")) if filepath.exists() else {}

for method, optimiser in pairs:
    if optimiser in experiment.get(method, {}):
        continue
    call = ours if method == "Ours" else ik
    metric_accumulator = torch.zeros(len(metrics), seeds, n_iter + 1, device=device)
    for seed in tqdm(range(seeds), f"[{method}][{optimiser}] Seed"):
        morphs = call(initial_morph[seed], task[seed], n_iter, optimiser)
        torch.manual_seed(seed)
        metric_accumulator[:, seed] = evaluate(morphs, task[seed])
        torch.cuda.empty_cache()

    experiment.setdefault(method, {})[optimiser] = {
        metric: torch.stack(bootstrap_mean_ci(metric_accumulator[i]), dim=1)
        for i, metric in enumerate(metrics)
    }
    with open(filepath, "wb") as file:
        pickle.dump(experiment, file)
