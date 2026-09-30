import numpy as np
import torch

import ram.dataset.se3 as se3
from torch import Tensor
from jaxtyping import Float

from ram.model import Model
from paper_archive.RQ3.trajectory.batch_cma import BatchCMA

SIGMA0 = 0.1

def objective(model: Model,
              morph: Float[Tensor, "dofp1 3"],
              target_trajectory: Float[Tensor, "num_samples 4 4"],
              offsets: Float[Tensor, "pop num_samples 6"]) \
        -> Float[Tensor, "pop num_samples"]:
    pop, num_samples = offsets.shape[0], offsets.shape[1]
    target = target_trajectory.expand(pop, -1, -1, -1)
    trajectory = se3.exp(target, offsets)

    logit = model(morph.expand(pop * num_samples, -1, -1), se3.to_vector(trajectory).view(-1, 9))
    deviation = se3.distance(target, trajectory).squeeze(-1)
    return -10.0 * torch.sigmoid(logit).view(pop, num_samples) + 0.5 * deviation


def ours(morph: Float[Tensor, "dofp1 3"],
         target_trajectory: Float[Tensor, "num_samples 4 4"],
         n_iter: int,
         optimiser: str = "GD") \
        -> Float[Tensor, "{n_iter+1} num_samples 4 4"]:
    """
    Args:
        morph: The (fixed) morphology.
        target_trajectory: Poses the trajectory should stay close to.
        n_iter: Number of optimisation steps. A gradient step costs one model evaluation, a CMA-ES generation costs
            `population_size` (4 + floor(3 ln 6)) of them.
        optimiser: "GD" to differentiate through the model with AdamW, "CMA" to query it as a black box with one CMA-ES
            per waypoint.
    """
    trajectories = torch.zeros(n_iter + 1, *target_trajectory.shape, device=morph.device)
    trajectories[0] = target_trajectory.clone()

    num_samples = target_trajectory.shape[0]
    model = Model.from_id(2069).to(morph.device)

    if optimiser == "GD":
        offset = torch.zeros(1, num_samples, 6, device=morph.device, requires_grad=True)
        optimizer = torch.optim.AdamW([offset], lr=0.002)

        for i in range(n_iter):
            optimizer.zero_grad()

            loss = objective(model, morph, target_trajectory, offset).mean()
            loss.backward()
            optimizer.step()

            trajectories[i + 1] = se3.exp(target_trajectory, offset[0]).detach()

    elif optimiser == "CMA":
        # A single CMA-ES over all num_samples * 6 offsets would ignore that the objective decomposes over waypoints.
        optimizer = BatchCMA(torch.zeros(num_samples, 6, device=morph.device), SIGMA0,
                             bounds=torch.tensor([[-1.0, 1.0]] * 6, device=morph.device))
        scale = torch.tensor([0.5, 0.5, 0.5, np.pi / 2, np.pi / 2, np.pi / 2], device=morph.device)

        def to_offsets(xs: Float[Tensor, "num_samples pop 6"]) -> Float[Tensor, "pop num_samples 6"]:
            return xs.to(morph.dtype).transpose(0, 1) * scale

        with torch.no_grad():
            for i in range(n_iter):
                xs = optimizer.ask()

                losses = objective(model, morph, target_trajectory, to_offsets(xs))

                optimizer.tell(xs, losses.T)

                trajectories[i + 1] = se3.exp(target_trajectory, to_offsets(optimizer.mean[:, None])[0])

    else:
        raise ValueError(f"Unknown optimiser {optimiser!r}.")

    return trajectories
