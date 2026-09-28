import numpy as np
import torch

import ram.dataset.se3 as se3
from cmaes import CMA
from torch import Tensor
from jaxtyping import Float

from ram.dataset.self_collision import LINK_RADIUS, EPS
from ram.model import Model

# Initial CMA-ES step size, in units of the (unit total) link lengths.
SIGMA0 = 0.1


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        mask = (param.abs() >= 2 * LINK_RADIUS).float()
        return param * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output):
        param, l2_norm, norm = ctx.saved_tensors
        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param)
        )
        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def decode(alpha: Float[Tensor, "dofp1 1"], lengths: Float[Tensor, "pop dofp1 2"]) \
        -> Float[Tensor, "pop dofp1 3"]:
    params = torch.stack([Normaliser.apply(SquasherSTE.apply(Normaliser.apply(length))) for length in lengths])
    return torch.cat([alpha.unsqueeze(0).expand(lengths.shape[0], -1, -1), params], dim=2)


def objective(model: Model,
              morphs: Float[Tensor, "pop dofp1 3"],
              task: Float[Tensor, "num_samples 9"]) \
        -> Float[Tensor, "pop"]:
    pop, num_samples = morphs.shape[0], task.shape[0]
    logit = model(morphs.repeat_interleave(num_samples, dim=0), task.repeat(pop, 1))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction='none')
    return loss.view(pop, num_samples).mean(dim=1)


def ours(initial_morph: Float[Tensor, "dofp1 3"],
         task: Float[Tensor, "num_samples 4 4"],
         n_iter: int,
         optimiser: str = "GD") \
        -> Float[Tensor, "{n_iter+1} dofp1 3"]:
    """
    Args:
        initial_morph: Morphology the search is started from.
        task: Poses the morphology should reach.
        n_iter: Number of optimisation steps. A gradient step costs one model evaluation, a CMA-ES generation costs
            `population_size` (4 + floor(3 ln d)) of them.
        optimiser: "GD" to differentiate through the model with AdamW, "CMA" to query it as a black box with CMA-ES.
    """
    morphs = torch.zeros(n_iter + 1, initial_morph.shape[0], initial_morph.shape[1], device=initial_morph.device)
    morphs[0] = initial_morph.clone()

    task = se3.to_vector(task)
    alpha = initial_morph[:, :1].detach()

    model = Model.from_id(2069).to(initial_morph.device)

    if optimiser == "GD":
        lengths = initial_morph[:, 1:].clone()
        lengths.requires_grad = True
        optimizer = torch.optim.AdamW([lengths], lr=0.01)

        for i in range(n_iter):
            optimizer.zero_grad()

            loss = objective(model, decode(alpha, lengths.unsqueeze(0)), task)[0]

            loss.backward()
            optimizer.step()

            morphs[i + 1] = decode(alpha, lengths.unsqueeze(0))[0].detach()

    elif optimiser == "CMA":
        x0 = initial_morph[:, 1:].reshape(-1).double().cpu().numpy()
        optimizer = CMA(mean=x0, sigma=SIGMA0, bounds=np.tile(np.array([-1.0, 1.0]), (x0.shape[0], 1)))

        def to_lengths(xs: Float[np.ndarray, "pop dim"]) -> Float[Tensor, "pop dofp1 2"]:
            return torch.as_tensor(xs, device=alpha.device, dtype=alpha.dtype).view(-1, alpha.shape[0], 2)

        with torch.no_grad():
            for i in range(n_iter):
                xs = np.stack([optimizer.ask() for _ in range(optimizer.population_size)])

                losses = objective(model, decode(alpha, to_lengths(xs)), task)

                optimizer.tell(list(zip(xs, losses.double().tolist())))

                morphs[i + 1] = decode(alpha, to_lengths(optimizer.mean))[0]

    else:
        raise ValueError(f"Unknown optimiser {optimiser!r}.")

    return morphs
