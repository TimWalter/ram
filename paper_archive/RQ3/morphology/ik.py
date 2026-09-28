import numpy as np
import torch.utils.dlpack

from cmaes import CMA
from torch import Tensor
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp
import optax
import jax.dlpack

import paper_archive.RQ3.jax_ram.se3 as jax_se3
import paper_archive.RQ3.jax_ram.kinematics as jax_kinematics
from paper_archive.RQ3.jax_ram.self_collision import collision_check, EPS, LINK_RADIUS

SIGMA0 = 0.1


@jax.custom_vjp
def squasher(param):
    mask = (jnp.abs(param) >= 2 * LINK_RADIUS).astype(param.dtype)
    return param * mask


def squasher_fwd(param):
    return squasher(param), None


def squasher_bwd(res, g):
    return (g,)


squasher.defvjp(squasher_fwd, squasher_bwd)


@jax.custom_vjp
def normaliser(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return param / safe_norm


def normaliser_fwd(param):
    l2_norm = jnp.hypot(param[:, 0:1], param[:, 1:2])
    norm = jnp.sum(l2_norm, axis=0, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-12)
    return (param / safe_norm), (param, l2_norm, safe_norm)


def normaliser_bwd(res, g):
    param, l2_norm, safe_norm = res
    safe_l2_norm = jnp.maximum(l2_norm, 1e-12)

    chain = jnp.where(
        jnp.any(jnp.abs(param) > EPS, axis=1, keepdims=True),
        param / safe_l2_norm,
        jnp.zeros_like(param)
    )
    grad_param = (g * safe_norm - chain * jnp.sum(g * param)) / (safe_norm ** 2)
    return (grad_param,)


normaliser.defvjp(normaliser_fwd, normaliser_bwd)


def decode(alpha: Float[Array, "dofp1 1"], lengths: Float[Array, "pop dofp1 2"]) \
        -> Float[Array, "pop dofp1 3"]:
    params = jax.vmap(lambda length: normaliser(squasher(normaliser(length))))(lengths)
    return jnp.concatenate([jnp.broadcast_to(alpha, (lengths.shape[0], *alpha.shape)), params], axis=2)


def loss_fn(current_morph: Float[Array, "dofp1 3"], task: Float[Array, "num_samples 4 4"]):
    bmorph = jnp.broadcast_to(current_morph, (task.shape[0], *current_morph.shape))

    optimal_joints = jax_kinematics.numerical_inverse_kinematics(current_morph, task)
    reached_poses = jax_kinematics.forward_kinematics(bmorph, optimal_joints)

    critical_distance = collision_check(bmorph, reached_poses, debug=True)

    ee_poses = reached_poses[:, -1, :, :]
    dists = jax_se3.distance(ee_poses, task)

    loss = jnp.mean(jax.nn.relu(dists[:, 0] - EPS) + 20 * jax.nn.relu(-critical_distance))

    return loss


def objective(alpha: Float[Array, "dofp1 1"],
              lengths: Float[Array, "pop dofp1 2"],
              task: Float[Array, "num_samples 4 4"]) \
        -> Float[Array, "pop"]:
    return jax.vmap(loss_fn, in_axes=(0, None))(decode(alpha, lengths), task)


decode_fn = jax.jit(decode)
objective_fn = jax.jit(objective)
grad_fn = jax.jit(jax.grad(lambda alpha, lengths, task: objective(alpha, lengths[None], task)[0], argnums=1))


def ik(initial_morph: Float[Tensor, "dofp1 3"],
       task: Float[Tensor, "num_samples 4 4"],
       n_iter: int,
       optimiser: str = "GD") \
        -> Float[Tensor, "{n_iter+1} dofp1 3"]:
    """
    Args:
        initial_morph: Morphology the search is started from.
        task: Poses the morphology should reach.
        n_iter: Number of optimisation steps. A gradient step costs one evaluation of the ground-truth kinematics, a
            CMA-ES generation costs `population_size` (4 + floor(3 ln d)) of them.
        optimiser: "GD" to differentiate through the kinematics with AdamW, "CMA" to query them as a black box with
            CMA-ES.
    """
    morphs = torch.zeros(n_iter + 1, initial_morph.shape[0], initial_morph.shape[1], device=initial_morph.device)
    morphs[0] = initial_morph.clone()

    device = initial_morph.device
    task = jax.dlpack.from_dlpack(task.contiguous().clone())
    alpha = jax.dlpack.from_dlpack(initial_morph[:, :1].contiguous().clone())

    if optimiser == "GD":
        lengths = jax.dlpack.from_dlpack(initial_morph[:, 1:].contiguous().clone())
        optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=0.01)
        )
        opt_state = optimizer.init(lengths)

        for i in range(n_iter):
            grads = grad_fn(alpha, lengths, task)

            updates, opt_state = optimizer.update(grads, opt_state, lengths)
            lengths = optax.apply_updates(lengths, updates)

            morphs[i + 1] = torch.from_dlpack(decode_fn(alpha, lengths[None])[0]).to(device)

    elif optimiser == "CMA":
        x0 = initial_morph[:, 1:].reshape(-1).double().cpu().numpy()
        optimizer = CMA(mean=x0, sigma=SIGMA0, bounds=np.tile(np.array([-1.0, 1.0]), (x0.shape[0], 1)))

        def to_lengths(xs: Float[np.ndarray, "pop dim"]) -> Float[Array, "pop dofp1 2"]:
            return jnp.asarray(xs, dtype=alpha.dtype).reshape(-1, alpha.shape[0], 2)

        for i in range(n_iter):
            xs = np.stack([optimizer.ask() for _ in range(optimizer.population_size)])

            losses = objective_fn(alpha, to_lengths(xs), task)

            optimizer.tell(list(zip(xs, np.asarray(losses, dtype=np.float64).tolist())))

            morphs[i + 1] = torch.from_dlpack(decode_fn(alpha, to_lengths(optimizer.mean))[0]).to(device)

    else:
        raise ValueError(f"Unknown optimiser {optimiser!r}.")

    return morphs
