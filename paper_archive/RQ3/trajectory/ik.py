import numpy as np
import torch.utils.dlpack

from torch import Tensor
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp
import optax
import jax.dlpack

import paper_archive.RQ3.jax_ram.se3 as jax_se3
import paper_archive.RQ3.jax_ram.kinematics as jax_kinematics
from paper_archive.RQ3.jax_ram.self_collision import collision_check, EPS
from paper_archive.RQ3.trajectory.batch_cma import BatchCMA

SIGMA0 = 0.1


def loss_fn(morph: Float[Array, "dofp1 3"],
            target_trajectory: Float[Array, "num_samples 4 4"],
            offsets: Float[Array, "num_samples 6"]) \
        -> Float[Array, "num_samples"]:
    """
    Per waypoint loss, it decomposes over waypoints as an offset only moves its own waypoint.
    """
    trajectory = jax_se3.exp(target_trajectory, offsets)
    bmorph = jnp.broadcast_to(morph, (trajectory.shape[0], *morph.shape))

    optimal_joints = jax_kinematics.numerical_inverse_kinematics(morph, trajectory)
    reached_poses = jax_kinematics.forward_kinematics(bmorph, optimal_joints)

    critical_distance = collision_check(bmorph, reached_poses, debug=True)
    dists = jax_se3.distance(reached_poses[:, -1, :, :], trajectory)[:, 0]
    reachability = jax.nn.relu(dists - EPS) + 1e4 * jax.nn.relu(-critical_distance)

    deviation = jax_se3.distance(target_trajectory, trajectory)[:, 0]
    return 10.0 * reachability + 0.5 * deviation


exp_fn = jax.jit(jax_se3.exp)
objective_fn = jax.jit(jax.vmap(loss_fn, in_axes=(None, None, 0)))
grad_fn = jax.jit(jax.grad(lambda morph, target, offsets: loss_fn(morph, target, offsets).mean(), argnums=2))


def ik(morph: Float[Tensor, "dofp1 3"],
       target_trajectory: Float[Tensor, "num_samples 4 4"],
       n_iter: int,
       optimiser: str = "GD") \
        -> Float[Tensor, "{n_iter+1} num_samples 4 4"]:
    """
    Args:
        morph: The (fixed) morphology.
        target_trajectory: Poses the trajectory should stay close to.
        n_iter: Number of optimisation steps. A gradient step costs one evaluation of the ground-truth kinematics, a
            CMA-ES generation costs `population_size` (4 + floor(3 ln 6)) of them.
        optimiser: "GD" to differentiate through the kinematics with AdamW, "CMA" to query them as a black box with one
            CMA-ES per waypoint.
    """
    trajectories = torch.zeros(n_iter + 1, *target_trajectory.shape, device=morph.device)
    trajectories[0] = target_trajectory.clone()

    device = morph.device
    num_samples = target_trajectory.shape[0]
    target = jax.dlpack.from_dlpack(target_trajectory.contiguous().clone())
    jax_morph = jax.dlpack.from_dlpack(morph.contiguous().clone())

    if optimiser == "GD":
        offset = jnp.zeros((num_samples, 6), dtype=jax_morph.dtype)
        optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=0.001)
        )
        opt_state = optimizer.init(offset)

        for i in range(n_iter):
            grads = grad_fn(jax_morph, target, offset)

            updates, opt_state = optimizer.update(grads, opt_state, offset)
            offset = optax.apply_updates(offset, updates)

            trajectories[i + 1] = torch.from_dlpack(exp_fn(target, offset)).to(device)

    elif optimiser == "CMA":
        # A single CMA-ES over all num_samples * 6 offsets would ignore that the objective decomposes over waypoints.
        optimizer = BatchCMA(torch.zeros(num_samples, 6, device=device), SIGMA0,
                             bounds=torch.tensor([[-1.0, 1.0]] * 6, device=device))
        scale = jnp.array([0.5, 0.5, 0.5, np.pi / 2, np.pi / 2, np.pi / 2])

        def to_offsets(xs: Float[Tensor, "num_samples pop 6"]) -> Float[Array, "pop num_samples 6"]:
            xs = jax.dlpack.from_dlpack(xs.to(morph.dtype).contiguous())
            return jnp.swapaxes(xs, 0, 1) * scale

        for i in range(n_iter):
            xs = optimizer.ask()

            losses = torch.from_dlpack(objective_fn(jax_morph, target, to_offsets(xs)))

            optimizer.tell(xs, losses.T)

            mean = to_offsets(optimizer.mean[:, None])[0]
            trajectories[i + 1] = torch.from_dlpack(exp_fn(target, mean)).to(device)

    else:
        raise ValueError(f"Unknown optimiser {optimiser!r}.")

    return trajectories
