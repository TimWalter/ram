import torch
from torch import Tensor
from jaxtyping import Float

import ram.dataset.se3 as se3
from ram.dataset.morphology import get_joint_limits
from ram.dataset.workspace import sample_workspace


def sample_target_trajectory(morph: Float[Tensor, "dofp1 3"], num_samples: int) \
        -> Float[Tensor, "num_samples 4 4"]:
    """
    Samples a geodesic in SE(3) between two reachable poses of the morphology. Only the end points are guaranteed to be
    reachable, the waypoints in between generally are not.
    """
    joint_limits = get_joint_limits(morph)
    reachable_poses = sample_workspace(morph.unsqueeze(0).expand(10000, -1, -1),
                                       joint_limits.unsqueeze(0).expand(10000, -1, -1))[0]
    start, end = reachable_poses[0], reachable_poses[1]

    t = torch.linspace(0, 1, num_samples, device=morph.device).view(-1, 1)
    return se3.exp(start.repeat(num_samples, 1, 1), t * se3.log(start, end))
