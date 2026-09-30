import math

import numpy as np
import torch
from cmaes import CMA
from torch import Tensor
from jaxtyping import Float

# Same constants as cmaes._cma.
_EPS = 1e-8
_SIGMA_MAX = 1e32


class BatchCMA:
    """
    `num` independent CMA-ES instances of the same dimension, stepped together with batched tensor operations instead
    of one Python object each. Every instance follows cmaes.CMA (v0.13, without learning rate adaptation) step by step,
    only the random numbers differ as they are drawn from a single torch generator.
    """

    def __init__(self,
                 mean: Float[Tensor, "num dim"],
                 sigma: float,
                 bounds: Float[Tensor, "dim 2"] | None = None,
                 population_size: int | None = None,
                 n_max_resampling: int = 100,
                 seed: int = 0):
        num, dim = mean.shape
        dd = dict(device=mean.device, dtype=torch.float64)

        # Borrow the strategy parameters from the reference implementation so they match exactly.
        ref = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.population_size = ref.population_size
        self._mu = ref._mu
        self._mu_eff = ref._mu_eff
        self._cc = ref._cc
        self._c1 = ref._c1
        self._cmu = ref._cmu
        self._c_sigma = ref._c_sigma
        self._d_sigma = ref._d_sigma
        self._cm = ref._cm
        self._chi_n = ref._chi_n
        self._weights = torch.as_tensor(ref._weights, **dd)

        self._num, self._dim = num, dim
        self.mean = mean.to(torch.float64).clone()
        self._sigma = torch.full((num,), float(sigma), **dd)
        self._C = torch.eye(dim, **dd).repeat(num, 1, 1)
        self._p_sigma = torch.zeros(num, dim, **dd)
        self._pc = torch.zeros(num, dim, **dd)
        self._B: Tensor | None = None
        self._D: Tensor | None = None

        self._bounds = None if bounds is None else bounds.to(**dd)
        self._n_max_resampling = n_max_resampling
        self._g = 0
        self._generator = torch.Generator(device=mean.device).manual_seed(seed)

    def _eigen_decomposition(self) -> tuple[Tensor, Tensor]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.transpose(1, 2)) / 2
        D2, B = torch.linalg.eigh(self._C)
        D = torch.sqrt(torch.where(D2 < 0, _EPS, D2))
        self._C = B @ torch.diag_embed(D ** 2) @ B.transpose(1, 2)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> Float[Tensor, "num pop dim"]:
        B, D = self._eigen_decomposition()
        z = torch.randn(self._num, self.population_size, self._dim, generator=self._generator,
                        device=self.mean.device, dtype=torch.float64)  # ~ N(0, I)
        y = z @ (B * D[:, None, :]).transpose(1, 2)  # ~ N(0, C)
        return self.mean[:, None, :] + self._sigma[:, None, None] * y  # ~ N(m, σ^2 C)

    def _is_infeasible(self, x: Float[Tensor, "num pop dim"]) -> Float[Tensor, "num pop"]:
        return ((x < self._bounds[:, 0]) | (x > self._bounds[:, 1])).any(dim=-1)

    def ask(self) -> Float[Tensor, "num pop dim"]:
        """Samples a full population for every instance."""
        x = self._sample_solution()
        if self._bounds is None:
            return x

        # Like cmaes.CMA.ask, resample infeasible solutions up to n_max_resampling times in total, then clip.
        infeasible = self._is_infeasible(x)
        for _ in range(self._n_max_resampling - 1):
            if not infeasible.any():
                return x
            x = torch.where(infeasible[..., None], self._sample_solution(), x)
            infeasible = self._is_infeasible(x)
        if infeasible.any():
            repaired = self._sample_solution().clamp(self._bounds[:, 0], self._bounds[:, 1])
            x = torch.where(infeasible[..., None], repaired, x)
        return x

    def tell(self, x: Float[Tensor, "num pop dim"], values: Float[Tensor, "num pop"]) -> None:
        """Updates every instance with its evaluated population."""
        self._g += 1
        order = torch.argsort(values.to(torch.float64), dim=1, stable=True)
        x_k = torch.gather(x.to(torch.float64), 1, order[..., None].expand(-1, -1, self._dim))

        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        y_k = (x_k - self.mean[:, None, :]) / self._sigma[:, None, None]  # ~ N(0, C)

        y_w = (y_k[:, :self._mu] * self._weights[:self._mu, None]).sum(dim=1)  # eq.41
        self.mean = self.mean + self._cm * self._sigma[:, None] * y_w

        C_2 = B @ torch.diag_embed(1 / D) @ B.transpose(1, 2)  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (C_2 @ y_w[..., None])[..., 0]

        norm_p_sigma = torch.linalg.norm(self._p_sigma, dim=-1)
        self._sigma = self._sigma * torch.exp((self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1))
        self._sigma = self._sigma.clamp(max=_SIGMA_MAX)

        h_sigma_cond_left = norm_p_sigma / math.sqrt(1 - (1 - self._c_sigma) ** (2 * (self._g + 1)))
        h_sigma_cond_right = (1.4 + 2 / (self._dim + 1)) * self._chi_n
        h_sigma = (h_sigma_cond_left < h_sigma_cond_right).to(torch.float64)  # (p.28)

        self._pc = (1 - self._cc) * self._pc + h_sigma[:, None] * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        w_io = self._weights * torch.where(
            self._weights >= 0,
            1.0,
            self._dim / (torch.linalg.norm(y_k @ C_2.transpose(1, 2), dim=-1) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)

        rank_one = self._pc[:, :, None] * self._pc[:, None, :]
        rank_mu = torch.einsum("np,npi,npj->nij", w_io, y_k, y_k)

        self._C = (
            (1 + self._c1 * delta_h_sigma - self._c1 - self._cmu * self._weights.sum())[:, None, None] * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )
