import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Bool, Float, Int
from scipy.stats import norm, spearmanr
from torch import Tensor
from tqdm import tqdm

from paper_archive.utils import get_plt_colour, latex_mean_and_ci
from ram.dataset.kinematics import inverse_kinematics, transformation_matrix
from ram.dataset.morphology import sample_morph
from ram.dataset.workspace import ball_approximation, sample_poses_in_reach
from ram.model import Model

CACHE_DIR = Path(__file__).parent / "cache"
DOFS = (5, 6, 7)
MAX_JOINTS = max(DOFS) + 1


def set_style():
    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsmath}",

        "axes.labelsize": 34,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34,
        "legend.fontsize": 34,
        "axes.titlesize": 34,
        "lines.linewidth": 3,
    })


@dataclass
class Attribute:
    """A scalar morphological attribute to be correlated against the latent space."""
    slug: str  # File name friendly identifier.
    table_label: str  # Semantic name followed by the LaTeX symbol, as used in the paper table.
    values: Float[Tensor, "batch"]

    @property
    def name(self) -> str:
        """Semantic name without the LaTeX symbol, used as axis label."""
        return self.table_label.split(" $")[0]


@dataclass
class Pairwise:
    """A pairwise (dis)similarity comparison to be evaluated with a Mantel test."""
    slug: str
    table_label: str
    latent_label: str
    other_label: str
    latent: Float[Tensor, "batch batch"]
    other: Float[Tensor, "batch batch"]


@jaxtyped(typechecker=beartype)
def pca(data: Float[Tensor, "batch dim"],
        standardise: bool = True,
        n_components: int = None,
        p_variance: float = None,
        factor_loading: bool = False) \
        -> Float[Tensor, "batch new_dim"]:
    """
    Perform Principial Component Analysis (PCA) on the data.
    Args:
        data: The input data to be PCA-ed.
        standardise: Whether to standardise the data before PCA.
        n_components: The number of principal components to keep. If None, all components are kept.
        p_variance: The proportion of variance to keep. If None, all variance is kept. Should be called with n_components=None and will otherwise take precendence.
        factor_loading: Whether to multiply the results by the eigenvalues.

    Returns:
        The PCA-ed data.
    """
    if standardise:
        data = (data - torch.mean(data, dim=0)) / (torch.std(data, dim=0))
    else:
        data = data - torch.mean(data, dim=0)
    u, s, vh = torch.linalg.svd(data, full_matrices=False)
    eigenvalues = (s ** 2) / (data.shape[0] - 1)

    k = s.shape[0]
    if p_variance:
        k = int(torch.where(torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues) >= p_variance)[0][0].item()) + 1
    elif n_components:
        k = n_components

    data = data @ vh[:k, :].T

    if factor_loading:
        data = data * torch.sqrt(eigenvalues[:k])

    return data


@jaxtyped(typechecker=beartype)
def correlation_profile(data: Float[Tensor, "batch dim"], attribute: Float[Tensor, "batch"]) \
        -> Float[Tensor, "dim 4"]:
    """
    Calculate the correlation coefficients for every data component vs a scalar.

    Args:
        data: The data to be correlated.
        attribute: The scalar property to be correlated against.
    Returns:
        Spearman correlation coefficients, Spearman CI bounds, and p-values.
    """
    spearman_result = spearmanr(data.cpu(), attribute.unsqueeze(1).cpu(), axis=0)
    spearman = torch.from_numpy(spearman_result.statistic[-1, :-1]).to(data.device)
    p_values = torch.from_numpy(spearman_result.pvalue[-1, :-1]).to(data.device)

    z      = torch.arctanh(spearman.clamp(-0.9999, 0.9999))
    se     = 1.0 / torch.sqrt(torch.tensor(data.shape[0] - 3, dtype=torch.float32, device=data.device))
    z_crit = torch.tensor(norm.ppf(1 - (1 - 0.95) / 2), dtype=torch.float32, device=data.device)
    ci_low  = torch.tanh(z - z_crit * se)
    ci_high = torch.tanh(z + z_crit * se)

    return torch.stack([spearman, ci_low, ci_high, p_values], dim=1)


@jaxtyped(typechecker=beartype)
def mantel_test(similarity_a: Float[Tensor, "batch batch"],
                similarity_b: Float[Tensor, "batch batch"],
                n_bootstrap: int = 9999,
                n_permutations: int = 9999) -> tuple[float, float, float, float]:
    """
    Mantel test: correlates the upper triangles of two symmetric matrices
    and estimates significance via permutation of rows/columns.

    Args:
        similarity_a: First symmetric matrix (similarity or distance).
        similarity_b: Second symmetric matrix (similarity or distance).
        n_bootstrap: Number of bootstrap samples for the CI.
        n_permutations: Number of permutations for the p-value estimate.
    Returns:
        Spearman correlation coefficient, Spearman CI bounds, and p-value.
    """
    idx = torch.triu_indices(similarity_a.shape[0], similarity_a.shape[0], offset=1)
    v_a = similarity_a[idx[0], idx[1]].cpu()
    v_b = similarity_b[idx[0], idx[1]].cpu()
    spearman = spearmanr(v_a, v_b).statistic

    bootstrap_r = torch.empty(n_bootstrap, device=similarity_a.device)
    for i in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        sample = torch.randint(0, similarity_a.shape[0], (similarity_a.shape[0],))
        a_boot = similarity_a[sample][:, sample]
        b_boot = similarity_b[sample][:, sample]
        bootstrap_r[i] = spearmanr(
            a_boot[idx[0], idx[1]].cpu(),
            b_boot[idx[0], idx[1]].cpu()
        ).statistic
    alpha = (1 - 0.95) / 2
    ci_low, ci_high = torch.quantile(bootstrap_r, torch.tensor([alpha, 1 - alpha], device=bootstrap_r.device))

    count = 0
    for _ in tqdm(range(n_permutations), desc="Permutations", leave=False):
        perm = torch.randperm(similarity_a.shape[0])
        b_perm = similarity_b[perm][:, perm]
        if spearmanr(v_a, b_perm[idx[0], idx[1]].cpu()).statistic >= spearman:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return spearman, ci_low.item(), ci_high.item(), p_value


@jaxtyped(typechecker=beartype)
def scatter(x: Float[Tensor, "batch"],
            y: Float[Tensor, "batch"],
            x_label: str,
            y_label: str,
            path: Path):
    """
    Scatter plot of two scalars, saved as PDF.

    Args:
        x: Values on the horizontal axis.
        y: Values on the vertical axis.
        x_label: Label of the horizontal axis.
        y_label: Label of the vertical axis.
        path: Destination of the PDF.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.scatter(x.cpu(), y.cpu(), color=get_plt_colour(0), alpha=0.5, s=30)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


@jaxtyped(typechecker=beartype)
def mdh_distance_matrix(morphs: Float[Tensor, "n max_joints 3"]) -> Float[Tensor, "n n"]:
    """
    Pairwise distance matrix between MDH parameter matrices.
    Handles variable DoF via zero-padding and normalises each parameter column.

    Args:
        morphs: MDH parameters (alpha, a, d), padded to max_joints with zeros.
    Returns:
        Symmetric pairwise distance matrix.
    """
    n = morphs.shape[0]

    # Normalise each of the 3 parameter columns independently
    # to put alpha, a, d on comparable scales
    flat = morphs.reshape(n, -1)                                  # (n, max_joints*3)
    std  = flat.std(dim=0).clamp(min=1e-8)                        # (max_joints*3,)
    flat_normed = flat / std                                       # (n, max_joints*3)

    # Pairwise L2 distance on the flattened, normalised parameters
    dist = torch.cdist(flat_normed, flat_normed)                   # (n, n)
    return dist


@jaxtyped(typechecker=beartype)
def sample_padded_morphs(num_robots: int, device: torch.device) -> Float[Tensor, "batch max_joints 3"]:
    """
    Sample robots of every DoF and zero-pad them to a common number of joints.

    Args:
        num_robots: Number of robots to sample per DoF.
        device: Torch device to operate on.
    Returns:
        MDH parameters of all sampled robots.
    """
    return torch.cat([
        torch.cat([sample_morph(num_robots, dof, False, device=device),
                   torch.zeros(num_robots, MAX_JOINTS - dof - 1, 3, device=device)], dim=1)
        for dof in DOFS
    ])


@jaxtyped(typechecker=beartype)
def compute_reachability(morphs: Float[Tensor, "batch max_joints 3"], num_poses: int) -> Float[Tensor, "batch"]:
    """
    Fraction of poses within the reachable ball that the robot can actually reach.

    Args:
        morphs: MDH parameters of the robots.
        num_poses: Number of poses sampled per robot.
    Returns:
        Reachability of every robot.
    """
    return torch.tensor([
        (inverse_kinematics(m, sample_poses_in_reach(num_poses, m))[-1] != -1).sum() / num_poses
        for m in tqdm(morphs, desc="Reachability")
    ])


@jaxtyped(typechecker=beartype)
def compute_reach_matrix(morphs: Float[Tensor, "batch max_joints 3"],
                         poses: Float[Tensor, "batch num_poses 4 4"],
                         chunk_size: int) -> Bool[Tensor, "batch batch num_poses"]:
    """
    Solve the IK of every robot for the poses sampled in reach of every robot.

    Args:
        morphs: MDH parameters of the robots.
        poses: Poses sampled within the reachable ball of each robot.
        chunk_size: Number of poses solved per IK call, trades memory for speed.
    Returns:
        Whether robot i reaches pose k of robot j.
    """
    num_robots, num_poses = poses.shape[0], poses.shape[1]
    flat = poses.reshape(num_robots * num_poses, 4, 4)

    reach = torch.zeros(num_robots, num_robots * num_poses, dtype=torch.bool, device=morphs.device)
    for i in tqdm(range(num_robots), desc="Workspace similarity"):
        reach[i] = torch.cat([
            inverse_kinematics(morphs[i], flat[start:start + chunk_size])[-1] != -1
            for start in range(0, flat.shape[0], chunk_size)
        ])
    return reach.reshape(num_robots, num_robots, num_poses)


@jaxtyped(typechecker=beartype)
def workspace_similarity(reach: Bool[Tensor, "batch batch num_poses"]) -> Float[Tensor, "batch batch"]:
    """
    Intersection over union of the reachable sets, evaluated on the poses sampled
    in reach of either of the two robots.

    Args:
        reach: Whether robot i reaches pose k of robot j.
    Returns:
        Symmetric pairwise workspace similarity.
    """
    self_reach = torch.diagonal(reach, dim1=0, dim2=1).T.unsqueeze(0)  # (1, batch, num_poses)

    # Both counts are evaluated on the poses of the second robot, the transpose
    # contributes the poses of the first one.
    intersection = (reach & self_reach).sum(dim=-1).float()
    union = (reach | self_reach).sum(dim=-1).float()

    similarity = (intersection + intersection.T) / (union + union.T + 1e-8)
    similarity.fill_diagonal_(1.0)
    return similarity


@jaxtyped(typechecker=beartype)
def morphological_attributes(morphs: Float[Tensor, "batch max_joints 3"],
                             num_per_dof: int,
                             reachability: Float[Tensor, "batch"]) -> list[Attribute]:
    """
    Assemble all scalar morphological attributes of the sampled robots.

    Args:
        morphs: MDH parameters of the robots.
        num_per_dof: Number of robots sampled per DoF.
        reachability: Precomputed reachability of every robot.
    Returns:
        All attributes to be correlated against the latent space.
    """
    device = morphs.device
    batch = morphs.shape[0]
    dof = torch.cat([torch.full((num_per_dof,), d) for d in DOFS]).to(device)
    eef_idx = dof.long()
    centre, radius = ball_approximation(morphs)

    link_lengths = torch.hypot(morphs[:, :, 1], morphs[:, :, 2])
    # Padded joints have zero length, offset the real ones to exclude the padding from the minimum.
    mask = (link_lengths != 0).int()
    shortest_link = torch.min(link_lengths.abs() - mask, dim=1).values + 1
    longest_link = torch.max(link_lengths.abs(), dim=1).values
    std_link = torch.std(link_lengths.abs(), dim=1)

    a = morphs[:, :, 1]
    d = morphs[:, :, 2]
    real_mask = (morphs.abs().sum(dim=-1) > 0)

    link_types = torch.zeros_like(a, dtype=torch.long)
    link_types[(a == 0) & (d != 0)] = 1
    link_types[(a != 0) & (d == 0)] = 2
    link_types[(a == 0) & (d == 0)] = 3
    link_types[~real_mask] = -1

    num_joints = (max(DOFS) - (link_types == -1).sum(dim=1))
    fraction_type = [(link_types == t).sum(dim=1) / num_joints for t in range(4)]
    pos3 = link_types.argmax(dim=1).float()

    return [
        Attribute("dof", r"DoF $n$", dof.float()),
        Attribute("moveable_length", r"Moveable Length $r_\text{mov}$", radius),
        Attribute("base_twist", r"Base Twist $\alpha_0$", morphs[:, 0, 0]),
        Attribute("base_length", r"Base Length $a_0$", morphs[:, 0, 1]),
        Attribute("base_offset", r"Base Offset $d_0$", morphs[:, 0, 2]),
        Attribute("first_twist", r"First Twist $\alpha_1$", morphs[:, 1, 0]),
        Attribute("first_length", r"First Length $a_1$", morphs[:, 1, 1]),
        Attribute("first_offset", r"First Offset $d_1$", morphs[:, 1, 2]),
        Attribute("eef_twist", r"EEF Twist $\alpha_n$", morphs[torch.arange(batch), eef_idx, 0]),
        Attribute("eef_length", r"EEF Length $a_n$", morphs[torch.arange(batch), eef_idx, 1]),
        Attribute("eef_offset", r"EEF Offset $d_n$", morphs[torch.arange(batch), eef_idx, 2]),
        Attribute("reachability", r"Reachability $\frac{\sum^{1000}_il_i}{1000}$",
                  reachability.to(device)),
        Attribute("centre_x", r"Centre X $\left(\bm{t}_0\right)_x$", centre[:, 0]),
        Attribute("centre_y", r"Centre Y $\left(\bm{t}_0\right)_y$", centre[:, 1]),
        Attribute("centre_z", r"Centre Z $\left(\bm{t}_0\right)_z$", centre[:, 2]),
        Attribute("centre_magnitude", r"Centre Magnitude $\anynorm{\bm{t}_0}_2$",
                  torch.linalg.norm(centre, dim=1)),
        Attribute("min_link", r"Min. Link Length $\min_i\sqrt{a_i^2+d_i^2}$", shortest_link),
        Attribute("max_link", r"Max. Link Length $\max_i\sqrt{a_i^2+d_i^2}$", longest_link),
        Attribute("std_link", r"Std. Link Length $\sigma\!\left(\sqrt{a_i^2+d_i^2}\right)$", std_link),
        *[Attribute(f"fraction_type{t}",
                    rf"Fraction Type {t} $\frac{{\sum_i\mathds{{1}}\left(j_i ={t}\right)}}{{n}}$",
                    fraction_type[t])
          for t in range(4)],
        Attribute("pos_type3", r"Max. Pos. Type 3 $\max_i\mathds{1}\left(j_i =3\right)$", pos3),
    ]


@jaxtyped(typechecker=beartype)
def format_p_value(p_value: float) -> str:
    """
    Format a p-value for the LaTeX table.

    Args:
        p_value: The p-value to be formatted.
    Returns:
        LaTeX string of the p-value.
    """
    return r"$<$0.001" if p_value < 1e-3 else rf"\num{{{p_value:.3f}}}"


def cached(path: Path, compute: Callable, use_cache: bool = True):
    """
    Evaluate an expensive stage, reusing a pickled result if available.

    Args:
        path: Location of the cache file.
        compute: Callable producing the result.
        use_cache: Whether an existing cache file may be reused.
    Returns:
        The (cached) result.
    """
    if use_cache and path.exists():
        print(f"Loading cache {path}")
        return pickle.load(open(path, "rb"))
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(result.cpu() if isinstance(result, Tensor) else result, open(path, "wb"))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-id", type=int, default=2069, help="Wandb ID of the evaluated model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to operate on.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed, each stage is seeded separately.")
    parser.add_argument("--num-robots", type=int, default=1000, help="Robots per DoF for the attribute correlations.")
    parser.add_argument("--num-pairs", type=int, default=50, help="Robots per DoF for the Mantel tests.")
    parser.add_argument("--num-poses", type=int, default=1000, help="Poses sampled per robot.")
    parser.add_argument("--num-bootstrap", type=int, default=9999, help="Bootstrap samples of the Mantel CI.")
    parser.add_argument("--num-permutations", type=int, default=9999, help="Permutations of the Mantel p-value.")
    parser.add_argument("--ik-chunk-size", type=int, default=50000, help="Poses solved per batched IK call.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Destination of the PDFs.")
    parser.add_argument("--plot-all", action="store_true", help="Plot every attribute, not only the strongest two.")
    parser.add_argument("--no-cache", action="store_true", help="Recompute the expensive stages.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_style()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = Model.from_id(args.model_id).to(device)

    # Individual attributes vs the principal components of the latent space.
    torch.manual_seed(args.seed)
    morphs = sample_padded_morphs(args.num_robots, device)
    latent = pca(model.encoder(morphs)[1][0][-1].detach())

    torch.manual_seed(args.seed + 1)
    reachability = cached(
        CACHE_DIR / f"reachability_{args.seed}_{args.num_robots}_{args.num_poses}.pkl",
        lambda: compute_reachability(morphs, args.num_poses),
        not args.no_cache,
    ).to(device)

    rows = []
    for attribute in morphological_attributes(morphs, args.num_robots, reachability):
        coefficients = correlation_profile(latent, attribute.values.float())
        component = int(torch.argmax(coefficients[:, 0].abs()).item())
        scc, ci_low, ci_high, p_value = coefficients[component].tolist()
        if scc < 0:
            # The sign of a principal component is arbitrary, hence report the magnitude.
            scc, ci_low, ci_high = -scc, -ci_high, -ci_low
        rows.append((attribute, component, scc, ci_low, ci_high, p_value))
        print(f"{attribute.slug:>18}: PC{component + 1:<4} SCC {scc * 100:+.1f}% "
              f"[{ci_low * 100:+.1f}%, {ci_high * 100:+.1f}%], p={p_value:.3g}")

    strongest = sorted(rows, key=lambda row: abs(row[2]), reverse=True)
    for attribute, component, *_ in (rows if args.plot_all else strongest[:2]):
        scatter(attribute.values.float(), latent[:, component], attribute.name,
                rf"PC {component + 1}", args.out_dir / f"exploration_{attribute.slug}.pdf")

    # Pairwise (dis)similarities vs the latent space.
    torch.manual_seed(args.seed + 2)
    morphs_pairs = sample_padded_morphs(args.num_pairs, device)
    latent_pairs = model.encoder(morphs_pairs)[1][0][-1].detach()
    latent_dist = torch.cdist(latent_pairs, latent_pairs)
    latent_normalised = torch.nn.functional.normalize(latent_pairs, dim=1)
    cosine_similarity = latent_normalised @ latent_normalised.T

    mdh_dist = mdh_distance_matrix(morphs_pairs)

    torch.manual_seed(args.seed + 3)
    similarity = cached(
        CACHE_DIR / f"workspace_similarity_{args.seed}_{args.num_pairs}_{args.num_poses}.pkl",
        lambda: workspace_similarity(compute_reach_matrix(
            morphs_pairs,
            torch.stack([sample_poses_in_reach(args.num_poses, m) for m in morphs_pairs]),
            args.ik_chunk_size,
        )),
        not args.no_cache,
    ).to(device)

    candidates = {
        "mdh": [
            Pairwise("mdh_distance", r"$d_\text{MDH}$ vs. $d_\text{latent}$",
                     r"$d_\text{latent}$", r"$d_\text{MDH}$", latent_dist, mdh_dist),
            Pairwise("mdh_distance", r"$d_\text{MDH}$ vs. $d^{\cos}_\text{latent}$",
                     r"$d^{\cos}_\text{latent}$", r"$d_\text{MDH}$", 1 - cosine_similarity, mdh_dist),
        ],
        "workspace": [
            Pairwise("workspace_similarity", r"$d_\text{workspace}$ vs. $d_\text{latent}$",
                     r"$d_\text{latent}$", r"$d_\text{workspace}$", latent_dist, 1 - similarity),
            Pairwise("workspace_similarity", r"$s_\text{workspace}$ vs. $s_\text{latent}$",
                     r"$s_\text{latent}$", r"$s_\text{workspace}$", cosine_similarity, similarity),
        ],
    }

    idx = torch.triu_indices(args.num_pairs * len(DOFS), args.num_pairs * len(DOFS), offset=1)
    mantel_rows = []
    for group, options in candidates.items():
        # Only the strongest latent (dis)similarity of each group is reported and plotted.
        best = max(options, key=lambda p: abs(spearmanr(p.latent[idx[0], idx[1]].cpu(),
                                                        p.other[idx[0], idx[1]].cpu()).statistic))
        r, ci_low, ci_high, p_value = mantel_test(best.latent, best.other,
                                                  args.num_bootstrap, args.num_permutations)
        mantel_rows.append((best, r, ci_low, ci_high, p_value))
        print(f"{group:>18}: SCC {r * 100:+.1f}% [{ci_low * 100:+.1f}%, {ci_high * 100:+.1f}%], p={p_value:.3g}")
        scatter(best.latent[idx[0], idx[1]], best.other[idx[0], idx[1]], best.latent_label, best.other_label,
                args.out_dir / f"exploration_{best.slug}.pdf")

    attribute_lines = "\n".join(
        rf"        {attribute.table_label} & {latex_mean_and_ci(scc * 100, ci_low * 100, ci_high * 100, decimals=0)}"
        rf" & {format_p_value(p_value)} \\"
        for attribute, _, scc, ci_low, ci_high, p_value in rows
    )
    mantel_lines = "\n".join(
        rf"        {pair.table_label} & {latex_mean_and_ci(r * 100, ci_low * 100, ci_high * 100, decimals=0)}"
        rf" & {format_p_value(p_value)} \\"
        for pair, r, ci_low, ci_high, p_value in mantel_rows
    )

    print(rf"""
\begin{{table}}
    \centering
    \caption{{Spearman correlation coefficients (SCC) between morphological
             attributes and the most correlated principal component of the
             latent space, and Mantel test correlations between pairwise
             similarity matrices.}}
    \label{{tab:exploration}}
    \begin{{tblr}}{{colspec={{l r r}}, row{{1}}={{font=\bfseries}}}}
        \toprule
        Attribute & SCC (\%) & $p$-value \\
        \midrule
{attribute_lines}
        \midrule
        \SetCell[c=3]{{l}}\textit{{Mantel test}} & & \\
{mantel_lines}
        \bottomrule
    \end{{tblr}}
\end{{table}}""")


if __name__ == "__main__":
    main()
