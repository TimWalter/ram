import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tabulate import tabulate

CACHE = Path(__file__).parent / "cache"

N_ITER = 100

METHODS = {
    "Ours": (r"\text{RAM}", 0),
    "IK": (r"\text{IK}", 1),
}

OPTIMISERS = {
    "GD": (r"\text{Adam}", "-"),
    "CMA": (r"\text{CMA-ES}", "--"),
}

METRICS = {
    "success_rate": (r"$\text{Success Rate (\%)}$", 100.0, "{:.1f}"),
    "self_collisions": (r"$\text{\# Self-Collisions}$", 1.0, "{:.1f}"),
    "pose_error": (r"$\text{Mean Pose Error}$", 1.0, "{:.4f}"),
}


def load() -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Loads the bootstrapped (mean, lower, upper) metrics of experiment.py per evaluated iteration."""
    path = CACHE / "experiment.pkl"
    if not path.exists():
        raise SystemExit(f"No results at {path}, run experiment.py first.")

    experiment = pickle.load(open(path, "rb"))
    return {
        method: {
            optimiser: {
                metric: torch.as_tensor(experiment[method][optimiser][metric]).cpu() * scale
                for metric, (_, scale, _) in METRICS.items()
            }
            for optimiser in OPTIMISERS
        }
        for method in METHODS
    }


def main() -> None:
    experiment = load()

    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": shutil.which("latex") is not None,
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

    colors = sns.color_palette("colorblind", len(METHODS))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10 * len(METRICS), 5))

    table = []
    for method, (method_label, color) in METHODS.items():
        for optimiser, (optimiser_label, style) in OPTIMISERS.items():
            row = [f"{method} / {optimiser}"]
            for ax, (metric, (_, _, fmt)) in zip(axes, METRICS.items()):
                mean, lower, upper = experiment[method][optimiser][metric].unbind(dim=1)
                iteration = torch.linspace(0, N_ITER, mean.shape[0])

                ax.plot(iteration, mean, label=rf"${method_label}, {optimiser_label}$", color=colors[color],
                        linestyle=style)
                ax.fill_between(iteration, lower, upper, color=colors[color], alpha=0.2)

                row.append(f"{fmt.format(mean[0])} → {fmt.format(mean[-1])} "
                           f"({fmt.format(lower[-1])}, {fmt.format(upper[-1])})")
            table.append(row)

    for ax, (label, _, _) in zip(axes, METRICS.values()):
        ax.set_ylabel(label)
        ax.set_xlabel(r"$\text{Iteration}$")
        ax.set_xlim(0, N_ITER)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS) * len(OPTIMISERS),
               bbox_to_anchor=(0.52, 0.0))
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "morphology_experiment.pdf", bbox_inches="tight")
    fig.savefig(Path(__file__).parent / "morphology_experiment.png", bbox_inches="tight", dpi=40)

    print(tabulate(table, headers=["Method / Optimiser", *(f"{metric} initial → final (CI)" for metric in METRICS)],
                   tablefmt="github"))


if __name__ == "__main__":
    main()
