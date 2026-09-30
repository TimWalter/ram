import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from tabulate import tabulate

CACHE = Path(__file__).parent / "cache"

SIZES = torch.logspace(0, 4, 10).int()
N_ITER = 100

METHODS = {
    "Ours": (r"\text{RAM}", 0),
    "IK": (r"\text{IK}", 1),
}

OPTIMISERS = {
    "GD": (r"\text{Adam}", "-"),
    "CMA": (r"\text{CMA-ES}", "--"),
}


def load() -> dict[str, dict[str, torch.Tensor]]:
    """Loads the nanosecond timings of runtime.py as seconds per optimisation step."""
    path = CACHE / "runtime.pkl"
    if not path.exists():
        raise SystemExit(f"No timings at {path}, run runtime.py first.")

    runtime = pickle.load(open(path, "rb"))
    return {
        method: {
            optimiser: torch.as_tensor(runtime[method][optimiser]).cpu() / 1e9 / N_ITER
            for optimiser in OPTIMISERS
        }
        for method in METHODS
    }


def main() -> None:
    runtime = load()

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
    fig, ax = plt.subplots(figsize=(10, 5))

    table = []
    reference = runtime["Ours"]["GD"][:, 0]
    for method, (method_label, color) in METHODS.items():
        for optimiser, (optimiser_label, style) in OPTIMISERS.items():
            mean, lower, upper = runtime[method][optimiser].unbind(dim=1)
            if mean.shape[0] != SIZES.shape[0]:
                raise SystemExit(f"{method}/{optimiser} has {mean.shape[0]} sizes, SIZES has {SIZES.shape[0]}.")

            ax.plot(SIZES, mean, label=rf"${method_label}, {optimiser_label}$", color=colors[color], linestyle=style)
            ax.fill_between(SIZES, lower, upper, color=colors[color], alpha=0.2)

            table.append([
                f"{method} / {optimiser}",
                f"{mean[0]:.3g} ({lower[0]:.3g}, {upper[0]:.3g})",
                f"{mean[-1]:.3g} ({lower[-1]:.3g}, {upper[-1]:.3g})",
                f"{mean[-1] / reference[-1]:.1f}x",
            ])

    ax.set_ylabel(r"$\text{Runtime (s)}$")
    ax.set_xlabel(r"$\text{\# Waypoints}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_minor_formatter(ticker.LogFormatterExponent())
    ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.52, 0.0))
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "trajectory_runtime.pdf", bbox_inches="tight")
    fig.savefig(Path(__file__).parent / "trajectory_runtime.png", bbox_inches="tight", dpi=40)

    print(tabulate(table, headers=["Method / Optimiser", f"{SIZES[0]} Poses (s/step)",
                                   f"{SIZES[-1]} Poses (s/step)", "vs Ours / GD"], tablefmt="github"))


if __name__ == "__main__":
    main()
