import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch

from paper_archive.RQ3.trajectory import experiment_plot, runtime_plot
from paper_archive.RQ3.trajectory.experiment_plot import METHODS, OPTIMISERS


def main() -> None:
    runtime = runtime_plot.load()
    experiment = experiment_plot.load()
    num_samples = torch.load(experiment_plot.CACHE / "experiment_inputs.pt")["target_trajectory"].shape[1]
    for method in METHODS:
        for optimiser in OPTIMISERS:
            experiment[method][optimiser]["self_collisions"] *= 100 / num_samples

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
    metrics = {
        "success_rate": r"$\text{Reached Poses}$",
        "deviation": r"$\text{Mean Deviation}$",
        "pose_error": r"$\text{Mean Pose Error}$",
        "self_collisions": r"$\text{Self-Collisions}$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(30, 10))
    axes = axes.flatten()

    for method, (method_label, color) in METHODS.items():
        for optimiser, (optimiser_label, style) in OPTIMISERS.items():
            label = rf"${method_label}, {optimiser_label}$"

            mean, lower, upper = runtime[method][optimiser].unbind(dim=1)
            axes[0].plot(runtime_plot.SIZES, mean, label=label, color=colors[color], linestyle=style)
            axes[0].fill_between(runtime_plot.SIZES, lower, upper, color=colors[color], alpha=0.2)

            for ax, metric in zip(axes[1:1 + len(metrics)], metrics):
                mean, lower, upper = experiment[method][optimiser][metric].unbind(dim=1)
                iteration = torch.linspace(0, experiment_plot.N_ITER, mean.shape[0])
                ax.plot(iteration, mean, label=label, color=colors[color], linestyle=style)
                ax.fill_between(iteration, lower, upper, color=colors[color], alpha=0.2)

    axes[0].set_ylabel(r"$\text{Runtime (s)}$")
    axes[0].set_xlabel(r"$\text{\# Waypoints}$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
    axes[0].yaxis.set_minor_formatter(ticker.LogFormatterExponent())

    for ax, (metric, label) in zip(axes[1:1 + len(metrics)], metrics.items()):
        ax.set_ylabel(label)
        ax.set_xlabel(r"$\text{Iteration}$")
        ax.set_xlim(0, experiment_plot.N_ITER)
        if metric == "success_rate":
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
        else:
            ax.set_ylim(bottom=0)

    for ax in axes[:1 + len(metrics)]:
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "trajectory.pdf", bbox_inches="tight")
    fig.savefig(Path(__file__).parent / "trajectory.png", bbox_inches="tight", dpi=40)


if __name__ == "__main__":
    main()
