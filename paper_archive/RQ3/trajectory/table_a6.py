import torch

from paper_archive.utils import latex_mean_and_ci
from paper_archive.RQ3.trajectory import experiment_plot, runtime_plot

OPTIMISERS = {
    "GD": "Adam",
    "CMA": "CMA-ES",
}

METHODS = {
    "Ours": "RAM",
    "IK": "IK",
}


def main() -> None:
    runtime = runtime_plot.load()
    experiment = experiment_plot.load()
    num_samples = torch.load(experiment_plot.CACHE / "experiment_inputs.pt")["target_trajectory"].shape[1]

    reference_deviation = experiment["Ours"]["GD"]["deviation"][-1, 0]

    # One row per metric, one column per (optimiser, classifier).
    cells = {label: [] for label in [r"Runtime (s)", r"Reached Poses (\%)", r"Relative Mean Deviation",
                                     r"Pose Error Reduction (\%)", r"Self-Collisions (\%)"]}
    for optimiser in OPTIMISERS:
        for method in METHODS:
            metrics = experiment[method][optimiser]
            step_time = runtime[method][optimiser][-1, 0].item()

            # The bootstrap CI can leave the valid range, so it is clipped before formatting.
            reached = metrics["success_rate"][-1].clamp(0, 100)

            deviation = (metrics["deviation"][-1] / reference_deviation).clamp(min=0)

            initial_error = metrics["pose_error"][0, 0]
            reduction = 100 * (1 - metrics["pose_error"][-1] / initial_error)
            reduction = reduction[[0, 2, 1]].clamp(max=100)
            collisions = (100 * metrics["self_collisions"][-1] / num_samples).clamp(0, 100)

            for column, cell in zip(cells.values(), [
                f"{step_time:.2g}",
                latex_mean_and_ci(*reached, decimals=0),
                rf"{latex_mean_and_ci(*deviation, decimals=1)}$\times$",
                latex_mean_and_ci(*reduction, decimals=0),
                latex_mean_and_ci(*collisions, decimals=0),
            ]):
                column.append(cell)

    n_methods = len(METHODS)
    optimiser_header = " & ".join(rf"\SetCell[c={n_methods}]{{c}} {label}" + " &" * (n_methods - 1)
                                  for label in OPTIMISERS.values())
    optimiser_rules = " ".join(rf"\cmidrule[lr]{{{2 + i * n_methods}-{1 + (i + 1) * n_methods}}}"
                               for i in range(len(OPTIMISERS)))
    classifier_header = " & ".join(list(METHODS.values()) * len(OPTIMISERS))
    rows = [rf"{label:<26} & " + " & ".join(values) + r" \\" for label, values in cells.items()]

    print(rf"""
\begin{{table}}[ht]
\centering
    \begin{{talltblr}}[
        caption = {{Trajectory optimisation with RAM compared to numerical inverse kinematics.}},
label = {{tab:rq3_trajectory}},
]{{
    colspec = {{l {"r " * n_methods * len(OPTIMISERS)}}},
row{{1-2}} = {{font=\bfseries}},
}}
\toprule
Optimiser & {optimiser_header} \\
{optimiser_rules}
Classifier & {classifier_header} \\
    \midrule""")
    print("\n".join(rows))
    print(r"""    \bottomrule
\end{talltblr}
\end{table}""")


if __name__ == "__main__":
    main()
