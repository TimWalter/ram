import torch

from paper_archive.utils import latex_mean_and_ci
from paper_archive.RQ3.morphology import experiment_plot, runtime_plot

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
    num_poses = torch.load(experiment_plot.CACHE / "experiment_inputs.pt")["task"].shape[1]

    rows = []
    for optimiser, optimiser_label in OPTIMISERS.items():
        for i, (method, method_label) in enumerate(METHODS.items()):
            metrics = experiment[method][optimiser]
            step_time = runtime[method][optimiser][-1, 0].item()

            reached = metrics["success_rate"][-1].clamp(0, 100)

            initial_error = metrics["pose_error"][0, 0]
            reduction = 100 * (1 - metrics["pose_error"][-1] / initial_error)
            reduction = reduction[[0, 2, 1]].clamp(max=100)
            collisions = (100 * metrics["self_collisions"][-1] / num_poses).clamp(0, 100)

            prefix = rf"\SetCell[r={len(METHODS)}]{{l}} {optimiser_label}" if i == 0 else ""
            rows.append(rf"{prefix:<24} & {method_label} & {step_time:.1f} & "
                        rf"{latex_mean_and_ci(*reached, decimals=0)} & {latex_mean_and_ci(*reduction, decimals=0)} & "
                        rf"{latex_mean_and_ci(*collisions, decimals=0)} \\")
        rows.append(r"    \midrule")
    rows.pop()

    print(r"""
\begin{table}[ht]
\centering
    \begin{talltblr}[
        caption = {Morphology optimisation with RAM compared to numerical inverse kinematics.},
label = {tab:rq3_morphology},
]{
    colspec = {l l r r r r},
row{1} = {font=\bfseries},
}
\toprule
Optimiser & Classifier & Runtime (s) & \SetCell{l}{Reached (\%)\\ Poses} & \SetCell{l}{Pose Error (\%)\\ Reduction} & \SetCell{l}{Self- (\%) \\ Collisions} \\
    \midrule""")
    print("\n".join(rows))
    print(r"""    \bottomrule
\end{talltblr}
\end{table}""")


if __name__ == "__main__":
    main()
