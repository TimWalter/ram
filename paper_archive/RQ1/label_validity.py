import pickle
from pathlib import Path

import torch
from tqdm import tqdm

import ram.dataset.r3 as r3
import ram.dataset.so3 as so3
import ram.dataset.se3 as se3

from ram.dataset.morphology import sample_morph
from ram.dataset.workspace import fk_approximation, synthesise_data
from ram.metrics import binary_confusion_matrix, bootstrap_mean_ci, counts_to_rates
from paper_archive.utils import latex_mean_and_ci

torch.manual_seed(0)
device = torch.device("cuda")

cache = Path(__file__).parent / "cache"
cache.mkdir(parents=True, exist_ok=True)

# Config
levels = [1, 2, 3, 4]
intervals = [1, 1, 10, 30]
num_robots = 50
num_samples = 100_000

# Data Generation
morphs = {}
labels = {}
cell_indices = {}
for dof in [5, 6, 7]:
    morphs[dof] = sample_morph(num_robots, dof, False, device)
    labels[dof] = []
    cell_indices[dof] = {1: [], 2: [], 3: [], 4: []}
    for m in morphs[dof]:
        p, l = synthesise_data(m, num_samples, True, True)
        labels[dof] += [l]
        for level in levels:
            se3.set_level(level)
            c = se3.index(p)
            cell_indices[dof][level] += [c]

if not any(cache.iterdir()):
    cell_distance = []
    benchmarks = []
    # Table 1
    n_cells = []
    runtime = []
    balanced_accuracy = []
    # Table A1
    confusion_matrix = {"All": [], 7: [], 6: [], 5: []}

    for level, interval in zip(levels, intervals):
        se3.set_level(level)
        print(f"[LEVEL{se3.LEVEL}]")

        cell_distance += [[se3.MIN_DISTANCE_BETWEEN_CELLS, se3.MAX_DISTANCE_BETWEEN_CELLS]]
        n_cells += [so3.N_CELLS * (torch.linalg.norm(r3.cell(torch.arange(0, r3.N_CELLS)), dim=1) < 1.0).sum()]

        level_benchmarks = []
        # Table 1
        level_runtime = []
        level_balanced_accuracy = []

        # Table A1
        level_confusion_matrix = {"All": [], 7: [], 6: [], 5: []}

        for dof in [5, 6, 7]:
            print(f"[DOF{dof}]")
            batch_size = None

            for (m, l, c) in tqdm(zip(morphs[dof],
                                      labels[dof], cell_indices[dof][level],
                                      ), desc="Looping Morphologies"):
                morph_runtime = 0.0
                morph_benchmarks = []

                hits = torch.zeros_like(l)
                while True:
                    new_r_indices, benchmark, batch_size = fk_approximation(m, True,
                                                                            seconds=interval,
                                                                            batch_size=batch_size)
                    hits |= torch.isin(c, new_r_indices)
                    morph_benchmarks.append(torch.tensor(benchmark))
                    morph_runtime += interval

                    tpr, fnr, fpr, tnr = counts_to_rates(
                        *binary_confusion_matrix(hits.float(), l).flatten(start_dim=1).unbind(dim=-1))

                    if tpr > 95.0 or morph_runtime > 600:
                        break

                # Aggregate
                morph_benchmarks = torch.stack(morph_benchmarks)
                aggregated_morph_benchmark = morph_benchmarks.mean(dim=0)
                aggregated_morph_benchmark[0] *= morph_benchmarks.shape[0]
                level_benchmarks += [aggregated_morph_benchmark]

                level_runtime += [morph_runtime]

                level_balanced_accuracy += [(tpr + tnr) / 2]
                level_confusion_matrix["All"] += [[tpr, fnr, fpr, tnr]]
                level_confusion_matrix[dof] += [[tpr, fnr, fpr, tnr]]

        # Aggregate (As this is final compute CIs
        benchmarks += [bootstrap_mean_ci(torch.stack(level_benchmarks))]

        runtime += [bootstrap_mean_ci(torch.tensor(level_runtime).unsqueeze(1))]
        balanced_accuracy += [bootstrap_mean_ci(torch.tensor(level_balanced_accuracy).unsqueeze(1))]

        for key in level_confusion_matrix.keys():
            confusion_matrix[key] += [bootstrap_mean_ci(torch.tensor(level_confusion_matrix[key]))]

    with open(cache / "cache.pkl", "wb") as file:
        pickle.dump([cell_distance, n_cells, runtime, balanced_accuracy, confusion_matrix, benchmarks], file)
else:
    (cell_distance, n_cells, runtime, balanced_accuracy, confusion_matrix, benchmarks) = pickle.load(
        open(cache / "cache.pkl", "rb"))

# Table 1
print(r"""
\begin{tblr}{
            colspec = {l r r r},
            row{1} = {font=\bfseries}, 
        }
        \toprule
        Cell Spacing & \# Cells & Runtime (s) & Balanced Accuracy (\%) \\
        \midrule""")
for i, level in enumerate(levels):
    current_runtime = ""
    if runtime[i][0] == 1.0:
        current_runtime = r"$\leq 1$"
    else:
        current_runtime = latex_mean_and_ci(*runtime[i])
    random = latex_mean_and_ci(*balanced_accuracy[i])
    print(
        rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$ & ${int(n_cells[i]):,}$ & {current_runtime} & {random} \\\addlinespace")
print(r"""\bottomrule
    \end{tblr}""")

# Table A1
print(r"""
    \begin{tblr}{
                colspec = {l r r r r r},
                row{1} = {font=\bfseries},
            }
            \toprule
            Cell Spacing & DoF & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)\\
            \midrule""")
for i, l in enumerate(levels):
    print(rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$")
    for dof in ("All", 5, 6, 7):
        row = rf"& {dof} \\"
        for cm in [confusion_matrix]:
            cm_inner = torch.stack(cm[dof][i], dim=-1)
            tpr = latex_mean_and_ci(*cm_inner[0])
            fnr = latex_mean_and_ci(*cm_inner[1])
            fpr = latex_mean_and_ci(*cm_inner[2])
            tnr = latex_mean_and_ci(*cm_inner[3])
            row += rf"& {tpr} & {fnr} & {fpr} & {tnr}"
        row += r"\\"
        print(row)
print(r"""\bottomrule
    \end{tblr}""")

# Benchmark summary (debug-only)
metric_names = ["total_samples",
                "total_efficiency (%)", "unique_efficiency (%)",
                "collision_efficiency (%)"]
for i, level in enumerate(levels):
    print(f"[LEVEL {level}]")
    mean, lower, upper = benchmarks[i]
    for name, m, lo, up in zip(metric_names, mean, lower, upper):
        print(f"  {name:<26} {latex_mean_and_ci(m, lo, up)}")
