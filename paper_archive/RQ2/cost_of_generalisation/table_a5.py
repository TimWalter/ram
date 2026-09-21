import torch
import pickle
from pathlib import Path

from ram.metrics import bootstrap_mean_ci
from paper_archive.utils import latex_mean_and_ci

import wandb

api = wandb.Api()

# RAM
runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM"}
)
run = runs[0]

bacc = latex_mean_and_ci(run.summary.get("Validation/Balanced Accuracy (Mean)"),
                         run.summary.get("Validation/Balanced Accuracy (CI Lower)"),
                         run.summary.get("Validation/Balanced Accuracy (CI Upper)"), decimals=0)
tpr = latex_mean_and_ci(run.summary.get("Validation/True Positive Rate (Mean)"),
                        run.summary.get("Validation/True Positive Rate (CI Lower)"),
                        run.summary.get("Validation/True Positive Rate (CI Upper)"), decimals=0)
fnr = latex_mean_and_ci(run.summary.get("Validation/False Negative Rate (Mean)"),
                        run.summary.get("Validation/False Negative Rate (CI Lower)"),
                        run.summary.get("Validation/False Negative Rate (CI Upper)"), decimals=0)
fpr = latex_mean_and_ci(run.summary.get("Validation/False Positive Rate (Mean)"),
                        run.summary.get("Validation/False Positive Rate (CI Lower)"),
                        run.summary.get("Validation/False Positive Rate (CI Upper)"), decimals=0)
tnr = latex_mean_and_ci(run.summary.get("Validation/True Negative Rate (Mean)"),
                        run.summary.get("Validation/True Negative Rate (CI Lower)"),
                        run.summary.get("Validation/True Negative Rate (CI Upper)"), decimals=0)

# Reachability Map
(cell_distance, n_cells, runtime, balanced_accuracy, confusion_matrix, benchmarks) = pickle.load(
    open(Path(__file__).parent.parent.parent / "RQ1" / "cache" / "cache.pkl", "rb"))

rm_bacc = latex_mean_and_ci(*balanced_accuracy[2], decimals=0)

confusion_matrix = torch.stack(confusion_matrix["All"][2], dim=-1)
rm_tpr = latex_mean_and_ci(*confusion_matrix[0], decimals=0)
rm_fnr = latex_mean_and_ci(*confusion_matrix[1], decimals=0)
rm_fpr = latex_mean_and_ci(*confusion_matrix[2], decimals=0)
rm_tnr = latex_mean_and_ci(*confusion_matrix[3], decimals=0)

# Single-Morphology

single_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "Cost of Cross"}
)
bacc_list = []
tpr_list = []
fnr_list = []
fpr_list = []
tnr_list = []
for run in single_runs:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]
    tpr_list += [run.summary.get("Validation/True Positive Rate (Mean)")]
    fnr_list += [run.summary.get("Validation/False Negative Rate (Mean)")]
    fpr_list += [run.summary.get("Validation/False Positive Rate (Mean)")]
    tnr_list += [run.summary.get("Validation/True Negative Rate (Mean)")]

single_bacc = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)
single_tpr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(tpr_list).unsqueeze(1)), decimals=0)
single_fnr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(fnr_list).unsqueeze(1)), decimals=0)
single_fpr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(fpr_list).unsqueeze(1)), decimals=0)
single_tnr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(tnr_list).unsqueeze(1)), decimals=0)

print(rf"""
\begin{{table}}[ht]
\centering
    \begin{{talltblr}}[
        caption = {{Comparison of RAM to single-morphology baselines.}},
        label = {{tab:ram_cost_of_cross}},
    ]{{
        colspec = {{l r r r r r}},
        row{{1}} = {{font=\bfseries}}, 
    }}
        \toprule
        Classifier & {{Balanced (\%)\\Accuracy}}  & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)  \\
        \midrule
        RAM & {bacc}  & {tpr} & {fnr} & {fpr} & {tnr} \\
        SMR & {single_bacc}  & {single_tpr} & {single_fnr} & {single_fpr} & {single_tnr} \\
        Reachability Map & {rm_bacc}  & {rm_tpr} & {rm_fnr} & {rm_fpr} & {rm_tnr} \\
        \bottomrule
    \end{{talltblr}}
\end{{table}}
""")
