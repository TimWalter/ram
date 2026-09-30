from ram.metrics import bootstrap_mean_ci
from paper_archive.utils import latex_mean_and_ci

import wandb
import torch

api = wandb.Api()

runs_1 = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM-1"}
)
bacc_list = []
for run in runs_1:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]
bacc_1 = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)

runs_2 = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM-2"}
)
bacc_list = []
for run in runs_2:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]
bacc_2 = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)

runs_3 = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM"}
)
bacc_list = []
for run in runs_3:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]
bacc_3 = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)




print(rf"""
\begin{{table}}[ht]
\centering
    \begin{{talltblr}}[
        caption = {{RAMs trained on different training label fidelities.}},
        label = {{tab:ram_discretisation}},
    ]{{
        colspec = {{l r r}},
        row{{1}} = {{font=\bfseries}}, 
    }}
        \toprule
        Cell Spacing & \SetCell{{l}}{{Training Labels \\ Balanced Accuracy (\%)}} & \SetCell{{l}}{{RAM \\ Balanced Accuracy (\%)}}\\
        \midrule
        $\left[0.158, 0.163\right]$ & \num{{71(2:3)}} & {bacc_1} \\
        $\left[0.080, 0.083\right]$ & \num{{83(2:2)}} & {bacc_2} \\
        $\left[0.040, 0.042\right]$ & \num{{87(2:2)}} & {bacc_3} \\
        \bottomrule
    \end{{talltblr}}
\end{{table}}
""")
