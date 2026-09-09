import pickle
from pathlib import Path

import wandb
import torch

from ram.metrics import bootstrap_mean_ci
from paper_archive.utils import latex_mean_and_ci

ggik = pickle.load(open(Path(__file__).parent / "cache" / "test" / "results.pickle", "rb"))["All"]
g_tpr = ggik["True Positive Rate (Mean)"]
g_fnr = ggik["False Negative Rate (Mean)"]
g_fpr = ggik["False Positive Rate (Mean)"]
g_tnr = ggik["True Negative Rate (Mean)"]

api = wandb.Api()

test_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM"}
)

tpr = []
fnr = []
fpr = []
tnr = []
for run in test_runs:
    tpr += [run.summary.get("Validation/True Positive Rate (Mean)")]
    fnr += [run.summary.get("Validation/False Negative Rate (Mean)")]
    fpr += [run.summary.get("Validation/False Positive Rate (Mean)")]
    tnr += [run.summary.get("Validation/True Negative Rate (Mean)")]

tpr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(tpr).unsqueeze(1)))
fnr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(fnr).unsqueeze(1)))
fpr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(fpr).unsqueeze(1)))
tnr = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(tnr).unsqueeze(1)))

print(r"""
\begin{table}[ht]
    \centering
    \begin{talltblr}[
        caption = {Binary confusion matrix of RAM and GGIK.).},
        label = {tab:rq2_binary_confusion},
    ]{
        colspec = {l r r r r},
        row{1} = {font=\bfseries}, 
    }
        \toprule
        Classifier & {TPR (\%)} & {FNR (\%)} & {FPR (\%)} & {TNR (\%)}\\
        \midrule
""")
print(rf"""
        RAM  & {tpr} & {fnr} & {fpr} & {tnr} \\
        GGIK & {round(g_tpr)} & {round(g_fnr)} & {round(g_fpr)} & {round(g_tnr)} \\
""")
print(r"""
        \bottomrule
    \end{talltblr}
\end{table}
""")
