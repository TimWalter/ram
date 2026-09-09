import pickle
from pathlib import Path

from paper_archive.utils import latex_mean_and_ci

import wandb

ggik = pickle.load(open(Path(__file__).parent / "cache" / "test" / "results.pickle", "rb"))["All"]
bacc = latex_mean_and_ci(ggik["Balanced Accuracy (Mean)"],
                         ggik["Balanced Accuracy (CI Lower)"],
                         ggik["Balanced Accuracy (CI Upper)"], decimals=0)
tpr = latex_mean_and_ci(ggik["True Positive Rate (Mean)"],
                        ggik["True Positive Rate (CI Lower)"],
                        ggik["True Positive Rate (CI Upper)"], decimals=0)
fnr = latex_mean_and_ci(ggik["False Negative Rate (Mean)"],
                        ggik["False Negative Rate (CI Lower)"],
                        ggik["False Negative Rate (CI Upper)"], decimals=0)
fpr = latex_mean_and_ci(ggik["False Positive Rate (Mean)"],
                        ggik["False Positive Rate (CI Lower)"],
                        ggik["False Positive Rate (CI Upper)"], decimals=0)
tnr = latex_mean_and_ci(ggik["True Negative Rate (Mean)"],
                        ggik["True Negative Rate (CI Lower)"],
                        ggik["True Negative Rate (CI Upper)"], decimals=0)

api = wandb.Api()
runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "Boundary"}
)
run = runs[0]


ram_bacc = latex_mean_and_ci(run.summary.get("Validation/Balanced Accuracy (Mean)"),
                         run.summary.get("Validation/Balanced Accuracy (CI Lower)"),
                         run.summary.get("Validation/Balanced Accuracy (CI Upper)"), decimals=0)
ram_tpr = latex_mean_and_ci(run.summary.get("Validation/True Positive Rate (Mean)"),
                         run.summary.get("Validation/True Positive Rate (CI Lower)"),
                         run.summary.get("Validation/True Positive Rate (CI Upper)"), decimals=0)
ram_fnr = latex_mean_and_ci(run.summary.get("Validation/False Negative Rate (Mean)"),
                         run.summary.get("Validation/False Negative Rate (CI Lower)"),
                         run.summary.get("Validation/False Negative Rate (CI Upper)"), decimals=0)
ram_fpr = latex_mean_and_ci(run.summary.get("Validation/False Positive Rate (Mean)"),
                         run.summary.get("Validation/False Positive Rate (CI Lower)"),
                         run.summary.get("Validation/False Positive Rate (CI Upper)"), decimals=0)
ram_tnr = latex_mean_and_ci(run.summary.get("Validation/True Negative Rate (Mean)"),
                         run.summary.get("Validation/True Negative Rate (CI Lower)"),
                         run.summary.get("Validation/True Negative Rate (CI Upper)"), decimals=0)



print(rf"""
\begin{{table}}[ht]
\centering
    \begin{{talltblr}}[
        caption = {{Performance of RAM and GGIK on boundary poses.}},
        label = {{tab:ram_boundary}},
    ]{{
        colspec = {{l r r r r r}},
        row{{1}} = {{font=\bfseries}}, 
    }}
        \toprule
        Classifier & Balanced Accuracy (\%)  & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)  \\
        \midrule
        RAM & {ram_bacc}  & {ram_tpr} & {ram_fnr} & {ram_fpr} & {ram_tnr} \\
        GGIK & {bacc}  & {tpr} & {fnr} & {fpr} & {tnr} \\
        \bottomrule
    \end{{talltblr}}
\end{{table}}
""")





