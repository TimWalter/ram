import pickle
from pathlib import Path

from ram.metrics import bootstrap_mean_ci
from paper_archive.utils import latex_mean_and_ci

import wandb
import torch

t_inference_time = pickle.load(open(Path(__file__).parent / "cache" / "runtime_transformer.pkl", "rb"))
inference_time = pickle.load(open(Path(__file__).parent.parent.parent / "RQ2" / "representation_accuracy"
                                  / "cache" / "runtime_ours.pkl", "rb"))

api = wandb.Api()
#RAM
test_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM"}
)
training_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM-Train"}
)

training_time_list = []
for run in training_runs:
    training_time_list += [run.summary.get("_runtime") / 3600]
bacc_list = []
for run in test_runs:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]

training_time = round(sum(training_time_list) / len(training_time_list))
bacc = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)

#Transformer
test_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "Transformer"}
)
training_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "Transformer-Train"}
)

training_time_list = []
for run in training_runs:
    training_time_list += [run.summary.get("_runtime") / 3600]
bacc_list = []
for run in test_runs:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]

t_training_time = round(sum(training_time_list) / len(training_time_list))
t_bacc = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)), decimals=0)


print(rf"""
\begin{{table}}[ht]
    \centering
    \begin{{talltblr}}[
        caption = {{Comparison of encoder types for RAM.}},
        label = {{tab:ram_transformer}},
    ]{{
        colspec = {{l r r r}},
        row{{1}} = {{font=\bfseries}}, 
    }}
        \toprule
        Classifier & {{Training  (h)}} & {{Inference (s)}} & \SetCell{{l}}{{Balanced (\%)\\ Accuracy }}  \\
        \midrule
        LSTM  & \textbf{{{training_time}}} & $\bm{{{round(inference_time)}\cdot10^ {{-9}}}}$ & {{\bfseries{bacc}}} \\
        Transformer  & {t_training_time} & ${round(t_inference_time)}\cdot10^{{-9}}$ & {t_bacc} \\
        \bottomrule
    \end{{talltblr}}
\end{{table}}
""")
