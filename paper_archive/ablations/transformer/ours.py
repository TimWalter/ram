import re
import time
import pickle
from pathlib import Path

import wandb
import torch

from paper_archive.ablations.transformer.model import Model
from paper_archive.ablations.transformer.validate import validate
from ram.dataset.loader import HomogeneousPoseSet

torch.set_float32_matmul_precision("high")

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda:1")
    batch_size = 100_000

    cache = Path(__file__).parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)


    api = wandb.Api()

    training_runs = api.runs(
        "tim-walter-tum/RAM",
        filters={"group": "Transformer-Train"}
    )
    model_ids = []
    for run in training_runs:
        model_id = int(re.findall(r'\d+', run.name)[-1])
        #validate(model_id, batch_size, None, "test", "Transformer")
        model_ids.append(model_id)

    # Runtime
    validation_set = HomogeneousPoseSet(batch_size, False, "test", device)
    model = Model.from_id(model_ids[-1]).to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    model.eval()
    # Warm-Up
    for batch_idx, (morph, pose, _, _) in enumerate(validation_set):
        logit = model.predict(morph, pose)
    runtime = []
    for batch_idx, (morph, pose, _, _) in enumerate(validation_set):
        start = time.perf_counter_ns()
        logit = model.predict(morph, pose)
        runtime += [time.perf_counter_ns() - start]

    runtime = sum(runtime) / len(runtime) / batch_size
    print(runtime)

    with open(cache / "runtime_transformer.pkl", "wb") as file:
        pickle.dump(runtime, file)