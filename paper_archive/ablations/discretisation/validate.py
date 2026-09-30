import re

import wandb

from ram.validate import validate

api = wandb.Api()

runs_1 = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM-Train Level1"}
)
model_id1 = [int(re.findall(r'\d+', run.name)[-1])  for run in runs_1]

for model_id in model_id1:
    validate(model_id, 1000, val_set_path=None, test_set_path="test", group="RAM-1")

runs_2 = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM-Train Level2"}
)

model_id2 = [int(re.findall(r'\d+', run.name)[-1]) for run in runs_2]

for model_id in model_id2:
    validate(model_id, 1000, val_set_path=None, test_set_path="test", group="RAM-2")
