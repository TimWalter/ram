#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1

model_id=4670
for dof in {5..7}
do
  for i in {1..100}
  do
    if ! ls data/trained_models/${model_id}-*/checkpoint.pth >/dev/null 2>&1; then
      echo "Model ID $model_id has no checkpoint, aborting"
      exit 1
    fi
    echo "Validating RAM for $dof [$i/100] with Model ID: $model_id"
    .venv/bin/python ram/validate.py --model_id="$model_id" --test_set_path="cost_of_cross/test_${dof}_${i}" --group="Cost of Cross"
    ((model_id++))
  done
done
