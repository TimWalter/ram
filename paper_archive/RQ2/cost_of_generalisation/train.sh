#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1

for dof in {5..7}
do
  for i in {1..100}
  do
    echo "Training RAM for $dof [$i/100]"
    .venv/bin/python ram/train.py --training_set_path="cost_of_cross/train_${dof}_${i}" --group="Cost of Cross - Train" --validation_set_path=""
  done
done

