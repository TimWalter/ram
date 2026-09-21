#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1
mkdir -p data/cost_of_cross

seed=0

for dof in {5..7}
do
  for i in {1..100}
  do
    echo "Creating data for $dof [$i/100] (Seed $seed)"
    .venv/bin/python  paper_archive/RQ2/cost_of_cross/generate_train_set.py --set="train_${dof}_${i}" --seed="$seed"
    mv data/"train_${dof}_${i}" data/cost_of_cross/"train_${dof}_${i}"
    .venv/bin/python paper_archive/RQ2/cost_of_cross/generate_evaluation_set.py --set="test_${dof}_${i}" --seed="$seed"
    mv data/"test_${dof}_${i}" data/cost_of_cross/"test_${dof}_${i}"

    ((seed++))
  done
done