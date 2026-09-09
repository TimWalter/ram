#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1
mkdir -p data/generalisation

for dof in {1..9}
do
    echo "Creating data for $dof"
    .venv/bin/python ram/dataset/generate_evaluation_set.py --dof="$dof" --set="$dof" --num_robots=100
    mv data/"$dof" data/generalisation/
done

for dof in {5..7}
do
    echo "Creating spherical wrist data for $dof"
    .venv/bin/python paper_archive/RQ2/generalisation/generate_evaluation_set.py --dof="$dof" --set="spherical_wrist" --num_robots=33
done
mv data/spherical_wrist data/generalisation/


