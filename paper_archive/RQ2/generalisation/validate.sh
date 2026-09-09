#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1

for dof in {1..9}
do
    .venv/bin/python ram/validate.py --test_set_path="generalisation/$dof" --group="Generalisation"
done
.venv/bin/python ram/validate.py --test_set_path="generalisation/spherical_wrist" --group="Generalisation"