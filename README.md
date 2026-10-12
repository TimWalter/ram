<div align="center">

# RAM: Reachability Across Morphologies

**Tim Walter**<sup>1</sup>, **Xinyu Chen**<sup>2</sup>, **Jonathan Külz**<sup>1</sup>, **Matthias Althoff**<sup>1</sup>

<sup>1</sup>Technical University of Munich, <sup>2</sup>German Electron Synchrotron (DESY) & University of Hamburg

**CoRL 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://timwalter.github.io/ram/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09108-b31b1b)](https://arxiv.org/abs/2606.09108)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-TimWalter%2FRAM-yellow)](https://huggingface.co/TimWalter/RAM)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-TimWalter%2FRAM-yellow)](https://huggingface.co/datasets/TimWalter/RAM)
[![Dataset](https://img.shields.io/badge/Dataset-Globus-green)](https://app.globus.org/file-manager?origin_id=fcf7ca2e-cd91-45fb-8252-c53fe972e4d2&origin_path=%2Fram%2F)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://www.youtube.com/watch?v=z9-zx4CzY8w)

</div>

**TL;DR:** RAM is a fast, differentiable surrogate for pose reachability that generalises across robot morphologies, enabling rapid design and planning workflows.

![RAM overview](https://raw.githubusercontent.com/TimWalter/ram/gh-pages/static/images/HeroFigure.svg)

## Abstract

Many stages of the robotic lifecycle, from morphology synthesis to operation, rely fundamentally on the reachable workspace. However, current methods for approximating workspaces are slow, imprecise, or tied to a single morphology. We introduce Reachability Across Morphologies (RAM): a morphology-conditioned, implicit neural representation that acts as a fast, differentiable surrogate for pose reachability, generalising to unseen morphologies while inherently accounting for self-collisions. RAM is trained on a large-scale dataset of 3·10<sup>10</sup> samples generated solely from forward kinematics, which we publish alongside the model. Experiments show that RAM reaches 84% balanced accuracy at nanosecond inference, 15 percentage points higher than a generative inverse kinematics baseline that also generalises across morphologies, while running three orders of magnitude faster. This speed carries over to downstream tasks, accelerating morphology and trajectory optimisation by one to two orders of magnitude.

## Installation

The project uses [uv](https://docs.astral.sh/uv/), requires Python ≥ 3.13, and a CUDA-capable GPU.

```bash
git clone https://github.com/TimWalter/ram.git
cd ram
uv sync
```

All commands below are run from the repository root with:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export SCIPY_ARRAY_API=1
```

## Quick start

A pretrained model is available on the [Hugging Face Hub](https://huggingface.co/TimWalter/RAM):

```python
import torch
from ram.model import Model
from ram.dataset.morphology import sample_morph
from ram.dataset.workspace import sample_poses_in_reach
import ram.dataset.se3 as se3

model = Model.from_pretrained("TimWalter/RAM").eval()

# A random 6-DoF morphology given as modified Denavit-Hartenberg parameters (alpha, a, d): shape (dof + 1, 3)
morph = sample_morph(num_robots=1, dof=6, analytically_solvable=False)[0]

# Poses as 4x4 homogeneous transforms, encoded as 9D vectors (position + first two rotation-matrix columns)
poses = se3.to_vector(sample_poses_in_reach(1000, morph))

logits = model.predict(morph.expand(poses.shape[0], -1, -1), poses)
reachable = torch.sigmoid(logits) > 0.5
```

`Model.forward` is fully differentiable with respect to both the morphology and the pose, so RAM can be dropped into gradient-based morphology or trajectory optimisation in place of inverse kinematics.

## Dataset

The dataset contains 3·10<sup>10</sup> samples across 3·10<sup>4</sup> serial morphologies with 5–7 revolute joints (202 GB). It is stored as Zarr arrays and split into `train`, `val`, `test`, `val_boundary` and `test_boundary`. All dataset paths in the scripts are relative to `data/`.

**Hugging Face.** Download from [TimWalter/RAM](https://huggingface.co/datasets/TimWalter/RAM) into `data/`, which places the splits under `data/RAM/`:

```bash
# Full dataset
hf download TimWalter/RAM --repo-type dataset --local-dir data

# Only selected splits, e.g. the evaluation sets (train is ~216 GB, every other split ~1 GB)
hf download TimWalter/RAM --repo-type dataset --local-dir data --include "RAM/val/*" "RAM/test/*"
```

The splits are then addressed as `RAM/train`, `RAM/val`, etc. (e.g. `--training_set_path RAM/train`).

**Globus.** Alternatively, download via [Globus](https://app.globus.org/file-manager?origin_id=fcf7ca2e-cd91-45fb-8252-c53fe972e4d2&origin_path=%2Fram%2F) and place the sets under `data/` (e.g. `data/train`, `data/val`).

To generate data yourself:

```bash
# Training set: discretised cell indices labelled via forward kinematics
python ram/dataset/generate_train_set.py --set train --dof 6 --num_robots 100 --num_samples 1000000

# Validation / test set: continuous poses labelled via inverse kinematics
python ram/dataset/generate_evaluation_set.py --set val --dof 6 --num_robots 100 --num_samples 100000
```

Both scripts append to an existing set, so they can be run repeatedly (and in parallel) for different DoFs.

## Training and evaluation

```bash
# Train (logs to Weights & Biases)
python ram/train.py --training_set_path train --validation_set_path val

# Hyperparameter search with Optuna
python ram/hyperparameter.py --training_set_path train --validation_set_path val

# Evaluate a trained model on a test set, optionally calibrating the threshold on a validation set
python ram/validate.py --model_id <id> --val_set_path val --test_set_path test
```

Trained models are stored in `data/trained_models/` and can be loaded with `Model.from_id(<id>)`.

## Reproducing the paper

All experiments of the paper are in `paper_archive/` on the [`CoRL2026`](https://github.com/TimWalter/ram/tree/CoRL2026) branch, organised by research question.                   # Video assets

The GGIK baseline is included as a git submodule; clone with `git clone --recurse-submodules -b CoRL2026 https://github.com/TimWalter/ram.git`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
RAM,
title={{RAM}: Reachability Across Morphologies},
author={Walter, Tim and Chen, Xinyu and K{\"u}lz, Jonathan and Althoff, Matthias},
booktitle={10th Annual Conference on Robot Learning},
year={2026},
url={https://arxiv.org/abs/2606.09108}
}
```
