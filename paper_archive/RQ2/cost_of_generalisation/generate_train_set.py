import re
import argparse
from pathlib import Path

import torch
import zarr
import fasteners
from tqdm import tqdm

from ram.dataset.workspace import synthesise_data
from ram.dataset.loader import HomogeneousPoseSet

CHUNK_SIZE = 100_000  # ~2.4MB
SHARD_SIZE = CHUNK_SIZE * 1000  # ~2.4GB

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="train", help="Where to store")
parser.add_argument("--num_samples", type=int, default=1_000_000, help="Number of samples per robot")
parser.add_argument("--seed", type=int, default=0, help="Seed, if -1 dont seed")
args = parser.parse_args()

assert args.num_samples % CHUNK_SIZE == 0, \
    f"Only full chunks are supported (chunk size {CHUNK_SIZE})"
assert SHARD_SIZE / args.num_samples == SHARD_SIZE // args.num_samples, \
    f"One robot must belong to one shard (shard size {SHARD_SIZE})"

SAFE_FOLDER = Path(__file__).parent.parent.parent.parent / "data" / args.set
lock = fasteners.InterProcessLock(SAFE_FOLDER.parent / "train_lock.file")
compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

with lock:
    SAFE_FOLDER.mkdir(parents=True, exist_ok=True)
root = zarr.open(SAFE_FOLDER, mode="a")

test_set = HomogeneousPoseSet(1000, False, "test", torch.device("cuda:1"))
morphs = test_set.morphologies[args.seed:args.seed + 1]
dof = ((morphs != 0).any(dim=-1).sum(dim=-1)[0] - 1).item()
morphs = morphs[:, :dof + 1, :]

with lock:
    file_indices = [
        int(match.group(1))
        for k in root.array_keys()
        if (match := re.search(r'^(\d+)_samples$', k))
    ]
    file_idx = (max(file_indices) + 1) if file_indices else 0
    morph_offset = sum([root[f"{idx}_morphologies"].shape[0] for idx in file_indices])

    morph_filename = str(file_idx) + "_morphologies"
    sample_filename = str(file_idx) + "_samples"

    root.create_array(
        morph_filename,
        shape=(1, dof + 1, 3),
        dtype="float32",
        chunks=(1, dof + 1, 3),
        compressors=compressor,
        overwrite=False,
    )
    root.create_array(sample_filename,
                      shape=(args.num_samples, 3),
                      dtype="int64",
                      chunks=(CHUNK_SIZE, 3),
                      shards=(SHARD_SIZE, 3),
                      compressors=compressor)

root[morph_filename][0:] = morphs.cpu().numpy()

file = root[sample_filename]
file_offset = 0
buffer = torch.zeros(min(1 * args.num_samples, SHARD_SIZE), 3, dtype=torch.int64)
buffer_offset = 0

desc = f"Train set file {file_idx} with morph offset {morph_offset} for {dof} DoF robots"
for idx, morph in enumerate(tqdm(morphs, desc=desc)):
    cell_indices, labels = synthesise_data(morph, args.num_samples, seconds=30, use_ik=False)

    poses = cell_indices.unsqueeze(1)
    labels = labels.long().unsqueeze(1)
    morph_ids = torch.full_like(labels, idx + morph_offset)

    samples = torch.cat([morph_ids, poses, labels], dim=1)

    buffer[buffer_offset: buffer_offset + samples.shape[0]] = samples
    buffer_offset += samples.shape[0]

    if buffer_offset == buffer.shape[0] or idx == morphs.shape[0] - 1:
        active_data = buffer[:buffer_offset]
        active_data = active_data[torch.randperm(active_data.shape[0])]

        file[file_offset: file_offset + active_data.shape[0]] = active_data.cpu().numpy()
        file_offset += active_data.shape[0]

        buffer_offset = 0
