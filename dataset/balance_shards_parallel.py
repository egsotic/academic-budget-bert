import argparse
from functools import partial
import glob
import os
import time
from typing import Dict, List
import h5py
import numpy as np
import tqdm
import multiprocessing as mp


def read_file_length(path):
    with h5py.File(path, "r") as f:
        r = (np.asarray(f[list(f.keys())[0]][:]).shape[0], path)
    return r

def find_num_samples_in_shard_parallel(files):
    lengths = []
    print("Finding optimal length of each shard")
    
    with mp.Pool() as pool:
        for r in pool.imap_unordered(read_file_length, tqdm.tqdm(files)):
            lengths.append(r)
    lengths = sorted(lengths, key=lambda x: x[0], reverse=True)

    return lengths


def batch_iter(files_indices, size: int):
    current_batch = []
    current_size = 0
    
    for (f, (start, end)) in tqdm.tqdm(files_indices, desc='batch'):
        current_batch.append([f, (start, end)])
        current_size += end - start
        
        if current_size >= size:
            yield current_batch
            
            current_batch = []
            current_size = 0
    
    if current_batch:
        yield current_batch


def read_dataset_from_file(path):
    with h5py.File(path, "r") as f:
        data = {
            k: np.asarray(v[:])
            for k, v in f.items()
        }

    return data

def write_dataset_to_file(data: List[Dict[str, np.array]], out_path):
    assert all(data[0].keys() == d.keys() for d in data[1:])
    
    with h5py.File(out_path, "w", libver="latest") as f:
        for k in data[0].keys():
            key_data = np.vstack([d[k] for d in data])
            dtype = data[0][k].dtype
            f.create_dataset(k, data=key_data, dtype=dtype, compression="gzip")

def slice_dataset(dataset: Dict[str, np.array], start: int, end: int):
    return {
        k: v[start:end]
        for k, v in dataset.items()
    }


def balance_worker(batch_files_indices, block_size, out_dir, prefix):
    current_data = []
    current_size = 0
    current_files_indices_splits = []
    
    for f, indices in split_files_indices(batch_files_indices, block_size):
        f_data = read_dataset_from_file(f)

        # aggregate to blocks
        offset = indices[0][0]
        for start, end in indices:
            current_data.append(slice_dataset(f_data, start - offset, end - offset))
            current_size += end - start
            current_files_indices_splits += [(f, (start, end))]
            
            if current_size == block_size:
                out_path = os.path.join(out_dir, f"{prefix}{time.time()}.hdf5")
                
                write_dataset_to_file(current_data, out_path)
                
                current_data = []
                current_size = 0
                current_files_indices_splits = []

    return current_files_indices_splits

def balance_parallel(files, out_dir, prefix):
    sizes = find_num_samples_in_shard_parallel(files)
    if len(sizes) == 0:
        return
    
    all_files_indices = [(f, (0, size)) for size, f in sizes if size > 0]
    # TODO: can be determined by script and not by amount of files
    block_size = sum([x[0] for x in sizes]) // len(sizes)
    
    print('total files', len(sizes))
    print('total size', sum([x[0] for x in sizes]))
    print('block size', block_size)
    
    worker = partial(balance_worker, block_size=block_size, out_dir=out_dir, prefix=prefix)
    
    remaining_files_indices = all_files_indices
    remaining_size = sum(end - start for _, (start, end) in remaining_files_indices)
    
    while remaining_size >= block_size:
        # by size ascending order (packing as many small files togethe as possible)
        remaining_files_indices.sort(key=lambda t: t[1][1] - t[1][0])
        
        with mp.Pool() as pool:
            remaining_files_indices = sum(
                pool.map(
                    worker,
                    batch_iter(remaining_files_indices, block_size),
                    ),
                start=[])
        
        remaining_size = sum(end - start for _, (start, end) in remaining_files_indices)

    print('deleted samples', remaining_size)

def split_files_indices(files_indices, block_size: int):
    current_size = 0

    # by length ascending order (to keep remainder with minimum files)
    for f, (start, end) in sorted(files_indices, key=lambda t: t[1][1] - t[1][0]):
        current_block = []
        
        l = end - start
        # add complete block
        if current_size + l < block_size:
            current_block = [(start, end)]
            current_size += l
        # split
        else:
            split_size = block_size - current_size
            # fill up previous block
            current_block.append(((start, start + split_size)))
            # full blocks + possible remainder
            current_block.extend([((i, min(i + block_size, end))) for i in range(start + split_size, end, block_size)])
            last_block_size = current_block[-1][1] - current_block[-1][0]
            current_size = last_block_size % block_size
        
        yield (f, [indices for indices in current_block])

def main(files, out_dir):
    train_files = [file for file in files if "train_shard" in file]
    test_files = [file for file in files if "test_shard" in file]
    balance_parallel(train_files, out_dir, "train_shard_")
    balance_parallel(test_files, out_dir, "test_shard_")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.dir, "**/*.hdf5"), recursive=True)
    
        
    main(files, args.out_dir)
