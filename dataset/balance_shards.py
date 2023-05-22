import argparse
import glob
import os
import h5py
import numpy as np
import time
from tqdm import tqdm
from collections import deque


def get_partial_numpy(lst_of_npy, num_rows):
    result = []
    current_rows = 0
    prev_shape = sum([x.shape[0] for x in lst_of_npy])
    n = len(lst_of_npy)
    # we parse the list in reverse order, since we want the most recently inserted array first
    for i, arr in enumerate(reversed(lst_of_npy)):
        if current_rows < num_rows and arr.shape[0] > 0:
            if arr.shape[0] + current_rows <= num_rows:
                result.append(arr)
                current_rows += arr.shape[0]
                lst_of_npy[n-i-1] = np.zeros(0)
            else:
                result.append(arr[0:(num_rows-current_rows)])
                lst_of_npy[n-i-1] = lst_of_npy[n-i-1][(num_rows-current_rows):]
                break
        elif current_rows > num_rows:
            raise ValueError("current rows cannot be greater than num_rows")
    
    curr_shape = sum([x.shape[0] for x in lst_of_npy])
    # print(prev_shape, curr_shape, sum([x.shape[0] for x in result]), num_rows)

    assert prev_shape - curr_shape == num_rows
    assert sum([x.shape[0] for x in result]) == num_rows
    return np.vstack(result)



def find_num_samples_in_shard(files):
    lengths = []
    print("Finding optimal length of each shard")
    for file in tqdm((files)):
        f = h5py.File(file, "r")
        lengths.append((np.asarray(f[list(f.keys())[0]][:]).shape[0], file))
        f.close()
    lengths = sorted(lengths, key=lambda x: -x[0])
    return lengths



def balance(files, out_dir, fix_rename, fix_rename_prefix):
    lengths = find_num_samples_in_shard(files)
    if len(lengths) == 0:
        return
    average_length = sum([x[0] for x in lengths]) // len(lengths)
    print("*"*72)
    print(sum([x[0] for x in lengths]))
    print(average_length)
    print(len(lengths))  
    ## We have to re-distribute the shards so that each shard contains equal number of samples.
    ## Since we are operating on tokenised data, re-distributing should not alter the perfornance (IID assumption)

    file_content = {}
    print("Re sharding data into equal blocks")
    for i, (length, file) in tqdm(enumerate(lengths)):
        f = h5py.File(file, "r")
        keys = list(f.keys())
        data_type = {}
        for key in keys:
            data_type[key] = f[key].dtype
            if key not in file_content:
                file_content[key] = [np.asarray(f[key][:])]
            else:
                file_content[key].append(np.asarray(f[key][:]))
                # file_content acts like a stack containing the most recent data that has been read
        f.close()
        
        if fix_rename:
            out_path = os.path.join(out_dir, fix_rename_prefix + str(i) + ".hdf5")
        else:
            out_path = os.path.join(out_dir, os.path.basename(file))
        fout = h5py.File(out_path, "w", libver="latest")

        for key in keys:
            fout.create_dataset(key, data=get_partial_numpy(file_content[key], average_length), dtype=data_type[key], compression="gzip")
            # remove the #average_length data points from `file_content`
            # since they have already been writtent into shards
        fout.close()

def main(files, out_dir, fix_rename):
    train_files = [file for file in files if "train_shard" in file]
    test_files = [file for file in files if "test_shard" in file]
    balance(train_files, out_dir, fix_rename, "train_shard_")
    balance(test_files, out_dir, fix_rename, "test_shard_")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--fix_rename", action='store_true')
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.dir, "**/*.hdf5"), recursive=True)
    
    # make sure no file name collision OR fix_rename is enabled
    if not args.fix_rename and len(set(os.path.basename(f) for f in files)) != len(files):
        print('detected same name of multiple files, use --fix_rename to rename names')
        exit(1)
        
    main(files, args.out_dir, args.fix_rename)
