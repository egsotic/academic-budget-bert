# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import logging
import os

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def write_shard(lines, f_idx, out_dir, prefix):
    filename = f"{prefix}_{f_idx}.txt"

    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as fw:
        for l in lines:
            fw.write(l)

def list_files_in_dir(dir, data_prefix=".txt", file_name_grep=""):
    return [
        f for f in glob.glob(os.path.join(dir, file_name_grep), recursive=True)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="input directory with sharded text files", required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--ratio",
        type=int,
        default=1,
        help="Number of files to merge into a single shard",
    )
    parser.add_argument(
        "--grep",
        type=str,
        default="",
        help="A string to filter a subset of files from input directory",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="shard",
        help="A string to be prefix of each files name (e.g. test/train)",
    )

    args = parser.parse_args()
    dataset_files = list_files_in_dir(args.data, file_name_grep=args.grep)
    num_files = len(dataset_files)
    assert (
        num_files % args.ratio == 0
    ), f"{num_files} % {args.ratio} != 0, make sure equal shards in each merged file"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Compacting input shards into {args.output_dir}")

    # merge input directory shards into num_files_shard
    file_lines = []
    f_idx = 0
    lines_idx = 0
    for f in tqdm(dataset_files, smoothing=1):
        with open(f) as fp:
            finish = False
            line = None
            while not finish:
                try:
                    line = fp.readline()
                    file_lines.append(line)
                    finish = line == ''
                except Exception as e:
                    print(e)
                    continue
        
        if lines_idx == args.ratio - 1:
            write_shard(file_lines, f_idx, args.output_dir, args.prefix)
            file_lines = []
            f_idx += 1
            lines_idx = 0
            continue
        lines_idx += 1
    logger.info("Done!")
