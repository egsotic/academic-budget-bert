import argparse
from functools import partial
import glob
import os
import re
import multiprocessing as mp
import tqdm
import re


def prepare_preseg_text_for_tokenizer(text: str, prefix_sep: str, suffix_sep: str, tokenizer_delimiter: str):
    prefix_sep_regex = []
    suffix_sep_regex = []
    
    if prefix_sep:
        prefix_sep_regex = [fr"(?:(?<=[משהוכלב]){re.escape(prefix_sep)})"]
    
    if suffix_sep:
        suffix_sep_regex = [fr"(?:{re.escape(suffix_sep)}(?=[יךהוםןכנ]))"]
    
    preseg_sep_regex = re.compile('|'.join(prefix_sep_regex + suffix_sep_regex))
    
    return preseg_sep_regex.sub(tokenizer_delimiter, text)

def worker(input_path, output_dir, prefix_sep: str, suffix_sep: str, tokenizer_delimiter: str):
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                f_out.write(prepare_preseg_text_for_tokenizer(line, prefix_sep, suffix_sep, tokenizer_delimiter))

def process_parallel(input_path, output_dir, n_processes: int, prefix_sep: str, suffix_sep: str, tokenizer_delimiter: str):
    os.makedirs(output_dir, exist_ok=True)
    
    worker_with_args = partial(worker,
                               output_dir=output_dir,
                               prefix_sep=prefix_sep,
                               suffix_sep=suffix_sep,
                               tokenizer_delimiter=tokenizer_delimiter)
    
    all_inputs_path = glob.glob(input_path)
    
    with mp.Pool(n_processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(worker_with_args, all_inputs_path), desc='files', total=len(all_inputs_path)):
            pass

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--prefix_sep")
    parser.add_argument("--suffix_sep")
    parser.add_argument("--tokenizer_delimiter")
    parser.add_argument("--n_processes", type=int)
    
    args = parser.parse_args()
    
    process_parallel(args.input_path,
                     args.output_dir,
                     args.n_processes,
                     args.prefix_sep,
                     args.suffix_sep,
                     args.tokenizer_delimiter)
    
if __name__ == "__main__":
    main()
