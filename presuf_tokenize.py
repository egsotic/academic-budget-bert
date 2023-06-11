import glob
import re
from typing import Dict, List
import itertools

from train_tokenizer_wp_preseg_rft import MORPH_DELIMITER


HEB_VALID_PREFIXES = 'ושמשה,וכשל,וה,וכש,שבכ,ל,מש,בכ,לכ,ושב,מ,ו,כש,שמשה,ומה,ושה,שב,ב,ושל,וכ,שלכ,ושכ,שמ,ה,שכשה,משה,שכשל,מה,ומש,ש,כשה,שמה,וכשה,שה,לכש,ובכ,של,כשב,ולכ,ושלכ,שמש,שכ,וש,ומשה,ומ,ול,ושמש,וכשב,וכשמ,וב,כשל,ושמ,כשמ,שכש,כ'.split(',')
HEB_VALID_SUFFIXES = 'י,נו,ך,ה,הם,הן,כם,כן,ו,ם'.split(',')


def get_trie(strings: List[str]):
    trie = {}

    for s in strings:
        trie_current = trie
        for c in s:
            if c not in trie_current:
                trie_current[c] = {}
            trie_current = trie_current[c]
    
    return trie

def trie_prefix_iter(trie: Dict, prefix: List[str]=[]):
    if len(trie) == 0 and len(prefix) > 0:
        return
    
    for c in trie:
        possible_next = trie[c].keys()
        result = prefix + [c]
        
        yield result, possible_next
        
        yield from trie_prefix_iter(trie[c], prefix + [c])
        

def get_non_overlapping_prefixes_regexes(prefixes: List[str]):
    prefix_regexes = []

    for prefix, possible_next in trie_prefix_iter(get_trie(prefixes)):
        p = ''.join(prefix)
        
        if len(possible_next) > 0:
            p += f"(?![{''.join(possible_next)}])"
        
        prefix_regexes.append(p)
    
    return prefix_regexes

def get_non_overlapping_suffixes_regexes(suffixes: List[str]):
    suffix_regexes = []

    for inv_suffix, possible_prev in trie_prefix_iter(get_trie(s[::-1] for s in suffixes)):
        s = ''.join(inv_suffix[::-1])
        
        if len(possible_prev) > 0:
            s = f"(?<!{''.join(possible_prev)})" + s
        
        suffix_regexes.append(s)
    
    return suffix_regexes

def get_prefixes_regex(prefixes: List[str]):
    prefixes_regexes = get_non_overlapping_prefixes_regexes(prefixes)
    prefixes_regex = re.compile(fr"\b(?:{'|'.join(prefixes_regexes)})(?!\b)")
    
    return prefixes_regex

def get_suffixes_regex(suffixes: List[str]):
    suffixes_regexes = get_non_overlapping_suffixes_regexes(suffixes)
    suffixes_regex = re.compile(fr"(?:(?<!\b))(?:{'|'.join(suffixes_regexes)})\b")
    
    return suffixes_regex

HEB_PREFIXES_REGEX = get_prefixes_regex(HEB_VALID_PREFIXES)
HEB_SUFFIXES_REGEX = get_suffixes_regex(HEB_VALID_SUFFIXES)

def process_prefix(m, sep: str='_'):
    prefixes = m.group()
    
    return ''.join(itertools.chain(*zip(prefixes, sep * len(prefixes))))


def process_suffix(m, sep: str='@'):
    suffixes = m.group()
    
    return ''.join(itertools.chain(*zip(sep * len(suffixes), suffixes)))

def heb_separate_prefixes_suffixes(text: str):
    text_sep_prefixes = re.sub(HEB_PREFIXES_REGEX,
                               process_prefix,
                               text)
    text_sep_suffixes = re.sub(HEB_SUFFIXES_REGEX,
                               process_suffix,
                               text_sep_prefixes)
    
    return text_sep_suffixes


if __name__ == "__main__":
    import argparse
    import tokenizers
    import multiprocessing as mp
    import os
    import tqdm
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--n_processes", type=int)
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    n_processes = args.n_processes
    
    tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()

    def tokenize_text(text):
        return [w for w, _ in tokenizer.pre_tokenize_str(text)]

    def batch_iter(items, batch_size):
        batch = []
        
        for x in items:
            batch.append(x)
            
            if len(batch) >= batch_size:
                yield batch[:]
                
                batch = []
        
        if len(batch) > 0:
            yield batch
    
    def worker(path):
        file_name = os.path.basename(path)
        output_path = os.path.join(output_dir, file_name)
        
        with open(path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for lines in batch_iter(f_in, batch_size=256):
                    try:
                        tokenized_text = ';;;'.join(' '.join(tokenize_text(line)) for line in lines)
                        presuf_separated_text = heb_separate_prefixes_suffixes(tokenized_text)
                        presuf_separated_text = presuf_separated_text.replace(';;;', '\n')
                        f_out.write(presuf_separated_text + os.linesep)
                        f_out.flush()
                    except Exception as e:
                        print(e)

    with mp.Pool(n_processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(worker, glob.glob(input_path))):
            pass
