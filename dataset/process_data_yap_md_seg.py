import argparse
import os
import tokenizers
import requests
import json
import pandas as pd
import io
import multiprocessing as mp
from collections import defaultdict, namedtuple
from typing import List

MORPH_DELIMITER = '‡'
TOKEN_ID_KEY = 'token_id'
MA_TAGS = ['form', 'lemma', 'cpostag', 'postag', 'feats']
MA_COLUMNS = ['from_ind', 'to_ind'] + MA_TAGS + [TOKEN_ID_KEY]

MORPH_FORM_KEY = 'form'
LEMMA_FORM_KEY = 'lemma'
POSTAG_FORM_KEY = 'postag'
MORPH_FEATS_FORM_KEY = 'feats'
MorphemeType = namedtuple('MorphemeType', ' '.join((MORPH_FORM_KEY,
                                                    LEMMA_FORM_KEY,
                                                    POSTAG_FORM_KEY,
                                                    MORPH_FEATS_FORM_KEY)))

feats_to_person_morph = {
    (None, 'Sing', '1'): 'אני',
    ('Fem', 'Sing', '1'): 'אני',
    ('Masc', 'Sing', '1'): 'אני',
    ('Fem,Masc', 'Sing', '1'): 'אני',
    
    (None, 'Plur', '1'): 'אנחנו',
    ('Fem,Masc', 'Plur', '1'): 'אנחנו',
    ('Masc', 'Plur', '1'): 'אנחנו',
    ('Fem', 'Plur', '1'): 'אנחנו',
    
    ('Masc', 'Sing', '2'): 'אתה',
    ('Masc', 'Sing', '3'): 'הוא',
    ('Masc', 'Plur', '2'): 'אתם',
    ('Masc', 'Plur', '3'): 'הם',
    ('Fem', 'Sing', '2'): 'את',
    ('Fem', 'Sing', '3'): 'היא',
    ('Fem', 'Plur', '2'): 'אתן',
    ('Fem', 'Plur', '3'): 'הן',
}

def get_feat_k(d, k):
    if 'feats' in d and isinstance(d['feats'], dict):
        return d['feats'].get(k, None)
    
    return None


def get_person_form(d):
    gen = get_feat_k(d, 'Gender')
    num = get_feat_k(d, 'Number')
    per = get_feat_k(d, 'Person')
    k = (gen, num, per)

    if k not in feats_to_person_morph:
        return d['form']
    
    return feats_to_person_morph[k]


'''
HTB
['Gender', 'Number', 'Person']
dict_keys(['Masc', 'Fem,Masc', 'Fem'])
dict_keys(['Sing', 'Plur', 'Plur,Sing', 'Dual', 'Dual,Plur'])
dict_keys(['3', '1,2,3', '1', '2'])

YAP
['suf_gen', 'suf_num', 'suf_per']
{('F', 'M'), ('F',), ('M',)}
{('P',), ('S',)}
{('3',), ('1',), ('2',)}
'''

YAP_to_HTB_fl_suf = {
    # suf_gen
    ('F', 'M'): 'Fem,Masc',
    ('F',): 'Fem',
    ('M',): 'Masc',
    
    # suf_num
    'S': 'Sing',
    'P': 'Plur',
    
    # suf_per
    '1': '1',
    '2': '2',
    '3': '3',
}

def convert_morphs_format_YAP_to_HTB(morphs: List[MorphemeType]):
    morphs_new = []
    
    for m in morphs:
        # של
        if m.feats is not None and 'suf_gen' in m.feats:
            yap_feats = defaultdict(set)
            for k, v in (kv.split('=') for kv in m.feats.split('|')):
                yap_feats[k].add(v)
            
            feats = {
                # can be multi
                'Gender': tuple(sorted(yap_feats['suf_gen'])),
                'Number': list(yap_feats['suf_num'])[0],
                'Person': list(yap_feats['suf_per'])[0],
            }
            
            # convert
            feats = {
                k: YAP_to_HTB_fl_suf[v]
                for k, v in feats.items()
            }
            
            per = get_person_form({
                'feats': feats,
                'form': None
            })
            
            m_new = MorphemeType(form=m.lemma, lemma=m.lemma, postag=m.postag,
                                 feats='|'.join('='.join((k, v))
                                                for k, v in (kv.split('=') for kv in m.feats.split('|'))
                                                if k not in ['suf_gen', 'suf_num', 'suf_per']))
            m_fl = MorphemeType(form='של', lemma='של', postag='_', feats='_')
            # TODO: add lemma and postag according to either הוא or את
            m_per = MorphemeType(form=per, lemma=per, postag='S_PRN',
                                 feats='|'.join('='.join(kv)
                                                for kv in [
                                                    *[('gen', gen) for gen in yap_feats['suf_gen']],
                                                    *[('num', num) for num in yap_feats['suf_num']],
                                                    *[('per', per) for per in yap_feats['suf_per']],
                                                ]))
            
            morphs_new.append(m_new)
            morphs_new.append(m_fl)
            morphs_new.append(m_per)
        else:
            morphs_new.append(m)
        
    return morphs_new


tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()

def tokenize_text(text):
    tokenized_text = ' '.join((w for w, _ in tokenizer.pre_tokenize_str(text)))
    
    return tokenized_text

def yap_md_query(tokenized_text):
    localhost_yap = "http://localhost:8000/yap/heb/joint"
    data = json.dumps({'text': "{}  ".format(tokenized_text)})  # input string ends with two space characters
    headers = {'content-type': 'application/json'}
    response = requests.get(url=localhost_yap, data=data, headers=headers)
    json_response = response.json()
    
    df = pd.read_csv(io.StringIO(json_response['md_lattice']), quoting=3, sep='\t', header=None, names=MA_COLUMNS)
    
    return df

def token_df_to_morphemes(token_df):
    morphs = []
    
    for m in token_df.itertuples():
        morph = MorphemeType(m.form, m.lemma, m.postag, m.feats)
        morphs.append(morph)
    
    morphs = convert_morphs_format_YAP_to_HTB(morphs)
    
    return morphs

def output_md_seg(df):
    try:
        return ' '.join(
            df.groupby('token_id').apply(
                # df -> morphemes (+ convert YAP to HTB e.g. add fl + person)
                lambda g: token_df_to_morphemes(g)
            ).apply(
                # join morphemes
                lambda g: MORPH_DELIMITER.join((m.form for m in g))
            )
        )
    except:
        df.to_csv('~/tmp/debug_log.csv')
        raise

def input_iter(path):
    with open(path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip() != '':
                yield line

def output_md_worker(text):
    try:
        return output_md_seg(yap_md_query(tokenize_text(text)))
    except Exception as e:
        print(e)
        return None


def process(input_path, output_path, n_processes: int = None):
    print_every = 50
    chunksize = 1
    
    with mp.Pool(n_processes) as pool:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, md_seg in enumerate(pool.imap_unordered(output_md_worker, input_iter(input_path), chunksize)):
                if i % print_every == 0:
                    print(i, flush=True)
                    
                if md_seg is not None:
                    print(md_seg, file=f_out, flush=True)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--n_processes", type=int, default=None)
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    n_processes = args.n_processes
    
    process(input_path, output_path, n_processes)

if __name__ == "__main__":
    main()
