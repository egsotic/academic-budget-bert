import argparse
import os
import pandas as pd
import io
import multiprocessing as mp
from collections import defaultdict, namedtuple
from typing import List

import tqdm

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


def token_df_to_morphemes(token_df):
    morphs = []
    
    for m in token_df.itertuples():
        morph = MorphemeType(m.form, m.lemma, m.postag, m.feats)
        morphs.append(morph)
    
    morphs = convert_morphs_format_YAP_to_HTB(morphs)
    
    return morphs


def md_lattice_to_df(raw_md_lattice_str):
    df = pd.read_csv(io.StringIO(raw_md_lattice_str), quoting=3, sep='\t', header=None, names=MA_COLUMNS)
    # to prevent missing form
    return df.fillna({'form': '[UNK]'})


def read_conllu_md_lattice(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = []
        
        for line in f:
            if line.strip() == '':
                yield '\n'.join(lines)
                lines = []
            else:
                lines.append(line)
        
        if len(lines) > 0:
            yield '\n'.join(lines)

def md_lattice_to_df_iter(raw_md_lattice_iter):
    for raw_md_lattice in raw_md_lattice_iter:
        yield md_lattice_to_df(raw_md_lattice)
    

def md_lattice_df_to_seg_text(md_lattice_df):
    return ' '.join(
        md_lattice_df.groupby('token_id').apply(
            # df -> morphemes (+ convert YAP to HTB e.g. add fl + person)
            lambda token_df: token_df_to_morphemes(token_df)
        ).apply(
            # join morphemes
            lambda morphemes: MORPH_DELIMITER.join((m.form for m in morphemes))
        )
    )


def process_worker(md_lattice_df):
    try:
        md_seg_text = md_lattice_df_to_seg_text(md_lattice_df)
        return md_seg_text
    except Exception as e:
        print(e)
        return None

def process(input_path, output_path, n_processes: int = None):
    chunksize = 1
    
    progress_bar = tqdm.tqdm(total=sum(1 for _ in read_conllu_md_lattice(input_path)))
    
    with mp.Pool(n_processes) as pool:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for md_seg_text in pool.imap_unordered(process_worker, 
                                                   md_lattice_to_df_iter(read_conllu_md_lattice(input_path)), chunksize):
                progress_bar.update()
                    
                if md_seg_text is not None:
                    f_out.write(md_seg_text + os.linesep)
                    f_out.flush()

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
