import argparse
import glob
import itertools
import os
import tokenizers
import re
import tqdm
from transformers import PreTrainedTokenizerFast


MORPH_DELIMITER = 'o'

def prepare_segmented_text_for_tokenizer(segmented_text: str, 
                                         seg_delimiter: str='|',
                                         tokenizer_delimiter: str=MORPH_DELIMITER):
    # bypass pre_tokenizer (splitting punct)
    text = segmented_text.strip().replace(seg_delimiter, tokenizer_delimiter)
    # separate prefix
    text_sep_prefixes = ' '.join(re.split(fr'\s|(?<=\b[משהוכלב]{tokenizer_delimiter})|(?<={tokenizer_delimiter}[משהוכלב]{tokenizer_delimiter})', text))
    # separate suffix
    tokens = re.split(fr'\s|(?={tokenizer_delimiter}\w+)', text_sep_prefixes.strip())
    
    return tokens


def train_tokenizer(corpus_file: str, output_dir: str, vocab_size: int):
    # constants
    unk_token = "[UNK]"
    
    morph_special_tokens = [
        # prefix
        *[p + MORPH_DELIMITER.lower() for p in list('משהוכלב') + ['כש', 'מש']],
        # suffix
        *[MORPH_DELIMITER.lower() + s for s in list('יךהוםן') + ['נו', 'כם', 'כן', 'הם', 'הן']],
    ]

    special_tokens = ["[PAD]", unk_token, "[CLS]", "[SEP]", "[MASK]"] + morph_special_tokens


    # config
    tokenizer_name = f'wp_preseg_rft_{vocab_size}'


    # create
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_token="[UNK]"))

    normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFD(),
            tokenizers.normalizers.Lowercase(),
            tokenizers.normalizers.StripAccents(),
        ]
    )
    tokenizer.normalizer = normalizer

    pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.pre_tokenizer = pre_tokenizer


    # train
    corpus_files = glob.glob(corpus_file)

    trainer = tokenizers.trainers.WordPieceTrainer(vocab_size=vocab_size,
                                                   special_tokens=special_tokens)
    total_sequences = sum(1 for f in tqdm.tqdm(corpus_files, desc='files') for _ in open(f, 'r', encoding='utf-8'))
    
    def preprocess_iter(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                text = normalizer.normalize_str(text)
                text = ' '.join(prepare_segmented_text_for_tokenizer(text))
                yield text

    tokenizer.train_from_iterator(itertools.chain.from_iterable(preprocess_iter(f) for f in corpus_files),
                                  trainer=trainer,
                                  length=total_sequences)


    # save
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )


    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        additional_special_tokens=morph_special_tokens,
    )


    output_path = os.path.join(output_dir, tokenizer_name)
    wrapped_tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--corpus_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--vocab_size", type=int)
    
    args = parser.parse_args()
    
    corpus_file = args.corpus_file
    output_dir = args.output_dir
    vocab_size = args.vocab_size
    
    train_tokenizer(corpus_file, output_dir, vocab_size)

if __name__ == "__main__":
    main()
