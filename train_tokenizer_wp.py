import argparse
import os
import tokenizers
from transformers import PreTrainedTokenizerFast

def train_tokenizer(corpus_file: str, output_dir: str, vocab_size: int):
    # constants
    unk_token = "[UNK]"
    special_tokens = ["[PAD]", unk_token, "[CLS]", "[SEP]", "[MASK]"]


    # config
    tokenizer_name = f'wp_{vocab_size}'


    # create
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFD(),
            tokenizers.normalizers.Lowercase(),
            tokenizers.normalizers.StripAccents(),
        ]
    )


    pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.pre_tokenizer = pre_tokenizer


    # train
    corpus_files = [corpus_file]

    trainer = tokenizers.trainers.WordPieceTrainer(vocab_size=vocab_size,
                                                   special_tokens=special_tokens)


    tokenizer.train(corpus_files, trainer=trainer)


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
