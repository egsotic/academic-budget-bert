import argparse
import tqdm
import tokenizers
import os


class BertPreTokenizerWrap:
    def __init__(self):
        self.tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    
    def tokenize_tokens(self, text):
        return [w for w, _ in self.tokenizer.pre_tokenize_str(text)]


def process(input_path, output_path, tokenizer):
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm.tqdm(f_in):
                line = line.strip()
                
                if line != '':
                    for token in tokenizer.tokenize_tokens(line):
                        print(token, file=f_out)
                    f_out.write(os.linesep)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    tokenizer = BertPreTokenizerWrap()
    
    process(input_path, output_path, tokenizer)


if __name__ == "__main__":
    main()
