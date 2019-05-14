#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from os.path import expanduser
from tqdm import tqdm


def main(vocab_file: str, word_file: FileType, threshold: int):
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    vocab_file = expanduser(vocab_file)
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False)

    print('count,original,splitted')  # file header
    for line in tqdm(word_file, 'words'):
        c_word = line.strip().split()
        if len(c_word) == 1:  # word is a space
            continue

        count, word = c_word
        count = int(count)
        if count < threshold:
            break

        tokens = tokenizer.tokenize(word)
        if len(tokens) > 1:  # we have subwords
            if len(tokens) == 2 and tokens[1] == '##s':
                continue
            print(count, word, '#'.join(t.strip('#') for t in tokens), sep=',') # create csv from that output


args = ArgumentParser()
args.add_argument(
    '--vocab_file',
    type=str,
    default='~/.data/cased_L-12_H-768_A-12/vocab.txt',
)
args.add_argument(
    '--word_file',
    type=FileType(mode='r', encoding='utf-8'),
    default='allwords.txt',
)
args.add_argument(
    '--threshold',
    '-t',
    type=int,
    default=5,
)
main(**vars(args.parse_args()))

