#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from os.path import expanduser
from tqdm import tqdm
import pandas as pd


def main(vocab_file: str, ouput_file: str, missing_tokens_file: str, limit: int):
    vocab_file = expanduser(vocab_file)
    ouput_file = expanduser(ouput_file)
    count_unused = 0

    vocab = []
    count = 0
    f = open("new_vocab.txt", "r")
    for x in f:
        ### the first line is a warning from bert
        #   if count > 0:
        vocab.append(x.replace("\n", "").split(','))
        count += 1
    new_vocab = pd.DataFrame(vocab[1:], columns=vocab[0])
    new_vocab['count'] = new_vocab['count'].apply(int)
    # new_vocab.sort_values('count', ascending=False).to_csv('new_vocab.csv')

    # missing_tokens = pd.read_csv(missing_tokens_file)
    missing_tokens = new_vocab
    with open(ouput_file, 'w') as write:
        with open(vocab_file, 'r') as read:
            for line in tqdm(read):
                if '[unused' in line and count_unused<missing_tokens.shape[0]:
                    write.write(missing_tokens.iloc[count_unused]['original']+'\n')
                    count_unused += 1
                else:
                    write.write(line)


args = ArgumentParser()
args.add_argument(
    '--vocab_file',
    type=str,
    default='~/.data/cased_L-12_H-768_A-12/vocab.txt',
)
args.add_argument(
    '--ouput_file',
    type=str,
    default='~/.data/cased_L-12_H-768_A-12/vocab_modified.txt',
)
args.add_argument(
    '--missing_tokens_file',
    type=str,
    default='missing_tokens.csv',
)
args.add_argument(
    '--limit',
    '-l',
    type=int,
    default=101,
)
main(**vars(args.parse_args()))

