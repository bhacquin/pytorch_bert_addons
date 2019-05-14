#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from tqdm import tqdm
from pathlib import Path
import spacy


def main(input_folder: str, ouput_file: FileType, trim: bool):
    nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'tagger', 'ner', 'textcat'])
    nlp.max_length = 2000000
    text_to_write = []
    for file in tqdm(input_folder.glob("*.txt")):
        with open(file, 'r') as f:
            raw_text = f.read()
        text = raw_text.replace('\n\n', ' ')
        doc = nlp(text)
        if trim:
            sentences = [sent.string.strip() for sent in doc.sents if len(sent.string.strip())>15]
        else:
            sentences = [sent.string.strip() for sent in doc.sents]
        text_to_write.append('\n'.join(sentences))
    ouput_file.write('\n\n'.join(text_to_write))


args = ArgumentParser()
args.add_argument(
    '--input_folder',
    type=Path,
    default='processed_texts',
)
args.add_argument(
    '--ouput_file',
    type=FileType(mode='w', encoding='utf-8'),
    default='all_texts.txt',
)
args.add_argument(
    '--trim',
    action='store_true',
)
main(**vars(args.parse_args()))
