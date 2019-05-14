#!/usr/bin/env python3

import json
import subprocess
from argparse import ArgumentParser, FileType
from os.path import expanduser
import pandas as pd
import spacy
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import os


def main(annotated_text_file : str, text_file : str, vocab_file: str, word_file: str, threshold: int, output_file : str):
    # nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'tagger', 'ner', 'textcat'])

###### Find missing tokens
    # os.system('sh ./1_extract_vocab.sh')
    # subprocess.call("./1_extract_vocab.sh", shell = True)
    # subprocess.check_call("./1_extract_vocab.sh -i %s -o %s" % (str(annotated_text_file), str(word_file)), shell = True)
    vocab_file = expanduser(vocab_file)
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
    f = open(word_file, "w+")
    print('count,original,splitted', file=f)  # file header
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
            print(count, word, '#'.join(t.strip('#') for t in tokens), sep=',', file=f) # create csv from that output
    ######2nd Stage
    count_unused = 0
    vocab = []
    count = 0
    f = open(word_file, "r")
    for x in f:
        ### the first line is a warning from bert
        #   if count > 0:
        vocab.append(x.replace("\n", "").split(','))
        count += 1
    new_vocab = pd.DataFrame(vocab[1:], columns=vocab[0])
    new_vocab['count'] = new_vocab['count'].apply(int)
    new_vocab.sort_values('count', ascending = False).to_csv('new_vocab.csv')
    missing_tokens = pd.read_csv('new_vocab.csv')
    with open(output_file, 'w') as write:
        with open(vocab_file, 'r') as read:
            for line in tqdm(read):
                if '[unused' in line and count_unused<missing_tokens.shape[0]:
                    write.write(missing_tokens.iloc[count_unused]['original']+'\n')
                    count_unused += 1
                else:
                    write.write(line)


    ### TO SEPARATE THE FULL TEXT INTO DOCUMENTS




    df = pd.read_csv(text_file, delimiter="\n\n", header=None, engine='python')
    docs = df[0].apply(lambda x: x.replace('Operator', ""))

    #### LOAD THE ANNOTATED DATA

    with open(annotated_text_file) as json_file:
        data_annotated = json.load(json_file)
    content = []
    sentiment = []
    for i in range(len(data_annotated['data'])):
        try:
            content.append(data_annotated['data'][i]['content'])

            try:
                sentiment.append(data_annotated['data'][i]['annotation']['sentiment'])

            except:
                print('pb sentiment')
                del content[i]
        except:
            print('pb')
        continue

    docs2 = pd.Series(content)
    docs = pd.concat([docs, docs2])

    # vocab = []
    # count = 0
    # f = open("new_vocab.txt", "r")
    # for x in f:
    #     vocab.append(x.replace("\n", "").split(','))
    #     count += 1
    #
    # ###### ADDING NEW VOCABULARY
    # new_vocab = pd.DataFrame(vocab[1:], columns=vocab[0])
    # new_vocab['count'] = new_vocab['count'].apply(int)
    # new_vocab.sort_values('count', ascending=False).to_csv('new_vocab.csv')

    documents_liste = docs.tolist()

    ## Writing the documents to separate files : propportion to fasten the execution

    for i, document in enumerate(documents_liste):
        if i < len(documents_liste) * 0.99:
            output_file = 'training/transcript_' + str(i) + '.txt'
        else:
            output_file = 'test/transcript_' + str(i) + '.txt'

        new_file = open(output_file, mode="w+", encoding="utf-8")
        new_file.write(document)
        new_file.close()

args = ArgumentParser()
args.add_argument(
    '--vocab_file',
    type=str,
    default='./bert-base-uncased-vocab.txt',
)


args.add_argument(
    '--word_file',
    type=str,
    default='allwords.txt',
)
args.add_argument(
    '--threshold',
    '-t',
    type=int,
    default=5,
)

args.add_argument(
    '--output_file',
    type=str,
    default='./vocab.txt',
)

args.add_argument(
    '--annotated_text_file',
    type=str,
    default='annotated_stuff.txt',
)

args.add_argument(
    '--text_file',
    type=str,
    default='transcripts_presentation.txt',
)

main(**vars(args.parse_args()))
