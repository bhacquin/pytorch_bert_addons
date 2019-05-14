#!/usr/bin/env bash
while getopts "i:o:" option
do
case $option
in
i) echo "input : $OPTARG"
INPUT="$OPTARG";;
o) echo "output : $OPTARG"
OUTPUT=$OPTARG;;
esac
done
cat $INPUT |  # get all the content from the specified files
  sed -e "s/’/\'/g" | # remove weird apostrophe
  tr '[:punct:][:digit:]' ' ' |  # remove punctuation
  tr '[:upper:]' '[:lower:]' |   # to lowercase
  tr ' ' '\n' |  # split on words
  tqdm |
  sort |
  tqdm |
  uniq -c |
sort -nr > $OUTPUT # sort by most frequent words
