#!jar xvf repo.zip
# sudo yum update
# source activate pytorch_p36
# pip install virtualenv
# virtualenv venv
# source venv/bin/activate

while getopts "a" option
do
case $option
in
a)pip install -r requirements.txt
python -m spacy download en_core_web_lg
mkdir data
mkdir test
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
$address = pytorch_pretrained_bert
chmod +x $address/1_extract_vocab.sh
# bash 1_extract_vocab.sh annotated_stuff.txt

echo "Vocabulary issues"
# python 2_find_missing_tokens.py --vocab_file bert-base-uncased-vocab.txt > new_vocab.txt
# python python1_data.py
# python 3_add_missing_tokens_to_vocab.py --vocab_file bert-base-uncased-vocab.txt --ouput_file full_vocab.txt --missing_tokens_file new_vocab.txt
# mv full_vocab.txt vocab.txt
# python python2_data.py
python $address/vocab_treatment.py

echo "Prepare text data"
python $address/prepare_all_texts_for_bert.py --input_folder data/ --ouput_file training_text.txt
python $address/prepare_all_texts_for_bert.py --input_folder test/ --ouput_file test_text.txt
rm -r training/
rm -r test/

echo 'generating data for train'
python $address/pregenerate_training_data.py --train_corpus 'training_text.txt' --bert_model vocab.txt --do_lower_case --output_dir training/ --epochs_to_generate 2 --max_seq_len 512
python $address/pregenerate_training_data.py --train_corpus 'test_text.txt' --bert_model vocab.txt --do_lower_case --output_dir test/ --epochs_to_generate 2 --max_seq_len 512
;;

esac
done


rm -r log
mkdir log
echo 'tensorboard setup'
tensorboard --logdir=/log --host 0.0.0.0 --port 6006 &


echo 'finetuning starting'
python $address/finetune_on_pregenerated.py --verbose --pregenerated_data training/ --bert_model bert-base-uncased --do_lower_case --output_dir finetuned_lm/ --epochs 10 --train_batch_size 16 --tensorboard  >> results.txt &&
