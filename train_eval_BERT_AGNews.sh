
#! /bin/sh

mkdir -p AGNews_model_save
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data


python3 model_BERT/classifier.py \
    --task agnews \
    --mode train \
    --train_cfg ./model_BERT/config/train_mrpc.json \
    --model_cfg ./model_BERT/config/bert_base.json \
    --data_train_file total_data/agtrain.tsv \
    --data_test_file total_data/ag_test.tsv \
    --pretrain_file ./model_BERT/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --vocab ./model_BERT/uncased_L-12_H-768_A-12/vocab.txt \
    --dataName AG \
    --max_len 256

