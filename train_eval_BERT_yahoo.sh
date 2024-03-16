
#! /bin/sh
mkdir -p yahoo_model_save
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data

CUDA_VISIBLE_DEVICES=2 python3 model_BERT/classifier.py \
    --task yahoo \
    --mode train \
    --train_cfg ./model_BERT/config/train_mrpc.json \
    --model_cfg ./model_BERT/config/bert_base.json \
    --data_train_file total_data/yahootrain.tsv \
    --data_test_file total_data/yahoo_test.tsv \
    --pretrain_file ./model_BERT/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --vocab ./model_BERT/uncased_L-12_H-768_A-12/vocab.txt \
    --dataName yahoo \
    --max_len 256

