#!/bin/bash
RANDOM=9929
DATA_DIR=''
RETRIEVAL_DIR=''
VOCAB_DIR=''
CONFIG_DIR=''
CKPT_DIR=''
for ((i=1; i<=5; i++))
do
  echo train ${i} data
  CUDA_VISIBLE_DEVICES=0 python entity_retrieval_biosess.py  --do_train=True   --do_eval=True  --do_predict=True --task_name=textsim --data_dir=$DATA_DIR --retrieve_dir=$RETRIEVAL_DIR --vocab_file=$VOCAB_DIR  --bert_config_file=$CONFIG_DIR --init_checkpoint=$CKPT_DIR  --max_seq_length=200  --train_batch_size=5 --eval_batch_size=5 --predict_batch_size=5  --learning_rate=2e-5 --other_learning_rate=2e-5  --num_train_epochs=8 --random_seed=$RANDOM --output_dir="BIOSSES_biobert_ea_re_"${i}
done
