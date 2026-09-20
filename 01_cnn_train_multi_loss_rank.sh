#!/bin/bash
#### Terminal窗口执行  ./01_cnn_train_multi_loss_rank.sh

# 1. 定义要测试的 nodel_type 列表
model_types=("vgg16")
patients=(200)

for model_type in "${model_types[@]}"; do
  echo "Running with model_type=$model_type"
  for patient in "${patients[@]}"; do
    echo "Running with patient=$patient"
    python ./cnn_train_multi_loss_rank.py \
        --model_type $model_type --is_pretrained \
        --do_train --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_glove --soft_weight 0.2 \
        --is_parall \
        --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_val.txt
    python ./cnn_train_multi_loss_rank.py \
        --model_type $model_type --is_pretrained \
        --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_glove --soft_weight 0.2 \
        --is_parall \
        --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_test.txt
  done
done


model_types=("vit_base")
patients=(50)
for model_type in "${model_types[@]}"; do
  echo "Running with model_type=$model_type"
  for patient in "${patients[@]}"; do
    echo "Running with patient=$patient"
    python ./cnn_train_multi_loss_rank.py \
        --model_type $model_type --is_pretrained \
        --do_train --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_glove --soft_weight 0.2 \
        --is_parall \
        --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_val.txt
    python ./cnn_train_multi_loss_rank.py \
        --model_type $model_type --is_pretrained \
        --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_glove --soft_weight 0.2 \
        --is_parall \
        --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_test.txt
  done
done