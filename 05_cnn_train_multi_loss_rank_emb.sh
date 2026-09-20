#!/bin/bash
#### Terminal窗口执行  ./05_cnn_train_multi_loss_rank_emb.sh

# 1. 定义要测试的 nodel_type 列表
model_types=("resnet50")
celoss_types=("celoss_glove")
# celoss_types=("celoss_clip" "celoss_node")
soft_weights=(0.2)

for model_type in "${model_types[@]}"; do
  echo "Running with model_type=$model_type"
  for celoss_type in "${celoss_types[@]}"; do
    echo "Running with celoss_type=$celoss_type"
    for soft_weight in "${soft_weights[@]}"; do
      echo "Running with soft_weight=$soft_weight"
      python ./cnn_train_multi_loss_rank.py \
          --model_type $model_type --is_pretrained \
          --do_train --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop 50 \
          --celoss_type $celoss_type \
          --soft_weight $soft_weight \
          --is_parall \
          --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_val.txt
      python ./cnn_train_multi_loss_rank.py \
          --model_type $model_type --is_pretrained \
          --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop 50 \
          --celoss_type $celoss_type \
          --soft_weight $soft_weight \
          --is_parall \
          --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_test.txt
    done
  done
done
