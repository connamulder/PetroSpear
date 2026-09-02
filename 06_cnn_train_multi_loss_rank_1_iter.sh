#!/bin/bash
#### Terminal窗口执行  ./06_cnn_train_multi_loss_rank_1_iter.sh

dataset_names=("plutonicrocks13_v3" "plutonicrocks13_v3_Iter2" "plutonicrocks13_v3_Iter3")
patients=(10 20 50)

for dataset_name in "${dataset_names[@]}"; do
  echo "Running with dataset_name=$dataset_name"
  for patient in "${patients[@]}"; do
    echo "Running with patient=$patient"
    python ./06_cnn_train_multi_loss_rank.py \
        --model_type resnet50 --is_pretrained \
        --do_train --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_node \
        --soft_weight 0.1 \
        --is_parall \
        --dataset_name $dataset_name --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_val.txt
    python ./06_cnn_train_multi_loss_rank.py \
        --model_type resnet50 --is_pretrained \
        --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop $patient \
        --celoss_type celoss_node \
        --soft_weight 0.1 \
        --is_parall \
        --dataset_name $dataset_name --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_test.txt
  done
done
