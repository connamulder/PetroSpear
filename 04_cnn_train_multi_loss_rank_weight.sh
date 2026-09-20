#!/bin/bash
#### Terminal窗口执行  ./04_cnn_train_multi_loss_rank_weight.sh

soft_weights=(0.0 0.2 0.5 0.9 1.0)

for soft_weight in "${soft_weights[@]}"; do
  echo "Running with soft_weight=$soft_weight"
  python ./cnn_train_multi_loss_rank.py \
      --model_type resnet50 --is_pretrained \
      --do_train --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop 10 \
      --celoss_type celoss_glove --soft_weight $soft_weight \
      --is_parall \
      --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_val.txt
  python ./cnn_train_multi_loss_rank.py \
      --model_type resnet50 --is_pretrained \
      --n_stage 2 --n_epochs 10000 --batch_size 32 --n_stop 10 \
      --celoss_type celoss_glove --soft_weight $soft_weight \
      --is_parall \
      --dataset_name plutonicrocks13_v3 --dataset_file plutonicrocks13_V3_train.txt --val_file plutonicrocks13_V3_test.txt
done
