#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

PORT=${PORT:-59500}

python3 -m torch.distributed.launch --master_port=$PORT --nproc_per_node=2 train.py \
  --data_dir ./LLD-MMRI/classification_dataset/images \
  --train_anno_file ./LLD-MMRI/labels/trainval.txt \
  --val_anno_file ./LLD-MMRI/labels/test.txt \
  --batch-size 4 \
  --model latent_fusion_lstm_uniformer_base \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --epochs 300 \
  --output output/ \
  --train_transform_list random_crop z_flip x_flip y_flip rotation edge emboss filter \
  --crop_size 14 112 112 \
  --pretrained \
  --mixup \
  --cb_loss \
  --smoothing 0.1 \
  --img_size 16 128 128 \
  --patch_size 2 4 4 \
  --drop-path 0.1 \
  --eval-metric f1 kappa \
  --num-classes 7 \
  --sampling class \
  --handcraft_feature

  
  

