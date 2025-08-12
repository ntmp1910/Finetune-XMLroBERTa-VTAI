#!/bin/bash

# Script chạy fine-tune với Docker để access GPU
# Giống cách NeMo hoạt động

echo "=== Chạy Fine-tune với Docker GPU Access ==="

# Kiểm tra GPU trước
echo "1. Kiểm tra GPU trong Docker:"
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi

# Chạy fine-tune với GPU access
echo "2. Chạy fine-tune:"
docker run --gpus all --rm -it \
  -v $PWD:/work -w /work \
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  bash -lc "
    pip install -r requirements.txt && \
    python finetune_xlm_roberta.py \
      --data data/dantri.jsonl \
      --output_dir model_output \
      --epochs 3 \
      --train_bs 8 \
      --eval_bs 8 \
      --lr 2e-5 \
      --gpu_device cuda:0
  "
