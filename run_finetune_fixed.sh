#!/bin/bash

# Script chạy fine-tune với environment variables để fix lỗi
echo "=== Chạy Fine-tune với Environment Variables ==="

# Set environment variables để tránh lỗi
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Chạy fine-tune
python finetune_xlm_roberta.py \
  --data data/dantri.jsonl \
  --output_dir model_output \
  --epochs 3 \
  --train_bs 4 \
  --eval_bs 4 \
  --lr 2e-5 \
  --gpu_device cuda:4
