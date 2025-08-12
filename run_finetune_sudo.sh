#!/bin/bash

# Script chạy fine-tune với sudo để access GPU
# Giống cách NeMo hoạt động với quyền root

echo "=== Chạy Fine-tune với Sudo GPU Access ==="

# Kiểm tra GPU với sudo
echo "1. Kiểm tra GPU với sudo:"
sudo nvidia-smi

# Chạy fine-tune với sudo
echo "2. Chạy fine-tune với sudo:"
sudo -E python finetune_xlm_roberta.py \
  --data data/dantri.jsonl \
  --output_dir model_output \
  --epochs 3 \
  --train_bs 8 \
  --eval_bs 8 \
  --lr 2e-5
