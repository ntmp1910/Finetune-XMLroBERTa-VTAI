# Fine-tune XLM-RoBERTa cho Vietnamese Text Classification

Project này fine-tune model XLM-RoBERTa để phân loại văn bản tiếng Việt với accuracy đạt được 62.72%.

## 🚀 Tính năng

- Fine-tune XLM-RoBERTa cho Vietnamese text classification
- Hỗ trợ format dữ liệu JSON/JSONL
- Tự động xử lý GPU/CPU
- Predict với confidence scores
- Tích hợp wandb logging (offline mode)

## 📁 Cấu trúc Project

```
FineTune xlmr-roBerta/
├── finetune_xlm_roberta.py      # Script chính để fine-tune
├── predict.py                   # Script để predict với model đã train
├── test_texts.txt              # File test texts để predict
├── run_finetune_fixed.sh       # Script chạy fine-tune với env vars
├── run_finetune_docker.sh      # Script chạy fine-tune với Docker
├── run_finetune_sudo.sh        # Script chạy fine-tune với sudo
├── data/
│   └── dantri.jsonl            # Dữ liệu training (JSONL format)
├── model_output/               # Thư mục lưu model đã train
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## 📊 Cấu trúc Dữ liệu

### JSON/JSONL Format
```json
{
  "title": "Tiêu đề bài viết",
  "summary": "Tóm tắt nội dung",
  "category": "Tên category"
}
```

### Ví dụ JSONL:
```jsonl
{"title": "Bão số 3 đổ bộ", "summary": "Thiệt hại nặng nề", "category": "Thời tiết"}
{"title": "Giá vàng tăng", "summary": "Thị trường sôi động", "category": "Kinh tế"}
```

## 🛠️ Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Đặt file dữ liệu JSON/JSONL vào thư mục `data/`

## 🚀 Sử dụng

### 1. Fine-tune Model

#### Chạy trực tiếp:
```bash
python finetune_xlm_roberta.py \
  --data data/dantri.jsonl \
  --output_dir model_output \
  --epochs 3 \
  --train_bs 8 \
  --eval_bs 8 \
  --lr 2e-5
```

#### Chạy với script (khuyến nghị):
```bash
chmod +x run_finetune_fixed.sh
./run_finetune_fixed.sh
```

#### Chạy với Docker:
```bash
chmod +x run_finetune_docker.sh
./run_finetune_docker.sh
```

### 2. Predict với Model đã Train

#### Predict với texts mặc định:
```bash
python predict.py --model_path ./model_output
```

#### Predict với texts cụ thể:
```bash
python predict.py --model_path ./model_output \
  --texts "Bão số 3 đổ bộ vào miền Trung" "Giá vàng hôm nay tăng mạnh"
```

#### Predict với file texts:
```bash
python predict.py --model_path ./model_output --file test_texts.txt
```

#### Predict với confidence scores:
```bash
python predict.py --model_path ./model_output --file test_texts.txt --with_confidence
```

## ⚙️ Tham số

### Fine-tune Parameters
- `--data`: Đường dẫn file dữ liệu JSON/JSONL
- `--output_dir`: Thư mục lưu model (default: `./model_output`)
- `--epochs`: Số epoch train (default: 3)
- `--train_bs`: Batch size train (default: 8)
- `--eval_bs`: Batch size eval (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--max_length`: Max sequence length (default: 256)

### Predict Parameters
- `--model_path`: Đường dẫn đến model đã train
- `--texts`: Các text cần predict
- `--file`: File chứa texts (mỗi dòng một text)
- `--with_confidence`: Hiển thị confidence scores

## 🔧 Cấu hình GPU

### Tự động detect GPU:
```bash
# Model sẽ tự động sử dụng GPU nếu có
python finetune_xlm_roberta.py --data data/dantri.jsonl
```

### Chỉ định GPU cụ thể:
```bash
# Sử dụng GPU 0
python finetune_xlm_roberta.py --data data/dantri.jsonl --gpu_device cuda:0

# Sử dụng CPU
python finetune_xlm_roberta.py --data data/dantri.jsonl --gpu_device cpu
```

## 📈 Kết quả

### Ví dụ Predictions
```
Text: "Bão số 3 đổ bộ vào miền Trung"
Prediction: "Thời tiết"

Text: "Giá vàng hôm nay tăng mạnh"
Prediction: "Kinh tế"

Text: "Công nghệ AI phát triển nhanh chóng"
Prediction: "Công nghệ"
```

## 🐛 Troubleshooting

### Lỗi GPU không tìm thấy:
```bash
# Kiểm tra GPU
nvidia-smi

# Chạy với CPU
python finetune_xlm_roberta.py --gpu_device cpu
```

### Lỗi NCCL:
```bash
# Sử dụng script với environment variables
./run_finetune_fixed.sh
```

### Lỗi Memory:
```bash
# Giảm batch size
python finetune_xlm_roberta.py --train_bs 4 --eval_bs 4
```

## 📝 Logs

### Wandb Logging
- Model sử dụng wandb offline mode
- Logs được lưu tại `wandb/offline-run-*`
- Sync logs: `wandb sync wandb/offline-run-*`

### Training Logs
- Model checkpoints: `model_output/checkpoint-*`
- Best model: `model_output/`
- Training logs: Console output

## 🤝 Đóng góp

1. Fork project
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License

## 📞 Liên hệ

- Email: [your-email@example.com]
- GitHub: [your-github-username]

---

**Lưu ý**: Model được train trên dữ liệu tiếng Việt và có thể cần fine-tune thêm cho domain cụ thể. 