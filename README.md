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
├── finetune_xlm_roberta.py      # Script chính để fine-tune (JSON/JSONL)
├── finetune_xlm_roberta_txt.py  # Script fine-tune với dữ liệu TXT
├── predict.py                   # Script predict với model JSON/JSONL
├── predict_txt.py              # Script predict với model TXT
├── test_texts.txt              # File test texts để predict
├── input_texts.txt             # File input texts để predict
├── data/
│   ├── dantri.jsonl            # Dữ liệu training (JSONL format)
│   └── sample_data.txt         # Dữ liệu training (TXT format)
├── model_output/               # Thư mục lưu model JSON/JSONL
├── model_output_txt/           # Thư mục lưu model TXT
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

### TXT Format
Format: `text\tlabel` hoặc `text|label` hoặc `text,label`

### Ví dụ TXT:
```
Bão số 3 đổ bộ vào miền Trung gây thiệt hại nặng nề	Thời tiết
Giá vàng hôm nay tăng mạnh lên mức cao nhất trong tháng	Kinh tế
Công nghệ AI phát triển nhanh chóng trong năm 2024	Công nghệ
```

## 🛠️ Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Đặt file dữ liệu JSON/JSONL hoặc TXT vào thư mục `data/`

### 3. Chuẩn bị file input cho predict (tùy chọn)
Tạo file text chứa các câu cần predict (mỗi dòng một câu) như `input_texts.txt`

## 🚀 Sử dụng

### 1. Fine-tune Model với JSON/JSONL

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

#### Chạy với environment variables (khuyến nghị):
```bash
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
  --lr 2e-5
```

### 2. Fine-tune Model với TXT

#### Chạy trực tiếp:
```bash
python finetune_xlm_roberta_txt.py \
  --data data/sample_data.txt \
  --delimiter "\t" \
  --output_dir model_output_txt \
  --epochs 3 \
  --train_bs 8 \
  --eval_bs 8 \
  --lr 2e-5
```

#### Chạy với environment variables:
```bash
# Set environment variables để tránh lỗi
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Chạy fine-tune với TXT
python finetune_xlm_roberta_txt.py \
  --data data/sample_data.txt \
  --delimiter "\t" \
  --output_dir model_output_txt \
  --epochs 3 \
  --train_bs 4 \
  --eval_bs 4 \
  --lr 2e-5
```

### 3. Predict với Model đã Train

#### Predict với model JSON/JSONL:
```bash
python predict.py --model_path ./model_output
```

#### Predict với model TXT:
```bash
python predict_txt.py --model_path ./model_output_txt
```

#### Predict với texts cụ thể:
```bash
# JSON/JSONL model
python predict.py --model_path ./model_output \
  --texts "Bão số 3 đổ bộ vào miền Trung" "Giá vàng hôm nay tăng mạnh"

# TXT model
python predict_txt.py --model_path ./model_output_txt \
  --texts "Bão số 3 đổ bộ vào miền Trung" "Giá vàng hôm nay tăng mạnh"
```

#### Predict với file texts:
```bash
# JSON/JSONL model
python predict.py --model_path ./model_output --file input_texts.txt

# TXT model
python predict_txt.py --model_path ./model_output_txt --file input_texts.txt
```

#### Predict với confidence scores:
```bash
# JSON/JSONL model
python predict.py --model_path ./model_output --file input_texts.txt --with_confidence

# TXT model
python predict_txt.py --model_path ./model_output_txt --file input_texts.txt --with_confidence
```

## ⚙️ Tham số

### Fine-tune Parameters (JSON/JSONL)
- `--data`: Đường dẫn file dữ liệu JSON/JSONL
- `--output_dir`: Thư mục lưu model (default: `./model_output`)
- `--epochs`: Số epoch train (default: 3)
- `--train_bs`: Batch size train (default: 8)
- `--eval_bs`: Batch size eval (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--max_length`: Max sequence length (default: 256)

### Fine-tune Parameters (TXT)
- `--data`: Đường dẫn file dữ liệu TXT
- `--delimiter`: Delimiter giữa text và label (default: `\t`)
- `--output_dir`: Thư mục lưu model (default: `./model_output_txt`)
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
# Sử dụng environment variables
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
python finetune_xlm_roberta.py --data data/dantri.jsonl
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

- Email: [ntmpwork@example.com]
- GitHub: [ntmp1910]

---

**Lưu ý**: Model được train trên dữ liệu tiếng Việt và có thể cần fine-tune thêm cho domain cụ thể. 