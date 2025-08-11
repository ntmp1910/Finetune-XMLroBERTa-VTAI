# Fine-tune XLM-RoBERTa cho Phân loại Văn bản Tiếng Việt

Dự án này fine-tune model XLM-RoBERTa để phân loại văn bản tiếng Việt sử dụng dữ liệu từ file JSON.

## 📁 Cấu trúc Dự án

```
FineTune xlmr-roBerta/
├── data_demo.json          # Dữ liệu JSON mẫu 1
├── data_demo2.json         # Dữ liệu JSON mẫu 2  
├── finetune_xlm_roberta.py # Script chính để fine-tune
├── quick_test.py           # Script test nhanh
├── requirements.txt         # Dependencies
├── analyze_data.py         # Script phân tích dữ liệu
└── README.md              # Hướng dẫn này
```

## 🚀 Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Kiểm tra môi trường:**
```bash
python quick_test.py
```

## 📊 Cấu trúc Dữ liệu

Dữ liệu JSON cần có cấu trúc như sau:

```json
[
  {
    "title": "Tiêu đề bài báo",
    "category": "Thể thao",
    "summary": "Tóm tắt nội dung bài báo",
    "datetime": "24/07/2025 12:30 GMT+7"
  }
]
```

### ️✅ Hỗ trợ JSONL (jsonlines/NDJSON)

Bạn cũng có thể dùng file `.jsonl` (mỗi dòng là một object JSON). Code đã tự động nhận diện và load theo từng dòng:

Ví dụ `data.jsonl`:

```
{"title": "Tiêu đề 1", "category": "Thể thao", "summary": "Nội dung bài 1"}
{"title": "Tiêu đề 2", "category": "Pháp luật", "summary": "Nội dung bài 2"}
```

- Trường bắt buộc: `title`, `category`, `summary`
- Trường khác (như `datetime`) là tùy chọn
- Chạy fine-tune với JSONL chỉ cần trỏ `--data` tới file `.jsonl`:

```bash
docker run --gpus all --rm -it \
  -e CUDA_VISIBLE_DEVICES="1" \
  -v $PWD:/work -w /work \
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  bash -lc "pip install -r requirements.txt && \
            python finetune_xlm_roberta.py \
              --data path/to/your_data.jsonl \
              --base_model xlm-roberta-base \
              --output_dir model_output"
```

### Các trường bắt buộc:
- `title`: Tiêu đề bài báo
- `category`: Danh mục phân loại
- `summary`: Tóm tắt nội dung

## 🔧 Sử dụng

### 1. Test nhanh
```bash
python quick_test.py
```

### 2. Fine-tune model trên GPU server (qua Docker)

- Chạy fine-tune (chọn GPU bằng CUDA_VISIBLE_DEVICES):

```bash
# Ví dụ: dùng GPU số 1, data ở data_demo2.json, lưu ra ./model_output
docker run --gpus all --rm -it \
  -e CUDA_VISIBLE_DEVICES="1" \
  -v $PWD:/work -w /work \
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  bash -lc "pip install -r requirements.txt && \
            python finetune_xlm_roberta.py \
              --data data_demo2.json \
              --base_model xlm-roberta-base \
              --output_dir model_output \
              --epochs 3 --train_bs 8 --eval_bs 8 --lr 2e-5"
```

- Resume từ model/ckpt sẵn có (đã có ở thư mục làm việc):

```bash
docker run --gpus all --rm -it \
  -e CUDA_VISIBLE_DEVICES="1" \
  -v $PWD:/work -w /work \
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  bash -lc "pip install -r requirements.txt && \
            python finetune_xlm_roberta.py \
              --data data_demo2.json \
              --base_model xlm-roberta-base \
              --resume_from model_output \
              --output_dir model_output \
              --epochs 1 --train_bs 8 --eval_bs 8 --lr 1e-5"
```

### 3. Sử dụng model đã train

```python
from finetune_xlm_roberta import VietnameseTextClassifier

# Load model đã train
classifier = VietnameseTextClassifier()
classifier.setup_model()

# Load model đã lưu
classifier.model = AutoModelForSequenceClassification.from_pretrained("./model_output")
classifier.tokenizer = AutoTokenizer.from_pretrained("./model_output")

# Dự đoán
texts = [
    "Truy nã nghi phạm lừa đảo 5,7 tỉ đồng tiền đặt cọc mua lúa",
    "Xung đột Thái Lan - Campuchia: Thái Lan ghi nhận ít nhất 12 người thiệt mạng"
]

predictions = classifier.predict(texts)
for text, pred in zip(texts, predictions):
    print(f"Text: {text[:50]}...")
    print(f"Category: {pred}")
```

## ⚙️ Cấu hình

### Hyperparameters có thể điều chỉnh:

```python
# Trong finetune_xlm_roberta.py
classifier = VietnameseTextClassifier(
    model_name="xlm-roberta-base",  # Có thể thay bằng "xlm-roberta-large"
    max_length=256                  # Độ dài tối đa của input
)

# Training arguments
training_args = TrainingArguments(
    learning_rate=2e-5,           # Learning rate
    per_device_train_batch_size=8, # Batch size
    num_train_epochs=3,           # Số epochs
    weight_decay=0.01,            # Weight decay
    warmup_steps=500,             # Warmup steps
)
```

## 📈 Kết quả

Script sẽ:
1. **Phân tích dữ liệu**: Hiển thị thống kê về categories và phân bố
2. **Chia dữ liệu**: Train (70%), Validation (10%), Test (20%)
3. **Fine-tune model**: Training với XLM-RoBERTa
4. **Đánh giá**: Accuracy và classification report
5. **Lưu model**: Tại thư mục `./model_output`

## 🎯 Các Categories được phát hiện

Từ dữ liệu `data_demo2.json`, các categories bao gồm:
- Thời sự
- Thể thao  
- Pháp luật
- Thế giới
- Xe
- Sức khỏe
- Giáo dục
- Giải trí
- Và nhiều categories khác...

## 🔍 Troubleshooting

### Lỗi thường gặp:

1. **Out of Memory (OOM)**:
   - Giảm `max_length` xuống 128 hoặc 256
   - Giảm `per_device_train_batch_size` xuống 4 hoặc 2
   - Sử dụng gradient accumulation

2. **CUDA not available**:
   - Cài đặt PyTorch với CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **Dữ liệu không đúng format**:
   - Kiểm tra cấu trúc JSON với `python quick_test.py`

## 📝 Ghi chú

- Model XLM-RoBERTa hỗ trợ đa ngôn ngữ, phù hợp cho tiếng Việt
- Dữ liệu training cần cân bằng giữa các categories để tránh bias
- Có thể sử dụng data augmentation để tăng số lượng mẫu training
- Model được lưu tại `./model_output/` sau khi training xong

## 🤝 Đóng góp

Nếu bạn muốn cải thiện dự án này, hãy:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới MIT License. 