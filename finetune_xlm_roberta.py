#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import wandb 
wandb.init(mode = "offline")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ====== THÊM: hỗ trợ gọi API OpenAI-compatible ======
import openai
try:
    from api_config import BASE_URL, API_KEY, API_MODEL_NAME
except Exception:
    BASE_URL, API_KEY, API_MODEL_NAME = None, None, None
# ====================================================

class VietnameseTextClassifier:
    def __init__(self, model_name="/home/dungdx4/BERT/xlm-roberta-base-language-detection", max_length=512,
                 api_base_url: str = None, api_key: str = None, api_model_name: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}

        # ====== THÊM: cấu hình API ======
        self.api_base_url = api_base_url or BASE_URL
        self.api_key = api_key or API_KEY
        self.api_model_name = api_model_name or API_MODEL_NAME
        self.api_client = None
        if self.api_base_url:
            self.api_client = openai.Client(base_url=self.api_base_url, api_key=self.api_key or "EMPTY")
        # =================================

    def load_json_auto(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            first_char = f.read(1); f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                return [json.loads(line) for line in f if line.strip()]

    def load_data(self, json_file_path):
        print(f"Đang load dữ liệu từ {json_file_path}...")
        data = self.load_json_auto(json_file_path)
        df = pd.DataFrame(data if isinstance(data, list) else data.get('train', []))
        df['text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        df['text'] = df['text'].str.strip()
        df = df[df['category'].notna() & (df['text'] != '')]
        unique_categories = sorted(df['category'].unique())
        self.label2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.id2label = {idx: cat for cat, idx in self.label2id.items()}
        df['labels'] = df['category'].map(self.label2id)
        counts = df['labels'].value_counts()
        valid_labels = counts[counts >= 2].index
        df = df[df['labels'].isin(valid_labels)]
        unique_categories = sorted(df['category'].unique())
        self.label2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.id2label = {idx: cat for cat, idx in self.label2id.items()}
        df['labels'] = df['category'].map(self.label2id)
        print(f"Tổng số mẫu: {len(df)}")
        print(f"Số lượng categories: {len(unique_categories)}")
        print(f"Categories: {unique_categories}")
        return df

    def prepare_datasets(self, df, test_size=0.2, val_size=0.1):
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['labels'])
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42, stratify=train_df['labels'])
        print(f"Train set: {len(train_df)} mẫu")
        print(f"Validation set: {len(val_df)} mẫu")
        print(f"Test set: {len(test_df)} mẫu")
        return (Dataset.from_pandas(train_df),
                Dataset.from_pandas(val_df),
                Dataset.from_pandas(test_df))

    def tokenize_data(self, dataset):
        def tokenize_function(examples):
            result = self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=self.max_length)
            if 'labels' in examples:
                result['labels'] = examples['labels']
            return result
        return dataset.map(
            tokenize_function, batched=True,
            remove_columns=[c for c in dataset.column_names if c not in ['labels', 'input_ids', 'attention_mask']]
        )

    def setup_model(self):
        print(f"Đang load tokenizer và model: {self.model_name}")
        
        # Sử dụng cách gọi GPU chính xác
        self.tokenizer = AutoTokenizer.from_pretrained("/home/dungdx4/BERT/xlm-roberta-base-language-detection")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        num_labels = len(self.label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/home/dungdx4/BERT/xlm-roberta-base-language-detection",
            num_labels=num_labels, 
            id2label=self.id2label, 
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        print(f"Model đã được khởi tạo với {num_labels} labels")
        
        # Move model to GPU theo cách chính xác
        self.model.to(device="cuda:0")
        print(f"[Info] Model moved to GPU: cuda:1")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        unique_label_ids = sorted(np.unique(np.concatenate([labels, predictions])))
        target_names = [self.id2label[i] for i in unique_label_ids]
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(
                labels, predictions, labels=unique_label_ids, target_names=target_names, digits=4, output_dict=True
            )
        }

    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir: str = "./model_output",
        learning_rate: float = 2e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        num_train_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 100,
        dataloader_num_workers: int = 1,
    ):
        tokenized_train = self.tokenize_data(train_dataset)
        tokenized_val = self.tokenize_data(val_dataset)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_steps=logging_steps,
            save_total_limit=2,
            warmup_steps=warmup_steps,
            dataloader_num_workers=dataloader_num_workers,
            report_to=None,  # Tắt tensorboard để tránh lỗi protobuf
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            
        )
        trainer = Trainer(
            model=self.model, args=training_args,
            train_dataset=tokenized_train, eval_dataset=tokenized_val,
            tokenizer=self.tokenizer, data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        print("Bắt đầu training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model đã được lưu tại: {output_dir}")
        return trainer

    def evaluate(self, test_dataset):
        tokenized_test = self.tokenize_data(test_dataset)
        trainer = Trainer(model=self.model, tokenizer=self.tokenizer, compute_metrics=self.compute_metrics)
        results = trainer.evaluate(tokenized_test)
        print("\n=== Kết quả đánh giá trên Test Set ===")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        return results

    def predict(self, texts):
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        
        devices = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return [self.id2label[p.item()] for p in predictions]

    # ====== THÊM: inference qua API OpenAI-compatible ======
    def predict_via_api(self, texts):
        if not self.api_client or not self.api_model_name:
            raise RuntimeError("Chưa cấu hình api_base_url/api_model_name cho predict_via_api")

        preds = []
        # Prompt đơn giản yêu cầu trả về đúng tên category
        system_prompt = (
            "Bạn là bộ phân loại văn bản tiếng Việt. "
            "Nhiệm vụ: trả về DUY NHẤT tên category phù hợp nhất cho đoạn văn sau. "
            "Không thêm giải thích hay ký tự thừa."
        )

        for t in texts:
            user_prompt = f"Văn bản:\n{t}\n\nTrả về category:"
            resp = self.api_client.chat.completions.create(
                model=self.api_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            label = resp.choices[0].message.content.strip()
            preds.append(label)
        return preds
    # =======================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Vietnamese text classification")
    parser.add_argument("--data", type=str, default="data_demo2.json", help="Đường dẫn file dữ liệu JSON")
    parser.add_argument("--base_model", type=str, default="xlm-roberta-base", help="Tên model base hoặc đường dẫn model local để resume")
    parser.add_argument("--resume_from", type=str, default=None, help="Đường dẫn checkpoint/model để tiếp tục train")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Thư mục lưu model")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch train")
    parser.add_argument("--train_bs", type=int, default=8, help="Batch size train mỗi GPU")
    parser.add_argument("--eval_bs", type=int, default=8, help="Batch size eval mỗi GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    # In thông tin GPU
    try:
        print(f"[Info] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Info] Total GPUs on system: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"[Error] GPU check failed: {e}")

    # 1) Load data để xác định label mapping trước
    classifier = VietnameseTextClassifier(
        model_name=args.base_model,
        max_length=args.max_length,
    )

    df = classifier.load_data(args.data)

    # 2) Nếu resume_from được cung cấp, dùng đường dẫn đó làm model nguồn
    if args.resume_from and os.path.isdir(args.resume_from):
        classifier.model_name = args.resume_from

    # 3) Setup model/tokenizer
    classifier.setup_model()

    # 4) Chuẩn bị datasets
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(df)

    # 5) Training
    classifier.train(
        train_dataset,
        val_dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
    )

    # 6) Evaluation
    classifier.evaluate(test_dataset)

    # 7) Ví dụ predict nhanh tại local model
    test_texts = [
        "Truy nã nghi phạm lừa đảo 5,7 tỉ đồng tiền đặt cọc mua lúa",
        "Soi Hyundai Palisade 2025 trước khi về Việt Nam",
    ]
    print("\n=== Local Predict ===")
    print(classifier.predict(test_texts))

if __name__ == "__main__":
    main()