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

class VietnameseTextClassifierTXT:
    def __init__(self, model_name="/home/dungdx4/BERT/xlm-roberta-base-language-detection", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}

    def load_txt_data(self, txt_file_path, delimiter='\t'):
        """
        Load dữ liệu từ file txt
        Format: text\tlabel hoặc text|label hoặc text,label
        """
        print(f"Đang load dữ liệu từ {txt_file_path}...")
        
        data = []
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Thử các delimiter khác nhau
                    if delimiter in line:
                        parts = line.split(delimiter, 1)
                    elif '|' in line:
                        parts = line.split('|', 1)
                    elif ',' in line:
                        parts = line.split(',', 1)
                    else:
                        print(f"Warning: Line {line_num} không có delimiter hợp lệ: {line}")
                        continue
                    
                    if len(parts) != 2:
                        print(f"Warning: Line {line_num} không đúng format: {line}")
                        continue
                        
                    text, label = parts[0].strip(), parts[1].strip()
                    
                    if text and label:
                        data.append({
                            'text': text,
                            'category': label
                        })
                        
                except Exception as e:
                    print(f"Error parsing line {line_num}: {line} - {e}")
                    continue
        
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("Không load được dữ liệu nào từ file txt")
        
        # Xử lý labels
        unique_categories = sorted(df['category'].unique())
        self.label2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.id2label = {idx: cat for cat, idx in self.label2id.items()}
        df['labels'] = df['category'].map(self.label2id)
        
        # Lọc categories có ít nhất 2 mẫu
        counts = df['labels'].value_counts()
        valid_labels = counts[counts >= 2].index
        df = df[df['labels'].isin(valid_labels)]
        
        # Cập nhật lại label mapping
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
        
        # Move model to GPU
        self.model.to(device="cuda:0")
        print(f"[Info] Model moved to GPU: cuda:0")

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
        output_dir: str = "./model_output_txt",
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
            report_to=None,
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
        
        # Lưu label mapping
        with open(os.path.join(output_dir, "label2id.json"), 'w') as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        
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
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return [self.id2label[p.item()] for p in predictions]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa với dữ liệu TXT")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file dữ liệu TXT")
    parser.add_argument("--delimiter", type=str, default="\t", help="Delimiter giữa text và label (default: tab)")
    parser.add_argument("--output_dir", type=str, default="./model_output_txt", help="Thư mục lưu model")
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
    classifier = VietnameseTextClassifierTXT(max_length=args.max_length)
    df = classifier.load_txt_data(args.data, delimiter=args.delimiter)

    # 2) Setup model/tokenizer
    classifier.setup_model()

    # 3) Chuẩn bị datasets
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(df)

    # 4) Training
    classifier.train(
        train_dataset,
        val_dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
    )

    # 5) Evaluation
    classifier.evaluate(test_dataset)

    # 6) Ví dụ predict nhanh
    test_texts = [
        "Truy nã nghi phạm lừa đảo 5,7 tỉ đồng tiền đặt cọc mua lúa",
        "Soi Hyundai Palisade 2025 trước khi về Việt Nam",
    ]
    print("\n=== Local Predict ===")
    print(classifier.predict(test_texts))

if __name__ == "__main__":
    main()
