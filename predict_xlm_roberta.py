import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from finetune_xlm_roberta import VietnameseTextClassifier

if __name__ == "__main__":
    import glob
    import argparse
    # Thêm argument để nhận đường dẫn thư mục
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục chứa các file .txt để dự đoán")
    args = parser.parse_args()
    input_dir = args.input_dir

    model_dir = "model_output"
    classifier = VietnameseTextClassifier(model_name=model_dir, max_length=256)
    classifier.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    classifier.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Lấy lại id2label từ config nếu cần
    import json
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
        classifier.id2label = {int(k): v for k, v in config["id2label"].items()}
        classifier.label2id = {k: int(v) for k, v in config["label2id"].items()}

    # Đọc tất cả file .txt trong thư mục
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    texts = []
    file_names = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            file_names.append(os.path.basename(file_path))

    predictions = classifier.predict(texts)
    print("\n=== Dự đoán nhãn cho các file .txt ===")
    for fname, text, pred in zip(file_names, texts, predictions):
        print(f"File: {fname}")
        print(f"Text: {text[:50]}...")
        print(f"Predicted Category: {pred}")
        print() 