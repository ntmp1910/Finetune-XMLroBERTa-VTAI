#!/usr/bin/env python3
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os
from finetune_xlm_roberta import VietnameseTextClassifier # Import the class

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Predict với model đã train")
    parser.add_argument("--model_path", type=str, default="./model_output", help="Đường dẫn đến model đã train")
    parser.add_argument("--texts", nargs='+', help="Các text cần predict")
    parser.add_argument("--file", type=str, help="File chứa texts (mỗi dòng một text)")
    parser.add_argument("--with_confidence", action='store_true', help="Hiển thị confidence scores")
    
    args = parser.parse_args()
    
    # Khởi tạo predictor
    predictor = VietnameseTextClassifier(model_name=args.model_path)
    predictor.setup_model()
    
    # Lấy texts để predict
    texts_to_predict = []
    
    if args.texts:
        texts_to_predict = args.texts
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts_to_predict = [line.strip() for line in f if line.strip()]
    else:
        # Default test texts
        texts_to_predict = [
            "Truy nã nghi phạm lừa đảo 5,7 tỉ đồng tiền đặt cọc mua lúa",
            "Soi Hyundai Palisade 2025 trước khi về Việt Nam",
            "Bão số 3 đổ bộ vào miền Trung",
            "Giá vàng hôm nay tăng mạnh",
            "Công nghệ AI phát triển nhanh chóng"
        ]
    
    print(f"\n=== Predict {len(texts_to_predict)} texts ===")
    
    #if args.with_confidence:
    #    results = predictor.predict_with_confidence(texts_to_predict)
    #    for result in results:
    #        print(f"Text: {result['text']}")
    #        print(f"Prediction: {result['prediction']}")
    #        print(f"Confidence: {result['confidence']:.4f}")
    #        print("-" * 50)
    #else:
    predictions = predictor.predict(texts_to_predict)
    for text, pred in zip(texts_to_predict, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print("-" * 50)

if __name__ == "__main__":
    main()
