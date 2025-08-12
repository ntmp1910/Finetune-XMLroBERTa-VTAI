#!/usr/bin/env python3
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VietnameseTextPredictorTXT:
    def __init__(self, model_path, max_length=256):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        
    def load_model(self):
        print(f"Đang load model từ: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Load label mapping
        if os.path.exists(os.path.join(self.model_path, "label2id.json")):
            with open(os.path.join(self.model_path, "label2id.json"), 'r') as f:
                self.label2id = json.load(f)
            self.id2label = {int(v): k for k, v in self.label2id.items()}
        else:
            # Fallback: try to get from model config
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label
            
        print(f"Model loaded với {len(self.label2id)} labels")
        print(f"Labels: {list(self.label2id.keys())}")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda:0")
            print("Model moved to GPU: cuda:0")
        else:
            print("Using CPU")
            
    def predict(self, texts):
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
            
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return [self.id2label[p.item()] for p in predictions]
    
    def predict_with_confidence(self, texts):
        """Predict với confidence scores"""
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
            
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        results = []
        for i, pred in enumerate(predictions):
            label = self.id2label[pred.item()]
            confidence = probabilities[i][pred].item()
            results.append({
                'text': texts[i],
                'prediction': label,
                'confidence': confidence
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Predict với model train từ dữ liệu TXT")
    parser.add_argument("--model_path", type=str, default="./model_output_txt", help="Đường dẫn đến model đã train")
    parser.add_argument("--texts", nargs='+', help="Các text cần predict")
    parser.add_argument("--file", type=str, help="File chứa texts (mỗi dòng một text)")
    parser.add_argument("--with_confidence", action='store_true', help="Hiển thị confidence scores")
    
    args = parser.parse_args()
    
    # Khởi tạo predictor
    predictor = VietnameseTextPredictorTXT(args.model_path)
    predictor.load_model()
    
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
    
    if args.with_confidence:
        results = predictor.predict_with_confidence(texts_to_predict)
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 50)
    else:
        predictions = predictor.predict(texts_to_predict)
        for text, pred in zip(texts_to_predict, predictions):
            print(f"Text: {text}")
            print(f"Prediction: {pred}")
            print("-" * 50)

if __name__ == "__main__":
    main()
