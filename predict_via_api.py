# predict_via_api.py
import argparse
from finetune_xlm_roberta import VietnameseTextClassifier
from api_config import BASE_URL, API_KEY, API_MODEL_NAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Một đoạn văn cần dự đoán (ưu tiên)")
    parser.add_argument("--file", type=str, help="Đường dẫn file .txt (mỗi dòng là một văn bản)")
    args = parser.parse_args()

    clf = VietnameseTextClassifier(
        api_base_url=BASE_URL,
        api_key=API_KEY,
        api_model_name=API_MODEL_NAME,
    )

    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit("Hãy truyền --text hoặc --file")

    preds = clf.predict_via_api(texts)
    for t, p in zip(texts, preds):
        print(f"[PRED] {p}\t|\t{t[:80]}")
