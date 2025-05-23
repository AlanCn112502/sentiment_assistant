# predict.py

import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification


class SentimentPredictor:
    def __init__(self, model_path="sentiment_bert_model"):
        # 自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型和分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

        label_str = "正面" if pred_label == 1 else "负面"
        return {
            "label": label_str,
            "score": round(confidence, 4)
        }


# 可独立运行测试
if __name__ == "__main__":
    predictor = SentimentPredictor("sentiment_bert_model")
    result = predictor.predict("这部电影太感人了，我哭了。")
    print(result)
