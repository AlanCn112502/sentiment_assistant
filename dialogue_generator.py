import random
import os

class DialogueGenerator:
    def __init__(self, corpus_dir="response_corpus"):
        try:
            self.positive_responses = self._load_corpus(os.path.join(corpus_dir, "positive.txt"))
            self.negative_responses = self._load_corpus(os.path.join(corpus_dir, "negative.txt"))
        except FileNotFoundError as e:
            raise RuntimeError(f"初始化失败: {str(e)}")

    def _load_corpus(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到语料文件: {filepath}")
        with open(filepath, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError(f"语料文件为空: {filepath}")
            return lines

    def generate(self, emotion_label):
        if emotion_label == "正面":
            return random.choice(self.positive_responses)
        elif emotion_label == "负面":
            return random.choice(self.negative_responses)
        else:
            return "我还不太理解你的情绪，但我会陪着你。"