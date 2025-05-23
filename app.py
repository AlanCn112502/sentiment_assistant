import gradio as gr
from predict import SentimentPredictor
from dialogue_generator import DialogueGenerator
import os

# 初始化组件（使用正确的本地路径）
try:
    # 模型路径
    model_path = "D:/sentiment_assistant/train1hmodel"
    
    # 语料路径（重要！改为你的实际路径）
    corpus_path = "D:/sentiment_assistant/response_corpus"
    
    # 验证路径
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"语料目录不存在: {corpus_path}")
    
    predictor = SentimentPredictor(model_path=model_path)
    generator = DialogueGenerator(corpus_dir=corpus_path)
except Exception as e:
    raise RuntimeError(f"初始化失败: {str(e)}")

def chat_interface(text):
    try:
        result = predictor.predict(text)
        label = result["label"]
        response = generator.generate(label)
        return f"识别为：{label}情绪\n\n回复：{response}"
    except Exception as e:
        return f"处理出错: {str(e)}"

iface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="text",
    title="情绪感知对话助手",
    examples=[["今天超级开心！"], ["最近压力好大..."]]
)

iface.launch(share=True)