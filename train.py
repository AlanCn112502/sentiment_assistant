import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import os

# 设置环境变量禁用CUDA（强制使用CPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 读取预处理数据（使用更高效的内存映射方式）
df = pd.read_csv('D:/sentiment_assistant/data/processed_douban.csv', engine='python')

# 转成 Hugging Face Dataset（仅保留必要列减少内存）
dataset = Dataset.from_pandas(df[['text', 'label']])

# 分割训练和验证集（使用更小的验证集）
dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 验证集从20%降到10%

# 加载中文BERT分词器（禁用并行tokenizer警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=64  # 从128减到64，减少计算量
    )

# 使用更高效的映射方式
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # 增大批处理大小提高效率
    remove_columns=['text']  # 移除原始文本减少内存
)

# 加载预训练BERT模型（使用更小的模型）
try:
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese', 
        num_labels=2,
        torch_dtype=torch.float32  # 明确使用float32避免自动混合精度
    )
except:
    # 如果网络问题，使用本地缓存
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese', 
        num_labels=2,
        local_files_only=True
    )

# 冻结部分层减少计算量（可选）
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for param in model.bert.encoder.layer[:6].parameters():  # 冻结前6层
    param.requires_grad = False

# 计算准确率函数（简化版）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# 训练参数（针对CPU优化）
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",  # 改为按steps评估更灵活
    eval_steps=200,         # 每200步评估一次
    save_strategy="steps",
    save_steps=200,
    learning_rate=1e-5,     # 降低学习率
    per_device_train_batch_size=8,   # 减小批大小
    per_device_eval_batch_size=16,
    num_train_epochs=2,     # 减少epoch数
    weight_decay=0.001,
    logging_steps=50,
    report_to="none",       # 禁用所有日志报告
    disable_tqdm=True,      # 禁用进度条减少开销
    no_cuda=True,           # 明确禁用CUDA
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# 开始训练（使用梯度累积模拟更大batch）
print("开始训练...")
trainer.train()

# 训练完成后保存模型（仅保存必要部分）
model.save_pretrained('./sentiment_bert_model', state_dict=model.state_dict())
tokenizer.save_pretrained('./sentiment_bert_model')

print("训练完成！模型已保存到 sentiment_bert_model 目录")