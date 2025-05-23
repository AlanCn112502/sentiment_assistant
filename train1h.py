import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 加载数据
DATA_PATH = '/kaggle/input/processed-douban/processed_douban.csv'
df = pd.read_csv(DATA_PATH)

# 数据子采样（临时加速训练，正式训练时可移除）
df = df.sample(frac=0.5, random_state=42) if len(df) > 50000 else df

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 2. 分词（缩短序列长度）
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
def tokenize_function(examples):
    return tokenizer(examples['text'], 
                   padding='max_length', 
                   truncation=True, 
                   max_length=64)  # 从128减到64
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. 加载模型（不检查检查点以节省时间）
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)
model.to('cuda')

# 4. 训练参数（优化为3小时配置）
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,                  # 减少epoch
    per_device_train_batch_size=64,       # 增大batch size
    per_device_eval_batch_size=64,
    fp16=True,                           # 启用混合精度
    gradient_accumulation_steps=1,       # 不使用梯度累积
    learning_rate=3e-5,                  # 稍高学习率
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,                      # 减少评估频率
    save_strategy="no",                  # 不保存检查点
    warmup_steps=100,                    # 增加warmup
    report_to="none",
    disable_tqdm=False,
)

# 5. 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# 6. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# 7. 训练
logger.info("开始训练...")
print("初始GPU状态:")
!nvidia-smi

trainer.train()

# 8. 保存最终模型
logger.info("训练完成，保存模型...")
model.save_pretrained("./sentiment_bert_model")
tokenizer.save_pretrained("./sentiment_bert_model")

# 压缩模型
!zip -r sentiment_bert_model.zip sentiment_bert_model/
logger.info("模型已保存并压缩")