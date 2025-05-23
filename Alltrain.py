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
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 2. 分词
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. 检查是否存在检查点
CHECKPOINT_DIR = "./checkpoints"
LAST_CHECKPOINT = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        LAST_CHECKPOINT = sorted(checkpoints)[-1]
        LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, LAST_CHECKPOINT)
        logger.info(f"发现最近检查点: {LAST_CHECKPOINT}")

# 4. 加载模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese' if not LAST_CHECKPOINT else LAST_CHECKPOINT,
    num_labels=2
)
model.to('cuda')

# 5. 训练参数（启用检查点保存）
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",  # 与eval_strategy同步
    save_steps=500,         # 每500步保存一次
    save_total_limit=2,     # 最多保留2个检查点
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    disable_tqdm=False,     # 显示进度条
)

# 6. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# 7. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# 8. 训练（带异常处理）
try:
    logger.info("开始训练...")
    print("初始GPU状态:")
    !nvidia-smi
    
    if LAST_CHECKPOINT:
        logger.info(f"从检查点 {LAST_CHECKPOINT} 恢复训练")
        trainer.train(resume_from_checkpoint=LAST_CHECKPOINT)
    else:
        trainer.train()
        
except Exception as e:
    logger.error(f"训练过程中出错: {str(e)}")
    logger.info("尝试保存当前进度...")
    trainer.save_model("./crash_save")
    logger.info("已保存应急模型到 ./crash_save")

# 9. 最终保存
logger.info("训练完成，保存最终模型...")
trainer.save_model("./sentiment_bert_model")
tokenizer.save_pretrained("./sentiment_bert_model")

# 压缩模型
!zip -r sentiment_bert_model.zip sentiment_bert_model/
logger.info("模型已保存并压缩")