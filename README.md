# 中文情感分析与对话生成助手

这是一个基于 PyTorch 的 NLP 项目，包含以下功能：

- 中文电影评论情感分类模型（正面/负面）
- 后续拓展：Gradio 可视化界面
- 数据来源：豆瓣电影短评（约20万条）

## 项目结构

- `prepare_data.py`：数据预处理
- `train_model.py`：模型训练
- `app.py`：Gradio 可视化界面
- `data/`：数据文件夹

## 环境要求

- Python 3.8+
- PyTorch
- jieba
- pandas, scikit-learn, matplotlib, gradio

## 使用方法

```bash
# 数据处理
python prepare_data.py

# 模型训练
python train_model.py

# 启动界面
python app.py

# 由于大小限制没有上传数据集，这是链接
https://www.kaggle.com/datasets/alanbrian/processed-douban
