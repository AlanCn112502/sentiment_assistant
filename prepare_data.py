import pandas as pd
import os

def prepare_data():
    # 确保 data 文件夹存在
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 读取 CSV 文件
    df = pd.read_csv('data/douban_sample.csv')
    
    # 选取评论和评分列
    data = df[['Comment', 'Star']]
    
    # 重命名列名，方便后续代码使用
    data = data.rename(columns={'Comment': 'text', 'Star': 'label'})
    
    # 简单处理标签，比如把评分转为情感类别（正面/负面）
    # 假设评分4及以上为正面，3及以下为负面
    data['label'] = data['label'].apply(lambda x: 1 if x >= 4 else 0)
    
    # 去掉空值
    data = data.dropna()
    
    # 保存处理后的数据为新文件
    data.to_csv('data/processed_douban.csv', index=False, encoding='utf-8')
    
    print("数据预处理完成，保存为 data/processed_douban.csv")
    print(data.head())

if __name__ == "__main__":
    prepare_data()
