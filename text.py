import torch
print(torch.__version__)          # 查看当前版本
print(torch.cuda.is_available())  # 是否支持CUDA
print(torch.version.cuda)         # 当前PyTorch关联的CUDA版本

