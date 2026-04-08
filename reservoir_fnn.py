import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_curve,auc

# 固定随机种子
torch.manual_seed(42)

class Reservoir_fnn(nn.Module):
    def __init__(self, rhow=1, input_dim=50, hidden_dim=150):
        super(Reservoir_fnn, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(256, 150, kernel_size=1)
        )
        self.rhow = rhow
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.xavier_uniform_(m.weight)
                self.weight_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def weight_init(self, tensor):
        # 初始化权重
        with torch.no_grad():
            tensor.uniform_(-1, 1)

            A = tensor[:, :, 0]

            # 使用 SVD 计算奇异值
            U, S, Vh = torch.svd(A)

            spectral_radius = S.max().item()

            # 归一化
            if spectral_radius != 0:
                tensor.mul_(self.rhow / spectral_radius)

    def forward(self, x):
        # 输入形状 (batch_size, 31, 50)
        return self.convs(x)

# random_array = np.random.rand(1, 31, 50).astype(np.float32)
# input_tensor = torch.from_numpy(random_array)
#
# # 创建模型并进行前向传播
# model = Reservoir_fnn(2)
# output = model(input_tensor)
#
# print(output.shape)  # 输出形状