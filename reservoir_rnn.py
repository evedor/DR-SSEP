import torch
import torch.nn as nn
# 固定随机种子
# torch.manual_seed(42)

class Reservoir_rnn(nn.Module):
    def __init__(self, rhow=1, input_dim=19, hidden_dim=150):
        super(Reservoir_rnn, self).__init__()

        self.rhow = rhow
        self.input_dim = input_dim
        # RNN层
        self.rnn1 = nn.RNN(input_size=self.input_dim, hidden_size=64, batch_first=True)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化RNN层的权重
        for m in self.modules():
            if isinstance(m, nn.RNN):
                # 初始化 RNN 的权重
                self.weight_init(m.weight_ih_l0)  # 输入到隐藏层的权重
                self.weight_init(m.weight_hh_l0)  # 隐藏层到隐藏层的权重

                # 如果有偏置，则初始化为零
                if m.bias_ih_l0 is not None:
                    nn.init.zeros_(m.bias_ih_l0)  # 输入到隐藏层的偏置
                if m.bias_hh_l0 is not None:
                    nn.init.zeros_(m.bias_hh_l0)  # 隐藏层到隐藏层的偏置

    def weight_init(self, tensor):
        # 使用uniform分布初始化权重
        with torch.no_grad():
            tensor.uniform_(-10, 10)
            # torch.nn.init.xavier_uniform_(tensor)
            A = tensor[:, :]
            # 使用 SVD 计算奇异值
            U, S, Vh = torch.linalg.svd(A)

            spectral_radius = S.max().item()

            # 归一化
            if spectral_radius != 0:
                tensor.mul_(self.rhow / spectral_radius)

    def forward(self, x):
        # 输入形状 (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)
        # x: (1, 31, 50) -> RNN输入是(batch_size, seq_len, input_dim)

        # RNN 层输出, output形状: (batch_size, seq_len, hidden_dim)
        x, _ = self.rnn1(x)  # output: (1, 31, 150)
        x, _ = self.rnn2(x)
        x = x.transpose(1, 2)

        return x

class Reservoir_rnn_deep(nn.Module):
    def __init__(self, rhow=1, input_dim=50, hidden_dim=150):
        super(Reservoir_rnn, self).__init__()

        self.rhow = rhow

        # RNN层
        self.rnn1 = nn.RNN(input_size=10, hidden_size=64, batch_first=True)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=hidden_dim, batch_first=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化RNN层的权重
        for m in self.modules():
            if isinstance(m, nn.RNN):
                # 初始化 RNN 的权重
                self.weight_init(m.weight_ih_l0)  # 输入到隐藏层的权重
                self.weight_init(m.weight_hh_l0)  # 隐藏层到隐藏层的权重

                # 如果有偏置，则初始化为零
                if m.bias_ih_l0 is not None:
                    nn.init.zeros_(m.bias_ih_l0)  # 输入到隐藏层的偏置
                if m.bias_hh_l0 is not None:
                    nn.init.zeros_(m.bias_hh_l0)  # 隐藏层到隐藏层的偏置
                    # for param in m.parameters():
                    #     if param.dim() == 2:  # 权重矩阵
                    #         self.weight_init(param)
                    #     elif param.dim() == 1:  # 偏置项
                    #         nn.init.zeros_(param)

    def weight_init(self, tensor):
        # 使用uniform分布初始化权重
        with torch.no_grad():
            tensor.uniform_(-10, 10)
            # torch.nn.init.xavier_uniform_(tensor)
            A = tensor[:, :]
            # 使用 SVD 计算奇异值
            U, S, Vh = torch.linalg.svd(A)

            spectral_radius = S.max().item()

            # 归一化
            if spectral_radius != 0:
                tensor.mul_(self.rhow / spectral_radius)

    def forward(self, x):
        # 输入形状 (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)
        # x: (1, 31, 50) -> RNN输入是(batch_size, seq_len, input_dim)

        # RNN 层输出, output形状: (batch_size, seq_len, hidden_dim)
        x, _ = self.rnn1(x)  # output: (1, 31, 150)
        x, _ = self.rnn2(x)
        x = x.transpose(1, 2)

        return x

# # 测试代码
# import numpy as np
#
# random_array = np.random.rand(1, 19, 50).astype(np.float32)  # 输入数据的形状是 (1, 31, 50)
# input_tensor = torch.from_numpy(random_array)
#
# # 创建模型并进行前向传播
# model = Reservoir_rnn(rhow=2, input_dim=19, hidden_dim=150)
# output = model(input_tensor)
#
# print(output.shape)  # 输出形状应为 (1, 31, 150)
