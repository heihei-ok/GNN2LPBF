import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(Conv, self).__init__()

        # 定义一维卷积层，假设输入数据是二维的 (batch_size, feature)

        self.conv1 = nn.Conv1d(in_channels=feature, out_channels=1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)

        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 将输入数据调整为适合Conv1d的形状 (batch_size, feature, sequence_length)
        x = x.unsqueeze(-1)  # 增加一个维度以匹配Conv1d的输入要求

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 将输出数据调整回适合Linear的形状 (batch_size, feature)
        x = x.squeeze(-1)  # 移除多余的维度

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
