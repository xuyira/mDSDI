import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_CNN_Optimized(nn.Module):
    """优化的EEG CNN，进一步提升性能"""
    
    def __init__(self, input_channels=22):
        super(EEG_CNN_Optimized, self).__init__()
        self.n_outputs = 256  # 增加特征维度
        
        # 卷积层 - 使用更深的网络
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc2 = nn.Linear(512, 256)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积 + 批归一化 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 22x64x64 -> 32x32x32
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32x32 -> 64x16x16
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 64x16x16 -> 128x8x8
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 128x8x8 -> 256x4x4
        x = self.dropout1(x)
        
        # 展平
        x = x.view(-1, 256 * 4 * 4)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        return x


class EEG_CNN_Attention(nn.Module):
    """带注意力机制的EEG CNN"""
    
    def __init__(self, input_channels=22):
        super(EEG_CNN_Attention, self).__init__()
        self.n_outputs = 256
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积 + 批归一化 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 展平
        x = x.view(-1, 128 * 8 * 8)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return x
