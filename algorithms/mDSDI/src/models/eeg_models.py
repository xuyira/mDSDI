import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_CNN(nn.Module):
    """专门为EEG数据设计的轻量级CNN"""
    
    def __init__(self, input_channels=22, num_classes=4):
        super(EEG_CNN, self).__init__()
        self.n_outputs = 128  # 大幅减少特征维度
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc2 = nn.Linear(256, 128)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 22x64x64 -> 32x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32x32 -> 64x16x16
        x = self.pool(F.relu(self.conv3(x)))  # 64x16x16 -> 128x8x8
        
        # 展平
        x = x.view(-1, 128 * 8 * 8)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return x


class EEG_ResNet(nn.Module):
    """轻量级ResNet，专门为EEG设计"""
    
    def __init__(self, input_channels=22):
        super(EEG_ResNet, self).__init__()
        self.n_outputs = 256
        
        # 初始卷积
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 第一个块处理维度变化
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        
        return x


class BasicBlock(nn.Module):
    """ResNet基础块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out
