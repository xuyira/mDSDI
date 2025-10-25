"""
BCI2a数据集的数据加载器
用于加载npy格式的EEG图像数据
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class BCI2a_Train_Dataset(Dataset):
    """BCI2a训练数据集"""
    
    def __init__(self, src_path, sample_paths, class_labels, domain_label):
        """
        参数:
            src_path: 数据根目录
            sample_paths: 样本路径列表（相对路径）
            class_labels: 类别标签列表
            domain_label: 域标签（受试者编号）
        """
        self.src_path = src_path
        self.sample_paths = sample_paths.tolist() if hasattr(sample_paths, 'tolist') else list(sample_paths)
        self.class_labels = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)
        self.domain_label = domain_label
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        # 加载npy文件
        npy_path = self.src_path + self.sample_paths[idx]
        
        # 加载图像数据 (22, 64, 64)
        img = np.load(npy_path).astype(np.float32)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img)
        
        # 获取标签
        class_label = self.class_labels[idx]
        domain_label = self.domain_label
        
        return img_tensor, class_label, domain_label


class BCI2a_Test_Dataset(Dataset):
    """BCI2a测试数据集"""
    
    def __init__(self, src_path, sample_paths, class_labels):
        """
        参数:
            src_path: 数据根目录
            sample_paths: 样本路径列表（相对路径）
            class_labels: 类别标签列表
        """
        self.src_path = src_path
        self.sample_paths = sample_paths if isinstance(sample_paths, list) else list(sample_paths)
        self.class_labels = class_labels if isinstance(class_labels, list) else list(class_labels)
        
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        # 加载npy文件
        npy_path = self.src_path + self.sample_paths[idx]
        
        # 加载图像数据 (22, 64, 64)
        img = np.load(npy_path).astype(np.float32)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img)
        
        # 获取标签
        class_label = self.class_labels[idx]
        domain_label = -1  # 测试集域标签设为-1
        
        return img_tensor, class_label, domain_label

