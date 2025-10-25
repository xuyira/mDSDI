#!/usr/bin/env python3
"""
将BCI2a数据集转换为mDSDI格式
- 每个受试者作为一个域
- 每个trial作为一张图片
- 使用时间延迟嵌入将EEG信号转换为图像
"""

import os
import numpy as np
import scipy.io
import torch
import torch as th 

class DelayEmbedder:
    """Delay embedding transformation"""
    
    def __init__(self, device, seq_len, delay, embedding):
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None
    
    def pad_to_square(self, x, mask=0):
        """Pads the input tensor x to make it square along the last two dimensions."""
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (0, max_side - rows, 0, max_side - cols)
        x_padded = th.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded
    
    def ts_to_img(self, signal, pad=True, mask=0):
        """
        将时间序列转换为图像
        Args:
            signal: (batch, length, features) - EEG数据
            pad: 是否填充到正方形
        Returns:
            x_image: (batch, features, H, W)
        """
        batch, length, features = signal.shape
        if self.seq_len != length:
            self.seq_len = length
        
        x_image = th.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1
        
        # 处理剩余部分
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1
        
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]
        
        if pad:
            x_image = self.pad_to_square(x_image, mask)
        
        return x_image

def prepare_bci2a_for_mDSDI(
    data_root='./data/bci2a/standard_2a_data/',
    output_root='./data/BCI2a/',
    delay=15,
    embedding=64,
    device='cuda'
):
    """
    将BCI2a数据集转换为mDSDI格式
    
    参数:
        data_root: BCI2a原始数据路径
        output_root: 输出根目录
        delay: 时间延迟参数
        embedding: 嵌入维度
        device: 计算设备
    """
    
    print('=' * 80)
    print('BCI2a数据集转换为mDSDI格式')
    print('=' * 80)
    
    # 检查设备
    if device == 'cuda' and not torch.cuda.is_available():
        print('⚠️  CUDA不可用，使用CPU')
        device = 'cpu'
    
    # 创建输出目录
    raw_images_dir = os.path.join(output_root, 'Raw images')
    splits_dir = os.path.join(output_root, 'Train val splits')
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    num_subjects = 9
    num_classes = 4
    
    # 遍历每个受试者
    for subj in range(1, num_subjects + 1):
        print(f'\n处理受试者 {subj}/9...')
        
        # ========== Step 1: 加载数据 ==========
        train_file = os.path.join(data_root, f'A0{subj}T.mat')
        test_file = os.path.join(data_root, f'A0{subj}E.mat')
        
        if not os.path.exists(train_file):
            print(f'  ⚠️  找不到文件: {train_file}')
            continue
        if not os.path.exists(test_file):
            print(f'  ⚠️  找不到文件: {test_file}')
            continue
        
        # 加载训练数据 (T - Training)
        train_mat = scipy.io.loadmat(train_file)
        data_T = train_mat['data']     # (time_points, channels, trials) = (1000, 22, 288)
        label_T = train_mat['label']   # (trials, 1) = (288, 1)
        
        # 加载测试数据 (E - Evaluation)
        test_mat = scipy.io.loadmat(test_file)
        data_E = test_mat['data']      # (time_points, channels, trials) = (1000, 22, 288)
        label_E = test_mat['label']    # (trials, 1) = (288, 1)
        
        print(f'  原始数据形状: T={data_T.shape}, E={data_E.shape}')
        
        # ========== Step 2: 数据预处理 ==========
        # 转换维度: (time_points, channels, trials) -> (trials, time_points, channels)
        data_T = np.transpose(data_T, (2, 0, 1))  # (288, 1000, 22)
        label_T = label_T.flatten()                # (288,)
        
        data_E = np.transpose(data_E, (2, 0, 1))  # (288, 1000, 22)
        label_E = label_E.flatten()                # (288,)
        
        # 合并训练和测试数据
        all_data = np.concatenate([data_T, data_E], axis=0)  # (576, 1000, 22)
        all_labels = np.concatenate([label_T, label_E], axis=0)  # (576,)
        
        # 标签从 1-4 转换为 0-3
        all_labels = all_labels - 1
        
        print(f'  合并后数据: {all_data.shape}, 标签: {all_labels.shape}')
        
        # ========== Step 3: 归一化 [-1, 1] ==========
        data_min = np.min(all_data)
        data_max = np.max(all_data)
        all_data = 2 * (all_data - data_min) / (data_max - data_min) - 1
        
        print(f'  归一化范围: [{data_min:.4f}, {data_max:.4f}] -> [{all_data.min():.4f}, {all_data.max():.4f}]')
        
        # ========== Step 4: 转换为图像 ==========
        # 转换为 torch tensor
        data_tensor = torch.from_numpy(all_data).float()  # (576, 1000, 22)
        
        # 获取序列长度
        seq_len = all_data.shape[1]  # 1000
        
        # 创建 DelayEmbedder
        embedder = DelayEmbedder(
            device=device,
            seq_len=seq_len,
            delay=delay,
            embedding=embedding
        )
        
        # 转换为图像
        data_tensor = data_tensor.to(device)
        img_data = embedder.ts_to_img(data_tensor, pad=True, mask=0)  # (576, 22, 64, 64)
        
        print(f'  图像数据形状: {img_data.shape}')
        
        # ========== Step 5: 保存为npy文件 ==========
        # 创建受试者目录
        subj_dir = os.path.join(raw_images_dir, f'sub{subj}')
        os.makedirs(subj_dir, exist_ok=True)
        
        # 保存每个trial的图像和创建元数据
        meta_lines = []
        
        for trial_idx in range(len(img_data)):
            # 获取当前trial的图像 (22, 64, 64)
            img = img_data[trial_idx].cpu().numpy()
            label = all_labels[trial_idx]
            
            # 保存为npy文件
            npy_filename = f'trial_{trial_idx:04d}.npy'
            npy_path = os.path.join(subj_dir, npy_filename)
            np.save(npy_path, img)
            
            # 添加到元数据
            # 相对路径: sub{subj}/trial_{trial_idx:04d}.npy
            relative_path = f'sub{subj}/{npy_filename}'
            meta_lines.append(f'{relative_path} {label}\n')
        
        print(f'  保存了 {len(img_data)} 个npy文件到: {subj_dir}')
        
        # ========== Step 6: 保存元数据文件 ==========
        meta_file = os.path.join(splits_dir, f'sub{subj}.txt')
        with open(meta_file, 'w') as f:
            f.writelines(meta_lines)
        
        print(f'  保存元数据到: {meta_file}')
        print(f'  ✅ 受试者 {subj} 处理完成! ({len(meta_lines)} 个样本)')
    
    # ========== 完成 ==========
    print('\n' + '=' * 80)
    print('✅ 所有数据转换完成!')
    print('=' * 80)
    print(f'\n生成的文件结构:')
    print(f'  {output_root}')
    print(f'    ├── Raw images/')
    print(f'    │   ├── sub1/ ({576} npy files)')
    print(f'    │   ├── sub2/ ({576} npy files)')
    print(f'    │   └── ...')
    print(f'    └── Train val splits/')
    print(f'        ├── sub1.txt')
    print(f'        ├── sub2.txt')
    print(f'        └── ...')
    print(f'\n每个npy文件格式: (22, 64, 64) - (channels, height, width)')
    print(f'每个txt文件格式: <relative_path> <class_label>')
    print('=' * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将BCI2a数据集转换为mDSDI格式')
    parser.add_argument('--data_root', type=str, 
                       default='../data/standard_2a_data/',
                       help='BCI2a原始数据路径')
    parser.add_argument('--output_root', type=str, 
                       default='../data/BCI2a/',
                       help='输出根目录')
    parser.add_argument('--delay', type=int, default=15,
                       help='时间延迟参数')
    parser.add_argument('--embedding', type=int, default=64,
                       help='嵌入维度')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 执行转换
    prepare_bci2a_for_mDSDI(
        data_root=args.data_root,
        output_root=args.output_root,
        delay=args.delay,
        embedding=args.embedding,
        device=args.device
    )

