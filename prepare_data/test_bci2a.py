#!/usr/bin/env python3
"""
测试BCI2a数据集是否正确设置
"""

import os
import numpy as np
import torch
from algorithms.mDSDI.src.dataloaders.BCI2a_Dataloader import BCI2a_Train_Dataset, BCI2a_Test_Dataset
from algorithms.mDSDI.src.models.resnet import ResNet


def test_data_structure():
    """测试数据结构是否正确"""
    print("=" * 80)
    print("测试1: 检查数据结构")
    print("=" * 80)
    
    # 检查目录结构
    data_root = "../data/BCI2a/"
    raw_images_dir = os.path.join(data_root, "Raw images")
    splits_dir = os.path.join(data_root, "Train val splits")
    
    if not os.path.exists(raw_images_dir):
        print(f"❌ 未找到目录: {raw_images_dir}")
        print(f"   请先运行: python prepare_bci2a_for_mDSDI.py")
        return False
    
    if not os.path.exists(splits_dir):
        print(f"❌ 未找到目录: {splits_dir}")
        return False
    
    print(f"✅ 目录结构正确")
    
    # 检查每个受试者的数据
    for subj in range(1, 10):
        subj_dir = os.path.join(raw_images_dir, f"sub{subj}")
        meta_file = os.path.join(splits_dir, f"sub{subj}.txt")
        
        if os.path.exists(subj_dir):
            npy_files = [f for f in os.listdir(subj_dir) if f.endswith('.npy')]
            print(f"  ✅ sub{subj}: {len(npy_files)} npy files")
        else:
            print(f"  ❌ sub{subj}: 目录不存在")
            
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                lines = f.readlines()
            print(f"  ✅ sub{subj}.txt: {len(lines)} 条记录")
        else:
            print(f"  ❌ sub{subj}.txt: 文件不存在")
    
    return True


def test_dataloader():
    """测试数据加载器"""
    print("\n" + "=" * 80)
    print("测试2: 数据加载器")
    print("=" * 80)
    
    # 读取一个元数据文件
    meta_file = "../data/BCI2a/Train val splits/sub1.txt"
    if not os.path.exists(meta_file):
        print(f"❌ 未找到元数据文件: {meta_file}")
        return False
    
    # 解析元数据
    sample_paths = []
    class_labels = []
    with open(meta_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sample_paths.append(parts[0])
                class_labels.append(int(parts[1]))
    
    print(f"✅ 读取元数据: {len(sample_paths)} 个样本")
    
    # 创建数据集
    dataset = BCI2a_Train_Dataset(
        src_path="../data/BCI2a/Raw images/",
        sample_paths=sample_paths[:10],  # 只测试前10个
        class_labels=class_labels[:10],
        domain_label=0
    )
    
    print(f"✅ 创建数据集: {len(dataset)} 个样本")
    
    # 加载一个样本
    img, label, domain = dataset[0]
    print(f"✅ 加载样本:")
    print(f"   - 图像形状: {img.shape}")
    print(f"   - 图像类型: {img.dtype}")
    print(f"   - 图像范围: [{img.min():.4f}, {img.max():.4f}]")
    print(f"   - 类别标签: {label}")
    print(f"   - 域标签: {domain}")
    
    # 检查形状
    if img.shape == torch.Size([22, 64, 64]):
        print(f"✅ 图像形状正确: (22, 64, 64)")
    else:
        print(f"❌ 图像形状错误: 期望 (22, 64, 64), 实际 {img.shape}")
        return False
    
    return True


def test_model():
    """测试模型"""
    print("\n" + "=" * 80)
    print("测试3: 模型")
    print("=" * 80)
    
    # 创建模型
    model = ResNet(input_channels=22)
    print(f"✅ 创建模型: ResNet18_Custom")
    print(f"   - 输出维度: {model.n_outputs}")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 22, 64, 64)  # batch_size=2
    try:
        output = model(dummy_input)
        print(f"✅ 前向传播成功:")
        print(f"   - 输入形状: {dummy_input.shape}")
        print(f"   - 输出形状: {output.shape}")
        
        if output.shape == torch.Size([2, 2048]):
            print(f"✅ 输出形状正确: (batch_size, 2048)")
        else:
            print(f"❌ 输出形状错误: 期望 (2, 2048), 实际 {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    return True


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 80)
    print("测试4: 配置文件")
    print("=" * 80)
    
    config_file = "../algorithms/mDSDI/configs/BCI2a_sub1.json"
    if not os.path.exists(config_file):
        print(f"❌ 未找到配置文件: {config_file}")
        return False
    
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"✅ 读取配置文件")
    print(f"   - 实验名称: {config['exp_name']}")
    print(f"   - 模型: {config['model']}")
    print(f"   - 特征维度: {config['feature_dim']}")
    print(f"   - 数据集: {config['dataset']}")
    print(f"   - 类别数: {config['n_classes']}")
    print(f"   - 域数: {config['n_domain_classes']}")
    print(f"   - 训练域数: {len(config['src_train_meta_filenames'])}")
    print(f"   - 测试域: {config['target_test_meta_filenames']}")
    
    # 检查配置
    if config['n_classes'] != 4:
        print(f"⚠️  警告: BCI2a有4个类别，当前配置为 {config['n_classes']}")
    
    if config['n_domain_classes'] != len(config['src_train_meta_filenames']):
        print(f"⚠️  警告: 域数量不匹配")
    
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("BCI2a数据集设置测试")
    print("=" * 80 + "\n")
    
    tests = [
        ("数据结构", test_data_structure),
        ("数据加载器", test_dataloader),
        ("模型", test_model),
        ("配置文件", test_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 出错: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练了。")
        print("\n运行训练命令:")
        print("  python main.py --config algorithms/mDSDI/configs/BCI2a_sub1.json --exp_idx 1 --gpu_idx 0")
    else:
        print("\n⚠️  部分测试失败，请检查上面的错误信息。")


if __name__ == '__main__':
    main()

