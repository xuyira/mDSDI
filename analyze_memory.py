#!/usr/bin/env python3
"""
分析EEG Conformer的参数量和显存占用
"""

import torch
import torch.nn as nn
from algorithms.mDSDI.src.models.eeg_conformer import EEG_Conformer_Simple
from algorithms.mDSDI.src.Trainer_mDSDI import Domain_Discriminator, ZS_Domain_Classifier, Classifier


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_size():
    """分析模型大小"""
    print("🔍 EEG Conformer + mDSDI 参数量分析")
    print("=" * 60)
    
    # 1. EEG Conformer参数量
    conformer = EEG_Conformer_Simple()
    conformer_params = count_parameters(conformer)
    print(f"📊 EEG Conformer参数量: {conformer_params:,}")
    
    # 2. mDSDI组件参数量
    feature_dim = 2440
    n_classes = 4
    n_domain_classes = 8
    
    # 域判别器
    domain_discriminator = Domain_Discriminator(feature_dim, n_domain_classes)
    domain_discriminator_params = count_parameters(domain_discriminator)
    print(f"📊 Domain Discriminator参数量: {domain_discriminator_params:,}")
    
    # 域分类器
    zs_domain_classifier = ZS_Domain_Classifier(feature_dim, n_domain_classes)
    zs_domain_classifier_params = count_parameters(zs_domain_classifier)
    print(f"📊 ZS Domain Classifier参数量: {zs_domain_classifier_params:,}")
    
    # 最终分类器
    classifier = Classifier(feature_dim, n_classes)
    classifier_params = count_parameters(classifier)
    print(f"📊 Classifier参数量: {classifier_params:,}")
    
    # 3. 总参数量
    total_params = conformer_params * 2 + domain_discriminator_params + zs_domain_classifier_params + classifier_params
    print(f"\n🎯 总参数量: {total_params:,}")
    
    # 4. 详细分析
    print(f"\n📈 详细分析:")
    print(f"   - 两个Conformer (zi_model + zs_model): {conformer_params * 2:,}")
    print(f"   - Domain Discriminator: {domain_discriminator_params:,}")
    print(f"   - ZS Domain Classifier: {zs_domain_classifier_params:,}")
    print(f"   - Classifier: {classifier_params:,}")
    
    # 5. 显存估算
    print(f"\n💾 显存估算 (假设float32):")
    memory_per_param = 4  # bytes
    total_memory_mb = (total_params * memory_per_param) / (1024 * 1024)
    print(f"   - 模型参数显存: {total_memory_mb:.1f} MB")
    
    # 6. 训练时显存 (包含梯度、优化器状态等)
    training_memory_mb = total_memory_mb * 3  # 参数 + 梯度 + 优化器状态
    print(f"   - 训练时显存 (估算): {training_memory_mb:.1f} MB")
    
    return total_params, total_memory_mb, training_memory_mb


def analyze_conformer_details():
    """详细分析Conformer各组件参数量"""
    print(f"\n🔬 Conformer详细分析:")
    print("=" * 60)
    
    conformer = EEG_Conformer_Simple()
    
    # PatchEmbedding
    patch_params = count_parameters(conformer.patch_embedding)
    print(f"📊 PatchEmbedding: {patch_params:,}")
    
    # TransformerEncoder
    transformer_params = count_parameters(conformer.transformer_encoder)
    print(f"📊 TransformerEncoder: {transformer_params:,}")
    
    # FeatureHead
    feature_head_params = count_parameters(conformer.feature_head)
    print(f"📊 FeatureHead: {feature_head_params:,}")
    
    # 验证
    total_conformer = patch_params + transformer_params + feature_head_params
    print(f"📊 总计: {total_conformer:,}")


def analyze_classifier_details():
    """分析Classifier的参数量"""
    print(f"\n🔬 Classifier详细分析:")
    print("=" * 60)
    
    classifier = Classifier(2440, 4)
    
    # 各层参数量
    layers = classifier.classifier
    total_params = 0
    
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            params = layer.in_features * layer.out_features + layer.out_features
            total_params += params
            print(f"📊 Linear层 {i//3 + 1}: {layer.in_features} → {layer.out_features}, 参数量: {params:,}")
    
    print(f"📊 Classifier总参数量: {total_params:,}")


def analyze_domain_discriminator_details():
    """分析Domain Discriminator的参数量"""
    print(f"\n🔬 Domain Discriminator详细分析:")
    print("=" * 60)
    
    discriminator = Domain_Discriminator(2440, 8)
    
    layers = discriminator.class_classifier
    total_params = 0
    
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            params = layer.in_features * layer.out_features + layer.out_features
            total_params += params
            print(f"📊 Linear层 {i//2 + 1}: {layer.in_features} → {layer.out_features}, 参数量: {params:,}")
    
    print(f"📊 Domain Discriminator总参数量: {total_params:,}")


def suggest_optimizations():
    """建议优化方案"""
    print(f"\n💡 显存优化建议:")
    print("=" * 60)
    
    print("1. 🔧 减少特征维度:")
    print("   - 将feature_dim从2440降到512或256")
    print("   - 在Conformer后添加降维层")
    
    print("\n2. 🔧 减少批次大小:")
    print("   - batch_size: 16 → 8 或 4")
    print("   - 使用梯度累积")
    
    print("\n3. 🔧 使用混合精度训练:")
    print("   - torch.cuda.amp.autocast()")
    print("   - 减少一半显存占用")
    
    print("\n4. 🔧 减少Conformer深度:")
    print("   - depth: 6 → 3 或 4")
    print("   - 减少Transformer层数")
    
    print("\n5. 🔧 使用梯度检查点:")
    print("   - torch.utils.checkpoint")
    print("   - 用计算换显存")


if __name__ == "__main__":
    try:
        total_params, model_memory, training_memory = analyze_model_size()
        analyze_conformer_details()
        analyze_classifier_details()
        analyze_domain_discriminator_details()
        suggest_optimizations()
        
        print(f"\n🎯 总结:")
        print(f"   总参数量: {total_params:,}")
        print(f"   模型显存: {model_memory:.1f} MB")
        print(f"   训练显存: {training_memory:.1f} MB")
        
        if training_memory > 8000:  # 8GB
            print(f"\n⚠️  警告: 显存占用可能超过8GB!")
            print(f"   建议使用优化方案减少显存占用")
        
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
