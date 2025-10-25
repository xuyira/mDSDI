import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def img_to_ts(image_data):
    """
    将图像数据转换为时间序列格式（使用时间延迟嵌入的逆过程）
    
    参数:
        image_data: torch tensor, shape=(batch, 22, 64, 64)
    
    返回:
        time_series: torch tensor, shape=(batch, 1000, 22)
    """
    batch, channels, rows, cols = image_data.shape
    
    # 参数设置
    seq_len = 1000
    delay = 15
    embedding = 64
    
    # 初始化重建的时间序列
    reconstructed_x_time_series = torch.zeros((batch, channels, seq_len), device=image_data.device)
    
    # 重建时间序列（时间延迟嵌入的逆过程）
    for i in range(cols - 1):
        start = i * delay
        end = start + embedding
        reconstructed_x_time_series[:, :, start:end] = image_data[:, :, :, i]
    
    # 处理最后一列（特殊情况）
    start = (cols - 1) * delay
    end = reconstructed_x_time_series[:, :, start:].shape[-1]
    reconstructed_x_time_series[:, :, start:] = image_data[:, :, :end, cols - 1]
    
    # 转换维度: (batch, channels, seq_len) -> (batch, seq_len, channels)
    reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)
    
    return reconstructed_x_time_series


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),      # 时间维度卷积
            nn.Conv2d(40, 40, (22, 1), (1, 1)),      # 空间维度卷积
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),         # 时间池化
            nn.Dropout(0.5),
        )
        
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 转换为patch序列
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class EEG_Conformer(nn.Module):
    
    def __init__(self, input_channels=22, emb_size=40, depth=6):
        super(EEG_Conformer, self).__init__()
        # 使用原始Conformer的输出维度
        self.n_outputs = 2440  # 原始Conformer的输出维度
        
        # Conformer组件
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        
        # 使用原始的分类头结构
        self.feature_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 2440)  # 保持原始输出维度
        )
        
    def forward(self, x):
        """
        前向传播
        
        输入: [batch_size, channels, height, width] = [batch, 22, 64, 64]
        输出: [batch_size, 2440]
        """
        # 1. 图像转时间序列
        time_series = img_to_ts(x)
        
        # 2. 添加通道维度
        time_series = time_series.permute(0, 2, 1).unsqueeze(1)
        
        # 3. Patch Embedding
        patches = self.patch_embedding(time_series)
        
        # 4. Transformer编码
        encoded_patches = self.transformer_encoder(patches)
        
        # 5. 特征提取
        features = self.feature_head(encoded_patches)
        
        return features
