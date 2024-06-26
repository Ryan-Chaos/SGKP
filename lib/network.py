from typing import List, Tuple

from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Encoding of geometric properties using MLP """
    def __init__(self, keypoint_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([keypoint_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, kpts):
        if self.use_dropout:
            return self.dropout(self.encoder(kpts))
        return self.encoder(kpts)


class DescriptorEncoder(nn.Module):
    """ Encoding of visual descriptor using MLP """
    def __init__(self, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.encoder = MLP([feature_dim] + layers + [feature_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, descs):
        residual = descs
        if self.use_dropout:
            return residual + self.dropout(self.encoder(descs))
        return residual + self.encoder(descs)

# AFT-Simple
class AFTAttention(nn.Module):
    """ Attention-free attention """
    def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.dim = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = torch.sigmoid(q)
        k = k.T
        k = torch.softmax(k, dim=-1)
        k = k.T
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv
        x = self.proj(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    
# MHA
def ScaledDotProductAttention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[2]
    scores = torch.einsum('hid, hjd -> hij', query, key) / (dim**.5)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('hij, hjd->hid', prob, value), prob


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-head self attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model * 3)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        qkv = self.proj(x)
        query, key, value = tuple(rearrange(qkv, 'n (d k h) -> k h n d ', k=3, h=self.num_heads))
        x, _ = ScaledDotProductAttention(query, key, value)
        x = rearrange(x, "h n d -> n (h d)")
        x = self.merge(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
        self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class AttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, use_mha: bool = False, dropout: bool = False, p: float = 0.1):
        super().__init__()
        if use_mha:
            self.attn = MultiHeadedSelfAttention(4, feature_dim)
        else:
            self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
        self.ffn = PositionwiseFeedForward(feature_dim, dropout=dropout, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, layer_num: int, use_mha: bool = False, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalLayer(feature_dim, use_mha=use_mha, dropout=dropout, p=p)
            for _ in range(layer_num)])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            desc = layer(desc)
        return desc


class FeatureBooster(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128],
        'Attentional_layers': 3,
        'last_activation': 'relu',
        'l2_normalization': True,
        'output_dim': 128
    }

    def __init__(self, config, dropout=False, p=0.1, use_kenc=True, use_cross=True, use_mha=False):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.use_kenc = use_kenc
        self.use_cross = use_cross

        if use_kenc:
            self.kenc = KeypointEncoder( # 关键点编码器
                self.config['keypoint_dim'], self.config['descriptor_dim'], self.config['keypoint_encoder'], dropout=dropout)

        if self.config.get('descriptor_encoder', False):
            self.denc = DescriptorEncoder( # 描述符编码器
                self.config['descriptor_dim'], self.config['descriptor_encoder'], dropout=dropout)
        else:
            self.denc = None

        if self.use_cross:
            self.attn_proj = AttentionalNN( # 注意力网络
                feature_dim=self.config['descriptor_dim'], layer_num=self.config['Attentional_layers'], use_mha=use_mha, dropout=dropout)

        self.final_proj = nn.Linear( # 最终的投影层
            self.config['descriptor_dim'], self.config['output_dim'])

        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

        self.layer_norm = nn.LayerNorm(self.config['descriptor_dim'], eps=1e-6)

        if self.config.get('last_activation', False):
            if self.config['last_activation'].lower() == 'relu':
                self.last_activation = nn.ReLU()
            elif self.config['last_activation'].lower() == 'sigmoid':
                self.last_activation = nn.Sigmoid()
            elif self.config['last_activation'].lower() == 'tanh':
                self.last_activation = nn.Tanh()
            else:
                raise Exception('Not supported activation "%s".' % self.config['last_activation'])
        else:
            self.last_activation = None

    def forward(self, desc, kpts):
        ## Self boosting
        # Descriptor MLP encoder
        """
        首先，进行自我增强（self-boosting）：
        如果存在描述符编码器，将描述符输入到描述符编码器中。
        如果使用关键点编码器，将关键点特征与描述符相加，并应用可选的dropout。
        """
        if self.denc is not None:
            desc = self.denc(desc)
        # Geometric MLP encoder
        if self.use_kenc:
            desc = desc + self.kenc(kpts)
            if self.use_dropout:
                desc = self.dropout(desc)
        
        ## Cross boosting
        # Multi-layer Transformer network.
        """
        然后，进行交叉增强（cross-boosting）：
        如果使用交叉增强，将描述符输入到注意力网络中
        """
        if self.use_cross:
            desc = self.attn_proj(self.layer_norm(desc))

        ## Post processing
        # Final MLP projection
        """
        最后，进行后处理：
        将描述符通过最终的投影层进行线性变换。
        如果指定了激活函数，应用相应的激活函数。
        如果启用了L2归一化，对描述符进行L2归一化
        """
        desc = self.final_proj(desc)
        if self.last_activation is not None:
            desc = self.last_activation(desc)
        # L2 normalization
        if self.config['l2_normalization']:
            desc = F.normalize(desc, dim=-1)

        return desc