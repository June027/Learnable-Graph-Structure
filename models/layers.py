# models/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


# === 位置编码 (不变) ===
class SinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, coords):
        coords = torch.clamp(coords.long(), 0, self.pe.size(0) - 1)
        return self.pe[coords[:, :, 0]] + self.pe[coords[:, :, 1]]


# === 🔥 万能异构层 (Universal Hetero Layer) ===
class UniversalHeteroLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_cell_types=6, cfg=None):
        # 如果使用注意力，aggr 设为 None (手动处理)；否则使用 'add'
        use_att = cfg['model']['use_attention']
        super().__init__(aggr=None if use_att else 'add', node_dim=0)

        self.use_att = use_att
        self.use_edge = cfg['model']['use_edge_weight']
        self.heads = cfg['model']['heads'] if use_att else 1
        self.out_channels = out_channels

        # 1. 异构投影 (针对不同细胞类型)
        self.cell_type_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels * self.heads) for _ in range(num_cell_types)
        ])

        # 2. 注意力参数 (仅当 use_attention=True 时初始化)
        if self.use_att:
            # 这里的 Attention 维度要考虑是否加入边特征
            # Node(Hi) + Node(Hj) = 2 * Out
            # If Edge: + Edge_Dim (假设距离是1维)
            att_input_dim = out_channels
            self.att = nn.Parameter(torch.Tensor(1, self.heads, att_input_dim))

            if self.use_edge:
                self.edge_encoder = nn.Linear(1, out_channels * self.heads)  # 把距离编码

        # 3. 输出映射
        self.root_lin = nn.Linear(in_channels, out_channels * self.heads)
        self.bias = nn.Parameter(torch.Tensor(out_channels * self.heads))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.cell_type_linears: lin.reset_parameters()
        self.root_lin.reset_parameters()
        if self.use_att:
            nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, cell_types, pos=None):
        # A. 异构投影
        x_proj = torch.zeros((x.size(0), self.out_channels * self.heads), device=x.device)
        for i, linear in enumerate(self.cell_type_linears):
            mask = (cell_types == i)
            if mask.sum() > 0:
                x_proj[mask] = linear(x[mask]).to(x_proj.dtype)

        # B. 准备边属性 (如果需要)
        edge_attr = None
        if self.use_edge and pos is not None:
            # 实时计算边距离作为权重
            row, col = edge_index
            dist = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
            # 归一化距离或者取倒数 (距离越近权重越大)
            edge_attr = 1.0 / (dist + 1e-5)

        # C. 消息传递
        if self.use_att:
            # Attention 模式
            x_view = x_proj.view(-1, self.heads, self.out_channels)
            out = self.propagate(edge_index, x=x_view, edge_attr=edge_attr)
        else:
            # Sum 模式 (GCN风格)
            # 如果开启了边权重但没开注意力，则做 加权求和
            out = self.propagate(edge_index, x=x_proj, edge_attr=edge_attr)

        # D. 残差连接
        res = self.root_lin(x)
        out = out.view(-1, self.out_channels * self.heads) + self.bias
        return F.relu(out + res)

    def message(self, x_j, x_i=None, edge_attr=None, index=None, ptr=None, size_i=None):
        # === 情况 1: 使用注意力 ===
        if self.use_att:
            # x_j, x_i: [E, Heads, Dim]
            # edge_attr: [E, 1]

            # 基础分数: Node i + Node j
            score_input = x_i + x_j

            # 如果有边权重，融合进去 (EGAT逻辑)
            if self.use_edge and edge_attr is not None:
                edge_emb = self.edge_encoder(edge_attr).view(-1, self.heads, self.out_channels)
                score_input = score_input + edge_emb

            # 计算 Attention Score
            alpha = (score_input * self.att).sum(dim=-1)  # [E, Heads]
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, index, ptr, size_i)

            return x_j * alpha.unsqueeze(-1)

        # === 情况 2: 不使用注意力 (GCN) ===
        else:
            # 如果有边权重，直接乘 (Weighted Sum)
            if self.use_edge and edge_attr is not None:
                return x_j * edge_attr
            # 否则直接返回 (Standard Sum)
            return x_j