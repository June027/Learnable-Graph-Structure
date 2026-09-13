# models/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import UniversalHeteroLayer, SinePositionalEncoding2D


class HGTs_NAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_feat = cfg['data']['num_features']
        gnn_dim = cfg['model']['gnn_dim']
        n_classes = cfg['data']['num_classes']
        n_types = cfg['model']['num_cell_types']
        self.heads = cfg['model'].get('heads', 1) if cfg['model']['use_attention'] else 1

        # 维度对齐：如果是多头，中间层的维度会变大
        # 策略：保持 hidden_dim 不变，每一层输出后投影回 hidden_dim
        # 或者让 GNN 内部消化。这里我们采用简单的 Linear 投影对齐。
        self.gnn_dim = gnn_dim

        self.input_norm = nn.BatchNorm1d(num_feat)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        # === 构建 GNN ===
        # 第一层
        self.gnn_layers.append(UniversalHeteroLayer(num_feat, gnn_dim, n_types, cfg))
        self.gnn_norms.append(nn.LayerNorm(gnn_dim * self.heads))

        # 投影层 (如果多头，把 features*heads 变回 features)
        self.proj = nn.Linear(gnn_dim * self.heads, gnn_dim)

        self.depth = cfg['model']['gnn_layers'] - 1
        for _ in range(self.depth):
            self.gnn_layers.append(UniversalHeteroLayer(gnn_dim, gnn_dim, n_types, cfg))
            self.gnn_norms.append(nn.LayerNorm(gnn_dim * self.heads))

        # === 冻结 GNN (如果配置要求) ===
        if cfg['train']['freeze_gnn']:
            print("❄️ Freezing GNN Layers (Fixed Structure Mode)...")
            for param in self.gnn_layers.parameters():
                param.requires_grad = False
            for param in self.gnn_norms.parameters():
                param.requires_grad = False
            for param in self.proj.parameters():
                param.requires_grad = False

        # === Attention Pooling ===
        self.att_lin1 = nn.Linear(gnn_dim, gnn_dim)
        self.att_lin2 = nn.Linear(gnn_dim, gnn_dim)
        self.att_lin3 = nn.Linear(gnn_dim, 1)

        # === Transformer ===
        self.pos_encoder = SinePositionalEncoding2D(gnn_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=gnn_dim, nhead=4, dim_feedforward=256,
                                               dropout=cfg['model']['dropout'], batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg['model']['transformer_layers'])

        self.classifier = nn.Sequential(
            nn.Linear(gnn_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_classes)
        )

    def get_attention_scores(self, x):
        return self.att_lin3(torch.tanh(self.att_lin1(x)) * torch.sigmoid(self.att_lin2(x)))

    def _process_gnn(self, x, edge_index, cell_types, pos=None):
        if x.shape[1] > 167: x = x[:, :-6]
        if x.shape[0] > 1: x = self.input_norm(x)

        # 循环 GNN 层
        # 第一层
        x = self.gnn_layers[0](x, edge_index, cell_types, pos)
        x = self.gnn_norms[0](x)
        x = F.relu(x)
        x = self.proj(x)  # 对齐维度

        # 后续层
        for i in range(self.depth):
            x_in = x
            x = self.gnn_layers[i + 1](x, edge_index, cell_types, pos)
            x = self.gnn_norms[i + 1](x)
            x = F.relu(x)
            x = self.proj(x)
            x = x + x_in  # Residual
        return x

    def forward(self, batch_graphs):
        if not batch_graphs:
            raise ValueError("batch_graphs must contain at least one graph")

        patch_embs, patch_coords = [], []
        for data in batch_graphs:
            # 传入 pos 以计算边权重
            x = self._process_gnn(data.x, data.edge_index, data.cell_type, getattr(data, 'pos', None))

            att = F.softmax(self.get_attention_scores(x), dim=0)
            patch_embs.append(torch.sum(x * att, dim=0))

            center = data.pos.mean(dim=0) if hasattr(data, 'pos') else torch.zeros(2, device=x.device)
            patch_coords.append(center)

        seq = torch.stack(patch_embs).unsqueeze(0)
        coords = torch.stack(patch_coords).unsqueeze(0)
        seq = seq + self.pos_encoder(coords).to(seq.dtype)

        feat = torch.mean(self.transformer(seq), dim=1)
        return self.classifier(feat)

    # 🔥🔥🔥 【核心保留】专门用于挖掘的推理函数 🔥🔥🔥
    def forward_interpret(self, batch_graphs):
        """
        运行一次推理，并返回所有详细的中间数据：
        1. patient_risk: 预测的风险值 (Logits)
        2. patch_data_list: 包含每个 Patch 的 [注意力权重, 细胞类型, 坐标, 边连接, 原始形态特征]
        """
        patch_embeddings = []
        patch_coords = []
        interpret_results = []

        for data in batch_graphs:
            x, edge_index, cell_types = data.x, data.edge_index, data.cell_type

            # --- 1. 特征预处理 ---
            if x.shape[1] > 167:
                x_feat = x[:, :-6]
            else:
                x_feat = x

            # ✅ 捕获原始特征用于分析
            raw_features = x_feat.detach().cpu().numpy()

            if x_feat.shape[0] > 1: x_feat = self.input_norm(x_feat)

            # --- 2. GNN 提取特征 ---
            pos = getattr(data, 'pos', None)
            x_gnn = self.gnn_layers[0](x_feat, edge_index, cell_types, pos)
            x_gnn = self.gnn_norms[0](x_gnn)
            x_gnn = F.relu(x_gnn)
            x_gnn = self.proj(x_gnn)
            for i in range(self.depth):
                x_in = x_gnn
                x_gnn = self.gnn_layers[i + 1](x_gnn, edge_index, cell_types, pos)
                x_gnn = self.gnn_norms[i + 1](x_gnn)
                x_gnn = F.relu(x_gnn)
                x_gnn = self.proj(x_gnn)
                x_gnn = x_gnn + x_in

            # --- 3. 获取 Attention (核心挖掘点) ---
            att_scores = self.get_attention_scores(x_gnn)
            att_weights = F.softmax(att_scores, dim=0)

            # --- 4. 聚合 Patch ---
            patch_emb = torch.sum(x_gnn * att_weights, dim=0)
            patch_embeddings.append(patch_emb)

            # 获取 Patch 坐标
            if hasattr(data, 'pos') and data.pos is not None:
                center = data.pos.mean(dim=0)
                raw_coords = data.pos
            else:
                center = torch.zeros(2, device=x.device)
                raw_coords = torch.zeros((x.size(0), 2), device=x.device)
            patch_coords.append(center)

            # 🔥 5. 打包挖掘数据
            interpret_results.append({
                "att_weights": att_weights.detach().cpu().numpy().flatten(),
                "cell_types": cell_types.detach().cpu().numpy(),
                "coords": raw_coords.detach().cpu().numpy(),
                "edge_index": edge_index.detach().cpu().numpy(),
                "node_features": x_gnn.detach().cpu().numpy(),
                "raw_features": raw_features
            })

        # --- 6. Transformer 全局预测 ---
        if len(patch_embeddings) == 0:
            return None, []

        seq_emb = torch.stack(patch_embeddings).unsqueeze(0)
        seq_coords = torch.stack(patch_coords).unsqueeze(0)
        pos_encoding = self.pos_encoder(seq_coords)
        seq_input = seq_emb + pos_encoding.to(seq_emb.dtype)

        global_features = self.transformer(seq_input)
        patient_feature = torch.mean(global_features, dim=1)
        logits = self.classifier(patient_feature)

        return logits, interpret_results
