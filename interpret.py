# interpret_advanced.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 引入模块
from utils.config import get_config
from data.dataset import NATGraphDataset, custom_collate
from models.network import HGTs_NAT

# ================= ⚙️ 配置 (顶刊配色) =================
# 0:黑, 1:红(Tumor), 2:绿(Immune), 3:蓝, 4:黄, 5:橙
NODE_COLORS = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FFA500']
TYPE_NAMES = ["Background", "Tumor", "Immune", "Stroma", "Necrosis", "Other"]
LABEL_MAP = {1: "pCR (Responder)", 0: "Non-pCR (Resistant)"}


# ================= 1. 深度挖掘引擎 =================
def mine_advanced_data(loader, model, device):
    slide_stats = []

    # 全局交互矩阵 (Group -> 6x6)
    global_attn = {k: np.zeros((6, 6)) for k in LABEL_MAP.values()}
    global_counts = {k: 0 for k in LABEL_MAP.values()}

    # 最佳子图存储
    best_subgraph = {
        k: {"score": -1, "graph": None, "pos": None, "types": None, "id": ""}
        for k in LABEL_MAP.values()
    }

    print("🚀 Starting Advanced Mining (Interactome / Spatial / Motifs)...")

    with torch.no_grad():
        for i, (g, l, sid) in enumerate(tqdm(loader)):
            if not g: continue

            true_label = l.item()
            group_name = LABEL_MAP.get(true_label, "Unknown")
            slide_id = sid[0]

            # 🔥 调用模型挖掘接口 (forward_interpret)
            # models/network.py 必须实现此接口
            logits, results = model.forward_interpret([x.to(device) for x in g[0]])

            # --- 单个病人的统计 ---
            slide_ti_dists = []  # Tumor-Immune 距离
            slide_matrix = np.zeros((6, 6))

            for p_idx, patch in enumerate(results):
                att = patch['att_weights']  # (N,)
                types = patch['cell_types']  # (N,)
                coords = patch['coords']  # (N, 2) or None
                # 注意：network.py 的 forward_interpret 需返回 edge_index
                # 如果没返回，这里只能跳过拓扑分析
                edge_index = patch.get('edge_index', np.zeros((2, 0)))

                # === A. 交互矩阵 (Interactome) ===
                # 使用边连接的节点类型统计，以 Attention 为权重
                if edge_index.shape[1] > 0:
                    src, dst = edge_index[0], edge_index[1]
                    # 简单策略：统计每条边的两端类型，权重为 dst 的 attention
                    # (假设 dst 是信息接收者)
                    for s, d, w in zip(src, dst, att[dst]):
                        st, dt = types[s], types[d]
                        if st < 6 and dt < 6:
                            slide_matrix[st, dt] += w

                # === B. 空间拓扑 (Spatial) ===
                if coords is not None:
                    t_mask = (types == 1)
                    i_mask = (types == 2)
                    if t_mask.any() and i_mask.any():
                        from scipy.spatial.distance import cdist
                        dists = cdist(coords[t_mask], coords[i_mask])
                        min_dists = dists.min(axis=1)  # 每个肿瘤细胞最近的免疫细胞距离
                        slide_ti_dists.extend(min_dists)

                # === C. 关键 Motif 捕获 ===
                # 寻找包含 Tumor+Immune 且 Attention 最高的 Patch
                max_att = np.max(att) if len(att) > 0 else 0
                has_t = (types == 1).any()
                has_i = (types == 2).any()

                if has_t and has_i and (max_att > best_subgraph[group_name]["score"]):
                    if edge_index.shape[1] > 20:  # 过滤太小的图
                        best_subgraph[group_name] = {
                            "score": max_att,
                            "graph": (edge_index, types, att),
                            "pos": coords,
                            "id": f"{slide_id}_P{p_idx}"
                        }

            # --- 汇总 ---
            if slide_matrix.sum() > 0:
                norm_mat = slide_matrix / slide_matrix.sum()
                global_attn[group_name] += norm_mat
                global_counts[group_name] += 1

            avg_dist = np.mean(slide_ti_dists) if slide_ti_dists else np.nan
            slide_stats.append({
                "ID": slide_id, "Group": group_name, "Avg_TI_Dist": avg_dist
            })

    # 平均化全局矩阵
    for k in global_attn:
        if global_counts[k] > 0: global_attn[k] /= global_counts[k]

    return pd.DataFrame(slide_stats), global_attn, best_subgraph


# ================= 2. 可视化绘图 =================
def plot_results(output_dir, df, matrices, best_subgraphs):
    # 1. 交互热力图
    print("🎨 Plotting Heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (grp, mat) in enumerate(matrices.items()):
        sns.heatmap(mat, ax=axes[i], annot=True, fmt=".3f", cmap="magma",
                    xticklabels=TYPE_NAMES, yticklabels=TYPE_NAMES)
        axes[i].set_title(f"{grp} Interactome")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Interactome_Heatmap.png"))
    plt.close()

    # 2. 空间距离小提琴图
    print("🎨 Plotting Spatial Stats...")
    plt.figure(figsize=(6, 6))
    clean_df = df.dropna()
    if len(clean_df) > 0:
        sns.violinplot(x='Group', y='Avg_TI_Dist', data=clean_df, inner="quartile", palette="Pastel1")
        sns.stripplot(x='Group', y='Avg_TI_Dist', data=clean_df, color='k', alpha=0.5, size=3)

        # 统计检验
        g1 = clean_df[clean_df['Group'] == LABEL_MAP[1]]['Avg_TI_Dist']
        g2 = clean_df[clean_df['Group'] == LABEL_MAP[0]]['Avg_TI_Dist']
        if len(g1) > 0 and len(g2) > 0:
            _, p = mannwhitneyu(g1, g2)
            plt.title(f"Tumor-Immune Proximity (P={p:.4e})")
        plt.savefig(os.path.join(output_dir, "Spatial_Distance.png"))
    plt.close()

    # 3. Motif 可视化 (NetworkX)
    print("🎨 Plotting Motifs...")
    for grp, data in best_subgraphs.items():
        if data["graph"] is None: continue
        edge_index, types, att = data["graph"]
        pos = data["pos"]

        G = nx.Graph()
        colors, sizes = [], []

        for i in range(len(types)):
            # 翻转Y轴适配图像
            G.add_node(i, pos=(pos[i][0], -pos[i][1]))
            t_idx = int(types[i])
            if t_idx >= 6: t_idx = 5
            colors.append(NODE_COLORS[t_idx])
            sizes.append(30 + att[i] * 1000)  # Attention越大点越大

        src, dst = edge_index
        for i in range(len(src)):
            G.add_edge(src[i], dst[i])

        plt.figure(figsize=(8, 8))
        nx_pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(G, nx_pos, alpha=0.2, edge_color='gray')
        nx.draw_networkx_nodes(G, nx_pos, node_size=sizes, node_color=colors, edgecolors='w')

        # 图例
        legends = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=n, markersize=10)
                   for c, n in zip(NODE_COLORS[1:], TYPE_NAMES[1:])]
        plt.legend(handles=legends, title="Cell Types")
        plt.title(f"{grp} Motif\n{data['id']}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"Motif_{grp.split()[0]}.png"))
        plt.close()


# ================= 3. 主入口 =================
def main():
    cfg = get_config(description="Advanced Interpretation")
    device = cfg['project']['device']
    out_dir = os.path.join(cfg['paths']['output_dir'], "advanced_mining")
    os.makedirs(out_dir, exist_ok=True)

    # 加载
    model_path = os.path.join(cfg['paths']['output_dir'], "best_model.pth")
    print(f"📥 Loading: {model_path}")

    ds = NATGraphDataset(cfg['paths']['graph_dir'], cfg['paths']['label_file'], cfg['data']['max_patches'], mode='val')
    # 为了演示，只取前 50 个做深度挖掘，全量跑可以去掉切片
    # ds.slide_ids = ds.slide_ids[:50]
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

    model = HGTs_NAT(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 挖掘 & 绘图
    df, matrices, best_graphs = mine_advanced_data(loader, model, device)

    # 保存数据
    df.to_csv(os.path.join(out_dir, "spatial_stats.csv"), index=False)
    plot_results(out_dir, df, matrices, best_graphs)

    print(f"✅ All Done! Check folder: {out_dir}")


if __name__ == '__main__':
    main()