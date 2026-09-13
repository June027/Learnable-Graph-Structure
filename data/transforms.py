# data/transforms.py
import torch
import copy
from torch_geometric.utils import subgraph


class GraphAugmenter:
    """图数据增强器：随机丢弃节点、添加特征噪声"""

    def __init__(self, cfg):
        self.enabled = cfg['data']['augmentation']['enabled']
        self.drop_prob = cfg['data']['augmentation'].get('drop_node_prob', 0.0)
        self.noise_scale = cfg['data']['augmentation'].get('noise_scale', 0.0)

    def __call__(self, data):
        if not self.enabled:
            return data

        # 深拷贝避免修改原始缓存
        data = copy.copy(data)
        num_nodes = data.x.size(0)

        # 1. 随机丢弃节点，并同步重建边索引。
        if self.drop_prob > 0:
            mask = torch.rand(num_nodes, device=data.x.device) > self.drop_prob
            if mask.sum() > 0:  # 保证不全扔掉了
                data.edge_index, data.edge_attr = subgraph(
                    mask,
                    data.edge_index,
                    edge_attr=getattr(data, 'edge_attr', None),
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )
                data.x = data.x[mask]
                data.cell_type = data.cell_type[mask]
                if hasattr(data, 'pos'):
                    data.pos = data.pos[mask]

        # 2. 特征噪声 (Feature Noise)
        if self.noise_scale > 0:
            noise = torch.randn_like(data.x) * self.noise_scale
            data.x = data.x + noise

        return data
