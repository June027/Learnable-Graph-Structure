# data/dataset.py
import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from .transforms import GraphAugmenter


class NATGraphDataset(Dataset):
    def __init__(self, data_root, label_file, max_patches=100, cfg=None, mode='val'):
        self.data_root = data_root
        self.df = pd.read_csv(label_file)
        self.df['case_id'] = self.df['case_id'].astype(str)
        self.slide_ids = self.df['case_id'].tolist()
        self.labels = self.df['label'].tolist()
        self.max_patches = max_patches
        self.cache_good_files = {}

        # 🔥 修复：增强器初始化
        # 只有传入 cfg 且 mode='train' 且配置文件开启时，才启用增强
        self.augmenter = None
        if mode == 'train' and cfg and cfg['data']['augmentation']['enabled']:
            self.augmenter = GraphAugmenter(cfg)

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]

        # 1. 查找文件 (保持您原有的缓存逻辑)
        if slide_id not in self.cache_good_files:
            search_path = os.path.join(self.data_root, slide_id, "*.pt")
            pt_files = glob.glob(search_path)
            if not pt_files:
                pt_files = glob.glob(os.path.join(self.data_root, "**", slide_id, "*.pt"), recursive=True)

            if len(pt_files) > self.max_patches:
                pt_files = pt_files[:self.max_patches]

            valid = []
            for p in pt_files:
                try:
                    torch.load(p)
                    valid.append(p)
                except:
                    pass
            self.cache_good_files[slide_id] = valid

        # 2. 加载
        graphs = []
        for p in self.cache_good_files[slide_id]:
            try:
                data = torch.load(p)
                if not hasattr(data, 'cell_type'):
                    if data.x.shape[1] >= 6:
                        data.cell_type = torch.argmax(data.x[:, -6:], dim=1)
                    else:
                        data.cell_type = torch.zeros(data.x.shape[0], dtype=torch.long)

                # 🔥 应用增强
                if self.augmenter:
                    data = self.augmenter(data)
                graphs.append(data)
            except:
                pass

        # 🔥 返回 Slide ID，这对于生成 Excel 至关重要
        return graphs, torch.tensor(label, dtype=torch.long), slide_id


def custom_collate(batch):
    """适配返回 Slide ID 的 Collate"""
    batch_graphs, batch_labels, batch_ids = [], [], []
    for graphs, label, slide_id in batch:
        if len(graphs) > 0:
            batch_graphs.append(graphs)
            batch_labels.append(label)
            batch_ids.append(slide_id)

    if not batch_labels:
        return [], torch.tensor([]), []

    return batch_graphs, torch.stack(batch_labels), batch_ids


def check_and_clean_dataset(dataset, log_path):
    if os.path.exists(log_path): return dataset
    print("🏥 Checking dataset health...")
    valid_idx, bad = [], []
    for i in tqdm(range(len(dataset))):
        try:
            g, _, _ = dataset[i]  # 解包3个变量
            if g:
                valid_idx.append(i)
            else:
                bad.append({"id": dataset.slide_ids[i], "err": "Empty"})
        except Exception as e:
            bad.append({"id": dataset.slide_ids[i], "err": str(e)})
    if bad:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        pd.DataFrame(bad).to_csv(log_path, index=False)
        dataset.slide_ids = [dataset.slide_ids[i] for i in valid_idx]
        dataset.labels = [dataset.labels[i] for i in valid_idx]
    return dataset