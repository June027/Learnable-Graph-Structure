import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
import os
import numpy as np

# 引入你的模块
from data.dataset import NATGraphDataset, custom_collate, check_and_clean_dataset
from models.network import HGTs_NAT
from trainer import Trainer
from utils.tools import setup_seed


# ================= 1. 定义参数解析器 =================
def parse_args():
    parser = argparse.ArgumentParser(description='NAT-HGT Training Project')
    parser.add_argument('--cfg', dest='cfg_file', help='配置文件路径', default='configs/full_experiment.yaml', type=str)
    parser.add_argument('opts', help='命令行修改配置', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# ================= 2. 配置加载与合并工具 =================
def merge_config(config, opts):
    if not opts: return config
    if len(opts) % 2 != 0: raise ValueError(f"opts must be key-value pairs: {opts}")
    for k, v in zip(opts[0::2], opts[1::2]):
        keys = k.split('.')
        sub = config
        try:
            for key in keys[:-1]: sub = sub[key]
            orig = sub[keys[-1]]
            if isinstance(orig, bool): v = str(v).lower() in ('true', '1', 'yes')
            elif orig is None:
                try: v = float(v) if '.' in v else int(v)
                except: pass
            else: v = type(orig)(v)
            print(f"⚡️ Override: {k} = {v}")
            sub[keys[-1]] = v
        except Exception as e: print(f"❌ Error merging {k}: {e}")
    return config

def load_config(args):
    try:
        with open(args.cfg_file, 'r', encoding='utf-8') as f: cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config {args.cfg_file} not found"); sys.exit(1)
    return merge_config(cfg, args.opts)


# ================= 3. 主程序 =================
def main():
    # 1. 初始化
    args = parse_args()
    cfg = load_config(args)
    setup_seed(cfg['project']['seed'])

    if cfg['project']['device'].startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        cfg['project']['device'] = 'cpu'

    print(f"\n🚀 Project: {cfg['project']['name']} | Device: {cfg['project']['device']}")
    print(f"🧪 Mode: Attn={cfg['model']['use_attention']} | Edge={cfg['model']['use_edge_weight']} | Freeze={cfg['train'].get('freeze_gnn', False)}")

    # 2. 数据准备
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)
    bad_log_path = f"{cfg['paths']['output_dir']}/bad_cases_record.csv"

    # 2.1 预扫描清洗 (获取干净ID)
    temp_dataset = NATGraphDataset(cfg['paths']['graph_dir'], cfg['paths']['label_file'], cfg['data']['max_patches'])
    if cfg['data'].get('check_files', False):
        temp_dataset = check_and_clean_dataset(temp_dataset, bad_log_path)

    clean_ids = temp_dataset.slide_ids
    clean_labels = temp_dataset.labels
    total_len = len(clean_ids)

    # 2.2 固定划分索引
    indices = torch.randperm(total_len).tolist()
    train_size = int(0.8 * total_len)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"📊 Data Split: Total={total_len} | Train={len(train_indices)} | Val={len(val_indices)}")

    # 2.3 实例化 (🔥 修复核心：Train开启增强，Val关闭增强)
    # Train Dataset (Mode='train')
    train_ds_base = NATGraphDataset(
        cfg['paths']['graph_dir'], cfg['paths']['label_file'], cfg['data']['max_patches'],
        cfg=cfg, mode='train' # 👈 关键：传入配置开启增强
    )
    train_ds_base.slide_ids, train_ds_base.labels = clean_ids, clean_labels

    # Val Dataset (Mode='val')
    val_ds_base = NATGraphDataset(
        cfg['paths']['graph_dir'], cfg['paths']['label_file'], cfg['data']['max_patches'],
        cfg=cfg, mode='val'   # 👈 关键：关闭增强
    )
    val_ds_base.slide_ids, val_ds_base.labels = clean_ids, clean_labels

    # 2.4 子集映射
    train_ds = Subset(train_ds_base, train_indices)
    val_ds = Subset(val_ds_base, val_indices)

    # 3. 动态权重
    train_labels_list = [clean_labels[i] for i in train_indices]
    n_pos = sum(train_labels_list)
    n_neg = len(train_labels_list) - n_pos
    pos_w = n_neg / (n_pos + 1e-5) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_w]).to(cfg['project']['device'])
    print(f"⚖️ Pos Weight: {pos_w:.4f} (Pos={n_pos}, Neg={n_neg})")

    # 4. Loader
    train_loader = DataLoader(train_ds, cfg['train']['batch_size'], shuffle=True,
                              collate_fn=custom_collate, num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_ds, cfg['train']['batch_size'], collate_fn=custom_collate)

    # 5. Model
    model = HGTs_NAT(cfg).to(cfg['project']['device'])

    # 6. Opt (🔥 修复：过滤冻结参数)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # 👈 关键：防止报错
        lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 7. Start
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, cfg)
    trainer.run()

if __name__ == '__main__':
    main()
