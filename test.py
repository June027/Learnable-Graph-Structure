# test.py
import torch
import pandas as pd
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import get_config
from data.dataset import NATGraphDataset, custom_collate
from models.network import HGTs_NAT
from utils.metrics import calculate_metrics


def main():
    # 1. 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/full_experiment.yaml')
    parser.add_argument('--ckpt', required=True, help='Path to best_model.pth')
    parser.add_argument('--data_dir', required=True, help='Path to external test data (graphs)')
    parser.add_argument('--label_file', required=True, help='Path to external labels csv')
    args = parser.parse_args()

    # 2. 加载配置
    # 注意：我们复用训练的 config 来构建模型架构，但数据路径覆盖掉
    class Args:
        pass

    dummy_args = Args();
    dummy_args.cfg_file = args.cfg;
    dummy_args.opts = []

    # 手动加载 Config
    import yaml
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    device = cfg['project']['device']
    print(f"🚀 External Testing on {args.data_dir}")
    print(f"📥 Loading Model: {args.ckpt}")

    # 3. 准备数据
    ds = NATGraphDataset(args.data_dir, args.label_file, cfg['data']['max_patches'], mode='val')
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # 4. 加载模型
    model = HGTs_NAT(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    # 兼容处理
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 5. 推理
    probs, labs, ids = [], [], []
    with torch.no_grad():
        for g, l, sid in tqdm(loader):
            if not g: continue
            logits = model([x.to(device) for x in g[0]])
            p = torch.sigmoid(logits).item()
            probs.append(p)
            labs.append(l.item())
            ids.append(sid[0])

    # 6. 计算指标
    acc, auc, spe, sen = calculate_metrics(labs, probs)
    print(f"\n📊 External Test Results:")
    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"SEN: {sen:.4f}")
    print(f"SPE: {spe:.4f}")

    # 7. 保存结果
    df = pd.DataFrame({
        "Slide_ID": ids,
        "True_Label": labs,
        "Probability": probs,
        "Prediction": [1 if p > 0.5 else 0 for p in probs]
    })
    save_path = os.path.join(os.path.dirname(args.ckpt), "external_test_results.xlsx")
    df.to_excel(save_path, index=False)
    print(f"💾 Results saved to: {save_path}")


if __name__ == '__main__':
    main()