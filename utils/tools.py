import torch
import numpy as np
import random
import os
import glob

def setup_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CheckpointManager:
    def __init__(self, save_dir, max_keep=5):
        self.dir = os.path.join(save_dir, "epoch_checkpoints")
        self.max_keep = max_keep
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, epoch, auc, score):
        path = os.path.join(self.dir, f"epoch_{epoch:03d}_auc_{auc:.4f}_score_{score:.4f}.pth")
        torch.save(model.state_dict(), path)
        files = glob.glob(os.path.join(self.dir, "epoch_*.pth"))
        files.sort(key=lambda x: float(x.split('_score_')[-1].replace('.pth','')), reverse=True)
        for f in files[self.max_keep:]:
            try: os.remove(f)
            except: pass