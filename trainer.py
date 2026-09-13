# trainer.py
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics
from utils.plotting import plot_training_curves
from utils.tools import CheckpointManager


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, cfg):
        self.model = model
        self.tr_loader = train_loader
        self.val_loader = val_loader
        self.opt = optimizer
        self.sch = scheduler
        self.crit = criterion
        self.cfg = cfg
        self.device = cfg['project']['device']

        self.out = cfg['paths']['output_dir']
        os.makedirs(self.out, exist_ok=True)
        self.scaler = GradScaler(enabled=self.device.startswith('cuda'))
        self.writer = SummaryWriter(log_dir=os.path.join(self.out, 'runs'))
        self.ckpt = CheckpointManager(self.out, cfg['train']['max_keep_ckpts'])
        self.last_path = os.path.join(self.out, "last_model.pth")

        self.hist = {k: [] for k in [
            'train_loss', 'train_acc', 'train_auc', 'train_spe', 'train_sen',
            'val_acc', 'val_auc', 'val_spe', 'val_sen', 'auc_gap', 'lr'
        ]}
        self.best_score = -999.0
        self.start_ep = 0

    def load(self):
        if os.path.exists(self.last_path):
            c = torch.load(self.last_path)
            self.model.load_state_dict(c['model_state_dict'])
            self.opt.load_state_dict(c['optimizer_state_dict'])
            self.start_ep = c['epoch'] + 1
            self.best_score = c.get('best_score', -999.0)
            print(f"🔄 Resumed from Ep {self.start_ep}")

    def run(self):
        self.load()
        pat = 0
        print(f"🔥 Training Start: {self.cfg['train']['epochs']} Epochs")

        for ep in range(self.start_ep, self.cfg['train']['epochs']):
            loss, tm = self._train(ep)
            # 🔥 获取验证集的详细结果 (含 ID)
            vm, val_df = self._val(ep)

            self.sch.step()
            curr_lr = self.opt.param_groups[0]['lr']

            # Gap-Aware Scoring
            gap = tm[1] - vm[1]
            score = (tm[1] + vm[1]) - 1.0 * abs(gap) + 0.2 * ((vm[2] + vm[3]) / 2.0)

            self._update_hist(loss, tm, vm, gap, curr_lr)

            print(f"Ep {ep + 1} | Loss:{loss:.4f} | T_AUC:{tm[1]:.4f} | V_AUC:{vm[1]:.4f} | Score:{score:.4f}")
            self.ckpt.save(self.model, ep + 1, vm[1], score)

            torch.save({'epoch': ep, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(), 'best_score': self.best_score}, self.last_path)

            # 🔥 自动保存 Excel 报表
            if score > self.best_score:
                self.best_score = score
                torch.save(self.model.state_dict(), os.path.join(self.out, "best_model.pth"))
                val_df.to_excel(os.path.join(self.out, "best_val_predictions.xlsx"), index=False)
                print("   🏆 New Best! Excel Saved.")
                pat = 0
            else:
                pat += 1
                if pat >= self.cfg['train']['patience']:
                    print("🛑 Early Stopping");
                    break

        self.writer.close()
        plot_training_curves(self.hist, os.path.join(self.out, "curves.png"))

    def _train(self, ep):
        self.model.train()
        tl, probs, labs = 0, [], []
        steps = self.cfg['train']['accum_steps']
        pbar = tqdm(self.tr_loader, desc=f"Tr {ep + 1}", leave=False)
        self.opt.zero_grad(set_to_none=True)
        accumulated = 0

        # 解包 (g, l, _) 忽略ID
        for i, (g, l, _) in enumerate(pbar):
            if not g: continue
            g = [x.to(self.device) for x in g[0]]
            l = l.float().to(self.device).view(-1, 1)

            with autocast(enabled=self.device.startswith('cuda')):
                logits = self.model(g)
                loss = self.crit(logits, l) / steps
            self.scaler.scale(loss).backward()
            accumulated += 1

            if accumulated == steps:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['clip_grad'])
                self.scaler.step(self.opt);
                self.scaler.update();
                self.opt.zero_grad(set_to_none=True)
                accumulated = 0

            tl += loss.item() * steps
            probs.append(torch.sigmoid(logits).detach().cpu().item())
            labs.append(l.item())

        # Apply gradients from the last partial accumulation window.
        if accumulated:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['clip_grad'])
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)

        return tl / len(self.tr_loader), calculate_metrics(labs, probs)

    def _val(self, ep):
        self.model.eval()
        probs, labs, ids = [], [], []
        with torch.no_grad():
            # 🔥 解包 (g, l, sid)
            for g, l, sid in tqdm(self.val_loader, desc=f"Va {ep + 1}", leave=False):
                if not g: continue
                logits = self.model([x.to(self.device) for x in g[0]])
                p = torch.sigmoid(logits).item()
                probs.append(p)
                labs.append(l.item())
                ids.append(sid[0])  # sid 是 tuple

        metrics = calculate_metrics(labs, probs)

        # 🔥 生成 Excel 所需的 DataFrame
        df = pd.DataFrame({
            "Slide_ID": ids,
            "True_Label": labs,
            "Pred_Prob": probs,
            "Pred_Label": [1 if p > 0.5 else 0 for p in probs]
        })
        return metrics, df

    def _update_hist(self, loss, tm, vm, gap, lr):
        h = self.hist
        h['train_loss'].append(loss);
        h['lr'].append(lr)
        h['train_acc'].append(tm[0]);
        h['train_auc'].append(tm[1])
        h['train_spe'].append(tm[2]);
        h['train_sen'].append(tm[3])
        h['val_acc'].append(vm[0]);
        h['val_auc'].append(vm[1])
        h['val_spe'].append(vm[2]);
        h['val_sen'].append(vm[3])
        h['auc_gap'].append(gap)
        pd.DataFrame(h).to_csv(os.path.join(self.out, "log.csv"), index=False)
