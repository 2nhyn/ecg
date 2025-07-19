# 간단한 structure의 모델 2layer linear probe 

#!/usr/bin/env python3
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
)
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from helper_code import compute_challenge_score
import random


# --------- Dataset -----------
class ECGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx): return self.signals[idx], self.labels[idx]


class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.orig = orig_linear
        self.rank = rank
        self.scaling = alpha / rank
        for p in self.orig.parameters(): p.requires_grad = False
        self.lora_A = nn.Linear(orig_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, orig_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 원본 weight/bias를 참조할 수 있도록 프록시
    @property
    def weight(self):
        return self.orig.weight

    @property
    def bias(self):
        return self.orig.bias

    def forward(self, x):
        return self.orig(x) + self.lora_B(self.lora_A(x)) * self.scaling
    
    def to(self, *args, **kwargs):
        # LoRA adapter 파라미터도 동일하게 이동
        self.lora_A.to(*args, **kwargs)
        self.lora_B.to(*args, **kwargs)
        self.orig.to(*args, **kwargs)
        return super().to(*args, **kwargs)

def apply_lora(model: nn.Module, rank: int = 4, alpha: float = 16.0):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and module is not model:
            parent, attr = name.rsplit('.', 1)
            parent_mod = dict(model.named_modules())[parent] if parent else model
            setattr(parent_mod, attr, LoRALinear(module, rank, alpha))


def find_best_threshold(labels, probs, beta=1.0):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f_betas = (1 + beta**2) * precisions * recalls / (beta**2 * precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f_betas)
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f_betas[best_idx]



# --------- Model Components -----------
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# ----- 4. Transformer Layer 편입 -----
class TransformerBlock1D(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    def forward(self, x):  # x: (B, L, C)
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        res2 = x
        x = res2 + self.mlp(self.norm2(res2))
        return x

# ----- 5. BasicBlock와 ResNet1D -----
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1D(out_c)
        self.down = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.down:
            identity = self.down(x)
        out += identity
        return self.relu(out)

class ResNet1DEncoder(nn.Module):
    def __init__(self, in_ch, emb_dim=128, layers=[2,2,2,2], dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], 2)
        self.layer3 = self._make_layer(128, 256, layers[2], 2)
        self.layer4 = self._make_layer(256, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Transformer on sequence length=1 (embedding)
        self.transformer = TransformerBlock1D(512)
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim)
        )
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        down = None
        if stride!=1 or in_c!=out_c:
            down = nn.Sequential(
                nn.Conv1d(in_c, out_c,1,stride,bias=False),
                nn.BatchNorm1d(out_c)
            )
        layers = [BasicBlock1D(in_c, out_c, stride, down)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_c, out_c))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)  # (B, 512)
        # Transformer expects seq len, here treat embedding as seq_len=1
        x = x.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        z = self.proj(x)
        return x, F.normalize(z, dim=1)

# 1dim Head
# class LinearProbeHead(nn.Module):
#     def __init__(self,in_dim=512,num_classes=2): super().__init__(); self.fc=nn.Linear(in_dim,num_classes)
#     def forward(self,x): return self.fc(x)

# 2dim Head
class LinearProbeHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=2, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)

# 3dim Head
# class LinearProbeHead(nn.Module):
#     def __init__(self, in_dim=512, num_classes=2, hidden_dim=128):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.BatchNorm1d(hidden_dim//2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim//2, num_classes)
#         )
#     def forward(self, x):
#         return self.mlp(x)


# --------- Trainer Class -----------
class LinearProbeTrainer:
    def __init__(self, encoder_path, signal_path, label_path, save_path, batch_size=32, lr=1e-3, device='cuda:2'):
        self.device = device or torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        signals, labels = self._load(signal_path,label_path)
        self.loader = DataLoader(ECGDataset(signals,labels),batch_size=batch_size,shuffle=True)
        self.encoder = ResNet1DEncoder(in_ch=signals.shape[1]).to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path,map_location=self.device))
        self.encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad=False
        # self.head = LinearProbeHead(in_dim=512,num_classes=len(np.unique(labels))).to(self.device) # 1head
        self.head = LinearProbeHead(in_dim=512, num_classes=len(np.unique(labels)), hidden_dim=128).to(self.device) # 2layer head
        self.opt = optim.Adam(self.head.parameters(),lr=lr)
        weights = torch.tensor([0.6, 1.0], dtype=torch.float32).to(self.device)
        self.crit=nn.CrossEntropyLoss(weight=weights)
        self.save_path=save_path
        self.best_f1 = 0.0

    def _load(self, sp, lp):
        sig=np.load(sp); lbl=np.load(lp)
        if sig.ndim==3 and sig.shape[1]!=sig.shape[2]: sig=sig.transpose(0,2,1)
        return sig,lbl

    def _evaluate_precision(self):
        """현재 head의 precision 평가"""
        self.head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in self.loader:  # validation loader가 있다면 그걸 써야 더 좋음
                xb, yb = xb.to(self.device), yb.to(self.device)
                h, _ = self.encoder(xb)
                logits = self.head(h)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return precision_score(labels, preds, average='binary')  # binary용
    
    def _evaluate_f1(self):
        """현재 head의 f1 평가"""
        self.head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in self.loader:  # validation loader가 있다면 그걸 써야 더 좋음
                xb, yb = xb.to(self.device), yb.to(self.device)
                h, _ = self.encoder(xb)
                logits = self.head(h)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return f1_score(labels, preds, average='macro')  # 여기가 핵심!

    def train(self, epochs=10):
        for e in range(epochs):
            self.head.train(); tot=0
            for xb,yb in tqdm(self.loader,desc=f"Epoch {e+1}/{epochs}"):
                xb,yb=xb.to(self.device),yb.to(self.device)
                with torch.no_grad(): h,_=self.encoder(xb)
                logits=self.head(h); loss=self.crit(logits,yb)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                tot+=loss.item()
            print(f"Epoch {e+1}: Loss={tot/len(self.loader):.4f}")
            
            # ⬇️ f1 평가
            f1 = self._evaluate_f1()
            print(f"Epoch {e+1}: F1={f1:.4f}")
            
            # ⬇️ 최고 f1이면 저장
            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(self.head.state_dict(), self.save_path)
                print(f"✅ New best model saved (F1={f1:.4f})")

# =====================================
# mlp linear
# =====================================
# --------- Evaluator Class -----------
class Evaluator_mlp:
    def __init__(self, encoder, head, dataloader, device):
        self.encoder = encoder.eval()
        self.head = head.eval()
        self.loader = dataloader
        self.device = device

    def evaluate(self):
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                h, _ = self.encoder(xb)
                probs = torch.softmax(self.head(h), dim=1)
                preds = probs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_p = np.concatenate(all_preds)
        all_l = np.concatenate(all_labels)
        all_pr = np.concatenate(all_probs)

        acc = accuracy_score(all_l, all_p)
        f1  = f1_score(all_l, all_p, average='macro')
        auc = roc_auc_score(all_l, all_pr[:,1] if all_pr.shape[1]==2 else all_pr, multi_class='ovr')
        print(f'[Eval] acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}')

        cm = confusion_matrix(all_l, all_p)
        print('Confusion Matrix:')
        print(cm)
        print('Classification Report:')
        print(classification_report(all_l, all_p))

        challenge = compute_challenge_score(all_l, all_pr[:,1] if all_pr.shape[1]==2 else all_pr)
        print(f'Challenge Score (TPR at 5% capacity): {challenge:.4f}')

        result_df = pd.DataFrame({
            'true_label': all_l,
            'pred_label': all_p,
            'probability': all_pr.tolist(),
            'confidence': np.max(all_pr, axis=1)
        })
        result_df.to_csv("/Data/nh25/ECG/2_trial_pretrain/outputs/linear_probe_predictions.csv", index=False)
        print("Saved results to linear_probe_predictions.csv")
        return result_df



# ================================
# LoRA
# # ==================================
# ----- LoRA Trainer -----
class LoRATrainer(LinearProbeTrainer):
    def __init__(self, *args, lora_rank=4, lora_alpha=16, **kwargs):
        super().__init__(*args, **kwargs)
        # LoRA 적용 후 디바이스 이동
        apply_lora(self.encoder, rank=lora_rank, alpha=lora_alpha)
        self.encoder.to(self.device)  # 이 줄이 핵심!
        # optimizer: head + LoRA adapter만
        trainable = list(self.head.parameters()) + [p for p in self.encoder.parameters() if p.requires_grad]
        self.opt = optim.Adam(trainable, lr=self.opt.param_groups[0]['lr'])


# --------- Evaluator Class -----------
class Evaluator_lora:
    def __init__(self, encoder, head, dataloader, device, save_csv_path):
        self.encoder = encoder.eval()
        self.head = head.eval()
        self.loader = dataloader
        self.device = device
        self.save_csv_path = save_csv_path

    def evaluate(self, beta=1.0):
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                h, _ = self.encoder(xb)
                probs = torch.softmax(self.head(h), dim=1)
                preds = probs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_p = np.concatenate(all_preds)
        all_l = np.concatenate(all_labels)
        all_pr = np.concatenate(all_probs)

        print(f'[Eval] acc={accuracy_score(all_l, all_p):.4f}, f1={f1_score(all_l, all_p, average="macro"):.4f}, auc={roc_auc_score(all_l, all_pr[:,1]):.4f}')

        print('Confusion Matrix:')
        cm = confusion_matrix(all_l, all_p)
        print(cm)
        print('Classification Report:')
        print(classification_report(all_l, all_p))

        # challenge score
        print(f'Challenge Score (TPR@5%): {compute_challenge_score(all_l, all_pr[:,1]):.4f}')

        # find best threshold
        thr, p, r, f = find_best_threshold(all_l, all_pr[:,1], beta=beta)
        print(f'Best thr@F{beta}: {thr:.3f}, Prec={p:.3f}, Rec={r:.3f}, F{beta}={f:.3f}')
        preds_thr = (all_pr[:,1] >= thr).astype(int)

        print('--- At optimal threshold ---')
        print(classification_report(all_l, preds_thr))
        cm_thr = confusion_matrix(all_l, preds_thr)
        ConfusionMatrixDisplay(cm_thr).plot()
        plt.show()

        # save results
        df = pd.DataFrame({
            'true_label': all_l,
            'pred_label': preds_thr,
            'probability': all_pr.tolist(),
            'confidence': np.max(all_pr, axis=1)
        })
        df.to_csv(self.save_csv_path, index=False)
        print(f"Saved results to {self.save_csv_path}")
        return df


# --------- Main -----------
if __name__ == "__main__":

    encoder_path = "/Data/nh25/ECG/2_trial_pretrain/outputs/9/contrastive_encoder_v8.pth"
    linear_signal_path = "/Data/nh25/ECG/0_Data/processed_strong_label_data/signals.npy"
    linear_label_path  = "/Data/nh25/ECG/0_Data/processed_strong_label_data/labels.npy"
    
    eval_signal_path = "/Data/nh25/ECG/0_Data/processed_eval_data/signals.npy"
    eval_label_path  = "/Data/nh25/ECG/0_Data/processed_eval_data/labels.npy"
    save_head_path = "/Data/nh25/ECG/2_trial_pretrain/outputs/linear_probe_fc_v8.pth"
    save_csv      = "/Data/nh25/ECG/2_trial_pretrain/outputs/lora_predictions.csv"

    # Trainer로 학습 (lora, linearprobe 중에 선택)
    trainer = LinearProbeTrainer(
        encoder_path=encoder_path,
        signal_path=linear_signal_path,
        label_path=linear_label_path,
        save_path=save_head_path,
        batch_size=64,
        lr=1e-3,
        device='cuda:2',
        # lora_rank=8, # 일반적으로 4~8 사용
        # lora_alpha=8 # 일반적으로 alpha=rank 또는 alpha=2×rank, 혹은 8~32 내외
    )
    trainer.train(epochs=15)

    # Best head 불러오기
    trainer.head.load_state_dict(torch.load(save_head_path, map_location=trainer.device))

    # Eval 세트 준비
    eval_s, eval_l = np.load(eval_signal_path), np.load(eval_label_path)
    if eval_s.ndim == 3 and eval_s.shape[1] != eval_s.shape[2]:
        eval_s = eval_s.transpose(0, 2, 1)
    eval_loader = DataLoader(ECGDataset(eval_s, eval_l), batch_size=32)

    # Evaluator로 평가
    evaluator = Evaluator_mlp(trainer.encoder, trainer.head, eval_loader, trainer.device)
    # evaluator = Evaluator_lora(trainer.encoder, trainer.head, eval_loader, trainer.device, save_csv)
    df = evaluator.evaluate(beta=1.0)
