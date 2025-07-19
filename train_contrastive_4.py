#!/usr/bin/env python3
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import wandb
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# --- Logging setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# ---------- 1. Curriculum Sampler ----------
class CurriculumSampler(Sampler):
    def __init__(self, difficulty, epoch, total_epochs):
        # lazy conversion to numpy
        if torch.is_tensor(difficulty):
            difficulty = difficulty.cpu().numpy()
        self.difficulty = np.asarray(difficulty)
        self.n = len(self.difficulty)
        ratio = (epoch + 1) / total_epochs
        cutoff = int(self.n * ratio)
        sorted_idx = sorted(range(self.n), key=lambda i: self.difficulty[i])
        easy, hard = sorted_idx[:cutoff], sorted_idx[cutoff:]
        random.shuffle(hard)
        self.indices = easy + hard
    def __iter__(self): return iter(self.indices)
    def __len__(self): return self.n

# ---------- 2. Utility functions ----------
def load_signals(path):
    sig = np.load(path)
    if sig.ndim == 3 and sig.shape[1] > sig.shape[2]:
        sig = sig.transpose(0, 2, 1)
    return torch.from_numpy(sig.astype(np.float32))

def detect_qrs_peaks(sig, fs):
    m = sig.abs().sum(dim=0).cpu().numpy()
    thr = m.mean() + m.std()
    peaks = [i for i in range(1, len(m)-1)
             if m[i] > thr and m[i] > m[i-1] and m[i] > m[i+1]]
    dist = int(0.2 * fs)
    filt = []
    for p in peaks:
        if all(abs(p - f) > dist for f in filt):
            filt.append(p)
    return filt

def compute_difficulty(signals, fs=400):
    diffs = []
    for sig in signals:
        pts = detect_qrs_peaks(sig, fs)
        if len(pts) < 2:
            diffs.append(0.0)
        else:
            diffs.append(float(np.diff(pts).std()))
    return torch.tensor(diffs, dtype=torch.float32)

# augmentations
def augment_basic(x):
    # 노이즈 추가
    x = x + torch.randn_like(x) * 0.01
    # 랜덤 시프트
    shift = random.randint(-5, 5)
    C, T = x.shape
    if shift > 0:
        pad = torch.zeros(C, shift, device=x.device)
        x = torch.cat([x[:, shift:], pad], dim=1)
    elif shift < 0:
        pad = torch.zeros(C, -shift, device=x.device)
        x = torch.cat([pad, x[:, :shift]], dim=1)
    return x

def random_crop_include_qrs(sig, crop_len, fs):
    C,T = sig.shape
    if T<=crop_len:
        return F.pad(sig, (0, crop_len-T))
    pts = detect_qrs_peaks(sig, fs)
    valid = [i for i in range(T-crop_len) if any(p>=i and p<i+crop_len for p in pts)]
    start = random.choice(valid) if valid else random.randint(0, T-crop_len)
    return sig[:, start:start+crop_len]

def mask_avoid_qrs(sig, fs, mask_frac=0.1):
    C,T = sig.shape
    mask_len = int(T*mask_frac)
    pts = detect_qrs_peaks(sig, fs)
    forbidden = np.zeros(T, bool)
    w = int(0.05*fs)
    for p in pts:
        s,e = max(0,p-w//2), min(T,p+w//2)
        forbidden[s:e] = True
    cand = [i for i in range(T-mask_len) if not forbidden[i:i+mask_len].any()]
    if cand:
        start = random.choice(cand)
        sig[:, start:start+mask_len] = 0
    return sig

# ContrastiveECGDatasetOptim
# augment 복잡 버전
class ContrastiveECGDatasetOptim(Dataset):
    def __init__(self, signal_np, crop_sec=4, fs=400, subset_k=3):
        self.signals = torch.from_numpy(signal_np.astype(np.float32))
        self.crop_len = int(crop_sec*fs)
        self.fs = fs
        self.n_leads = self.signals.shape[1]
        self.subset_k = subset_k
        self.difficulty = compute_difficulty(self.signals, fs)
    def __len__(self): return len(self.signals)
    def __getitem__(self, idx):
        sig = self.signals[idx]
        # full 12ch
        v1, v2 = sig.clone(), sig.clone()
        # random mask leads
        mask1 = torch.ones(self.n_leads, dtype=torch.bool)
        mask2 = torch.ones(self.n_leads, dtype=torch.bool)
        drop1 = random.sample(range(self.n_leads), self.n_leads-self.subset_k)
        drop2 = random.sample(range(self.n_leads), self.n_leads-self.subset_k)
        mask1[drop1]=False; mask2[drop2]=False
        v1[~mask1]=0; v2[~mask2]=0
        # crop & augment
        v1 = random_crop_include_qrs(v1, self.crop_len, self.fs)
        v2 = random_crop_include_qrs(v2, self.crop_len, self.fs)
        v1 = augment_basic(v1); v1 = mask_avoid_qrs(v1, self.fs)
        v2 = augment_basic(v2); v2 = mask_avoid_qrs(v2, self.fs)
        return v1, v2, idx

# 간단 버전
# class ContrastiveECGDatasetOptim(Dataset):
#     def __init__(self, signal_np, crop_sec=4, fs=400, subset_k=3):
#         self.signals = torch.from_numpy(signal_np.astype(np.float32))
        
#     def __len__(self): return len(self.signals)

#     def __getitem__(self, idx):
#         sig = self.signals[idx]          # (12, T)
#         # 두 뷰 모두 원본에 기본 증강만 적용
#         v1 = augment_basic(sig.clone())
#         v2 = augment_basic(sig.clone())
#         return v1, v2, idx


# ---------- 4. Model & Loss ----------

# ----- 3. SE 블록 -----
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

# ----- 6. NT-Xent -----
class NTXentLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2],0)
        sim = torch.matmul(z, z.T)/self.temp
        mask = ~torch.eye(2*B, device=sim.device).bool()
        pos = torch.cat([sim.diag(B), sim.diag(-B)],0).unsqueeze(1)
        neg = sim[mask].view(2*B, -1)
        logits = torch.cat([pos, neg],1)
        labels = torch.zeros(2*B, dtype=torch.long, device=sim.device)
        return self.ce(logits, labels)

    
class SupConLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp
        self.eps = 1e-7

    def forward(self, z1, z2, labels=None):
        """
        z1, z2: (B, D) / labels: (B,) or None
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], 0)    # (2B, D)
        sim = torch.matmul(z, z.T) / self.temp   # (2B, 2B)
        sim = torch.exp(sim)

        # mask: 같은 클래스끼리만 positive로
        if labels is not None:
            labels = labels.repeat(2)   # (2B,)
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(z.device)
            mask = mask - torch.eye(2*B, device=z.device)
        else:
            # unsupervised: (i, i+B)만 positive
            mask = torch.zeros(2*B, 2*B, device=z.device)
            for i in range(B):
                mask[i, i+B] = mask[i+B, i] = 1
        
        denom = (sim * (1-mask)).sum(dim=1, keepdim=True) + self.eps
        pos = (sim * mask).sum(dim=1, keepdim=True)
        loss = -torch.log(pos / denom + self.eps)
        return loss.mean()

import math

def get_temp(epoch, total_epochs,
             temp_min=0.1, temp_max=0.3, warmup_epochs=0):
    # 이제 warmup_epochs는 쓰지 않고 단순 cosine
    progress = epoch / (total_epochs - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return temp_min + (temp_max - temp_min) * cosine

class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    def forward(self, z1, z2):
        # Invariance (MSE between two views)
        sim_loss = F.mse_loss(z1, z2)
        # Variance
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        # Covariance (decorrelation)
        N, D = z1.size()
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (off_diagonal(cov_z1).pow(2).sum() / D) + (off_diagonal(cov_z2).pow(2).sum() / D)
        return self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

from torch.optim.lr_scheduler import CosineAnnealingLR

def pretrain_contrastive(
    signals_path,
    epochs=30, 
    batch_size=256,
    lr=1e-3,
    crop_sec=4,
    fs=400,
    subset_k=3,
    mask_frac=0.1,
    loss_type='ntxent',
    model_save_path='outputs/contrastive_encoder_v9.pth',
    val_ratio=0.2,
    random_state=42,
    checkpoint_interval=5,
    smoothing_alpha=0.8
):
    # — W&B init —
    wandb.init(project="ecg-contrastive-ntxent",
               name = 'encoder_contrastive_v9',
               config=dict(
                   epochs=epochs,
                   batch_size=batch_size,
                   lr=lr,
                   crop_sec=crop_sec,
                   fs=fs,
                   subset_k=subset_k,
                   mask_frac=mask_frac,
                   loss_type=loss_type
               ))
    cfg = wandb.config

    # — Load & split signals —
    sig = np.load(signals_path)
    if sig.ndim == 3 and sig.shape[1] > sig.shape[2]:
        sig = sig.transpose(0, 2, 1)
    idxs = np.arange(len(sig))
    train_i, val_i = train_test_split(
        idxs, test_size=val_ratio,
        random_state=random_state, shuffle=True
    )
    tr_arr, vl_arr = sig[train_i], sig[val_i]

    # — Datasets & Loaders —
    tr_ds = ContrastiveECGDatasetOptim(tr_arr,
                                       crop_sec=cfg.crop_sec,
                                       fs=cfg.fs,
                                       subset_k=cfg.subset_k)
    vl_ds = ContrastiveECGDatasetOptim(vl_arr,
                                       crop_sec=cfg.crop_sec,
                                       fs=cfg.fs,
                                       subset_k=cfg.subset_k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet1DEncoder(in_ch=tr_arr.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    scaler = GradScaler()

    # — choose loss —
    if loss_type == 'vicreg':
        criterion = VICRegLoss()
    elif loss_type == 'supcon':
        criterion = SupConLoss()
    else:
        criterion = NTXentLoss()

    best_val = float('inf')
    val_ema = None

    for epoch in range(cfg.epochs):
        # — Training with CurriculumSampler —
        sampler = CurriculumSampler(tr_ds.difficulty, epoch, cfg.epochs)
        tr_loader = DataLoader(tr_ds,
                               batch_size=cfg.batch_size,
                               sampler=sampler,
                               num_workers=4,
                               pin_memory=True,
                               drop_last=True)

        model.train()
        running_loss = 0.0
        for v1, v2, _ in tqdm(tr_loader, desc=f"Train {epoch+1}/{cfg.epochs}"):
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = criterion(z1, z2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
        scheduler.step()
        train_loss = running_loss / len(tr_loader)

        # — Validation & EMA smoothing —
        vl_loader = DataLoader(vl_ds,
                               batch_size=cfg.batch_size,
                               shuffle=False,
                               num_workers=4,
                               pin_memory=True)
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for v1, v2, _ in vl_loader:
                v1, v2 = v1.to(device), v2.to(device)
                _, z1 = model(v1); _, z2 = model(v2)
                val_running += criterion(z1, z2).item()
        val_loss = val_running / len(vl_loader)
        val_ema = val_loss if val_ema is None else (
            smoothing_alpha * val_loss + (1 - smoothing_alpha) * val_ema
        )

        # — Checkpoints —
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       model_save_path.replace('.pth', '_best.pth'))
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(),
                       model_save_path.replace('.pth', f'_ep{epoch+1}.pth'))

        # — Logging —
        temp = get_temp(epoch, cfg.epochs)
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_loss_ema': val_ema,
            'temp': temp,
            'lr': scheduler.get_last_lr()[0]
        })
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"val_ema={val_ema:.4f}, temp={temp:.3f}")

    # — Final save —
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str,
                        default='/Data/nh25/ECG/2_trial_pretrain/outputs/9/contrastive_encoder_v9.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--crop_sec', type=float, default=4)
    parser.add_argument('--fs', type=int, default=400)
    parser.add_argument('--subset_k', type=int, default=3)
    parser.add_argument('--mask_frac', type=float, default=0.1)
    parser.add_argument('--loss_type',
                        choices=['supcon', 'ntxent', 'vicreg'],
                        default='ntxent')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)
    args, _ = parser.parse_known_args()

    pretrain_contrastive(
        signals_path=args.signals_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_sec=args.crop_sec,
        fs=args.fs,
        subset_k=args.subset_k,
        mask_frac=args.mask_frac,
        loss_type=args.loss_type,
        model_save_path=args.model_save_path,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        checkpoint_interval=args.checkpoint_interval,
        smoothing_alpha=args.smoothing_alpha
    )

# python /Data/nh25/ECG/2_trial_pretrain/train_contrastive_4.py --signals_path /Data/nh25/ECG/0_Data/processed_full_train_data/signals.npy