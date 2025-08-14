#!/usr/bin/env python

"""
ECG 모델 정의 모듈
- EnhancedECGEncoder: 메인 ECG 인코더
- Heads: 분류 및 contrastive 헤드
- 관련 빌딩 블록들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util_nh import *
# ---------------------------------
# 4) Building Blocks & Encoders
# ---------------------------------

class MultiScaleConv1D(nn.Module):
    """Multi-scale convolution to capture different ECG wave components."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Ensure divisibility by 3
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        k_small, k_medium, k_large = 3, 7, 15
        self.conv_small  = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_small, padding=k_small // 2)
        self.conv_medium = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_medium, padding=k_medium // 2)
        self.conv_large  = nn.Conv1d(in_channels, out_channels // 3, kernel_size=k_large, padding=k_large // 2)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x1 = self.conv_small(x)
        x2 = self.conv_medium(x)
        x3 = self.conv_large(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return F.relu(self.bn(x))


class LeadGate(nn.Module):
    """Learnable per-lead gating at the raw input stage."""
    def __init__(self, num_leads=12):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_leads, 1))
    
    def forward(self, x):
        # x: (B, C=12, T)
        return x * self.alpha


class TrueTemporalAttention(nn.Module):
    """Time-wise attention mask via 1D conv over channel-averaged trace."""
    def __init__(self, kernel_size=9):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, T)
        t_score = x.mean(dim=1, keepdim=True)   # (B, 1, T)
        mask = self.sigmoid(self.conv(t_score)) # (B, 1, T)
        return x * mask


class ResidualLSTMBlock(nn.Module):
    """BiLSTM with residual connection (keeps (B, C, T) interface)."""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # x: (B, C, T)
        res = x
        x = x.transpose(1, 2)        # (B, T, C)
        lstm_out, _ = self.lstm(x)   # (B, T, 2H)
        x = self.proj(lstm_out)      # (B, T, C)
        x = self.norm(x + res.transpose(1, 2))
        return x.transpose(1, 2)     # (B, C, T)


class EnhancedECGEncoder(nn.Module):
    """
    Enhanced backbone for ECG:
      - LeadGate on raw leads
      - Multi-scale conv -> deep conv blocks
      - True time-wise attention after each stage
      - Residual BiLSTM
      - Dual pooling (avg+max) -> feature vector
    """
    def __init__(self, in_channels=12, dropout=0.3):
        super().__init__()
        self.lead_gate = LeadGate(in_channels)

        self.stem = nn.Sequential(
            MultiScaleConv1D(in_channels, 66),
            nn.MaxPool1d(2)
        )

        self.layer1 = self._make_layer(66, 128, blocks=2, stride=1)
        self.tatt1  = TrueTemporalAttention()

        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)
        self.tatt2  = TrueTemporalAttention()

        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)
        self.tatt3  = TrueTemporalAttention()

        self.lstm_block = ResidualLSTMBlock(512, 256, num_layers=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.in_dim = 512 * 2  # avg+max concat -> 1024

        self.dropout = nn.Dropout(dropout)

    def _make_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 7, 1, 3, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = [self._make_block(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(self._make_block(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # x: (B, C, T)
        x = self.lead_gate(x)
        x = self.stem(x)

        x = self.layer1(x); x = self.tatt1(x)
        x = self.layer2(x); x = self.tatt2(x)
        x = self.layer3(x); x = self.tatt3(x)

        x = self.lstm_block(x)
        avg_feat = self.avg_pool(x).squeeze(-1)
        max_feat = self.max_pool(x).squeeze(-1)
        feat = torch.cat([avg_feat, max_feat], dim=1)  # (B, 1024)
        feat = self.dropout(feat)
        return feat

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------
# 5) Heads (classification + contrastive)
# ---------------------------------

class Heads(nn.Module):
    """
    Two heads:
      - classification logits
      - contrastive projection (normalized)
    """
    def __init__(self, in_dim=1024, n_classes=2, proj_dim=128, dropout=0.3):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, feat):
        logits = self.cls(feat)
        z = F.normalize(self.proj(feat), dim=1)
        return logits, z


# ---------------------------------
# 추가 유틸리티 모델들
# ---------------------------------

class LinearProbeHead(nn.Module):
    """ 2-Layer MLP classifier head """
    def __init__(self, in_dim=512, num_classes=2, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.mlp(x)


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration"""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_T)))))

    def forward(self, logits):
        T = self.logT.exp()
        return logits / T


# ---------------------------------
# Trainers (for making sure of encoder performance)
# ---------------------------------

class LinearProbeTrainer:
    """
    Linear probe 학습
    """
    def __init__(self, encoder, signals_tensor, labels_array, save_path, batch_size=64, lr=1e-4, device='cpu', verbose=False):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # auto in_dim
        in_dim = getattr(self.encoder, "in_dim", 512)
        n_classes = int(len(np.unique(labels_array)))
        self.head = LinearProbeHead(in_dim=in_dim, num_classes=n_classes).to(self.device)
        self.save_path = save_path
        self.verbose = verbose

        dataset = SupervisedECGDataset(signals_tensor, labels_array)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_safe)

        # class weight (placeholder; adjust as needed)
        class_weights = torch.tensor([0.5, 1.0], dtype=torch.float32).to(self.device)

        self.opt = optim.Adam(self.head.parameters(), lr=lr)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)

    def train(self, epochs=5):
        if self.verbose:
            try:
                from tqdm import tqdm
                epoch_iterator = tqdm(range(epochs), desc="Fine-tuning Head")
            except ImportError:
                epoch_iterator = range(epochs)
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:
            self.head.train()
            total_loss = 0.0
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)   # (B, C, T)
                with torch.no_grad():
                    if hasattr(self.encoder, "forward_features"):
                        h = self.encoder.forward_features(xb)
                    else:
                        h, _ = self.encoder(xb)
                logits = self.head(h)
                loss = self.crit(logits, yb)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.loader)
            if self.verbose:
                print(f'[Fine-tune] Epoch {e+1}/{epochs}, Loss: {avg_loss:.4f}')

        torch.save(self.head.state_dict(), self.save_path)
        if self.verbose:
            print(f"   -> Head saved after {epochs} epochs to {self.save_path}")
