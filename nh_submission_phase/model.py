#!/usr/bin/env python

"""
개선된 ECG 모델 정의 모듈
- 정보 소실 문제 해결
- 분리도(Separability) 개선
- 코사인 유사도 문제 해결
- 실용적인 임베딩 생성

주요 개선사항:
1. LeakyReLU 사용으로 정보 보존
2. Skip Connection으로 gradient flow 개선
3. 얕은 네트워크로 과도한 정보 소실 방지
4. 다양성 확보를 위한 Dual Branch 구조
5. 적절한 가중치 초기화
"""

import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ImprovedMultiScaleConv1D(nn.Module):
    """개선된 Multi-scale convolution - 정보 보존 강화"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        
        k_small, k_medium, k_large = 3, 7, 15
        branch_channels = out_channels // 3
        
        self.conv_small = nn.Conv1d(in_channels, branch_channels, kernel_size=k_small, padding=k_small // 2)
        self.conv_medium = nn.Conv1d(in_channels, branch_channels, kernel_size=k_medium, padding=k_medium // 2)
        self.conv_large = nn.Conv1d(in_channels, branch_channels, kernel_size=k_large, padding=k_large // 2)
        
        # GroupNorm 사용 (BatchNorm보다 안정적)
        self.norm = nn.GroupNorm(min(32, out_channels//4), out_channels)
        
        # LeakyReLU로 정보 보존
        self.activation = nn.LeakyReLU(0.1)
        
        # 초기화 개선
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv_small, self.conv_medium, self.conv_large]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.conv_small(x)
        x2 = self.conv_medium(x)
        x3 = self.conv_large(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.activation(self.norm(x))


class LeadGate(nn.Module):
    """개선된 Lead Gate - 더 안정적인 초기화"""
    def __init__(self, num_leads=12):
        super().__init__()
        # 1 대신 1.1로 초기화 (더 적극적인 학습)
        self.alpha = nn.Parameter(torch.ones(1, num_leads, 1) * 1.1)
    
    def forward(self, x):
        return x * self.alpha


class LightTemporalAttention(nn.Module):
    """가벼운 Temporal Attention - 과도한 억제 방지"""
    def __init__(self, kernel_size=9):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, T)
        t_score = x.mean(dim=1, keepdim=True)   # (B, 1, T)
        mask = self.sigmoid(self.conv(t_score)) # (B, 1, T)
        
        # 최소 0.3, 최대 1.2로 제한 (과도한 억제 방지)
        mask = torch.clamp(mask, min=0.3, max=1.2)
        
        return x * mask


class ResidualBlock(nn.Module):
    """Skip Connection이 있는 Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        
        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(min(32, out_channels//4), out_channels)
            )
        
        self.activation = nn.LeakyReLU(0.1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.skip is not None:
            residual = self.skip(x)
        
        out += residual * 0.3  # Skip connection 강도 조절
        out = self.activation(out)
        
        return out


class DualBranchECGEncoder(nn.Module):
    """
    개선된 ECG Encoder
    - 정보 소실 최소화
    - 분리도 개선
    - 다양성 확보
    """
    def __init__(self, in_channels=12, dropout=0.1):
        super().__init__()
        
        self.lead_gate = LeadGate(in_channels)
        
        # Stem - 더 보수적으로
        self.stem = nn.Sequential(
            ImprovedMultiScaleConv1D(in_channels, 128),  # 더 넓게 시작
            nn.MaxPool1d(2)
        )
        
        # 메인 브랜치 (양수 특징)
        self.main_branch = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            LightTemporalAttention(),
            ResidualBlock(256, 512, stride=2),
            # 더 이상 깊게 가지 않음 (정보 보존)
        )
        
        # 보조 브랜치 (음수/대비 특징)
        self.aux_branch = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.GroupNorm(32, 256),
            nn.Tanh(),  # -1~1 범위로 다양성 확보
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.GroupNorm(32, 256),
            nn.Tanh(),
        )
        
        # Pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 메인: 512*2, 보조: 256*2 = 총 1536
        self.in_dim = 512 * 2 + 256 * 2
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.lead_gate(x)
        x = self.stem(x)
        
        # 메인 브랜치
        main_feat = self.main_branch(x)
        main_avg = self.avg_pool(main_feat).squeeze(-1)
        main_max = self.max_pool(main_feat).squeeze(-1)
        
        # 보조 브랜치  
        aux_feat = self.aux_branch(x)
        aux_avg = self.avg_pool(aux_feat).squeeze(-1)
        aux_max = self.max_pool(aux_feat).squeeze(-1)
        
        # 결합
        feat = torch.cat([main_avg, main_max, aux_avg, aux_max], dim=1)
        feat = self.dropout(feat)
        
        return feat
    
    def forward(self, x):
        return self.forward_features(x)


class ShallowECGEncoder(nn.Module):
    """
    얕은 네트워크 버전 - 안정성 우선
    """
    def __init__(self, in_channels=12, dropout=0.1):
        super().__init__()
        
        self.lead_gate = LeadGate(in_channels)
        
        # 단 3개 레이어만 사용
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 256, kernel_size=7, padding=3),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            # Layer 2
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            # Layer 3 (마지막)
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.in_dim = 512
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.2)  # 약간 큰 초기값
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.lead_gate(x)
        x = self.features(x)
        feat = x.squeeze(-1)
        feat = self.dropout(feat)
        return feat
    
    def forward(self, x):
        return self.forward_features(x)


# 원본 EnhancedECGEncoder도 개선
class EnhancedECGEncoder(nn.Module):
    """
    개선된 원본 모델 - 기존 구조 유지하면서 문제점만 수정
    """
    def __init__(self, in_channels=12, dropout=0.1):
        super().__init__()
        self.lead_gate = LeadGate(in_channels)

        self.stem = nn.Sequential(
            ImprovedMultiScaleConv1D(in_channels, 66),
            nn.MaxPool1d(2)
        )

        # ResidualBlock 사용
        self.layer1 = ResidualBlock(66, 128, stride=1)
        self.tatt1 = LightTemporalAttention()

        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.tatt2 = LightTemporalAttention()

        # Layer3는 제거 (너무 깊음)
        
        # 최종 처리
        self.final_conv = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.final_norm = nn.GroupNorm(32, 512)
        self.final_act = nn.LeakyReLU(0.1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.in_dim = 512 * 2
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.lead_gate(x)
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.tatt1(x)
        
        x = self.layer2(x)
        x = self.tatt2(x)
        
        x = self.final_conv(x)
        x = self.final_norm(x)
        x = self.final_act(x)
        
        avg_feat = self.avg_pool(x).squeeze(-1)
        max_feat = self.max_pool(x).squeeze(-1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        feat = self.dropout(feat)
        return feat

    def forward(self, x):
        return self.forward_features(x)


# ---------------------------------
#  개선된 Heads
# ---------------------------------

class ImprovedHeads(nn.Module):
    """
    개선된 헤드 - 분리도 향상
    """
    def __init__(self, in_dim=1024, n_classes=2, proj_dim=128, dropout=0.2):
        super().__init__()
        
        # 분류 헤드 - Residual connection 추가
        self.cls_fc1 = nn.Linear(in_dim, 512)
        self.cls_norm1 = nn.LayerNorm(512)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_norm2 = nn.LayerNorm(256)
        self.cls_out = nn.Linear(256, n_classes)
        
        # Skip connection을 위한 projection
        self.cls_skip = nn.Linear(in_dim, 256)
        
        # Contrastive 헤드 - 다양성 강화
        self.proj_fc1 = nn.Linear(in_dim, 512)
        self.proj_norm1 = nn.LayerNorm(512)
        self.proj_fc2 = nn.Linear(512, proj_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feat):
        # 분류 헤드 (skip connection 포함)
        x = self.cls_fc1(feat)
        x = self.cls_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.cls_fc2(x)
        x = self.cls_norm2(x)
        
        # Skip connection
        skip = self.cls_skip(feat)
        x = x + skip * 0.3
        x = self.activation(x)
        
        logits = self.cls_out(x)
        
        # Contrastive 헤드
        z = self.proj_fc1(feat)
        z = self.proj_norm1(z)
        z = self.activation(z)
        z = self.proj_fc2(z)
        z = F.normalize(z, dim=1)
        
        return logits, z


# ---------------------------------
# Factory 함수
# ---------------------------------

def create_encoder(version='dual', **kwargs):
    """
    인코더 생성 팩토리 함수
    
    Args:
        version: 'dual', 'shallow', 'enhanced' 중 선택
    """
    if version == 'dual':
        return DualBranchECGEncoder(**kwargs)
    elif version == 'shallow':
        return ShallowECGEncoder(**kwargs)
    elif version == 'enhanced':
        return EnhancedECGEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")


def test_encoder_quality(encoder, test_signals, test_sources, device='cuda'):
    """
    인코더 품질 테스트 함수
    """
    encoder.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for sig, source in zip(test_signals, test_sources):
            if sig.shape[0] != 12:
                sig = sig.T
            
            sig_tensor = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
            emb = encoder(sig_tensor)
            embeddings.append(emb.cpu().numpy().flatten())
            labels.append(source)
    
    # 분리도 계산
    same_distances = []
    diff_distances = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            
            src_i = labels[i].split('_')[0]
            src_j = labels[j].split('_')[0]
            
            if src_i == src_j:
                same_distances.append(dist)
            else:
                diff_distances.append(dist)
    
    if same_distances and diff_distances:
        separation_ratio = np.mean(diff_distances) / np.mean(same_distances)
        print(f"분리비: {separation_ratio:.3f} ({'우수' if separation_ratio > 1.5 else '보통' if separation_ratio > 1.2 else '부족'})")
        return separation_ratio
    
    return 0


if __name__ == "__main__":
    # 테스트 코드
    x = torch.randn(2, 12, 4096)
    
    print("=== 개선된 모델들 테스트 ===")
    
    models = {
        'DualBranch': DualBranchECGEncoder(),
        'Shallow': ShallowECGEncoder(), 
        'Enhanced': EnhancedECGEncoder()
    }
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
            print(f"{name}: {out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")
    
    print("\n권장 모델: ShallowECGEncoder (안정성) 또는 DualBranchECGEncoder (성능)")