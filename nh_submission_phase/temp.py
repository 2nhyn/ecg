import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 해결책 1: 음수/양수 균형 유지
class DiverseECGEncoder(nn.Module):
    """다양성을 강화한 ECG Encoder - 코사인 유사도 해결"""
    def __init__(self, in_channels=12, dropout=0.1, debug=True):
        super().__init__()
        self.debug = debug
        
        # 첫 번째 브랜치: 양수 특징
        self.positive_branch = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),  # 양수만
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 두 번째 브랜치: 음수 특징 (Tanh 사용)
        self.negative_branch = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.Tanh(),  # -1 ~ +1 범위
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.in_dim = 512  # 256 + 256
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)  # 균등 분포
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)  # 작은 랜덤 bias
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.uniform_(m.weight, 0.8, 1.2)  # 다양한 scale
                nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def _debug_print(self, x, stage_name):
        if self.debug:
            print(f"  {stage_name}: shape={x.shape}, mean={x.mean().item():.6f}, "
                  f"std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
    
    def forward(self, x, debug_mode=None):
        if debug_mode is None:
            debug_mode = self.debug
            
        if debug_mode:
            self._debug_print(x, "Input")
        
        # 양수 브랜치
        pos_feat = self.positive_branch(x).squeeze(-1)
        if debug_mode:
            self._debug_print(pos_feat, "Positive Branch")
        
        # 음수 브랜치  
        neg_feat = self.negative_branch(x).squeeze(-1)
        if debug_mode:
            self._debug_print(neg_feat, "Negative Branch")
        
        # 결합
        feat = torch.cat([pos_feat, neg_feat], dim=1)
        if debug_mode:
            self._debug_print(feat, "Combined Features")
        
        # Dropout
        feat = self.dropout(feat)
        if debug_mode:
            self._debug_print(feat, "After Dropout")
        
        return feat


# 해결책 2: 랜덤 노이즈 추가
class NoiseECGEncoder(nn.Module):
    """노이즈를 추가해서 다양성 확보"""
    def __init__(self, in_channels=12, dropout=0.1, debug=True):
        super().__init__()
        self.debug = debug
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.in_dim = 512
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.1)  # 약간의 변동
                nn.init.normal_(m.bias, 0, 0.01)
    
    def _debug_print(self, x, stage_name):
        if self.debug:
            print(f"  {stage_name}: shape={x.shape}, mean={x.mean().item():.6f}, "
                  f"std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
    
    def forward(self, x, debug_mode=None):
        if debug_mode is None:
            debug_mode = self.debug
            
        if debug_mode:
            self._debug_print(x, "Input")
        
        # 특징 추출
        feat = self.features(x).squeeze(-1)
        if debug_mode:
            self._debug_print(feat, "After Features")
        
        # 학습 시에만 작은 노이즈 추가 (다양성 확보)
        if self.training:
            noise = torch.randn_like(feat) * 0.01  # 작은 노이즈
            feat = feat + noise
            if debug_mode:
                self._debug_print(feat, "After Noise")
        
        # Dropout
        feat = self.dropout(feat)
        if debug_mode:
            self._debug_print(feat, "After Dropout")
        
        return feat


# 해결책 3: 서로 다른 활성화 함수 조합
class MultiActivationECGEncoder(nn.Module):
    """여러 활성화 함수를 사용해서 다양성 확보"""
    def __init__(self, in_channels=12, dropout=0.1, debug=True):
        super().__init__()
        self.debug = debug
        
        # Block 1: ReLU 
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Block 2: GELU
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        
        # Block 3: Swish
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),  # Swish
            nn.MaxPool1d(2)
        )
        
        # Block 4: Tanh
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.in_dim = 128
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _debug_print(self, x, stage_name):
        if self.debug:
            print(f"  {stage_name}: shape={x.shape}, mean={x.mean().item():.6f}, "
                  f"std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
    
    def forward(self, x, debug_mode=None):
        if debug_mode is None:
            debug_mode = self.debug
            
        if debug_mode:
            self._debug_print(x, "Input")
        
        x = self.block1(x)
        if debug_mode:
            self._debug_print(x, "After Block1 (ReLU)")
        
        x = self.block2(x)
        if debug_mode:
            self._debug_print(x, "After Block2 (GELU)")
        
        x = self.block3(x)
        if debug_mode:
            self._debug_print(x, "After Block3 (Swish)")
        
        x = self.block4(x)
        if debug_mode:
            self._debug_print(x, "After Block4 (Tanh)")
        
        feat = x.squeeze(-1)
        if debug_mode:
            self._debug_print(feat, "After Squeeze")
        
        feat = self.dropout(feat)
        if debug_mode:
            self._debug_print(feat, "After Dropout")
        
        return feat


def test_diversity_encoders():
    """다양성 강화 encoder들 테스트"""
    print("=== 다양성 강화 Encoder 테스트 (코사인 유사도 해결) ===")
    
    try:
        # 실제 데이터 로드
        signals = np.load("/Data/nh25/ECG/0_Data/processed_eval_data/signals.npy")
        sources = np.load("/Data/nh25/ECG/0_Data/processed_eval_data/sources.npy", allow_pickle=True)
        
        # PTB vs SaMi 찾기
        ptb_indices = [i for i, s in enumerate(sources) if 'ptb' in str(s).lower()]
        sami_indices = [i for i, s in enumerate(sources) if 'sami' in str(s).lower()]
        
        if len(ptb_indices) == 0 or len(sami_indices) == 0:
            print("❌ PTB 또는 SaMi 데이터가 없습니다!")
            return
        
        ptb_sig = signals[ptb_indices[0]]
        sami_sig = signals[sami_indices[0]]
        
        # 전처리
        if ptb_sig.shape[0] != 12: ptb_sig = ptb_sig.T
        if sami_sig.shape[0] != 12: sami_sig = sami_sig.T
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_ptb = torch.from_numpy(ptb_sig.astype(np.float32)).unsqueeze(0).to(device)
        x_sami = torch.from_numpy(sami_sig.astype(np.float32)).unsqueeze(0).to(device)
        
        # 다양성 강화 모델들
        encoders = {
            'Diverse (Pos+Neg Branch)': DiverseECGEncoder(debug=True),
            'Noise (Random Noise)': NoiseECGEncoder(debug=True),
            'MultiAct (Multi Activation)': MultiActivationECGEncoder(debug=True)
        }
        
        best_encoder = None
        best_cos_score = 0  # 코사인 유사도가 낮을수록 좋음
        
        for name, model in encoders.items():
            print(f"\n{'='*70}")
            print(f"테스트 중: {name}")
            print(f"코사인 유사도 개선 중점 테스트")
            print(f"{'='*70}")
            
            model = model.to(device).eval()
            
            with torch.no_grad():
                print(f"🔵 PTB 신호:")
                out_ptb = model(x_ptb, debug_mode=True)
                
                print(f"\n🔴 SaMi 신호:")
                out_sami = model(x_sami, debug_mode=True)
                
                # 상세 분석
                diff = torch.abs(out_ptb - out_sami).mean().item()
                cosine_sim = F.cosine_similarity(out_ptb, out_sami).item()
                
                ptb_magnitude = torch.abs(out_ptb).mean().item()
                sami_magnitude = torch.abs(out_sami).mean().item()
                avg_magnitude = (ptb_magnitude + sami_magnitude) / 2
                
                # 임베딩 분포 분석
                ptb_pos_ratio = (out_ptb > 0).float().mean().item()
                sami_pos_ratio = (out_sami > 0).float().mean().item()
                
                print(f"\n=== {name} - 다양성 분석 ===")
                print(f"  임베딩 크기: {avg_magnitude:.6f}")
                print(f"  임베딩 차이: {diff:.6f}")
                print(f"  🎯 코사인 유사도: {cosine_sim:.6f}")
                print(f"  PTB 양수 비율: {ptb_pos_ratio:.3f}")
                print(f"  SaMi 양수 비율: {sami_pos_ratio:.3f}")
                print(f"  구별비율: {diff/avg_magnitude:.6f}" if avg_magnitude > 0 else "  구별비율: N/A")
                
                # 점수 계산 (코사인 유사도 중심)
                score = 0
                
                # 1. 코사인 유사도가 낮은가? (가장 중요)
                if cosine_sim < 0.8:
                    score += 4
                    print("  🏆 코사인 유사도 매우 우수 (+4점)")
                elif cosine_sim < 0.9:
                    score += 3
                    print("  ✅ 코사인 유사도 우수 (+3점)")
                elif cosine_sim < 0.95:
                    score += 2
                    print("  ⚠️ 코사인 유사도 보통 (+2점)")
                elif cosine_sim < 0.98:
                    score += 1
                    print("  ❌ 코사인 유사도 아직 높음 (+1점)")
                else:
                    print("  ❌ 코사인 유사도 너무 높음 (0점)")
                
                # 2. 임베딩 크기
                if avg_magnitude > 0.01:
                    score += 2
                    print("  ✅ 임베딩 크기 우수 (+2점)")
                elif avg_magnitude > 0.001:
                    score += 1
                    print("  ⚠️ 임베딩 크기 보통 (+1점)")
                else:
                    print("  ❌ 임베딩 크기 작음 (0점)")
                
                # 3. 구별 능력
                if diff > 0.01:
                    score += 2
                    print("  ✅ 구별 능력 우수 (+2점)")
                elif diff > 0.001:
                    score += 1
                    print("  ⚠️ 구별 능력 보통 (+1점)")
                else:
                    print("  ❌ 구별 능력 부족 (0점)")
                
                print(f"\n  📊 총점: {score}/8")
                
                # 코사인 유사도 개선 정도 계산
                cos_improvement = (1 - cosine_sim) * 100  # 0에 가까울수록 좋음
                print(f"  📈 코사인 유사도 개선도: {cos_improvement:.2f}%")
                
                if score >= 6:
                    status = "🏆 매우 우수함"
                elif score >= 4:
                    status = "✅ 우수함"
                elif score >= 2:
                    status = "⚠️ 보통"
                else:
                    status = "❌ 개선 필요"
                
                print(f"  📈 평가: {status}")
                
                # 코사인 유사도 기준으로 최고 모델 선택
                if cos_improvement > best_cos_score:
                    best_cos_score = cos_improvement
                    best_encoder = name
        
        # 최종 추천
        print(f"\n{'='*70}")
        print(f"🎯 코사인 유사도 개선 최종 결과")
        print(f"{'='*70}")
        
        if best_encoder:
            print(f"🏆 최고 다양성: {best_encoder}")
            print(f"🎯 코사인 유사도 개선도: {best_cos_score:.2f}%")
            
            if best_cos_score > 15:  # 코사인 유사도 0.85 미만
                print(f"✅ {best_encoder}를 강력히 추천합니다!")
                print(f"   코사인 유사도가 크게 개선되었습니다.")
            elif best_cos_score > 5:
                print(f"⚠️ {best_encoder}가 최선이지만 추가 개선 필요합니다.")
            else:
                print(f"❌ 여전히 코사인 유사도가 높습니다.")
                print(f"   더 근본적인 해결책이 필요합니다.")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    test_diversity_encoders()