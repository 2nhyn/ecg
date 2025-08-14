import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DebugSimpleECGEncoder(nn.Module):
    """디버깅 기능이 추가된 SimpleECGEncoder"""
    def __init__(self, in_channels=12, dropout=0.1, debug=True):
        super().__init__()
        self.debug = debug
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer1 = self._make_layer(64, 128, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.in_dim = 512
        self.dropout = nn.Dropout(dropout)
    
    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_c, out_c, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_c, out_c, 1))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )
    
    def _debug_print(self, x, stage_name):
        """디버깅 정보 출력"""
        if self.debug:
            print(f"  {stage_name}: shape={x.shape}, mean={x.mean().item():.6f}, "
                  f"std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
    
    def forward_features_debug(self, x):
        """각 단계별 출력을 확인할 수 있는 forward"""
        self._debug_print(x, "Input")
        
        # Stem
        x = self.stem(x)
        self._debug_print(x, "After Stem")
        
        # Layer 1
        x = self.layer1(x)
        self._debug_print(x, "After Layer1")
        
        # Layer 2
        x = self.layer2(x)
        self._debug_print(x, "After Layer2")
        
        # Layer 3
        x = self.layer3(x)
        self._debug_print(x, "After Layer3")
        
        # Global Pooling
        feat = self.global_pool(x).squeeze(-1)
        self._debug_print(feat, "After Global Pool")
        
        # Dropout
        feat = self.dropout(feat)
        self._debug_print(feat, "After Dropout")
        
        return feat
    
    def forward_features(self, x):
        """일반 forward (디버깅 없음)"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        feat = self.global_pool(x).squeeze(-1)
        feat = self.dropout(feat)
        return feat
    
    def forward(self, x, debug_mode=None):
        """forward 함수 - debug_mode로 제어 가능"""
        if debug_mode is None:
            debug_mode = self.debug
            
        if debug_mode:
            return self.forward_features_debug(x)
        else:
            return self.forward_features(x)


def test_with_real_ecg_data():
    """실제 ECG 데이터로만 테스트 - 랜덤 데이터 절대 사용 안함"""
    print("=== 실제 ECG 데이터로만 테스트 ===")
    
    try:
        # 실제 데이터 로드 (eval 데이터 사용)
        signals = np.load("/Data/nh25/ECG/0_Data/processed_eval_data/signals.npy")
        sources = np.load("/Data/nh25/ECG/0_Data/processed_eval_data/sources.npy", allow_pickle=True)
        
        print(f"로드된 신호 개수: {len(signals)}")
        print(f"신호 shape: {signals[0].shape}")
        
        # 서로 다른 소스의 신호들 찾기
        ptb_indices = [i for i, s in enumerate(sources) if 'ptb' in str(s).lower()]
        sami_indices = [i for i, s in enumerate(sources) if 'sami' in str(s).lower()]
        code_indices = [i for i, s in enumerate(sources) if 'code' in str(s).lower()]
        
        print(f"PTB 신호: {len(ptb_indices)}개")
        print(f"SaMi 신호: {len(sami_indices)}개") 
        print(f"CODE 신호: {len(code_indices)}개")
        
        # 테스트할 신호들 선택 (각 소스에서 여러 개씩)
        test_signals = []
        test_sources = []
        
        # PTB에서 3개
        for i in range(min(3, len(ptb_indices))):
            idx = ptb_indices[i]
            test_signals.append(signals[idx])
            test_sources.append(f"PTB_{i}")
        
        # SaMi에서 3개
        for i in range(min(3, len(sami_indices))):
            idx = sami_indices[i]
            test_signals.append(signals[idx])
            test_sources.append(f"SaMi_{i}")
        
        # CODE에서 3개
        for i in range(min(3, len(code_indices))):
            idx = code_indices[i]
            test_signals.append(signals[idx])
            test_sources.append(f"CODE_{i}")
        
        print(f"\n총 {len(test_signals)}개 신호로 테스트")
        
        # 신호들이 실제로 다른지 확인
        print(f"\n=== 신호 차이 확인 ===")
        for i in range(min(5, len(test_signals))):
            sig = test_signals[i]
            print(f"{test_sources[i]}: mean={sig.mean():.6f}, std={sig.std():.6f}, min={sig.min():.6f}, max={sig.max():.6f}")
        
        # 서로 다른 소스 간 차이
        if len(test_signals) >= 3:
            ptb_sig = test_signals[0]  # PTB
            sami_sig = test_signals[3] if len(test_signals) > 3 else test_signals[1]  # SaMi 또는 다른 것
            
            diff = np.abs(ptb_sig - sami_sig).mean()
            print(f"\n서로 다른 소스 간 차이: {diff:.6f}")
            
            if diff < 1e-6:
                print("❌ 신호가 거의 동일함! 데이터 문제")
                return None, None
            else:
                print("✅ 신호가 다름. 모델 테스트 진행")
        
        return test_signals, test_sources
        
    except Exception as e:
        print(f"❌ 실제 데이터 로드 실패: {e}")
        print("경로를 확인하거나 데이터가 존재하는지 확인하세요.")
        return None, None


def test_simple_model_with_real_data():
    """SimpleECGEncoder를 실제 데이터로 테스트"""
    
    # 실제 데이터 로드
    test_signals, test_sources = test_with_real_ecg_data()
    if test_signals is None:
        return
    
    print(f"\n=== SimpleECGEncoder 실제 데이터 테스트 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DebugSimpleECGEncoder(debug=True).to(device)
    model.eval()
    
    embeddings = []
    embedding_sources = []
    
    # 각 신호별로 임베딩 생성
    for i, (sig, source) in enumerate(zip(test_signals[:6], test_sources[:6])):  # 최대 6개
        print(f"\n{'='*50}")
        print(f"테스트 중: {source}")
        print(f"{'='*50}")
        
        # 신호 전처리
        if sig.shape[0] != 12:  # (T, C) -> (C, T)
            sig = sig.T
        
        sig_tensor = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model(sig_tensor, debug_mode=True)
            embeddings.append(emb.cpu().numpy())
            embedding_sources.append(source)
    
    # 임베딩 비교 분석
    print(f"\n{'='*60}")
    print(f"=== 임베딩 분석 ===")
    print(f"{'='*60}")
    
    for i, (emb, source) in enumerate(zip(embeddings, embedding_sources)):
        print(f"{source}: mean={emb.mean():.8f}, std={emb.std():.8f}")
        print(f"  범위: {emb.min():.8f} ~ {emb.max():.8f}")
        print(f"  첫 5개 값: {emb[0][:5]}")
        print()
    
    # 모든 쌍별 비교
    print(f"=== 임베딩 간 차이 분석 ===")
    
    same_source_diffs = []
    diff_source_diffs = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            emb1, emb2 = embeddings[i], embeddings[j]
            source1, source2 = embedding_sources[i], embedding_sources[j]
            
            # L1 차이
            diff = np.abs(emb1 - emb2).mean()
            
            # 코사인 유사도
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            cos_sim = np.dot(emb1_norm.flatten(), emb2_norm.flatten())
            
            print(f"{source1} vs {source2}:")
            print(f"  차이: {diff:.8f}, 코사인 유사도: {cos_sim:.6f}")
            
            # 같은 소스인지 다른 소스인지 구분
            source1_type = source1.split('_')[0]
            source2_type = source2.split('_')[0]
            
            if source1_type == source2_type:
                same_source_diffs.append(diff)
                print(f"  -> 같은 소스 ({source1_type})")
            else:
                diff_source_diffs.append(diff)
                print(f"  -> 다른 소스 ({source1_type} vs {source2_type})")
            print()
    
    # 통계 분석
    print(f"=== 최종 통계 ===")
    if same_source_diffs:
        print(f"같은 소스 간 평균 차이: {np.mean(same_source_diffs):.8f} ± {np.std(same_source_diffs):.8f}")
    if diff_source_diffs:
        print(f"다른 소스 간 평균 차이: {np.mean(diff_source_diffs):.8f} ± {np.std(diff_source_diffs):.8f}")
    
    # 결론
    print(f"\n=== 결론 ===")
    if embeddings:
        avg_magnitude = np.mean([np.abs(emb).mean() for emb in embeddings])
        avg_diff = np.mean(diff_source_diffs) if diff_source_diffs else 0
        
        print(f"임베딩 평균 크기: {avg_magnitude:.8f}")
        print(f"서로 다른 소스 간 평균 차이: {avg_diff:.8f}")
        
        if avg_magnitude < 1e-4:
            print("❌ 임베딩 크기가 너무 작음 - 정보 소실!")
        elif avg_diff < 1e-6:
            print("❌ 서로 다른 신호의 임베딩이 너무 유사함!")
        else:
            print("✅ 정상적인 임베딩 생성됨")


if __name__ == "__main__":
    test_simple_model_with_real_data()