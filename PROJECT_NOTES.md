# UAV-MEC MADDPG 구현 — 결정 사항, 변형, 인사이트

Shi et al. 2026 *"A Deep Reinforcement Learning Based Approach for Optimizing
Trajectory and Frequency in Energy Constrained Multi-UAV Assisted MEC System"*
를 PyTorch로 재현하면서 마주친 모호한 설계, 우리가 내린 결정, 그리고
실험을 통해 얻은 인사이트를 정리.

---

## 1. 알고리즘 개선 (버전별)

논문 그대로 시작 → 발견한 문제 → 수정 반복.

### v1 — 논문 충실 재현
- MADDPG 표준: 각 UAV마다 독립 Actor/Critic + 중앙집중 Critic
- 보상: `r_m = N_m × F_sd_m × F_uav - 10 × penalty_count`
- 에너지: 초과 시 UAV 즉시 비활성화 (논문 명시 없음, 우리 가정)
- noise σ: 0.30 → 0.05 선형 감쇠

**결과**: F_uav = 0.33 — **Lazy Agent 심각** (UAV 1만 일하고 0, 2는 정지)

### v2 — 파라미터 공유 + UAV ID
변경:
- 독립 Actor 3개 → **단일 공유 Actor**
- 관측에 UAV one-hot ID 추가 (obs_dim 57 → 60)
- noise σ_final 0.05 → 0.10 (탐색 더 길게 유지)

**결과**: F_uav = 0.78 (Lazy 부분 해소) / 하지만 **return 음수(-100), bits 절반**.
공정성은 좋아졌지만 전체 성능은 더 나빠짐. 페널티 회피로 움직임 자체가 줄어든 패턴.

### v3 — 페널티 축소 + 개인 보너스 + Soft 에너지 제약
변경:
- penalty: **10 → 1** (위험 회피 완화)
- **개인 기여 보너스 추가**: `r_m += 0.5 × N_m` (F_sd × F_uav 곱셈 우회로 직접 신호)
- 에너지 정책: 소진 시 정지 → **계속 동작 + 슬롯당 페널티** (논문 C7에 충실)
- 에피소드: 50k → 70k → 다시 50k

**결과**: F_uav = **0.98** (4시드 평균), bits 7.7e8 ± 0.6e8, return +640.
모든 지표가 v1, v2보다 우월. 다만 보상 함수 변경이라 return 직접 비교는 부정확.

### v4 — Fairness 항 제거 (Ablation)
변경:
- `use_fairness_reward: false`: `N × F_sd × F_uav` 항 제거
- 개인 보너스(0.5 × N)와 페널티만 유지

**목적**: Fairness 곱셈이 학습에 실제 기여하는가 검증.
**현재**: 4시드 학습 진행 중.

### MATD3 — 코드만 구현
변경:
- Twin critics (clipped double Q)
- Target policy smoothing
- Delayed actor + target updates (policy_delay=2)
- `agents/matd3.py` (신규), train.py / visualize에 분기 추가

학습은 아직 진행 안 함.

---

## 2. 추가한 인프라/스크립트

| 파일 | 목적 |
|------|------|
| `agents/matd3.py` | TD3의 멀티에이전트 변형 |
| `plot_multiseed.py` | 여러 시드 평균±표준편차 학습곡선 |
| `analyze_task_processing.py` | 생성/처리 태스크 비율, 미처리 히트맵 |
| `analyze_energy.py` | UAV별 에너지 사용량, 위반률 통계 |
| `analyze_frequency.py` | 학습된 정책의 주파수 선택 분포 |

환경에 추가한 추적 변수:
- `generated_per_sd`, `served_per_sd` (태스크 처리율 분석)
- `energy_violated`, `violation_slot`, `cumulative_excess`, `peak_excess`, `slot_excess` (에너지 위반 추적)

---

## 3. 논문이 모호하거나 미명시한 항목 (우리가 결정한 것)

### 3.1 에너지 모델 단위
- **k1 = 10**: 단위 미명시. 표준 CMOS의 k_eff (10^-27 정도)와 10^28배 차이
- **f_max = 10^6, f_min > 0**: Hz/MHz/GHz 단위 미명시
- **cyc = 0.125 cycles/unit-data**: cycles/bit인지 cycles/byte인지 명시 안 됨 (우리는 cycles/bit로 가정)
- **p_tran = 5**: W 단위 미명시 (우리 초기 가정 0.1, 후에 5로 수정 필요)
- **w_uav = 5 kg**: kg 단위로 명시됨
- **E_uav = 2000 mAh**: 실제 배터리 단위, 다른 정규화 단위와 혼합

**결정**: 논문이 추상/혼합 단위계임을 인정. `energy_scale = 1e16`으로 정규화.

### 3.2 보상 함수 세부
- **N 스케일링**: 논문 미명시. `reward_scale_N_divisor = 1e6` 가정 (1 SD 서비스 ≈ 1 단위)
- **에너지 초과 시 처리**: 논문이 C7 제약만 정의, 위반 시 행동 미명시 → 슬롯당 페널티로 구현

### 3.3 환경 디테일
- **UAV 초기 위치**: 논문 p.323에 명시 — (10,10), (10,90), (90,90) ✓
- **매 에피소드 SD 위치 재샘플**: 우리 결정 (랜덤). 논문 미명시
- **α 학습 분포**: 우리는 uniform. 논문은 평가 5가지 분포만 명시 (g1~g5)

### 3.4 네트워크 구조
- **은닉층**: 우리 [256, 128] / 논문 미명시
- **활성함수**: ReLU + 최종 Tanh
- **초기화**: PyTorch 기본 (Xavier 등)

### 3.5 알고리즘 디테일
- **Actor 독립 vs 공유**: 우리 v2+ 공유 / 원본 MADDPG는 독립 — 우리만의 변형
- **탐색 노이즈**: Gaussian, 선형 감쇠 σ 0.3 → 0.10 (논문 미명시)
- **버퍼 크기, 배치 크기**: 100k, 256 (우리 결정)
- **τ (soft update)**: 0.01

### 3.6 페널티 정의
- **penalty=10**: 경계 위반 + 충돌 모두 동일 가중치 (논문 명시)
- 우리 v3+에서 1로 축소 → 논문 일탈

---

## 4. 우리만의 변형 (논문에서 명백히 벗어난 부분)

| 항목 | 논문 | 우리 v3 | 이유 |
|------|------|--------|------|
| Actor 개수 | M개 독립 | 1개 공유 | Lazy agent 완화 |
| 관측 차원 | 57 | 60 (+UAV ID) | 공유 Actor에서 정체성 구분 |
| 보상 추가 | - | `+ 0.5 × N` | Lazy agent 직접 해소 |
| 페널티 | 10 | 1 | 위험 회피 완화 |
| 에너지 위반 시 | 비활성화 (가정) | 소프트 페널티 | 논문 C7 충실 + 학습 신호 유지 |
| 이동 에너지 | 식 그대로 | `× 1e12` 가중치 | 이동/계산 비율 균형 |
| 에너지 초과 페널티 | 명시 없음 | base 3 + per-excess 0.5 | 의미있는 제약 |

---

## 5. 실험을 통해 얻은 인사이트

### 5.1 Lazy Agent의 진짜 원인
**가설 1 (틀림)**: 에너지 소모가 두려워서
**가설 2 (맞음)**: 보상 신호 구조 문제

- N=0인 UAV는 보상 = 0 (F_sd × F_uav 곱해도 0)
- 움직이면 boundary/collision penalty 위험
- "가만히 있는 것이 로컬 최적" → 자기충족적 패턴

**해결**: 페널티 축소 + 개인 보너스로 직접 양의 신호 부여.

### 5.2 파라미터 공유의 한계
공유 Actor + UAV ID one-hot으로도 lazy agent 완전 해소 안 됨:
- 네트워크가 "ID에 따라 다른 행동" 학습 가능
- ID = [1,0,0]인 UAV는 정지, [0,1,0]은 활동 — 같은 가중치라도 가능
- 즉, 파라미터 공유는 **lazy를 구조적으로 막지 않음**

### 5.3 에너지 모델의 구조적 문제
논문 식 그대로:
```
E_com ∝ k1 × f × N × cyc   (1e16 규모)
E_oper_move ∝ d/v          (1e3 규모)
ratio E_com : E_move = 10^9 : 1
```
**비행은 거의 무료**. 우리가 movement_mult로 보정해도 의도적 trade-off는 아님.

### 5.4 주파수 선택은 사실상 학습 안 됨
MADDPG의 평균 f: **5.0e5** (= 50% of f_max)
RANDOM의 평균 f: **4.98e5** (= 49.8%)
GREEDY의 평균 f: **1.0e6** (= 100%, 하드코딩)

→ MADDPG의 f 선택이 RANDOM과 통계적으로 동일. **학습 실패**.

원인: 환경에서 f가 N(처리량)에 영향을 주지 않음. f는 처리 시간만 바꿈, 시간은 슬롯 단위로 고정 → trade-off 없음.

### 5.5 GREEDY가 MADDPG보다 좋아 보이는 이유
- **정보 격차**: GREEDY는 모든 SD의 현재 큐 크기를 봄, MADDPG는 못 봄
- **MADDPG는 SD 위치도 직접 관측 못 함** (누적 서비스 이력으로만 추론)
- 그럼에도 MADDPG가 GREEDY의 **85~90% 성능** 달성 (정보 격차 고려 시 합리적)
- bits/energy 효율은 MADDPG가 1.85배 더 좋음 (단 이는 f 학습 실패의 부작용)

### 5.6 에너지 예산이 너무 넉넉
50 에피소드 분석 결과:
- GREEDY 최대 사용: 16.2% of E_uav
- MADDPG 평균 사용: 5.9~9.1%
- **0 위반** (50 에피소드)

E_uav=2000은 사실상 제약이 아님. 우리 환경에서 에너지 학습 신호가 약함.

### 5.7 다중 시드의 중요성
단일 시드(seed=42) vs 4시드 평균:
- 단일: bits 5.5e8 ~ 8.6e8 (시드별 변동)
- 4시드 평균: bits 7.72e8 ± 0.60e8 (CV 8%)
- F_uav는 시드 간 일관성 매우 높음 (CV 0.6%)

→ RL 결과는 **최소 4시드**, 권장 10시드 평균 보고 필요.

### 5.8 단일 에피소드 평가의 변동성
같은 모델, 다른 평가 시드:
- seed 100 단일: MADDPG 85.3% > GREEDY 81.6%
- seed 100-109 평균: MADDPG 73% < GREEDY 86%

→ **단일 에피소드 결과는 신뢰 불가**. 50+ 에피소드 필요.

---

## 6. 남은 문제 / 한계

### 6.1 알고리즘 구현 검증 안 됨
우리 MADDPG, MATD3가 정확한 구현인지 검증 안 함:
- 표준 환경(MPE simple_spread 등)에서 알려진 결과 재현 안 함
- SB3와 단일 에이전트(DDPG) 비교 안 함
- 그래디언트 흐름, target update 등 미세 검증 안 함

### 6.2 평가의 통계적 약함
- 한 모델에 10 episode 평가 → 표준편차 크게 나옴
- 4시드 학습 후에도 신뢰구간이 큼

### 6.3 환경 단위/스케일 모호성
- E_uav (mAh) ↔ E_com (정규화) 단위 불일치
- 우리가 energy_scale로 hack한 부분이 결과에 영향 줄 수 있음

### 6.4 보상 구조 변경의 부작용
v3, v4의 보상은 논문과 다름:
- "이것이 v1보다 좋다"는 비교가 부분적으로만 valid
- bits, F_uav 같은 환경 지표 비교만 직접 가능, return 비교는 불가

### 6.5 GREEDY 베이스라인이 너무 강함
- 오라클 정보 사용
- 충돌 회피, 에너지 관리 포함된 "스마트 GREEDY"
- 논문 원본 GREEDY는 더 단순할 가능성 → 우리가 너무 강한 baseline 구현

---

## 7. 결정 우선순위 (앞으로)

진행 가능한 작업, 우선순위 순:

### 즉시 가능
1. v4 학습 완료 후 결과 분석 (~4시간 후)
2. 시드별 에너지 사용 패턴 비교 (analyze_energy.py)
3. v3 vs v4 F_sd, F_uav 비교로 fairness 항 효과 검증

### 단기
4. 에너지 제약 강화 (energy_scale 1e14 또는 movement_mult 1e13)
5. f가 의미 있는 환경 변수가 되도록 설계 변경
6. GREEDY를 더 단순하게 구현하여 공정 비교

### 중기
7. MATD3 학습 실행 후 MADDPG와 비교
8. 표준 MARL 벤치마크(MPE)에서 우리 MADDPG 검증
9. SB3와 단일 에이전트 비교

### 장기
10. 평가 프로토콜 강화 (50+ episodes per scenario)
11. 학습 중 주기적 평가 곡선 추가
12. 보상 구조 다양화 ablation study

---

## 8. 디렉터리 / 결과물

```
/home/cai/lmg/UAV-MEC Baseline/
├── env/uav_mec_env.py          # 환경
├── agents/
│   ├── maddpg.py               # MADDPG (파라미터 공유)
│   ├── matd3.py                # MATD3 (방금 구현)
│   ├── ddpg.py                 # DDPG 베이스라인
│   ├── networks.py             # Actor, Critic
│   └── replay_buffer.py
├── baselines/                  # RANDOM, CIRCLE, GREEDY
├── train.py                    # 학습 entry point
├── evaluate.py                 # 5시나리오 평가
├── visualize_trajectory.py     # 궤적 시각화
├── plot_training.py            # 학습 곡선
├── plot_multiseed.py           # 다중 시드 평균
├── analyze_task_processing.py  # 처리율 / 히트맵
├── analyze_energy.py           # 에너지 사용 / 위반 분석
├── analyze_frequency.py        # 주파수 선택 분포
├── config.json                 # 전역 하이퍼파라미터
└── results/
    ├── v1/                     # 초기 MADDPG 결과
    ├── v2/                     # 파라미터 공유
    ├── v3/                     # 페널티 축소 + 보너스 (4시드)
    ├── baselines/              # RANDOM, CIRCLE, GREEDY 궤적
    ├── comparisons/            # 버전 간 비교
    ├── task_analysis_*/        # 처리율 분석
    ├── energy_analysis/        # 에너지 분석
    └── freq_analysis/          # 주파수 분석
```

---

## 9. 핵심 교훈

1. **논문 재현 ≠ 결과 재현**: 미명시 부분 결정이 결과를 크게 바꿈
2. **Lazy agent는 보상 설계 문제**: 알고리즘 변경보다 보상 직접 수정이 효과적
3. **단위 명시되지 않은 RL 환경은 정규화 단위로 봐야 함**
4. **에너지/주파수 trade-off가 환경에 없으면 학습되지 않음** (당연한데 종종 간과)
5. **단일 시드 결과는 신뢰 불가**, 최소 4시드 (권장 10시드)
6. **베이스라인 구현이 강하면 알고리즘 비교가 불공정해질 수 있음** (정보 격차 명시 필요)
7. **측정 없는 파라미터 튜닝은 추측**: 추적/로그 먼저, 결정 나중
