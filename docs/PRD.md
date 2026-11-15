# Product Requirements Document (PRD)
## Project: Jeju Renewable Energy Forecast (J-REF)

---

## 1. 개요 (Overview)

| 항목 | 내용 |
|------|------|
| **프로젝트명** | J-REF (Jeju Renewable Energy Forecast) |
| **목표** | 제주도 태양광·풍력 발전량(KPX 전력거래량)을 하루 전(D-1 10시 제출) 조건으로 예측하고, 대시보드로 시각화 |
| **핵심 가치** | KPX 예측오차율(FER)로 평가한 하루 전 예측 정확도. 정산 구간(2021~ ≤6%, 2025 개정 육지 ≤4%)은 비교 기준으로 쓴다 (제주 예측제도는 2025-06-01 종료) |
| **타겟 사용자** | 전력거래소(KPX) 관계자, 신재생에너지 발전 사업자, 에너지 연구원 |

---

## 2. 문제 정의 (Problem Statement)

### 2.1 현황 분석
- 제주도는 날씨 변동성이 커서 기존의 광역 기상 데이터 기반 예측은 오차가 큼
- 출력 제어(Curtailment) 발생 시점을 예측하지 못하면 경제적 손실 발생
- 기존 서비스는 '현황' 중심이며, '고정밀 예측' 기능은 부족함

### 2.2 기회 영역
- **Micro-climate 모델링**: 제주도 특유의 지형(한라산, 해안, 중산간)에 따른 국지 기상 반영
- **Dual-Stream Architecture**: 태양광/풍력의 물리적 특성 차이를 반영한 전문화된 모델
- **Real-time Dashboard**: 예측값과 실측값의 실시간 비교 시각화

---

## 3. 주요 기능 (Key Features)

### 3.1 데이터 수집 및 가공 (Data Pipeline)

```
[Raw Data Sources]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  기상청 데이터                                           │
│  - ASOS (종관기상관측): 주요 기상대 관측값               │
│  - AWS (방재기상관측): 제주도 내 상세 관측망             │
│  - LDAPS (수치예보모델): 격자형 예보 데이터              │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  전력거래소(KPX) 데이터                                  │
│  - 시간대별 발전량 (태양광/풍력)                         │
│  - 발전소 위치 정보                                      │
│  - 계통 연계 정보                                        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
[Feature Engineering] → [Model Training] → [Prediction]
```

#### 데이터 소스 상세

| 데이터 | 출처 | 주기 | 활용 |
|--------|------|------|------|
| ASOS | 기상청 | 1시간 | 기본 기상 변수 |
| AWS | 기상청 | 10분 | 상세 지역 기상 |
| LDAPS | 기상청 API허브 | 6시간 | 예보 기반 예측 (미확보, 향후 과제) |
| JMA MSM 과거 예보 | Open-Meteo Previous Runs API (CC BY 4.0, 비상업) | 1시간 | 하루 전 조건 학습·평가 (확보) |
| 발전량 | data.go.kr 15065269 (KPX 전력거래량) | 1시간 | 타겟 변수 (2024년) |

> 요구사항: 발전량과 기상은 **구간 끝 시각**으로 맞춘다 (KPX 거래시간 h = (h−1)~h시, ASOS 일사 = 직전 1시간 누적). 2024-06-01 이후 공개 거래량의 포괄 범위 축소를 반영한다. 상세: [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md)

### 3.2 예측 모델링 (AI Core)

> **현황 (2026-11-07)**: 아래 Dual-Stream 딥러닝 구조는 초기 설계안이다. 운영 경로는 태양광·풍력 모두 **LightGBM**이다 (태양광 V2 `island_mean`, 풍력 Phase 5). Bi-LSTM·Transformer는 합성 데이터 초기 실험에만 쓰였고, TFT와 그래디언트 부스팅 메타 앙상블은 구현하지 않았다 (PyTorch 앙상블 `src/models/ensemble.py`는 초기 실험 코드로만 있다).

#### 3.2.1 Dual-Stream Model Architecture (초기 설계안)

```
                    ┌──────────────────────┐
                    │   Input Features     │
                    │  (Weather + Time)    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   Solar-Net     │               │   Wind-Net      │
    │   (LSTM/GRU)    │               │   (Transformer) │
    │                 │               │                 │
    │ Features:       │               │ Features:       │
    │ - 일사량        │               │ - 풍속          │
    │ - 운량          │               │ - 풍향          │
    │ - 기온          │               │ - 기압          │
    │ - 습도          │               │ - 난류강도      │
    └────────┬────────┘               └────────┬────────┘
             │                                  │
             └────────────┬─────────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Ensemble Layer     │
               │  (Weighted Average  │
               │   or Meta-Learner)  │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Final Prediction   │
               │  (MW/hour)          │
               └─────────────────────┘
```

#### 3.2.2 모델 상세 스펙

| 모델 | 대상 | 설계안 | 현재 구현 | 상태 |
|------|------|--------|-----------|------|
| Solar-Net | 태양광 | Bi-LSTM + Attention | LightGBM V2 (경사면 일사 물리 피처, 직전 1시간 기상) | 운영 (API `island_mean`) |
| Wind-Net | 풍력 | Temporal Fusion Transformer | LightGBM Phase 5 (V2 피처 + 2020-2023 패턴) | 운영, 재학습 필요 |
| Ensemble | 통합 | Gradient Boosting Meta | `src/models/ensemble.py` (PyTorch 가중 평균/메타 러너, 초기 실험) | 운영 미사용, GB 메타 미구현 |

**예측 시점 요구사항**: KPX 하루 전 제출(D-1 10시 마감)에 맞춰, 입력은 그 시점에 확실히 있는 수치예보(48시간 전 발표분)와 D-2까지의 발전량만 쓴다. 관측 기상 입력 결과는 상한으로만 보고한다.

### 3.3 시각화 대시보드 (Frontend)

#### 3.3.1 Map View
- 제주도 지도 위에 발전소 위치 마커 표시
- 실시간 예측 발전량 히트맵 오버레이
- 지역별 발전 효율 색상 구분

#### 3.3.2 Chart View
- 실측값 vs 예측값 시계열 비교 그래프
- 오차율 표시: 발전량은 KPX 예측오차율(FER), 수요는 MAPE
- 24시간/48시간/72시간 예측 구간 선택

#### 3.3.3 Alert System
- 출력 제어(Curtailment) 예상 시점 경고
- 급격한 발전량 변동 알림
- 예측 정확도 임계치 이탈 알림

---

## 4. 기술 스택 (Tech Stack)

### 4.1 Backend & AI

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| Language | Python | 3.10+ | 메인 개발 언어 |
| ML Model | LightGBM | - | 운영 예측 모델 |
| DL Framework | PyTorch | 2.0+ | 초기 딥러닝 실험 (선택) |
| ML Library | Scikit-learn | 1.3+ | 전처리, 앙상블 |
| Data | Pandas, NumPy | latest | 데이터 처리 |
| API | FastAPI | 0.100+ | REST API 서빙 |

### 4.2 Frontend & Visualization

| 분류 | 기술 | 용도 |
|------|------|------|
| Dashboard | Streamlit | MVP 대시보드 |
| Map | PyDeck / Folium | 지도 시각화 |
| Chart | Plotly | 인터랙티브 차트 |

### 4.3 Infrastructure

| 분류 | 기술 | 용도 |
|------|------|------|
| Container | Docker | 환경 표준화 |
| VCS | Git | 버전 관리 |
| GPU | Apple MPS / CUDA | 초기 딥러닝 실험 가속 (LightGBM은 불필요) |

---

## 5. 성공 지표 (Metrics)

### 5.1 모델 성능 지표

| 지표 | 목표값 | 측정 방법 |
|------|--------|-----------|
| **FER** (태양광·풍력, 주 지표) | 하루 전 조건 ≤ 6% (내부 목표, 2021~ 정산 구간 기준. 2025 개정 육지 ≤ 4%는 비교값) | \|예측 − 실측\| ÷ 설비용량 × 100, 실측 ≥ 용량 10% 시간만 (전력시장운영규칙 제14.1.1조) |
| 스킬 점수 | > 0 | 1 − RMSE / RMSE(48시간 전 지속성) |
| MAE·RMSE·R² | 보조 | MW 단위 참고값 |

- 발전량 MAPE·R²는 주 지표로 쓰지 않는다 (MAPE는 저출력 시간에 폭주, R²는 평가 기간 분산에 좌우). 수요 예측은 MAPE를 쓴다.
- 정의·분모(설비용량)·정산 구간: [solar_evaluation_criteria.md](solar_evaluation_criteria.md)

**현재 달성 (태양광, 테스트 2024-11-07~12-31, 분모 359.4 MW)**: 관측 기상 입력 FER 4.18% (`island_mean`, 상한), 하루 전 조건(48시간 전 예보) 8.53~8.65% → 하루 전 목표 미달. 풍력은 재학습 후 FER로 평가한다.

### 5.2 시스템 성능 지표

| 지표 | 목표값 | 비고 |
|------|--------|------|
| API 응답 시간 | < 500ms | P95 기준 |
| 모델 추론 시간 | < 100ms | 단일 예측 기준 |
| 데이터 갱신 주기 | 1시간 | 실시간 파이프라인 |

---

## 6. 마일스톤 (Milestones)

### Phase 1: Foundation
- [x] 데이터 파이프라인 구축 (기상청 ASOS + KPX 2024)
- [x] EDA 및 베이스라인 모델 구축
- [x] 기본 API 서버 구현

### Phase 2: Model Development
- [x] 태양광 모델 (LightGBM V2, FER 평가)
- [x] 풍력 모델 (LightGBM Phase 5)
- [ ] Ensemble 레이어 (보류)

### Phase 3: Integration
- [x] 대시보드 MVP (KPX 실측 데이터, `streamlit_app.py`)
- [x] API 1.1.0 (태양광 V2 서빙, 구간 시각 명시, 입력 검증)
- [x] API 1.3.0 (2026-11-15): 계통 라벨 서빙 모델, 발전 수준 입력 `level_history`, Render 배포 (https://jref-api.onrender.com)
- [x] 하루 전 예측 파이프라인 `scripts/predict_dayahead.py` — 수치예보(GFS·ECMWF)를 받아 태양광 10시·17시, 풍력 10시 제출용 예측. 실행은 수동(스케줄러 연동은 안 함)
- [ ] 실시간 예측 파이프라인 연동 (수치예보 자동 수집 스케줄링)
- [ ] 알림 시스템 구현

### Phase 4: Optimization
- [x] 하루 전 조건(JMA MSM 예보) 평가
- [ ] 풍력 재학습 (구간 끝 라벨, 2024-06 포괄 범위 단절 처리)
- [ ] 다중 예보 모델·17시 2차 제출·예보 일사 MOS 보정
- [ ] 2025년 자료로 새 hold-out 검증
- [ ] main 병합 및 배포 (현재 `feature/solar-v2-kpx-fer` 브랜치)

---

## 7. 리스크 및 대응 방안

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| 데이터 품질 이슈 | 높음 | 다중 데이터 소스 활용, 이상치 탐지 파이프라인 |
| 공개 거래량 포괄 범위 변화 (2024-06-01~) | 높음 | 분모를 실효 설비용량으로 보정, 단절 이후 구간 별도 처리 |
| 테스트 구간 재사용 (2024-11~12 이미 2회 사용) | 중간 | 모델 선택은 CV, 최종 검증은 2025년 hold-out |
| 모델 과적합 | 중간 | 교차 검증, 정규화, Early Stopping |
| API 한도 초과 | 중간 | 캐싱 전략, 배치 요청 최적화 |
| 계산 자원 부족 | 낮음 | MPS 활용, 모델 경량화 |

---

## 8. 참고 자료

- [햇빛바람지도 (60hz.io)](https://map.60hz.io/page/map) - 벤치마크 서비스
- [기상자료개방포털](https://data.kma.go.kr/) - 기상 데이터
- [전력통계정보시스템](https://epsis.kpx.or.kr/) - 발전량 데이터
- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363) - 초기 설계 참조 (미구현)
- [KPX 공개 발전량 (data.go.kr 15065269)](https://www.data.go.kr/data/15065269/fileData.do) - 타겟 데이터
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) - 과거 수치예보

---

*Last Updated: 2026-11-07*
*Document Version: 1.1*
