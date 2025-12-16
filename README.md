# Jeju Renewable Energy Demand Forecast (J-REF)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

제주도의 **태양광 및 풍력 발전량**을 딥러닝 기반으로 정밀하게 예측하는 시스템입니다.

기존의 통계적 기법이나 단순 시각화 서비스를 넘어, **제주도의 국지적 기상 특성(Micro-climate)을 반영한 고성능 예측 모델**을 구축하는 것을 목표로 합니다.

> **Target:** 시각화는 [햇빛바람지도](https://map.60hz.io/page/map)를 지향하되, 예측 성능(Accuracy)은 이를 초월합니다.

---

## Architecture Philosophy

**Clean Architecture** 원칙을 준수하여 유지보수성과 확장성을 확보합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│                   (Streamlit Dashboard, API)                     │
├─────────────────────────────────────────────────────────────────┤
│                        Application Layer                         │
│              (Training, Inference, Visualization)                │
├─────────────────────────────────────────────────────────────────┤
│                         Domain Layer                             │
│           (Solar-Net, Wind-Net, Ensemble Models)                 │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                        │
│            (Data Loaders, API Clients, File I/O)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

- **Modularity:** 데이터 전처리, 모델 학습, 추론 로직이 완벽히 분리
- **Strict Typing:** Python Type Hint를 적극 사용하여 코드 안정성 확보
- **Hybrid Modeling:** 태양광(Solar)과 풍력(Wind)의 물리적 특성 차이를 고려하여 별도의 모델을 학습 후 앙상블

---

## Directory Structure

```
kpx-demand-forecast/
├── .github/                   # CI/CD workflows
├── data/                      # 데이터 저장소 (Git ignore)
│   ├── raw/                   # 원천 데이터 (KPX, 기상청)
│   ├── processed/             # 전처리 완료 데이터
│   └── external/              # 외부 데이터 (휴일 정보 등)
├── docs/                      # 프로젝트 문서
│   └── PRD.md                 # 제품 요구사항 정의서
├── models/                    # 학습된 모델 체크포인트
├── notebooks/                 # EDA 및 프로토타이핑
├── src/                       # 핵심 소스 코드
│   ├── config.py              # 환경 설정
│   ├── data/                  # 데이터 로더
│   │   ├── weather_fetcher.py # 기상청 API 연동
│   │   └── kpx_loader.py      # KPX 데이터 로더
│   ├── features/              # Feature Engineering
│   ├── models/                # 모델 아키텍처
│   │   ├── solar_lstm.py      # 태양광 예측 모델
│   │   ├── wind_transformer.py# 풍력 예측 모델
│   │   └── ensemble.py        # 앙상블 모델
│   ├── training/              # 학습 루프
│   └── visualization/         # 시각화 유틸리티
├── api/                       # FastAPI 서버
├── tests/                     # 테스트 코드
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Dual-Stream Ensemble

태양광과 풍력은 물리적 특성이 완전히 다르므로, 각각 특화된 모델을 개발합니다.

```
                    Weather Features
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌───────────┐                   ┌───────────┐
    │ Solar-Net │                   │ Wind-Net  │
    │  (LSTM)   │                   │  (TFT)    │
    └─────┬─────┘                   └─────┬─────┘
          │                               │
          └───────────┬───────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │  Ensemble   │
               │   Layer     │
               └──────┬──────┘
                      │
                      ▼
              Final Prediction (MW)
```

| Model | Target | Architecture | Input Sequence |
|-------|--------|--------------|----------------|
| Solar-Net | 태양광 | Bi-LSTM + Attention | 24h |
| Wind-Net | 풍력 | Temporal Fusion Transformer | 48h |
| Ensemble | 통합 | Gradient Boosting | Model Outputs |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (MPS) 또는 NVIDIA GPU (CUDA)

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-repo/kpx-demand-forecast.git
   cd kpx-demand-forecast
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Data**

   `data/raw/` 폴더에 데이터 파일을 위치시킵니다:
   - KPX 발전량 데이터
   - 기상청 ASOS/AWS 데이터

---

## Usage

### Training

```bash
# 태양광 모델 학습
python -m src.training.train --model solar --epochs 50

# 풍력 모델 학습
python -m src.training.train --model wind --epochs 100

# 앙상블 모델 학습
python -m src.training.train --model ensemble
```

### Inference

```bash
# 단일 예측
python -m src.training.predict --date 2024-12-17

# 배치 예측
python -m src.training.predict --start 2024-12-17 --end 2024-12-24
```

### API Server

```bash
# 개발 서버 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# API 문서 확인
# http://localhost:8000/docs
```

### Dashboard

```bash
# Streamlit 대시보드 실행
streamlit run src/visualization/dashboard.py
```

---

## Data Sources

| Source | Data | URL |
|--------|------|-----|
| 기상청 | ASOS/AWS 관측 데이터 | [data.kma.go.kr](https://data.kma.go.kr/) |
| 기상청 | LDAPS 수치예보 | [data.kma.go.kr](https://data.kma.go.kr/) |
| KPX | 발전량 통계 | [epsis.kpx.or.kr](https://epsis.kpx.or.kr/) |

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| MAPE | < 10% | - |
| RMSE | Baseline -20% | - |
| R² Score | > 0.85 | - |

---

## Tech Stack

**Core**
- Python 3.10+
- PyTorch 2.0+
- Pandas, NumPy, Scikit-learn

**API & Visualization**
- FastAPI
- Streamlit
- Plotly, PyDeck

**Infrastructure**
- Docker
- Git

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [햇빛바람지도 (60hz.io)](https://map.60hz.io/page/map) - 벤치마크 서비스
- [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) - 모델 참조
- 기상청 및 한국전력거래소 공공데이터 제공

---

*Developed with Claude Code*
