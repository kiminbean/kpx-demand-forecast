"""
Configuration Management for J-REF Project

환경 설정, 하이퍼파라미터, 경로 관리를 담당합니다.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch


# ========================
# Path Configuration
# ========================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


# ========================
# Device Configuration
# ========================

def get_device() -> torch.device:
    """
    최적의 연산 디바이스를 자동 선택합니다.

    우선순위: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: 선택된 디바이스
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ========================
# Data Configuration
# ========================

@dataclass
class DataConfig:
    """데이터 관련 설정"""

    # 시퀀스 길이
    solar_seq_length: int = 24  # 태양광: 24시간 입력
    wind_seq_length: int = 48   # 풍력: 48시간 입력

    # 예측 범위
    forecast_horizon: int = 24  # 24시간 예측

    # 데이터 분할 비율
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 결측치 처리
    max_missing_ratio: float = 0.1  # 10% 이상 결측 시 해당 구간 제외

    # 스케일링
    scaling_method: str = "standard"  # "standard" or "minmax"


# ========================
# Model Configuration
# ========================

@dataclass
class SolarModelConfig:
    """태양광 예측 모델 설정 (LSTM)"""

    input_size: int = 10  # 입력 피처 수
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention: bool = True


@dataclass
class WindModelConfig:
    """풍력 예측 모델 설정 (Transformer)"""

    input_size: int = 12  # 입력 피처 수
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1


@dataclass
class EnsembleConfig:
    """앙상블 모델 설정"""

    method: str = "weighted_average"  # "weighted_average" or "meta_learner"
    solar_weight: float = 0.5
    wind_weight: float = 0.5


# ========================
# Training Configuration
# ========================

@dataclass
class TrainingConfig:
    """학습 관련 설정"""

    # 기본 설정
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # 스케줄러
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_patience: int = 10

    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Gradient Clipping
    gradient_clip_val: float = 1.0

    # 체크포인트
    save_top_k: int = 3
    checkpoint_metric: str = "val_loss"

    # 재현성
    seed: int = 42


# ========================
# API Configuration
# ========================

@dataclass
class APIConfig:
    """API 서버 설정"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # 기상청 API
    kma_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("KMA_API_KEY")
    )

    # KPX API
    kpx_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("KPX_API_KEY")
    )


# ========================
# Feature Names
# ========================

SOLAR_FEATURES = [
    "temperature",      # 기온
    "humidity",         # 습도
    "solar_radiation",  # 일사량
    "cloud_cover",      # 운량
    "visibility",       # 시정
    "hour_sin",         # 시간 (sin)
    "hour_cos",         # 시간 (cos)
    "month_sin",        # 월 (sin)
    "month_cos",        # 월 (cos)
    "is_holiday",       # 공휴일 여부
]

WIND_FEATURES = [
    "wind_speed",       # 풍속
    "wind_direction",   # 풍향
    "temperature",      # 기온
    "pressure",         # 기압
    "humidity",         # 습도
    "turbulence",       # 난류 강도
    "hour_sin",         # 시간 (sin)
    "hour_cos",         # 시간 (cos)
    "month_sin",        # 월 (sin)
    "month_cos",        # 월 (cos)
    "is_holiday",       # 공휴일 여부
    "elevation",        # 고도
]


# ========================
# Instantiate Configs
# ========================

data_config = DataConfig()
solar_model_config = SolarModelConfig()
wind_model_config = WindModelConfig()
ensemble_config = EnsembleConfig()
training_config = TrainingConfig()
api_config = APIConfig()


# ========================
# Utility Functions
# ========================

def setup_directories() -> None:
    """프로젝트 디렉토리 구조를 생성합니다."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 시드 설정"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 설정 확인
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Data Config: {data_config}")
    print(f"Training Config: {training_config}")
