"""
API Configuration
==================

환경 변수 및 설정 관리
"""

import os
from pathlib import Path
from typing import Literal, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # API 기본 설정
    APP_NAME: str = "J-REF API"
    APP_VERSION: str = "1.3.0"
    APP_DESCRIPTION: str = "제주도 신재생에너지 발전량 예측 API (태양광/풍력)"
    DEBUG: bool = False

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False

    # 모델 설정
    MODEL_DIR: Optional[str] = None
    WIND_MODEL_NAME: str = "wind_lgbm_fair.lgb"
    # API는 관측소 구분 없는 기상 1세트만 받으므로 섬 평균 변형을 쓴다.
    # 다른 변형(관측소별·발전 수준 lvl_p 사용)은 API 입력으로 피처를 만들 수 없어 설정 단계에서 막는다
    SOLAR_V2_VARIANT: Literal["island_mean"] = "island_mean"
    # 예측 대상. system = KPX 제주계통운영정보(입찰자원 포함, 2024-01~2026-10 학습),
    # public = 공개 거래량의 태양광·풍력(입찰자원 제외, 2024년 학습) — 이전 서빙 모델
    SERVING_LABEL: Literal["system", "public"] = "system"

    # CORS 설정
    CORS_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 기상청 API
    KMA_API_KEY: Optional[str] = None

    @property
    def project_root(self) -> Path:
        """프로젝트 루트 경로"""
        return Path(__file__).parent.parent

    @property
    def model_path(self) -> Path:
        """모델 디렉토리 경로"""
        if self.MODEL_DIR:
            return Path(self.MODEL_DIR)
        return self.project_root / "models"

    @property
    def wind_model_path(self) -> Path:
        """풍력 모델 경로"""
        return self._wind_dir / self.WIND_MODEL_NAME

    @property
    def wind_patterns_path(self) -> Path:
        """풍력 모델 학습에 쓴 2020-2023년 월·시간별 기상 평년값"""
        return self._wind_dir / "weather_patterns_2020_2023.csv"

    @property
    def solar_model_dir(self) -> Path:
        """태양광 V2 모델 디렉토리 (metadata.json, metrics_test.json, 시드별 .lgb)"""
        root = "solar_v2_system" if self.SERVING_LABEL == "system" else "solar_v2"
        return self.model_path / root / self.SOLAR_V2_VARIANT

    @property
    def solar_level_model_dir(self) -> Optional[Path]:
        """발전 수준(lvl_p) 입력을 쓰는 태양광 변형. 계통 라벨에서만 서빙한다 (없으면 None)"""
        if self.SERVING_LABEL != "system":
            return None
        return self.model_path / "solar_v2_system" / "island_level"

    @property
    def _wind_dir(self) -> Path:
        return self.model_path / ("wind_system" if self.SERVING_LABEL == "system" else "wind_phase5_fair")

    @property
    def data_path(self) -> Path:
        """데이터 디렉토리 경로"""
        return self.project_root / "data" / "processed"

    @property
    def cors_origins_list(self) -> list:
        """CORS origins 리스트"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤 반환"""
    return Settings()


settings = get_settings()
