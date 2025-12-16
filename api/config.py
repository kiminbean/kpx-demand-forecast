"""
API Configuration
==================

환경 변수 및 설정 관리
"""

import os
from pathlib import Path
from typing import Optional
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
    APP_VERSION: str = "1.0.0"
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
    SOLAR_MODEL_NAME: str = "solar_lgbm_baseline.lgb"

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
        return self.model_path / "wind_phase5_fair" / self.WIND_MODEL_NAME

    @property
    def solar_model_path(self) -> Path:
        """태양광 모델 경로"""
        return self.model_path / "real_baseline" / self.SOLAR_MODEL_NAME

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
