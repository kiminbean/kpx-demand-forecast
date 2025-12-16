"""
API Schemas (Pydantic Models)
=============================

요청/응답 데이터 검증을 위한 Pydantic 모델 정의
"""

from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# Enums
# ============================================================

class EnergyType(str, Enum):
    """에너지 타입"""
    SOLAR = "solar"
    WIND = "wind"
    BOTH = "both"


# ============================================================
# Weather Input
# ============================================================

class WeatherInput(BaseModel):
    """기상 데이터 입력"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "datetime": "2024-12-17T14:00:00",
                "temperature": 8.5,
                "humidity": 65.0,
                "wind_speed": 5.2,
                "wind_direction": 270.0,
                "pressure": 1013.5,
                "solar_radiation": 2.5,
                "cloud_cover": 3,
                "visibility": 15.0,
                "precipitation": 0.0
            }
        }
    )

    datetime: dt = Field(..., description="관측 시간")
    temperature: float = Field(..., description="기온 (°C)")
    humidity: float = Field(..., ge=0, le=100, description="습도 (%)")
    wind_speed: float = Field(..., ge=0, description="풍속 (m/s)")
    wind_direction: float = Field(..., ge=0, lt=360, description="풍향 (°)")
    pressure: float = Field(..., description="기압 (hPa)")
    solar_radiation: Optional[float] = Field(None, ge=0, description="일사량 (MJ/m²)")
    cloud_cover: Optional[float] = Field(None, ge=0, le=10, description="운량 (0-10)")
    visibility: Optional[float] = Field(None, ge=0, description="시정 (km)")
    precipitation: Optional[float] = Field(None, ge=0, description="강수량 (mm)")


# ============================================================
# Prediction Request/Response
# ============================================================

class PredictionRequest(BaseModel):
    """단일 예측 요청"""
    model_config = ConfigDict(protected_namespaces=())

    weather: WeatherInput = Field(..., description="기상 데이터")
    energy_type: EnergyType = Field(
        default=EnergyType.BOTH,
        description="예측 대상 (solar, wind, both)"
    )
    include_uncertainty: bool = Field(
        default=True,
        description="불확실성 구간 포함 여부"
    )


class PredictionResult(BaseModel):
    """개별 예측 결과"""
    model_config = ConfigDict(protected_namespaces=())

    energy_type: EnergyType
    prediction_mw: float = Field(..., description="예측 발전량 (MW)")
    lower_bound_mw: Optional[float] = Field(None, description="80% 신뢰구간 하한 (MW)")
    upper_bound_mw: Optional[float] = Field(None, description="80% 신뢰구간 상한 (MW)")
    model_r2: float = Field(..., description="모델 R² 성능")


class PredictionResponse(BaseModel):
    """예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    timestamp: dt = Field(..., description="예측 대상 시간")
    predictions: List[PredictionResult] = Field(..., description="예측 결과")
    total_mw: Optional[float] = Field(None, description="총 발전량 (MW)")
    processing_time_ms: float = Field(..., description="처리 시간 (ms)")


# ============================================================
# Batch Prediction
# ============================================================

class BatchPredictionRequest(BaseModel):
    """배치 예측 요청"""
    model_config = ConfigDict(protected_namespaces=())

    weather_data: List[WeatherInput] = Field(
        ...,
        min_length=1,
        max_length=168,
        description="기상 데이터 리스트 (최대 168시간 = 7일)"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.BOTH,
        description="예측 대상"
    )


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    predictions: List[Dict[str, Any]] = Field(..., description="시간별 예측 결과")
    total_hours: int = Field(..., description="총 예측 시간")
    statistics: Dict[str, Any] = Field(..., description="예측 통계")
    processing_time_ms: float = Field(..., description="처리 시간 (ms)")


# ============================================================
# Realtime Prediction (기상청 API 연동)
# ============================================================

class RealtimePredictionRequest(BaseModel):
    """실시간 예측 요청 (기상청 API 자동 연동)"""
    model_config = ConfigDict(protected_namespaces=())

    target_datetime: Optional[dt] = Field(
        None,
        description="예측 대상 시간 (None이면 현재 시간)"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.BOTH,
        description="예측 대상"
    )
    station_ids: Optional[List[int]] = Field(
        None,
        description="기상 관측소 ID (None이면 제주도 4개소 평균)"
    )


class RealtimePredictionResponse(BaseModel):
    """실시간 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    timestamp: dt
    weather_source: str = Field(..., description="기상 데이터 출처")
    weather_data: Dict[str, Any] = Field(..., description="사용된 기상 데이터")
    predictions: List[PredictionResult]
    total_mw: Optional[float] = None
    processing_time_ms: float


# ============================================================
# Health & Info
# ============================================================

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {"wind": True, "solar": True},
                "uptime_seconds": 3600.5
            }
        }
    )

    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    models_loaded: Dict[str, bool] = Field(..., description="모델 로드 상태")
    uptime_seconds: float = Field(..., description="서비스 가동 시간 (초)")


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {
                        "name": "wind",
                        "type": "LightGBM",
                        "r2": 0.7576,
                        "rmse": 14.96,
                        "features": 137,
                        "status": "loaded"
                    },
                    {
                        "name": "solar",
                        "type": "LightGBM",
                        "r2": 0.9134,
                        "rmse": 17.63,
                        "features": 45,
                        "status": "loaded"
                    }
                ]
            }
        }
    )

    models: List[Dict[str, Any]] = Field(..., description="로드된 모델 목록")


# ============================================================
# Error Response
# ============================================================

class ErrorResponse(BaseModel):
    """에러 응답"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error_code": "INVALID_INPUT",
                "error_message": "기상 데이터가 유효하지 않습니다",
                "detail": "wind_speed must be >= 0"
            }
        }
    )

    success: bool = Field(default=False)
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 정보")
