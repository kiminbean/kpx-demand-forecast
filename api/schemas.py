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

    datetime: dt = Field(
        ...,
        description=(
            "시각 T (KST, 시간대 없으면 KST로 간주, 시간대가 있으면 서버에서 KST로 변환). 기상은 T의 관측값. "
            "태양광 예측은 KPX 거래시간 구간 (T-1h, T] 발전량 (거래시간 h = T의 시, T=00:00은 전날 24시), "
            "풍력 예측은 구간 (T, T+1h] 발전량 (학습 라벨이 구간 시작 시각 기준) — 같은 T라도 서로 다른 1시간이다"
        ),
    )
    temperature: float = Field(..., description="기온 (°C)")
    humidity: float = Field(..., ge=0, le=100, description="습도 (%)")
    wind_speed: float = Field(..., ge=0, description="풍속 (m/s)")
    wind_direction: float = Field(..., ge=0, lt=360, description="풍향 (°)")
    pressure: float = Field(..., description="기압 (hPa)")
    solar_radiation: Optional[float] = Field(
        None,
        ge=0,
        # 대기권 밖 수평 일사의 1시간 최대가 약 4.9 MJ/m²이므로 이를 넘으면 단위 오류(W/m² 등)로 본다
        le=5.0,
        description=(
            "일사량 (MJ/m² per hour, 구간 (T-1h, T] 누적 = ASOS 시간 일사, 제주·고산 관측소 평균). "
            "W/m²(1시간 평균 일사 강도)는 ×0.0036으로 변환해서 보낼 것 (최대 5.0). "
            "태양광 예측 시 낮 시간에는 필수 (없으면 400)"
        ),
    )
    cloud_cover: Optional[float] = Field(
        None, ge=0, le=10,
        description="운량 (0-10). 태양광 예측 시 낮 시간에는 필수 (없으면 400), 풍력은 없으면 기본값 사용",
    )
    visibility: Optional[float] = Field(None, ge=0, description="시정 (km, 서버에서 학습 단위인 10m로 변환)")
    precipitation: Optional[float] = Field(None, ge=0, description="강수량 (mm)")


# ============================================================
# Prediction Request/Response
# ============================================================

_EXAMPLE_HISTORY_HOUR = {
    "datetime": "2024-12-17T13:00:00",
    "temperature": 8.2,
    "humidity": 66.0,
    "wind_speed": 5.0,
    "wind_direction": 270.0,
    "pressure": 1013.6,
    "solar_radiation": 2.3,
    "cloud_cover": 3,
    "visibility": 15.0,
    "precipitation": 0.0,
}
_EXAMPLE_WEATHER_1400 = WeatherInput.model_config["json_schema_extra"]["example"]
_EXAMPLE_WEATHER_1500 = {
    **_EXAMPLE_WEATHER_1400,
    "datetime": "2024-12-17T15:00:00",
    "temperature": 8.4,
    "solar_radiation": 1.8,
    "cloud_cover": 4,
}


class PredictionRequest(BaseModel):
    """단일 예측 요청"""
    # 예시는 태양광에 필요한 직전 1시간 history를 포함해야 /docs에서 그대로 실행해도 400이 나지 않는다
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "weather": _EXAMPLE_WEATHER_1400,
                "history": [_EXAMPLE_HISTORY_HOUR],
                "energy_type": "both",
                "include_uncertainty": True,
            }
        },
    )

    weather: WeatherInput = Field(..., description="기상 데이터")
    history: Optional[List[WeatherInput]] = Field(
        None,
        max_length=168,
        description=(
            "예측 시점 직전의 연속된 시간별 기상 "
            "(풍력 지연·이동평균 피처용 24시간 권장, 태양광은 직전 1시간 필수)"
        )
    )
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
    prediction_mw: float = Field(..., description="예측 발전량 (MW, interval_start~interval_end 1시간 평균)")
    interval_start: Optional[dt] = Field(
        None, description="예측 구간 시작 (KST naive). 풍력 = T, 태양광 = T-1h"
    )
    interval_end: Optional[dt] = Field(
        None, description="예측 구간 끝 (KST naive). 풍력 = T+1h, 태양광 = T"
    )
    lower_bound_mw: Optional[float] = Field(
        None,
        description=(
            "대략적 하한 = 예측 - 1.28·RMSE (0 이상). RMSE는 관측 기상을 넣은 테스트 기간 잔차 "
            "(풍력: 전체 시간, 태양광: 낮/밤 구간 따로). 보정된 80% 구간이 아니며 "
            "수치예보(NWP)로 하루 전 예측할 때는 실제 오차가 더 크다"
        ),
    )
    upper_bound_mw: Optional[float] = Field(
        None, description="대략적 상한 = 예측 + 1.28·RMSE (하한 설명 참고, 보정된 80% 구간 아님)"
    )
    model_r2: Optional[float] = Field(None, description="모델 R² 성능 (테스트 기간, 지표 파일이 없으면 null)")


class PredictionResponse(BaseModel):
    """예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    timestamp: dt = Field(
        ...,
        description=(
            "요청한 시각 T (입력 그대로). 태양광은 구간 (T-1h, T], 풍력은 (T, T+1h] 발전량이므로 "
            "실제 구간은 각 결과의 interval_start/interval_end를 볼 것"
        ),
    )
    predictions: List[PredictionResult] = Field(..., description="예측 결과")
    total_mw: Optional[float] = Field(
        None,
        description=(
            "총 발전량 (MW). 호환용 필드로 항상 null: energy_type=both여도 풍력·태양광이 서로 다른 1시간이라 "
            "합하면 두 시간을 섞게 된다. 합계가 필요하면 같은 구간의 결과끼리 더할 것"
        ),
    )
    history_hours: int = Field(0, description="피처 계산에 사용한 과거 기상 시간 수 (풍력 지연·이동평균, 태양광 직전 1시간)")
    processing_time_ms: float = Field(..., description="처리 시간 (ms)")


# ============================================================
# Batch Prediction
# ============================================================

class BatchPredictionRequest(BaseModel):
    """배치 예측 요청"""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "weather_data": [_EXAMPLE_WEATHER_1400, _EXAMPLE_WEATHER_1500],
                "history": [_EXAMPLE_HISTORY_HOUR],
                "energy_type": "both",
            }
        },
    )

    weather_data: List[WeatherInput] = Field(
        ...,
        min_length=1,
        max_length=168,
        description="연속된 시간별 기상 데이터 리스트 (최대 168시간 = 7일)"
    )
    history: Optional[List[WeatherInput]] = Field(
        None,
        max_length=168,
        description="weather_data 첫 시점 직전의 연속된 시간별 기상 (풍력 24시간 권장, 태양광은 직전 1시간 필수)"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.BOTH,
        description="예측 대상"
    )


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    predictions: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "시간별 예측 결과: datetime(요청 T), predictions{에너지: MW}, "
            "intervals{에너지: {start, end}} (풍력 (T, T+1h], 태양광 (T-1h, T])"
        ),
    )
    total_hours: int = Field(..., description="총 예측 시간")
    history_hours: int = Field(0, description="피처 계산에 사용한 과거 기상 시간 수 (풍력 지연·이동평균, 태양광 직전 1시간)")
    statistics: Dict[str, Any] = Field(
        ...,
        description=(
            "에너지원별 예측 통계 (mean/min/max MW, total_mwh). 풍력은 (T, T+1h], 태양광은 (T-1h, T] 구간 기준이라 "
            "같은 요청이라도 두 통계의 대상 기간이 1시간 어긋난다 (합산 통계는 내지 않음)"
        ),
    )
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
                "version": "1.1.0",
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
                        "type": "LightGBM (Solar V2, island mean)",
                        "r2": 0.9685,
                        "rmse": 10.63,
                        "features": 24,
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
