"""
J-REF API Main
===============

제주도 신재생에너지(태양광/풍력) 발전량 예측 REST API

Features:
- 실시간 발전량 예측 (Wind: interval-END 재학습 모델, 구간 (T-1h, T] / Solar: Solar V2 섬 평균 모델, 구간 (T-1h, T])
- 배치 예측 (최대 168시간 = 7일)
- 대략적 불확실성 폭 (테스트 잔차 ±1.28·RMSE, 보정된 80% 구간 아님)
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import (
    EnergyType,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from .service import ModelUnavailableError, get_prediction_service

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# Service startup time
_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global _start_time
    _start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Load models
    service = get_prediction_service()
    status = service.load_models()

    logger.info(f"Model status: {status}")

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS.split(",") if settings.CORS_ALLOW_METHODS != "*" else ["*"],
    allow_headers=settings.CORS_ALLOW_HEADERS.split(",") if settings.CORS_ALLOW_HEADERS != "*" else ["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API 루트 엔드포인트"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """헬스체크 엔드포인트"""
    service = get_prediction_service()
    model_status = service.get_model_status()

    uptime = time.time() - _start_time if _start_time else 0

    all_loaded = all(model_status.values())
    status = "healthy" if all_loaded else "degraded"

    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        models_loaded=model_status,
        uptime_seconds=uptime,
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """로드된 모델 정보 조회"""
    service = get_prediction_service()
    model_status = service.get_model_status()

    models = []

    if model_status.get("wind"):
        models.append({
            "name": "wind",
            "type": "LightGBM",
            "r2": service.wind_r2,
            "rmse": service.wind_rmse,
            "features": len(service.wind_model.feature_name()) if service.wind_model else 0,
            "status": "loaded",
            "description": (
                "Wind LightGBM (interval-END labels, post-2024-06-break training, phase5_tw0999); "
                "prediction at T = generation over (T-1h, T]; "
                "RMSE/R2 are CV validation-month (Aug/Sep/Oct 2024) means"
            ),
        })

    if model_status.get("solar"):
        models.append({
            "name": "solar",
            "type": "LightGBM (Solar V2, island mean)",
            "r2": service.solar_r2,
            "rmse": service.solar_rmse,
            "features": len(service.solar_model.feature_names) if service.solar_model else 0,
            "status": "loaded",
            "description": (
                "Solar V2 on island-mean weather (plane-of-array physics + cloud trend); "
                "prediction at T = generation over (T-1h, T], needs the previous hour's weather; "
                "bounds = ±1.28·RMSE of observed-weather test residuals, day/night separately (not a calibrated interval)"
            ),
        })

    return ModelInfoResponse(models=models)


# =============================================================================
# Prediction Endpoints
# =============================================================================

def _require_models(energy_type: EnergyType) -> None:
    """요청한 에너지원 모델이 없으면 500 (단일·배치 공통: 0 MW로 조용히 응답하지 않기 위함)"""
    model_status = get_prediction_service().get_model_status()
    needed = ["wind", "solar"] if energy_type == EnergyType.BOTH else [energy_type.value]
    missing = [name for name in needed if not model_status.get(name)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {', '.join(missing)} (requested energy_type={energy_type.value})",
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="단일 예측",
    description="기상 데이터를 기반으로 발전량 예측을 수행합니다.",
)
async def predict(request: PredictionRequest):
    """
    단일 시점 발전량 예측

    - **weather**: 기상 데이터 (기온, 습도, 풍속, 풍향, 기압, 일사량 등)
    - **energy_type**: 예측 대상 (solar, wind, both)
    - **include_uncertainty**: 대략적 ±1.28·RMSE 폭 포함 여부 (보정된 80% 구간 아님)

    풍력·태양광 모두 (T-1h, T] 발전량이므로 결과별 interval_start/interval_end를 확인하고,
    total_mw는 모델 오차가 독립이라 항상 null이다.
    """
    service = get_prediction_service()
    _require_models(request.energy_type)

    try:
        response = service.predict(
            weather=request.weather,
            energy_type=request.energy_type,
            include_uncertainty=request.include_uncertainty,
            history=request.history,
            level_history=request.level_history,
        )
        return response
    except ModelUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"],
    summary="배치 예측",
    description="여러 시점의 기상 데이터를 기반으로 배치 예측을 수행합니다. (최대 168시간)",
)
async def predict_batch(request: BatchPredictionRequest):
    """
    배치 발전량 예측 (최대 168시간 = 7일)

    - **weather_data**: 시간별 기상 데이터 리스트
    - **energy_type**: 예측 대상 (solar, wind, both)
    """
    service = get_prediction_service()
    _require_models(request.energy_type)

    try:
        response = service.predict_batch(
            weather_data=request.weather_data,
            energy_type=request.energy_type,
            history=request.history,
            level_history=request.level_history,
        )
        return response
    except ModelUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "error_message": exc.detail,
            "detail": None,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "error_message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
    )
