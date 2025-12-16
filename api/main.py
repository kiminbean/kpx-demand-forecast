"""
J-REF API Main
===============

제주도 신재생에너지(태양광/풍력) 발전량 예측 REST API

Features:
- 실시간 발전량 예측 (Wind: R² 0.76, Solar: R² 0.91)
- 배치 예측 (최대 168시간 = 7일)
- 불확실성 정량화 (80% 신뢰구간)
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
from .service import get_prediction_service

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
            "rmse": 14.96,
            "features": len(service.wind_model.feature_name()) if service.wind_model else 0,
            "status": "loaded",
            "description": "Phase 5 Fair Wind Model with V2 Features",
        })

    if model_status.get("solar"):
        models.append({
            "name": "solar",
            "type": "LightGBM",
            "r2": service.solar_r2,
            "rmse": 17.63,
            "features": len(service.solar_model.feature_name()) if service.solar_model else 0,
            "status": "loaded",
            "description": "Baseline Solar Model with Physics-Informed Features",
        })

    return ModelInfoResponse(models=models)


# =============================================================================
# Prediction Endpoints
# =============================================================================

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
    - **include_uncertainty**: 80% 신뢰구간 포함 여부
    """
    service = get_prediction_service()
    model_status = service.get_model_status()

    # Check model availability
    if request.energy_type == EnergyType.WIND and not model_status.get("wind"):
        raise HTTPException(status_code=500, detail="Wind model not loaded")
    if request.energy_type == EnergyType.SOLAR and not model_status.get("solar"):
        raise HTTPException(status_code=500, detail="Solar model not loaded")
    if request.energy_type == EnergyType.BOTH:
        if not model_status.get("wind") or not model_status.get("solar"):
            raise HTTPException(status_code=500, detail="One or more models not loaded")

    try:
        response = service.predict(
            weather=request.weather,
            energy_type=request.energy_type,
            include_uncertainty=request.include_uncertainty,
        )
        return response
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

    try:
        response = service.predict_batch(
            weather_data=request.weather_data,
            energy_type=request.energy_type,
        )
        return response
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
