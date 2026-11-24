"""
API Service Layer
==================

모델 로딩 및 예측 비즈니스 로직
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features.wind_features_v2 import create_wind_features_v2

from .config import settings
from .schemas import (
    EnergyType,
    WeatherInput,
    PredictionResult,
    PredictionResponse,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)

# 태양광 모듈이 깨져도(의존성 누락 등) 풍력 API까지 죽지 않도록 import 실패를 '태양광 사용 불가'로 격리한다
try:
    from src.features.solar_features_v2 import (
        create_island_mean_features,
        frame_from_point_weather,
        generation_level,
        interval_irradiance,
        physics_features,
        to_island_mean,
        to_kst_naive,
    )
    from src.models.solar_v2 import LEVEL_COL, SolarV2Model
    SOLAR_IMPORT_ERROR: Optional[Exception] = None
except ImportError as e:
    SOLAR_IMPORT_ERROR = e
    create_island_mean_features = frame_from_point_weather = interval_irradiance = SolarV2Model = None
    logger.error(f"Solar V2 modules unavailable, solar prediction disabled: {e}")

    def to_kst_naive(times) -> pd.DatetimeIndex:
        """solar_features_v2.to_kst_naive와 같은 규칙 (풍력 시각 정규화에 필요해 import 실패 시에만 둔다)"""
        times = pd.DatetimeIndex(times)
        if times.tz is not None:
            times = times.tz_convert("Asia/Seoul").tz_localize(None)
        return times

# 풍력 피처의 최대 과거 참조 길이 (lag 24h, rolling 24h)
WIND_LOOKBACK_HOURS = 24
# API 시정은 km, 학습 데이터(KMA ASOS)는 10m 단위
VISIBILITY_KM_TO_TRAINING_UNIT = 100
# 입력 누락 시 기본값 (학습 데이터 단위)
# 일사 기본값은 풍력 피처에만 쓴다. 태양광은 낮 일사 결측을 0으로 채우면 조용히 과소예측하므로 결측 그대로 넘겨 모델이 거부하게 한다
DEFAULT_SOLAR_RADIATION = 0.0
DEFAULT_CLOUD_COVER = 5.0
DEFAULT_VISIBILITY_KM = 10.0
DEFAULT_PRECIPITATION = 0.0
# 태양광 /models 지표와 불확실성 구간 기준: 실효 설비 359MW, 2025 육지 정산 규칙 (학습 스크립트의 대표 지표)
SOLAR_METRICS_FILE = "metrics_test.json"
SOLAR_HEADLINE_CAPACITY = "eff_359"
SOLAR_HEADLINE_RULE = "mainland_2025_benchmark"
# 정규 오차를 가정한 대략적 ±1.28·RMSE 폭 (보정된 80% 구간은 아님)
UNCERTAINTY_Z = 1.28
# 풍력 구간 폭에 쓰는 RMSE (MW, interval-END 재학습 모델의 CV 검증월 8/9/10 평균 — 12.15)
WIND_TEST_RMSE_MW = 12.15
LEGACY_WIND_R2 = 0.5517
# 계통 라벨 풍력 모델은 지표(R², RMSE)를 이 파일로 갖고 다닌다 (scripts/train_serving_models_system.py)
WIND_SERVING_METADATA_FILE = "wind_serving_metadata.json"
# 태양광 낮/밤 RMSE를 계산할 테스트 예측 파일 (index = 구간 끝 T, actual_mw, pred_mw)
SOLAR_TEST_PREDICTIONS_FILE = "predictions_test.csv"
# 1시간 구간 길이: 풍력·태양광 모두 (T-1h, T] — KPX 거래시간 h = T의 시
INTERVAL = pd.Timedelta(hours=1)


class ModelUnavailableError(RuntimeError):
    """요청한 에너지원 모델이 로드되지 않았다 (0 MW로 조용히 응답하지 않기 위해 예외로 알린다)"""


class PredictionService:
    """
    발전량 예측 서비스

    Models:
    - Wind: LightGBM (R² 0.76, Phase 5 Fair)
    - Solar: Solar V2 섬 평균 변형 (물리 피처 LightGBM, models/solar_v2/island_mean)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.wind_model = None
        self.solar_model = None
        self.weather_patterns = None
        # interval-END 재학습 모델의 CV 검증월(8/9/10) 평균 R² (전체 시간 기준)
        self.wind_r2 = LEGACY_WIND_R2
        self.wind_rmse = WIND_TEST_RMSE_MW
        # 발전 수준(lvl_p) 입력을 쓰는 태양광 변형 (level_history가 충분할 때만 사용)
        self.solar_level_model = None
        # 지표 파일이 없으면 None으로 두고 예측은 계속한다 (load_models에서 채움)
        self.solar_r2: Optional[float] = None
        self.solar_rmse: Optional[float] = None
        # 불확실성 폭용 낮/밤 RMSE (테스트 예측 파일이 없으면 None → 구간 생략)
        self.solar_rmse_day: Optional[float] = None
        self.solar_rmse_night: Optional[float] = None

        self._initialized = True

    def load_models(self) -> Dict[str, bool]:
        """모델 로드"""
        status = {"wind": False, "solar": False}

        # Wind model
        try:
            wind_path = settings.wind_model_path
            if wind_path.exists():
                self.wind_model = lgb.Booster(model_file=str(wind_path))
                status["wind"] = True
                logger.info(f"Wind model loaded: {wind_path}")
                meta_path = wind_path.parent / WIND_SERVING_METADATA_FILE
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self.wind_r2, self.wind_rmse = float(meta["r2"]), float(meta["rmse_mw"])
                else:
                    self.wind_r2, self.wind_rmse = LEGACY_WIND_R2, WIND_TEST_RMSE_MW
            else:
                logger.warning(f"Wind model not found: {wind_path}")
        except Exception as e:
            logger.error(f"Failed to load wind model: {e}")

        # Solar model (V2 섬 평균)
        solar_dir = settings.solar_model_dir
        self.solar_model = None
        if SOLAR_IMPORT_ERROR is not None:
            logger.error(f"Solar V2 model not loaded: modules failed to import ({SOLAR_IMPORT_ERROR})")
        else:
            try:
                model = SolarV2Model.load(solar_dir)
                self._validate_solar_model(model)
                self.solar_model = model
                status["solar"] = True
                logger.info(f"Solar V2 model loaded: {solar_dir}")
            except Exception as e:
                logger.error(f"Failed to load solar V2 model ({solar_dir}): {e}")
        self.solar_r2, self.solar_rmse = self._load_solar_metrics(solar_dir)
        self.solar_level_model = self._load_solar_level_model(settings.solar_level_model_dir)
        # 낮/밤 판정(interval_irradiance)도 태양광 모듈이 있어야 하므로 import 실패 시에는 구간 없이 둔다
        if SOLAR_IMPORT_ERROR is None:
            self.solar_rmse_day, self.solar_rmse_night = self._load_solar_daylight_rmse(solar_dir)

        # 풍력 모델은 학습 때와 같은 평년값 없이는 피처가 틀어지므로, 없으면 풍력을 비활성화한다
        patterns_path = settings.wind_patterns_path
        try:
            self.weather_patterns = pd.read_csv(patterns_path)
            logger.info(f"Weather patterns loaded: {patterns_path}")
        except Exception as e:
            logger.error(f"Weather patterns not available ({patterns_path}): {e}; wind model disabled")
            self.weather_patterns = None
            self.wind_model = None
            status["wind"] = False

        return status

    @staticmethod
    def _load_solar_metrics(model_dir: Path) -> Tuple[Optional[float], Optional[float]]:
        """테스트 기간 태양광 R²·RMSE(MW). 없거나 깨졌으면 (None, None) — 지표 때문에 예측까지 막지 않는다"""
        path = model_dir / SOLAR_METRICS_FILE
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            # 계통 라벨 모델은 "serving" 키에 walk-forward 지표를 둔다. 이전 모델은 테스트 구간 지표
            head = (metrics["serving"] if "serving" in metrics
                    else metrics["by_capacity"][SOLAR_HEADLINE_CAPACITY][SOLAR_HEADLINE_RULE])
            r2, rmse = float(head["r2"]), float(head["rmse_mw"])
            if not (np.isfinite(r2) and np.isfinite(rmse) and rmse >= 0):
                raise ValueError(f"유효하지 않은 지표 r2={r2}, rmse={rmse}")
        except Exception as e:
            logger.warning(f"Solar metrics not available ({path}): {e}")
            return None, None
        return r2, rmse

    @staticmethod
    def _load_solar_level_model(model_dir: Optional[Path]):
        """발전 수준 변형을 불러온다. 없거나 깨졌으면 None — 기상만 쓰는 모델로 계속 서빙한다"""
        if model_dir is None or SOLAR_IMPORT_ERROR is not None or not Path(model_dir).exists():
            return None
        try:
            model = SolarV2Model.load(model_dir)
            if LEVEL_COL not in model.feature_names:
                raise ValueError(f"{LEVEL_COL} 피처가 없는 모델입니다")
            logger.info(f"Solar level model loaded: {model_dir}")
            return model
        except Exception as e:
            logger.error(f"Failed to load solar level model ({model_dir}): {e}")
            return None

    def _solar_level(self, solar_frame: pd.DataFrame, n_history: int, level_history) -> Optional[pd.Series]:
        """
        level_history로 대상 시점의 발전 수준(lvl_p)을 계산한다. 쓸 수 없으면 None

        학습과 같은 함수(generation_level)를 쓴다: 최근 14일 중 D-2까지의 Σ발전량 ÷ Σ물리 출력대용값.
        D-1 이후 기록은 지연 규칙 때문에 자동으로 빠진다.

        Raises:
            ValueError: level_history에 같은 시각이 두 번 있을 때
        """
        if self.solar_level_model is None or not level_history:
            return None
        hist = pd.DataFrame([{
            "datetime": r.datetime, "temperature": r.temperature, "humidity": r.humidity,
            "solar_radiation": r.solar_radiation, "cloud_cover": r.cloud_cover, "solar_mw": r.solar_mw,
        } for r in level_history])
        hist.index = to_kst_naive(hist.pop("datetime"))
        hist = hist.sort_index().astype(float)
        if not hist.index.is_unique:
            raise ValueError("level_history에 같은 시각이 두 번 있습니다")
        p_hist = physics_features(to_island_mean(frame_from_point_weather(hist.drop(columns="solar_mw"))))["p_mean"]
        targets = solar_frame.index[n_history:]
        p_target = physics_features(to_island_mean(solar_frame))["p_mean"].iloc[n_history:]
        index = hist.index.union(targets)
        p_all = pd.concat([p_hist, p_target[~p_target.index.isin(p_hist.index)]]).reindex(index)
        level = generation_level(pd.DataFrame(index=index), hist["solar_mw"].reindex(index), p_all).reindex(targets)
        day = solar_frame["daylight_frac"].iloc[n_history:].to_numpy() > 0
        usable = np.isfinite(level.to_numpy()) & (level.to_numpy() > 0)
        if (day & ~usable).any():
            return None   # 이력이 모자라면 조용히 낡은 수준을 쓰지 않고 기상만 쓰는 모델로 돌아간다
        return level

    @staticmethod
    def _validate_solar_model(model) -> None:
        """
        API가 만들 수 있는 피처만 쓰는 모델인지 확인 (다른 변형 파일이 섞여 들어와도 엉뚱한 예측을 막는다)

        Raises:
            ValueError: 발전 수준(lvl_p)·정규화 모델이 필요하거나, create_island_mean_features가 만들지 않는 피처가 있을 때
        """
        if "lvl_p" in model.feature_names or model.norm_models:
            raise ValueError("발전 수준(lvl_p)·정규화 모델이 필요한 변형은 API에서 쓸 수 없습니다 (island_mean 필요)")
        # 피처 열 이름만 얻으면 되므로 낮 시간 2시점짜리 임의 기상으로 피처를 만든다
        probe = pd.DataFrame(
            {"temperature": 10.0, "humidity": 60.0, "solar_radiation": 1.0, "cloud_cover": 5.0},
            index=pd.date_range("2024-06-01 11:00", periods=2, freq="h"),
        )
        produced = set(create_island_mean_features(frame_from_point_weather(probe)).columns)
        unknown = [c for c in model.feature_names if c not in produced]
        if not model.feature_names or unknown:
            raise ValueError(f"create_island_mean_features가 만들지 않는 피처가 필요합니다: {unknown or '(피처 목록 없음)'}")

    @staticmethod
    def _load_solar_daylight_rmse(model_dir: Path) -> Tuple[Optional[float], Optional[float]]:
        """
        테스트 예측 잔차의 낮/밤 RMSE (MW). 없거나 깨졌으면 (None, None) — 구간만 생략하고 예측은 계속한다

        전체 시간 RMSE 하나로 폭을 잡으면 밤(오차 ≈ 0)에는 과대, 낮에는 과소가 되므로
        구간 (T-1h, T]에 해가 떠 있었는지(daylight_frac > 0)로 나눈다.
        """
        path = model_dir / SOLAR_TEST_PREDICTIONS_FILE
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            resid = (df["actual_mw"] - df["pred_mw"]).to_numpy(dtype=float)
            day = interval_irradiance(df.index)["daylight_frac"].to_numpy() > 0
            if not np.isfinite(resid).all() or day.all() or not day.any():
                raise ValueError("잔차에 결측이 있거나 낮/밤 표본이 한쪽뿐입니다")
            rmse_day = float(np.sqrt(np.mean(resid[day] ** 2)))
            rmse_night = float(np.sqrt(np.mean(resid[~day] ** 2)))
        except Exception as e:
            logger.warning(f"Solar daylight RMSE not available ({path}): {e}; uncertainty bounds disabled")
            return None, None
        return rmse_day, rmse_night

    def get_model_status(self) -> Dict[str, bool]:
        """모델 로드 상태 반환"""
        return {
            "wind": self.wind_model is not None,
            "solar": self.solar_model is not None,
        }

    def predict(
        self,
        weather: WeatherInput,
        energy_type: EnergyType = EnergyType.BOTH,
        include_uncertainty: bool = True,
        history: Optional[List[WeatherInput]] = None,
        level_history=None,
    ) -> PredictionResponse:
        """
        단일 예측 수행

        Args:
            weather: 기상 데이터
            energy_type: 예측 타입 (solar, wind, both)
            include_uncertainty: 불확실성 구간 포함 여부
            history: 예측 시점 직전의 연속된 시간별 기상
                (풍력 지연·이동평균 피처용, 태양광은 직전 1시간 필수)

        Returns:
            PredictionResponse (T = weather.datetime, 풍력·태양광 모두 구간 (T-1h, T] 발전량,
            total_mw는 모델 오차가 독립이라 항상 None)

        Raises:
            ValueError: history와 weather가 1시간 간격으로 연속되지 않을 때,
                태양광 예측인데 history가 없거나 낮 시간 일사량·운량이 없을 때
            ModelUnavailableError: 요청한 에너지원 모델이 로드되지 않았을 때 (0 MW로 대신 응답하지 않음)
        """
        start_time = time.time()

        history = history or []
        frame = self._weather_to_dataframe(history + [weather])
        predictions = self._predict_targets(frame, len(history), energy_type, include_uncertainty,
                                            level_history)[0]

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            success=True,
            timestamp=weather.datetime,
            predictions=predictions,
            # 풍력·태양광 모두 (T-1h, T]라 통계 대상 기간이 같다 (합산은 여전히 내지 않는다)
            total_mw=None,
            history_hours=len(history),
            processing_time_ms=processing_time,
        )

    def predict_batch(
        self,
        weather_data: List[WeatherInput],
        energy_type: EnergyType = EnergyType.BOTH,
        history: Optional[List[WeatherInput]] = None,
        level_history=None,
    ) -> BatchPredictionResponse:
        """
        배치 예측 수행

        weather_data는 연속된 시간으로 보고, 앞 시점들이 뒤 시점의 과거 기상으로 쓰인다.

        Args:
            weather_data: 기상 데이터 리스트
            energy_type: 예측 타입
            history: weather_data 첫 시점 직전의 연속된 시간별 기상 (태양광은 직전 1시간 필수)

        Returns:
            BatchPredictionResponse

        Raises:
            ValueError: 입력 시점들이 1시간 간격으로 연속되지 않을 때,
                태양광 예측인데 history가 없거나 낮 시간 일사량·운량이 없을 때
            ModelUnavailableError: 요청한 에너지원 모델이 로드되지 않았을 때 (0 MW로 대신 응답하지 않음)
        """
        start_time = time.time()

        history = history or []
        frame = self._weather_to_dataframe(history + list(weather_data))
        per_hour = self._predict_targets(frame, len(history), energy_type, include_uncertainty=False,
                                         level_history=level_history)

        results = []
        all_predictions = {"wind": [], "solar": []}
        for weather, predictions in zip(weather_data, per_hour):
            results.append({
                "datetime": weather.datetime.isoformat(),
                "predictions": {p.energy_type.value: p.prediction_mw for p in predictions},
                "intervals": {
                    p.energy_type.value: {
                        "start": p.interval_start.isoformat(),
                        "end": p.interval_end.isoformat(),
                    }
                    for p in predictions
                },
            })
            for p in predictions:
                all_predictions[p.energy_type.value].append(p.prediction_mw)

        # Statistics
        statistics = {}
        for key, values in all_predictions.items():
            if values:
                statistics[key] = {
                    "mean_mw": float(np.mean(values)),
                    "min_mw": float(np.min(values)),
                    "max_mw": float(np.max(values)),
                    "total_mwh": float(np.sum(values)),
                }

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            success=True,
            predictions=results,
            total_hours=len(weather_data),
            history_hours=len(history),
            statistics=statistics,
            processing_time_ms=processing_time,
        )

    def _predict_targets(
        self,
        frame: pd.DataFrame,
        n_history: int,
        energy_type: EnergyType,
        include_uncertainty: bool,
        level_history=None,
    ) -> List[List[PredictionResult]]:
        """frame의 history 이후 각 시점에 대해 [풍력, 태양광] 예측 결과 리스트를 반환"""
        n_targets = len(frame) - n_history
        per_hour: List[List[PredictionResult]] = [[] for _ in range(n_targets)]

        if energy_type in [EnergyType.WIND, EnergyType.BOTH]:
            for i, result in enumerate(self._predict_wind(frame, n_history, include_uncertainty)):
                per_hour[i].append(result)

        if energy_type in [EnergyType.SOLAR, EnergyType.BOTH]:
            for i, result in enumerate(self._predict_solar(frame, n_history, include_uncertainty, level_history)):
                per_hour[i].append(result)

        return per_hour

    def _weather_to_dataframe(self, weathers: List[WeatherInput]) -> pd.DataFrame:
        """
        WeatherInput 리스트를 학습 데이터 단위의 시간별 DataFrame으로 변환

        일사량·운량 결측은 NaN으로 남긴다 (풍력 피처에서만 기본값으로 채움).
        시각은 여기서 한 번만 KST naive로 맞춘다: 시간대가 붙은 입력은 KST로 바꾼 뒤 시간대를 떼므로
        풍력·태양광이 같은 시간을 보고, 서로 다른 오프셋이 섞여도 pandas 내부 오류가 나지 않는다.

        Raises:
            ValueError: 시점들이 1시간 간격으로 연속되지 않을 때
        """
        def _or(value: Optional[float], default: float) -> float:
            # 0(맑음, 무강수 등)도 유효한 관측값이므로 None일 때만 기본값을 쓴다
            return default if value is None else value

        df = pd.DataFrame({
            # 시간대 오프셋이 섞인 리스트를 한 번에 변환하면 실패하므로 시점별로 정규화한다
            "datetime": [to_kst_naive([w.datetime])[0] for w in weathers],
            "temperature": [w.temperature for w in weathers],
            "humidity": [w.humidity for w in weathers],
            "wind_speed": [w.wind_speed for w in weathers],
            "wind_direction": [w.wind_direction for w in weathers],
            "pressure": [w.pressure for w in weathers],
            "solar_radiation": [np.nan if w.solar_radiation is None else w.solar_radiation for w in weathers],
            "cloud_cover": [np.nan if w.cloud_cover is None else w.cloud_cover for w in weathers],
            "visibility": [
                _or(w.visibility, DEFAULT_VISIBILITY_KM) * VISIBILITY_KM_TO_TRAINING_UNIT
                for w in weathers
            ],
            "precipitation": [_or(w.precipitation, DEFAULT_PRECIPITATION) for w in weathers],
        })

        df["datetime"] = pd.to_datetime(df["datetime"])
        steps = df["datetime"].diff().dropna()
        if not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError("history와 weather는 시간순으로 1시간 간격 연속이어야 합니다")

        return df.set_index("datetime")

    def _predict_wind(
        self,
        frame: pd.DataFrame,
        n_history: int,
        include_uncertainty: bool = True,
    ) -> List[PredictionResult]:
        """frame의 history 이후 각 시점 풍력 예측 (시점 T 예측 = 구간 (T-1h, T] 발전량)"""
        if self.wind_model is None:
            raise ModelUnavailableError("Wind model not loaded")

        # Feature engineering
        features_df = self._create_wind_features(frame, n_history)

        # Get model feature names
        model_features = self.wind_model.feature_name()

        # Align features
        X = self._align_features(features_df, model_features)

        # Predict
        predictions = np.clip(self.wind_model.predict(X), 0, None)

        # 대략적 ±1.28·RMSE 폭 (interval-END 재학습 모델의 CV 검증월 평균 RMSE ~12.15 MW)
        rmse = self.wind_rmse
        results = []
        # 학습 라벨이 interval-END(T = 거래시간 h)이므로 T 예측은 (T-1h, T] 발전량이다
        for t, prediction in zip(frame.index[n_history:], predictions):
            prediction = float(prediction)
            results.append(PredictionResult(
                energy_type=EnergyType.WIND,
                prediction_mw=prediction,
                interval_start=t - INTERVAL,
                interval_end=t,
                lower_bound_mw=max(0, prediction - UNCERTAINTY_Z * rmse) if include_uncertainty else None,
                upper_bound_mw=prediction + UNCERTAINTY_Z * rmse if include_uncertainty else None,
                model_r2=self.wind_r2,
            ))
        return results

    def _predict_solar(
        self,
        frame: pd.DataFrame,
        n_history: int,
        include_uncertainty: bool = True,
        level_history=None,
    ) -> List[PredictionResult]:
        """
        frame의 history 이후 각 시점 태양광 예측 (대상 시점 전체를 한 번에)

        시점 T의 예측은 KPX 거래시간 구간 (T-1h, T] 발전량이다 (거래시간 h = T의 시, T=00:00은 전날 24시).

        Raises:
            ValueError: 첫 대상 시점의 직전 1시간 기상(history)이 없을 때, 낮 시간 일사량·운량이 없을 때
            ModelUnavailableError: 태양광 모델이 로드되지 않았을 때
        """
        if self.solar_model is None:
            raise ModelUnavailableError("Solar model not loaded")

        # 운량 추세(cc_m1)와 구간 기온이 직전 1시간 관측을 쓴다. 없으면 학습 때 없던 결측이라 거부한다
        if n_history == 0:
            raise ValueError(
                "태양광 예측에는 첫 예측 시점 직전 1시간의 기상이 필요합니다 (history에 1시간 이상 넣어 주세요)"
            )

        # @MX:NOTE: [AUTO] 학습(scripts/train_solar_v2.py --variant island_mean)과 같은 함수로 피처를 만든다.
        # 대상 시점만 넘겨야 직전 관측이 없는 history 첫 행이 결측 검사에 걸리지 않는다.
        solar_frame = frame_from_point_weather(frame)
        # 낮 운량 결측은 모델이 아니라 여기서 막는다 (학습 자료에 운량 결측이 있으면 모델은 받아들이기 때문)
        day_target = solar_frame["daylight_frac"].iloc[n_history:].to_numpy() > 0
        no_cloud = day_target & solar_frame["cloud_cover_jeju"].iloc[n_history:].isna().to_numpy()
        if no_cloud.any():
            raise ValueError(f"낮 시간 운량 결측 {int(no_cloud.sum())}개: 기본값으로 채우면 조용히 틀린 예측이 되므로 거부합니다")
        X = create_island_mean_features(solar_frame).iloc[n_history:]
        level = self._solar_level(solar_frame, n_history, level_history)
        level_used = level is not None
        if level_used:
            X = pd.concat([X, level.rename(LEVEL_COL)], axis=1)
            predictions = self.solar_level_model.predict(X, solar_frame["daylight_frac"])
        else:
            predictions = self.solar_model.predict(X, solar_frame["daylight_frac"])

        targets = solar_frame.index[n_history:]
        with_bounds = (
            include_uncertainty and self.solar_rmse_day is not None and self.solar_rmse_night is not None
        )
        daylight = solar_frame["daylight_frac"].iloc[n_history:].to_numpy() > 0
        results = []
        for t, prediction, is_day in zip(targets, predictions, daylight):
            prediction = float(prediction)
            rmse = (self.solar_rmse_day if is_day else self.solar_rmse_night) if with_bounds else None
            results.append(PredictionResult(
                energy_type=EnergyType.SOLAR,
                prediction_mw=prediction,
                interval_start=t - INTERVAL,
                interval_end=t,
                lower_bound_mw=max(0.0, prediction - UNCERTAINTY_Z * rmse) if with_bounds else None,
                upper_bound_mw=prediction + UNCERTAINTY_Z * rmse if with_bounds else None,
                model_r2=self.solar_r2,
                level_used=level_used,
            ))
        return results

    def _create_wind_features(self, frame: pd.DataFrame, n_history: int) -> pd.DataFrame:
        """
        풍력 모델용 피처 생성 (학습 파이프라인과 동일한 함수 사용)

        Args:
            frame: history + 예측 대상 시점이 이어진 시간별 기상 (datetime index)
            n_history: frame 앞쪽의 history 행 수

        Returns:
            예측 대상 시점 행만 담은 피처 DataFrame
        """
        # @MX:NOTE: [AUTO] 학습(scripts/phase5_bias_correction.py)과 같은 순서로 피처를 만든다.
        # 손으로 재구현한 피처는 학습값과 어긋나 API 경로 R²가 0.71로 떨어졌었다.
        n_targets = len(frame) - n_history
        # 일사·운량 결측은 태양광 검증을 위해 NaN으로 넘어오므로, 풍력 피처에서는 기존처럼 기본값으로 채운다
        frame = frame.fillna({"solar_radiation": DEFAULT_SOLAR_RADIATION, "cloud_cover": DEFAULT_CLOUD_COVER})

        # history가 부족하면 첫 관측값이 그대로 이어졌다고 가정해 앞을 채운다 (지속성 가정)
        n_pad = max(0, WIND_LOOKBACK_HOURS - n_history)
        if n_pad:
            first = frame.iloc[[0]]
            pad_index = frame.index[0] - pd.to_timedelta(np.arange(n_pad, 0, -1), unit="h")
            pad = pd.DataFrame(
                np.repeat(first.values, n_pad, axis=0), columns=frame.columns, index=pad_index
            )
            frame = pd.concat([pad, frame])

        df = frame.rename_axis("datetime").reset_index()
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month
        df["dayofyear"] = df["datetime"].dt.dayofyear

        # 2020-2023년 월·시간별 평년값 (학습 때와 같은 기간)
        df = pd.merge(df, self.weather_patterns, on=["month", "hour"], how="left")
        df["wind_speed_anomaly"] = df["wind_speed"] - df["wind_speed_mean"]
        df["temperature_anomaly"] = df["temperature"] - df["temperature_mean"]
        df["solar_radiation_anomaly"] = df["solar_radiation"] - df["solar_radiation_mean"]

        df = create_wind_features_v2(df.set_index("datetime"))
        return df.iloc[-n_targets:]

    def _align_features(
        self,
        features_df: pd.DataFrame,
        model_features: List[str],
    ) -> np.ndarray:
        """모델이 요구하는 피처 순서로 정렬"""
        columns = {}
        for feat in model_features:
            if feat in features_df.columns:
                columns[feat] = features_df[feat]
            else:
                # Missing feature: use default value
                columns[feat] = 0.0
                logger.debug(f"Missing feature '{feat}', using default 0.0")

        aligned_df = pd.DataFrame(columns, index=features_df.index)
        # Handle inf/nan
        aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan)
        aligned_df = aligned_df.fillna(0.0)

        return aligned_df.values


# Singleton instance
_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """예측 서비스 인스턴스 반환"""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
