"""
API Service Layer
==================

모델 로딩 및 예측 비즈니스 로직
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import settings
from .schemas import (
    EnergyType,
    WeatherInput,
    PredictionResult,
    PredictionResponse,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    발전량 예측 서비스

    Models:
    - Wind: LightGBM (R² 0.76, Phase 5 Fair)
    - Solar: LightGBM (R² 0.91, Baseline)
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
        self.wind_r2 = 0.7576
        self.solar_r2 = 0.9134

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
            else:
                logger.warning(f"Wind model not found: {wind_path}")
        except Exception as e:
            logger.error(f"Failed to load wind model: {e}")

        # Solar model
        try:
            solar_path = settings.solar_model_path
            if solar_path.exists():
                self.solar_model = lgb.Booster(model_file=str(solar_path))
                status["solar"] = True
                logger.info(f"Solar model loaded: {solar_path}")
            else:
                logger.warning(f"Solar model not found: {solar_path}")
        except Exception as e:
            logger.error(f"Failed to load solar model: {e}")

        # Weather patterns for anomaly features
        try:
            patterns_path = settings.data_path / "weather_5year_patterns.csv"
            if patterns_path.exists():
                self.weather_patterns = pd.read_csv(patterns_path)
                logger.info(f"Weather patterns loaded: {patterns_path}")
        except Exception as e:
            logger.warning(f"Weather patterns not available: {e}")

        return status

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
    ) -> PredictionResponse:
        """
        단일 예측 수행

        Args:
            weather: 기상 데이터
            energy_type: 예측 타입 (solar, wind, both)
            include_uncertainty: 불확실성 구간 포함 여부

        Returns:
            PredictionResponse
        """
        start_time = time.time()

        predictions = []
        total_mw = 0.0

        # Convert weather input to DataFrame
        weather_df = self._weather_to_dataframe(weather)

        # Wind prediction
        if energy_type in [EnergyType.WIND, EnergyType.BOTH]:
            wind_result = self._predict_wind(weather_df, include_uncertainty)
            predictions.append(wind_result)
            total_mw += wind_result.prediction_mw

        # Solar prediction
        if energy_type in [EnergyType.SOLAR, EnergyType.BOTH]:
            solar_result = self._predict_solar(weather_df, include_uncertainty)
            predictions.append(solar_result)
            total_mw += solar_result.prediction_mw

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            success=True,
            timestamp=weather.datetime,
            predictions=predictions,
            total_mw=total_mw if energy_type == EnergyType.BOTH else None,
            processing_time_ms=processing_time,
        )

    def predict_batch(
        self,
        weather_data: List[WeatherInput],
        energy_type: EnergyType = EnergyType.BOTH,
    ) -> BatchPredictionResponse:
        """
        배치 예측 수행

        Args:
            weather_data: 기상 데이터 리스트
            energy_type: 예측 타입

        Returns:
            BatchPredictionResponse
        """
        start_time = time.time()

        results = []
        all_predictions = {"wind": [], "solar": []}

        for weather in weather_data:
            response = self.predict(weather, energy_type, include_uncertainty=False)

            result = {
                "datetime": weather.datetime.isoformat(),
                "predictions": {p.energy_type.value: p.prediction_mw for p in response.predictions},
            }
            results.append(result)

            for p in response.predictions:
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
            statistics=statistics,
            processing_time_ms=processing_time,
        )

    def _weather_to_dataframe(self, weather: WeatherInput) -> pd.DataFrame:
        """WeatherInput을 DataFrame으로 변환"""
        data = {
            "datetime": [weather.datetime],
            "temperature": [weather.temperature],
            "humidity": [weather.humidity],
            "wind_speed": [weather.wind_speed],
            "wind_direction": [weather.wind_direction],
            "pressure": [weather.pressure],
            "solar_radiation": [weather.solar_radiation or 0.0],
            "cloud_cover": [weather.cloud_cover or 5.0],
            "visibility": [weather.visibility or 10.0],
            "precipitation": [weather.precipitation or 0.0],
        }

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        return df

    def _predict_wind(
        self,
        weather_df: pd.DataFrame,
        include_uncertainty: bool = True,
    ) -> PredictionResult:
        """풍력 예측"""
        if self.wind_model is None:
            return PredictionResult(
                energy_type=EnergyType.WIND,
                prediction_mw=0.0,
                lower_bound_mw=0.0,
                upper_bound_mw=0.0,
                model_r2=0.0,
            )

        # Feature engineering
        features_df = self._create_wind_features(weather_df)

        # Get model feature names
        model_features = self.wind_model.feature_name()

        # Align features
        X = self._align_features(features_df, model_features)

        # Predict
        prediction = float(np.clip(self.wind_model.predict(X)[0], 0, None))

        # Uncertainty bounds (based on model RMSE ~15 MW)
        rmse = 14.96
        lower_bound = max(0, prediction - 1.28 * rmse) if include_uncertainty else None
        upper_bound = prediction + 1.28 * rmse if include_uncertainty else None

        return PredictionResult(
            energy_type=EnergyType.WIND,
            prediction_mw=prediction,
            lower_bound_mw=lower_bound,
            upper_bound_mw=upper_bound,
            model_r2=self.wind_r2,
        )

    def _predict_solar(
        self,
        weather_df: pd.DataFrame,
        include_uncertainty: bool = True,
    ) -> PredictionResult:
        """태양광 예측"""
        if self.solar_model is None:
            return PredictionResult(
                energy_type=EnergyType.SOLAR,
                prediction_mw=0.0,
                lower_bound_mw=0.0,
                upper_bound_mw=0.0,
                model_r2=0.0,
            )

        # Feature engineering
        features_df = self._create_solar_features(weather_df)

        # Get model feature names
        model_features = self.solar_model.feature_name()

        # Align features
        X = self._align_features(features_df, model_features)

        # Predict
        prediction = float(np.clip(self.solar_model.predict(X)[0], 0, None))

        # Uncertainty bounds (based on model RMSE ~18 MW)
        rmse = 17.63
        lower_bound = max(0, prediction - 1.28 * rmse) if include_uncertainty else None
        upper_bound = prediction + 1.28 * rmse if include_uncertainty else None

        return PredictionResult(
            energy_type=EnergyType.SOLAR,
            prediction_mw=prediction,
            lower_bound_mw=lower_bound,
            upper_bound_mw=upper_bound,
            model_r2=self.solar_r2,
        )

    def _create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """풍력 모델용 피처 생성"""
        df = df.copy()
        dt = df.index

        # Basic temporal features
        df["hour"] = dt.hour
        df["dayofweek"] = dt.dayofweek
        df["month"] = dt.month
        df["dayofyear"] = dt.dayofyear

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        # Binary features
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
        df["season"] = df["month"].apply(lambda m: (m % 12 + 3) // 3 - 1)

        # Wind physics features
        wind_speed = df["wind_speed"].fillna(3.0)
        wind_dir = df["wind_direction"].fillna(0.0)
        temp = df["temperature"].fillna(15.0)
        pressure = df["pressure"].fillna(1013.25)

        # Air density
        R = 287.05
        T_kelvin = temp + 273.15
        df["air_density"] = (pressure * 100) / (R * T_kelvin)
        rho = df["air_density"]

        # Wind Power Density
        df["wind_power_density"] = 0.5 * rho * (wind_speed ** 3)
        df["wind_speed_cubed"] = wind_speed ** 3

        # Wind shear (hub height correction)
        h_ref, h_hub, alpha = 10, 80, 0.14
        df["wind_speed_hub"] = wind_speed * ((h_hub / h_ref) ** alpha)
        df["wind_shear_factor"] = (h_hub / h_ref) ** alpha
        df["wpd_hub"] = 0.5 * rho * (df["wind_speed_hub"] ** 3)

        # Turbine power curve
        cut_in, rated, cut_out = 3.0, 12.0, 25.0
        v_hub = df["wind_speed_hub"]
        df["is_cut_in"] = (v_hub >= cut_in).astype(int)
        df["is_rated"] = ((v_hub >= rated) & (v_hub < cut_out)).astype(int)
        df["is_cut_out"] = (v_hub >= cut_out).astype(int)

        partial = ((v_hub - cut_in) / (rated - cut_in)).clip(0, 1)
        df["theoretical_output"] = np.where(
            v_hub < cut_in, 0,
            np.where(v_hub >= cut_out, 0, np.where(v_hub >= rated, 1, partial))
        )

        # Wind direction features
        wind_dir_rad = np.radians(wind_dir)
        df["wind_dir_sin"] = np.sin(2 * np.pi * wind_dir / 360)
        df["wind_dir_cos"] = np.cos(2 * np.pi * wind_dir / 360)
        df["wind_dir_8"] = ((wind_dir + 22.5) // 45).astype(int) % 8
        df["wind_u"] = -wind_speed * np.sin(wind_dir_rad)
        df["wind_v"] = -wind_speed * np.cos(wind_dir_rad)
        df["wind_u_hub"] = -v_hub * np.sin(wind_dir_rad)
        df["wind_v_hub"] = -v_hub * np.cos(wind_dir_rad)

        # Wind direction sectors
        df["wind_sector_8"] = ((wind_dir + 22.5) // 45).astype(int) % 8
        df["is_northwest"] = ((wind_dir >= 270) | (wind_dir < 45)).astype(int)
        df["is_southeast"] = ((wind_dir >= 90) & (wind_dir < 180)).astype(int)

        # Atmospheric stability (simplified)
        hour = df["hour"].values[0]
        is_daytime = 1 if 7 <= hour <= 18 else 0
        df["stability_class"] = 4.0
        df["dynamic_alpha"] = 0.17 if is_daytime else 0.22
        df["thermal_stability"] = 0.14 if wind_speed.values[0] >= 3 else (0.1 if is_daytime else 0.25)

        # Dynamic wind shear
        df["wind_speed_hub_dynamic"] = wind_speed * ((h_hub / h_ref) ** df["dynamic_alpha"])
        df["wpd_hub_dynamic"] = 0.5 * rho * (df["wind_speed_hub_dynamic"] ** 3)

        # Efficiency features
        df["temp_efficiency_wind"] = (1 - 0.002 * np.abs(temp - 15)).clip(0.9, 1)
        df["combined_efficiency"] = df["temp_efficiency_wind"]
        df["theoretical_output_v2"] = df["theoretical_output"]
        df["theoretical_output_adjusted"] = df["theoretical_output"] * df["combined_efficiency"]

        # Temporal wind features
        df["is_sea_breeze_time"] = ((hour >= 10) & (hour <= 16)).astype(int)
        df["is_land_breeze_time"] = ((hour >= 22) | (hour <= 6)).astype(int)
        df["wind_sea_breeze"] = wind_speed * df["is_sea_breeze_time"]
        df["wind_land_breeze"] = wind_speed * df["is_land_breeze_time"]
        df["is_winter_wind"] = ((df["month"] >= 11) | (df["month"] <= 2)).astype(int)
        df["is_typhoon_season"] = ((df["month"] >= 6) & (df["month"] <= 9)).astype(int)
        df["hour_month_interaction"] = df["hour"] * df["month"] / 100
        df["is_peak_wind_time"] = ((hour >= 2) & (hour <= 10)).astype(int)

        # Historical pattern features (anomaly)
        if self.weather_patterns is not None:
            month_val = df["month"].values[0]
            hour_val = df["hour"].values[0]

            pattern = self.weather_patterns[
                (self.weather_patterns["month"] == month_val) &
                (self.weather_patterns["hour"] == hour_val)
            ]

            if not pattern.empty:
                p = pattern.iloc[0]
                df["wind_speed_anomaly"] = wind_speed.values[0] - p["wind_speed_mean"]
                df["temperature_anomaly"] = temp.values[0] - p["temperature_mean"]
                df["solar_radiation_anomaly"] = df["solar_radiation"].values[0] - p["solar_radiation_mean"]

                df["temperature_mean"] = p["temperature_mean"]
                df["temperature_min"] = p["temperature_min"]
                df["temperature_max"] = p["temperature_max"]
                df["humidity_mean"] = p["humidity_mean"]
                df["humidity_min"] = p["humidity_min"]
                df["humidity_max"] = p["humidity_max"]
                df["humidity_std"] = p["humidity_std"]
                df["pressure_mean"] = p["pressure_mean"]
                df["pressure_min"] = p["pressure_min"]
                df["pressure_max"] = p["pressure_max"]
                df["cloud_cover_mean"] = p["cloud_cover_mean"]
            else:
                df["wind_speed_anomaly"] = 0.0
                df["temperature_anomaly"] = 0.0
                df["solar_radiation_anomaly"] = 0.0
        else:
            df["wind_speed_anomaly"] = 0.0
            df["temperature_anomaly"] = 0.0
            df["solar_radiation_anomaly"] = 0.0

        # Lag/rolling features (use current value for single prediction)
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"wind_speed_lag_{lag}h"] = wind_speed

        df["wind_speed_change_1h"] = 0.0
        df["wind_speed_change_3h"] = 0.0
        df["wind_speed_change_6h"] = 0.0
        df["wind_speed_pct_change_1h"] = 0.0
        df["wind_speed_pct_change_3h"] = 0.0

        for window in [3, 6, 12, 24]:
            df[f"wind_speed_rolling_mean_{window}h"] = wind_speed
            df[f"wind_speed_rolling_std_{window}h"] = 0.0
            df[f"wind_speed_rolling_max_{window}h"] = wind_speed
            df[f"wind_speed_rolling_min_{window}h"] = wind_speed

        df["wind_acceleration"] = 0.0
        df["is_ramp_up"] = 0
        df["is_ramp_down"] = 0
        df["wind_dir_change_1h"] = 0.0
        df["wind_dir_stability_6h"] = 0.0

        # Pressure lag features
        for lag in [3, 6]:
            df[f"pressure_lag_{lag}h"] = pressure

        df["pressure_change_1h"] = 0.0
        df["pressure_change_3h"] = 0.0
        df["pressure_change_6h"] = 0.0
        df["is_pressure_drop"] = 0
        df["is_pressure_rise"] = 0
        df["pressure_trend"] = 0.0

        # Temperature lag features
        df["temp_lag_3h"] = temp
        df["temp_change_1h"] = 0.0
        df["temp_change_6h"] = 0.0
        df["temp_rise_rate"] = 0.0

        # Richardson number (simplified)
        df["richardson_number"] = 0.0
        df["stability_index"] = 0.0
        df["alpha_rolling_6h"] = df["dynamic_alpha"]
        df["wind_gradient"] = 0.0

        # Wind sector speeds
        for sector in range(8):
            mask = df["wind_sector_8"] == sector
            df[f"wind_speed_sector_{sector}"] = np.where(mask, wind_speed, 0)

        df["wind_speed_nw"] = wind_speed * df["is_northwest"]
        df["wind_speed_se"] = wind_speed * df["is_southeast"]
        df["strong_nw_wind"] = (wind_speed > 5).astype(int) * df["is_northwest"]
        df["strong_se_wind"] = (wind_speed > 5).astype(int) * df["is_southeast"]

        # Spatial lag features
        df["upstream_wind_1h"] = wind_speed
        df["upstream_wind_2h"] = wind_speed
        df["upstream_wind_change"] = 0.0
        df["wind_arriving"] = 0
        df["wind_leaving"] = 0
        df["spatial_weight_west"] = 1.0 if df["is_northwest"].values[0] else 0.3
        df["spatial_weight_east"] = 1.0 if df["is_southeast"].values[0] else 0.3
        df["weighted_upstream_wind"] = wind_speed * df["spatial_weight_west"]

        # Turbulence
        df["wind_speed_std_3h"] = 0.0
        df["wind_speed_mean_3h"] = wind_speed
        df["turbulence_intensity"] = 0.1

        return df

    def _create_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """태양광 모델용 피처 생성"""
        df = df.copy()
        dt = df.index

        # Basic temporal features
        df["hour"] = dt.hour
        df["dayofweek"] = dt.dayofweek
        df["month"] = dt.month
        df["dayofyear"] = dt.dayofyear

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        # Binary features
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
        df["season"] = df["month"].apply(lambda m: (m % 12 + 3) // 3 - 1)

        # Solar position (Jeju: 33.5N, 126.5E)
        latitude, longitude = 33.5, 126.5
        day_of_year = dt.dayofyear.values[0]
        hour = dt.hour.values[0] + dt.minute.values[0] / 60.0

        # Declination
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        declination_rad = np.radians(declination)

        # Hour angle
        solar_noon = 12.0 - (longitude - 135) / 15
        hour_angle = 15 * (hour - solar_noon)
        hour_angle_rad = np.radians(hour_angle)

        lat_rad = np.radians(latitude)

        # Solar altitude
        sin_altitude = (
            np.sin(lat_rad) * np.sin(declination_rad) +
            np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        )
        altitude = np.degrees(np.arcsin(np.clip(sin_altitude, -1, 1)))
        zenith = 90 - altitude

        # Azimuth
        altitude_safe = max(0.1, altitude)
        cos_azimuth = (
            np.sin(declination_rad) - np.sin(lat_rad) * sin_altitude
        ) / (np.cos(lat_rad) * np.cos(np.radians(altitude_safe)))
        azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))
        if hour_angle > 0:
            azimuth = 360 - azimuth

        df["solar_zenith"] = zenith
        df["solar_azimuth"] = azimuth
        df["solar_declination"] = declination
        df["solar_altitude"] = altitude
        df["altitude_sin"] = np.sin(np.radians(max(0, altitude)))
        df["altitude_cos"] = np.cos(np.radians(max(0, altitude)))

        # Air mass
        zenith_rad = np.radians(min(zenith, 89))
        air_mass = 1 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** -1.6364)
        air_mass = np.clip(air_mass, 1, 40)
        df["air_mass"] = air_mass

        # Theoretical DNI
        I0 = 1361
        k = 0.1
        altitude_sin_clipped = max(0, df["altitude_sin"].values[0])
        theoretical_dni = I0 * np.exp(-k * air_mass) * altitude_sin_clipped
        df["theoretical_dni"] = max(0, theoretical_dni)

        # Clear sky index
        solar_rad = df["solar_radiation"].values[0]
        if df["theoretical_dni"].values[0] > 0:
            df["clear_sky_index"] = np.clip(solar_rad / df["theoretical_dni"].values[0], 0, 2)
        else:
            df["clear_sky_index"] = 0.0

        df["solar_potential"] = solar_rad * altitude_sin_clipped

        # Sun elevation
        sun_elevation = np.sin(np.pi * (hour - 6) / 12)
        df["sun_elevation"] = max(0, sun_elevation)

        # Daylight flag
        df["is_daylight"] = (altitude > 0).astype(int)

        # Cloud factor
        cloud = df["cloud_cover"].fillna(5).values[0]
        df["cloud_factor"] = 1 - (cloud / 10)
        df["cloud_attenuation"] = np.exp(-0.1 * cloud)

        # Temperature efficiency
        temp = df["temperature"].fillna(20).values[0]
        temp_diff = max(0, temp - 25)
        df["temp_efficiency"] = np.clip(1 - 0.004 * temp_diff, 0.7, 1.0)

        # Humidity factor
        humidity = df["humidity"].fillna(60).values[0]
        df["humidity_factor"] = 1 - 0.001 * humidity

        return df

    def _align_features(
        self,
        features_df: pd.DataFrame,
        model_features: List[str],
    ) -> np.ndarray:
        """모델이 요구하는 피처 순서로 정렬"""
        aligned_df = pd.DataFrame(index=features_df.index)

        for feat in model_features:
            if feat in features_df.columns:
                aligned_df[feat] = features_df[feat]
            else:
                # Missing feature: use default value
                aligned_df[feat] = 0.0
                logger.debug(f"Missing feature '{feat}', using default 0.0")

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
