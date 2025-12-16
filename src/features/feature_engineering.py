"""
Feature Engineering Module

시계열 예측을 위한 특성 공학 모듈

주요 기능:
- 시간적 특성 생성 (cyclical encoding)
- 지연 특성 (lag features)
- 롤링 통계 (rolling statistics)
- 기상 파생 변수
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    시계열 예측을 위한 특성 공학 클래스

    Usage:
        fe = FeatureEngineer()
        df = fe.fit_transform(df, target_col='solar_mw')
    """

    def __init__(
        self,
        add_time_features: bool = True,
        add_lag_features: bool = True,
        add_rolling_features: bool = True,
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None,
    ):
        """
        Args:
            add_time_features: 시간 특성 추가 여부
            add_lag_features: 지연 특성 추가 여부
            add_rolling_features: 롤링 통계 추가 여부
            lag_periods: 지연 기간 목록 (기본: [1, 2, 3, 6, 12, 24])
            rolling_windows: 롤링 윈도우 크기 (기본: [3, 6, 12, 24])
        """
        self.add_time_features = add_time_features
        self.add_lag_features = add_lag_features
        self.add_rolling_features = add_rolling_features

        self.lag_periods = lag_periods or [1, 2, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows or [3, 6, 12, 24]

        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        datetime_col: str = 'datetime',
    ) -> pd.DataFrame:
        """
        특성 생성 및 스케일러 학습

        Args:
            df: 입력 DataFrame
            target_col: 타겟 컬럼 (지연/롤링 특성 대상)
            datetime_col: 날짜시간 컬럼

        Returns:
            특성이 추가된 DataFrame
        """
        df = df.copy()

        # datetime 인덱스 설정
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("datetime 인덱스 또는 datetime_col이 필요합니다.")

        # 시간 특성
        if self.add_time_features:
            df = self._add_time_features(df)

        # 지연 특성
        if self.add_lag_features and target_col:
            df = self._add_lag_features(df, target_col)

        # 롤링 특성
        if self.add_rolling_features and target_col:
            df = self._add_rolling_features(df, target_col)

        # 결측치 처리 (지연/롤링으로 발생한 NaN)
        df = df.dropna()

        self.feature_names = df.columns.tolist()
        self._is_fitted = True

        return df

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        datetime_col: str = 'datetime',
    ) -> pd.DataFrame:
        """
        학습된 설정으로 특성 생성 (스케일러 학습 없음)
        """
        return self.fit_transform(df, target_col, datetime_col)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 관련 특성 추가"""
        df = df.copy()

        # 기본 시간 특성
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear

        # Cyclical Encoding (원형 특성)
        # 시간 (24시간 주기)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 요일 (7일 주기)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 월 (12개월 주기)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 연간 일자 (365일 주기)
        df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

        # 이진 특성
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)

        # 계절 (0: 봄, 1: 여름, 2: 가을, 3: 겨울)
        df['season'] = df['month'].apply(lambda m: (m % 12 + 3) // 3 - 1)

        return df

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """지연 특성 추가"""
        df = df.copy()

        for lag in self.lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # 차분 특성 (변화량)
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24'] = df[target_col].diff(24)  # 전일 동시간 대비

        return df

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """롤링 통계 특성 추가"""
        df = df.copy()

        for window in self.rolling_windows:
            # 이동 평균
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).mean()

            # 이동 표준편차
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).std()

            # 이동 최대/최소
            df[f'{target_col}_max_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).max()
            df[f'{target_col}_min_{window}'] = df[target_col].rolling(
                window=window, min_periods=1
            ).min()

        return df

    def get_feature_names(self) -> List[str]:
        """생성된 특성 이름 반환"""
        return self.feature_names


def calculate_solar_position(datetime_index: pd.DatetimeIndex, latitude: float = 33.5, longitude: float = 126.5):
    """
    태양 위치 계산 (천문학적 계산 기반)

    Args:
        datetime_index: DatetimeIndex
        latitude: 위도 (제주: 33.5)
        longitude: 경도 (제주: 126.5)

    Returns:
        (zenith_angle, azimuth_angle, declination) 튜플
    """
    # Day of year
    day_of_year = datetime_index.dayofyear
    hour = datetime_index.hour + datetime_index.minute / 60.0

    # 태양 적위 (declination) - 계절에 따른 변화
    declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    declination_rad = np.radians(declination)

    # 시간각 (hour angle)
    # 태양 정오 = 12시 (대략적)
    solar_noon = 12.0 - (longitude - 135) / 15  # 한국 표준시 기준 보정
    hour_angle = 15 * (hour - solar_noon)  # 도 단위
    hour_angle_rad = np.radians(hour_angle)

    # 위도를 라디안으로
    lat_rad = np.radians(latitude)

    # 태양 고도각 (solar altitude / elevation)
    sin_altitude = (
        np.sin(lat_rad) * np.sin(declination_rad) +
        np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    altitude = np.degrees(np.arcsin(np.clip(sin_altitude, -1, 1)))

    # 천정각 (zenith angle) = 90 - 고도각
    zenith = 90 - altitude

    # 방위각 (azimuth) - 간략화된 계산
    altitude_safe = np.clip(altitude, 0.1, None)  # numpy array로 변환
    cos_azimuth = (
        np.sin(declination_rad) - np.sin(lat_rad) * sin_altitude
    ) / (np.cos(lat_rad) * np.cos(np.radians(altitude_safe)))
    azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))

    # 오후에는 방위각 보정
    azimuth = np.where(hour_angle > 0, 360 - azimuth, azimuth)

    return zenith, azimuth, declination


def create_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    태양광 발전 예측을 위한 물리 기반 특성 생성

    Physics-Informed Features:
    - Solar Zenith/Azimuth Angle (천문학적 태양 위치)
    - Clear Sky Index (맑은 하늘 대비 실제 일사량)
    - Temperature Efficiency (패널 온도 효율)

    Args:
        df: 기상+발전량 DataFrame

    Returns:
        태양광 특성이 추가된 DataFrame
    """
    df = df.copy()

    # datetime 인덱스 확보
    if isinstance(df.index, pd.DatetimeIndex):
        dt_index = df.index
        hour = df.index.hour
        day_of_year = df.index.dayofyear
    else:
        dt_index = pd.to_datetime(df['datetime'])
        hour = dt_index.hour
        day_of_year = dt_index.dayofyear

    # === Physics-Informed: 태양 위치 계산 (제주도 기준) ===
    zenith, azimuth, declination = calculate_solar_position(dt_index)

    # numpy array로 변환 (pandas Index 호환성)
    zenith = np.array(zenith)
    azimuth = np.array(azimuth)
    declination = np.array(declination)

    df['solar_zenith'] = zenith
    df['solar_azimuth'] = azimuth
    df['solar_declination'] = declination

    # 태양 고도각 (zenith의 보각)
    df['solar_altitude'] = 90 - zenith

    # 고도각의 sin/cos (모델 학습에 유용)
    df['altitude_sin'] = np.sin(np.radians(np.clip(df['solar_altitude'].values, 0, 90)))
    df['altitude_cos'] = np.cos(np.radians(np.clip(df['solar_altitude'].values, 0, 90)))

    # === 이론적 최대 일사량 (Clear Sky Model) ===
    # 대기질량 (Air Mass) 계산
    zenith_rad = np.radians(np.clip(zenith, 0, 89))
    air_mass = 1 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    air_mass = np.clip(air_mass, 1, 40)
    df['air_mass'] = air_mass

    # 이론적 직달 일사량 (W/m²)
    # I = I0 * exp(-k * AM) 형태의 간략화
    I0 = 1361  # 태양 상수 (W/m²)
    k = 0.1   # 대기 소산 계수 (맑은 날 기준)
    altitude_sin_clipped = np.clip(df['altitude_sin'].values, 0, 1)
    theoretical_dni = I0 * np.exp(-k * air_mass) * altitude_sin_clipped
    df['theoretical_dni'] = np.clip(theoretical_dni, 0, None)

    # Clear Sky Index (실제/이론)
    if 'solar_radiation' in df.columns:
        df['clear_sky_index'] = df['solar_radiation'] / df['theoretical_dni'].replace(0, np.nan)
        df['clear_sky_index'] = np.clip(df['clear_sky_index'].fillna(0).values, 0, 2)
        df['solar_potential'] = df['solar_radiation'] * altitude_sin_clipped

    # === 기존 특성 (개선) ===
    # 태양 위치 기반 발전 가능 시간대
    sun_elevation = np.sin(np.pi * (np.array(hour) - 6) / 12)
    df['sun_elevation'] = np.clip(sun_elevation, 0, 1)

    # 일출/일몰 플래그 (태양 고도 > 0)
    df['is_daylight'] = (df['solar_altitude'] > 0).astype(int)

    # 구름 영향
    if 'cloud_cover' in df.columns:
        cloud = df['cloud_cover'].fillna(5)
        df['cloud_factor'] = 1 - (cloud / 10)  # 0-10 스케일
        df['cloud_attenuation'] = np.exp(-0.1 * cloud)  # 지수 감쇠 모델

    # 기온-효율 관계 (고온에서 효율 저하)
    if 'temperature' in df.columns:
        temp = df['temperature'].fillna(20).values
        # 25도 기준, 1도당 0.4~0.5% 효율 감소 (crystalline silicon 기준)
        temp_diff = np.clip(temp - 25, 0, None)
        df['temp_efficiency'] = np.clip(1 - 0.004 * temp_diff, 0.7, 1.0)

    # 습도 영향 (높은 습도는 산란 증가)
    if 'humidity' in df.columns:
        humidity = df['humidity'].fillna(60)
        df['humidity_factor'] = 1 - 0.001 * humidity

    return df


def create_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    풍력 발전 예측을 위한 물리 기반 특성 생성

    Physics-Informed Features:
    - Wind Power Density (WPD = 0.5 * ρ * v³)
    - Wind Shear (고도 보정)
    - Turbine Power Curve (cut-in, rated, cut-out)
    - Air Density (이상기체 방정식)

    Args:
        df: 기상+발전량 DataFrame

    Returns:
        풍력 특성이 추가된 DataFrame
    """
    df = df.copy()

    # === 공기 밀도 계산 (먼저 계산 - WPD에 필요) ===
    # 표준 공기 밀도
    rho_std = 1.225  # kg/m³ (해수면, 15°C 기준)

    if 'pressure' in df.columns and 'temperature' in df.columns:
        # 이상기체 방정식: ρ = P / (R * T)
        R = 287.05  # J/(kg·K) - 건조 공기 기체 상수
        pressure = df['pressure'].fillna(1013.25)  # hPa
        temperature = df['temperature'].fillna(15)

        T_kelvin = temperature + 273.15
        df['air_density'] = (pressure * 100) / (R * T_kelvin)
        rho = df['air_density']
    else:
        df['air_density'] = rho_std
        rho = rho_std

    # === 풍속 관련 특성 ===
    if 'wind_speed' in df.columns:
        # 결측치 처리
        wind_speed = df['wind_speed'].fillna(df['wind_speed'].median())

        # === Physics-Informed: Wind Power Density (WPD) ===
        # WPD = 0.5 * ρ * v³ (W/m²)
        df['wind_power_density'] = 0.5 * rho * (wind_speed ** 3)

        # 풍속의 3승 (명시적)
        df['wind_speed_cubed'] = wind_speed ** 3

        # === Wind Shear (고도 보정) ===
        # Power Law: v(h) = v(h_ref) * (h / h_ref)^α
        # α = 풍속 지수 (평지: 0.14, 거친 지형: 0.25)
        # 기상청 관측 높이: 10m, 일반 풍력터빈 허브 높이: 80-100m
        h_ref = 10  # 기상청 관측 높이 (m)
        h_hub = 80  # 터빈 허브 높이 (m)
        alpha = 0.14  # 풍속 지수 (해안가/평지 기준)

        df['wind_speed_hub'] = wind_speed * ((h_hub / h_ref) ** alpha)
        df['wind_shear_factor'] = (h_hub / h_ref) ** alpha

        # 허브 높이에서의 WPD
        df['wpd_hub'] = 0.5 * rho * (df['wind_speed_hub'] ** 3)

        # === Turbine Power Curve ===
        # 일반적인 풍력터빈 기준 (2-3MW급)
        cut_in = 3.0    # m/s - 발전 시작
        rated = 12.0    # m/s - 정격 출력 도달
        cut_out = 25.0  # m/s - 과풍속 정지

        v_hub = df['wind_speed_hub']

        # 풍속 구간 플래그
        df['is_cut_in'] = (v_hub >= cut_in).astype(int)
        df['is_rated'] = ((v_hub >= rated) & (v_hub < cut_out)).astype(int)
        df['is_cut_out'] = (v_hub >= cut_out).astype(int)

        # 이론적 출력 비율 (0~1)
        # 부분 출력 구간: (v - cut_in) / (rated - cut_in)
        partial_output = ((v_hub - cut_in) / (rated - cut_in)).clip(0, 1)
        df['theoretical_output'] = np.where(
            v_hub < cut_in, 0,
            np.where(v_hub >= cut_out, 0,
                     np.where(v_hub >= rated, 1, partial_output))
        )

        # 풍속 구간 (카테고리)
        wind_zone = pd.cut(
            wind_speed,
            bins=[-np.inf, cut_in, rated, cut_out, np.inf],
            labels=[0, 1, 2, 0],  # 0: 발전불가, 1: 부분출력, 2: 정격출력
            ordered=False
        )
        df['wind_zone'] = wind_zone.astype(float).fillna(0).astype(int)

        # === Turbulence Intensity (근사) ===
        # TI = σ_v / v_mean (표준편차 / 평균)
        # 롤링 윈도우로 근사
        df['wind_speed_std_3h'] = wind_speed.rolling(3, min_periods=1).std()
        df['wind_speed_mean_3h'] = wind_speed.rolling(3, min_periods=1).mean()
        df['turbulence_intensity'] = (
            df['wind_speed_std_3h'] / df['wind_speed_mean_3h'].replace(0, np.nan)
        ).fillna(0)

    # === 풍향 특성 (원형 인코딩) ===
    if 'wind_direction' in df.columns:
        wind_dir = df['wind_direction'].fillna(0)
        df['wind_dir_sin'] = np.sin(2 * np.pi * wind_dir / 360)
        df['wind_dir_cos'] = np.cos(2 * np.pi * wind_dir / 360)

        # 주요 풍향 (8방위)
        df['wind_dir_8'] = ((wind_dir + 22.5) // 45).astype(int) % 8

    # === 기압 변화 (전선 통과 지표) ===
    if 'pressure' in df.columns:
        pressure = df['pressure'].fillna(method='ffill')
        df['pressure_change_1h'] = pressure.diff(1)
        df['pressure_change_3h'] = pressure.diff(3)
        df['pressure_change_6h'] = pressure.diff(6)

        # 급격한 기압 변화 플래그 (폭풍 전조)
        df['pressure_drop_flag'] = (df['pressure_change_3h'] < -3).astype(int)

    return df


def create_sequences(
    data: np.ndarray,
    seq_length: int,
    target_col_idx: int = 0,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 시퀀스 생성 (LSTM/Transformer 입력용)

    Args:
        data: 입력 배열 (samples, features)
        seq_length: 입력 시퀀스 길이
        target_col_idx: 타겟 컬럼 인덱스
        forecast_horizon: 예측 기간

    Returns:
        (X, y) 튜플
        - X: (samples, seq_length, features)
        - y: (samples, forecast_horizon)
    """
    X, y = [], []

    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, target_col_idx])

    return np.array(X), np.array(y)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시계열 데이터 분할 (시간 순서 유지)

    Args:
        df: 입력 DataFrame
        train_ratio: 학습 비율
        val_ratio: 검증 비율
        test_ratio: 테스트 비율

    Returns:
        (train_df, val_df, test_df) 튜플
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df):,} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df):,} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df):,} ({len(test_df)/n:.1%})")

    return train_df, val_df, test_df


class DataScaler:
    """
    데이터 스케일링 유틸리티

    시계열 데이터의 스케일링을 관리합니다.
    학습 데이터에서 fit하고, 검증/테스트에서 transform합니다.
    """

    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (StandardScaler) 또는 'minmax' (MinMaxScaler)
        """
        self.method = method
        self.scalers: Dict[str, object] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """학습 및 변환"""
        df = df.copy()
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if self.method == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler

        return df

    def transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """변환만 수행"""
        df = df.copy()
        columns = columns or list(self.scalers.keys())

        for col in columns:
            if col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]])

        return df

    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """역변환"""
        df = df.copy()
        columns = columns or list(self.scalers.keys())

        for col in columns:
            if col in self.scalers:
                df[col] = self.scalers[col].inverse_transform(df[[col]])

        return df


if __name__ == "__main__":
    # 테스트
    print("Feature Engineering 모듈 테스트")

    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'solar_mw': np.random.rand(100) * 100,
        'temperature': np.random.rand(100) * 30,
        'wind_speed': np.random.rand(100) * 15,
    })

    # Feature Engineering
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df, target_col='solar_mw')

    print(f"\n원본 컬럼: {len(df.columns)}")
    print(f"특성 추가 후: {len(df_features.columns)}")
    print(f"\n생성된 특성:")
    print(df_features.columns.tolist())
