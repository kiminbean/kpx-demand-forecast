"""
Wind Feature Engineering V2 - Phase 1 Enhancement

Gemini 협업 토론 결과 반영:
1. 대기 안정도 지수 (Richardson Number, Pasquill-Gifford)
2. 동적 Wind Shear (시간대별 가변 alpha)
3. 기상 변수의 Lag/Rolling Features
4. 풍향-풍속 상호작용 Features (후류효과 간접 모델링)
5. 공간적 시차(Spatial Lag) Features
6. 발전 효율 관련 Features

목표: Wind R² 0.25 → 0.5~0.6
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class WindFeatureEngineerV2:
    """
    풍력 발전량 예측을 위한 고급 Feature Engineering

    Gemini 토론 결과 반영:
    - Physics-informed features
    - Atmospheric stability
    - Spatial-temporal features
    """

    def __init__(
        self,
        hub_height: float = 80.0,
        ref_height: float = 10.0,
        cut_in_speed: float = 3.0,
        rated_speed: float = 12.0,
        cut_out_speed: float = 25.0,
    ):
        """
        Args:
            hub_height: 터빈 허브 높이 (m)
            ref_height: 기상 관측 높이 (m)
            cut_in_speed: 발전 시작 풍속 (m/s)
            rated_speed: 정격 출력 풍속 (m/s)
            cut_out_speed: 과풍속 정지 풍속 (m/s)
        """
        self.hub_height = hub_height
        self.ref_height = ref_height
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 풍력 관련 피처 생성

        Args:
            df: 기상+발전량 DataFrame

        Returns:
            피처가 추가된 DataFrame
        """
        df = df.copy()

        # 1. 대기 안정도 피처
        df = self._add_atmospheric_stability_features(df)

        # 2. 동적 Wind Shear 피처
        df = self._add_dynamic_wind_shear_features(df)

        # 3. 기상 변수 Lag/Rolling 피처
        df = self._add_weather_lag_rolling_features(df)

        # 4. 풍향-풍속 상호작용 피처
        df = self._add_wind_direction_interaction_features(df)

        # 5. 발전 효율 관련 피처
        df = self._add_efficiency_features(df)

        # 6. 시간대별 피처
        df = self._add_temporal_wind_features(df)

        return df

    def _add_atmospheric_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        대기 안정도 관련 피처 추가

        Physics:
        - Richardson Number: 대기 안정도의 정량적 지표
        - 안정(Ri > 0.25): Wind Shear 강함, alpha 증가
        - 불안정(Ri < 0): 대류 활발, alpha 감소
        - 중립(0 < Ri < 0.25): 표준 조건
        """
        df = df.copy()

        # 필요한 컬럼 확인
        has_temp = 'temperature' in df.columns
        has_pressure = 'pressure' in df.columns
        has_wind = 'wind_speed' in df.columns

        if has_temp and has_wind:
            temp = df['temperature'].fillna(15)
            wind_speed = df['wind_speed'].fillna(df['wind_speed'].median())

            # 기온의 시간 변화율 (dT/dt를 dT/dz의 프록시로 사용)
            temp_change = temp.diff().fillna(0)

            # Gradient Richardson Number (간략화)
            # Ri = (g/T) * (dT/dz) / (du/dz)^2
            # 여기서는 시간 변화율로 근사
            g = 9.81  # 중력가속도
            T_kelvin = temp + 273.15

            # 풍속 변화율
            wind_change = wind_speed.diff().fillna(0)
            wind_change_sq = wind_change ** 2 + 1e-6  # 0 방지

            # Richardson Number (근사)
            df['richardson_number'] = (g / T_kelvin) * temp_change / wind_change_sq
            df['richardson_number'] = np.clip(df['richardson_number'], -10, 10)

            # 대기 안정도 클래스 (Pasquill-Gifford 근사)
            # A-F: 매우 불안정 ~ 매우 안정
            ri = df['richardson_number']
            stability_class = pd.cut(
                ri,
                bins=[-np.inf, -0.5, -0.1, 0.1, 0.25, 1.0, np.inf],
                labels=[1, 2, 3, 4, 5, 6]  # A=1(불안정) ~ F=6(안정)
            )
            df['stability_class'] = stability_class.astype(float).fillna(4)

            # 안정도 지표 (연속값)
            df['stability_index'] = np.where(
                ri < 0, -np.sqrt(-ri.clip(upper=0)),  # 불안정
                np.sqrt(ri.clip(lower=0))  # 안정
            )

            # 주간/야간 + 풍속 기반 간이 안정도
            if isinstance(df.index, pd.DatetimeIndex):
                hour = df.index.hour
            else:
                hour = pd.to_datetime(df.get('datetime', df.index)).dt.hour

            is_daytime = ((hour >= 7) & (hour <= 18)).astype(int)

            # 주간: 일사량 있으면 불안정 경향
            # 야간: 바람 약하면 안정 경향
            df['thermal_stability'] = np.where(
                is_daytime == 1,
                np.where(wind_speed < 3, 0.1, 0.14),  # 주간: 약풍=약간불안정, 강풍=중립
                np.where(wind_speed < 3, 0.25, 0.14)  # 야간: 약풍=안정, 강풍=중립
            )

        return df

    def _add_dynamic_wind_shear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        동적 Wind Shear 피처 추가

        Physics:
        - Wind Shear: v(h) = v(h_ref) * (h/h_ref)^α
        - α는 대기 안정도에 따라 변화
          - 불안정: α ≈ 0.10-0.15
          - 중립: α ≈ 0.14-0.20
          - 안정: α ≈ 0.25-0.40
        """
        df = df.copy()

        has_wind = 'wind_speed' in df.columns

        if has_wind:
            wind_speed = df['wind_speed'].fillna(df['wind_speed'].median())

            # 동적 alpha 계산
            if 'stability_class' in df.columns:
                stability = df['stability_class']
                # 안정도 클래스별 alpha 매핑
                alpha_map = {1: 0.10, 2: 0.12, 3: 0.14, 4: 0.17, 5: 0.22, 6: 0.30}
                df['dynamic_alpha'] = stability.map(alpha_map).fillna(0.14)
            elif 'thermal_stability' in df.columns:
                df['dynamic_alpha'] = df['thermal_stability']
            else:
                df['dynamic_alpha'] = 0.14

            # 동적 허브 높이 풍속
            height_ratio = self.hub_height / self.ref_height
            df['wind_speed_hub_dynamic'] = wind_speed * (height_ratio ** df['dynamic_alpha'])

            # 동적 WPD (허브 높이)
            if 'air_density' in df.columns:
                rho = df['air_density']
            else:
                rho = 1.225

            df['wpd_hub_dynamic'] = 0.5 * rho * (df['wind_speed_hub_dynamic'] ** 3)

            # Wind Shear Exponent의 롤링 평균 (안정성)
            df['alpha_rolling_6h'] = df['dynamic_alpha'].rolling(6, min_periods=1).mean()

            # 연직 풍속 기울기 (m/s per 10m)
            df['wind_gradient'] = (df['wind_speed_hub_dynamic'] - wind_speed) / (
                (self.hub_height - self.ref_height) / 10
            )

        return df

    def _add_weather_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기상 변수의 Lag/Rolling 피처 추가

        핵심: 타겟(발전량)이 아닌 기상 변수에 대한 시계열 피처
        - 풍속 변화 추세 학습
        - 기압 변화로 전선 통과 감지
        - 기온 변화로 대기 안정도 변화 감지
        """
        df = df.copy()

        # 풍속 Lag Features
        if 'wind_speed' in df.columns:
            wind = df['wind_speed'].fillna(method='ffill')

            # Lag features (1, 2, 3, 6, 12, 24시간 전)
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'wind_speed_lag_{lag}h'] = wind.shift(lag)

            # 풍속 변화량
            df['wind_speed_change_1h'] = wind.diff(1)
            df['wind_speed_change_3h'] = wind.diff(3)
            df['wind_speed_change_6h'] = wind.diff(6)

            # 풍속 변화율 (%)
            df['wind_speed_pct_change_1h'] = wind.pct_change(1) * 100
            df['wind_speed_pct_change_3h'] = wind.pct_change(3) * 100

            # Rolling Statistics
            for window in [3, 6, 12, 24]:
                df[f'wind_speed_rolling_mean_{window}h'] = wind.rolling(window, min_periods=1).mean()
                df[f'wind_speed_rolling_std_{window}h'] = wind.rolling(window, min_periods=1).std()
                df[f'wind_speed_rolling_max_{window}h'] = wind.rolling(window, min_periods=1).max()
                df[f'wind_speed_rolling_min_{window}h'] = wind.rolling(window, min_periods=1).min()

            # 풍속 가속도 (2차 미분)
            df['wind_acceleration'] = df['wind_speed_change_1h'].diff()

            # Ramp Event Detection (급격한 풍속 변화)
            df['is_ramp_up'] = (df['wind_speed_change_3h'] > 3).astype(int)
            df['is_ramp_down'] = (df['wind_speed_change_3h'] < -3).astype(int)

        # 풍향 Lag Features
        if 'wind_direction' in df.columns:
            wind_dir = df['wind_direction'].fillna(method='ffill')

            # 풍향 변화 (각도 차이 계산 시 360도 경계 처리)
            dir_diff = wind_dir.diff()
            dir_diff = np.where(dir_diff > 180, dir_diff - 360, dir_diff)
            dir_diff = np.where(dir_diff < -180, dir_diff + 360, dir_diff)
            df['wind_dir_change_1h'] = dir_diff

            # 풍향의 안정성 (표준편차)
            df['wind_dir_stability_6h'] = wind_dir.rolling(6, min_periods=1).std()

        # 기압 변화 Features (전선 통과 감지)
        if 'pressure' in df.columns:
            pressure = df['pressure'].fillna(method='ffill')

            df['pressure_lag_3h'] = pressure.shift(3)
            df['pressure_lag_6h'] = pressure.shift(6)

            df['pressure_change_1h'] = pressure.diff(1)
            df['pressure_change_3h'] = pressure.diff(3)
            df['pressure_change_6h'] = pressure.diff(6)

            # 급격한 기압 변화 (폭풍 전조)
            df['is_pressure_drop'] = (df['pressure_change_3h'] < -3).astype(int)
            df['is_pressure_rise'] = (df['pressure_change_3h'] > 3).astype(int)

            # 기압 경향 (상승/하강/정체)
            df['pressure_trend'] = np.sign(df['pressure_change_6h']).fillna(0)

        # 기온 변화 Features
        if 'temperature' in df.columns:
            temp = df['temperature'].fillna(method='ffill')

            df['temp_lag_3h'] = temp.shift(3)
            df['temp_change_1h'] = temp.diff(1)
            df['temp_change_6h'] = temp.diff(6)

            # 주간 기온 상승률 (대류 발달 지표)
            df['temp_rise_rate'] = df['temp_change_6h'].clip(lower=0)

        return df

    def _add_wind_direction_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        풍향-풍속 상호작용 피처 추가

        목적: 후류효과(Wake Effect) 간접 모델링
        - 특정 풍향에서 발전 효율이 다름
        - 풍향을 섹터로 나누어 효율 특성 학습
        """
        df = df.copy()

        has_wind = 'wind_speed' in df.columns
        has_dir = 'wind_direction' in df.columns

        if has_wind and has_dir:
            wind_speed = df['wind_speed'].fillna(df['wind_speed'].median())
            wind_dir = df['wind_direction'].fillna(0)

            # 풍향을 8방위로 분류
            df['wind_sector_8'] = ((wind_dir + 22.5) // 45).astype(int) % 8

            # 풍향을 16방위로 분류 (더 세밀)
            df['wind_sector_16'] = ((wind_dir + 11.25) // 22.5).astype(int) % 16

            # 풍향별 풍속 (상호작용)
            for sector in range(8):
                mask = df['wind_sector_8'] == sector
                df[f'wind_speed_sector_{sector}'] = np.where(mask, wind_speed, 0)

            # 풍향 벡터 성분
            wind_dir_rad = np.radians(wind_dir)
            df['wind_u'] = -wind_speed * np.sin(wind_dir_rad)  # 동서 성분
            df['wind_v'] = -wind_speed * np.cos(wind_dir_rad)  # 남북 성분

            # 허브 높이 풍속 기준
            if 'wind_speed_hub_dynamic' in df.columns:
                v_hub = df['wind_speed_hub_dynamic']
            elif 'wind_speed_hub' in df.columns:
                v_hub = df['wind_speed_hub']
            else:
                v_hub = wind_speed * 1.34  # 기본 보정

            df['wind_u_hub'] = -v_hub * np.sin(wind_dir_rad)
            df['wind_v_hub'] = -v_hub * np.cos(wind_dir_rad)

            # 제주도 주요 풍향 특성
            # 북서풍(겨울): 270-360, 남동풍(여름): 90-180
            df['is_northwest'] = ((wind_dir >= 270) | (wind_dir < 45)).astype(int)
            df['is_southeast'] = ((wind_dir >= 90) & (wind_dir < 180)).astype(int)

            # 주 풍향에서의 풍속 강화
            df['wind_speed_nw'] = wind_speed * df['is_northwest']
            df['wind_speed_se'] = wind_speed * df['is_southeast']

            # 풍향과 풍속의 조합 Feature
            # 강풍(>5m/s)이면서 특정 방향인 경우
            strong_wind = (wind_speed > 5).astype(int)
            df['strong_nw_wind'] = strong_wind * df['is_northwest']
            df['strong_se_wind'] = strong_wind * df['is_southeast']

        return df

    def _add_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        발전 효율 관련 피처 추가

        목적: 이론 대비 실제 발전량의 효율 학습
        """
        df = df.copy()

        # 이론적 출력 계산 (허브 높이 풍속 기준)
        if 'wind_speed_hub_dynamic' in df.columns:
            v = df['wind_speed_hub_dynamic']
        elif 'wind_speed_hub' in df.columns:
            v = df['wind_speed_hub']
        elif 'wind_speed' in df.columns:
            v = df['wind_speed'].fillna(3) * 1.34
        else:
            return df

        # Power Curve 기반 이론적 출력 비율
        partial = ((v - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)).clip(0, 1)
        theoretical = np.where(
            v < self.cut_in_speed, 0,
            np.where(v >= self.cut_out_speed, 0,
                     np.where(v >= self.rated_speed, 1, partial))
        )
        df['theoretical_output_v2'] = theoretical

        # 정격 출력 대비 비율 (추정)
        # 풍속의 3승에 비례하는 이론적 출력
        v_cubed_norm = (v ** 3) / (self.rated_speed ** 3)
        df['power_ratio_theoretical'] = np.clip(v_cubed_norm, 0, 1)

        # 효율 저하 요인들
        # 1. 난류 강도에 의한 효율 저하
        if 'turbulence_intensity' in df.columns:
            ti = df['turbulence_intensity'].fillna(0.1)
            df['ti_efficiency_loss'] = np.clip(1 - 0.5 * ti, 0.5, 1)

        # 2. 공기 밀도에 의한 효율 변화
        if 'air_density' in df.columns:
            rho = df['air_density']
            rho_std = 1.225
            df['density_efficiency'] = rho / rho_std

        # 3. 기온에 의한 효율 변화 (극저온/고온)
        if 'temperature' in df.columns:
            temp = df['temperature'].fillna(15)
            # 15도 기준, 극단적 기온에서 효율 저하
            df['temp_efficiency_wind'] = 1 - 0.002 * np.abs(temp - 15)
            df['temp_efficiency_wind'] = df['temp_efficiency_wind'].clip(0.9, 1)

        # 종합 효율 추정
        efficiency = pd.Series(np.ones(len(df)), index=df.index)
        if 'ti_efficiency_loss' in df.columns:
            efficiency *= df['ti_efficiency_loss']
        if 'density_efficiency' in df.columns:
            efficiency *= df['density_efficiency']
        if 'temp_efficiency_wind' in df.columns:
            efficiency *= df['temp_efficiency_wind']
        df['combined_efficiency'] = efficiency

        # 효율 적용된 이론적 출력
        df['theoretical_output_adjusted'] = df['theoretical_output_v2'] * efficiency

        return df

    def _add_temporal_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간대별 풍력 특성 피처

        목적: 일주기, 계절 패턴 학습 강화
        """
        df = df.copy()

        # datetime 인덱스 확보
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        elif 'datetime' in df.columns:
            dt = pd.to_datetime(df['datetime'])
        else:
            return df

        hour = dt.hour
        month = dt.month

        # 시간대별 풍력 특성 (제주도 해륙풍)
        # 낮: 해풍 (바다→육지), 밤: 육풍 (육지→바다)
        df['is_sea_breeze_time'] = ((hour >= 10) & (hour <= 16)).astype(int)
        df['is_land_breeze_time'] = ((hour >= 22) | (hour <= 6)).astype(int)

        # 풍속과 시간대 상호작용
        if 'wind_speed' in df.columns:
            wind = df['wind_speed'].fillna(3)
            df['wind_sea_breeze'] = wind * df['is_sea_breeze_time']
            df['wind_land_breeze'] = wind * df['is_land_breeze_time']

        # 계절별 풍력 특성
        # 겨울(11-2월): 강한 북서풍
        # 여름(6-8월): 태풍 시즌
        df['is_winter_wind'] = ((month >= 11) | (month <= 2)).astype(int)
        df['is_typhoon_season'] = ((month >= 6) & (month <= 9)).astype(int)

        # 시간-계절 조합
        df['hour_month_interaction'] = hour * month / 100

        # 피크 발전 시간대 (통계 기반)
        # 일반적으로 새벽~오전에 풍속이 높음
        df['is_peak_wind_time'] = ((hour >= 2) & (hour <= 10)).astype(int)

        return df


def add_spatial_lag_features(
    df: pd.DataFrame,
    station_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    공간적 시차(Spatial Lag) 피처 추가

    제주도 특성:
    - 서쪽(고산)에서 관측된 풍속 → 1-2시간 후 동쪽(성산)에 영향
    - 북쪽(제주)에서 관측된 풍속 → 남쪽(서귀포)에 영향

    Args:
        df: 메인 DataFrame
        station_data: 관측소별 데이터 딕셔너리 (없으면 현재 데이터 기반 생성)

    Returns:
        공간 시차 피처가 추가된 DataFrame
    """
    df = df.copy()

    # 단일 데이터셋인 경우, 시간 시차로 근사
    # (실제로는 여러 관측소 데이터가 필요)
    if 'wind_speed' in df.columns:
        wind = df['wind_speed'].fillna(method='ffill')

        # "서쪽에서 오는 바람" 시뮬레이션
        # 1-2시간 전 풍속을 upstream 풍속으로 간주
        df['upstream_wind_1h'] = wind.shift(1)
        df['upstream_wind_2h'] = wind.shift(2)

        # upstream 풍속 변화
        df['upstream_wind_change'] = df['upstream_wind_1h'] - df['upstream_wind_2h']

        # "바람이 온다" 지표
        df['wind_arriving'] = (df['upstream_wind_change'] > 0.5).astype(int)
        df['wind_leaving'] = (df['upstream_wind_change'] < -0.5).astype(int)

    if 'wind_direction' in df.columns:
        wind_dir = df['wind_direction'].fillna(method='ffill')

        # 풍향 기반 공간 시차 가중치
        # 북서풍: 서쪽 데이터가 중요
        # 남동풍: 동쪽 데이터가 중요
        nw_weight = np.where((wind_dir >= 270) | (wind_dir < 45), 1.0, 0.3)
        se_weight = np.where((wind_dir >= 90) & (wind_dir < 180), 1.0, 0.3)

        df['spatial_weight_west'] = nw_weight
        df['spatial_weight_east'] = se_weight

        # 가중 upstream 풍속
        if 'upstream_wind_1h' in df.columns:
            df['weighted_upstream_wind'] = df['upstream_wind_1h'] * df['spatial_weight_west']

    return df


def create_wind_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wind Feature Engineering V2 메인 함수

    모든 Phase 1 개선사항 적용

    Args:
        df: 기상+발전량 DataFrame

    Returns:
        개선된 피처가 추가된 DataFrame
    """
    # 기본 WindFeatureEngineerV2 적용
    fe = WindFeatureEngineerV2()
    df = fe.fit_transform(df)

    # 공간 시차 피처 추가
    df = add_spatial_lag_features(df)

    # 결측치 처리 (forward fill 후 backward fill)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # 무한값 처리
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    return df


if __name__ == "__main__":
    # 테스트
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config import PROCESSED_DATA_DIR

    print("=" * 60)
    print("Wind Feature Engineering V2 Test")
    print("=" * 60)

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "cleaned_kpx_dataset_v2.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        print(f"Original columns: {len(df.columns)}")

        # V2 피처 적용
        df_v2 = create_wind_features_v2(df)

        print(f"After V2 features: {len(df_v2.columns)}")
        print(f"\nNew features added: {len(df_v2.columns) - len(df.columns)}")

        # 새로 추가된 피처 목록
        new_cols = [c for c in df_v2.columns if c not in df.columns]
        print(f"\nNew feature names:")
        for col in sorted(new_cols):
            print(f"  - {col}")
    else:
        print(f"Data file not found: {data_path}")
