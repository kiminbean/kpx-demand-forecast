"""
Data Cleaning Module

Gemini 권장 데이터 클렌징:
1. ESS 방전 데이터 필터링 (야간 Solar > 0)
2. 출력 제어(Curtailment) 구간 탐지

Reference:
- ESS 방전: 태양광 설비는 야간에 직접 발전 불가, KPX 데이터의 야간 발전은 ESS 방전
- 출력 제어: 계통 안정화를 위해 풍력 발전기 출력을 인위적으로 제한

Gemini 팩트체크 (2024-12):
- 기존 ratio 방식(풍속>=5m/s & ratio<0.3)은 심각한 오탐 포함
- 5-7m/s 풍속은 터빈 특성상 낮은 출력이 정상 (파워커브)
- 실제 출력 제어는 '풍속 유지 + 발전량 급락' 패턴 (디커플링)
- 시간대 조건: 태양광 피크 시간(11-15시)에 집중
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mask_nighttime_solar(
    df: pd.DataFrame,
    solar_col: str = 'solar_mw',
    zenith_col: str = 'solar_zenith',
    zenith_threshold: float = 90.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    야간 ESS 방전 데이터를 0으로 마스킹

    KPX 데이터의 야간 태양광 발전량은 실제 태양광이 아닌 ESS 방전량.
    Solar Zenith Angle > 90도 = 해가 지평선 아래 = 야간

    Args:
        df: 입력 DataFrame
        solar_col: 태양광 발전량 컬럼명
        zenith_col: 태양 천정각 컬럼명
        zenith_threshold: 야간 판단 임계값 (기본: 90도)
        verbose: 상세 출력 여부

    Returns:
        클렌징된 DataFrame
    """
    df = df.copy()

    if zenith_col not in df.columns:
        logger.warning(f"Column '{zenith_col}' not found. Skipping nighttime masking.")
        return df

    if solar_col not in df.columns:
        logger.warning(f"Column '{solar_col}' not found. Skipping nighttime masking.")
        return df

    # 야간 마스크 (Solar Zenith > 90)
    is_night = df[zenith_col] > zenith_threshold

    # 야간 + 발전량 > 0 인 경우 = ESS 방전 의심
    ess_discharge_mask = is_night & (df[solar_col] > 0)
    n_ess_records = ess_discharge_mask.sum()

    if verbose:
        logger.info(f"[ESS Discharge Detection]")
        logger.info(f"  - Total records: {len(df):,}")
        logger.info(f"  - Nighttime records: {is_night.sum():,}")
        logger.info(f"  - ESS discharge (night + solar > 0): {n_ess_records:,}")

    # ESS 방전 데이터를 0으로 마스킹
    original_sum = df[solar_col].sum()
    df.loc[ess_discharge_mask, solar_col] = 0
    masked_sum = df[solar_col].sum()

    if verbose:
        logger.info(f"  - Original total: {original_sum:,.2f} MWh")
        logger.info(f"  - After masking: {masked_sum:,.2f} MWh")
        logger.info(f"  - Removed: {original_sum - masked_sum:,.2f} MWh ({(original_sum - masked_sum) / original_sum * 100:.2f}%)")

    return df


def detect_wind_curtailment(
    df: pd.DataFrame,
    wind_col: str = 'wind_mw',
    wind_speed_col: str = 'wind_speed',
    wpd_col: str = 'wind_power_density',
    method: str = 'ratio',
    threshold: float = 0.3,
    min_wind_speed: float = 5.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    풍력 출력 제어(Curtailment) 구간 탐지

    출력 제어: 풍속은 높은데 발전량이 비정상적으로 낮은 구간
    제주도는 계통 안정성을 위해 빈번하게 출력 제어 실시

    Methods:
    1. 'ratio': 발전량/이론출력 비율이 threshold 미만
    2. 'residual': 예상 발전량 대비 잔차가 큰 구간
    3. 'zscore': 풍속 구간별 발전량의 Z-score 기반

    Args:
        df: 입력 DataFrame
        wind_col: 풍력 발전량 컬럼명
        wind_speed_col: 풍속 컬럼명
        wpd_col: Wind Power Density 컬럼명
        method: 탐지 방법 ('ratio', 'residual', 'zscore')
        threshold: 탐지 임계값
        min_wind_speed: 최소 풍속 (이하는 정상으로 판단)
        verbose: 상세 출력 여부

    Returns:
        (df, curtailment_mask) 튜플
    """
    df = df.copy()

    # 컬럼 확인
    required_cols = [wind_col, wind_speed_col]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found. Cannot detect curtailment.")
            return df, pd.Series([False] * len(df), index=df.index)

    wind_speed = df[wind_speed_col].fillna(0)
    wind_gen = df[wind_col].fillna(0)

    # 출력 제어 마스크 초기화
    curtailment_mask = pd.Series([False] * len(df), index=df.index)

    if method == 'ratio':
        # 방법 1: 이론 출력 대비 실제 출력 비율
        # 풍속 >= min_wind_speed 인데 발전량이 비정상적으로 낮은 경우

        # 이론적 최대 출력 추정 (풍속 구간별 최대값)
        # 풍속을 0.5m/s 구간으로 나누어 각 구간별 최대 발전량 참조
        wind_bins = pd.cut(wind_speed, bins=np.arange(0, 30.5, 0.5))
        max_gen_per_bin = df.groupby(wind_bins, observed=True)[wind_col].max()

        # 각 레코드에 해당 풍속 구간의 최대 발전량 매핑
        expected_max = wind_bins.map(max_gen_per_bin).astype(float)
        expected_max = expected_max.fillna(0)

        # 실제/최대 비율 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            output_ratio = wind_gen / expected_max
            output_ratio = output_ratio.replace([np.inf, -np.inf], 0).fillna(0)

        # 출력 제어 탐지: 풍속 충분 + 출력 비율 낮음
        curtailment_mask = (
            (wind_speed >= min_wind_speed) &  # 풍속이 충분함
            (expected_max > 0) &              # 해당 풍속에서 발전 가능
            (output_ratio < threshold)         # 비율이 임계값 미만
        )

        df['expected_max_gen'] = expected_max
        df['output_ratio'] = output_ratio

    elif method == 'zscore':
        # 방법 2: 풍속 구간별 Z-score 기반
        # 같은 풍속에서 발전량이 평균 대비 얼마나 낮은지

        wind_bins = pd.cut(wind_speed, bins=np.arange(0, 30.5, 1.0))

        # 구간별 평균/표준편차 계산
        bin_stats = df.groupby(wind_bins, observed=True)[wind_col].agg(['mean', 'std'])

        # Z-score 계산
        zscore = pd.Series(index=df.index, dtype=float)
        for bin_label, group_idx in df.groupby(wind_bins, observed=True).groups.items():
            if bin_label in bin_stats.index:
                mean_val = bin_stats.loc[bin_label, 'mean']
                std_val = bin_stats.loc[bin_label, 'std']
                if pd.notna(std_val) and std_val > 0:
                    zscore.loc[group_idx] = (df.loc[group_idx, wind_col] - mean_val) / std_val
                else:
                    zscore.loc[group_idx] = 0

        zscore = zscore.fillna(0)

        # Z-score가 -threshold 이하 + 풍속 충분 = 출력 제어 의심
        curtailment_mask = (
            (wind_speed >= min_wind_speed) &
            (zscore < -threshold)
        )

        df['wind_zscore'] = zscore

    elif method == 'residual':
        # 방법 3: WPD 기반 잔차
        if wpd_col in df.columns:
            wpd = df[wpd_col].fillna(0)

            # 단순 선형 관계 가정: Generation ~ WPD
            # 비율로 정규화
            with np.errstate(divide='ignore', invalid='ignore'):
                wpd_ratio = wind_gen / wpd
                wpd_ratio = wpd_ratio.replace([np.inf, -np.inf], 0).fillna(0)

            # WPD가 높은데 발전량이 낮은 경우
            wpd_median_ratio = wpd_ratio.median()
            curtailment_mask = (
                (wind_speed >= min_wind_speed) &
                (wpd > wpd.quantile(0.5)) &           # WPD가 중앙값 이상
                (wpd_ratio < wpd_median_ratio * threshold)  # 비율이 낮음
            )

            df['wpd_ratio'] = wpd_ratio
        else:
            logger.warning(f"WPD column '{wpd_col}' not found. Using 'ratio' method instead.")
            return detect_wind_curtailment(df, wind_col, wind_speed_col, wpd_col, 'ratio', threshold, min_wind_speed, verbose)

    n_curtailment = curtailment_mask.sum()

    if verbose:
        logger.info(f"[Wind Curtailment Detection]")
        logger.info(f"  - Method: {method}")
        logger.info(f"  - Threshold: {threshold}")
        logger.info(f"  - Min wind speed: {min_wind_speed} m/s")
        logger.info(f"  - Total records: {len(df):,}")
        logger.info(f"  - Curtailment detected: {n_curtailment:,} ({n_curtailment / len(df) * 100:.2f}%)")

        if n_curtailment > 0:
            curt_df = df[curtailment_mask]
            logger.info(f"  - Avg wind speed (curtailed): {curt_df[wind_speed_col].mean():.2f} m/s")
            logger.info(f"  - Avg generation (curtailed): {curt_df[wind_col].mean():.2f} MW")

    df['is_curtailed'] = curtailment_mask.astype(int)

    return df, curtailment_mask


def detect_wind_curtailment_v2(
    df: pd.DataFrame,
    wind_col: str = 'wind_mw',
    wind_speed_col: str = 'wind_speed',
    solar_col: str = 'solar_mw',
    datetime_col: str = 'datetime',
    wind_speed_change_threshold: float = 1.0,
    generation_drop_threshold: float = 20.0,
    min_wind_speed: float = 7.0,
    solar_peak_hours: Tuple[int, int] = (11, 15),
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    개선된 풍력 출력 제어(Curtailment) 탐지 (Gemini 피드백 반영)

    디커플링 패턴 기반:
    - 풍속은 유지되는데 (변화 < threshold)
    - 발전량만 급락 (drop > threshold)
    - 태양광 피크 시간대 가중

    Args:
        df: 입력 DataFrame
        wind_col: 풍력 발전량 컬럼명
        wind_speed_col: 풍속 컬럼명
        solar_col: 태양광 발전량 컬럼명
        datetime_col: datetime 컬럼명
        wind_speed_change_threshold: 풍속 변화 임계값 (m/s)
        generation_drop_threshold: 발전량 급락 임계값 (MW)
        min_wind_speed: 최소 풍속 (정격 출력 근접 구간)
        solar_peak_hours: 태양광 피크 시간대 (시작, 끝)
        verbose: 상세 출력 여부

    Returns:
        (df, curtailment_mask) 튜플
    """
    df = df.copy()

    # datetime 처리
    if datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col])
        hour = dt.dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        logger.warning("No datetime information found. Skipping time-based filtering.")
        hour = pd.Series([12] * len(df))  # 기본값

    # 변화율 계산 (시계열 정렬 필요)
    df_sorted = df.sort_values(datetime_col) if datetime_col in df.columns else df

    wind_speed = df_sorted[wind_speed_col].fillna(0)
    wind_gen = df_sorted[wind_col].fillna(0)

    wind_speed_change = wind_speed.diff().abs()
    wind_gen_change = wind_gen.diff()

    # 조건 1: 디커플링 패턴
    # 풍속 변화 적음 + 발전량 급락
    decoupling_mask = (
        (wind_speed_change < wind_speed_change_threshold) &  # 풍속 거의 변화 없음
        (wind_gen_change < -generation_drop_threshold) &     # 발전량 급락
        (wind_speed >= min_wind_speed)                       # 풍속 충분
    )

    # 조건 2: 태양광 피크 시간대 (가중)
    # 제주도 출력 제어는 태양광 피크 시 발생
    is_solar_peak = (hour >= solar_peak_hours[0]) & (hour <= solar_peak_hours[1])

    # 조건 3: 태양광 발전량이 높을 때
    if solar_col in df.columns:
        solar_high = df[solar_col].quantile(0.7)  # 상위 30%
        is_solar_high = df[solar_col] > solar_high
    else:
        is_solar_high = pd.Series([True] * len(df), index=df.index)

    # 복합 조건: 디커플링 + (태양광 피크 OR 태양광 높음)
    curtailment_mask = decoupling_mask & (is_solar_peak | is_solar_high)

    # 인덱스 정렬 복원
    if datetime_col in df.columns:
        curtailment_mask = curtailment_mask.reindex(df.index)
        wind_speed_change = wind_speed_change.reindex(df.index)
        wind_gen_change = wind_gen_change.reindex(df.index)

    n_curtailment = curtailment_mask.sum()

    if verbose:
        logger.info(f"[Wind Curtailment Detection V2 - Decoupling Pattern]")
        logger.info(f"  - Wind speed change threshold: {wind_speed_change_threshold} m/s")
        logger.info(f"  - Generation drop threshold: {generation_drop_threshold} MW")
        logger.info(f"  - Min wind speed: {min_wind_speed} m/s")
        logger.info(f"  - Solar peak hours: {solar_peak_hours[0]}-{solar_peak_hours[1]}시")
        logger.info(f"  - Total records: {len(df):,}")
        logger.info(f"  - Decoupling patterns: {decoupling_mask.sum():,}")
        logger.info(f"  - Final curtailment: {n_curtailment:,} ({n_curtailment / len(df) * 100:.2f}%)")

        if n_curtailment > 0:
            curt_df = df[curtailment_mask]
            logger.info(f"  - Avg wind speed (curtailed): {curt_df[wind_speed_col].mean():.2f} m/s")
            logger.info(f"  - Avg generation (curtailed): {curt_df[wind_col].mean():.2f} MW")

    # 파생 변수 추가
    df['wind_speed_change'] = wind_speed_change
    df['wind_gen_change'] = wind_gen_change
    df['is_decoupling'] = decoupling_mask.astype(int)
    df['is_curtailed_v2'] = curtailment_mask.astype(int)

    return df, curtailment_mask


def filter_curtailment(
    df: pd.DataFrame,
    curtailment_mask: pd.Series,
    strategy: str = 'mask',
    fill_value: float = np.nan,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    출력 제어 구간 처리

    Strategies:
    1. 'drop': 해당 레코드 제거
    2. 'mask': 발전량을 특정 값으로 대체
    3. 'keep': 플래그만 추가하고 유지 (학습 시 필터링)

    Args:
        df: 입력 DataFrame
        curtailment_mask: 출력 제어 마스크
        strategy: 처리 전략
        fill_value: mask 전략 시 대체값 (default: NaN)
        verbose: 상세 출력 여부

    Returns:
        처리된 DataFrame
    """
    df = df.copy()
    n_curtailed = curtailment_mask.sum()

    if verbose:
        logger.info(f"[Curtailment Filtering]")
        logger.info(f"  - Strategy: {strategy}")
        logger.info(f"  - Records to process: {n_curtailed:,}")

    if strategy == 'drop':
        df = df[~curtailment_mask]
        if verbose:
            logger.info(f"  - Records after drop: {len(df):,}")

    elif strategy == 'mask':
        if 'wind_mw' in df.columns:
            df.loc[curtailment_mask, 'wind_mw'] = fill_value
        if verbose:
            logger.info(f"  - Wind generation masked to: {fill_value}")

    elif strategy == 'keep':
        if verbose:
            logger.info(f"  - Keeping all records with 'is_curtailed' flag")

    return df


def clean_generation_data(
    df: pd.DataFrame,
    solar_col: str = 'solar_mw',
    wind_col: str = 'wind_mw',
    zenith_col: str = 'solar_zenith',
    wind_speed_col: str = 'wind_speed',
    datetime_col: str = 'datetime',
    mask_ess: bool = True,
    detect_curtailment: bool = True,
    curtailment_strategy: str = 'keep',
    curtailment_method: str = 'v2',
    curtailment_threshold: float = 0.3,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    통합 발전량 데이터 클렌징

    Args:
        df: 입력 DataFrame
        solar_col: 태양광 발전량 컬럼명
        wind_col: 풍력 발전량 컬럼명
        zenith_col: 태양 천정각 컬럼명
        wind_speed_col: 풍속 컬럼명
        datetime_col: datetime 컬럼명
        mask_ess: ESS 방전 마스킹 여부
        detect_curtailment: 출력 제어 탐지 여부
        curtailment_strategy: 출력 제어 처리 전략
        curtailment_method: 출력 제어 탐지 방법 ('v2': 디커플링, 'ratio': 기존)
        curtailment_threshold: 출력 제어 탐지 임계값
        verbose: 상세 출력 여부

    Returns:
        (cleaned_df, stats_dict) 튜플
    """
    stats = {
        'original_records': len(df),
        'ess_discharge_masked': 0,
        'curtailment_detected': 0,
        'final_records': 0,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("Data Cleaning Pipeline")
        logger.info("=" * 60)

    # Step 1: ESS 방전 마스킹
    if mask_ess and zenith_col in df.columns:
        original_solar = df[solar_col].sum() if solar_col in df.columns else 0

        df = mask_nighttime_solar(
            df,
            solar_col=solar_col,
            zenith_col=zenith_col,
            verbose=verbose
        )

        new_solar = df[solar_col].sum() if solar_col in df.columns else 0
        stats['ess_discharge_masked'] = int(original_solar - new_solar)

    # Step 2: 출력 제어 탐지
    curtailment_mask = pd.Series([False] * len(df), index=df.index)

    if detect_curtailment and wind_col in df.columns and wind_speed_col in df.columns:
        df, curtailment_mask = detect_wind_curtailment(
            df,
            wind_col=wind_col,
            wind_speed_col=wind_speed_col,
            method=curtailment_method,
            threshold=curtailment_threshold,
            verbose=verbose
        )

        stats['curtailment_detected'] = int(curtailment_mask.sum())

        # 출력 제어 처리
        df = filter_curtailment(
            df,
            curtailment_mask,
            strategy=curtailment_strategy,
            verbose=verbose
        )

    stats['final_records'] = len(df)

    if verbose:
        logger.info("=" * 60)
        logger.info("Cleaning Summary")
        logger.info("=" * 60)
        logger.info(f"  - Original: {stats['original_records']:,}")
        logger.info(f"  - ESS masked: {stats['ess_discharge_masked']:,} MWh")
        logger.info(f"  - Curtailment detected: {stats['curtailment_detected']:,}")
        logger.info(f"  - Final: {stats['final_records']:,}")

    return df, stats


def add_dynamic_wind_shear(
    df: pd.DataFrame,
    wind_speed_col: str = 'wind_speed',
    wind_dir_col: str = 'wind_direction',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    동적 Wind Shear 계수 적용 (Gemini 권장)

    풍향에 따라 Wind Shear 계수(α) 조정:
    - 바다 방향 (남쪽): α = 0.11 (평탄한 수면)
    - 산악 방향 (북쪽): α = 0.22 (거친 지형)

    제주도 기준:
    - 남풍 (135~225도): 해상 풍속 프로파일
    - 북풍 (315~45도): 한라산 영향

    Args:
        df: 입력 DataFrame
        wind_speed_col: 풍속 컬럼명
        wind_dir_col: 풍향 컬럼명
        verbose: 상세 출력 여부

    Returns:
        동적 Wind Shear가 적용된 DataFrame
    """
    df = df.copy()

    if wind_dir_col not in df.columns:
        logger.warning(f"Column '{wind_dir_col}' not found. Using static α = 0.14")
        return df

    if wind_speed_col not in df.columns:
        logger.warning(f"Column '{wind_speed_col}' not found.")
        return df

    wind_dir = df[wind_dir_col].fillna(0)
    wind_speed = df[wind_speed_col].fillna(0)

    # 풍향별 Wind Shear 계수
    # 남풍 (바다): 0.11, 북풍 (산악): 0.22, 기타: 0.14
    alpha = pd.Series(0.14, index=df.index)

    # 남풍 (135-225도): 바다
    south_mask = (wind_dir >= 135) & (wind_dir <= 225)
    alpha[south_mask] = 0.11

    # 북풍 (315-360 또는 0-45도): 산악
    north_mask = (wind_dir >= 315) | (wind_dir <= 45)
    alpha[north_mask] = 0.22

    df['wind_shear_alpha'] = alpha

    # 동적 허브 높이 풍속 계산
    h_ref = 10   # 기상청 관측 높이 (m)
    h_hub = 80   # 터빈 허브 높이 (m)

    df['wind_speed_hub_dynamic'] = wind_speed * ((h_hub / h_ref) ** alpha)

    # 동적 WPD
    if 'air_density' in df.columns:
        rho = df['air_density']
    else:
        rho = 1.225

    df['wpd_hub_dynamic'] = 0.5 * rho * (df['wind_speed_hub_dynamic'] ** 3)

    if verbose:
        logger.info("[Dynamic Wind Shear Applied]")
        logger.info(f"  - South wind (sea, α=0.11): {south_mask.sum():,} records")
        logger.info(f"  - North wind (mountain, α=0.22): {north_mask.sum():,} records")
        logger.info(f"  - Other (α=0.14): {(~south_mask & ~north_mask).sum():,} records")

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config import PROCESSED_DATA_DIR
    from src.features import FeatureEngineer, create_solar_features, create_wind_features

    print("=" * 60)
    print("Data Cleaning Test")
    print("=" * 60)

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"\nLoaded: {len(df)} rows")

    # Feature Engineering
    fe = FeatureEngineer(add_lag_features=False, add_rolling_features=False)
    df_features = fe.fit_transform(df, target_col='solar_mw')
    df_features = df_features.reset_index()

    # Solar/Wind 물리 특성 추가
    df_solar = create_solar_features(df_features)
    df_wind = create_wind_features(df_solar)

    print(f"Features: {len(df_wind.columns)} columns")

    # 데이터 클렌징
    cleaned_df, stats = clean_generation_data(
        df_wind,
        mask_ess=True,
        detect_curtailment=True,
        curtailment_strategy='keep',
        curtailment_method='ratio',
        curtailment_threshold=0.3,
        verbose=True
    )

    # 동적 Wind Shear 적용
    cleaned_df = add_dynamic_wind_shear(cleaned_df, verbose=True)

    # 저장
    output_path = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
