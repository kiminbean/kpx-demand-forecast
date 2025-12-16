#!/usr/bin/env python3
"""
Data Leakage Detection Script

Gemini 권장: 타겟과 상관관계가 0.9 이상인 변수 탐지
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR
from src.features import FeatureEngineer, create_solar_features, create_wind_features


def detect_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.9):
    """
    타겟 변수와 높은 상관관계를 가진 누수 의심 변수 탐지

    Args:
        df: 데이터프레임
        target_col: 타겟 컬럼명
        threshold: 상관관계 임계값

    Returns:
        누수 의심 변수 리스트
    """
    # 수치형 컬럼만
    numeric_df = df.select_dtypes(include=[np.number])

    if target_col not in numeric_df.columns:
        print(f"Warning: {target_col} not in numeric columns")
        return []

    # 상관관계 계산
    corr_with_target = numeric_df.corr()[target_col].abs()

    # 자기 자신 제외하고 threshold 이상인 변수
    leakage_candidates = corr_with_target[
        (corr_with_target > threshold) & (corr_with_target.index != target_col)
    ].sort_values(ascending=False)

    return leakage_candidates


def main():
    print("=" * 60)
    print("Data Leakage Detection")
    print("=" * 60)

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\nLoaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Feature Engineering
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df, target_col='solar_mw')
    df_solar = create_solar_features(df_features)
    df_wind = create_wind_features(df_solar)

    print(f"\nAfter feature engineering: {len(df_wind.columns)} columns")

    # 태양광 누수 탐지
    print("\n" + "=" * 60)
    print("Solar (solar_mw) Leakage Detection")
    print("=" * 60)

    solar_leakage = detect_leakage(df_wind, 'solar_mw', threshold=0.9)

    if len(solar_leakage) > 0:
        print(f"\n⚠️  Leakage candidates (corr > 0.9):")
        for col, corr in solar_leakage.items():
            print(f"   - {col}: {corr:.4f}")
    else:
        print("\n✓ No obvious leakage detected")

    # 풍력 누수 탐지
    print("\n" + "=" * 60)
    print("Wind (wind_mw) Leakage Detection")
    print("=" * 60)

    wind_leakage = detect_leakage(df_wind, 'wind_mw', threshold=0.9)

    if len(wind_leakage) > 0:
        print(f"\n⚠️  Leakage candidates (corr > 0.9):")
        for col, corr in wind_leakage.items():
            print(f"   - {col}: {corr:.4f}")
    else:
        print("\n✓ No obvious leakage detected")

    # 제거해야 할 변수 목록
    print("\n" + "=" * 60)
    print("Variables to Remove")
    print("=" * 60)

    # 명백한 누수 변수
    obvious_leakage = [
        'solar_cf', 'wind_cf', 'total_mw',  # 타겟에서 직접 파생
        'solar_mw', 'wind_mw',  # 타겟 자체 (cross-prediction 시)
    ]

    # 롤링/지연 특성 중 타겟 기반
    target_derived = [col for col in df_wind.columns if
                      col.startswith('solar_mw_') or col.startswith('wind_mw_')]

    all_leakage = list(set(obvious_leakage + target_derived))

    print(f"\n🚫 Obvious leakage variables:")
    for var in obvious_leakage:
        if var in df_wind.columns:
            print(f"   - {var}")

    print(f"\n🚫 Target-derived variables:")
    for var in target_derived:
        print(f"   - {var}")

    # 안전한 특성 목록
    safe_features = [col for col in df_wind.columns
                     if col not in all_leakage and col != 'datetime']

    print(f"\n✓ Safe features ({len(safe_features)}):")
    for var in safe_features[:20]:
        print(f"   - {var}")
    if len(safe_features) > 20:
        print(f"   ... and {len(safe_features) - 20} more")

    return all_leakage, safe_features


if __name__ == "__main__":
    leakage_vars, safe_vars = main()
