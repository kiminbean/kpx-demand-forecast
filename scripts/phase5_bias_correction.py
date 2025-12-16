"""
Phase 5: Look-ahead Bias 수정 및 공정한 재검증

문제 발견:
- 5년 기상 패턴(월-시간별 평균) 계산 시 2024년 전체(테스트셋 포함) 데이터 사용
- wind_speed_anomaly = 현재풍속 - 5년평균(2020-2024 전체)
- 테스트셋(11-12월)의 분포 정보가 이미 평균에 포함됨 → Look-ahead Bias

수정:
1. 2020-2023년 데이터만으로 기상 패턴 baseline 계산
2. 2024년 데이터는 이 baseline에 투영
3. 시간 가중치 제거 버전도 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.wind_features_v2 import create_wind_features_v2


def get_clean_features(df, target_col='wind_mw'):
    """누수 없는 클린 피처 선택"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_patterns = [
        'solar_mw', 'wind_mw', 'total_mw',
        'solar_cf', 'wind_cf',
        'output_ratio', 'expected_max',
        'is_curtailed', 'is_decoupling',
        '_gen_change', '_mw_lag', '_mw_ma', '_mw_std',
    ]
    feature_cols = []
    for col in numeric_cols:
        is_safe = True
        for pattern in exclude_patterns:
            if pattern in col:
                is_safe = False
                break
        if is_safe and col != target_col:
            feature_cols.append(col)
    return feature_cols


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def create_weather_patterns_no_leakage(df_weather_all: pd.DataFrame,
                                        cutoff_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Look-ahead Bias 없는 기상 패턴 생성

    Args:
        df_weather_all: 전체 기상 데이터
        cutoff_date: 이 날짜 이전 데이터만 사용하여 패턴 계산

    Returns:
        월-시간별 기상 패턴 통계 (2020-2023년만 사용)
    """
    df = df_weather_all.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # cutoff_date 이전 데이터만 사용 (2020-2023)
    df_past = df[df['datetime'] < cutoff_date].copy()

    df_past['hour'] = df_past['datetime'].dt.hour
    df_past['month'] = df_past['datetime'].dt.month

    stats_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                  'pressure', 'solar_radiation', 'cloud_cover']
    stats_cols = [c for c in stats_cols if c in df_past.columns]

    # 그룹별 통계 (2020-2023년만)
    patterns = df_past.groupby(['month', 'hour'])[stats_cols].agg(['mean', 'std', 'min', 'max'])
    patterns.columns = ['_'.join(col).strip() for col in patterns.columns.values]
    patterns = patterns.reset_index()

    return patterns


def add_historical_features_no_leakage(df: pd.DataFrame,
                                        weather_patterns: pd.DataFrame) -> pd.DataFrame:
    """
    Look-ahead Bias 없는 과거 기상 패턴 피처 추가

    Args:
        df: 2024년 데이터
        weather_patterns: 2020-2023년 기반 패턴 통계

    Returns:
        피처가 추가된 DataFrame
    """
    df = df.copy()

    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear

    # 병합 (2020-2023년 통계 사용)
    df = pd.merge(df, weather_patterns, on=['month', 'hour'], how='left', suffixes=('', '_hist'))

    # anomaly 피처 생성 (이제 공정함)
    if 'wind_speed' in df.columns and 'wind_speed_mean' in df.columns:
        df['wind_speed_anomaly'] = df['wind_speed'] - df['wind_speed_mean']

    if 'temperature' in df.columns and 'temperature_mean' in df.columns:
        df['temperature_anomaly'] = df['temperature'] - df['temperature_mean']

    if 'solar_radiation' in df.columns and 'solar_radiation_mean' in df.columns:
        df['solar_radiation_anomaly'] = df['solar_radiation'] - df['solar_radiation_mean']

    return df


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       weights=None, model_name="Model"):
    """모델 학습 및 평가"""

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.02,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'max_depth': 8,
        'verbose': -1,
        'seed': 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = calc_metrics(y_test, y_pred)

    print(f"  [{model_name}]")
    print(f"    iterations: {model.best_iteration}")
    print(f"    R²: {metrics['r2']:.4f}")
    print(f"    RMSE: {metrics['rmse']:.2f} MW")
    print(f"    MAE: {metrics['mae']:.2f} MW")

    return model, metrics, y_pred


def main():
    print("=" * 70)
    print("Phase 5: Look-ahead Bias 수정 및 공정한 재검증")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ================================================================
    # 1. 데이터 로드
    # ================================================================
    print("\n[1] 데이터 로드")

    # 5년치 기상 데이터
    weather_5year_path = PROCESSED_DATA_DIR / "weather_5year_hourly.csv"
    df_weather_all = pd.read_csv(weather_5year_path)
    df_weather_all['datetime'] = pd.to_datetime(df_weather_all['datetime'])
    print(f"5년 기상 데이터: {len(df_weather_all)} rows")
    print(f"기간: {df_weather_all['datetime'].min()} ~ {df_weather_all['datetime'].max()}")

    # 2024년 KPX + 기상 병합 데이터
    merged_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df_original = pd.read_csv(merged_path)
    df_original['datetime'] = pd.to_datetime(df_original['datetime'])
    print(f"병합 데이터: {len(df_original)} rows")

    # ================================================================
    # 2. Look-ahead Bias 수정
    # ================================================================
    print("\n[2] Look-ahead Bias 수정")
    print("  기존: 2020-2024 전체로 패턴 계산 (테스트셋 포함 → 누수)")
    print("  수정: 2020-2023년만으로 패턴 계산 (테스트셋 제외 → 공정)")

    # 2020-2023년 데이터만으로 기상 패턴 계산
    weather_patterns_fair = create_weather_patterns_no_leakage(
        df_weather_all,
        cutoff_date='2024-01-01'
    )
    print(f"  공정한 패턴 생성 완료: {len(weather_patterns_fair)} 조합")

    # 기존 anomaly 피처 제거
    drop_cols = [c for c in df_original.columns if 'anomaly' in c or
                 c.endswith('_mean') or c.endswith('_std') or
                 c.endswith('_min') or c.endswith('_max')]
    df_base = df_original.drop(columns=[c for c in drop_cols if c in df_original.columns])
    print(f"  기존 패턴 피처 제거: {len(drop_cols)}개")

    # 공정한 패턴 피처 추가
    df_fair = add_historical_features_no_leakage(df_base, weather_patterns_fair)
    print(f"  공정한 패턴 피처 추가 완료")

    # ================================================================
    # 3. V2 피처 엔지니어링
    # ================================================================
    print("\n[3] V2 Feature Engineering")
    df_fair = df_fair.set_index('datetime')
    df_v2 = create_wind_features_v2(df_fair)
    print(f"  피처 수: {len(df_v2.columns)}")

    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"  클린 피처 수: {len(feature_cols)}")

    # ================================================================
    # 4. 데이터 분할
    # ================================================================
    print("\n[4] 데이터 분할")
    n = len(df_v2)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_v2.iloc[:train_end]
    val_df = df_v2.iloc[train_end:val_end]
    test_df = df_v2.iloc[val_end:]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df['wind_mw'].values
    y_val = val_df['wind_mw'].values
    y_test = test_df['wind_mw'].values

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 시간 가중치
    T = len(y_train)
    weights_strong = np.array([0.999 ** (T - t - 1) for t in range(T)])
    weights_strong = weights_strong / weights_strong.sum() * T

    weights_mild = np.array([0.9999 ** (T - t - 1) for t in range(T)])
    weights_mild = weights_mild / weights_mild.sum() * T

    print(f"  강한 가중치 비율 (첫/끝): {weights_strong[-1]/weights_strong[0]:.1f}x")
    print(f"  약한 가중치 비율 (첫/끝): {weights_mild[-1]/weights_mild[0]:.1f}x")

    # ================================================================
    # 5. 모델 학습 및 비교
    # ================================================================
    print("\n" + "=" * 70)
    print("모델 학습 및 비교")
    print("=" * 70)

    results = {}

    # 5.1 기존 모델 (누수 + 강한 시간 가중치) - 참조용
    print("\n[참조] 기존 성능 (누수 포함)")
    print("  R²: 0.7441, RMSE: 15.37 MW (이전 측정값)")
    results['기존(누수+강한가중치)'] = {'r2': 0.7441, 'rmse': 15.37, 'mae': 12.00}

    # 5.2 누수 수정 + 가중치 없음
    print("\n[실험 1] 누수 수정 + 가중치 없음")
    model1, metrics1, pred1 = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights=None, model_name="누수수정_가중치없음"
    )
    results['누수수정_가중치없음'] = metrics1

    # 5.3 누수 수정 + 약한 가중치 (0.9999)
    print("\n[실험 2] 누수 수정 + 약한 가중치 (decay=0.9999)")
    model2, metrics2, pred2 = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights=weights_mild, model_name="누수수정_약한가중치"
    )
    results['누수수정_약한가중치'] = metrics2

    # 5.4 누수 수정 + 강한 가중치 (0.999)
    print("\n[실험 3] 누수 수정 + 강한 가중치 (decay=0.999)")
    model3, metrics3, pred3 = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights=weights_strong, model_name="누수수정_강한가중치"
    )
    results['누수수정_강한가중치'] = metrics3

    # ================================================================
    # 6. Feature Importance 비교
    # ================================================================
    print("\n" + "=" * 70)
    print("Feature Importance 분석 (누수 수정 후)")
    print("=" * 70)

    importance = model3.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n[Top 15 Features - 누수 수정 후]")
    print(fi_df.head(15).to_string(index=False))

    # anomaly 피처의 순위 변화 확인
    anomaly_features = fi_df[fi_df['feature'].str.contains('anomaly')]
    print("\n[Anomaly 피처 중요도 - 누수 수정 후]")
    if len(anomaly_features) > 0:
        for _, row in anomaly_features.iterrows():
            rank = fi_df[fi_df['feature'] == row['feature']].index[0] + 1
            print(f"  {row['feature']}: 순위 {rank}, 중요도 {row['importance']:.0f}")
    else:
        print("  anomaly 피처 없음")

    # ================================================================
    # 7. 최종 결과 요약
    # ================================================================
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    print(f"\n{'모델':<30} {'R²':>10} {'RMSE':>12} {'MAE':>10}")
    print("-" * 65)

    for name, m in results.items():
        print(f"{name:<30} {m['r2']:>10.4f} {m['rmse']:>10.2f} MW {m['mae']:>8.2f} MW")

    # 성능 변화 분석
    print("\n[성능 변화 분석]")
    baseline_r2 = 0.7441

    for name, m in results.items():
        if name != '기존(누수+강한가중치)':
            diff = m['r2'] - baseline_r2
            print(f"  {name}: R² {diff:+.4f} (기존 대비)")

    # 누수 기여도 추정
    leak_contribution = baseline_r2 - metrics3['r2']
    print(f"\n[Look-ahead Bias 기여도 추정]")
    print(f"  누수로 인한 R² 과대평가: 약 {leak_contribution:.4f} ({leak_contribution/baseline_r2*100:.1f}%)")

    # 공정한 최고 성능
    best_fair = max([(k, v) for k, v in results.items() if k != '기존(누수+강한가중치)'],
                    key=lambda x: x[1]['r2'])

    print(f"\n[공정한 최고 성능]")
    print(f"  모델: {best_fair[0]}")
    print(f"  R²: {best_fair[1]['r2']:.4f}")
    print(f"  RMSE: {best_fair[1]['rmse']:.2f} MW")

    # Phase별 공정한 성능 비교
    print(f"\n[Phase별 공정한 성능 비교]")
    print(f"{'Phase':<40} {'R²':>10} {'RMSE':>12}")
    print("-" * 65)
    print(f"{'Phase 0 (Baseline - 원본)':<40} {'0.2530':>10} {'26.28 MW':>12}")
    print(f"{'Phase 1 (V2 Physics Features)':<40} {'0.5153':>10} {'21.16 MW':>12}")
    print(f"{'Phase 2 (5년 패턴 - 누수 포함)':<40} {'0.6357':>10} {'18.34 MW':>12}")
    print(f"{'Phase 3 (시간 가중치 - 누수 포함)':<40} {'0.7441':>10} {'15.37 MW':>12}")
    print(f"{'Phase 5 (누수 수정 - 공정)':<40} {best_fair[1]['r2']:>10.4f} {best_fair[1]['rmse']:>7.2f} MW")

    # 실제 개선율
    fair_improvement = (best_fair[1]['r2'] - 0.2530) / 0.2530 * 100
    print(f"\n[실제 개선율 (공정 평가)]")
    print(f"  Phase 0 → Phase 5: R² 0.2530 → {best_fair[1]['r2']:.4f}")
    print(f"  개선율: +{fair_improvement:.1f}%")

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_phase5_fair'
    save_dir.mkdir(parents=True, exist_ok=True)

    model3.save_model(str(save_dir / 'wind_lgbm_fair.lgb'))
    fi_df.to_csv(save_dir / 'feature_importance_fair.csv', index=False)

    print(f"\n모델 저장: {save_dir}")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = main()
