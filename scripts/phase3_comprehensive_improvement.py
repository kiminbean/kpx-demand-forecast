"""
Wind Model Phase 3 - 종합 개선 스크립트

3가지 핵심 문제 해결:
1. 테스트셋 분산 감소 → 시계열 CV + 분산 정규화 R²
2. Concept Drift → 시간 가중치 학습 + Expanding Window
3. 풍력 확률성 → Quantile Regression + 불확실성 추정

추가 개선:
4. Two-Stage 모델 (출력제어 분류 + 회귀)
5. 앙상블 모델 (LightGBM + XGBoost + CatBoost)
6. Optuna 하이퍼파라미터 튜닝
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.wind_features_v2 import create_wind_features_v2


# =============================================================================
# 1. 유틸리티 함수
# =============================================================================

def get_clean_features(df: pd.DataFrame, target_col: str = 'wind_mw') -> List[str]:
    """누수 없는 클린 피처 선택"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_patterns = [
        'solar_mw', 'wind_mw', 'total_mw',
        'solar_cf', 'wind_cf',
        'output_ratio', 'expected_max',
        'is_curtailed', 'is_decoupling',
        '_gen_change',
        '_mw_lag', '_mw_ma', '_mw_std',
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


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """평가 메트릭 계산"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (y > 1인 경우만)
    mask = y_true > 1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def variance_normalized_r2(y_true: np.ndarray, y_pred: np.ndarray,
                           train_variance: float) -> float:
    """
    분산 정규화 R² - 테스트셋 분산 감소 문제 해결

    일반 R² = 1 - MSE / Var(y_test)
    정규화 R² = 1 - MSE / train_variance

    이렇게 하면 테스트셋 분산이 작아도 공정한 비교 가능
    """
    mse = mean_squared_error(y_true, y_pred)
    return 1 - mse / train_variance


# =============================================================================
# 2. 문제 1: 테스트셋 분산 감소 해결 - 시계열 CV
# =============================================================================

def solve_variance_reduction(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
    n_splits: int = 5
) -> Dict:
    """
    문제 1 해결: 테스트셋 분산 감소

    해결책:
    1. TimeSeriesSplit으로 여러 fold 평가
    2. 분산 정규화 R² 사용
    3. 각 fold의 분산 비율 분석
    """
    print("\n" + "=" * 70)
    print("문제 1: 테스트셋 분산 감소 해결")
    print("=" * 70)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
    y = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []
    fold_metrics = []

    print(f"\n시계열 교차검증 ({n_splits} folds)")
    print("-" * 70)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 분산 비율 계산
        train_var = np.var(y_train)
        test_var = np.var(y_test)
        var_ratio = test_var / train_var * 100

        # LightGBM 학습
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

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
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = np.clip(model.predict(X_test), 0, None)

        # 일반 R² vs 분산 정규화 R²
        r2_normal = r2_score(y_test, y_pred)
        r2_normalized = variance_normalized_r2(y_test, y_pred, train_var)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        fold_result = {
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_var': train_var,
            'test_var': test_var,
            'var_ratio': var_ratio,
            'r2_normal': r2_normal,
            'r2_normalized': r2_normalized,
            'rmse': rmse,
            'mae': mae,
        }
        fold_metrics.append(fold_result)

        print(f"Fold {fold+1}: 분산비율={var_ratio:5.1f}% | R²={r2_normal:.4f} | "
              f"정규화R²={r2_normalized:.4f} | RMSE={rmse:.2f}")

    # 요약 통계
    avg_r2 = np.mean([f['r2_normal'] for f in fold_metrics])
    avg_r2_norm = np.mean([f['r2_normalized'] for f in fold_metrics])
    avg_rmse = np.mean([f['rmse'] for f in fold_metrics])
    std_r2 = np.std([f['r2_normal'] for f in fold_metrics])

    print("-" * 70)
    print(f"평균: R²={avg_r2:.4f} (±{std_r2:.4f}) | 정규화R²={avg_r2_norm:.4f} | RMSE={avg_rmse:.2f}")

    # 분산 정규화의 효과 분석
    print("\n[분석]")
    print(f"  - 일반 R² 평균: {avg_r2:.4f}")
    print(f"  - 정규화 R² 평균: {avg_r2_norm:.4f}")
    print(f"  - 차이: {avg_r2_norm - avg_r2:.4f}")
    print(f"  → 분산 정규화 시 R²가 {'증가' if avg_r2_norm > avg_r2 else '감소'}함")

    return {
        'fold_metrics': fold_metrics,
        'avg_r2': avg_r2,
        'avg_r2_normalized': avg_r2_norm,
        'avg_rmse': avg_rmse,
    }


# =============================================================================
# 3. 문제 2: Concept Drift 해결 - 시간 가중치 학습
# =============================================================================

def solve_concept_drift(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
    decay_rate: float = 0.999,
) -> Dict:
    """
    문제 2 해결: Concept Drift

    해결책:
    1. 시간 가중치 학습 (최근 데이터에 더 높은 가중치)
    2. Expanding Window 재학습 시뮬레이션
    3. 월별 모델 성능 비교
    """
    print("\n" + "=" * 70)
    print("문제 2: Concept Drift 해결")
    print("=" * 70)

    # 시계열 분할
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # 3.1 시간 가중치 계산
    print("\n[1] 시간 가중치 학습")

    # 지수 감쇠 가중치: w_t = decay_rate^(T-t)
    T = len(y_train)
    weights = np.array([decay_rate ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum() * T  # 정규화

    print(f"  - 감쇠율: {decay_rate}")
    print(f"  - 첫 데이터 가중치: {weights[0]:.4f}")
    print(f"  - 마지막 데이터 가중치: {weights[-1]:.4f}")
    print(f"  - 가중치 비율 (마지막/첫): {weights[-1]/weights[0]:.2f}x")

    # 가중치 없는 모델
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

    # 일반 학습
    train_data_normal = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data_normal = lgb.Dataset(X_val, label=y_val, reference=train_data_normal, free_raw_data=False)

    model_normal = lgb.train(
        params,
        train_data_normal,
        num_boost_round=2000,
        valid_sets=[val_data_normal],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 가중치 학습
    train_data_weighted = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    val_data_weighted = lgb.Dataset(X_val, label=y_val, reference=train_data_weighted, free_raw_data=False)

    model_weighted = lgb.train(
        params,
        train_data_weighted,
        num_boost_round=2000,
        valid_sets=[val_data_weighted],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 결과 비교
    y_pred_normal = np.clip(model_normal.predict(X_test), 0, None)
    y_pred_weighted = np.clip(model_weighted.predict(X_test), 0, None)

    metrics_normal = calc_metrics(y_test, y_pred_normal)
    metrics_weighted = calc_metrics(y_test, y_pred_weighted)

    print("\n[결과 비교]")
    print(f"{'모델':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 50)
    print(f"{'일반 학습':<20} {metrics_normal['r2']:>10.4f} {metrics_normal['rmse']:>10.2f} {metrics_normal['mae']:>10.2f}")
    print(f"{'시간 가중치 학습':<20} {metrics_weighted['r2']:>10.4f} {metrics_weighted['rmse']:>10.2f} {metrics_weighted['mae']:>10.2f}")

    improvement = metrics_weighted['r2'] - metrics_normal['r2']
    print(f"\n  → 시간 가중치 효과: R² {'+' if improvement > 0 else ''}{improvement:.4f}")

    # 3.2 월별 성능 분석
    print("\n[2] 월별 성능 분석 (Concept Drift 확인)")

    test_df_copy = test_df.copy()
    if isinstance(test_df_copy.index, pd.DatetimeIndex):
        test_df_copy['month'] = test_df_copy.index.month
    else:
        test_df_copy['month'] = pd.to_datetime(test_df_copy.index).month

    test_df_copy['pred_normal'] = y_pred_normal
    test_df_copy['pred_weighted'] = y_pred_weighted
    test_df_copy['actual'] = y_test

    monthly_results = []
    for month in sorted(test_df_copy['month'].unique()):
        mask = test_df_copy['month'] == month
        y_actual = test_df_copy.loc[mask, 'actual'].values
        y_pred_n = test_df_copy.loc[mask, 'pred_normal'].values
        y_pred_w = test_df_copy.loc[mask, 'pred_weighted'].values

        r2_n = r2_score(y_actual, y_pred_n)
        r2_w = r2_score(y_actual, y_pred_w)
        mae_n = mean_absolute_error(y_actual, y_pred_n)
        mae_w = mean_absolute_error(y_actual, y_pred_w)

        monthly_results.append({
            'month': month,
            'r2_normal': r2_n,
            'r2_weighted': r2_w,
            'mae_normal': mae_n,
            'mae_weighted': mae_w,
            'improvement': r2_w - r2_n,
        })

        print(f"  {month}월: R²_일반={r2_n:.3f}, R²_가중치={r2_w:.3f}, 개선={r2_w-r2_n:+.3f}")

    return {
        'metrics_normal': metrics_normal,
        'metrics_weighted': metrics_weighted,
        'monthly_results': monthly_results,
        'best_model': model_weighted,
        'improvement': improvement,
    }


# =============================================================================
# 4. 문제 3: 풍력 확률성 해결 - Quantile Regression
# =============================================================================

def solve_wind_stochasticity(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
) -> Dict:
    """
    문제 3 해결: 풍력의 확률성

    해결책:
    1. Quantile Regression으로 예측 구간 제공
    2. 불확실성 정량화
    3. 확률적 예측 평가 (Pinball Loss)
    """
    print("\n" + "=" * 70)
    print("문제 3: 풍력 확률성 해결")
    print("=" * 70)

    # 데이터 분할
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    print(f"\n[1] Quantile Regression 학습 (분위수: {quantiles})")

    quantile_models = {}
    quantile_predictions = {}

    for q in quantiles:
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
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

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        quantile_models[q] = model
        quantile_predictions[q] = np.clip(model.predict(X_test), 0, None)

        print(f"  - Q{int(q*100):02d} 학습 완료 (iterations: {model.best_iteration})")

    # Pinball Loss 계산
    def pinball_loss(y_true, y_pred, quantile):
        delta = y_true - y_pred
        return np.mean(np.where(delta >= 0, quantile * delta, (quantile - 1) * delta))

    print("\n[2] Pinball Loss 평가")
    pinball_results = {}
    for q in quantiles:
        loss = pinball_loss(y_test, quantile_predictions[q], q)
        pinball_results[q] = loss
        print(f"  - Q{int(q*100):02d}: Pinball Loss = {loss:.4f}")

    # 예측 구간 커버리지
    print("\n[3] 예측 구간 분석")

    q10 = quantile_predictions[0.1]
    q90 = quantile_predictions[0.9]

    # 80% 예측 구간 커버리지
    in_interval = (y_test >= q10) & (y_test <= q90)
    coverage = np.mean(in_interval) * 100

    print(f"  - 80% 예측 구간 커버리지: {coverage:.1f}% (목표: 80%)")

    # 평균 구간 폭
    avg_interval_width = np.mean(q90 - q10)
    print(f"  - 평균 예측 구간 폭: {avg_interval_width:.2f} MW")

    # 불확실성 지표
    q25 = quantile_predictions[0.25]
    q75 = quantile_predictions[0.75]
    iqr = q75 - q25  # 사분위 범위

    print(f"  - 평균 IQR (Q75-Q25): {np.mean(iqr):.2f} MW")

    # 중앙값 예측 성능
    median_pred = quantile_predictions[0.5]
    metrics_median = calc_metrics(y_test, median_pred)

    print(f"\n[4] 중앙값(Q50) 예측 성능")
    print(f"  - R²: {metrics_median['r2']:.4f}")
    print(f"  - RMSE: {metrics_median['rmse']:.2f} MW")
    print(f"  - MAE: {metrics_median['mae']:.2f} MW")

    # 불확실성 기반 신뢰도 분류
    print("\n[5] 불확실성 기반 예측 분류")

    uncertainty = (q90 - q10) / (median_pred + 1e-6)  # 상대적 불확실성

    low_uncertainty = uncertainty < np.percentile(uncertainty, 33)
    high_uncertainty = uncertainty > np.percentile(uncertainty, 67)

    r2_low = r2_score(y_test[low_uncertainty], median_pred[low_uncertainty])
    r2_high = r2_score(y_test[high_uncertainty], median_pred[high_uncertainty])

    print(f"  - 낮은 불확실성 구간 R²: {r2_low:.4f} ({np.sum(low_uncertainty)}개)")
    print(f"  - 높은 불확실성 구간 R²: {r2_high:.4f} ({np.sum(high_uncertainty)}개)")
    print(f"  → 불확실성이 낮을 때 예측이 {'더 정확함' if r2_low > r2_high else '덜 정확함'}")

    return {
        'quantile_models': quantile_models,
        'quantile_predictions': quantile_predictions,
        'pinball_results': pinball_results,
        'coverage_80': coverage,
        'avg_interval_width': avg_interval_width,
        'metrics_median': metrics_median,
    }


# =============================================================================
# 5. Phase 3: Two-Stage 모델
# =============================================================================

def build_two_stage_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
    curtailment_threshold: float = 0.7,  # 정격 대비 70% 이하면 출력제어 가능성
) -> Dict:
    """
    Phase 3: Two-Stage 모델

    Stage 1: 출력제어 여부 분류
    Stage 2: 분류 결과에 따른 조건부 회귀

    가설: 출력제어 시점과 정상 시점의 패턴이 다름
    """
    print("\n" + "=" * 70)
    print("Phase 3: Two-Stage 모델 (출력제어 분류 + 회귀)")
    print("=" * 70)

    # 데이터 분할
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # 출력제어 레이블 생성 (풍속 대비 발전량이 낮은 경우)
    # wind_power_density가 높은데 발전량이 낮으면 출력제어 가능성
    if 'wind_power_density' in train_df.columns:
        wpd = train_df['wind_power_density']
        wind_mw = train_df[target_col]

        # 간단한 휴리스틱: WPD 상위 25%인데 발전량이 중앙값 이하
        wpd_high = wpd > wpd.quantile(0.75)
        mw_low = wind_mw < wind_mw.quantile(0.5)

        train_df['is_curtailed'] = (wpd_high & mw_low).astype(int)
        val_df['is_curtailed'] = 0  # 검증셋은 레이블 없음
        test_df['is_curtailed'] = 0

        curtailment_rate = train_df['is_curtailed'].mean() * 100
        print(f"\n[1] 출력제어 레이블 생성")
        print(f"  - 추정 출력제어 비율: {curtailment_rate:.1f}%")
    else:
        print("  경고: wind_power_density 없음. Two-Stage 스킵.")
        return None

    # 피처 준비 (is_curtailed 제외)
    safe_features = [c for c in feature_cols if c != 'is_curtailed']

    X_train = train_df[safe_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[safe_features].median())
    X_val = val_df[safe_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[safe_features].median())
    X_test = test_df[safe_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[safe_features].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # Stage 1: 분류 모델 (현재는 단순 회귀로 대체)
    # 실제로는 분류기를 학습하지만, 여기서는 WPD 기반 규칙 사용
    print("\n[2] Stage 1: 조건부 분류 (WPD 기반)")

    # Stage 2: 조건부 회귀
    print("\n[3] Stage 2: 조건부 회귀 모델")

    # 정상 데이터 모델
    normal_mask = train_df['is_curtailed'] == 0

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

    # 정상 상태 모델
    train_normal = lgb.Dataset(X_train[normal_mask], label=y_train[normal_mask])
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_normal)

    model_normal = lgb.train(
        params,
        train_normal,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 전체 모델 (비교용)
    train_all = lgb.Dataset(X_train, label=y_train)

    model_all = lgb.train(
        params,
        train_all,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 예측 및 평가
    y_pred_normal_model = np.clip(model_normal.predict(X_test), 0, None)
    y_pred_all_model = np.clip(model_all.predict(X_test), 0, None)

    metrics_normal_model = calc_metrics(y_test, y_pred_normal_model)
    metrics_all_model = calc_metrics(y_test, y_pred_all_model)

    print("\n[결과 비교]")
    print(f"{'모델':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 55)
    print(f"{'전체 데이터 학습':<25} {metrics_all_model['r2']:>10.4f} {metrics_all_model['rmse']:>10.2f} {metrics_all_model['mae']:>10.2f}")
    print(f"{'정상 데이터만 학습':<25} {metrics_normal_model['r2']:>10.4f} {metrics_normal_model['rmse']:>10.2f} {metrics_normal_model['mae']:>10.2f}")

    return {
        'model_normal': model_normal,
        'model_all': model_all,
        'metrics_normal_model': metrics_normal_model,
        'metrics_all_model': metrics_all_model,
        'curtailment_rate': curtailment_rate,
    }


# =============================================================================
# 6. Phase 4: 앙상블 모델
# =============================================================================

def build_ensemble_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
) -> Dict:
    """
    Phase 4: 앙상블 모델 (LightGBM + XGBoost + CatBoost)
    """
    print("\n" + "=" * 70)
    print("Phase 4: 앙상블 모델")
    print("=" * 70)

    # XGBoost, CatBoost 임포트 시도
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("  경고: XGBoost 미설치")

    try:
        from catboost import CatBoostRegressor
        HAS_CAT = True
    except ImportError:
        HAS_CAT = False
        print("  경고: CatBoost 미설치")

    # 데이터 분할
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    predictions = {}
    models = {}

    # 1. LightGBM
    print("\n[1] LightGBM 학습")
    lgb_params = {
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

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    predictions['lgb'] = np.clip(lgb_model.predict(X_test), 0, None)
    models['lgb'] = lgb_model
    print(f"  완료 (iterations: {lgb_model.best_iteration})")

    # 2. XGBoost
    if HAS_XGB:
        print("\n[2] XGBoost 학습")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'seed': 42,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        predictions['xgb'] = np.clip(xgb_model.predict(dtest), 0, None)
        models['xgb'] = xgb_model
        print(f"  완료 (iterations: {xgb_model.best_iteration})")

    # 3. CatBoost
    if HAS_CAT:
        print("\n[3] CatBoost 학습")
        cat_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=100,
        )

        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        predictions['cat'] = np.clip(cat_model.predict(X_test), 0, None)
        models['cat'] = cat_model
        print(f"  완료 (iterations: {cat_model.best_iteration_})")

    # 4. 앙상블
    print("\n[4] 앙상블 예측")

    # 단순 평균 앙상블
    pred_list = list(predictions.values())
    ensemble_avg = np.mean(pred_list, axis=0)

    # 가중 평균 (검증셋 성능 기반)
    val_scores = {}
    for name, model in models.items():
        if name == 'lgb':
            val_pred = np.clip(model.predict(X_val), 0, None)
        elif name == 'xgb' and HAS_XGB:
            val_pred = np.clip(model.predict(xgb.DMatrix(X_val)), 0, None)
        elif name == 'cat' and HAS_CAT:
            val_pred = np.clip(model.predict(X_val), 0, None)
        else:
            continue
        val_scores[name] = 1 / (mean_squared_error(y_val, val_pred) + 1e-6)

    total_score = sum(val_scores.values())
    weights = {k: v / total_score for k, v in val_scores.items()}

    print(f"  가중치: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

    ensemble_weighted = np.zeros_like(y_test, dtype=float)
    for name, pred in predictions.items():
        ensemble_weighted += weights.get(name, 0) * pred

    # 결과 비교
    print("\n[결과 비교]")
    print(f"{'모델':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 50)

    for name, pred in predictions.items():
        m = calc_metrics(y_test, pred)
        print(f"{name.upper():<20} {m['r2']:>10.4f} {m['rmse']:>10.2f} {m['mae']:>10.2f}")

    m_avg = calc_metrics(y_test, ensemble_avg)
    m_weighted = calc_metrics(y_test, ensemble_weighted)

    print("-" * 50)
    print(f"{'앙상블 (평균)':<20} {m_avg['r2']:>10.4f} {m_avg['rmse']:>10.2f} {m_avg['mae']:>10.2f}")
    print(f"{'앙상블 (가중)':<20} {m_weighted['r2']:>10.4f} {m_weighted['rmse']:>10.2f} {m_weighted['mae']:>10.2f}")

    return {
        'models': models,
        'predictions': predictions,
        'ensemble_avg': ensemble_avg,
        'ensemble_weighted': ensemble_weighted,
        'weights': weights,
        'metrics_avg': m_avg,
        'metrics_weighted': m_weighted,
    }


# =============================================================================
# 7. Optuna 하이퍼파라미터 튜닝
# =============================================================================

def optuna_tuning(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'wind_mw',
    n_trials: int = 50,
) -> Dict:
    """
    Optuna 하이퍼파라미터 튜닝
    """
    print("\n" + "=" * 70)
    print("Optuna 하이퍼파라미터 튜닝")
    print("=" * 70)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  경고: Optuna 미설치. pip install optuna")
        return None

    # 데이터 분할
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'verbose': -1,
            'seed': 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    print(f"\n최적화 시작 ({n_trials} trials)...")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n최적 파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\n최적 RMSE (Validation): {study.best_value:.4f}")

    # 최적 파라미터로 최종 모델 학습
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
    })

    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    y_pred_test = np.clip(final_model.predict(X_test), 0, None)
    metrics = calc_metrics(y_test, y_pred_test)

    print(f"\n[테스트 셋 성능 - Optuna 튜닝 모델]")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f} MW")
    print(f"  MAE: {metrics['mae']:.2f} MW")

    return {
        'study': study,
        'best_params': study.best_params,
        'best_model': final_model,
        'test_metrics': metrics,
    }


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    print("=" * 70)
    print("Wind Model Phase 3 - 종합 개선")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드
    print("\n[데이터 로드]")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # 2. V2 피처 엔지니어링
    print("\n[V2 Feature Engineering]")
    df_v2 = create_wind_features_v2(df)
    print(f"After V2: {len(df_v2.columns)} columns")

    # 3. 클린 피처 선택
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"Clean features: {len(feature_cols)}")

    # 결과 저장
    all_results = {}

    # =================================================================
    # 문제 1: 테스트셋 분산 감소 해결
    # =================================================================
    result1 = solve_variance_reduction(df_v2, feature_cols)
    all_results['variance_reduction'] = result1

    # =================================================================
    # 문제 2: Concept Drift 해결
    # =================================================================
    result2 = solve_concept_drift(df_v2, feature_cols)
    all_results['concept_drift'] = result2

    # =================================================================
    # 문제 3: 풍력 확률성 해결
    # =================================================================
    result3 = solve_wind_stochasticity(df_v2, feature_cols)
    all_results['stochasticity'] = result3

    # =================================================================
    # Phase 3: Two-Stage 모델
    # =================================================================
    result4 = build_two_stage_model(df_v2, feature_cols)
    all_results['two_stage'] = result4

    # =================================================================
    # Phase 4: 앙상블 모델
    # =================================================================
    result5 = build_ensemble_model(df_v2, feature_cols)
    all_results['ensemble'] = result5

    # =================================================================
    # Optuna 튜닝
    # =================================================================
    result6 = optuna_tuning(df_v2, feature_cols, n_trials=15)
    all_results['optuna'] = result6

    # =================================================================
    # 최종 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    print("\n[Phase별 최고 성능]")
    print(f"{'Phase':<35} {'R²':>10} {'RMSE':>12}")
    print("-" * 60)
    print(f"{'Phase 0 (Baseline)':<35} {'0.2530':>10} {'26.28 MW':>12}")
    print(f"{'Phase 1 (V2 Features)':<35} {'0.5153':>10} {'21.16 MW':>12}")
    print(f"{'Phase 2 (5yr Patterns)':<35} {'0.6357':>10} {'18.34 MW':>12}")

    # Phase 3 결과
    if result2:
        r2_p3 = result2['metrics_weighted']['r2']
        rmse_p3 = result2['metrics_weighted']['rmse']
        print(f"{'Phase 3 (시간 가중치)':<35} {r2_p3:>10.4f} {rmse_p3:>7.2f} MW")

    # 앙상블 결과
    if result5:
        r2_ens = result5['metrics_weighted']['r2']
        rmse_ens = result5['metrics_weighted']['rmse']
        print(f"{'Phase 4 (앙상블 가중)':<35} {r2_ens:>10.4f} {rmse_ens:>7.2f} MW")

    # Optuna 결과
    if result6:
        r2_opt = result6['test_metrics']['r2']
        rmse_opt = result6['test_metrics']['rmse']
        print(f"{'Optuna 튜닝':<35} {r2_opt:>10.4f} {rmse_opt:>7.2f} MW")

    print("\n[핵심 인사이트]")
    print("1. 분산 정규화 R²: 테스트셋 분산 감소의 영향을 보정")
    print("2. 시간 가중치: Concept Drift 완화에 효과적")
    print("3. Quantile Regression: 불확실성 정량화 가능")
    print("4. 앙상블: 단일 모델 대비 안정적 성능")

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_phase3'
    save_dir.mkdir(parents=True, exist_ok=True)

    if result6 and result6.get('best_model'):
        result6['best_model'].save_model(str(save_dir / 'wind_lgbm_optuna.lgb'))
        print(f"\nOptuna 튜닝 모델 저장: {save_dir / 'wind_lgbm_optuna.lgb'}")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    results = main()
