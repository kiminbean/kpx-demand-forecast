"""
Wind Model Training Phase 2 - 5년치 기상 패턴 활용

Phase 2 개선사항:
1. 5년치 기상 패턴 피처 (월-시간 조합별 통계)
2. 현재 값과 과거 평균의 편차 (anomaly features)
3. V2 피처 엔지니어링 적용

기대 효과:
- 과거 기상 패턴 대비 이상치 감지
- 계절적 패턴의 더 정확한 학습
- Concept Drift 완화
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
from sklearn.linear_model import Ridge

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.wind_features_v2 import create_wind_features_v2


def get_clean_features(df: pd.DataFrame, target_col: str = 'wind_mw'):
    """
    누수 없는 클린 피처 선택
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 완전 제외 패턴
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


def train_val_test_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """시계열 순서 유지 데이터 분할"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def calc_metrics(y_true, y_pred):
    """메트릭 계산"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true > 1
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def train_lgbm_model(X_train, y_train, X_val, y_val, feature_cols):
    """LightGBM 모델 학습"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

    callbacks = [
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(200),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
    )

    return model


def main():
    print("=" * 70)
    print("Wind Model Phase 2 - 5년치 기상 패턴 활용")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 5년치 병합 데이터 로드
    print("\n[1] Loading 5-year merged dataset...")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")

    # 기존 과거 기상 패턴 피처 확인
    hist_features = [c for c in df.columns if '_mean' in c or '_std' in c or 'anomaly' in c]
    print(f"Historical pattern features from 5-year data: {len(hist_features)}")

    # 2. V2 피처 엔지니어링 적용
    print("\n[2] Applying V2 Feature Engineering...")
    df_v2 = create_wind_features_v2(df)

    print(f"After V2 features: {len(df_v2.columns)} columns")

    # 3. 클린 피처 선택
    print("\n[3] Selecting clean features (no leakage)...")
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"Clean features selected: {len(feature_cols)}")

    # 피처 카테고리별 분류
    v2_features = [c for c in feature_cols if any(x in c for x in
        ['stability', 'richardson', 'shear', 'power_density', 'turbulence', 'efficiency'])]
    hist_features_used = [c for c in feature_cols if '_mean' in c or '_std' in c or 'anomaly' in c]
    weather_lag = [c for c in feature_cols if '_lag_' in c or '_rolling_' in c]

    print(f"  - V2 physics features: {len(v2_features)}")
    print(f"  - Historical pattern features: {len(hist_features_used)}")
    print(f"  - Weather lag/rolling features: {len(weather_lag)}")

    # 4. 데이터 분할
    print("\n[4] Splitting data...")
    train_df, val_df, test_df = train_val_test_split(df_v2)

    target_col = 'wind_mw'

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    print(f"Train: {len(X_train)} ({train_df.index.min()} ~ {train_df.index.max()})")
    print(f"Val:   {len(X_val)} ({val_df.index.min()} ~ {val_df.index.max()})")
    print(f"Test:  {len(X_test)} ({test_df.index.min()} ~ {test_df.index.max()})")

    # 5. 이전 결과 (비교용)
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 60)
    print(f"{'Phase 0 (Original)':<30} {'0.2530':>10} {'26.28':>10} {'21.88':>10}")
    print(f"{'Phase 1 (V2 Features)':<30} {'0.5153':>10} {'21.16':>10} {'16.43':>10}")

    # 6. Phase 2 모델 학습
    print("\n" + "=" * 60)
    print("TRAINING PHASE 2 MODEL")
    print("=" * 60)

    model = train_lgbm_model(X_train, y_train, X_val, y_val, feature_cols)
    print(f"Best iteration: {model.best_iteration}")

    # 예측
    y_pred_train = np.clip(model.predict(X_train, num_iteration=model.best_iteration), 0, None)
    y_pred_val = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0, None)
    y_pred_test = np.clip(model.predict(X_test, num_iteration=model.best_iteration), 0, None)

    # 7. 결과 평가
    metrics_train = calc_metrics(y_train, y_pred_train)
    metrics_val = calc_metrics(y_val, y_pred_val)
    metrics_test = calc_metrics(y_test, y_pred_test)

    print("\n[Results]")
    print(f"{'Split':<10} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print("-" * 50)
    print(f"{'Train':<10} {metrics_train['r2']:>10.4f} {metrics_train['rmse']:>10.2f} {metrics_train['mae']:>10.2f} {metrics_train['mape']:>10.1f}%")
    print(f"{'Val':<10} {metrics_val['r2']:>10.4f} {metrics_val['rmse']:>10.2f} {metrics_val['mae']:>10.2f} {metrics_val['mape']:>10.1f}%")
    print(f"{'Test':<10} {metrics_test['r2']:>10.4f} {metrics_test['rmse']:>10.2f} {metrics_test['mae']:>10.2f} {metrics_test['mape']:>10.1f}%")

    # 8. Phase 비교
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: Phase 0 vs Phase 1 vs Phase 2")
    print("=" * 70)
    print(f"{'Phase':<25} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Phase 0 (Original)':<25} {'0.2530':>10} {'26.28 MW':>12} {'21.88 MW':>12} {'-':>15}")
    print(f"{'Phase 1 (V2 Features)':<25} {'0.5153':>10} {'21.16 MW':>12} {'16.43 MW':>12} {'R² +103.7%':>15}")
    print(f"{'Phase 2 (5yr Patterns)':<25} {metrics_test['r2']:>10.4f} {metrics_test['rmse']:>7.2f} MW {metrics_test['mae']:>7.2f} MW ", end="")

    # 개선율 계산
    r2_improvement = (metrics_test['r2'] - 0.2530) / 0.2530 * 100
    rmse_improvement = (26.28 - metrics_test['rmse']) / 26.28 * 100

    if metrics_test['r2'] > 0.5153:
        print(f"R² +{r2_improvement:.1f}%")
    else:
        print(f"R² +{r2_improvement:.1f}%")

    # 9. Feature Importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n[Top 20 Features - Phase 2]")
    print(feature_importance.head(20).to_string(index=False))

    # 과거 기상 패턴 피처 중요도
    hist_importance = feature_importance[feature_importance['feature'].str.contains('_mean|_std|anomaly')]
    if len(hist_importance) > 0:
        print("\n[Historical Pattern Features Importance]")
        print(hist_importance.head(10).to_string(index=False))

    # 10. 모델 저장
    save_dir = MODELS_DIR / 'wind_phase2'
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(save_dir / 'wind_lgbm_phase2.lgb'))
    feature_importance.to_csv(save_dir / 'feature_importance_phase2.csv', index=False)

    # 예측 결과 저장
    results_df = test_df[[target_col]].copy()
    results_df['predicted'] = y_pred_test
    results_df['error'] = results_df[target_col] - results_df['predicted']
    results_df.to_csv(save_dir / 'test_predictions_phase2.csv')

    print(f"\nModel saved to: {save_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 요약 반환
    return {
        'phase2_test': metrics_test,
        'phase1_baseline': {'r2': 0.5153, 'rmse': 21.16, 'mae': 16.43},
        'phase0_baseline': {'r2': 0.2530, 'rmse': 26.28, 'mae': 21.88},
        'improvement_over_phase0': r2_improvement,
    }


if __name__ == "__main__":
    results = main()
