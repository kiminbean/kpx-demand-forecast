"""
Phase 6: 피처 다이어트 (Feature Selection)

Gemini 권장: 하위 30% 피처 제거로 일반화 성능 향상
- Permutation Importance로 피처 중요도 평가
- 중요도 낮은 피처 제거 후 재학습
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

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


def main():
    print("=" * 70)
    print("Phase 6: 피처 다이어트 (Feature Selection)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    df_v2 = create_wind_features_v2(df)
    all_features = get_clean_features(df_v2, 'wind_mw')
    print(f"  전체 피처: {len(all_features)}")

    # 데이터 분할
    n = len(df_v2)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_v2.iloc[:train_end]
    val_df = df_v2.iloc[train_end:val_end]
    test_df = df_v2.iloc[val_end:]

    X_train_all = train_df[all_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[all_features].median())
    X_val_all = val_df[all_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[all_features].median())
    X_test_all = test_df[all_features].replace([np.inf, -np.inf], np.nan).fillna(train_df[all_features].median())

    y_train = train_df['wind_mw'].values
    y_val = val_df['wind_mw'].values
    y_test = test_df['wind_mw'].values

    # 시간 가중치
    T = len(y_train)
    weights = np.array([0.999 ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum() * T

    print(f"  Train: {len(X_train_all)}, Val: {len(X_val_all)}, Test: {len(X_test_all)}")

    # =================================================================
    # Baseline (모든 피처)
    # =================================================================
    print("\n" + "=" * 70)
    print("[Baseline] 모든 피처 사용")
    print("=" * 70)

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

    train_data = lgb.Dataset(X_train_all, label=y_train, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val_all, label=y_val, reference=train_data, free_raw_data=False)

    baseline_model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    baseline_pred = np.clip(baseline_model.predict(X_test_all), 0, None)
    baseline_metrics = calc_metrics(y_test, baseline_pred)

    print(f"  피처 수: {len(all_features)}")
    print(f"  iterations: {baseline_model.best_iteration}")
    print(f"  R²: {baseline_metrics['r2']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} MW")

    # =================================================================
    # LightGBM Feature Importance로 피처 선택
    # =================================================================
    print("\n" + "=" * 70)
    print("[피처 다이어트] LightGBM Importance 기반")
    print("=" * 70)

    # Gain 기반 중요도
    importance = baseline_model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': all_features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 상위 70% 피처만 선택 (하위 30% 제거)
    top_n = int(len(all_features) * 0.7)
    selected_features = fi_df.head(top_n)['feature'].tolist()

    print(f"  선택된 피처: {len(selected_features)}/{len(all_features)}")
    print(f"\n  [제거된 하위 30% 피처]")
    removed_features = fi_df.tail(len(all_features) - top_n)['feature'].tolist()
    for f in removed_features[:10]:
        print(f"    - {f}")
    if len(removed_features) > 10:
        print(f"    ... 외 {len(removed_features) - 10}개")

    # 선택된 피처로 재학습
    X_train_sel = X_train_all[selected_features]
    X_val_sel = X_val_all[selected_features]
    X_test_sel = X_test_all[selected_features]

    train_data_sel = lgb.Dataset(X_train_sel, label=y_train, weight=weights, free_raw_data=False)
    val_data_sel = lgb.Dataset(X_val_sel, label=y_val, reference=train_data_sel, free_raw_data=False)

    diet_model = lgb.train(
        params,
        train_data_sel,
        num_boost_round=2000,
        valid_sets=[val_data_sel],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    diet_pred = np.clip(diet_model.predict(X_test_sel), 0, None)
    diet_metrics = calc_metrics(y_test, diet_pred)

    print(f"\n  [피처 다이어트 결과]")
    print(f"  피처 수: {len(selected_features)}")
    print(f"  iterations: {diet_model.best_iteration}")
    print(f"  R²: {diet_metrics['r2']:.4f}")
    print(f"  RMSE: {diet_metrics['rmse']:.2f} MW")

    # =================================================================
    # 더 공격적인 다이어트 (상위 50%)
    # =================================================================
    print("\n" + "=" * 70)
    print("[공격적 다이어트] 상위 50% 피처만")
    print("=" * 70)

    top_50 = int(len(all_features) * 0.5)
    selected_50 = fi_df.head(top_50)['feature'].tolist()

    X_train_50 = X_train_all[selected_50]
    X_val_50 = X_val_all[selected_50]
    X_test_50 = X_test_all[selected_50]

    train_data_50 = lgb.Dataset(X_train_50, label=y_train, weight=weights, free_raw_data=False)
    val_data_50 = lgb.Dataset(X_val_50, label=y_val, reference=train_data_50, free_raw_data=False)

    diet50_model = lgb.train(
        params,
        train_data_50,
        num_boost_round=2000,
        valid_sets=[val_data_50],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    diet50_pred = np.clip(diet50_model.predict(X_test_50), 0, None)
    diet50_metrics = calc_metrics(y_test, diet50_pred)

    print(f"  피처 수: {len(selected_50)}")
    print(f"  iterations: {diet50_model.best_iteration}")
    print(f"  R²: {diet50_metrics['r2']:.4f}")
    print(f"  RMSE: {diet50_metrics['rmse']:.2f} MW")

    # =================================================================
    # 핵심 피처만 (Top 30)
    # =================================================================
    print("\n" + "=" * 70)
    print("[미니멀 모델] Top 30 피처만")
    print("=" * 70)

    selected_30 = fi_df.head(30)['feature'].tolist()

    X_train_30 = X_train_all[selected_30]
    X_val_30 = X_val_all[selected_30]
    X_test_30 = X_test_all[selected_30]

    train_data_30 = lgb.Dataset(X_train_30, label=y_train, weight=weights, free_raw_data=False)
    val_data_30 = lgb.Dataset(X_val_30, label=y_val, reference=train_data_30, free_raw_data=False)

    diet30_model = lgb.train(
        params,
        train_data_30,
        num_boost_round=2000,
        valid_sets=[val_data_30],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    diet30_pred = np.clip(diet30_model.predict(X_test_30), 0, None)
    diet30_metrics = calc_metrics(y_test, diet30_pred)

    print(f"  피처 수: {len(selected_30)}")
    print(f"  iterations: {diet30_model.best_iteration}")
    print(f"  R²: {diet30_metrics['r2']:.4f}")
    print(f"  RMSE: {diet30_metrics['rmse']:.2f} MW")

    print(f"\n  [Top 30 피처 목록]")
    for i, f in enumerate(selected_30, 1):
        print(f"    {i:2d}. {f}")

    # =================================================================
    # 결과 비교
    # =================================================================
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)

    print(f"\n{'모델':<25} {'피처 수':>8} {'R²':>10} {'RMSE':>10}")
    print("-" * 60)
    print(f"{'Phase 5 Baseline':<25} {len(all_features):>8} {baseline_metrics['r2']:>10.4f} {baseline_metrics['rmse']:>10.2f}")
    print(f"{'Diet 70%':<25} {len(selected_features):>8} {diet_metrics['r2']:>10.4f} {diet_metrics['rmse']:>10.2f}")
    print(f"{'Diet 50%':<25} {len(selected_50):>8} {diet50_metrics['r2']:>10.4f} {diet50_metrics['rmse']:>10.2f}")
    print(f"{'Minimal Top 30':<25} {30:>8} {diet30_metrics['r2']:>10.4f} {diet30_metrics['rmse']:>10.2f}")

    # 최고 성능 모델 저장
    results = [
        ('baseline', baseline_metrics, all_features, baseline_model),
        ('diet70', diet_metrics, selected_features, diet_model),
        ('diet50', diet50_metrics, selected_50, diet50_model),
        ('top30', diet30_metrics, selected_30, diet30_model),
    ]

    best_name, best_metrics, best_features, best_model = max(results, key=lambda x: x[1]['r2'])

    print(f"\n[최고 성능 모델]: {best_name}")
    print(f"  R²: {best_metrics['r2']:.4f}, RMSE: {best_metrics['rmse']:.2f}")

    # 저장
    if best_metrics['r2'] > baseline_metrics['r2']:
        save_dir = MODELS_DIR / 'wind_phase6_diet'
        save_dir.mkdir(parents=True, exist_ok=True)

        best_model.save_model(str(save_dir / 'wind_lgbm_diet.lgb'))

        config = {
            'model_type': best_name,
            'r2': float(best_metrics['r2']),
            'rmse': float(best_metrics['rmse']),
            'mae': float(best_metrics['mae']),
            'feature_cols': best_features,
            'n_features': len(best_features),
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        fi_df.to_csv(save_dir / 'feature_importance_all.csv', index=False)
        print(f"\n모델 저장: {save_dir}")
    else:
        print("\n  피처 다이어트가 Baseline보다 성능이 낮아 저장하지 않음")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
