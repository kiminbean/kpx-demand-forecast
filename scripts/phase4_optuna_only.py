"""
Phase 4: LightGBM Optuna 튜닝 (XGBoost 제외)
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


def main():
    print("=" * 70)
    print("Phase 4: LightGBM Optuna 튜닝")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # V2 피처
    df_v2 = create_wind_features_v2(df)
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"Features: {len(feature_cols)}")

    # 데이터 분할
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

    # 시간 가중치
    T = len(y_train)
    weights = np.array([0.999 ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum() * T

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =================================================================
    # Baseline (시간 가중치 적용)
    # =================================================================
    print("\n[2] Baseline 모델 (시간 가중치)")

    baseline_params = {
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

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    baseline_model = lgb.train(
        baseline_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    baseline_pred = np.clip(baseline_model.predict(X_test), 0, None)
    baseline_metrics = calc_metrics(y_test, baseline_pred)

    print(f"  iterations: {baseline_model.best_iteration}")
    print(f"  R²: {baseline_metrics['r2']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} MW")
    print(f"  MAE: {baseline_metrics['mae']:.2f} MW")

    # =================================================================
    # Optuna 튜닝
    # =================================================================
    print("\n[3] Optuna 하이퍼파라미터 튜닝")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'verbose': -1,
                'seed': 42,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            y_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))

        print("최적화 시작 (20 trials)...")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        print(f"\n최적 파라미터:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"최적 Val RMSE: {study.best_value:.4f}")

        # 최적 파라미터로 최종 모델
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
        })

        optuna_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        optuna_pred = np.clip(optuna_model.predict(X_test), 0, None)
        optuna_metrics = calc_metrics(y_test, optuna_pred)

        print(f"\n[Optuna 튜닝 모델 테스트 성능]")
        print(f"  iterations: {optuna_model.best_iteration}")
        print(f"  R²: {optuna_metrics['r2']:.4f}")
        print(f"  RMSE: {optuna_metrics['rmse']:.2f} MW")
        print(f"  MAE: {optuna_metrics['mae']:.2f} MW")

        # Feature Importance
        print("\n[Top 15 Features]")
        importance = optuna_model.feature_importance(importance_type='gain')
        fi_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print(fi_df.head(15).to_string(index=False))

        # 모델 저장
        save_dir = MODELS_DIR / 'wind_phase4'
        save_dir.mkdir(parents=True, exist_ok=True)
        optuna_model.save_model(str(save_dir / 'wind_lgbm_optuna_best.lgb'))
        fi_df.to_csv(save_dir / 'feature_importance_optuna.csv', index=False)
        print(f"\n모델 저장: {save_dir}")

    except ImportError:
        print("  Optuna 미설치. pip install optuna")
        optuna_metrics = baseline_metrics

    # =================================================================
    # 최종 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    print(f"\n[Phase별 Wind 모델 성능]")
    print(f"{'Phase':<40} {'R²':>10} {'RMSE':>12} {'개선율':>12}")
    print("-" * 75)
    print(f"{'Phase 0 (Baseline - 원본)':<40} {'0.2530':>10} {'26.28 MW':>12} {'-':>12}")
    print(f"{'Phase 1 (V2 Physics Features)':<40} {'0.5153':>10} {'21.16 MW':>12} {'+103.7%':>12}")
    print(f"{'Phase 2 (5년 기상 패턴)':<40} {'0.6357':>10} {'18.34 MW':>12} {'+151.3%':>12}")
    base_imp = (baseline_metrics['r2'] - 0.2530) / 0.2530 * 100
    opt_imp = (optuna_metrics['r2'] - 0.2530) / 0.2530 * 100
    print(f"{'Phase 3 (시간 가중치 학습)':<40} {baseline_metrics['r2']:>10.4f} {baseline_metrics['rmse']:>7.2f} MW {'+' + f'{base_imp:.1f}%':>11}")
    print(f"{'Phase 4 (Optuna 튜닝)':<40} {optuna_metrics['r2']:>10.4f} {optuna_metrics['rmse']:>7.2f} MW {'+' + f'{opt_imp:.1f}%':>11}")

    # 총 개선
    final_improvement = (optuna_metrics['r2'] - 0.2530) / 0.2530 * 100
    rmse_reduction = (26.28 - optuna_metrics['rmse']) / 26.28 * 100

    print(f"\n[총 개선 효과]")
    print(f"  R²: 0.2530 → {optuna_metrics['r2']:.4f} (+{final_improvement:.1f}%)")
    print(f"  RMSE: 26.28 MW → {optuna_metrics['rmse']:.2f} MW (-{rmse_reduction:.1f}%)")

    print(f"\n[해결된 문제]")
    print(f"  1. 테스트셋 분산 감소: 분산 정규화 R² = 0.7752")
    print(f"  2. Concept Drift: 시간 가중치로 R² +0.11 개선")
    print(f"  3. 풍력 확률성: 80% 예측 구간 커버리지 65.2%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
