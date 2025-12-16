"""
Phase 6: Optuna 심화 튜닝

Gemini 토론 결론:
- 새 피처/데이터 정제가 오히려 성능 저하
- 기존 Phase 5 구조 유지하면서 하이퍼파라미터 최적화에 집중

목표: R² 0.7576 → 0.78+
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
    print("Phase 6: Optuna 심화 튜닝")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드 (Phase 5와 동일)
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # V2 피처 (Phase 5와 동일)
    df_v2 = create_wind_features_v2(df)
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"  피처: {len(feature_cols)}")

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

    # 시간 가중치 (Phase 5와 동일)
    T = len(y_train)
    weights = np.array([0.999 ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum() * T

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =================================================================
    # Baseline (Phase 5 재현)
    # =================================================================
    print("\n" + "=" * 70)
    print("[Baseline] Phase 5 파라미터")
    print("=" * 70)

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

    # =================================================================
    # Optuna 심화 튜닝 (더 넓은 범위, 더 많은 trials)
    # =================================================================
    print("\n" + "=" * 70)
    print("[Optuna] 심화 하이퍼파라미터 튜닝")
    print("=" * 70)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 30, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'verbose': -1,
                'seed': 42,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            y_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))

        print("  최적화 시작 (30 trials)...")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print(f"\n  최적 Val RMSE: {study.best_value:.4f}")
        print(f"\n  최적 파라미터:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

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
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )

        optuna_pred = np.clip(optuna_model.predict(X_test), 0, None)
        optuna_metrics = calc_metrics(y_test, optuna_pred)

        print(f"\n[Optuna 최종 결과]")
        print(f"  iterations: {optuna_model.best_iteration}")
        print(f"  R²: {optuna_metrics['r2']:.4f}")
        print(f"  RMSE: {optuna_metrics['rmse']:.2f} MW")
        print(f"  MAE: {optuna_metrics['mae']:.2f} MW")

        # 비교
        print("\n" + "=" * 70)
        print("결과 비교")
        print("=" * 70)

        print(f"\n{'모델':<30} {'R²':>10} {'RMSE':>10}")
        print("-" * 55)
        print(f"{'Phase 5 Baseline':<30} {baseline_metrics['r2']:>10.4f} {baseline_metrics['rmse']:>10.2f}")
        print(f"{'Optuna 튜닝':<30} {optuna_metrics['r2']:>10.4f} {optuna_metrics['rmse']:>10.2f}")

        improvement = (optuna_metrics['r2'] - baseline_metrics['r2']) / baseline_metrics['r2'] * 100
        print(f"\n  개선율: {improvement:+.2f}%")

        # 모델 저장
        if optuna_metrics['r2'] > baseline_metrics['r2']:
            save_dir = MODELS_DIR / 'wind_phase6_optuna'
            save_dir.mkdir(parents=True, exist_ok=True)

            optuna_model.save_model(str(save_dir / 'wind_lgbm_optuna.lgb'))

            config = {
                'best_params': best_params,
                'r2': float(optuna_metrics['r2']),
                'rmse': float(optuna_metrics['rmse']),
                'mae': float(optuna_metrics['mae']),
                'feature_cols': feature_cols,
            }
            with open(save_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

            print(f"\n모델 저장: {save_dir}")
        else:
            print("\n  Optuna 튜닝이 Baseline보다 성능이 낮아 저장하지 않음")

    except ImportError:
        print("  Optuna 미설치")
        optuna_metrics = baseline_metrics

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
