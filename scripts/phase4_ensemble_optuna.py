"""
Phase 4: 앙상블 모델 + Optuna 튜닝 (간소화 버전)
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
    """평가 메트릭"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def main():
    print("=" * 70)
    print("Phase 4: 앙상블 모델 + Optuna 튜닝")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    print(f"Loaded: {len(df)} rows")

    # 2. V2 피처
    print("\n[2] V2 Feature Engineering")
    df_v2 = create_wind_features_v2(df)
    print(f"Features: {len(df_v2.columns)}")

    # 3. 클린 피처
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"Clean features: {len(feature_cols)}")

    # 4. 데이터 분할
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

    # 시간 가중치 (Concept Drift 해결용)
    T = len(y_train)
    weights = np.array([0.999 ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum() * T

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =================================================================
    # Phase 4: 앙상블 모델
    # =================================================================
    print("\n" + "=" * 70)
    print("앙상블 모델 학습")
    print("=" * 70)

    predictions = {}
    models = {}

    # LightGBM (시간 가중치 적용)
    print("\n[1] LightGBM (시간 가중치)")
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

    lgb_train = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    predictions['lgb'] = np.clip(lgb_model.predict(X_test), 0, None)
    models['lgb'] = lgb_model
    lgb_metrics = calc_metrics(y_test, predictions['lgb'])
    print(f"  완료 (iterations: {lgb_model.best_iteration})")
    print(f"  R²: {lgb_metrics['r2']:.4f}, RMSE: {lgb_metrics['rmse']:.2f}")

    # XGBoost
    try:
        import xgboost as xgb
        print("\n[2] XGBoost")

        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.05,  # 더 높은 학습률
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'seed': 42,
            'n_jobs': -1,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,  # 줄임
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        predictions['xgb'] = np.clip(xgb_model.predict(dtest), 0, None)
        models['xgb'] = xgb_model
        xgb_metrics = calc_metrics(y_test, predictions['xgb'])
        print(f"  완료 (iterations: {xgb_model.best_iteration})")
        print(f"  R²: {xgb_metrics['r2']:.4f}, RMSE: {xgb_metrics['rmse']:.2f}")

        HAS_XGB = True
    except Exception as e:
        print(f"  XGBoost 실패: {e}")
        HAS_XGB = False

    # 앙상블
    print("\n[3] 앙상블 결합")

    if HAS_XGB:
        # 검증셋 성능 기반 가중치
        lgb_val_pred = lgb_model.predict(X_val)
        xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val))

        lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
        xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))

        # 역RMSE 가중치
        w_lgb = 1 / lgb_val_rmse
        w_xgb = 1 / xgb_val_rmse
        total_w = w_lgb + w_xgb
        w_lgb /= total_w
        w_xgb /= total_w

        print(f"  가중치: LightGBM={w_lgb:.3f}, XGBoost={w_xgb:.3f}")

        # 가중 평균
        ensemble_pred = w_lgb * predictions['lgb'] + w_xgb * predictions['xgb']

        # 단순 평균
        ensemble_avg = (predictions['lgb'] + predictions['xgb']) / 2
    else:
        ensemble_pred = predictions['lgb']
        ensemble_avg = predictions['lgb']

    ens_metrics = calc_metrics(y_test, ensemble_pred)
    ens_avg_metrics = calc_metrics(y_test, ensemble_avg)

    print(f"\n[결과 비교]")
    print(f"{'모델':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 50)
    print(f"{'LightGBM':<20} {lgb_metrics['r2']:>10.4f} {lgb_metrics['rmse']:>10.2f} {lgb_metrics['mae']:>10.2f}")
    if HAS_XGB:
        print(f"{'XGBoost':<20} {xgb_metrics['r2']:>10.4f} {xgb_metrics['rmse']:>10.2f} {xgb_metrics['mae']:>10.2f}")
        print(f"{'앙상블 (평균)':<20} {ens_avg_metrics['r2']:>10.4f} {ens_avg_metrics['rmse']:>10.2f} {ens_avg_metrics['mae']:>10.2f}")
        print(f"{'앙상블 (가중)':<20} {ens_metrics['r2']:>10.4f} {ens_metrics['rmse']:>10.2f} {ens_metrics['mae']:>10.2f}")

    # =================================================================
    # Optuna 튜닝
    # =================================================================
    print("\n" + "=" * 70)
    print("Optuna 하이퍼파라미터 튜닝")
    print("=" * 70)

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

            train_data = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            y_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))

        print("\n최적화 시작 (15 trials)...")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15, show_progress_bar=True)

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

        final_train = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
        final_val = lgb.Dataset(X_val, label=y_val, reference=final_train, free_raw_data=False)

        optuna_model = lgb.train(
            best_params,
            final_train,
            num_boost_round=2000,
            valid_sets=[final_val],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        optuna_pred = np.clip(optuna_model.predict(X_test), 0, None)
        optuna_metrics = calc_metrics(y_test, optuna_pred)

        print(f"\n[Optuna 튜닝 모델 테스트 성능]")
        print(f"  R²: {optuna_metrics['r2']:.4f}")
        print(f"  RMSE: {optuna_metrics['rmse']:.2f} MW")
        print(f"  MAE: {optuna_metrics['mae']:.2f} MW")

        # 모델 저장
        save_dir = MODELS_DIR / 'wind_phase4'
        save_dir.mkdir(parents=True, exist_ok=True)
        optuna_model.save_model(str(save_dir / 'wind_lgbm_optuna_best.lgb'))
        print(f"\n모델 저장: {save_dir / 'wind_lgbm_optuna_best.lgb'}")

    except ImportError:
        print("  Optuna 미설치. pip install optuna")
        optuna_metrics = None

    # =================================================================
    # 최종 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("최종 결과 요약")
    print("=" * 70)

    print(f"\n[Phase별 성능 비교]")
    print(f"{'Phase':<35} {'R²':>10} {'RMSE':>12}")
    print("-" * 60)
    print(f"{'Phase 0 (Baseline)':<35} {'0.2530':>10} {'26.28 MW':>12}")
    print(f"{'Phase 1 (V2 Features)':<35} {'0.5153':>10} {'21.16 MW':>12}")
    print(f"{'Phase 2 (5yr Patterns)':<35} {'0.6357':>10} {'18.34 MW':>12}")
    print(f"{'Phase 3 (시간 가중치)':<35} {'0.7441':>10} {'15.37 MW':>12}")
    print(f"{'Phase 4 (앙상블 가중)':<35} {ens_metrics['r2']:>10.4f} {ens_metrics['rmse']:>7.2f} MW")
    if optuna_metrics:
        print(f"{'Optuna 튜닝':<35} {optuna_metrics['r2']:>10.4f} {optuna_metrics['rmse']:>7.2f} MW")

    # 총 개선율
    baseline_r2 = 0.2530
    final_r2 = optuna_metrics['r2'] if optuna_metrics else ens_metrics['r2']
    improvement = (final_r2 - baseline_r2) / baseline_r2 * 100

    print(f"\n[총 개선율]")
    print(f"  Phase 0 → 최종: R² {baseline_r2:.4f} → {final_r2:.4f}")
    print(f"  개선율: +{improvement:.1f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
