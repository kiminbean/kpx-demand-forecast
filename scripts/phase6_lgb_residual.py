"""
Phase 6: LightGBM + 잔차 학습 (XGBoost 제외)

Gemini 협업 합의사항 중 구현 가능한 것:
1. 출력제어(Curtailment) 데이터 정제
2. Ramp Rate 피처 강화 (가속도, 방향성)
3. 잔차 학습 2단계 모델

XGBoost는 M1 Mac에서 지연 문제로 제외
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


def add_ramp_rate_features(df):
    """Ramp Rate 피처 강화"""
    df = df.copy()

    # 1차 미분 (속도 변화)
    df['wind_delta_1h'] = df['wind_speed'].diff(1)
    df['wind_delta_3h'] = df['wind_speed'].diff(3)
    df['wind_delta_6h'] = df['wind_speed'].diff(6)

    # 2차 미분 (가속도)
    df['wind_accel_1h'] = df['wind_delta_1h'].diff(1)
    df['wind_accel_3h'] = df['wind_delta_3h'].diff(1)

    # 방향성 플래그
    df['wind_ramping_up'] = (df['wind_delta_1h'] > 0.5).astype(int)
    df['wind_ramping_down'] = (df['wind_delta_1h'] < -0.5).astype(int)

    # 급변 구간
    df['wind_rapid_change'] = (df['wind_delta_3h'].abs() > 3).astype(int)

    # 변화 크기 (절대값)
    df['wind_delta_abs_1h'] = df['wind_delta_1h'].abs()
    df['wind_delta_abs_3h'] = df['wind_delta_3h'].abs()

    return df


def detect_curtailment(df):
    """출력제어 감지"""
    CAPACITY = 300  # MW
    MIN_THRESHOLD = CAPACITY * 0.05  # 15 MW

    if 'is_curtailed' in df.columns:
        curtailed_mask = df['is_curtailed'] == 1
    else:
        curtailed_mask = pd.Series([False] * len(df), index=df.index)

    # 추가 감지
    additional_curtailed = (
        (df['wind_speed'] > 5) &
        (df['wind_mw'] < MIN_THRESHOLD) &
        (df['wind_mw'] > 0)
    )

    return curtailed_mask | additional_curtailed


def main():
    print("=" * 70)
    print("Phase 6: LightGBM + 잔차 학습")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    print(f"  데이터: {len(df)} rows")

    # 2. V2 피처 + Ramp Rate 피처
    print("\n[2] Feature Engineering")
    df_v2 = create_wind_features_v2(df)
    df_v2 = add_ramp_rate_features(df_v2)

    # 3. 출력제어 감지
    curtailed = detect_curtailment(df_v2)
    print(f"  출력제어 감지: {curtailed.sum()}건 ({curtailed.sum()/len(df_v2)*100:.1f}%)")

    # 가중치 계산용
    df_v2['curtailment_weight'] = 1.0
    df_v2.loc[curtailed, 'curtailment_weight'] = 0.1

    # 피처 선택
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    ramp_cols = ['wind_delta_1h', 'wind_delta_3h', 'wind_delta_6h',
                 'wind_accel_1h', 'wind_accel_3h', 'wind_ramping_up',
                 'wind_ramping_down', 'wind_rapid_change',
                 'wind_delta_abs_1h', 'wind_delta_abs_3h']
    for col in ramp_cols:
        if col in df_v2.columns and col not in feature_cols:
            feature_cols.append(col)

    print(f"  총 피처: {len(feature_cols)}")

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

    # 시간 가중치 + 출력제어 가중치
    T = len(y_train)
    time_weights = np.array([0.999 ** (T - t - 1) for t in range(T)])
    time_weights = time_weights / time_weights.sum() * T
    curt_weights = train_df['curtailment_weight'].values
    weights = time_weights * curt_weights

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =================================================================
    # Stage 1: 메인 LightGBM 모델
    # =================================================================
    print("\n" + "=" * 70)
    print("Stage 1: 메인 LightGBM 모델")
    print("=" * 70)

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

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    stage1_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    stage1_pred = np.clip(stage1_model.predict(X_test), 0, None)
    stage1_metrics = calc_metrics(y_test, stage1_pred)

    print(f"  iterations: {stage1_model.best_iteration}")
    print(f"  R²: {stage1_metrics['r2']:.4f}")
    print(f"  RMSE: {stage1_metrics['rmse']:.2f} MW")
    print(f"  MAE: {stage1_metrics['mae']:.2f} MW")

    # =================================================================
    # Stage 2: 잔차 학습 모델
    # =================================================================
    print("\n" + "=" * 70)
    print("Stage 2: 잔차 학습 모델")
    print("=" * 70)

    # Stage 1 예측으로 잔차 계산
    train_pred_s1 = stage1_model.predict(X_train)
    val_pred_s1 = stage1_model.predict(X_val)

    train_residuals = y_train - train_pred_s1
    val_residuals = y_val - val_pred_s1

    print(f"  Train 잔차: mean={train_residuals.mean():.2f}, std={train_residuals.std():.2f}")
    print(f"  Val 잔차: mean={val_residuals.mean():.2f}, std={val_residuals.std():.2f}")

    # 잔차 예측용 피처 (시간/계절 + Ramp Rate)
    residual_features = ['hour', 'month', 'dayofyear',
                         'wind_delta_1h', 'wind_delta_3h',
                         'wind_accel_1h', 'wind_ramping_up', 'wind_ramping_down',
                         'wind_rapid_change', 'wind_speed', 'wind_direction']
    residual_features = [f for f in residual_features if f in X_train.columns]

    print(f"  잔차 피처: {len(residual_features)}")

    X_train_res = X_train[residual_features]
    X_val_res = X_val[residual_features]
    X_test_res = X_test[residual_features]

    # 잔차 모델 (더 단순하게)
    residual_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'max_depth': 4,
        'verbose': -1,
        'seed': 42,
    }

    res_train = lgb.Dataset(X_train_res, label=train_residuals, free_raw_data=False)
    res_val = lgb.Dataset(X_val_res, label=val_residuals, reference=res_train, free_raw_data=False)

    stage2_model = lgb.train(
        residual_params,
        res_train,
        num_boost_round=500,
        valid_sets=[res_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Stage 2 예측
    residual_pred = stage2_model.predict(X_test_res)
    print(f"\n  Stage 2 iterations: {stage2_model.best_iteration}")
    print(f"  잔차 예측 범위: [{residual_pred.min():.2f}, {residual_pred.max():.2f}]")

    # =================================================================
    # 최종 예측: Stage 1 + Stage 2
    # =================================================================
    print("\n" + "=" * 70)
    print("최종 결과: Stage 1 + Stage 2")
    print("=" * 70)

    final_pred = np.clip(stage1_pred + residual_pred, 0, None)
    final_metrics = calc_metrics(y_test, final_pred)

    print(f"  R²: {final_metrics['r2']:.4f}")
    print(f"  RMSE: {final_metrics['rmse']:.2f} MW")
    print(f"  MAE: {final_metrics['mae']:.2f} MW")

    # =================================================================
    # 비교 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("성능 비교")
    print("=" * 70)

    print(f"\n{'모델':<35} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)
    print(f"{'Phase 5 (이전 최고)':<35} {'0.7576':>10} {'14.96':>10} {'11.57':>10}")
    print(f"{'Stage 1 (LightGBM + Ramp Rate)':<35} {stage1_metrics['r2']:>10.4f} {stage1_metrics['rmse']:>10.2f} {stage1_metrics['mae']:>10.2f}")
    print(f"{'Stage 1+2 (+ 잔차 학습)':<35} {final_metrics['r2']:>10.4f} {final_metrics['rmse']:>10.2f} {final_metrics['mae']:>10.2f}")

    # 개선율 계산
    phase5_r2 = 0.7576
    improvement_s1 = (stage1_metrics['r2'] - phase5_r2) / phase5_r2 * 100
    improvement_final = (final_metrics['r2'] - phase5_r2) / phase5_r2 * 100
    total_improvement = (final_metrics['r2'] - 0.2530) / 0.2530 * 100

    print(f"\n[개선 효과]")
    print(f"  Stage 1 vs Phase 5: {improvement_s1:+.2f}%")
    print(f"  Final vs Phase 5: {improvement_final:+.2f}%")
    print(f"  Final vs Phase 0: +{total_improvement:.1f}%")

    # Feature Importance (Stage 1)
    print("\n[Top 15 Features - Stage 1]")
    importance = stage1_model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(fi_df.head(15).to_string(index=False))

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_phase6'
    save_dir.mkdir(parents=True, exist_ok=True)

    stage1_model.save_model(str(save_dir / 'wind_lgbm_stage1.lgb'))
    stage2_model.save_model(str(save_dir / 'wind_lgbm_stage2_residual.lgb'))
    fi_df.to_csv(save_dir / 'feature_importance_phase6.csv', index=False)

    # 설정 저장
    config = {
        'stage1_r2': float(stage1_metrics['r2']),
        'stage1_rmse': float(stage1_metrics['rmse']),
        'final_r2': float(final_metrics['r2']),
        'final_rmse': float(final_metrics['rmse']),
        'feature_cols': feature_cols,
        'residual_features': residual_features,
    }
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n모델 저장: {save_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
