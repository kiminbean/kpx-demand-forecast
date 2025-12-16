"""
Phase 6: Gemini 협업 개선사항 적용

합의된 개선 방안:
1. 출력제어(Curtailment) 데이터 정제
2. Ramp Rate 피처 강화 (가속도, 방향성)
3. LightGBM + XGBoost 앙상블
4. 잔차 학습 2단계 모델

목표: R² 0.7576 → 0.80+
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


def add_ramp_rate_features(df):
    """
    Gemini 제안: Ramp Rate 피처 강화
    - 풍속 가속도 (2차 미분)
    - 상승/하강 방향 구분
    - 급변 구간 플래그
    """
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

    # 급변 구간 (3시간 내 풍속 변화 > 3m/s)
    df['wind_rapid_change'] = (df['wind_delta_3h'].abs() > 3).astype(int)

    # 변화 크기 (절대값)
    df['wind_delta_abs_1h'] = df['wind_delta_1h'].abs()
    df['wind_delta_abs_3h'] = df['wind_delta_3h'].abs()

    # 변화율 (%)
    df['wind_pct_change_1h'] = df['wind_speed'].pct_change(1) * 100
    df['wind_pct_change_3h'] = df['wind_speed'].pct_change(3) * 100

    return df


def clean_curtailment_data(df, method='mask'):
    """
    Gemini 제안: 출력제어 데이터 정제

    방법:
    - 'mask': 출력제어 구간을 학습에서 제외 (가중치=0)
    - 'remove': 출력제어 구간 완전 제거
    - 'impute': 이론값으로 대체
    """
    df = df.copy()

    # 출력제어 감지: 풍속 > 5m/s 인데 발전량 < 용량의 5%
    # 제주도 풍력 설비용량 약 300MW 기준
    CAPACITY = 300  # MW
    MIN_THRESHOLD = CAPACITY * 0.05  # 15 MW

    # 기존 is_curtailed 플래그 활용 + 추가 감지
    if 'is_curtailed' in df.columns:
        curtailed_mask = df['is_curtailed'] == 1
    else:
        curtailed_mask = pd.Series([False] * len(df), index=df.index)

    # 추가 감지: 풍속 높은데 발전량 낮은 이상 구간
    additional_curtailed = (
        (df['wind_speed'] > 5) &
        (df['wind_mw'] < MIN_THRESHOLD) &
        (df['wind_mw'] > 0)  # 완전 0은 유지보수일 수 있음
    )

    total_curtailed = curtailed_mask | additional_curtailed
    curtailment_ratio = total_curtailed.sum() / len(df) * 100

    print(f"  출력제어 감지: {total_curtailed.sum()}건 ({curtailment_ratio:.1f}%)")

    if method == 'mask':
        # 가중치 방식: 출력제어 구간 가중치 = 0.1 (완전 제거 X)
        df['curtailment_weight'] = 1.0
        df.loc[total_curtailed, 'curtailment_weight'] = 0.1
    elif method == 'remove':
        # 완전 제거
        df = df[~total_curtailed].copy()
    elif method == 'impute':
        # 이론값으로 대체 (Wind Power Density 기반)
        if 'wind_power_density' in df.columns:
            # 이론 발전량 = WPD * 효율계수 * 로터면적
            theoretical = df['wind_power_density'] * 0.35 * 5000  # 가정
            df.loc[total_curtailed, 'wind_mw'] = theoretical[total_curtailed]

    return df


def main():
    print("=" * 70)
    print("Phase 6: Gemini 협업 개선사항 적용")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    data_path = PROCESSED_DATA_DIR / "merged_5year_dataset.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    print(f"  원본 데이터: {len(df)} rows")

    # 2. V2 피처 생성
    print("\n[2] V2 Feature Engineering")
    df_v2 = create_wind_features_v2(df)

    # 3. Ramp Rate 피처 추가 (Gemini 제안 #1)
    print("\n[3] Ramp Rate 피처 강화 (Gemini 제안)")
    df_v2 = add_ramp_rate_features(df_v2)
    new_ramp_features = [c for c in df_v2.columns if 'delta' in c or 'accel' in c or 'ramp' in c]
    print(f"  추가된 피처: {len(new_ramp_features)}개")
    print(f"  {new_ramp_features}")

    # 4. 출력제어 데이터 정제 (Gemini 제안 #2)
    print("\n[4] 출력제어 데이터 정제 (Gemini 제안)")
    df_clean = clean_curtailment_data(df_v2, method='mask')

    # 피처 선택
    feature_cols = get_clean_features(df_clean, 'wind_mw')
    # Ramp Rate 피처 추가
    ramp_cols = [c for c in df_clean.columns if any(x in c for x in ['delta', 'accel', 'ramp', 'rapid'])]
    for col in ramp_cols:
        if col not in feature_cols and col in df_clean.columns:
            feature_cols.append(col)

    print(f"  총 피처 수: {len(feature_cols)}")

    # 5. 데이터 분할
    n = len(df_clean)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]

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

    if 'curtailment_weight' in train_df.columns:
        curt_weights = train_df['curtailment_weight'].values
        weights = time_weights * curt_weights
    else:
        weights = time_weights

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # =================================================================
    # 개선 1: Baseline (Phase 5 재현)
    # =================================================================
    print("\n" + "=" * 70)
    print("개선 1: Baseline + Ramp Rate 피처")
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

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    baseline_pred = np.clip(lgb_model.predict(X_test), 0, None)
    baseline_metrics = calc_metrics(y_test, baseline_pred)

    print(f"  iterations: {lgb_model.best_iteration}")
    print(f"  R²: {baseline_metrics['r2']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} MW")
    print(f"  MAE: {baseline_metrics['mae']:.2f} MW")

    # =================================================================
    # 개선 2: LightGBM + XGBoost 앙상블 (Gemini 제안 #3)
    # =================================================================
    print("\n" + "=" * 70)
    print("개선 2: LightGBM + XGBoost 앙상블 (Gemini 제안)")
    print("=" * 70)

    try:
        import xgboost as xgb

        print("\n[XGBoost 학습]")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,  # 빠른 학습
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'seed': 42,
            'tree_method': 'hist',  # 빠른 학습 방법
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

        xgb_pred = np.clip(xgb_model.predict(dtest), 0, None)
        xgb_metrics = calc_metrics(y_test, xgb_pred)
        print(f"  XGBoost R²: {xgb_metrics['r2']:.4f}, RMSE: {xgb_metrics['rmse']:.2f}")

        # 검증셋 기반 가중치 계산
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

        print(f"\n[앙상블 가중치]")
        print(f"  LightGBM: {w_lgb:.3f}, XGBoost: {w_xgb:.3f}")

        # 앙상블 예측
        ensemble_pred = w_lgb * baseline_pred + w_xgb * xgb_pred
        ensemble_metrics = calc_metrics(y_test, ensemble_pred)

        print(f"\n[앙상블 결과]")
        print(f"  R²: {ensemble_metrics['r2']:.4f}")
        print(f"  RMSE: {ensemble_metrics['rmse']:.2f} MW")
        print(f"  MAE: {ensemble_metrics['mae']:.2f} MW")

        HAS_XGB = True

    except Exception as e:
        print(f"  XGBoost 실패: {e}")
        ensemble_pred = baseline_pred
        ensemble_metrics = baseline_metrics
        HAS_XGB = False

    # =================================================================
    # 개선 3: 잔차 학습 2단계 모델 (Gemini 제안 #4)
    # =================================================================
    print("\n" + "=" * 70)
    print("개선 3: 잔차 학습 2단계 모델 (Gemini 제안)")
    print("=" * 70)

    # 1단계: 앙상블 예측의 잔차 계산
    train_pred_lgb = lgb_model.predict(X_train)
    val_pred_lgb = lgb_model.predict(X_val)

    if HAS_XGB:
        train_pred_xgb = xgb_model.predict(xgb.DMatrix(X_train))
        val_pred_xgb = xgb_model.predict(xgb.DMatrix(X_val))
        train_pred_stage1 = w_lgb * train_pred_lgb + w_xgb * train_pred_xgb
        val_pred_stage1 = w_lgb * val_pred_lgb + w_xgb * val_pred_xgb
    else:
        train_pred_stage1 = train_pred_lgb
        val_pred_stage1 = val_pred_lgb

    # 잔차 계산
    train_residuals = y_train - train_pred_stage1
    val_residuals = y_val - val_pred_stage1

    print(f"  Train 잔차 통계: mean={train_residuals.mean():.2f}, std={train_residuals.std():.2f}")
    print(f"  Val 잔차 통계: mean={val_residuals.mean():.2f}, std={val_residuals.std():.2f}")

    # 2단계: 잔차 예측 모델 (시간/계절 피처 중심)
    residual_features = ['hour', 'month', 'dayofyear', 'wind_delta_1h', 'wind_delta_3h',
                         'wind_accel_1h', 'wind_ramping_up', 'wind_ramping_down',
                         'wind_rapid_change']
    residual_features = [f for f in residual_features if f in X_train.columns]

    X_train_res = X_train[residual_features]
    X_val_res = X_val[residual_features]
    X_test_res = X_test[residual_features]

    residual_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 20,  # 더 단순한 모델
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

    res_train_data = lgb.Dataset(X_train_res, label=train_residuals, free_raw_data=False)
    res_val_data = lgb.Dataset(X_val_res, label=val_residuals, reference=res_train_data, free_raw_data=False)

    residual_model = lgb.train(
        residual_params,
        res_train_data,
        num_boost_round=500,
        valid_sets=[res_val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # 2단계 예측 (잔차 보정)
    residual_pred = residual_model.predict(X_test_res)

    # 최종 예측 = 1단계 + 2단계
    final_pred = np.clip(ensemble_pred + residual_pred, 0, None)
    final_metrics = calc_metrics(y_test, final_pred)

    print(f"\n[2단계 잔차 모델]")
    print(f"  iterations: {residual_model.best_iteration}")
    print(f"  잔차 예측 범위: [{residual_pred.min():.2f}, {residual_pred.max():.2f}]")

    print(f"\n[최종 결과 (앙상블 + 잔차 학습)]")
    print(f"  R²: {final_metrics['r2']:.4f}")
    print(f"  RMSE: {final_metrics['rmse']:.2f} MW")
    print(f"  MAE: {final_metrics['mae']:.2f} MW")

    # =================================================================
    # 결과 요약
    # =================================================================
    print("\n" + "=" * 70)
    print("Phase 6 결과 요약")
    print("=" * 70)

    print(f"\n{'모델':<40} {'R²':>10} {'RMSE':>12} {'MAE':>10}")
    print("-" * 75)
    print(f"{'Phase 5 (이전 최고)':<40} {'0.7576':>10} {'14.96 MW':>12} {'11.57 MW':>10}")
    print(f"{'Phase 6 Baseline + Ramp Rate':<40} {baseline_metrics['r2']:>10.4f} {baseline_metrics['rmse']:>7.2f} MW {baseline_metrics['mae']:>7.2f} MW")

    if HAS_XGB:
        print(f"{'Phase 6 XGBoost':<40} {xgb_metrics['r2']:>10.4f} {xgb_metrics['rmse']:>7.2f} MW {xgb_metrics['mae']:>7.2f} MW")
        print(f"{'Phase 6 앙상블 (LGB + XGB)':<40} {ensemble_metrics['r2']:>10.4f} {ensemble_metrics['rmse']:>7.2f} MW {ensemble_metrics['mae']:>7.2f} MW")

    print(f"{'Phase 6 최종 (앙상블 + 잔차)':<40} {final_metrics['r2']:>10.4f} {final_metrics['rmse']:>7.2f} MW {final_metrics['mae']:>7.2f} MW")

    # 개선율
    baseline_r2 = 0.7576
    improvement = (final_metrics['r2'] - baseline_r2) / baseline_r2 * 100
    total_improvement = (final_metrics['r2'] - 0.2530) / 0.2530 * 100

    print(f"\n[개선 효과]")
    print(f"  Phase 5 대비: R² {baseline_r2:.4f} → {final_metrics['r2']:.4f} ({improvement:+.2f}%)")
    print(f"  Phase 0 대비: R² 0.2530 → {final_metrics['r2']:.4f} (+{total_improvement:.1f}%)")

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_phase6_gemini'
    save_dir.mkdir(parents=True, exist_ok=True)

    lgb_model.save_model(str(save_dir / 'wind_lgbm_phase6.lgb'))
    if HAS_XGB:
        xgb_model.save_model(str(save_dir / 'wind_xgb_phase6.json'))
    residual_model.save_model(str(save_dir / 'wind_residual_phase6.lgb'))

    # 앙상블 가중치 저장
    import json
    ensemble_config = {
        'lgb_weight': float(w_lgb) if HAS_XGB else 1.0,
        'xgb_weight': float(w_xgb) if HAS_XGB else 0.0,
        'final_r2': float(final_metrics['r2']),
        'final_rmse': float(final_metrics['rmse']),
        'feature_cols': feature_cols,
        'residual_features': residual_features,
    }
    with open(save_dir / 'ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\n모델 저장: {save_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
