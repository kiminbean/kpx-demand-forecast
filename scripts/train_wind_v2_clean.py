"""
Wind Model Training V2 Clean - 누수 변수 완전 제거 버전

문제점 분석:
1. wind_gen_change: 타겟(wind_mw)의 변화량 → 누수
2. is_curtailed: 출력제어 여부 → 타겟과 직접 연관
3. output_ratio: 출력 비율 → 누수

해결:
- 모든 타겟 관련 파생 변수 제거
- 순수 기상 변수 기반 피처만 사용
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

    완전 제외 목록:
    - 타겟 자체 및 타겟 파생 변수
    - 발전량 관련 변수 (capacity factor, output ratio 등)
    - 출력제어 관련 변수
    """
    # 수치형 컬럼만
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 완전 제외 패턴
    exclude_patterns = [
        'solar_mw', 'wind_mw', 'total_mw',  # 타겟
        'solar_cf', 'wind_cf',  # capacity factor
        'output_ratio', 'expected_max',  # 출력 관련
        'is_curtailed', 'is_decoupling',  # 출력제어
        '_gen_change',  # 발전량 변화
        '_mw_lag', '_mw_ma', '_mw_std',  # 타겟 지연/롤링
    ]

    # 안전한 피처만 선택
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


def train_hybrid_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """
    Hybrid Model: Linear + LightGBM Residual

    Gemini 제안:
    - Linear Model로 추세(trend) 학습
    - LightGBM으로 잔차(residual) 학습
    - 외삽(extrapolation) 문제 완화
    """
    print("\n[Hybrid Model: Linear + LightGBM Residual]")

    # Step 1: Linear Model (Ridge)
    print("Step 1: Training Ridge Linear Model...")
    linear_model = Ridge(alpha=1.0)
    linear_model.fit(X_train, y_train)

    # Linear 예측
    linear_pred_train = linear_model.predict(X_train)
    linear_pred_val = linear_model.predict(X_val)
    linear_pred_test = linear_model.predict(X_test)

    # 잔차 계산
    residual_train = y_train - linear_pred_train
    residual_val = y_val - linear_pred_val

    print(f"  Linear Train R²: {r2_score(y_train, linear_pred_train):.4f}")
    print(f"  Linear Val R²: {r2_score(y_val, linear_pred_val):.4f}")
    print(f"  Linear Test R²: {r2_score(y_test, linear_pred_test):.4f}")

    # Step 2: LightGBM for Residuals
    print("\nStep 2: Training LightGBM on Residuals...")

    train_data = lgb.Dataset(X_train, label=residual_train)
    val_data = lgb.Dataset(X_val, label=residual_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'verbose': -1,
        'seed': 42,
    }

    lgbm_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    # LightGBM 잔차 예측
    residual_pred_train = lgbm_model.predict(X_train, num_iteration=lgbm_model.best_iteration)
    residual_pred_val = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
    residual_pred_test = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

    # Step 3: 최종 예측 = Linear + LightGBM Residual
    y_pred_train = np.clip(linear_pred_train + residual_pred_train, 0, None)
    y_pred_val = np.clip(linear_pred_val + residual_pred_val, 0, None)
    y_pred_test = np.clip(linear_pred_test + residual_pred_test, 0, None)

    return y_pred_train, y_pred_val, y_pred_test, linear_model, lgbm_model


def train_pure_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """
    Pure LightGBM (개선된 하이퍼파라미터)
    """
    print("\n[Pure LightGBM with Tuned Hyperparameters]")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.02,  # 더 낮은 학습률
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_child_samples': 50,  # 더 높은 min samples
        'reg_alpha': 1.0,  # 강한 정규화
        'reg_lambda': 1.0,
        'max_depth': 8,  # 깊이 제한
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

    print(f"Best iteration: {model.best_iteration}")

    y_pred_train = np.clip(model.predict(X_train, num_iteration=model.best_iteration), 0, None)
    y_pred_val = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0, None)
    y_pred_test = np.clip(model.predict(X_test, num_iteration=model.best_iteration), 0, None)

    return y_pred_train, y_pred_val, y_pred_test, model


def calc_metrics(y_true, y_pred):
    """메트릭 계산"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (0 제외)
    mask = y_true > 1  # 1MW 이상만
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def main():
    print("=" * 70)
    print("Wind Model V2 Clean - 누수 제거 + 하이브리드 모델")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "cleaned_kpx_dataset_v2.csv"
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(f"\nLoaded data: {len(df)} rows")

    # V2 피처 적용
    print("\n[1] Applying V2 Feature Engineering...")
    df_v2 = create_wind_features_v2(df)

    # 클린 피처 선택
    feature_cols = get_clean_features(df_v2, 'wind_mw')
    print(f"Clean features selected: {len(feature_cols)}")

    # 제거된 누수 변수 확인
    all_cols = df_v2.select_dtypes(include=[np.number]).columns.tolist()
    removed_cols = [c for c in all_cols if c not in feature_cols and c != 'wind_mw']
    print(f"Removed (potential leakage): {len(removed_cols)}")

    # 데이터 분할
    print("\n[2] Splitting data...")
    train_df, val_df, test_df = train_val_test_split(df_v2)

    # X, y 준비
    target_col = 'wind_mw'

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 이전 결과
    print("\n[Previous Baseline]")
    print("  Wind R²: 0.2530, RMSE: 26.28 MW, MAE: 21.88 MW")

    # ===== Method 1: Pure LightGBM =====
    print("\n" + "=" * 60)
    print("METHOD 1: Pure LightGBM (Tuned)")
    print("=" * 60)

    pred_train_1, pred_val_1, pred_test_1, lgbm_model = train_pure_lgbm(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    )

    metrics_1 = calc_metrics(y_test, pred_test_1)
    print(f"\nPure LightGBM Test Results:")
    print(f"  R²:   {metrics_1['r2']:.4f}")
    print(f"  RMSE: {metrics_1['rmse']:.2f}")
    print(f"  MAE:  {metrics_1['mae']:.2f}")

    # ===== Method 2: Hybrid (Linear + LightGBM) =====
    print("\n" + "=" * 60)
    print("METHOD 2: Hybrid (Linear + LightGBM Residual)")
    print("=" * 60)

    pred_train_2, pred_val_2, pred_test_2, linear_model, residual_model = train_hybrid_model(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    )

    metrics_2 = calc_metrics(y_test, pred_test_2)
    print(f"\nHybrid Model Test Results:")
    print(f"  R²:   {metrics_2['r2']:.4f}")
    print(f"  RMSE: {metrics_2['rmse']:.2f}")
    print(f"  MAE:  {metrics_2['mae']:.2f}")

    # ===== 결과 비교 =====
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 60)
    print(f"{'Baseline (Previous)':<25} {'0.2530':>10} {'26.28':>10} {'21.88':>10}")
    print(f"{'Pure LightGBM (Tuned)':<25} {metrics_1['r2']:>10.4f} {metrics_1['rmse']:>10.2f} {metrics_1['mae']:>10.2f}")
    print(f"{'Hybrid (Linear+LGBM)':<25} {metrics_2['r2']:>10.4f} {metrics_2['rmse']:>10.2f} {metrics_2['mae']:>10.2f}")

    # 개선율
    best_r2 = max(metrics_1['r2'], metrics_2['r2'])
    best_rmse = min(metrics_1['rmse'], metrics_2['rmse'])
    best_mae = min(metrics_1['mae'], metrics_2['mae'])

    print("\n[Best Improvement over Baseline]")
    print(f"  R² Change: {best_r2 - 0.2530:+.4f} ({(best_r2 - 0.2530)/0.2530*100:+.1f}%)")
    print(f"  RMSE Change: {best_rmse - 26.28:+.2f} ({(26.28 - best_rmse)/26.28*100:+.1f}%)")
    print(f"  MAE Change: {best_mae - 21.88:+.2f} ({(21.88 - best_mae)/21.88*100:+.1f}%)")

    # Feature Importance (Pure LightGBM)
    importance = lgbm_model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n[Top 15 Features (Clean)]")
    print(feature_importance.head(15).to_string(index=False))

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_v2_clean'
    save_dir.mkdir(parents=True, exist_ok=True)

    lgbm_model.save_model(str(save_dir / 'wind_lgbm_v2_clean.lgb'))
    feature_importance.to_csv(save_dir / 'feature_importance_clean.csv', index=False)

    print(f"\nModel saved to: {save_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'pure_lgbm': metrics_1,
        'hybrid': metrics_2,
    }


if __name__ == "__main__":
    results = main()
