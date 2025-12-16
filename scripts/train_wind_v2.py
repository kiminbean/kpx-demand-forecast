"""
Wind Model Training V2 - Phase 1 Enhancement

Gemini 협업 결과 반영:
1. 개선된 Physics-informed Features 적용
2. 기상 변수 Lag/Rolling Features 추가
3. 풍향-풍속 상호작용 Features 추가
4. 공간적 시차(Spatial Lag) Features 추가

목표: Wind R² 0.25 → 0.5~0.6
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Error: LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.features.wind_features_v2 import create_wind_features_v2


def prepare_data(df: pd.DataFrame, target_col: str = 'wind_mw'):
    """
    데이터 준비 및 누수 변수 제거

    Args:
        df: 입력 DataFrame
        target_col: 타겟 컬럼

    Returns:
        X, y, feature_names
    """
    # 수치형 컬럼만
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 누수 변수 목록
    leakage_vars = [
        'solar_cf', 'wind_cf', 'total_mw',  # 타겟에서 직접 파생
        'solar_mw', 'wind_mw',  # 타겟 자체
        'output_ratio', 'expected_max_gen',  # 출력 관련
    ]

    # 타겟 기반 지연/롤링 특성 (누수)
    target_derived = [col for col in numeric_cols if
                      col.startswith('solar_mw_') or col.startswith('wind_mw_')]

    # 제외 목록
    exclude_cols = set(leakage_vars + target_derived + [target_col])

    # 안전한 피처만 선택
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # X, y 분리
    X = df[feature_cols].copy()
    y = df[target_col].values

    # 무한값 처리
    X = X.replace([np.inf, -np.inf], np.nan)

    # 결측치 처리 (중앙값)
    X = X.fillna(X.median())

    return X, y, feature_cols


def train_val_test_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """시계열 순서 유지 데이터 분할"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def train_wind_model_v2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'wind_mw',
):
    """
    개선된 Wind 모델 학습

    Args:
        train_df: 학습 데이터
        val_df: 검증 데이터
        test_df: 테스트 데이터
        target_col: 타겟 컬럼

    Returns:
        모델, 메트릭, feature importance
    """
    # 데이터 준비
    X_train, y_train, feature_cols = prepare_data(train_df, target_col)
    X_val, y_val, _ = prepare_data(val_df, target_col)
    X_test, y_test, _ = prepare_data(test_df, target_col)

    # 동일 컬럼 순서 보장
    X_val = X_val[feature_cols]
    X_test = X_test[feature_cols]

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 하이퍼파라미터 (Gemini 권장 조정)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,  # 증가: 더 복잡한 패턴 학습
        'learning_rate': 0.03,  # 감소: 더 안정적인 학습
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,  # 과적합 방지
        'reg_alpha': 0.1,  # L1 정규화
        'reg_lambda': 0.1,  # L2 정규화
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }

    # 콜백
    callbacks = [
        lgb.early_stopping(100, verbose=True),
        lgb.log_evaluation(100),
    ]

    # 학습
    print("\n" + "=" * 60)
    print("Training Wind LightGBM V2")
    print("=" * 60)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
    )

    # 예측
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 음수 방지
    y_train_pred = np.clip(y_train_pred, 0, None)
    y_val_pred = np.clip(y_val_pred, 0, None)
    y_test_pred = np.clip(y_test_pred, 0, None)

    # 메트릭 계산
    def calc_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (0 제외)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')

        print(f"\n{name} Metrics:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    train_metrics = calc_metrics(y_train, y_train_pred, "Train")
    val_metrics = calc_metrics(y_val, y_val_pred, "Validation")
    test_metrics = calc_metrics(y_test, y_test_pred, "Test")

    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 60)
    print("Top 20 Feature Importance")
    print("=" * 60)
    print(feature_importance.head(20).to_string(index=False))

    return model, test_metrics, feature_importance, feature_cols


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("Wind Model V2 Training - Phase 1 Enhancement")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "cleaned_kpx_dataset_v2.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")

    # V2 피처 적용
    print("\n[1] Applying V2 Feature Engineering...")
    df_v2 = create_wind_features_v2(df)
    print(f"After V2 features: {len(df_v2.columns)} columns")
    print(f"New features added: {len(df_v2.columns) - len(df.columns)}")

    # 데이터 분할
    print("\n[2] Splitting data (70/15/15)...")
    train_df, val_df, test_df = train_val_test_split(df_v2)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 이전 결과와 비교를 위한 정보
    print("\n[Previous Results (Baseline)]")
    print("  Wind R²: 0.2530, RMSE: 26.28 MW")

    # V2 모델 학습
    print("\n[3] Training Wind Model V2...")
    model, metrics, feature_importance, feature_cols = train_wind_model_v2(
        train_df, val_df, test_df
    )

    # 결과 비교
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs V2")
    print("=" * 70)
    print(f"{'Metric':<10} {'Baseline':>12} {'V2':>12} {'Change':>12}")
    print("-" * 50)

    baseline = {'r2': 0.2530, 'rmse': 26.28, 'mae': 21.88}

    r2_change = metrics['r2'] - baseline['r2']
    rmse_change = metrics['rmse'] - baseline['rmse']
    mae_change = metrics['mae'] - baseline['mae']

    print(f"{'R²':<10} {baseline['r2']:>12.4f} {metrics['r2']:>12.4f} {r2_change:>+12.4f}")
    print(f"{'RMSE':<10} {baseline['rmse']:>12.2f} {metrics['rmse']:>12.2f} {rmse_change:>+12.2f}")
    print(f"{'MAE':<10} {baseline['mae']:>12.2f} {metrics['mae']:>12.2f} {mae_change:>+12.2f}")

    # 개선율
    r2_improvement = (metrics['r2'] - baseline['r2']) / abs(baseline['r2']) * 100
    rmse_improvement = (baseline['rmse'] - metrics['rmse']) / baseline['rmse'] * 100

    print(f"\nR² Improvement: {r2_improvement:+.1f}%")
    print(f"RMSE Improvement: {rmse_improvement:+.1f}%")

    # 모델 저장
    save_dir = MODELS_DIR / 'wind_v2'
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(save_dir / 'wind_lgbm_v2.lgb'))
    feature_importance.to_csv(save_dir / 'feature_importance_v2.csv', index=False)

    # 피처 목록 저장
    with open(save_dir / 'feature_cols.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    print(f"\nModel saved to: {save_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return model, metrics, feature_importance


if __name__ == "__main__":
    model, metrics, feature_importance = main()
