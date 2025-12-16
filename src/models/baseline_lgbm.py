"""
LightGBM Baseline Model

Gemini 권장: 적은 데이터에서 비선형성을 잘 잡는 트리 기반 모델
Transformer보다 데이터 효율적이며, 베이스라인으로 적합
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import pickle

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LightGBMBaseline:
    """
    LightGBM 베이스라인 모델

    시계열 예측을 위한 Tabular 접근법.
    DL 모델 대비 빠른 학습과 해석 가능성 제공.

    Usage:
        model = LightGBMBaseline(target_col='solar_mw')
        model.fit(train_df, val_df)
        predictions = model.predict(test_df)
    """

    def __init__(
        self,
        target_col: str = 'solar_mw',
        model_type: str = 'solar',
        params: Optional[Dict] = None,
    ):
        """
        Args:
            target_col: 타겟 컬럼명
            model_type: 'solar' 또는 'wind'
            params: LightGBM 파라미터 (None이면 기본값 사용)
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.target_col = target_col
        self.model_type = model_type
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = []

        # 기본 하이퍼파라미터
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
        }

        # 학습 파라미터
        self.num_boost_round = 1000
        self.early_stopping_rounds = 50

        # 학습 이력
        self.history: Dict[str, List] = {
            'train_rmse': [],
            'val_rmse': [],
        }
        self.best_iteration = 0
        self.feature_importance: Optional[pd.DataFrame] = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        피처 준비 (수치형만 선택, 누수 변수 제외)

        Data Leakage Prevention:
        - solar_cf, wind_cf: 타겟에서 직접 파생
        - total_mw: 타겟들의 합
        - *_mw_lag_*, *_mw_ma_*, etc.: 타겟의 지연/롤링 특성
        """
        # 수치형 컬럼만
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 누수 변수 목록 (Gemini 권장)
        leakage_vars = [
            'solar_cf', 'wind_cf', 'total_mw',  # 타겟에서 직접 파생
            'solar_mw', 'wind_mw',  # 타겟 자체
        ]

        # 타겟 기반 지연/롤링 특성 (누수)
        target_derived = [col for col in numeric_cols if
                          col.startswith('solar_mw_') or col.startswith('wind_mw_')]

        # 제외 목록
        exclude_cols = set(leakage_vars + target_derived + [self.target_col])

        # 안전한 피처만 선택
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # 무한값 처리
        df_clean = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # 결측치 처리 (중앙값)
        df_clean = df_clean.fillna(df_clean.median())

        return df_clean, feature_cols

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: int = 100,
    ) -> 'LightGBMBaseline':
        """
        모델 학습

        Args:
            train_df: 학습 데이터 (특성 + 타겟 포함)
            val_df: 검증 데이터 (optional)
            verbose: 출력 주기

        Returns:
            self
        """
        # 피처 준비
        X_train, self.feature_cols = self._prepare_features(train_df)
        y_train = train_df[self.target_col].values

        # LightGBM Dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if val_df is not None:
            X_val, _ = self._prepare_features(val_df)
            X_val = X_val[self.feature_cols]  # 동일 컬럼 순서 보장
            y_val = val_df[self.target_col].values
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # 콜백
        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=verbose > 0),
            lgb.log_evaluation(verbose),
        ]

        # 학습
        print(f"Training LightGBM ({self.model_type})...")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Train samples: {len(X_train)}")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration

        # Feature Importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\nBest iteration: {self.best_iteration}")
        print(f"\nTop 10 Features:")
        print(self.feature_importance.head(10).to_string(index=False))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        예측 수행

        Args:
            df: 입력 DataFrame

        Returns:
            예측값 배열
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X, _ = self._prepare_features(df)
        X = X[self.feature_cols]  # 학습 시 피처 순서 유지

        predictions = self.model.predict(X, num_iteration=self.best_iteration)

        # 음수 방지 (발전량은 0 이상)
        predictions = np.clip(predictions, 0, None)

        return predictions

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        모델 평가

        Args:
            df: 평가 데이터

        Returns:
            평가 메트릭 딕셔너리
        """
        predictions = self.predict(df)
        actual = df[self.target_col].values

        # 메트릭 계산
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)

        # MAPE (0 제외)
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
        else:
            mape = float('inf')

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
        }

        return metrics

    def save(self, path: Path):
        """모델 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # LightGBM 모델 저장
        self.model.save_model(str(path.with_suffix('.lgb')))

        # 메타데이터 저장
        metadata = {
            'target_col': self.target_col,
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'params': self.params,
            'best_iteration': self.best_iteration,
        }

        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Feature Importance 저장
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                path.with_suffix('.importance.csv'), index=False
            )

        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> 'LightGBMBaseline':
        """모델 로드"""
        path = Path(path)

        # 메타데이터 로드
        with open(path.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)

        model = cls(
            target_col=metadata['target_col'],
            model_type=metadata['model_type'],
            params=metadata['params'],
        )

        model.feature_cols = metadata['feature_cols']
        model.best_iteration = metadata['best_iteration']

        # LightGBM 모델 로드
        model.model = lgb.Booster(model_file=str(path.with_suffix('.lgb')))

        return model


def train_lgbm_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = 'solar',
    save_dir: Optional[Path] = None,
) -> Tuple[LightGBMBaseline, Dict[str, float]]:
    """
    LightGBM 베이스라인 학습 및 평가

    Args:
        train_df: 학습 데이터
        val_df: 검증 데이터
        test_df: 테스트 데이터
        model_type: 'solar' 또는 'wind'
        save_dir: 저장 디렉토리

    Returns:
        (학습된 모델, 테스트 메트릭)
    """
    target_col = 'solar_mw' if model_type == 'solar' else 'wind_mw'

    # 모델 생성
    model = LightGBMBaseline(
        target_col=target_col,
        model_type=model_type,
    )

    # 학습
    model.fit(train_df, val_df)

    # 평가
    print("\n" + "=" * 50)
    print(f"Test Evaluation ({model_type.upper()})")
    print("=" * 50)

    metrics = model.evaluate(test_df)

    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")

    # 저장
    if save_dir:
        save_dir = Path(save_dir)
        model.save(save_dir / f'{model_type}_lgbm_baseline')

    return model, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.config import PROCESSED_DATA_DIR, MODELS_DIR
    from src.features import (
        FeatureEngineer,
        create_solar_features,
        create_wind_features,
        train_val_test_split,
    )

    print("=" * 60)
    print("LightGBM Baseline Training")
    print("=" * 60)

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded data: {len(df)} rows")

    # Feature Engineering
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df, target_col='solar_mw')
    print(f"Features added: {len(df_features.columns)} columns")

    # Solar features
    df_solar = create_solar_features(df_features)

    # Wind features
    df_wind = create_wind_features(df_features)

    # Train/Val/Test Split
    print("\n[Data Split]")
    train_df, val_df, test_df = train_val_test_split(df_solar)

    # Solar Model
    print("\n" + "=" * 60)
    print("Training Solar LightGBM Baseline")
    print("=" * 60)
    solar_model, solar_metrics = train_lgbm_baseline(
        train_df, val_df, test_df,
        model_type='solar',
        save_dir=MODELS_DIR / 'baseline'
    )

    # Wind Model
    train_wind, val_wind, test_wind = train_val_test_split(df_wind)

    print("\n" + "=" * 60)
    print("Training Wind LightGBM Baseline")
    print("=" * 60)
    wind_model, wind_metrics = train_lgbm_baseline(
        train_wind, val_wind, test_wind,
        model_type='wind',
        save_dir=MODELS_DIR / 'baseline'
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary: LightGBM Baseline Results")
    print("=" * 60)
    print(f"\nSolar Model:")
    print(f"  R²: {solar_metrics['r2']:.4f}, RMSE: {solar_metrics['rmse']:.4f}")
    print(f"\nWind Model:")
    print(f"  R²: {wind_metrics['r2']:.4f}, RMSE: {wind_metrics['rmse']:.4f}")
