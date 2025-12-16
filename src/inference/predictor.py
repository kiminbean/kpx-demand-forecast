"""
Inference Predictor

학습된 모델을 사용한 예측 수행
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE, MODELS_DIR, data_config
from src.models.solar_lstm import create_solar_model, SolarLSTM
from src.models.wind_transformer import create_wind_model, WindTransformer
from src.features import (
    FeatureEngineer,
    DataScaler,
    create_solar_features,
    create_wind_features,
)


class Predictor:
    """
    발전량 예측기

    학습된 모델을 로드하여 예측을 수행합니다.

    Usage:
        predictor = Predictor(model_type='solar')
        predictor.load('models/solar_model/best_model.pt')
        predictions = predictor.predict(input_data)
    """

    def __init__(
        self,
        model_type: str = "solar",
        device: torch.device = DEVICE,
    ):
        """
        Args:
            model_type: 'solar' 또는 'wind'
            device: 연산 디바이스
        """
        self.model_type = model_type
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.scaler: Optional[DataScaler] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.feature_cols: Optional[List[str]] = None

        # 시퀀스 길이
        if model_type == "solar":
            self.seq_length = data_config.solar_seq_length
            self.target_col = "solar_mw"
        else:
            self.seq_length = data_config.wind_seq_length
            self.target_col = "wind_mw"

        self._is_loaded = False

    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ) -> "Predictor":
        """
        모델 및 스케일러 로드

        Args:
            checkpoint_path: 체크포인트 경로 (None이면 최신 모델 자동 선택)
            scaler_path: 스케일러 경로 (선택적)

        Returns:
            self
        """
        # 체크포인트 경로 결정
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        print(f"Loading model from {checkpoint_path}")

        # 체크포인트 로드 (weights_only=False: 자체 생성 체크포인트이므로 안전)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 모델 생성 및 가중치 로드
        # 체크포인트에서 input_size 추론
        state_dict = checkpoint["model_state_dict"]

        if self.model_type == "solar":
            # LSTM의 첫 번째 레이어에서 input_size 추론
            input_size = state_dict["lstm.weight_ih_l0"].shape[1]
            self.model = create_solar_model(input_size=input_size)
        else:
            # Transformer의 input_embedding에서 input_size 추론
            input_size = state_dict["input_embedding.weight"].shape[1]
            self.model = create_wind_model(input_size=input_size)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 스케일러 로드 (있는 경우)
        if scaler_path:
            import pickle
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self._is_loaded = True
        print(f"Model loaded successfully. Input size: {input_size}")

        return self

    def _find_best_checkpoint(self) -> Path:
        """최상의 체크포인트 자동 탐색"""
        model_dir = MODELS_DIR / f"{self.model_type}_model"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"모델 디렉토리가 없습니다: {model_dir}\n"
                "먼저 'python scripts/train.py' 실행하세요."
            )

        # val_loss가 가장 낮은 체크포인트 찾기
        checkpoints = list(model_dir.glob("model_epoch*.pt"))

        if not checkpoints:
            raise FileNotFoundError(f"체크포인트가 없습니다: {model_dir}")

        # 파일명에서 val_loss 추출하여 정렬
        def extract_val_loss(path: Path) -> float:
            name = path.stem
            try:
                loss_str = name.split("val_loss")[1].split("_")[0]
                return float(loss_str)
            except (IndexError, ValueError):
                return float("inf")

        best_checkpoint = min(checkpoints, key=extract_val_loss)
        return best_checkpoint

    def _preprocess(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str], DataScaler]:
        """
        입력 데이터 전처리

        Args:
            df: 입력 DataFrame (기상 + 발전량 데이터)

        Returns:
            (전처리된 배열, 피처 컬럼 리스트, 스케일러)
        """
        df = df.copy()

        # datetime 처리
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("datetime 인덱스 또는 datetime 컬럼이 필요합니다.")

        # Feature Engineering
        fe = FeatureEngineer(
            add_time_features=True,
            add_lag_features=True,
            add_rolling_features=True,
        )
        df_features = fe.fit_transform(df, target_col=self.target_col)

        # 모델별 특화 피처
        if self.model_type == "solar":
            df_features = create_solar_features(df_features)
        else:
            df_features = create_wind_features(df_features)

        # 수치형 컬럼만 선택
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

        # 타겟 컬럼을 첫 번째로
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        feature_cols = [self.target_col] + numeric_cols

        # 스케일링 (스케일러 저장)
        if self.scaler:
            scaler = self.scaler
            df_scaled = scaler.transform(df_features[feature_cols])
        else:
            scaler = DataScaler(method="standard")
            df_scaled = scaler.fit_transform(df_features[feature_cols])

        return df_scaled.values.astype(np.float32), feature_cols, scaler

    @torch.no_grad()
    def predict(
        self,
        df: pd.DataFrame,
        return_scaled: bool = False,
    ) -> np.ndarray:
        """
        예측 수행

        Args:
            df: 입력 DataFrame
            return_scaled: True면 스케일된 예측값 반환, False면 원본 스케일로 역변환

        Returns:
            예측값 배열
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. load() 먼저 호출하세요.")

        # 전처리
        data, feature_cols, scaler = self._preprocess(df)

        # 시퀀스 생성
        sequences = []
        for i in range(len(data) - self.seq_length + 1):
            sequences.append(data[i:i + self.seq_length])

        if not sequences:
            raise ValueError(
                f"데이터가 부족합니다. 최소 {self.seq_length}개의 연속 데이터가 필요합니다."
            )

        # 텐서 변환
        X = torch.FloatTensor(np.array(sequences)).to(self.device)

        # 예측
        self.model.eval()
        predictions = self.model(X).cpu().numpy().flatten()

        # 역변환 (원본 스케일로 복원)
        if not return_scaled and self.target_col in scaler.scalers:
            target_scaler = scaler.scalers[self.target_col]
            predictions = target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions

    @torch.no_grad()
    def predict_next(
        self,
        df: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """
        미래 시점 예측 (Autoregressive)

        Args:
            df: 최근 데이터 DataFrame
            horizon: 예측 기간 (시간)

        Returns:
            예측 결과 DataFrame
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        # 현재 시점 예측
        predictions = self.predict(df)

        # 마지막 시점부터 미래 예측
        last_datetime = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["datetime"]).iloc[-1]

        future_times = pd.date_range(
            start=last_datetime + timedelta(hours=1),
            periods=min(horizon, len(predictions)),
            freq="h",
        )

        result_df = pd.DataFrame({
            "datetime": future_times,
            f"predicted_{self.target_col}": predictions[:len(future_times)],
        })

        return result_df


class EnsemblePredictor:
    """
    앙상블 예측기

    태양광과 풍력 모델을 결합하여 총 발전량을 예측합니다.
    """

    def __init__(
        self,
        solar_weight: float = 0.5,
        wind_weight: float = 0.5,
        device: torch.device = DEVICE,
    ):
        """
        Args:
            solar_weight: 태양광 모델 가중치
            wind_weight: 풍력 모델 가중치
            device: 연산 디바이스
        """
        self.solar_predictor = Predictor(model_type="solar", device=device)
        self.wind_predictor = Predictor(model_type="wind", device=device)
        self.solar_weight = solar_weight
        self.wind_weight = wind_weight

    def load(
        self,
        solar_checkpoint: Optional[str] = None,
        wind_checkpoint: Optional[str] = None,
    ) -> "EnsemblePredictor":
        """모델 로드"""
        self.solar_predictor.load(solar_checkpoint)
        self.wind_predictor.load(wind_checkpoint)
        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        앙상블 예측

        Args:
            df: 입력 DataFrame

        Returns:
            예측 결과 딕셔너리
        """
        solar_preds = self.solar_predictor.predict(df)
        wind_preds = self.wind_predictor.predict(df)

        # 배열 길이 맞추기
        min_len = min(len(solar_preds), len(wind_preds))
        solar_preds = solar_preds[:min_len]
        wind_preds = wind_preds[:min_len]

        # 가중 평균
        total_preds = (
            self.solar_weight * solar_preds +
            self.wind_weight * wind_preds
        )

        return {
            "solar": solar_preds,
            "wind": wind_preds,
            "total": total_preds,
        }


def load_predictor(
    model_type: str = "solar",
    checkpoint_path: Optional[str] = None,
) -> Predictor:
    """
    Predictor 팩토리 함수

    Args:
        model_type: 'solar' 또는 'wind'
        checkpoint_path: 체크포인트 경로 (선택적)

    Returns:
        로드된 Predictor
    """
    predictor = Predictor(model_type=model_type)
    predictor.load(checkpoint_path)
    return predictor


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_DIR

    print("=" * 60)
    print("Predictor 테스트")
    print("=" * 60)

    # 테스트 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 최근 100시간 데이터로 테스트
        test_df = df.tail(100).copy()

        print(f"\nTest data: {len(test_df)} rows")

        # Solar Predictor 테스트
        try:
            print("\n[Solar Predictor]")
            solar_predictor = load_predictor("solar")

            predictions = solar_predictor.predict(test_df)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Prediction range: {predictions.min():.2f} ~ {predictions.max():.2f}")
        except Exception as e:
            print(f"Solar prediction failed: {e}")

        # Wind Predictor 테스트
        try:
            print("\n[Wind Predictor]")
            wind_predictor = load_predictor("wind")

            predictions = wind_predictor.predict(test_df)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Prediction range: {predictions.min():.2f} ~ {predictions.max():.2f}")
        except Exception as e:
            print(f"Wind prediction failed: {e}")

    else:
        print(f"테스트 데이터 없음: {data_path}")
