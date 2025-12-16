"""
Ensemble Model

태양광 + 풍력 모델 앙상블
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.config import ensemble_config, DEVICE
from .solar_lstm import SolarLSTM
from .wind_transformer import WindTransformer


class EnsemblePredictor(nn.Module):
    """
    앙상블 예측기

    태양광(Solar)과 풍력(Wind) 모델의 예측을 통합합니다.

    Strategies:
        - weighted_average: 가중 평균
        - meta_learner: 메타 러너 (학습된 결합 가중치)
    """

    def __init__(
        self,
        solar_model: SolarLSTM,
        wind_model: WindTransformer,
        method: str = ensemble_config.method,
        solar_weight: float = ensemble_config.solar_weight,
        wind_weight: float = ensemble_config.wind_weight,
    ):
        super().__init__()

        self.solar_model = solar_model
        self.wind_model = wind_model
        self.method = method

        if method == "weighted_average":
            # 고정 가중치 사용
            self.register_buffer(
                "solar_weight", torch.tensor(solar_weight)
            )
            self.register_buffer(
                "wind_weight", torch.tensor(wind_weight)
            )

        elif method == "meta_learner":
            # 학습 가능한 가중치
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def forward(
        self,
        solar_input: torch.Tensor,
        wind_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        순전파

        Args:
            solar_input: 태양광 모델 입력 (batch, solar_seq_len, solar_features)
            wind_input: 풍력 모델 입력 (batch, wind_seq_len, wind_features)

        Returns:
            통합 예측값 (batch, 1)
        """
        # 개별 모델 예측
        solar_pred = self.solar_model(solar_input)  # (batch, 1)
        wind_pred = self.wind_model(wind_input)      # (batch, 1)

        if self.method == "weighted_average":
            # 가중 평균
            output = (
                self.solar_weight * solar_pred +
                self.wind_weight * wind_pred
            )

        elif self.method == "meta_learner":
            # 메타 러너로 가중치 학습
            stacked = torch.cat([solar_pred, wind_pred], dim=-1)  # (batch, 2)
            weights = self.meta_fc(stacked)  # (batch, 2)

            output = (
                weights[:, 0:1] * solar_pred +
                weights[:, 1:2] * wind_pred
            )

        return output

    def predict_individual(
        self,
        solar_input: torch.Tensor,
        wind_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        개별 모델 예측값 반환

        Args:
            solar_input: 태양광 모델 입력
            wind_input: 풍력 모델 입력

        Returns:
            개별 예측값 딕셔너리
        """
        self.eval()

        with torch.no_grad():
            solar_pred = self.solar_model(solar_input)
            wind_pred = self.wind_model(wind_input)
            ensemble_pred = self.forward(solar_input, wind_input)

        return {
            "solar": solar_pred,
            "wind": wind_pred,
            "ensemble": ensemble_pred,
        }

    def get_weights(self) -> Tuple[float, float]:
        """
        현재 앙상블 가중치 반환

        Returns:
            (solar_weight, wind_weight) 튜플
        """
        if self.method == "weighted_average":
            return (
                self.solar_weight.item(),
                self.wind_weight.item(),
            )
        else:
            # Meta learner의 경우 평균 가중치 반환
            # 실제로는 입력에 따라 동적으로 변함
            return (0.5, 0.5)


class SimpleEnsemble:
    """
    간단한 앙상블 (nn.Module 미사용)

    학습된 두 모델의 예측을 단순 결합합니다.
    """

    def __init__(
        self,
        solar_model_path: Optional[str] = None,
        wind_model_path: Optional[str] = None,
        solar_weight: float = 0.5,
        wind_weight: float = 0.5,
    ):
        self.solar_weight = solar_weight
        self.wind_weight = wind_weight

        self.solar_model = None
        self.wind_model = None

        if solar_model_path:
            self.load_solar_model(solar_model_path)
        if wind_model_path:
            self.load_wind_model(wind_model_path)

    def load_solar_model(self, path: str) -> None:
        """태양광 모델 로드"""
        self.solar_model = SolarLSTM()
        self.solar_model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.solar_model.to(DEVICE)
        self.solar_model.eval()

    def load_wind_model(self, path: str) -> None:
        """풍력 모델 로드"""
        self.wind_model = WindTransformer()
        self.wind_model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.wind_model.to(DEVICE)
        self.wind_model.eval()

    def predict(
        self,
        solar_input: torch.Tensor,
        wind_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        예측 수행

        Args:
            solar_input: 태양광 입력
            wind_input: 풍력 입력

        Returns:
            예측 결과 딕셔너리
        """
        solar_input = solar_input.to(DEVICE)
        wind_input = wind_input.to(DEVICE)

        with torch.no_grad():
            solar_pred = self.solar_model(solar_input) if self.solar_model else None
            wind_pred = self.wind_model(wind_input) if self.wind_model else None

        ensemble_pred = None
        if solar_pred is not None and wind_pred is not None:
            ensemble_pred = (
                self.solar_weight * solar_pred +
                self.wind_weight * wind_pred
            )

        return {
            "solar": solar_pred,
            "wind": wind_pred,
            "ensemble": ensemble_pred,
        }


if __name__ == "__main__":
    from .solar_lstm import create_solar_model
    from .wind_transformer import create_wind_model

    # 모델 생성
    solar = create_solar_model()
    wind = create_wind_model()
    ensemble = EnsemblePredictor(solar, wind)

    print(f"Ensemble method: {ensemble.method}")
    print(f"Weights: {ensemble.get_weights()}")

    # 더미 입력으로 테스트
    batch_size = 16
    solar_input = torch.randn(batch_size, 24, 10).to(DEVICE)
    wind_input = torch.randn(batch_size, 48, 12).to(DEVICE)

    output = ensemble(solar_input, wind_input)
    print(f"Ensemble output shape: {output.shape}")
