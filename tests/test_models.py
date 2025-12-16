"""
Model Unit Tests

모델 아키텍처 및 추론 테스트
"""

import pytest
import torch

from src.config import DEVICE
from src.models.solar_lstm import SolarLSTM, create_solar_model
from src.models.wind_transformer import WindTransformer, create_wind_model
from src.models.ensemble import EnsemblePredictor


class TestSolarLSTM:
    """Solar LSTM 모델 테스트"""

    def test_model_creation(self):
        """모델 생성 테스트"""
        model = create_solar_model()
        assert model is not None
        assert isinstance(model, SolarLSTM)

    def test_forward_pass(self):
        """순전파 테스트"""
        model = create_solar_model()
        batch_size, seq_len, input_size = 16, 24, 10

        x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_attention_weights(self):
        """어텐션 가중치 테스트"""
        model = create_solar_model()
        batch_size, seq_len, input_size = 16, 24, 10

        x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
        output, attention = model(x, return_attention=True)

        assert attention.shape == (batch_size, seq_len)
        # 어텐션 가중치 합이 1에 가까운지 확인
        assert torch.allclose(attention.sum(dim=1), torch.ones(batch_size).to(DEVICE), atol=1e-5)


class TestWindTransformer:
    """Wind Transformer 모델 테스트"""

    def test_model_creation(self):
        """모델 생성 테스트"""
        model = create_wind_model()
        assert model is not None
        assert isinstance(model, WindTransformer)

    def test_forward_pass(self):
        """순전파 테스트"""
        model = create_wind_model()
        batch_size, seq_len, input_size = 16, 48, 12

        x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_variable_sequence_length(self):
        """가변 시퀀스 길이 테스트"""
        model = create_wind_model()
        batch_size, input_size = 8, 12

        for seq_len in [24, 48, 72]:
            x = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
            output = model(x)
            assert output.shape == (batch_size, 1)


class TestEnsemblePredictor:
    """앙상블 모델 테스트"""

    def test_weighted_average(self):
        """가중 평균 앙상블 테스트"""
        solar = create_solar_model()
        wind = create_wind_model()
        ensemble = EnsemblePredictor(solar, wind, method="weighted_average")

        batch_size = 8
        solar_input = torch.randn(batch_size, 24, 10).to(DEVICE)
        wind_input = torch.randn(batch_size, 48, 12).to(DEVICE)

        output = ensemble(solar_input, wind_input)
        assert output.shape == (batch_size, 1)

    def test_individual_predictions(self):
        """개별 예측 테스트"""
        solar = create_solar_model()
        wind = create_wind_model()
        ensemble = EnsemblePredictor(solar, wind)

        batch_size = 8
        solar_input = torch.randn(batch_size, 24, 10).to(DEVICE)
        wind_input = torch.randn(batch_size, 48, 12).to(DEVICE)

        predictions = ensemble.predict_individual(solar_input, wind_input)

        assert "solar" in predictions
        assert "wind" in predictions
        assert "ensemble" in predictions
        assert predictions["solar"].shape == (batch_size, 1)
        assert predictions["wind"].shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
