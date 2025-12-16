"""
Solar LSTM Model

태양광 발전량 예측을 위한 Bi-LSTM + Attention 모델
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from src.config import solar_model_config, DEVICE


class Attention(nn.Module):
    """
    Self-Attention 메커니즘

    시퀀스 내에서 중요한 시점에 가중치를 부여합니다.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: LSTM 출력 (batch, seq_len, hidden_size)

        Returns:
            context: 어텐션 적용된 컨텍스트 벡터 (batch, hidden_size)
            weights: 어텐션 가중치 (batch, seq_len)
        """
        # (batch, seq_len, 1)
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # (batch, hidden_size)
        context = torch.sum(attention_weights * lstm_output, dim=1)

        return context, attention_weights.squeeze(-1)


class SolarLSTM(nn.Module):
    """
    태양광 발전량 예측 모델

    Bidirectional LSTM + Self-Attention 구조로
    일사량, 운량, 기온 등의 기상 변수로부터 태양광 발전량을 예측합니다.

    Architecture:
        Input → Bi-LSTM → Attention → FC → Output
    """

    def __init__(
        self,
        input_size: int = solar_model_config.input_size,
        hidden_size: int = solar_model_config.hidden_size,
        num_layers: int = solar_model_config.num_layers,
        dropout: float = solar_model_config.dropout,
        bidirectional: bool = solar_model_config.bidirectional,
        use_attention: bool = solar_model_config.attention,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention 레이어
        lstm_output_size = hidden_size * self.num_directions
        if use_attention:
            self.attention = Attention(lstm_output_size)

        # Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, input_size)
            return_attention: 어텐션 가중치 반환 여부

        Returns:
            output: 예측값 (batch, 1)
            attention_weights: (선택적) 어텐션 가중치
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)

        if self.use_attention:
            # Attention 적용
            context, attention_weights = self.attention(lstm_out)
        else:
            # 마지막 시점의 출력 사용
            if self.bidirectional:
                # Forward와 Backward의 마지막 hidden state 연결
                context = torch.cat([
                    lstm_out[:, -1, :self.hidden_size],
                    lstm_out[:, 0, self.hidden_size:]
                ], dim=1)
            else:
                context = lstm_out[:, -1, :]
            attention_weights = None

        # FC 레이어
        output = self.fc(context)

        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output

    def predict(
        self,
        x: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        추론 모드 예측

        Args:
            x: 입력 텐서
            device: 연산 디바이스

        Returns:
            예측값 텐서
        """
        device = device or DEVICE
        self.eval()
        self.to(device)

        with torch.no_grad():
            x = x.to(device)
            output = self.forward(x)

        return output


def create_solar_model(
    input_size: Optional[int] = None,
    **kwargs,
) -> SolarLSTM:
    """
    Solar 모델 팩토리 함수

    Args:
        input_size: 입력 피처 수 (기본값: config에서 로드)
        **kwargs: 추가 모델 파라미터

    Returns:
        초기화된 SolarLSTM 모델
    """
    if input_size is None:
        input_size = solar_model_config.input_size

    model = SolarLSTM(input_size=input_size, **kwargs)
    return model.to(DEVICE)


if __name__ == "__main__":
    # 모델 테스트
    model = create_solar_model()
    print(f"Model device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 입력으로 테스트
    batch_size, seq_len, input_size = 32, 24, 10
    dummy_input = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
