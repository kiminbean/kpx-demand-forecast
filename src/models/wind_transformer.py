"""
Wind Transformer Model

풍력 발전량 예측을 위한 Transformer 기반 모델
"""

import math
from typing import Optional
import torch
import torch.nn as nn

from src.config import wind_model_config, DEVICE


class PositionalEncoding(nn.Module):
    """
    위치 인코딩 (Positional Encoding)

    Transformer의 시퀀스 위치 정보를 인코딩합니다.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩 계산
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 (seq_len, batch, d_model)

        Returns:
            위치 인코딩이 추가된 텐서
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class WindTransformer(nn.Module):
    """
    풍력 발전량 예측 Transformer 모델

    풍속, 풍향, 기압 등의 기상 변수로부터 풍력 발전량을 예측합니다.
    Temporal Fusion Transformer에서 영감을 받은 구조입니다.

    Architecture:
        Input Embedding → Positional Encoding → Transformer Encoder
        → Global Average Pooling → FC → Output
    """

    def __init__(
        self,
        input_size: int = wind_model_config.input_size,
        d_model: int = wind_model_config.d_model,
        nhead: int = wind_model_config.nhead,
        num_encoder_layers: int = wind_model_config.num_encoder_layers,
        dim_feedforward: int = wind_model_config.dim_feedforward,
        dropout: float = wind_model_config.dropout,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model

        # 입력 임베딩
        self.input_embedding = nn.Linear(input_size, d_model)

        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq, batch, feature) 형식 사용
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, input_size)
            src_mask: 소스 마스크 (선택적)

        Returns:
            output: 예측값 (batch, 1)
        """
        # 입력 임베딩
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        # (batch, seq_len, d_model)

        # (seq_len, batch, d_model) 형식으로 변환
        x = x.permute(1, 0, 2)

        # 위치 인코딩
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, mask=src_mask)
        # (seq_len, batch, d_model)

        # Global Average Pooling
        x = x.mean(dim=0)  # (batch, d_model)

        # 출력 레이어
        output = self.fc(x)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Causal Mask 생성

        미래 시점의 정보를 참조하지 못하도록 하는 마스크를 생성합니다.

        Args:
            sz: 시퀀스 길이

        Returns:
            마스크 텐서
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

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


def create_wind_model(
    input_size: Optional[int] = None,
    **kwargs,
) -> WindTransformer:
    """
    Wind 모델 팩토리 함수

    Args:
        input_size: 입력 피처 수 (기본값: config에서 로드)
        **kwargs: 추가 모델 파라미터

    Returns:
        초기화된 WindTransformer 모델
    """
    if input_size is None:
        input_size = wind_model_config.input_size

    model = WindTransformer(input_size=input_size, **kwargs)
    return model.to(DEVICE)


if __name__ == "__main__":
    # 모델 테스트
    model = create_wind_model()
    print(f"Model device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 입력으로 테스트
    batch_size, seq_len, input_size = 32, 48, 12
    dummy_input = torch.randn(batch_size, seq_len, input_size).to(DEVICE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
