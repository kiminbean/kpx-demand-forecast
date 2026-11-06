"""
Models Module

딥러닝 모델 아키텍처 정의
"""

import importlib

# torch 모델은 처음 접근할 때 불러온다. LightGBM 전용 모듈(solar_v2 등)을 torch 없이 import하기 위함
_LAZY_EXPORTS = {
    "SolarLSTM": ".solar_lstm",
    "WindTransformer": ".wind_transformer",
    "EnsemblePredictor": ".ensemble",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
