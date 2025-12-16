"""
PyTorch Dataset Classes

시계열 예측을 위한 Dataset 클래스 정의
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict


class TimeSeriesDataset(Dataset):
    """
    시계열 예측용 Dataset

    시퀀스 기반으로 입력(X)과 타겟(y)을 생성합니다.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        forecast_horizon: int = 1,
        target_col_idx: int = 0,
    ):
        """
        Args:
            data: 입력 배열 (samples, features)
            seq_length: 입력 시퀀스 길이
            forecast_horizon: 예측 기간
            target_col_idx: 타겟 컬럼 인덱스
        """
        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.target_col_idx = target_col_idx

        # 유효한 샘플 수 계산
        self.n_samples = len(data) - seq_length - forecast_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"데이터 길이({len(data)})가 seq_length({seq_length}) + "
                f"forecast_horizon({forecast_horizon})보다 커야 합니다."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: 입력 시퀀스 (seq_length, features)
            y: 타겟 값 (forecast_horizon,) 또는 (1,)
        """
        # 입력 시퀀스
        X = self.data[idx : idx + self.seq_length]

        # 타겟 (단일 값 또는 시퀀스)
        if self.forecast_horizon == 1:
            y = self.data[idx + self.seq_length, self.target_col_idx]
        else:
            y = self.data[
                idx + self.seq_length : idx + self.seq_length + self.forecast_horizon,
                self.target_col_idx,
            ]

        return (
            torch.FloatTensor(X),
            torch.FloatTensor([y]) if self.forecast_horizon == 1 else torch.FloatTensor(y),
        )


class MultiStepDataset(Dataset):
    """
    다중 스텝 예측용 Dataset

    여러 시점을 동시에 예측합니다.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        forecast_steps: List[int],
        target_col_idx: int = 0,
    ):
        """
        Args:
            data: 입력 배열 (samples, features)
            seq_length: 입력 시퀀스 길이
            forecast_steps: 예측할 시점 리스트 (예: [1, 6, 12, 24])
            target_col_idx: 타겟 컬럼 인덱스
        """
        self.data = data
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps
        self.target_col_idx = target_col_idx

        max_step = max(forecast_steps)
        self.n_samples = len(data) - seq_length - max_step

        if self.n_samples <= 0:
            raise ValueError("데이터 길이가 부족합니다.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.data[idx : idx + self.seq_length]
        y = np.array([
            self.data[idx + self.seq_length + step - 1, self.target_col_idx]
            for step in self.forecast_steps
        ])

        return torch.FloatTensor(X), torch.FloatTensor(y)


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    seq_length: int = 24,
    forecast_horizon: int = 1,
    target_col_idx: int = 0,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    DataLoader 생성 유틸리티

    Args:
        train_data: 학습 데이터
        val_data: 검증 데이터
        test_data: 테스트 데이터 (선택적)
        seq_length: 입력 시퀀스 길이
        forecast_horizon: 예측 기간
        target_col_idx: 타겟 컬럼 인덱스
        batch_size: 배치 크기
        num_workers: DataLoader 워커 수

    Returns:
        DataLoader 딕셔너리 {'train': ..., 'val': ..., 'test': ...}
    """
    train_dataset = TimeSeriesDataset(
        train_data, seq_length, forecast_horizon, target_col_idx
    )
    val_dataset = TimeSeriesDataset(
        val_data, seq_length, forecast_horizon, target_col_idx
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    if test_data is not None:
        test_dataset = TimeSeriesDataset(
            test_data, seq_length, forecast_horizon, target_col_idx
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    if test_data is not None:
        print(f"Test samples: {len(dataloaders['test'].dataset):,}")

    return dataloaders


def prepare_data_for_training(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    DataFrame을 학습용 배열로 변환

    Args:
        df: 입력 DataFrame
        target_col: 타겟 컬럼명
        feature_cols: 피처 컬럼 리스트 (None이면 숫자형 컬럼 전체)
        train_ratio: 학습 비율
        val_ratio: 검증 비율

    Returns:
        (train_data, val_data, test_data, target_col_idx) 튜플
    """
    # 피처 컬럼 선택
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 타겟 컬럼이 피처에 포함되어 있는지 확인
    if target_col not in feature_cols:
        feature_cols = [target_col] + feature_cols

    # 타겟 컬럼 인덱스
    target_col_idx = feature_cols.index(target_col)

    # 배열 변환
    data = df[feature_cols].values.astype(np.float32)

    # 시간 순서대로 분할
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Data shape: {data.shape}")
    print(f"Target column: {target_col} (index: {target_col_idx})")
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    return train_data, val_data, test_data, target_col_idx


if __name__ == "__main__":
    # 테스트
    import numpy as np

    # 더미 데이터
    n_samples = 1000
    n_features = 10
    data = np.random.randn(n_samples, n_features).astype(np.float32)

    # Dataset 테스트
    dataset = TimeSeriesDataset(data, seq_length=24, forecast_horizon=1, target_col_idx=0)
    print(f"Dataset size: {len(dataset)}")

    X, y = dataset[0]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # DataLoader 테스트
    train_data = data[:700]
    val_data = data[700:850]
    test_data = data[850:]

    dataloaders = create_dataloaders(
        train_data, val_data, test_data,
        seq_length=24, batch_size=32
    )

    for batch_X, batch_y in dataloaders["train"]:
        print(f"Batch X shape: {batch_X.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        break
