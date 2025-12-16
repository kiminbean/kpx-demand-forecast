#!/usr/bin/env python
"""
Training Script

태양광/풍력 발전량 예측 모델 학습 스크립트

Usage:
    # 태양광 모델 학습
    python scripts/train.py --model solar --epochs 100

    # 풍력 모델 학습
    python scripts/train.py --model wind --epochs 100

    # 모든 모델 학습
    python scripts/train.py --model all

    # 샘플 데이터로 빠른 테스트
    python scripts/train.py --model solar --quick
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    data_config,
    training_config,
    set_seed,
    setup_directories,
)
from src.features import (
    FeatureEngineer,
    DataScaler,
    create_solar_features,
    create_wind_features,
    train_val_test_split,
)
from src.training import (
    create_dataloaders,
    prepare_data_for_training,
    Trainer,
    create_optimizer,
    create_scheduler,
)
from src.models.solar_lstm import create_solar_model
from src.models.wind_transformer import create_wind_model


def load_and_prepare_data(
    model_type: str = "solar",
) -> tuple:
    """
    데이터 로드 및 전처리

    Args:
        model_type: 'solar' 또는 'wind'

    Returns:
        (train_data, val_data, test_data, scaler, feature_cols, target_col_idx)
    """
    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {data_path}\n"
            "먼저 'python scripts/download_data.py --sample' 실행하세요."
        )

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    # 타겟 컬럼 선택
    if model_type == "solar":
        target_col = "solar_mw"
        seq_length = data_config.solar_seq_length
    else:
        target_col = "wind_mw"
        seq_length = data_config.wind_seq_length

    print(f"Target: {target_col}")
    print(f"Sequence length: {seq_length}")

    # Feature Engineering
    fe = FeatureEngineer(
        add_time_features=True,
        add_lag_features=True,
        add_rolling_features=True,
    )
    df_features = fe.fit_transform(df, target_col=target_col)

    # 모델별 특화 피처 추가
    if model_type == "solar":
        df_features = create_solar_features(df_features)
    else:
        df_features = create_wind_features(df_features)

    # 결측치 제거
    df_features = df_features.dropna()

    print(f"Features: {len(df_features.columns)} columns")
    print(f"Samples: {len(df_features):,}")

    # 수치형 컬럼만 선택
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    # 타겟 컬럼을 첫 번째로 이동
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    feature_cols = [target_col] + numeric_cols

    # 데이터 분할
    train_df, val_df, test_df = train_val_test_split(
        df_features[feature_cols],
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
    )

    # 스케일링
    scaler = DataScaler(method=data_config.scaling_method)
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # numpy 배열로 변환
    train_data = train_scaled.values.astype(np.float32)
    val_data = val_scaled.values.astype(np.float32)
    test_data = test_scaled.values.astype(np.float32)

    target_col_idx = 0  # 타겟 컬럼이 첫 번째

    return train_data, val_data, test_data, scaler, feature_cols, target_col_idx


def train_model(
    model_type: str = "solar",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    quick: bool = False,
) -> dict:
    """
    모델 학습

    Args:
        model_type: 'solar' 또는 'wind'
        epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        quick: 빠른 테스트 모드

    Returns:
        학습 결과 딕셔너리
    """
    print("\n" + "=" * 60)
    print(f"Training {model_type.upper()} Model")
    print("=" * 60)

    # 시드 설정
    set_seed(training_config.seed)

    # 데이터 준비
    train_data, val_data, test_data, scaler, feature_cols, target_col_idx = (
        load_and_prepare_data(model_type)
    )

    # 시퀀스 길이
    if model_type == "solar":
        seq_length = data_config.solar_seq_length
    else:
        seq_length = data_config.wind_seq_length

    # Quick 모드: 데이터 축소
    if quick:
        train_data = train_data[:500]
        val_data = val_data[:200]
        test_data = test_data[:200]
        epochs = 5
        print("\n[Quick Mode] Data and epochs reduced for testing")

    # DataLoader 생성
    dataloaders = create_dataloaders(
        train_data,
        val_data,
        test_data,
        seq_length=seq_length,
        forecast_horizon=1,
        target_col_idx=target_col_idx,
        batch_size=batch_size,
    )

    # 입력 크기 (타겟 포함 전체 피처 수)
    input_size = train_data.shape[1]
    print(f"Input size: {input_size}")

    # 모델 생성
    if model_type == "solar":
        model = create_solar_model(input_size=input_size)
    else:
        model = create_wind_model(input_size=input_size)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 옵티마이저 & 스케줄러
    optimizer = create_optimizer(
        model,
        lr=learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=training_config.scheduler,
        epochs=epochs,
    )

    # 손실 함수
    criterion = nn.MSELoss()

    # Trainer 생성
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        model_name=f"{model_type}_model",
        gradient_clip_val=training_config.gradient_clip_val,
    )

    # 학습
    history = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        epochs=epochs,
        early_stopping_patience=training_config.early_stopping_patience,
        early_stopping_min_delta=training_config.early_stopping_min_delta,
        save_top_k=training_config.save_top_k,
        verbose=True,
    )

    # 테스트 평가
    print("\n" + "=" * 40)
    print("Test Set Evaluation")
    print("=" * 40)
    test_metrics = trainer.evaluate(dataloaders["test"])

    return {
        "model_type": model_type,
        "history": history,
        "test_metrics": test_metrics,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="Train renewable energy prediction models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["solar", "wind", "all"],
        default="solar",
        help="Model type to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_config.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_config.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=training_config.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced data",
    )
    args = parser.parse_args()

    # 디렉토리 생성
    setup_directories()

    print("\n" + "=" * 60)
    print("J-REF Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    results = {}

    if args.model in ["solar", "all"]:
        results["solar"] = train_model(
            model_type="solar",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            quick=args.quick,
        )

    if args.model in ["wind", "all"]:
        results["wind"] = train_model(
            model_type="wind",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            quick=args.quick,
        )

    # 결과 요약
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Model:")
        test_metrics = result["test_metrics"]
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  R²:   {test_metrics['r2']:.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Models saved to: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
