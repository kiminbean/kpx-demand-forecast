#!/usr/bin/env python
"""
Prediction Script

학습된 모델로 발전량 예측 수행

Usage:
    # 태양광 예측
    python scripts/predict.py --model solar --input data/processed/merged_dataset.csv

    # 풍력 예측
    python scripts/predict.py --model wind --input data/processed/merged_dataset.csv

    # 앙상블 예측
    python scripts/predict.py --model ensemble --input data/processed/merged_dataset.csv

    # 특정 체크포인트 사용
    python scripts/predict.py --model solar --checkpoint models/solar_model/best.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.inference import Predictor, EnsemblePredictor, load_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Renewable energy generation prediction")
    parser.add_argument(
        "--model",
        type=str,
        choices=["solar", "wind", "ensemble"],
        default="solar",
        help="Model type",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "merged_dataset.csv"),
        help="Input data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (optional)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=100,
        help="Use last N rows for prediction",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon (hours)",
    )
    return parser.parse_args()


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """예측 성능 지표 계산"""
    # Flatten
    predictions = predictions.flatten()
    actuals = actuals.flatten()

    # 길이 맞추기
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    # 메트릭 계산
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # MAPE (0 제외)
    mask = actuals != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = 0.0

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
    }


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("J-REF Prediction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Last N rows: {args.last_n}")
    print("=" * 60)

    # 데이터 로드
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 입력 파일이 없습니다: {input_path}")
        return

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 최근 N개 데이터 사용
    test_df = df.tail(args.last_n).copy()
    print(f"\nTest data: {len(test_df)} rows")
    print(f"Period: {test_df['datetime'].min()} ~ {test_df['datetime'].max()}")

    # 예측기 로드
    if args.model == "ensemble":
        predictor = EnsemblePredictor()
        predictor.load()
        predictions = predictor.predict(test_df)

        print("\n" + "=" * 40)
        print("Ensemble Predictions")
        print("=" * 40)

        for model_name, preds in predictions.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Mean: {preds.mean():.2f}")
            print(f"  Std:  {preds.std():.2f}")
            print(f"  Min:  {preds.min():.2f}")
            print(f"  Max:  {preds.max():.2f}")

    else:
        predictor = load_predictor(args.model, args.checkpoint)
        predictions = predictor.predict(test_df)

        print("\n" + "=" * 40)
        print(f"{args.model.upper()} Predictions")
        print("=" * 40)
        print(f"Predictions: {len(predictions)}")
        print(f"Mean: {predictions.mean():.2f}")
        print(f"Std:  {predictions.std():.2f}")
        print(f"Min:  {predictions.min():.2f}")
        print(f"Max:  {predictions.max():.2f}")

        # 실제값과 비교 (가능한 경우)
        target_col = "solar_mw" if args.model == "solar" else "wind_mw"
        if target_col in test_df.columns:
            # 예측은 시퀀스 길이만큼 시작이 밀림
            seq_length = predictor.seq_length
            actuals = test_df[target_col].values[seq_length - 1:]

            if len(actuals) >= len(predictions):
                actuals = actuals[:len(predictions)]

                metrics = calculate_metrics(predictions, actuals)

                print("\n" + "=" * 40)
                print("Evaluation Metrics")
                print("=" * 40)
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAE:  {metrics['mae']:.4f}")
                print(f"MAPE: {metrics['mape']:.2f}%")
                print(f"R²:   {metrics['r2']:.4f}")

    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.model == "ensemble":
            result_df = pd.DataFrame(predictions)
        else:
            # datetime 정렬
            seq_length = predictor.seq_length
            prediction_times = test_df["datetime"].values[seq_length - 1:][:len(predictions)]

            result_df = pd.DataFrame({
                "datetime": prediction_times,
                f"predicted_{args.model}_mw": predictions,
            })

            # 실제값 추가
            if target_col in test_df.columns:
                result_df[f"actual_{args.model}_mw"] = actuals[:len(predictions)]

        result_df.to_csv(output_path, index=False)
        print(f"\n결과 저장: {output_path}")

    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
