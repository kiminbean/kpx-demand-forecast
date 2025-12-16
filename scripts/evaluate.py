#!/usr/bin/env python3
"""
Model Evaluation Script

학습된 모델의 종합 평가 및 보고서 생성
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, DEVICE
from src.inference import load_predictor


def evaluate_model(model_type: str):
    """모델 종합 평가"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} Model")
    print(f"{'='*60}")

    # 데이터 로드
    data_path = PROCESSED_DATA_DIR / "merged_dataset.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return None

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 테스트 데이터 (마지막 15%)
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:].copy()
    print(f"Test data: {len(test_df)} samples")

    # Predictor 로드
    try:
        predictor = load_predictor(model_type)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # 예측 수행
    try:
        predictions = predictor.predict(test_df)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

    # 실제값과 비교
    target_col = 'solar_mw' if model_type == 'solar' else 'wind_mw'
    seq_length = predictor.seq_length

    actual = test_df[target_col].values[seq_length:]
    min_len = min(len(actual), len(predictions))
    actual = actual[:min_len]
    predictions = predictions[:min_len]

    # 메트릭 계산
    mse = np.mean((actual - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predictions))

    # MAPE (0 제외)
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
    else:
        mape = float('inf')

    # R²
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # 결과 출력
    print(f"\n[Test Metrics]")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    # 예측 범위
    print(f"\n[Prediction Statistics]")
    print(f"  Actual   - Min: {actual.min():.2f}, Max: {actual.max():.2f}, Mean: {actual.mean():.2f}")
    print(f"  Predicted - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")

    return {
        'model_type': model_type,
        'test_samples': len(actual),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'actual_mean': float(actual.mean()),
        'predicted_mean': float(predictions.mean()),
        'actual': actual,
        'predictions': predictions
    }


def create_evaluation_plots(solar_results, wind_results, output_dir: Path):
    """평가 시각화 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (results, name) in enumerate([(solar_results, 'Solar'), (wind_results, 'Wind')]):
        if results is None:
            continue

        actual = results['actual']
        pred = results['predictions']

        # 1. 시계열 비교
        ax = axes[idx, 0]
        ax.plot(actual[:200], label='Actual', alpha=0.7)
        ax.plot(pred[:200], label='Predicted', alpha=0.7)
        ax.set_title(f'{name} - Time Series Comparison (First 200)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('MW')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 산점도
        ax = axes[idx, 1]
        ax.scatter(actual, pred, alpha=0.3, s=10)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Perfect')
        ax.set_title(f'{name} - Actual vs Predicted (R²={results["r2"]:.3f})')
        ax.set_xlabel('Actual (MW)')
        ax.set_ylabel('Predicted (MW)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 잔차 분포
        ax = axes[idx, 2]
        residuals = actual - pred
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title(f'{name} - Residual Distribution (MAE={results["mae"]:.3f})')
        ax.set_xlabel('Residual (MW)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'model_evaluation.png'}")


def generate_report(solar_results, wind_results, output_dir: Path):
    """최종 평가 보고서 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'generated_at': datetime.now().isoformat(),
        'device': str(DEVICE),
        'models': {}
    }

    for results, name in [(solar_results, 'solar'), (wind_results, 'wind')]:
        if results:
            report['models'][name] = {
                'test_samples': results['test_samples'],
                'metrics': {
                    'mse': results['mse'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'mape': results['mape'],
                    'r2': results['r2']
                },
                'statistics': {
                    'actual_mean': results['actual_mean'],
                    'predicted_mean': results['predicted_mean']
                }
            }

    # JSON 저장
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Markdown 보고서
    md_report = f"""# J-REF Model Evaluation Report

Generated: {report['generated_at']}
Device: {report['device']}

## Summary

| Model | Test Samples | MSE | RMSE | MAE | MAPE (%) | R² |
|-------|-------------|-----|------|-----|----------|-----|
"""

    for name, data in report['models'].items():
        m = data['metrics']
        md_report += f"| {name.upper()} | {data['test_samples']} | {m['mse']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['mape']:.2f} | {m['r2']:.4f} |\n"

    md_report += """
## Model Details

### Solar Model (Bi-LSTM + Attention)
- Architecture: 2-layer Bi-LSTM with Self-Attention
- Sequence Length: 24 hours
- Features: Time cyclical encoding, lag features, rolling statistics

### Wind Model (Transformer Encoder)
- Architecture: Transformer Encoder with Positional Encoding
- Sequence Length: 48 hours
- Features: Time cyclical encoding, wind power density, wind direction encoding

## Data

- Training Period: 2024-01-01 ~ 2024-12-31
- Weather Data: KMA (기상청) ASOS data from 4 Jeju stations
- Generation Data: Physically-modeled synthetic data based on weather patterns

## Notes

1. **Solar Model**: Achieves reasonable R² with real weather data correlation
2. **Wind Model**: Lower performance due to simplified power curve model
3. Both models use early stopping to prevent overfitting

## Recommendations

1. Collect actual KPX generation data for improved accuracy
2. Fine-tune hyperparameters with Optuna
3. Consider ensemble methods for production deployment
"""

    md_path = output_dir / 'EVALUATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"Saved: {md_path}")

    return report


def main():
    print("="*60)
    print("J-REF Model Final Evaluation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Data: {PROCESSED_DATA_DIR / 'merged_dataset.csv'}")

    # 각 모델 평가
    solar_results = evaluate_model('solar')
    wind_results = evaluate_model('wind')

    # 시각화 생성
    output_dir = MODELS_DIR / 'evaluation'
    if solar_results or wind_results:
        create_evaluation_plots(solar_results, wind_results, output_dir)

    # 보고서 생성
    report = generate_report(solar_results, wind_results, output_dir)

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")

    return report


if __name__ == "__main__":
    main()
