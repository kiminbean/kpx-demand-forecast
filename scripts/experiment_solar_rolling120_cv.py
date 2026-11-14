#!/usr/bin/env python3
"""Exploratory 2024-only CV: 120-day rolling training vs expanding history.

Protocol fixed before execution: island_mean, JMA MSM day2, existing six
rolling-origin folds and 384.87 MW selection denominator. Compare the same
SolarV2Model and causal D-1 10:00 training deadline, differing only in the
120-day training window. FER is pooled over identical eligible OOF hours.
No validation, test, 2025 data, post-processing, model saving or deployment.
The day2 issuance timestamp is a proxy, not verified availability evidence.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_solar_v2 import (  # noqa: E402
    CAPACITY_SELECTION, CV_FOLDS, KPX_RAW, TEST_START, _starts,
    fit_predict, load_kpx_hourly, load_nwp_weather, training_mask_for_forecast,
)
from src.evaluation import evaluate_generation  # noqa: E402
from src.features.solar_features_v2 import (  # noqa: E402
    build_solar_frame, create_island_mean_features,
)


def main() -> None:
    """Run paired causal folds, reporting bounds and exact comparison."""
    # End timestamps through TEST_START belong to intervals starting before it.
    gen = load_kpx_hourly(KPX_RAW).loc[lambda x: x.index <= TEST_START]
    wx = load_nwp_weather(2).loc[lambda x: x.index <= TEST_START]
    frame = build_solar_frame(wx, generation=gen)
    X = create_island_mean_features(frame)
    y = frame['generation_mw']
    starts = _starts(frame.index)
    outputs = {name: pd.Series(np.nan, index=frame.index) for name in ('expanding', 'rolling120')}
    for fs_text, fe_text in CV_FOLDS:
        fs, fe = pd.Timestamp(fs_text), pd.Timestamp(fe_text)
        assert fe <= TEST_START
        deadline = fs.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
        base = training_mask_for_forecast(frame.index, fs)
        ev = ((starts >= fs) & (starts < fe)).values
        assert frame.index[ev].max() <= TEST_START
        for name, tr in [('expanding', base), ('rolling120', base & (frame.index > deadline - pd.Timedelta(days=120)))]:
            assert frame.index[tr].max() <= deadline
            p = fit_predict(X, y, frame, tr, ev)
            outputs[name].loc[ev] = p
            fer = evaluate_generation(y[ev], p, CAPACITY_SELECTION)['fer_mean_pct']
            print(f'{fs_text} {name} train={tr.sum()} train_end={frame.index[tr].max()} eval={ev.sum()} FER={fer:.6f}', flush=True)
    masks = [s.notna().values for s in outputs.values()]
    assert np.array_equal(*masks)
    for name, oof in outputs.items():
        m = oof.notna().values
        result = evaluate_generation(y[m], oof[m], CAPACITY_SELECTION)
        print(f"{name} pooled FER={result['fer_mean_pct']:.9f}% MAE={result['mae_mw']:.9f} n_eval={result['n_eval_hours']} all_hours={m.sum()}", flush=True)


if __name__ == '__main__':
    main()
