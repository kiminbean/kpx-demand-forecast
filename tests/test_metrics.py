"""
KPX 예측오차율 지표 테스트
=========================

손으로 계산한 값과 비교한다. 분모(설비용량), 10% 평가 필터, 정산 구간 경계(이하 포함)를 확인한다.
"""

import numpy as np
import pytest

from src.evaluation import (
    SETTLEMENT_TIERS_2021,
    SETTLEMENT_TIERS_2025,
    evaluate_generation,
    kpx_error_rate,
    settlement_price,
    skill_score,
)

CAP = 100.0


def test_error_rate_uses_capacity_and_10pct_filter():
    # 실측 0·5는 용량의 10% 미만이라 평가 제외, 10·50만 평가
    fer, mask = kpx_error_rate([0, 5, 10, 50], [1, 7, 13, 44], CAP)
    assert mask.tolist() == [False, False, True, True]
    np.testing.assert_allclose(fer, [3.0, 6.0])


@pytest.mark.parametrize(
    "tiers, fer, expected",
    [
        (SETTLEMENT_TIERS_2025, [3, 4, 5, 6, 7], [4, 4, 3, 3, 0]),
        (SETTLEMENT_TIERS_2021, [3, 6, 7, 8, 9], [4, 4, 3, 3, 0]),
    ],
    ids=["tiers-2025", "tiers-2021"],
)
def test_settlement_price_boundaries_inclusive(tiers, fer, expected):
    np.testing.assert_allclose(settlement_price(np.array(fer, float), tiers), expected)


def test_evaluate_generation_summary():
    r = evaluate_generation([0, 5, 10, 50], [1, 7, 13, 44], CAP)
    assert r["n_eval_hours"] == 2
    assert r["fer_mean_pct"] == pytest.approx(4.5)
    assert r["hit_4_pct"] == pytest.approx(50.0)
    assert r["hit_6_pct"] == pytest.approx(100.0)
    # 발전량 가중 단가: (10MWh×4원 + 50MWh×3원) / 60MWh
    assert r["settle_won_per_kwh"] == pytest.approx((10 * 4 + 50 * 3) / 60)
    assert r["settle_won"] == pytest.approx((10 * 4 + 50 * 3) * 1000)
    assert r["mae_mw"] == pytest.approx((1 + 2 + 3 + 6) / 4)


def test_negative_prediction_is_clipped():
    r = evaluate_generation([0, 20], [-5, 20], CAP)
    assert r["mae_mw"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "y, p, cap",
    [([1, 2], [1], CAP), ([1, np.nan], [1, 2], CAP), ([1, 2], [1, 2], 0.0)],
    ids=["length-mismatch", "nan-input", "zero-capacity"],
)
def test_invalid_input_fails_fast(y, p, cap):
    with pytest.raises(ValueError):
        evaluate_generation(y, p, cap)


def test_jfpf_scales_settlement_only():
    full = evaluate_generation([0, 5, 10, 50], [1, 7, 13, 44], CAP)
    half = evaluate_generation([0, 5, 10, 50], [1, 7, 13, 44], CAP, jfpf=0.5)
    assert half["settle_won_per_kwh"] == pytest.approx(full["settle_won_per_kwh"] * 0.5)
    assert half["fer_mean_pct"] == full["fer_mean_pct"]
    with pytest.raises(ValueError):
        evaluate_generation([0, 50], [0, 50], CAP, jfpf=0.0)


def test_skill_score_reference_points():
    y = np.array([0.0, 10.0, 20.0, 30.0])
    ref = np.array([5.0, 5.0, 25.0, 20.0])
    assert skill_score(y, y, ref) == pytest.approx(1.0)
    assert skill_score(y, ref, ref) == pytest.approx(0.0)
    # 기준 예측의 NaN 행은 제외
    assert skill_score(np.r_[y, 1.0], np.r_[y, 9.0], np.r_[ref, np.nan]) == pytest.approx(1.0)
