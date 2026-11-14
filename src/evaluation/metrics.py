"""
발전량 예측 평가 지표
=====================

태양광·풍력 발전량 예측은 MAPE가 아니라 KPX 재생에너지 발전량 예측제도의 예측오차율(FER)로 평가한다.

- FER_t = |예측_t − 실측_t| / 설비용량 × 100 (%)  — 분모가 실측값이 아니라 설비용량
- 평가 대상: 실측 발전량이 설비용량의 10% 이상인 시간만 (그 밖의 시간은 정산·평가 제외)
- 정산: 시간별 FER 구간에 따라 실측 발전량(kWh)에 단가를 곱함
  - 2025년 개정 후(육지 현행): ≤4% → 4원/kWh, 4~6% → 3원/kWh, 6% 초과 → 0원
  - 2021~2025년 개정 전: ≤6% → 4원/kWh, 6~8% → 3원/kWh, 8% 초과 → 0원
  - 제주: 2024-06-01 입찰제도 시행 후 1년간 개정 전 구간 × 50%(JFPF 0.5)만 지급, 2025-06-01 예측제도 종료.
    2025년 구간은 제주에 적용된 적이 없으므로 제주 모델에는 '육지 기준 비교값'으로만 쓴다.
- 근거: 전력시장운영규칙 제14.1.1조 제12호, 별표2 Ⅴ (KPX), 제24-6차 개정 공지 (2025-02-11)

MAPE를 쓰지 않는 이유: 해 뜰 무렵·해 질 무렵처럼 실측이 0에 가까운 시간에 분모가 작아져 수백 %로 폭주한다.
수요 예측은 실측(부하)이 0에 가까워지지 않으므로 MAPE가 표준이다.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

KPX_MIN_CAPACITY_FACTOR = 0.10

# (FER 상한 %, 정산단가 원/kWh) — 상한 이하이면 해당 단가
SETTLEMENT_TIERS_2025: Tuple[Tuple[float, float], ...] = ((4.0, 4.0), (6.0, 3.0))
SETTLEMENT_TIERS_2021: Tuple[Tuple[float, float], ...] = ((6.0, 4.0), (8.0, 3.0))

# 제주 전환기(2024-06-01 ~ 2025-05-31) 예측제도 정산 계수 JFPF (육지 1.0)
JEJU_TRANSITION_JFPF = 0.5

MWH_TO_KWH = 1000.0


def _as_arrays(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true와 y_pred 길이가 다릅니다: {y.shape} vs {p.shape}")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("y_true와 y_pred에 NaN/inf가 있습니다")
    return y, p


def kpx_error_rate(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    capacity_mw: float,
    min_capacity_factor: float = KPX_MIN_CAPACITY_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """시간별 KPX 예측오차율(%)과 평가 대상 마스크를 반환한다.

    Args:
        y_true: 실측 시간 발전량 (MWh = 평균 MW)
        y_pred: 예측 시간 발전량
        capacity_mw: 실측 시계열에 대응하는 설비용량 (출력제어·정지 설비 제외 실효 용량)
        min_capacity_factor: 평가 대상 최소 이용률 (KPX 10%)

    Returns:
        (fer, mask): fer는 평가 대상 시간의 오차율(%), mask는 전체 길이의 평가 대상 여부
    """
    if not capacity_mw > 0:
        raise ValueError(f"capacity_mw는 양수여야 합니다: {capacity_mw}")
    y, p = _as_arrays(y_true, y_pred)
    mask = y >= min_capacity_factor * capacity_mw
    fer = np.abs(p[mask] - y[mask]) / capacity_mw * 100.0
    return fer, mask


def settlement_price(fer: np.ndarray, tiers: Sequence[Tuple[float, float]] = SETTLEMENT_TIERS_2025) -> np.ndarray:
    """시간별 FER(%) → 정산단가(원/kWh). 구간 상한은 이하(≤) 기준."""
    price = np.zeros_like(np.asarray(fer, dtype=float))
    # 느슨한 구간부터 채우고 엄격한 구간으로 덮어써서 가장 높은 단가가 남게 한다
    for limit, won in sorted(tiers, key=lambda t: -t[0]):
        price = np.where(fer <= limit, won, price)
    return price


def evaluate_generation(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    capacity_mw: float,
    tiers: Sequence[Tuple[float, float]] = SETTLEMENT_TIERS_2025,
    min_capacity_factor: float = KPX_MIN_CAPACITY_FACTOR,
    jfpf: float = 1.0,
) -> Dict[str, float]:
    """발전량 예측 종합 지표.

    예측의 음수는 0으로 자른다 (발전량은 음수가 될 수 없다).
    jfpf: 정산단가 계수 (육지 1.0, 제주 전환기 0.5). 오차율·적중률에는 영향 없음.

    Returns:
        fer_mean_pct: 평가 대상 시간 평균 KPX 예측오차율 (1차 지표)
        n_eval_hours: 평가 대상 시간 수
        hit_{상한}_pct: 평가 대상 시간 중 FER ≤ 상한 비율 (정산 구간별)
        settle_won_per_kwh: 발전량 가중 평균 정산단가 (평가 대상 시간 기준)
        settle_won: 예상 정산금 합계 (원)
        mae_mw, rmse_mw, mbe_mw, r2: 전체 시간 기준
        nmae_all_pct, nrmse_all_pct: 전체 시간 MAE·RMSE / 설비용량
    """
    if not (np.isfinite(jfpf) and 0.0 < jfpf <= 1.0):
        raise ValueError(f"jfpf는 (0, 1] 범위여야 합니다: {jfpf}")
    y, p = _as_arrays(y_true, y_pred)
    p = np.clip(p, 0.0, None)
    fer, mask = kpx_error_rate(y, p, capacity_mw, min_capacity_factor)
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out: Dict[str, float] = {
        "capacity_mw": float(capacity_mw),
        "n_hours": int(len(y)),
        "n_eval_hours": int(mask.sum()),
        "fer_mean_pct": float(fer.mean()) if fer.size else float("nan"),
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "mbe_mw": float(np.mean(err)),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "nmae_all_pct": float(np.mean(np.abs(err)) / capacity_mw * 100.0),
        "nrmse_all_pct": float(np.sqrt(np.mean(err ** 2)) / capacity_mw * 100.0),
    }
    for limit, _ in sorted(tiers):
        out[f"hit_{limit:g}_pct"] = float(np.mean(fer <= limit) * 100.0) if fer.size else float("nan")
    if fer.size:
        price = settlement_price(fer, tiers) * jfpf
        gen_kwh = y[mask] * MWH_TO_KWH
        out["settle_won"] = float(np.sum(price * gen_kwh))
        out["settle_won_per_kwh"] = float(np.sum(price * gen_kwh) / np.sum(gen_kwh))
    else:
        out["settle_won"] = 0.0
        out["settle_won_per_kwh"] = float("nan")
    return out


def skill_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_ref: Sequence[float],
    mask: Optional[np.ndarray] = None,
) -> float:
    """RMSE 스킬 점수 1 − RMSE_model / RMSE_ref (IEA PVPS Task 16 권고). 0보다 커야 기준 예측보다 낫다.

    y_ref의 NaN(예: 첫날 지속성 예측)은 세 배열 공통으로 제외한다.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    r = np.asarray(y_ref, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(r)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if not ok.any():
        raise ValueError("스킬 점수를 계산할 공통 유효 표본이 없습니다")
    rmse = np.sqrt(np.mean((np.clip(p[ok], 0, None) - y[ok]) ** 2))
    rmse_ref = np.sqrt(np.mean((r[ok] - y[ok]) ** 2))
    return float(1.0 - rmse / rmse_ref)
