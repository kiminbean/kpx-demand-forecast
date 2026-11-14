"""
Solar V2 Model - 물리 피처 + 발전 수준 정규화 블렌드
====================================================

예측 = 0.5 × LGB(X → y) + 0.5 × lvl_p × LGB(X → y / lvl_p)   (lvl_p가 없으면 첫 항만)
야간(구간 내내 해가 지평선 아래)은 학습 기간 마지막 28일의 시각별 평균(태양광 연계 ESS 저녁 방전)으로 대체한다.

- 원 모델은 기상·기하 → 발전량 관계를, 정규화 모델은 '발전 수준' 변화(공개 거래량 포괄 범위 축소 등)와
  분리된 모양을 배운다. 두 모델의 계절별 오차가 달라 평균이 더 안정적이다 (CV 6개 월 중 5개 개선).
- 각 LightGBM은 학습 데이터 마지막 10%로 반복 수를 정한 뒤 전체로 재학습하고, 시드 3개 평균을 쓴다.
- n_jobs를 고정한다: LightGBM 결과는 스레드 수에 따라 미세하게 달라질 수 있어 재현성을 위해 실험값(2)과 맞춘다.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.solar_features_v2 import apply_night_profile, night_profile

DEFAULT_PARAMS: Dict = dict(
    objective="regression", metric="rmse", boosting_type="gbdt", num_leaves=31,
    learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    verbose=-1, n_jobs=2,
)
DEFAULT_SEEDS = (42, 43, 44)
ES_FRAC = 0.1
MAX_ROUNDS = 3000
ES_ROUNDS = 50
MIN_ROUNDS = 10
BLEND_ALPHA = 0.5
LEVEL_COL = "lvl_p"
MIN_NORM_ROWS = 500          # 정규화 모델을 학습할 최소 표본 (수준 프록시 유효 행)
NIGHT_WINDOW_DAYS = 28
META_FILE = "metadata.json"


def _fit_booster(X: pd.DataFrame, y: np.ndarray, w: Optional[np.ndarray], params: Dict) -> lgb.Booster:
    """마지막 ES_FRAC로 반복 수를 정하고 전체 데이터로 재학습 (평가 구간을 조기종료에 쓰지 않기 위함)"""
    k = int(len(X) * (1 - ES_FRAC))
    dtr = lgb.Dataset(X.iloc[:k], y[:k], weight=None if w is None else w[:k])
    des = lgb.Dataset(X.iloc[k:], y[k:], weight=None if w is None else w[k:], reference=dtr)
    probe = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[des],
                      callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    n_iter = max(probe.best_iteration, MIN_ROUNDS)
    return lgb.train(params, lgb.Dataset(X, y, weight=w), n_iter)


class SolarV2Model:
    """태양광 V2 예측기 (학습·예측·저장·불러오기)"""

    def __init__(self, params: Optional[Dict] = None, seeds: Sequence[int] = DEFAULT_SEEDS,
                 alpha: float = BLEND_ALPHA, night_window_days: int = NIGHT_WINDOW_DAYS):
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.seeds = list(seeds)
        self.alpha = alpha
        self.night_window_days = night_window_days
        self.feature_names: List[str] = []
        self.raw_models: List[lgb.Booster] = []
        self.norm_models: List[lgb.Booster] = []
        self.night_profile: Dict[int, float] = {}
        # 학습 때 결측이 한 번도 없던 피처에 예측 시 결측이 오면 LightGBM이 조용히 결측 분기로 보내므로 막는다
        self.nan_allowed: Dict[str, bool] = {}
        self.last_level: Optional[float] = None
        self.last_level_time: Optional[str] = None

    # ------------------------------------------------------------------ 학습
    def fit(self, X: pd.DataFrame, y: pd.Series, daylight_frac: pd.Series) -> "SolarV2Model":
        """X에 lvl_p 열이 없으면(발전 이력 없는 운영형) 원 모델만 학습한다"""
        if not X.index.equals(y.index):
            raise ValueError("X와 y의 인덱스가 다릅니다")
        if y.isna().any():
            raise ValueError("학습 타깃에 결측이 있습니다")
        self.feature_names = list(X.columns)
        self.nan_allowed = {c: bool(X[c].isna().any()) for c in X.columns}
        yv = y.values.astype(float)
        use_level = LEVEL_COL in X.columns
        lvl = X[LEVEL_COL].values if use_level else np.full(len(X), np.nan)
        ok = np.isfinite(lvl) & (lvl > 0)
        if use_level and ok.any():
            self.last_level = float(lvl[ok][-1])
            self.last_level_time = str(X.index[ok][-1])
        self.raw_models, self.norm_models = [], []
        for sd in self.seeds:
            p = dict(self.params, seed=sd)
            self.raw_models.append(_fit_booster(X, yv, None, p))
            if use_level and ok.sum() >= MIN_NORM_ROWS:
                # 원 공간 MSE와 같아지도록 정규화 공간 가중치 = 분모²
                w = lvl[ok] ** 2
                self.norm_models.append(_fit_booster(X[ok], yv[ok] / lvl[ok], w / w.mean(), p))
        self.night_profile = night_profile(y, daylight_frac, self.night_window_days)
        return self

    # ------------------------------------------------------------------ 예측
    def _validate(self, X: pd.DataFrame, day: np.ndarray, allow_stale_level: bool) -> pd.DataFrame:
        """주간 행 입력 검증. 야간 행은 프로파일로 대체되므로 검사하지 않는다."""
        if "p_mean" in X.columns:
            bad = day & ~np.isfinite(X["p_mean"].values)
            if bad.any():
                raise ValueError(f"주간 {int(bad.sum())}개 시각에 쓸 수 있는 일사·기온 관측이 없습니다: "
                                 f"{list(X.index[bad][:3])}")
        never_nan = [c for c in self.feature_names if not self.nan_allowed.get(c, True) and c != LEVEL_COL]
        for c in never_nan:
            bad = day & X[c].isna().values
            if bad.any():
                raise ValueError(f"학습 때 결측이 없던 피처 {c}에 주간 결측 {int(bad.sum())}개 (예: 직전 1시간 관측 누락)")
        if LEVEL_COL in self.feature_names:
            lvl = X[LEVEL_COL].values
            bad = day & ~(np.isfinite(lvl) & (lvl > 0))
            if bad.any():
                if not allow_stale_level or self.last_level is None:
                    raise ValueError(f"{LEVEL_COL}(D-2까지 발전 수준)이 주간 {int(bad.sum())}개 시각에 없습니다. "
                                     "발전량 이력을 넣거나 allow_stale_level=True로 마지막 학습 수준을 쓰세요")
                warnings.warn(f"{LEVEL_COL} 결측 {int(bad.sum())}개를 학습 마지막 수준 {self.last_level:.4f}"
                              f"({self.last_level_time})로 채웁니다", RuntimeWarning, stacklevel=3)
                X = X.copy()
                X.loc[bad, LEVEL_COL] = self.last_level
        return X

    def predict(self, X: pd.DataFrame, daylight_frac: pd.Series, allow_stale_level: bool = False) -> np.ndarray:
        """
        Args:
            X: create_solar_features_v2 결과 (+ lvl_p)
            daylight_frac: 구간 낮 비율 (X.index 전부 있어야 함)
            allow_stale_level: lvl_p가 없을 때 실패 대신 학습 마지막 수준을 쓸지 (경고 발생)

        Raises:
            ValueError: 피처 누락, 주간 입력 결측, lvl_p 결측(allow_stale_level=False), daylight_frac 누락
        """
        if not self.raw_models:
            raise RuntimeError("학습되지 않은 모델입니다")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"피처가 없습니다: {missing}")
        dl = daylight_frac.reindex(X.index)
        if dl.isna().any():
            raise ValueError("daylight_frac에 X 시각이 빠져 있습니다 (시간대 불일치 확인)")
        day = dl.values > 0
        X = self._validate(X[self.feature_names], day, allow_stale_level)
        p = np.mean([m.predict(X) for m in self.raw_models], axis=0)
        lvl = X[LEVEL_COL].values if LEVEL_COL in X.columns else np.full(len(X), np.nan)
        ok = np.isfinite(lvl) & (lvl > 0)
        if self.norm_models and ok.any():
            p_n = np.mean([m.predict(X[ok]) for m in self.norm_models], axis=0) * lvl[ok]
            p[ok] = (1 - self.alpha) * p[ok] + self.alpha * p_n
        return apply_night_profile(p, X.index, dl.values, self.night_profile)

    # ------------------------------------------------------------------ 저장
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        files = {"raw": [], "norm": []}
        for kind, models in (("raw", self.raw_models), ("norm", self.norm_models)):
            for sd, m in zip(self.seeds, models):
                name = f"solar_v2_{kind}_seed{sd}.lgb"
                m.save_model(str(d / name))
                files[kind].append({"file": name, "seed": sd, "num_trees": m.num_trees()})
        meta = {
            "feature_names": self.feature_names,
            "params": self.params,
            "seeds": self.seeds,
            "alpha": self.alpha,
            "night_window_days": self.night_window_days,
            "night_profile": {str(h): v for h, v in self.night_profile.items()},
            "nan_allowed": self.nan_allowed,
            "last_level": self.last_level,
            "last_level_time": self.last_level_time,
            "models": files,
        }
        (d / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory) -> "SolarV2Model":
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        obj = cls(params=meta["params"], seeds=meta["seeds"], alpha=meta["alpha"],
                  night_window_days=meta["night_window_days"])
        obj.feature_names = meta["feature_names"]
        obj.raw_models = [lgb.Booster(model_file=str(d / f["file"])) for f in meta["models"]["raw"]]
        obj.norm_models = [lgb.Booster(model_file=str(d / f["file"])) for f in meta["models"]["norm"]]
        obj.night_profile = {int(h): float(v) for h, v in meta["night_profile"].items()}
        obj.nan_allowed = {k: bool(v) for k, v in meta.get("nan_allowed", {}).items()}
        obj.last_level = meta.get("last_level")
        obj.last_level_time = meta.get("last_level_time")
        return obj
