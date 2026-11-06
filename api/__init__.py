"""
J-REF API Module
================

제주도 신재생에너지(태양광/풍력) 발전량 예측 REST API

Features:
- Wind: LightGBM Phase 5 Fair 모델. 시각 T 예측 = 구간 (T, T+1h] 발전량 (학습 라벨이 구간 시작 기준)
- Solar: Solar V2 섬 평균(island_mean) 모델 (물리 피처 LightGBM, 발전 이력 불필요).
  시각 T 예측 = KPX 거래시간 구간 (T-1h, T] 발전량이며 직전 1시간 기상이 필요하다
- 두 에너지원은 서로 다른 1시간을 가리키므로 응답의 interval_start/interval_end로 구간을 명시하고 합계(total_mw)는 내지 않는다
- 불확실성: 관측 기상 테스트 잔차 RMSE 기반 대략적 ±1.28·RMSE 구간 (보정된 80% 구간 아님, 태양광은 낮/밤 RMSE 분리)
"""

__version__ = "1.1.0"
__author__ = "J-REF Team"
