# 운영 반영 — 계통 라벨 서빙 모델, 발전 수준 입력, 하루 전 파이프라인 (2026-11-14)

`solar_dayahead_verified_2026-11-14.md` 6절의 권고 네 가지를 사용자 결정에 따라 적용했다.

## 무엇이 바뀌었나

| 항목 | 이전 | 지금 |
|---|---|---|
| API 버전 | 1.2.0 | 1.3.0 |
| 태양광 예측 대상 | 공개 거래량의 `태양광` (입찰자원 제외) | **KPX 제주계통운영정보 태양광** (입찰자원 포함) |
| 풍력 예측 대상 | 공개 거래량의 `풍력` (2024-06 이후 약 100 MW) | **KPX 제주계통운영정보 풍력** (제주 전체) |
| 학습 기간 | 2024년 | 2024-01 ~ 2026-10 |
| 태양광 모델 | `models/solar_v2/island_mean` | `models/solar_v2_system/island_mean` + `island_level` |
| 풍력 모델 | `models/wind_phase5_fair/` | `models/wind_system/` |
| 발전 수준 입력 | 없음 | 선택 입력 `level_history` |
| 하루 전 예측 | 없음 | `scripts/predict_dayahead.py` |

- **값의 크기가 달라진다.** 태양광은 이전보다 약 1.5배, 풍력은 훨씬 커진다(제주 전체). 이 API를 쓰는 쪽은 대상이 바뀐 것을 알아야 한다.
- 이전 모델 파일은 그대로 있다. 환경변수 `SERVING_LABEL=public`으로 이전 모델을 서빙할 수 있다.
- 모델 구성은 새로 고르지 않았다. 이미 평가한 구성을 최신 자료로 다시 학습했을 뿐이다 (`scripts/train_serving_models_system.py`).

## API의 `level_history`

- 내용: 최근 14일(예측일 D 기준 D-2까지)의 시간별 기록 — `datetime`(구간 끝), `temperature`, `humidity`, `solar_radiation`, `cloud_cover`, `solar_mw`(실제 발전량).
- 낮 시간 기록이 4일치 이상이면 발전 수준 모델(`island_level`)로 예측하고 응답에 `level_used: true`가 온다. 없거나 모자라면 기상만 쓰는 모델이 답하고 `level_used: false`다.
- D-1 이후 기록은 넣어도 쓰이지 않는다 (학습과 같은 D-2 지연 규칙).
- `solar_mw`는 KPX 제주계통운영정보 값이어야 한다 (`scripts/download_kpx_jeju_system.py`, 10분 지연 보정한 시간 평균). 다른 기준의 발전량을 넣으면 수준이 어긋난다.
- 근거: 관측 기상·고정 모델 조건에서 발전 수준 피처가 2025년 −1.56, 2023년 −0.79, 2026년 −1.32%p.

## 불확실성 폭

- 태양광·풍력 모두 2026-01 ~ 10 walk-forward(월별 재학습, **관측 기상 입력**) 잔차의 RMSE로 ±1.28·RMSE를 낸다. 태양광 낮 RMSE는 `predictions_test.csv`에서, 풍력은 36.5 MW(`wind_serving_metadata.json`)다.
- 보정된 80% 구간이 아니다. 예보 기상을 넣으면 실제 오차는 이보다 크다.

## 하루 전 파이프라인

```bash
python scripts/predict_dayahead.py --refresh-label                 # 내일, 10시 제출용: GFS + IFS 평균
python scripts/predict_dayahead.py --submission 17                 # 17시 제출용: GFS D-1 00 UTC
```
- 실행할 때마다 마감 시각까지의 라벨과 과거 예보로 다시 학습한다 (재학습 주기 = 매 실행).
- 예보가 없거나 마감 뒤에 올라온 날은 실패한다. 대체 예보로 메우지 않는다.
- 2026-11-15분을 실제로 돌려 확인했다 (10시용 일 합계 929 MWh, 17시용 909 MWh).
- 기대 성능: 10시용 구성의 2026-01 ~ 10 FER 9.04% (분모 490 MW). 발전 수준 피처를 붙인 구성(8.80%)은 선택 단계를 통과하지 못해 쓰지 않았다.

## 알려진 한계

- 풍력 서빙 모델의 지표(FER 9.5%)는 관측 기상 상한이다. 풍속 예보 입력 성능은 재지 않았다.
- 계통운영정보는 정산 계량값이 아니고 출력제어가 반영돼 있다.
- 17시용 다중 예보(GFS + IFS 18 UTC)는 자료를 받지 못해 평가·적용하지 않았다.
- `render.yaml`의 자동 배포는 main 브랜치 기준이다. 이 변경은 `feature/solar-v2-kpx-fer`에만 있다.

## 검증

`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_models.py -k 'not test_saved_model_reproduces_reported_test_metrics'`
- `tests/test_api_system_models.py`: 기본 설정이 계통 모델을 가리키는지, `level_history` 유무·부족·D-1 이후 기록 처리.
- `tests/test_api_service.py`: 이전 모델(`SERVING_LABEL=public`) 동작이 그대로인지.
