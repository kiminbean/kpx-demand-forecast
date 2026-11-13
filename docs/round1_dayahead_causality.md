# 1차 라운드: D-1 10:00 마감 기준 인과성(causality) 정합화

> 라운드: 총 3차 중 1차 (2026-11-14)
> 상위 지시: 이전 라운드의 금지·연기 항목 반복 금지, 1차 라운드 문서 작성 후 커밋·푸시.

## 배경

사전 점검에서 day-ahead(NWP) 파이프라인에 두 가지 인과성 위반이 확인됐다.

1. **학습 마스크 미래 누출**: `train_solar_v2.py`의 모든 학습 구간 마스크(CV 폴드·검증·테스트)가
   "구간 시작 < CUT"까지만 걸러서, 폴드 마감 시점 직전의 **마지막 부분일(마감 이후 종료 시각)** 학습
   샘플이 그대로 포함됐다. 특히 `run_test`는 D-1 10:00 마감 이후에 끝나는 2024-11-06 오후 구간까지
   학습에 썼다.
2. **day1 폴백 재평가**: `run_cv`/`run_validation`의 NWP 조건에서 `previous_day1`로 재시도하는
   폴백이 있었다. day1 실행 시점은 D-1 10:00 마감을 넘길 수 있어(AGENTS.md 데이터 함정 항목),
   day2 전용 조건으로 평가하려는 목적 자체를 훼손한다.

## 변경 사항

- `scripts/train_solar_v2.py`
  - `training_mask_for_forecast(df, cutoff)` 도입: 학습 샘플을 **구간 종료 시각** 기준으로
    `<= cutoff`(D-1 10:00 마감)로 제한. CV 폴드·검증·테스트 학습 마스크 전부에 적용.
  - NWP day2 조건의 day1 폴백 제거. day2 결측은 이제 `ValueError`로 즉시 실패(조용한 재평가 금지).
  - `run_test`가 마감 기준 마스크를 쓰도록 수정 → `--weather nwp_msm_day2 --evaluate-test` 재실행은
    기존 8.53/8.65% 재현이 아니라 **정정된(더 작은) 학습 집합**으로 평가하게 됨.
- `scripts/mos_correct_nwp.py` (MOS 선형/분위 보정, b0b98b2에서 "미적용" 판정된 실험)
  - `main()`에 실험 재실행 방지 가드 추가: `--evaluate-test`와 `--select`는 명시적
    `--i-know-this-blocks-reproduction` 플래그 없이 실행 불가(사전 등록 결과 보존).
  - `cv_fer`가 폴드별로 MOS를 폴드 마감 시점까지의 데이터로만 적합하도록 수정(폴드 누출 제거).
  - 정리: 미사용 상수·임포트 제거(FINAL_VARIANTS, FINAL_WEATHERS, SolarV2Model 등).
- 테스트 추가: `tests/test_mos_causal.py` (72개)
  - `training_mask_for_forecast` 마감 경계(포함/배제), day2 폴백 부재(ValueError), 검증·테스트
    마스크가 day1 전환을 요구하지 않는지, MOS 폴드 적합 구간이 마감을 넘지 않는지 검증.

## 금지 목록 준수 확인

- 2024-11-07~12-31 테스트 윈도우를 모델 선택/튜닝에 **사용하지 않음** (재실행·재선택 없음,
  저장된 모델 아티팩트 변경 없음).
- MOS 보정(미채택 판정) 재실행 없음 — 가드로 봉인만 함.
- 해줌 특허 2단 구조(테스트 동률 미채택, ecdeb64) 재시도 없음.

## 검증

- `pytest tests/ -q --ignore=tests/test_models.py` → **119 passed** (기존 47 + 신규 72).
- 기존 관측기상 재현 테스트(`test_saved_model_reproduces_reported_test_metrics`) 통과:
  관측기상(asos) 경로의 기존 결과와 API 모델 아티팩트는 불변.

## 기존 수치에 미치는 영향 (중요)

- AGENTS.md·문서의 **NWP day-ahead FER 8.53~8.65%는 위반 마스크 + day1 폴백 하에서 얻은 역사적
  수치**가 된다. 방향적 결론(목표 ≤6% 미달, 관측기상 4.18%가 상한)은 수치의 정확성과 무관하게
  유효하나, 정확한 정정 수치는 추후 라운드에서 CV로 다시 산출해야 한다(이번 라운드에서는 산출 안 함
  — 테스트 윈도우 보호 규칙 준수).
- `--evaluate-test`를 정정된 마스크로 다시 돌리는 것은 테스트 윈도우 규칙상 금지(선택·튜닝 목적).

## 남은 후속 (2·3차 라운드 후보)

- 정정된 인과 파이프라인으로 `--weather nwp_msm_day2 --cv` 재산출 (테스트 미사용).
- 2025 KPX 홀드아웃 확보 시 위 수치와 비교.
