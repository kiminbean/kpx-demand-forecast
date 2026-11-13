import numpy as np
import pandas as pd
import pytest

import scripts.mos_correct_nwp as mos


def test_observed_weather_loader_checks_safety_before_access(monkeypatch):
    folds = list(mos.T.CV_FOLDS) + [("2024-11-01", "2024-11-20")]
    monkeypatch.setattr(mos.T, "CV_FOLDS", folds)

    def unsafe_load(path):
        pytest.fail("observed weather accessed before safety check")

    monkeypatch.setattr(mos, "load_asos_stations", unsafe_load)
    with pytest.raises(RuntimeError, match="테스트 시작"):
        mos.load_observed_weather()


def test_cv_fits_only_pre_deadline_and_transforms_all_rows(monkeypatch):
    # 합성 입력(테스트 시작 이전)으로 실제 안전 검증을 통과해
    # 폴드별 계산 로직을 단위 검증한다.
    idx = pd.date_range('2024-04-20', '2024-05-03', freq='h')
    raw = pd.DataFrame({'solar_radiation_jeju': np.arange(len(idx), dtype=float)}, index=idx)
    obs = raw * 2
    obs.loc['2024-04-30 11:00':] = 1000000
    raw = raw.sample(frac=1, random_state=42)
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    monkeypatch.setattr(mos.T, 'CV_FOLDS', [('2024-05-01', '2024-05-03')])
    captured = {}

    def build(variant, weather):
        assert weather.index.is_monotonic_increasing
        captured['weather'] = weather.copy()
        frame = weather.assign(daylight_frac=1, generation_mw=100)
        return frame, weather, frame.generation_mw

    def predict(X, y, frame, tr, ev):
        captured['train'] = frame.index[tr]
        return np.full(ev.sum(), 100.)

    monkeypatch.setattr(mos, 'build_dataset_corrected', build)
    monkeypatch.setattr(mos.T, 'fit_predict', predict)
    mos.cv_fer('island_mean', raw, obs, 'linear')
    assert captured['train'].max() == pd.Timestamp('2024-04-30 10:00')
    pd.testing.assert_series_equal(captured['weather'].solar_radiation_jeju, raw.sort_index().solar_radiation_jeju * 2, check_names=False, rtol=1e-10, atol=1e-10)


def test_future_observations_do_not_change_fold_coefficients():
    idx = pd.date_range('2024-04-20', '2024-05-02', freq='h')
    raw = pd.DataFrame({f'solar_radiation_{st}': np.arange(len(idx), dtype=float) for st in mos.RAD_STATIONS}, index=idx)
    obs = raw * 2
    cutoff = pd.Timestamp('2024-04-30 10:00')
    changed = obs.copy()
    changed.loc[changed.index > cutoff] = 999999
    mask = mos.T.training_mask_for_forecast(raw.index, pd.Timestamp('2024-05-01'))
    first = mos.fit_mos(raw.loc[mask], obs, 'linear')
    second = mos.fit_mos(raw.loc[mask], changed, 'linear')
    assert first == second


@pytest.mark.parametrize("cloud_pairs", [0, 99, 100])
def test_linear_cloud_requires_100_complete_pairs(monkeypatch, cloud_pairs):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=120, freq='h')
    forecast = pd.DataFrame({
        'solar_radiation_jeju': np.arange(120, dtype=float),
        'cloud_cover_jeju': np.r_[np.arange(cloud_pairs, dtype=float),
                                  np.full(120 - cloud_pairs, np.nan)],
    }, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': np.arange(120, dtype=float) * 2}, index=idx)
    entry = mos.fit_mos(forecast, observed, 'linear_cloud')['stations']['jeju']
    assert entry['n_pairs'] == cloud_pairs
    assert entry['type'] == ('linear_cloud' if cloud_pairs >= 100 else 'identity')


def test_linear_cloud_counts_complete_pairs_even_if_radiation_pairs_are_insufficient(monkeypatch):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=99, freq='h')
    forecast = pd.DataFrame({'solar_radiation_jeju': 1.0,
                             'cloud_cover_jeju': np.nan}, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': 2.0}, index=idx)
    entry = mos.fit_mos(forecast, observed, 'linear_cloud')['stations']['jeju']
    assert entry == {'type': 'identity', 'n_pairs': 0}


def test_linear_cloud_missing_cloud_column_falls_back_to_identity(monkeypatch):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=120, freq='h')
    forecast = pd.DataFrame({'solar_radiation_jeju': 1.0}, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': 2.0}, index=idx)
    entry = mos.fit_mos(forecast, observed, 'linear_cloud')['stations']['jeju']
    assert entry == {'type': 'identity', 'n_pairs': 0}


@pytest.mark.parametrize('method', ['linear', 'linear_cloud', 'quantile'])
def test_nonfinite_fit_pairs_are_excluded(monkeypatch, method):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=100, freq='h')
    forecast = pd.DataFrame({'solar_radiation_jeju': np.arange(100, dtype=float),
                             'cloud_cover_jeju': 1.0}, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': np.arange(100, dtype=float)}, index=idx)
    forecast.iloc[-1, 0] = np.inf
    entry = mos.fit_mos(forecast, observed, method)['stations']['jeju']
    assert entry == {'type': 'identity', 'n_pairs': 99}


@pytest.mark.parametrize('extreme_input', ['forecast', 'observed'])
def test_quantile_fit_rejects_nonfinite_computed_knots(monkeypatch, extreme_input):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=100, freq='h')
    normal = np.linspace(0., 1., len(idx))
    extreme = np.r_[np.full(50, -1e308), np.full(50, 1e308)]
    forecast = pd.DataFrame({'solar_radiation_jeju': extreme if extreme_input == 'forecast' else normal}, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': extreme if extreme_input == 'observed' else normal}, index=idx)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='jeju'):
            mos.fit_mos(forecast, observed, 'quantile')


def test_quantile_fit_rejects_underflowing_slope(monkeypatch):
    monkeypatch.setattr(mos, 'RAD_STATIONS', ['jeju'])
    idx = pd.date_range('2024-04-01', periods=100, freq='h')
    forecast = pd.DataFrame({'solar_radiation_jeju': np.linspace(0., 1e308, 100)}, index=idx)
    observed = pd.DataFrame({'solar_radiation_jeju': np.linspace(0., 1e-308, 100)}, index=idx)
    with pytest.raises(ValueError, match='jeju'):
        mos.fit_mos(forecast, observed, 'quantile')


def test_require_safe_inputs_passes_for_preregistered_windows():
    assert mos._require_safe_inputs() is None


def test_require_safe_inputs_rejects_post_test_fold(monkeypatch):
    folds = list(mos.T.CV_FOLDS) + [("2024-11-01", "2024-11-20")]
    monkeypatch.setattr(mos.T, "CV_FOLDS", folds)
    with pytest.raises(RuntimeError, match="테스트 시작"):
        mos._require_safe_inputs()


def test_require_safe_inputs_rejects_post_test_index():
    with pytest.raises(RuntimeError, match="테스트 시작 이후"):
        mos._require_safe_inputs(max_index=pd.Timestamp("2024-11-08 00:00"))


@pytest.mark.parametrize('cloud', [None, np.nan, np.inf, -np.inf])
def test_apply_linear_cloud_rejects_missing_cloud_for_finite_radiation(cloud):
    forecast = pd.DataFrame({'solar_radiation_jeju': [1.0]})
    if cloud is not None:
        forecast['cloud_cover_jeju'] = cloud
    coeffs = {'stations': {'jeju': {'type': 'linear_cloud', 'a': 1., 'b': 0., 'c': 1.}}}
    with pytest.raises(ValueError, match='cloud_cover_jeju'):
        mos.apply_mos(forecast, coeffs)


def test_apply_linear_cloud_allows_missing_cloud_when_radiation_missing():
    forecast = pd.DataFrame({'solar_radiation_jeju': [np.nan], 'cloud_cover_jeju': [np.nan]})
    coeffs = {'stations': {'jeju': {'type': 'linear_cloud', 'a': 1., 'b': 0., 'c': 1.}}}
    assert np.isnan(mos.apply_mos(forecast, coeffs)['solar_radiation_jeju'].iloc[0])


@pytest.mark.parametrize('method,key', [('linear', 'a'), ('linear', 'b'),
                                        ('linear_cloud', 'a'), ('linear_cloud', 'b'),
                                        ('linear_cloud', 'c')])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_apply_rejects_nonfinite_linear_coefficients(method, key, value):
    forecast = pd.DataFrame({'solar_radiation_jeju': [1.0], 'cloud_cover_jeju': [1.0]})
    entry = {'type': method, 'a': 1., 'b': 0., 'c': 1.}
    entry[key] = value
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


@pytest.mark.parametrize('method', ['linear', 'linear_cloud'])
@pytest.mark.parametrize('slope', [2., -2.])
def test_apply_rejects_linear_arithmetic_overflow(method, slope):
    forecast = pd.DataFrame({'solar_radiation_jeju': [1e308], 'cloud_cover_jeju': [1.0]})
    entry = {'type': method, 'a': slope, 'b': 0., 'c': 1.}
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


def test_apply_rejects_unknown_mos_type():
    forecast = pd.DataFrame({'solar_radiation_jeju': [1.0]})
    entry = {'type': 'unknown', 'fcst_q': [0., 1.], 'obs_q': [0., 1.], 'obs_max': 1.}
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


@pytest.mark.parametrize('changes', [
    {'fcst_q': [], 'obs_q': []},
    {'fcst_q': [0., 1.], 'obs_q': [0.]},
    {'fcst_q': [0., np.nan]},
    {'obs_q': [0., np.inf]},
    {'fcst_q': [1., 1.]},
    {'fcst_q': [1., 0.]},
    {'obs_q': [1., 0.]},
    {'obs_max': np.nan},
    {'obs_max': np.inf},
    {'obs_max': -1.},
    {'fcst_q': [[0., 1.]]},
])
def test_apply_quantile_rejects_malformed_coefficients(changes):
    entry = {'type': 'quantile', 'fcst_q': [0., 1.],
             'obs_q': [0., 1.], 'obs_max': 1.}
    entry.update(changes)
    forecast = pd.DataFrame({'solar_radiation_jeju': [0.5]})
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


def test_apply_quantile_bounds_missing_and_input_immutability():
    forecast = pd.DataFrame({'solar_radiation_jeju': [-1., 0.5, 2., np.nan]})
    original = forecast.copy(deep=True)
    entry = {'type': 'quantile', 'fcst_q': [0., 1.],
             'obs_q': [-1., 3.], 'obs_max': 2.}
    result = mos.apply_mos(forecast, {'stations': {'jeju': entry}})
    np.testing.assert_allclose(result.solar_radiation_jeju, [0., 1., 2., np.nan], equal_nan=True)
    pd.testing.assert_frame_equal(forecast, original)


def test_apply_quantile_single_knot_is_constant():
    forecast = pd.DataFrame({'solar_radiation_jeju': [0., 1., 2., np.nan]})
    entry = {'type': 'quantile', 'fcst_q': [1.], 'obs_q': [0.5], 'obs_max': 1.}
    result = mos.apply_mos(forecast, {'stations': {'jeju': entry}})
    np.testing.assert_allclose(result.solar_radiation_jeju, [0.5, 0.5, 0.5, np.nan], equal_nan=True)


@pytest.mark.parametrize('fcst_q,obs_q,value', [
    ([-1e308, 1e308], [0., 1.], 0.),
    ([0., 1.], [-1e308, 1e308], 0.5),
])
def test_apply_quantile_rejects_overflowing_interpolation_span(fcst_q, obs_q, value):
    forecast = pd.DataFrame({'solar_radiation_jeju': [value]})
    entry = {'type': 'quantile', 'fcst_q': fcst_q, 'obs_q': obs_q, 'obs_max': 1e308}
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


def test_apply_quantile_rejects_underflowing_interpolation_slope():
    forecast = pd.DataFrame({'solar_radiation_jeju': [5e307]})
    entry = {'type': 'quantile', 'fcst_q': [0., 1e308],
             'obs_q': [0., 1e-308], 'obs_max': 1.}
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


@pytest.mark.parametrize('method', ['linear', 'linear_cloud', 'quantile', 'identity'])
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_apply_rejects_infinite_radiation(method, value):
    forecast = pd.DataFrame({'solar_radiation_jeju': [value], 'cloud_cover_jeju': [1.]})
    entry = {'type': method, 'a': 1., 'b': 0., 'c': 1.,
             'fcst_q': [0., 1.], 'obs_q': [0., 1.], 'obs_max': 1.}
    with pytest.raises(ValueError, match='jeju'):
        mos.apply_mos(forecast, {'stations': {'jeju': entry}})


def test_cv_fails_before_processing_inputs(monkeypatch):
    # 안전 조건 위반(테스트 창 포함 폴드)은 입력 처리 전에 실패해야 한다.
    folds = list(mos.T.CV_FOLDS) + [("2024-11-01", "2024-11-20")]
    monkeypatch.setattr(mos.T, "CV_FOLDS", folds)

    class UnprocessableInput(pd.DataFrame):
        def sort_index(self, *args, **kwargs):
            pytest.fail("input processed before safety check")

    with pytest.raises(RuntimeError, match="테스트 시작"):
        mos.cv_fer("island_mean", UnprocessableInput(), UnprocessableInput(), "linear")


@pytest.mark.parametrize("entry", ["selection", "dataset"])
def test_selection_paths_check_safety_before_unsafe_data_access(monkeypatch, entry):
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe data access or fitting occurred")

    folds = list(mos.T.CV_FOLDS) + [("2024-11-01", "2024-11-20")]
    monkeypatch.setattr(mos.T, "CV_FOLDS", folds)
    monkeypatch.setattr(mos.T, "load_nwp_weather", forbidden)
    monkeypatch.setattr(mos, "load_observed_weather", forbidden)
    monkeypatch.setattr(mos, "load_kpx_hourly", forbidden)
    monkeypatch.setattr(mos, "fit_mos", forbidden)
    with pytest.raises(RuntimeError, match="테스트 시작"):
        if entry == "selection":
            mos.run_selection(2)
        else:
            mos.build_dataset_corrected("island_mean", pd.DataFrame())


@pytest.mark.parametrize("entry", ["evaluate", "final"])
def test_exhausted_test_entrypoints_fail_before_data_access(monkeypatch, tmp_path, entry):
    def forbidden(*args, **kwargs):
        pytest.fail("data access or evaluation occurred")

    monkeypatch.setattr(mos.T, "load_nwp_weather", forbidden)
    monkeypatch.setattr(mos, "evaluate_test_corrected", forbidden if entry == "final" else mos.evaluate_test_corrected)
    with pytest.raises(RuntimeError, match="테스트"):
        if entry == "evaluate":
            mos.evaluate_test_corrected("island_mean", "nwp_msm_day2", "linear", tmp_path)
        else:
            mos.run_final_test("linear")
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_test_flag_before_selection(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mos_correct_nwp.py", "--select", "--evaluate-test"])
    monkeypatch.setattr(mos, "run_selection", lambda *_: pytest.fail("selection started"))
    with pytest.raises(SystemExit) as exc:
        mos.main()
    assert exc.value.code != 0
    assert "테스트" in capsys.readouterr().err


def test_help_does_not_advertise_exhausted_test_evaluation(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mos_correct_nwp.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        mos.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--select" in help_text
    assert "python scripts/mos_correct_nwp.py --evaluate-test" not in help_text
    assert "테스트 1회 평가" not in help_text
