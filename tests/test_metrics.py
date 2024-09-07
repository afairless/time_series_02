 
import pytest
import numpy as np

from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st

from sklearn.metrics import (
    root_mean_squared_error as skl_rmse, 
    mean_absolute_error as skl_mae, 
    median_absolute_error as skl_mdae)

from skforecast.metrics.metrics import (
    mean_squared_error as skf_mse,
    mean_absolute_error as skf_mae,
    median_absolute_error as skf_mdae)

import statsmodels.tsa.statespace.sarimax as sarimax

from src.s02_model import (
    root_median_squared_error,
    TimeSeriesDifferencing,
    )


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_mean_squared_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_rmse(y1, y2)
    skf_result = np.sqrt(skf_mse(y1, y2))
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_mean_absolute_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_mae(y1, y2)
    skf_result = skf_mae(y1, y2)
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_median_absolute_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = low + (high - low) * rng.random(arr_len)
    skl_result = skl_mdae(y1, y2)
    skf_result = skf_mdae(y1, y2)
    np.testing.assert_almost_equal(skl_result, skf_result)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_median_squared_error_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test valid input
    """
    rng = np.random.default_rng(seed)
    y1 = low + (high - low) * rng.random(arr_len)
    y2 = y1
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([0]))


def test_median_squared_error_02():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 2, 1])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([0]))


def test_median_squared_error_03():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 1, 1])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([1]))


def test_median_squared_error_04():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3])
    y2 = np.array([0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([2]))


def test_median_squared_error_05():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3, 4])
    y2 = np.array([0, 0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([np.sqrt(13/2)]))


def test_median_squared_error_06():
    """
    Test valid input
    """
    y1 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([0, 0, 0, 0, 0])
    result = root_median_squared_error(y1, y2)
    np.testing.assert_almost_equal(result, np.array([3]))


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_01(
    low: int, high: int, arr_len: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(time_series, k_diff=1, k_seasonal_diff=0)
    ts_diff_1 = time_series[1:] - time_series[:-1]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - 1


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    period=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_02(
    low: int, high: int, arr_len: int, period: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=0, k_seasonal_diff=1, seasonal_periods=period)
    ts_diff_1 = time_series[period:] - time_series[:-period]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - period


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    period=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_statsmodels_sarimax_differencing_03(
    low: int, high: int, arr_len: int, period: int, seed: int):
    """
    Verify how sarimax differencing function works
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=1, k_seasonal_diff=1, seasonal_periods=period)

    ts_diff_1 = time_series[period:] - time_series[:-period]
    ts_diff_2 = ts_diff_1[1:] - ts_diff_1[:-1]

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_2)
    assert len(ts_diff_0) == len(time_series) - (period+1)


def test_difference_time_series_01():
    """
    Test invalid input:  array with >1 dimension
    """

    time_series = np.array([[1, 2, 3], [4, 5, 6]])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    with pytest.raises(AssertionError):
        _ = ts_diff.difference_time_series(time_series)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_difference_time_series_02(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test no differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    ts_diff_1 = ts_diff.difference_time_series(time_series)
    np.testing.assert_almost_equal(time_series, ts_diff_1)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=4, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4))
@settings(print_blob=True)
def test_difference_time_series_03(
    low: int, high: int, arr_len: int, seed: int, k_diff: int):
    """
    Test simple differencing only
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(time_series, k_diff=k_diff)

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - k_diff


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=50))
@settings(print_blob=True)
def test_difference_time_series_04(
    low: int, high: int, arr_len: int, seed: int, k_seasonal_diff: int, 
    seasonal_periods: int):
    """
    Test seasonal differencing only
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=0, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)

    ts_diff = TimeSeriesDifferencing(
        k_diff=0, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    assert len(ts_diff_0) == max(0, (len(time_series) - season_period_diff_len))


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=100, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4),
    k_seasonal_diff=st.integers(min_value=1, max_value=4),
    seasonal_periods=st.integers(min_value=1, max_value=50))
@settings(print_blob=True)
def test_difference_time_series_05(
    low: int, high: int, arr_len: int, seed: int, k_diff: int, 
    k_seasonal_diff: int, seasonal_periods: int):
    """
    Test simple and seasonal differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff_0 = sarimax.diff(
        time_series, k_diff=k_diff, k_seasonal_diff=k_seasonal_diff, 
        seasonal_periods=seasonal_periods)

    ts_diff = TimeSeriesDifferencing(
        k_diff=k_diff, k_seasonal_diff=k_seasonal_diff,
        seasonal_periods=seasonal_periods)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)

    season_period_diff_len = k_seasonal_diff * seasonal_periods
    diff_total = k_diff + season_period_diff_len 
    assert len(ts_diff_0) == max(0, (len(time_series) - diff_total))


def test_de_difference_time_series_01():
    """
    Test invalid input:  no differencing before de-differencing
    """

    time_series = np.array([1, 2, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    with pytest.raises(ValueError):
        _ = ts_diff.de_difference_time_series(time_series)


def test_de_difference_time_series_02():
    """
    Test invalid input:  array with >1 dimension
    """

    time_series_1 = np.array([1, 2, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(AssertionError):
        _ = ts_diff.de_difference_time_series(time_series_2)


@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000))
@settings(print_blob=True)
def test_de_difference_time_series_03(
    low: int, high: int, arr_len: int, seed: int):
    """
    Test no differencing
    """

    rng = np.random.default_rng(seed)
    time_series = low + (high - low) * rng.random(arr_len)

    ts_diff = TimeSeriesDifferencing(k_diff=0)

    _ = ts_diff.difference_time_series(time_series)

    ts_diff_2 = ts_diff.de_difference_time_series(time_series)
    np.testing.assert_almost_equal(time_series, ts_diff_2)


def test_de_difference_time_series_04():
    """
    Test simple differencing only:  empty de-differencing input vector
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 0, 4, -8, -3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    ts_diff_2 = ts_diff.de_difference_time_series()
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_05():
    """
    Test simple differencing only:  empty de-differencing input vector
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 0, 4, -8, -3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([])

    ts_diff_2 = ts_diff.de_difference_time_series(time_series_2)
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_de_difference_time_series_06():
    """
    Test simple differencing only:  k_diff = 1
    """

    k_diff = 1
    time_series_1 = np.array([1, 2, 5, 10, 8])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1, 1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([1, 3, 7, 13, 12])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_07():
    """
    Test simple differencing only:  k_diff = 2
    """

    k_diff = 2
    time_series_1 = np.array([7, 3, 9, 1, 3])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([7, 3, 10, 4, 9])

    np.testing.assert_almost_equal(result, correct_result)


def test_de_difference_time_series_08():
    """
    Test simple differencing only:  k_diff = 3
    """

    k_diff = 3
    time_series_1 = np.array([2, 8, 7, 2, 4])

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    _ = ts_diff.difference_time_series(time_series_1)

    time_series_2 = np.array([1, 1])
    result = ts_diff.de_difference_time_series(time_series_2)

    correct_result = np.array([2, 8, 7, 3, 8])

    np.testing.assert_almost_equal(result, correct_result)


'''
@given(
    low=st.integers(min_value=-1000, max_value=1000),
    high=st.integers(min_value=-1000, max_value=1000),
    arr_len=st.integers(min_value=4, max_value=1000),
    seed=st.integers(min_value=1, max_value=1_000_000),
    k_diff=st.integers(min_value=1, max_value=4))
@settings(print_blob=True)
def test_de_difference_time_series_90(
    low_1: int, high_1: int, arr_len_1: int, seed_1: int, 
    k_diff: int):
    """
    Test simple differencing only
    """

    rng = np.random.default_rng(seed_1)
    time_series_1 = low_1 + (high_1 - low_1) * rng.random(arr_len_1)

    ts_diff_0 = sarimax.diff(time_series_1, k_diff=k_diff)

    ts_diff = TimeSeriesDifferencing(k_diff=k_diff)
    ts_diff_1 = ts_diff.difference_time_series(time_series_1)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series_1) - k_diff

    ts_diff_2 = ts_diff.de_difference_time_series(time_series_1)
    np.testing.assert_almost_equal(time_series_1, ts_diff_2)


def test_difference_time_series_99():
    """
    Test simple differencing
    """

    # rng = np.random.default_rng(seed)
    # time_series = low + (high - low) * rng.random(arr_len)
    time_series = np.array([1, 2, 3, 4, 5, 6])
    time_series = np.array([1, 2, 4, 7, 2, 5])
    time_series = np.array([9, 2, 4, 7, 2, 5])

    ts_diff_0 = sarimax.diff(time_series, k_diff=1)

    ts_diff = TimeSeriesDifferencing(k_diff=1)
    ts_diff_1 = ts_diff.difference_time_series(time_series)

    np.testing.assert_almost_equal(ts_diff_0, ts_diff_1)
    assert len(ts_diff_0) == len(time_series) - 1

    ts_diff_2 = ts_diff.de_difference_time_series(ts_diff_1)

    # test that de-differenced series is linear transformation of 'time_series'
    slope_diffs = np.std(time_series - ts_diff_2)
    np.testing.assert_almost_equal(slope_diffs, 0)
'''


