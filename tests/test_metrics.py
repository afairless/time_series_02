 
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st

from sklearn.metrics import (
    root_mean_squared_error as skl_rmse, 
    mean_absolute_error as skl_mae, 
    median_absolute_error as skl_mdae)

from skforecast.metrics.metrics import (
    mean_squared_error as skf_mse,
    mean_absolute_error as skf_mae,
    median_absolute_error as skf_mdae)

from src.s02_model import root_median_squared_error


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


