#! /usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import dirichlet
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.deterministic import DeterministicProcess

import matplotlib.pyplot as plt


@dataclass
class TimeSeriesTrendSegments:
    trend_lengths: np.ndarray
    trend_slopes: np.ndarray
    time_series: np.ndarray


def create_time_series_01(
    n: int, n_trends: int, 
    autogressive_order: int=3, moving_average_order: int=2, 
    arma_std_factor: float=4, season_std_factor: float=9, 
    seed: int=231562) -> np.ndarray:
    """
    Create time series data with multiple trends and seasonality added to output
        from ARMA model
    """

    np.random.seed(seed)
    arma = sm.tsa.arma_generate_sample(
        autogressive_order, moving_average_order, n, burnin=max(20, n//10))

    # add multiple trends across time series
    trend_len_mean = n / n_trends
    trend_len_min = math.floor(trend_len_mean / 2)
    trend_len_max = math.ceil(trend_len_mean * 2)
    trend_lens = np.random.randint(trend_len_min, trend_len_max, n_trends)
    trend_lens = np.round((trend_lens / trend_lens.sum()) * n).astype(int)
    trend_lens[-1] += n - trend_lens.sum()
    assert trend_lens.sum() == n

    trend_slopes = np.random.uniform(low=-3, high=3, size=n_trends)

    assert len(trend_lens) == len(trend_slopes)
    trends_by_n = [
        range(1, trend_lens[i]+1) * trend_slopes[i] 
        for i in range(n_trends)]
    trend_ends = [e[-1] for e in trends_by_n]
    trend_continues = [0] + trend_ends[:-1]
    trend_continues = np.array(trend_continues).cumsum()
    trends = np.concatenate([
        e + trend_continues[i] 
        for i, e in enumerate(trends_by_n)])

    season = np.sin(np.pi * np.arange(n) / 6)

    # re-scale ARMA output and seasonality
    arma_factor = np.std(trends) / arma_std_factor
    season_factor = np.std(trends) / season_std_factor 

    # assemble all components of time series
    time_series = arma_factor * arma + season_factor * season + trends

    return time_series


def randomize_segment_lengths(
    total_n: int, segment_n: int, seed=282840) -> np.ndarray:
    """
    Divide 'total_n' into 'segment_n' segments and randomize the lengths of the
        segments
    """

    proportions = dirichlet.rvs([2] * segment_n, size=1, random_state=seed)
    factor = total_n / np.sum(proportions)
    discrete_segment_lengths = np.round(proportions * factor).astype(int)[0]

    if np.sum(discrete_segment_lengths) > total_n:
        difference = np.sum(discrete_segment_lengths) - total_n
        max_idx = np.argmax(discrete_segment_lengths)
        discrete_segment_lengths[max_idx] -= difference

    elif np.sum(discrete_segment_lengths) < total_n:
        difference = np.sum(discrete_segment_lengths) - total_n
        min_idx = np.argmin(discrete_segment_lengths)
        discrete_segment_lengths[min_idx] -= difference

    assert np.sum(discrete_segment_lengths) == total_n

    return discrete_segment_lengths


def flatten_list_of_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def generate_and_combine_trends(
    time_n: int, trend_n: int, 
    trend_slope_min: float=-1, trend_slope_max: float=1,
    seed: int=459170) -> TimeSeriesTrendSegments:
    """
    Generate 'trend_n' trend segments with a total number of 'time_n' time steps 
        where each segment has a randomized length and slope
    """

    assert trend_n > 0

    trend_lens = randomize_segment_lengths(time_n, trend_n)
    rng = np.random.default_rng(seed)
    trend_slopes = rng.uniform(trend_slope_min, trend_slope_max, trend_n)
    assert len(trend_lens) == len(trend_slopes)

    trend_slopes_extended_lists = [
        [trend_slopes[i]] * trend_lens[i] for i in range(len(trend_lens))]
    trend_slopes_extended = flatten_list_of_lists(trend_slopes_extended_lists)
    assert len(trend_slopes_extended) == time_n

    # set first slope to zero so that doesn't change first time series value
    trend_slopes_extended[0] = 0
    trend = np.array(trend_slopes_extended).cumsum()

    trend_segments = TimeSeriesTrendSegments(trend_lens, trend_slopes, trend)

    return trend_segments 


def create_time_series_02(
    time_n: int=100, series_n: int=1, constant: Sequence[float]=[0]*100, 
    trend_n: int=1, trend_slope_min: float=-1, trend_slope_max: float=1,
    season_period: int=10, sin_amplitude: float=1, cos_amplitude: float=1, 
    autogressive_lag_polynomial_coefficients: np.ndarray=np.array([1, 1]), 
    moving_average_lag_polynomial_coefficients: np.ndarray=np.array([1, 1]), 
    arma_scale: float=1, seed: int=231562) -> TimeSeriesTrendSegments:
    """
    Create time series data with multiple trends and seasonality added to output
        from ARMA model
    """

    assert len(constant) == time_n


    # set multiple trends across time series
    ##################################################

    trend_segments = generate_and_combine_trends(
        time_n, trend_n, trend_slope_min, trend_slope_max, seed)

    trend_lengths = trend_segments.trend_lengths
    trend_slopes = trend_segments.trend_slopes
    trend = trend_segments.time_series


    # set seasonality across time series
    ##################################################

    time_idx = np.arange(time_n)
    season_sin = sin_amplitude * np.sin(2 * np.pi * time_idx / season_period)
    season_cos = cos_amplitude * np.cos(2 * np.pi * time_idx / season_period)


    # set ARMA noise across time series
    ##################################################

    # ar = np.array([1, autogressive_lag_polynomial_coefficients])
    # ma = np.array([1, moving_average_lag_polynomial_coefficients])
    ar = autogressive_lag_polynomial_coefficients
    ma = moving_average_lag_polynomial_coefficients
    arma_process = ArmaProcess(ar, ma) 

    np.random.seed(seed+1)
    arma_noise = arma_process.generate_sample(
        nsample=(series_n, time_n), axis=1, scale=arma_scale, 
        burnin=max(20, time_n//10))


    # combine all components into a single time series
    ##################################################

    constant_arr = np.tile(constant, (series_n, 1))
    trend_arr = np.tile(trend, (series_n, 1))
    season_sin_arr = np.tile(season_sin, (series_n, 1))
    season_cos_arr = np.tile(season_cos, (series_n, 1))
    time_series = (
        constant_arr + trend_arr + season_sin_arr + season_cos_arr + arma_noise)

    time_series_with_trends = TimeSeriesTrendSegments(
        trend_lengths, trend_slopes, time_series)

    return time_series_with_trends 


def main():

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()

    # n = 400
    # n_trends = 7
    # srs = create_time_series_01(n, n_trends, 3, 2, 4, 9, 84558)

    time_n = 100
    series_n = 2
    constant = [0] * time_n

    # trend parameters
    trend_n = 5
    trend_slope_min = -9 
    trend_slope_max = 9

    # season parameters
    season_period = 20
    sin_amplitude = 20
    cos_amplitude = 20

    # ARMA parameters
    ar_lag_coef = np.array([1, 0.9, 0.8])
    ma_lag_coef = np.array([1, 0.9, 0.8])
    arma_scale = 2

    time_series = create_time_series_02(
        time_n, series_n, constant, 
        trend_n, trend_slope_min, trend_slope_max, 
        season_period, sin_amplitude, cos_amplitude, 
        ar_lag_coef, ma_lag_coef, arma_scale, 761824)
    ts = time_series.time_series


    plt.clf()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(ts)
    plt.title('Generated Time Series with Trend, Seasonality, and ARMA Terms')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


if __name__ == '__main__':
    main()
