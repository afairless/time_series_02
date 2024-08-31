#! /usr/bin/env python3

import polars as pl
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.stats import dirichlet
from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.deterministic import DeterministicProcess

import matplotlib.pyplot as plt


@dataclass
class TimeSeriesTrendSegments:

    trend_lengths: np.ndarray
    trend_slopes: np.ndarray
    time_series: np.ndarray

    def __post_init__(self):

        assert len(self.trend_lengths) > 0
        assert len(self.trend_slopes) > 0
        assert len(self.time_series) > 0

        assert len(self.trend_lengths) == len(self.trend_slopes)

        if self.time_series.ndim == 1:
            assert len(self.time_series) == self.trend_lengths.sum()
        else:
            assert self.time_series.shape[1] == self.trend_lengths.sum()


@dataclass
class TimeSeriesParameters:

    time_n: int
    series_n: int
    constant: np.ndarray
    trend_n: int
    trend_slope_min: float
    trend_slope_max: float
    season_period: int
    sin_amplitude: float
    cos_amplitude: float
    autogressive_lag_polynomial_coefficients: np.ndarray
    moving_average_lag_polynomial_coefficients: np.ndarray
    arma_scale: float
    seed: int
    trend_lengths: np.ndarray
    trend_slopes: np.ndarray
    time_series: np.ndarray

    def __post_init__(self):

        assert len(self.constant) == self.time_n
        assert len(self.trend_lengths) == self.trend_n
        assert len(self.trend_slopes) == self.trend_n
        assert self.time_series.shape == (self.series_n, self.time_n)
        assert self.autogressive_lag_polynomial_coefficients[0] == 1
        assert self.moving_average_lag_polynomial_coefficients[0] == 1
        assert self.arma_scale > 0


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

    assert time_n > 0
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


def create_time_series(
    time_n: int=100, series_n: int=1, constant: np.ndarray=np.zeros(100),
    trend_n: int=1, trend_slope_min: float=-1, trend_slope_max: float=1,
    season_period: int=10, sin_amplitude: float=1, cos_amplitude: float=1, 
    autogressive_lag_polynomial_coefficients: np.ndarray=np.array([1, 1]), 
    moving_average_lag_polynomial_coefficients: np.ndarray=np.array([1, 1]), 
    arma_scale: float=1, seed: int=231562) -> TimeSeriesTrendSegments:
    """
    Generate time series data with multiple trends and seasonality added to 
        output from ARMA model
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


def convert_time_series_parameters_to_dataframe(
    ts_params: TimeSeriesParameters) -> pl.DataFrame:
    """
    Extract the time series and its generating parameters from the dataclass and
        store them in a DataFrame
    If there is more than one generated time series, store each series on a 
        separate row with its corresponding parameters
    """

    # verify that the intended number of time series matches the actual number
    assert ts_params.time_series.shape[0] == ts_params.series_n


    # extract parameters from dataclass fields and store in dataframe
    ##################################################

    df_dict = {}
    for field in fields(ts_params):

        is_scalar = (field.type == int) or (field.type == float)
        if is_scalar:
            df_dict[field.name] = [getattr(ts_params, field.name)]

        elif field.type == np.ndarray and field.name != 'time_series':

            # remove arrays that are too long and unwieldy to include in table
            #   these arrays will usually include the vector of constants
            # if len(getattr(ts_params, field.name)) < 20:
            #     for i in range(len(getattr(ts_params, field.name))):
            #         df_dict[f'{field.name}_{i}'] = [
            #             getattr(ts_params, field.name)[i]]

            # CORRECTION:  include the vector of constants
            for i in range(len(getattr(ts_params, field.name))):
                df_dict[f'{field.name}_{i}'] = [
                    getattr(ts_params, field.name)[i]]

    params_row_df = pl.DataFrame(df_dict)
    params_df = pl.concat([params_row_df] * ts_params.series_n, how='vertical')


    # extract time series from dataclass and store in dataframe
    ##################################################

    colnames = ['ts_' + str(i) for i in range(ts_params.time_n)]
    ts_df = pl.DataFrame(ts_params.time_series)
    ts_df.columns = colnames


    # combine parameters and time series into a single dataframe
    ##################################################

    ts_params_df = pl.concat([params_df, ts_df], how='horizontal')

    return ts_params_df


def plot_time_series(
    time_series: np.ndarray, series_n_to_plot: int=1, 
    output_filepath: Path=Path('plot.png')):

    assert series_n_to_plot <= time_series.shape[0]

    # plt.figure(figsize=(12, 3))
    for srs in time_series[:series_n_to_plot]:
        plt.plot(srs, alpha=0.5)

    plt.title('Time Series')
    plt.xlabel('Time Index')
    plt.ylabel('Value')

    plt.savefig(output_filepath)

    plt.clf()
    plt.close()


def create_time_series_with_params_example_01() -> TimeSeriesParameters:
    """
    Generate time series data with specified parameters for trends, seasonality, 
        ARMA error, etc. and return the parameters and series packaged together
        in a dataclass
    """

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()

    time_n = 100
    series_n = 2
    constant = np.zeros(time_n)

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
    arma_scale = 20

    seed = 761824
    time_series = create_time_series(
        time_n, series_n, constant, 
        trend_n, trend_slope_min, trend_slope_max, 
        season_period, sin_amplitude, cos_amplitude, 
        ar_lag_coef, ma_lag_coef, arma_scale, seed)

    ts_params = TimeSeriesParameters(
        time_n, series_n, constant, trend_n, trend_slope_min, trend_slope_max,
        season_period, sin_amplitude, cos_amplitude, ar_lag_coef, ma_lag_coef,
        arma_scale, seed, time_series.trend_lengths, time_series.trend_slopes,
        time_series.time_series)

    return ts_params


def create_arma_coefficients(
    coefficient_n: int, 
    start_beta_0: int, start_beta_1: int, 
    decay_beta_0: int, decay_beta_1: int, 
    seed: int=753197) -> np.ndarray:
    """
    Generate ARMA coefficients with a start value and a decay factor chosen
        randomly from a beta distribution
    The first lag polynomial coefficient is always set to 1
    
    'coefficient_n' - the number of coefficients to generate, including the
        first coefficient which is always set to 1
    """

    assert start_beta_0 > 0
    assert start_beta_1 > 0
    assert decay_beta_0 > 0
    assert decay_beta_1 > 0

    rng = np.random.default_rng(seed)

    coef_start = rng.beta(start_beta_0, start_beta_1, 1)[0]
    coef_decay = rng.beta(decay_beta_0, decay_beta_1, 1)[0]
    lag_coef = np.array(
     [coef_start * coef_decay**i 
      for i in range(coefficient_n-1)])
    lag_coef = np.concatenate((np.array([1.]), lag_coef))

    return lag_coef


def create_time_series_with_params() -> TimeSeriesParameters:
    """
    Generate time series data with specified parameters for trends, seasonality, 
        ARMA error, etc. and return the parameters and series packaged together
        in a dataclass
    """

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()
    seed = 761824

    time_n = 1000
    series_n = 10
    constant = np.zeros(time_n)

    # trend parameters
    trend_n = 5
    trend_slope_min = -9 
    trend_slope_max = 9

    # season parameters
    rng = np.random.default_rng(seed)
    season_period = rng.integers(4, time_n//10, 1)[0]
    sin_amplitude = rng.integers(0, int(np.sqrt(time_n)), 1)[0]
    cos_amplitude = rng.integers(0, int(np.sqrt(time_n)), 1)[0]

    # ARMA parameters
    coef_n = 10
    ar_lag_coef = create_arma_coefficients(coef_n, 4, 2, 4, 2, seed+1) 
    ma_lag_coef = create_arma_coefficients(coef_n, 4, 2, 4, 2, seed+2) 
    arma_scale = rng.integers(0, int(np.sqrt(time_n)), 1)[0]

    time_series = create_time_series(
        time_n, series_n, constant, 
        trend_n, trend_slope_min, trend_slope_max, 
        season_period, sin_amplitude, cos_amplitude, 
        ar_lag_coef, ma_lag_coef, arma_scale, seed)

    ts_params = TimeSeriesParameters(
        time_n, series_n, constant, trend_n, trend_slope_min, trend_slope_max,
        season_period, sin_amplitude, cos_amplitude, ar_lag_coef, ma_lag_coef,
        arma_scale, seed, time_series.trend_lengths, time_series.trend_slopes,
        time_series.time_series)

    return ts_params


def main():

    ts_params = create_time_series_with_params()
    # ts_params = create_time_series_with_params_example_01()
    ts_params_df = convert_time_series_parameters_to_dataframe(ts_params)

    series_plot_n = max(10, ts_params.series_n)

    output_path = Path.cwd() / 'output'

    output_filepath = output_path / 'time_series.png'
    plot_time_series(
        ts_params.time_series, series_plot_n, output_filepath=output_filepath)

    time_idx_n = 100
    output_filepath = output_path / 'time_series_segment01.png'
    plot_time_series(
        ts_params.time_series[:, :time_idx_n], series_plot_n, 
        output_filepath=output_filepath)

    output_path = Path.cwd() / 'output'
    output_filepath = output_path / 'time_series.parquet'
    ts_params_df.write_parquet(output_filepath)

    output_filepath = output_path / 'time_series.csv'
    ts_params_df.write_csv(output_filepath)


if __name__ == '__main__':
    main()
