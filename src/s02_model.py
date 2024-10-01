#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots

from sklearn.metrics import (
    root_mean_squared_error as skl_rmse, 
    mean_absolute_error as skl_mae, 
    median_absolute_error as skl_mdae)


if __name__ == '__main__':

    from src.s01_generate_data import (
        expand_values_by_lengths_into_vector,
        )

    from src.common import (
        TimeSeriesDifferencing,
        write_list_to_text_file,
        root_median_squared_error,
        plot_time_series_autocorrelation,
        plot_time_series,
        )

else:
    from s01_generate_data import (
        expand_values_by_lengths_into_vector,
        )

    from common import (
        TimeSeriesDifferencing,
        write_list_to_text_file,
        root_median_squared_error,
        plot_time_series_autocorrelation,
        plot_time_series,
        )


@dataclass
class TimeSeriesMetrics:
    rmse: float
    rmdse: float
    mae: float
    mdae: float


def extract_trend_from_time_series_dataframe(
    df: pl.DataFrame, row_idx: int) -> np.ndarray:

    trend_len_colnames = [e for e in df.columns if e[:13] == 'trend_lengths']
    trend_slope_colnames = [e for e in df.columns if e[:12] == 'trend_slopes']

    trend_lengths = df[row_idx, trend_len_colnames].to_numpy()[0]
    trend_slopes = df[row_idx, trend_slope_colnames].to_numpy()[0]

    trend_slopes_extended = expand_values_by_lengths_into_vector(
        trend_slopes, trend_lengths)
    assert len(trend_slopes_extended) == df[row_idx, 'time_n']

    trend_slopes_extended[0] = 0
    trend = np.array(trend_slopes_extended).cumsum()

    return trend


def calculate_time_series_metrics(
    series_1: np.ndarray, series_2: np.ndarray) -> TimeSeriesMetrics:
    """
    Calculate basic error metrics for two time series
    """

    rmse = skl_rmse(series_1, series_2).item()
    rmdse = root_median_squared_error(series_1, series_2).item()
    mae = skl_mae(series_1, series_2).item()
    mdae = skl_mdae(series_1, series_2).item()

    metrics = TimeSeriesMetrics(rmse, rmdse, mae, mdae)

    return metrics


def stationarity_tests(
    ts_train: np.ndarray, detrend_ts_train: np.ndarray, md: list[str]
    ) -> list[str]:

    # Augmented Dickey-Fuller unit root test')
    ################################################## 
    # https://otexts.com/fpp3/stationarity.html
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    # https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

    md.append('### Augmented Dickey-Fuller unit root test')
    md.append(
        'https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html')
    md.append('\n')

    md.append(
        '"The Augmented Dickey-Fuller test can be used to test for a unit root '
        'in a univariate process in the presence of serial correlation."')
    md.append('\n')
    md.append(
        '"The null hypothesis of the Augmented Dickey-Fuller is that there is '
        'a unit root, with the alternative that there is no unit root."')
    md.append('\n')

    adf = tsa_tools.adfuller(ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
    md.append('Time series training segment ADF')
    md.append('\n')
    md.append(
        f'ADF = {adf[0]}\n, p = {adf[1]}\n, number of lags used = {adf[2]}\n, '
        f'nobs = {adf[3]}\n')
    md.append(f'critical values = {adf[4]}')
    md.append('\n')
    md.append(
        'Null hypothesis that there is a unit root clearly not rejected; '
        'trend is present ')
    md.append('\n')

    adf = tsa_tools.adfuller(detrend_ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
    md.append('Detrended time series training segment ADF')
    md.append('\n')
    md.append(
        f'ADF = {adf[0]}\n, p = {adf[1]}\n, number of lags used = {adf[2]}\n, '
        f'nobs = {adf[3]}\n')
    md.append(f'critical values = {adf[4]}')
    md.append('\n')
    md.append(
        'Null hypothesis that there is a unit root clearly rejected; '
        'trend is not present.')
    md.append('\n')


    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    ################################################## 

    md.append('## Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test')
    md.append(
        'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html')
    md.append('\n')

    md.append(
        '"Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the '
        'null hypothesis that x is level or trend stationary."')

    kpss = tsa_tools.kpss(ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
    md.append('Time series training segment KPSS')
    md.append('\n')
    md.append(
        f'KPSS = {kpss[0]}\n, p = {kpss[1]}\n, lag truncation = {kpss[2]}\n')
    md.append(f'critical values = {kpss[3]}')
    md.append('\n')
    md.append(
        '<stdin>:1: InterpolationWarning: The test statistic is outside of '
        'the range of p-values available in the look-up table. The actual '
        'p-value is smaller than the p-value returned.')
    md.append('\n')
    md.append(
        'Null hypothesis of stationarity clearly rejected; trend is present ')
    md.append('\n')


    kpss = tsa_tools.kpss(detrend_ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
    md.append('Detrended Time series training segment KPSS')
    md.append('\n')
    md.append(
        f'KPSS = {kpss[0]}\n, p = {kpss[1]}\n, lag truncation = {kpss[2]}\n')
    md.append(f'critical values = {kpss[3]}')
    md.append('\n')
    md.append(
        '<stdin>:1: InterpolationWarning: The test statistic is outside of '
        'the range of p-values available in the look-up table. The actual '
        'p-value is greater than the p-value returned.')
    md.append('\n')
    md.append(
        'Null hypothesis of stationarity not rejected; trend is not present ')
    md.append('\n')

    return md


def plot_time_series_differenced_and_autocorrelation(
    ts: np.ndarray, output_filepath: Path=Path('plot.png')):
    """

    Adapted from:
        https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    """

    ts_srs = pl.DataFrame(ts)[:, 0]

    plt.rcParams.update({'figure.figsize': (16, 10)})

    fig, axes = plt.subplots(3, 3, sharex=False)

    # Original Series
    axes[0, 0].plot(ts_srs); axes[0, 0].set_title('Original Series')
    tsa_plots.plot_acf(ts_srs, ax=axes[0, 1], lags=len(ts_srs)-1)
    tsa_plots.plot_pacf(ts_srs, ax=axes[0, 2], lags=len(ts_srs)//4)
    # tsa_plots.plot_pacf(ts_srs, ax=axes[0, 2])

    # 1st Differencing
    ts_1diff = ts_srs.diff()
    axes[1, 0].plot(ts_1diff); axes[1, 0].set_title('1st Order Differencing')
    tsa_plots.plot_acf(ts_1diff.drop_nulls(), ax=axes[1, 1])
    tsa_plots.plot_pacf(
        ts_1diff.drop_nulls(), ax=axes[1, 2], lags=len(ts_srs)//4)

    # 2nd Differencing
    ts_2diff = ts_srs.diff()
    axes[2, 0].plot(ts_2diff); axes[2, 0].set_title('2nd Order Differencing')
    tsa_plots.plot_acf(ts_2diff.drop_nulls(), ax=axes[2, 1])
    tsa_plots.plot_pacf(
        ts_2diff.drop_nulls(), ax=axes[2, 2], lags=len(ts_srs)//4)

    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_differencing_and_autocorrelation(
    ts: np.ndarray, output_path: Path, md: list[str]) -> list[str]:

    # plot differencing and (partial) autocorrelation
    ################################################## 

    md.append('## Plot differencing and (partial) autocorrelation')
    md.append('\n')

    output_filepath = output_path / 'time_series_diff_autocorr.png'
    plot_time_series_differenced_and_autocorrelation(ts, output_filepath)
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    plt.rcParams.update({'figure.figsize': (6, 4)})

    # autocorrelation

    output_filepath = output_path / 'ts_1st_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    # partial autocorrelation

    output_filepath = output_path / 'ts_1st_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    return md


def exploratory01():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'explore'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'model01.md'
    md = []
    md.append('# Classical ARIMA-style analysis/modeling')
    md.append('\n')


    row_idx = 0
    trend = extract_trend_from_time_series_dataframe(df, row_idx)

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']


    # plot full time series
    ################################################## 

    md.append('## Plot full time series')
    md.append('\n')

    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    plot_filepath = output_path / 'time_series.png'
    ts_and_trend = np.vstack([ts, trend])
    plot_time_series(ts_and_trend, 2, plot_filepath)
    md.append('Full time series with true trend')
    md.append('\n')
    md.append('![Image](' + plot_filepath.name + '){width=1024}')
    md.append('\n')

    detrend_ts = ts - trend
    plot_filepath = output_path / 'time_series_detrend.png'
    plot_time_series(detrend_ts.reshape(1, -1), 1, plot_filepath)
    md.append('Full detrended time series')
    md.append('\n')
    md.append('![Image](' + plot_filepath.name + '){width=1024}')
    md.append('\n')


    # plot time series training segment
    ################################################## 

    md.append('## Plot time series training segment')
    md.append('\n')

    ts_train = ts[:test_start_idx]
    plot_filepath = output_path / 'time_series_train.png'
    plot_time_series(ts_train.reshape(1, -1), 1, plot_filepath)
    md.append('Time series training segment')
    md.append('\n')
    md.append('![Image](' + plot_filepath.name + '){width=1024}')
    md.append('\n')

    detrend_ts_train = detrend_ts[:test_start_idx]
    plot_filepath = output_path / 'time_series_train_detrend.png'
    plot_time_series(detrend_ts_train.reshape(1, -1), 1, plot_filepath)
    md.append('Detrended time series training segment')
    md.append('\n')
    md.append('![Image](' + plot_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(f'Length of time series training segment:  {len(ts_train)}')
    md.append('\n')

    md.append(
        'https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html')
    md.append('\n')


    # Augmented Dickey-Fuller unit root test')
    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    ################################################## 

    md = stationarity_tests(ts_train, detrend_ts_train, md)


    # plot differencing and (partial) autocorrelation, full series
    ################################################## 

    md = plot_differencing_and_autocorrelation(ts, output_path, md)


    # detrend and deseason time series train segment
    ################################################## 

    md.append('## Detrend and deseason time series train segment')
    md.append('\n')

    ts_train_srs = pl.DataFrame(ts_train)[:, 0]
    ts_train_1diff = ts_train_srs.diff().drop_nulls()
    adf = tsa_tools.adfuller(ts_train_1diff)
    kpss = tsa_tools.kpss(ts_train_1diff)
    md.append(
        'The 1st-order difference of the series training segment has ADF '
        f'p-value = {adf[1]:.3f} and KPSS p-value = {kpss[1]:.3f}, verifying '
        'a lack of trend')
    md.append('\n')


    write_list_to_text_file(md, md_filepath, True)


def exploratory02():
    # The ACF and PACF plots show the seasonality and the PACF shows notable
    #   down-spikes for several of the first lags
    # Because the data are synthetic, I know that the AR coefficients are
    #   substantially higher than the MA coefficients, but with both in play
    #   this isn't obvious from the plots, and this following advice doesn't 
    #   provide a firm direction

    # https://otexts.com/fpp3/non-seasonal-arima.html
    #
    # The data may follow an ARIMA(p,d, 0) model if the ACF and PACF plots of 
    #   the differenced data show the following patterns:
    #
    # the ACF is exponentially decaying or sinusoidal;
    # there is a significant spike at lag p in the PACF, but none beyond lag p.
    #
    # The data may follow an ARIMA(0,d, q) model if the ACF and PACF plots of 
    #   the differenced data show the following patterns:
    #
    # the PACF is exponentially decaying or sinusoidal;
    # there is a significant spike at lag q in the ACF, but none beyond lag q.

    # Based on the following discussion of seasonal ARMA terms, the PACF shows
    #   exponential decay while the ACF shows spikes at regular intervals, 
    #   suggesting a seasonal MA term is appropriate

    # https://otexts.com/fpp3/seasonal-arima.html
    #
    # The seasonal part of an AR or MA model will be seen in the seasonal lags 
    #   of the PACF and ACF. For example, an ARIMA(0,0,0)(0,0,1)12 model will 
    #   show:
    #
    # a spike at lag 12 in the ACF but no other significant spikes;
    # exponential decay in the seasonal lags of the PACF (i.e., at lags 12, 24, 36, …).
    #
    # Similarly, an ARIMA(0,0,0)(1,0,0)12 model will show:
    #
    # exponential decay in the seasonal lags of the ACF;
    # a single significant spike at lag 12 in the PACF.

    # According to this, one does seasonal differencing first:
    #   https://otexts.com/fpp3/seasonal-arima.html

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing01'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing01.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff_1 = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_train_season_diff_2 = sarimax.diff(
        ts_train, k_diff=1, k_seasonal_diff=1, seasonal_periods=6)


    md.append('# Looking at differencing')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(ts_train_season_diff_1, alpha=0.5, color='green')
    plt.plot(ts_train_season_diff_2, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series (blue), with seasonal differencing only (green), and '
        'with seasonal and simple differencing (orange)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff_1, ts_train_season_diff_2]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series (Series #0), with seasonal differencing only '
        '(Series #1), and with seasonal and simple differencing (Series #2)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(
        'Time series with seasonal differencing only (Series #1) shows best '
        'autocorrelation profile')

    write_list_to_text_file(md, md_filepath, True)


def exploratory03():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing02'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing02.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff_0 = sarimax.diff(
        ts_train, k_diff=1, k_seasonal_diff=0, seasonal_periods=6)
    ts_train_season_diff_1 = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)


    md.append('# Looking at differencing')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(ts_train_season_diff_0, alpha=0.5, color='green')
    plt.plot(ts_train_season_diff_1, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series (blue), with simple differencing only (green), and '
        'with seasonal differencing only (orange)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff_0, ts_train_season_diff_1]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series (Series #0), with simple differencing only (Series #1), '
        'and with seasonal differencing only (Series #2)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(
        'The seasonal differencing only (Series #1) does better than the '
        'simple differencing')
    md.append('\n')

    md.append(
        'The PACF spikes at 6, 12, 18, and 24 (where 6 is the seasonal period) '
        'suggest a seasonal AR term.  The spikes at 18 and 24 are very small, '
        'and the spike at 12 is not large, so I am unsure how large the term '
        'should be.  Probably best to start at 1, see the effects, and then '
        'perhaps increase from there.')
    md.append('\n')
    md.append(
        'Both ACF and PACF show spikes at one, but only ACF spikes at 7. '
        'That might suggest a seasonal MA term of 2, or else a non-seasonal '
        'AR or MA term of 1, but it might be best again to apply one change at '
        'a time.')
    md.append('\n')

    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    write_list_to_text_file(md, md_filepath, True)


def exploratory04():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing03'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing03.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    # ts_train_season_diff = sarimax.diff(
    #     ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_diff = TimeSeriesDifferencing(
        k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_train_season_diff = ts_diff.difference_time_series(ts_train)

    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    fittedvalues = model.fittedvalues
    assert isinstance(fittedvalues, np.ndarray)


    md.append('# Looking at model fit on differenced time series')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(fittedvalues, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series with seasonal differencing only (blue), and fitted values '
        'from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(orange)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [ts_train, fittedvalues]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series with seasonal differencing only (Series #0), and fitted '
        'values from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(
        'Adding the seasonal AR = 1 term with the fitted model nearly '
        'eliminated the seasonal PACF spikes.  Both ACF and PACF show some '
        'small spikes at 6 and 12, so an additional seasonal term of order 2 '
        'might help.  Additionally, both ACF and PACF show large spikes at 1, '
        'so a non-seasonal AR or MA term of order 1 would be useful. ')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_dediff.png'
    fittedvalues_dediff = ts_diff.de_difference_time_series(fittedvalues)
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(fittedvalues_dediff, alpha=0.5, color='orange')
    plt.title('Time series, with de-differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Original time series (blue), and de-differenced fitted values from '
        'SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    # magnitudes of spike at 1 on ACF and PACF are nearly identical, so I don't
    #   think it matters much whether I put the term on AR or MA
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    write_list_to_text_file(md, md_filepath, True)


def exploratory05():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing04'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing04.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    # ts_train_season_diff = sarimax.diff(
    #     ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_diff = TimeSeriesDifferencing(
        k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_train_season_diff = ts_diff.difference_time_series(ts_train)


    # model 1
    ##################################################
    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_1 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    assert isinstance(model_1, sarimax.SARIMAXResultsWrapper)
    fittedvalues_1 = model_1.fittedvalues


    # model 2
    ##################################################
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_2 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    assert isinstance(model_2, sarimax.SARIMAXResultsWrapper)
    fittedvalues_2 = model_2.fittedvalues


    md.append('# Looking at model fit on differenced time series')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(fittedvalues_1, alpha=0.5, color='green')
    plt.plot(fittedvalues_2, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series with seasonal differencing only (blue), and fitted values '
        'from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(green), and with p, d, q = 0, 0, 1 and P, D, Q = 1, 1, 0 (orange)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [ts_train_season_diff, fittedvalues_1, fittedvalues_2]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series with seasonal differencing only (Series #0), and fitted '
        'values from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1), and with p, d, q = 0, 0, 1 and P, D, Q = 1, 1, 0 (Series '
        '#2)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(
        'Adding the non-seasonal MA = 1 term made the ACF and PACF spikes at 1 '
        'smaller and reversed their direction, but it magnified spikes at 6 '
        'and 11, especially on the PACF. ')
    md.append('\n')

    md.append(
        'Also, the variability of the fitted values for both models is much '
        'smaller than for the differenced series, suggesting that a more '
        'complex model would be needed -- a conclusion supported by the small, '
        'persisting spikes in the ACF and PACF and by the parameters by which '
        'I created the synthetic data.')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_dediff.png'
    fittedvalues_dediff_1 = ts_diff.de_difference_time_series(fittedvalues_1)
    fittedvalues_dediff_2 = ts_diff.de_difference_time_series(fittedvalues_2)
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(fittedvalues_dediff_1, alpha=0.5, color='green')
    plt.plot(fittedvalues_dediff_2, alpha=0.5, color='orange')
    plt.title('Time series, with de-differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Original time series (blue), and de-differenced fitted values from '
        'SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1)')
    md.append(
        'Original time series with seasonal differencing only (blue), and '
        'de-differenced fitted values from SARIMAX model with p, d, q = 0, 0, 0 '
        ' and P, D, Q = 1, 1, 0 (green), and with p, d, q = 0, 0, 1 and '
        'P, D, Q = 1, 1, 0 (orange)')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    write_list_to_text_file(md, md_filepath, True)


def main():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    # df.columns
    # arma_colnames = [e for e in df.columns if 'lag_polynomial_coefficients' in e]

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    output_path = input_path / 'model01' / 'sarima02'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'sarima02.md'
    md = []

    row_idx = 0
    # trend = extract_trend_from_time_series_dataframe(df, row_idx)

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    # detrend_ts = ts - trend

    ts_train = ts[:test_start_idx]
    ts_test = ts[test_start_idx:]
    ts_train_season_diff = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)


    # model 1
    ##################################################
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_1 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    assert isinstance(model_1, sarimax.SARIMAXResultsWrapper)
    fitted_values_1 = model_1.fittedvalues
    forecast_values_1 = model_1.forecast(steps=len(ts_test))

    len_diff = len(ts_train_season_diff) - len(fitted_values_1)
    train_metrics = calculate_time_series_metrics(
        ts_train_season_diff[len_diff:], fitted_values_1)
    assert train_metrics.rmse == np.sqrt(model_1.mse)
    assert train_metrics.mae == model_1.mae
    test_metrics = calculate_time_series_metrics(
        ts_test, forecast_values_1)


    dir(model_1)
    print('\n')

    output_filepath = output_path / 'time_series_predictions.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(ts_test, alpha=0.5, color='green')
    plt.plot(ts_pred, alpha=0.5, color='orange')
    plt.title('Time series')
    plt.tight_layout()
    plt.show()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()



if __name__ == '__main__':

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   plus those 4 relative to benchmark (probably naive and seasonal naive) 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # statsmodels
    # skforecast
    # pmdarima

    exploratory01()
    exploratory02()
    exploratory03()
    exploratory04()
    exploratory05()
