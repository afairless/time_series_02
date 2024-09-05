#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots

if __name__ == '__main__':
    from s01_generate_data import (
        expand_values_by_lengths_into_vector,
        plot_time_series,
        )

else:
    from src.s01_generate_data import (
        expand_values_by_lengths_into_vector,
        plot_time_series,
        )


def write_list_to_text_file(
    a_list: list[str], text_filename: Path | str, overwrite: bool=False):
    """
    Writes a list of strings to a text file
    If 'overwrite' is 'True', any existing file by the name of 'text_filename'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file by the name of 'text_filename'

    :param a_list: a list of strings to be written to a text file
    :param text_filename: a string denoting the filepath or filename of text
        file
    :param overwrite: Boolean indicating whether to overwrite any existing text
        file or to append 'a_list' to that file's contents
    :return:
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    with open(text_filename, append_or_overwrite, encoding='utf-8') as txt_file:
        for e in a_list:
            txt_file.write(str(e))
            txt_file.write('\n')


def convert_path_to_relative_path_str(path: Path) -> str:
    return str(path).replace(str(Path.cwd()), '.')


def root_median_squared_error(y_true, y_pred):
    return np.sqrt(np.median((y_true - y_pred) ** 2))


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
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    plt.rcParams.update({'figure.figsize': (6, 4)})

    # autocorrelation

    output_filepath = output_path / 'ts_1st_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    # partial autocorrelation

    output_filepath = output_path / 'ts_1st_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    return md


def plot_time_series_autocorrelation(
    ts: list[np.ndarray], output_filepath: Path=Path('plot.png')):
    """

    Adapted from:
        https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    """

    plt.rcParams.update({'figure.figsize': (16, 3*len(ts))})

    fig, axes = plt.subplots(len(ts), 3, sharex=False)

    for i, _ in enumerate(ts):
        axes[i, 0].plot(ts[i]); axes[i, 0].set_title(f'Series #{i}')
        tsa_plots.plot_acf(ts[i], ax=axes[i, 1])
        tsa_plots.plot_pacf(ts[i], ax=axes[i, 2])

    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def exploratory01():

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   plus those 4 relative to benchmark (probably naive and seasonal naive) 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # statsmodels
    # skforecast
    # pmdarima

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'explore'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'model01.md'
    md = []
    md.append('# Classical ARIMA-style analysis/modeling')
    md.append('\n')

    # df.columns

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)


    row_idx = 0
    trend = extract_trend_from_time_series_dataframe(df, row_idx)

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    arma_colnames = [e for e in df.columns if 'lag_polynomial_coefficients' in e]


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
    md.append(f'![Image]({plot_filepath.name})')
    md.append('\n')

    detrend_ts = ts - trend
    plot_filepath = output_path / 'time_series_detrend.png'
    plot_time_series(detrend_ts.reshape(1, -1), 1, plot_filepath)
    md.append('Full detrended time series')
    md.append('\n')
    md.append(f'![Image]({plot_filepath.name})')
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
    md.append(f'![Image]({plot_filepath.name})')
    md.append('\n')

    detrend_ts_train = detrend_ts[:test_start_idx]
    plot_filepath = output_path / 'time_series_train_detrend.png'
    plot_time_series(detrend_ts_train.reshape(1, -1), 1, plot_filepath)
    md.append('Detrended time series training segment')
    md.append('\n')
    md.append(f'![Image]({plot_filepath.name})')
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

    output_path = input_path / 'model01' / 'sarima01'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'sarima01.md'
    md = []


    row_idx = 0
    # trend = extract_trend_from_time_series_dataframe(df, row_idx)

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    # detrend_ts = ts - trend
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
        'with seasonal and regular differencing (orange)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff_1, ts_train_season_diff_2]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series (Series #0), with seasonal differencing only '
        '(Series #1), and with seasonal and regular differencing (Series #2)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    md.append(
        'Time series with seasonal differencing only (Series #1) shows best '
        'autocorrelation profile')

    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    seasonal_order = (0, 0, 1, season_period)

    # dir(sarimax.diff())
    # sari = sarimax.SARIMAX(


    write_list_to_text_file(md, md_filepath, True)


def main():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

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

    ts_train_season_diff = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)



if __name__ == '__main__':
    exploratory01()
    exploratory02()
