#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools

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


def main():

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

    output_path = input_path / 'model01'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'model01.md'
    md = []
    md.append('# Classical ARIMA-style analysis/modeling')
    md.append('\n')

    df.columns

    for e in df.columns:
        if 'constant' not in e and e[:3] != 'ts_':
            print(e)


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

    # 
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
    md.append(
        '"The null hypothesis of the Augmented Dickey-Fuller is that there is '
        'a unit root, with the alternative that there is no unit root."')
    md.append('\n')

    adf = tsa_tools.adfuller(ts_train)
    md.append('--------------------')
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
    md.append('--------------------')
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


    # 
    ################################################## 
    # https://otexts.com/fpp3/stationarity.html
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    # https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

    md.append('### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test')
    md.append(
    'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html')
    md.append('\n')

    md.append(
        '"Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the '
        'null hypothesis that x is level or trend stationary."')

    kpss = tsa_tools.kpss(ts_train)
    md.append('--------------------')
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
    md.append('--------------------')
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




    write_list_to_text_file(md, md_filepath, True)


if __name__ == '__main__':
    main()
