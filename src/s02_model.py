#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools

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

    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    plot_filepath = output_path / 'time_series.png'
    plot_time_series(ts.reshape(1, -1), 1, plot_filepath)
    md.append('Full time series')
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

    ts_train = ts[:test_start_idx]
    detrend_ts_train = detrend_ts[:test_start_idx]

    

    md.append(
        'https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html')

    tsa_tools.adfuller(ts_train)
    tsa_tools.adfuller(detrend_ts_train)





    write_list_to_text_file(md, md_filepath, True)


if __name__ == '__main__':
    main()
