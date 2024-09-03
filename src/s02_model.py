#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

from src.s01_generate_data import expand_values_by_lengths_into_vector


def root_median_squared_error(y_true, y_pred):
    return np.sqrt(np.median((y_true - y_pred) ** 2))


def main():

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   plus those 4 relative to benchmark (probably naive and seasonal naive) 
    #   maybe also relative to in-sample, i.e., scaled errors

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    df.columns

    for e in df.columns:
        if 'constant' not in e and e[:3] != 'ts_':
            print(e)


    row_idx = 0

    trend_len_colnames = [e for e in df.columns if e[:13] == 'trend_lengths']
    trend_slope_colnames = [e for e in df.columns if e[:12] == 'trend_slopes']

    trend_lengths = df[row_idx, trend_len_colnames].to_numpy()[0]
    trend_slopes = df[row_idx, trend_slope_colnames].to_numpy()[0]

    trend_slopes_extended = expand_values_by_lengths_into_vector(
        trend_slopes, trend_lengths)
    assert len(trend_slopes_extended) == df[row_idx, 'time_n']

    trend_slopes_extended[0] = 0
    trend = np.array(trend_slopes_extended).cumsum()

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts = df[row_idx, ts_colnames]

    plt.clf()
    plt.close()

    # ts.to_numpy().shape
    plt.plot(ts.to_numpy().reshape(-1))
    plt.show()




if __name__ == '__main__':
    main()
