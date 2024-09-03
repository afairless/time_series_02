#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt


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

    
    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']

    row_idx = 0
    ts = df[row_idx, ts_colnames]

    plt.clf()
    plt.close()

    # ts.to_numpy().shape
    plt.plot(ts.to_numpy().reshape(-1))
    plt.show()




if __name__ == '__main__':
    main()
