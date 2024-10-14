#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots


try:
    from src.common import (
        TimeSeriesDifferencing,
        write_list_to_text_file,
        calculate_forecasts_and_metrics,
        plot_time_series,
        plot_time_series_autocorrelation,
        plot_time_series_and_model_values_2,
        )

except:
    from common import (
        TimeSeriesDifferencing,
        write_list_to_text_file,
        calculate_forecasts_and_metrics,
        plot_time_series,
        plot_time_series_autocorrelation,
        plot_time_series_and_model_values_2,
        )


def run_model(
    order: tuple[int, int, int],
    season_pdq: tuple[int, int, int],
    run_dir_name: str,
    train_diff: bool = False,
    simple_differencing: bool = False,
    hamilton_representation: bool = False):
    """
    """

    plt.rcParams.update({'figure.figsize': (16, 10)})

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model_s06' / run_dir_name
    output_path.mkdir(exist_ok=True, parents=True)


    # set up data
    ##################################################

    row_idx = 0
    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]
    ts_test = ts[test_start_idx:]


    # model
    ##################################################

    season_period = df[0, 'season_period']
    seasonal_order = (
        season_pdq[0], season_pdq[1], season_pdq[2], season_period)

    ts_diff_0 = TimeSeriesDifferencing(
        k_diff=order[1], k_seasonal_diff=seasonal_order[1], 
        seasonal_periods=season_period)
    ts_train_diff_model = ts_diff_0.difference_time_series(ts_train)

    if train_diff:
        train_data = ts_train_diff_model
    else:
        train_data = ts_train

    model_result = sarimax.SARIMAX(
        train_data, order=order, seasonal_order=seasonal_order,
        simple_differencing=simple_differencing, 
        hamilton_representation=hamilton_representation).fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)


    # save model results
    ##################################################

    prediction = model_result.predict(start=0, end=len(train_data)-1)
    if train_diff:
        prediction = ts_diff_0.de_difference_time_series(prediction)

    plot_srs = np.vstack([ts_train, prediction])

    output_filepath = output_path / 'original_and_predictions.png'
    plot_time_series(
        plot_srs, plot_srs.shape[0], output_filepath=output_filepath)


def main():

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    # best manual model
    # order, AR/p, d, MA/q
    # seasonal order, AR/P, D, MA/Q
    order = (0, 0, 1)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_no_simple_diff'
    train_diff = False
    simple_differencing = False
    hamilton_representation = False
    run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    # this produces the best-looking results


    # NO HAMILTON REPRESENTATION
    ##################################################

    # same settings as best model, except with simple differencing
    order = (0, 0, 1)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff'
    train_diff = False
    simple_differencing = True
    hamilton_representation = False
    run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    # this has poor results, as expected

    # same settings as simple differencing model, except with manual differncing
    order = (0, 0, 1)
    season_pdq = (0, 0, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff'
    train_diff = True
    simple_differencing = True
    hamilton_representation = False
    run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    # this has decent-looking results, though the de-differencing doesn't fully
    #   capture the original trends


    # HAMILTON REPRESENTATION
    ##################################################

    # same settings as best model, except with simple differencing
    order = (0, 0, 1)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff_hamilton'
    train_diff = False
    simple_differencing = True
    hamilton_representation = True
    run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    # this has poor results, as expected
    # Hamilton representation doesn't seem to matter much

    # same settings as simple differencing model, except with manual differncing
    order = (0, 0, 1)
    season_pdq = (0, 0, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff_hamilton'
    train_diff = True
    simple_differencing = True
    hamilton_representation = True
    run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    # this has decent-looking results, though the de-differencing doesn't fully
    #   capture the original trends
    # Hamilton representation doesn't seem to matter much

    # maybe save prediction vectors and compare hamilton and non-hamilton

if __name__ == '__main__':
    main()
