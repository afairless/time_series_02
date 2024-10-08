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


def exploratory08():
    """
    """

    plt.rcParams.update({'figure.figsize': (16, 10)})

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model_s03' / 'exp01'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing.md'
    md = []


    # true AR and MA coefficients
    ##################################################

    row_idx = 0
    arma_colnames = [
        e for e in df.columns if 'lag_polynomial_coefficients' in e]
    true_coefs_1 = df[row_idx, arma_colnames].to_dict()
    true_coefs_2 = [(e, round(true_coefs_1[e][0], 2)) for e in true_coefs_1]
    true_coefs_3 = [
        e[0] + '        ' + str(e[1]) + '  \n' for e in true_coefs_2]

    md.append('## True AR and MA coefficients that generated original series')
    md.append('\n')
    for e in true_coefs_3:
        md.append(e)


    # set up data
    ##################################################

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]
    ts_test = ts[test_start_idx:]


    # model
    ##################################################

    # order, AR/p, d, MA/q
    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal order, AR/P, D, MA/Q
    seasonal_order = (1, 1, 0, season_period)

    model_result = sarimax.SARIMAX(
        ts_train, order=order, seasonal_order=seasonal_order).fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)


    # differencing
    ##################################################

    ts_diff = TimeSeriesDifferencing(
        k_diff=order[1], k_seasonal_diff=seasonal_order[1], 
        seasonal_periods=season_period)
    ts_train_season_diff = ts_diff.difference_time_series(ts_train)

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [ts_train, ts_train_season_diff]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append('## Original time series and differenced time series')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')


    # model results:  metrics
    ##################################################

    output_filepath = output_path / 'decomposition.png'
    forecast_df, metrics_df = calculate_forecasts_and_metrics(
        ts, test_start_idx, model_result, season_period, True, True, 
        output_filepath)

    md.append('## Model and naive forecast metrics')
    md.append('\n')
    # md.append(f'{metrics_df.round(2).to_string()}')
    # md.append('\n')
    md.append(f'{metrics_df.round(2).to_html()}')
    md.append('\n')
    md.append('## Original series decomposition')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'metrics.csv'
    metrics_df.to_csv(output_filepath)


    # model results:  plots
    ##################################################

    output_filepath = output_path / 'original_and_predictions.png'
    plot_time_series_and_model_values_2(ts, model_result, output_filepath)

    md.append('## Original time series and model predictions')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    output_filepath = output_path / 'original_and_forecasts.png'
    title = 'Time series and naive, seasonal naive, and model forecasts'
    original_and_forecasts = np.vstack([ts_test, forecast_df.values.T])
    original_and_forecasts = original_and_forecasts[[0, 3, 1, 2], :]
    plot_time_series(
        original_and_forecasts, original_and_forecasts.shape[0], title, 
        output_filepath)

    md.append('## Original time series forecast segment and forecasts')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    write_list_to_text_file(md, md_filepath, True)


def main():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)






if __name__ == '__main__':

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # statsmodels
    # skforecast
    # pmdarima
    exploratory08()
