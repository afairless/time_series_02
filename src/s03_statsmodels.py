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
    run_dir_name: str):
    """
    """

    plt.rcParams.update({'figure.figsize': (16, 10)})

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model_s03' / run_dir_name
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
    md.append('\n')


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
    # order = (0, 0, 0)
    season_period = df[0, 'season_period']
    # assert season_period == 6
    # seasonal order, AR/P, D, MA/Q
    # season_pdq = (1, 1, 0)
    seasonal_order = (
        season_pdq[0], season_pdq[1], season_pdq[2], season_period)

    model_result = sarimax.SARIMAX(
        ts_train, order=order, seasonal_order=seasonal_order).fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    md.append('## Model coefficients')
    md.append('\n')
    md.append(f'{str(model_result.param_terms)}')
    md.append('\n')
    md.append(f'{np.array2string(model_result.params)}')
    md.append('\n')


    # differencing
    ##################################################

    ts_diff_1 = TimeSeriesDifferencing(
        k_diff=order[1], k_seasonal_diff=seasonal_order[1], 
        seasonal_periods=season_period)
    ts_train_season_diff = ts_diff_1.difference_time_series(ts_train)

    ts_diff_2 = TimeSeriesDifferencing(
        k_diff=order[1], k_seasonal_diff=seasonal_order[1], 
        seasonal_periods=season_period)
    ts_train_fitted_diff = ts_diff_2.difference_time_series(
        model_result.fittedvalues)

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff, ts_train_fitted_diff]
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

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    order = (0, 0, 0)
    season_pdq = (0, 1, 0)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)
    # ACF and PACF show large spikes at seasonal period 6
    # PACF might show small decay at multiples of 6, whereas ACF does not, 
    #   suggesting we should add an MA term
    # https://otexts.com/fpp3/seasonal-arima.html
    #   9.9 Seasonal ARIMA models

    order = (0, 0, 0)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)
    # adding MA term reduces seasonal ACF/PACF spikes
    # both ACF and PACF show large spikes at lag 1, but it's unclear whether
    #   adding a non-seasonal AR or MA term is preferred


    # compare adding AR or MA term
    ##################################################

    # add AR term
    order = (1, 0, 0)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)

    # add MA term
    order = (0, 0, 1)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)

    # unclear which model to prefer:
    #  both models essentially eliminate the ACF/PACF spikes at lag 1
    #  AR-term model shows smaller spikes at 5 and 6 than MA-term model
    #  however, MA-term model shows lower error metrics than AR-term model

    # a conservative modeling approach might stop here, because there's little
    #   difference between the two models and the path forward isn't clear
    # but with the remaining spikes, it's worth trying to extend the two models


    # extend AR- and MA-term models
    ##################################################

    # add another AR term
    order = (2, 0, 0)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)

    # add another MA term
    order = (0, 0, 2)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)

    # both the AR(2) and MA(2) models above make the ACF/PACF spiking worse and 
    #   show worse error metrics, compared to their AR(1) and MA(1) counterparts
    #   (where all those models are also seasonal PDQ(0, 1, 1))

    # combine AR and MA terms
    order = (1, 0, 1)
    season_pdq = (0, 1, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str
    run_model(order, season_pdq, run_dir_name)

    # this pdq(1, 0, 1) PDQ(0, 1, 1) model has error metrics comparable to the
    #   pdq(1, 0, 0) PDQ(0, 1, 1) "AR(1)" model but worse than the pdq(0, 0, 1)
    #   PDQ(0, 1, 1) "MA(1)" model, which still has the best error metrics on
    #   the test data of all the models



if __name__ == '__main__':

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # statsmodels
    # skforecast
    # pmdarima
    main()
