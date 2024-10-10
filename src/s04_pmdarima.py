#! /usr/bin/env python3

import numpy as np
import polars as pl
import pmdarima as pm
from pathlib import Path

import matplotlib.pyplot as plt

import statsmodels.tsa.statespace.sarimax as sarimax


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
    """
    'This (1, 1, 2)(0, 0. 6, 6) model found by auto-ARIMA is '
    'interesting:  it fails to fully account for the seasonality, as shown '
    'by the regular ACF/PACF spikes, but it gets lower metrics than any of '
    'the simpler models I fit manually, including the best one '
    '(0, 0, 1)(0, 1, 1, 6).'
    """

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    plt.rcParams.update({'figure.figsize': (16, 10)})

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model_s04'
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


    # differencing
    ##################################################

    md.append('## Differencing tests')
    md.append('\n')

    diff = pm.arima.utils.ndiffs(ts_train, test='kpss', max_d=4)
    md.append('"d" estimated by KPSS')
    md.append(f'{diff}')
    md.append('\n')

    diff = pm.arima.utils.ndiffs(ts_train, test='adf', max_d=4)
    md.append('"d" estimated by ADF')
    md.append(f'{diff}')
    md.append('\n')

    diff = pm.arima.utils.ndiffs(ts_train, test='pp', max_d=4)
    md.append('"d" estimated by PP')
    md.append(f'{diff}')
    md.append('\n')

    diff = pm.arima.utils.nsdiffs(ts_train, m=6, test='ch', max_D=4)
    md.append('seasonal "D" estimated by CH')
    md.append(f'{diff}')
    md.append('\n')

    diff = pm.arima.utils.nsdiffs(ts_train, m=6, test='ocsb', max_D=4)
    md.append('seasonal "D" estimated by OCSB')
    md.append(f'{diff}')
    md.append('\n')


    # model
    ##################################################

    md.append('## Model results')
    md.append('\n')

    season_period = df[0, 'season_period']
    model_result = pm.auto_arima(
        ts_train, 
        max_p=6, max_q=6, max_P=6, max_Q=6, 
        seasonal=True, m=season_period, 
        stepwise=True, stationary=False, 
        information_criterion='aic', n_jobs=12, trace=True)

    order = model_result.order
    seasonal_order = model_result.seasonal_order

    md.append('model order')
    md.append(f'{order}')
    md.append('\n')

    md.append('model seasonal order')
    md.append(f'{seasonal_order}')
    md.append('\n')

    md.append('model AR params')
    md.append(f'{model_result.arparams()}')
    md.append('\n')

    md.append('model MA params')
    md.append(f'{model_result.maparams()}')
    md.append('\n')

    md.append('model params')
    md.append(f'{model_result.params()}')
    md.append('\n')

    md.append('model summary')
    md.append(f'{model_result.summary()}')
    md.append('\n')

    md.append('statsmodels SARIMAXResultsWrapper')
    md.append(f'{model_result.arima_res_}')
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
        model_result.fittedvalues())

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff, ts_train_fitted_diff]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append('## Original time series and differenced time series')
    md.append('\n')
    md.append('![Image](' + output_filepath.name + '){width=1024}')
    md.append('\n')

    md.append(
        'This (1, 1, 2)(0, 0. 6, 6) model found by auto-ARIMA is '
        'interesting:  it fails to fully account for the seasonality, as shown '
        'by the regular ACF/PACF spikes, but it gets lower metrics than any of '
        'the simpler models I fit manually, including the best one '
        '(0, 0, 1)(0, 1, 1, 6).')


    # model results:  metrics
    ##################################################

    output_filepath = output_path / 'decomposition.png'
    forecast_df, metrics_df = calculate_forecasts_and_metrics(
        ts, test_start_idx, model_result.arima_res_, season_period, True, True, 
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
    plot_time_series_and_model_values_2(
        ts, model_result.arima_res_, output_filepath)

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


if __name__ == '__main__':
    """
    This (1, 1, 2)(0, 0. 6, 6) model found by auto-ARIMA is 
    interesting:  it fails to fully account for the seasonality, as shown 
    by the regular ACF/PACF spikes, but it gets lower metrics than any of 
    the simpler models I fit manually, including the best one 
    (0, 0, 1)(0, 1, 1, 6).

    A few notes to remember about the 'pmdarima' package:

        1) 'model_result.arima_res_' wraps the statsmodels SARIMAX results
        2) fitting 'auto_arima' with 'stepwise=True' using Hyndman's method is 
          faster than the grid search used by 'stepwise=False'

          https://alkaline-ml.com/pmdarima/tips_and_tricks.html
          The stepwise approach follows the strategy laid out by Hyndman and 
          Khandakar in their 2008 paper, “Automatic Time Series Forecasting: 
          The forecast Package for R”.

        3) 'pmdarima' has some good differencing and de-differencing utilities
        4) 'pmdarima' has a context manager that can set limits on the model 
            fitting process
        5) the method for fitting the model can be set manually:

            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

            method : str, optional (default=’lbfgs’)

            The method determines which solver from scipy.optimize is used, and it can be chosen from among the following strings:

                ‘newton’ for Newton-Raphson
                ‘nm’ for Nelder-Mead
                ‘bfgs’ for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                ‘lbfgs’ for limited-memory BFGS with optional box constraints
                ‘powell’ for modified Powell’s method
                ‘cg’ for conjugate gradient
                ‘ncg’ for Newton-conjugate gradient
                ‘basinhopping’ for global basin-hopping solver
    """

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # skforecast
    main()
