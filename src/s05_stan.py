#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from itertools import product as it_product

from cmdstanpy import CmdStanModel
import statsmodels.tsa.statespace.sarimax as sarimax

import arviz as az
import matplotlib.pyplot as plt

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


def calculate_gaussian_kernel_density_bandwidth_silverman_rule(
    df: pd.DataFrame) -> pd.Series:
    """
    Calculate Gaussian kernel density bandwidth based on Silverman's rule from:
        Silverman, B. W. (1986).  Density Estimation for Statistics and Data
            Analysis.  London: Chapman & Hall/CRC. p. 45
            ISBN 978-0-412-24620-3

    Wikipedia is a useful reference:

        https://en.wikipedia.org/wiki/Kernel_density_estimation

    :param df: a Pandas DataFrame where the Gaussian kernel density will be
        calculated for each column
    :return: scalar float representing bandwidth
    """

    # find interquartile range and divide it by 1.34
    iqr_div134 = (df.quantile(0.75) - df.quantile(0.25)) / 1.34

    # choose minimum of 'iqr_div134' and standard deviation for each variable
    a = pd.concat([iqr_div134, df.std()], axis=1).min(axis=1)

    h = 0.9 * a * len(df)**(-1/5)

    # check bandwidths/std on each variable

    return h


def resample_variables_by_gaussian_kernel_density(
    df: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    """
    For each column in Pandas DataFrame 'df', calculates a new sample of that
        variable based on Gaussian kernel density

    :param df: a Pandas DataFrame with columns of numerical data
    :param sample_n: the number of new samples to calculate for each column
    :return: a Pandas DataFrame with 'sample_n' rows and the same number of
        columns as 'df'
    """

    bandwidths = calculate_gaussian_kernel_density_bandwidth_silverman_rule(df)
    resample = df.sample(n=sample_n, replace=True)
    density_resample = np.random.normal(
        loc=resample, scale=bandwidths, size=(sample_n, df.shape[1]))

    density_resample = pd.DataFrame(density_resample, columns=df.columns)

    return density_resample


def create_grid_regular_intervals_two_variables(
    df: pd.DataFrame, intervals_num: int) -> pd.DataFrame:
    """
    1) Accepts Pandas DataFrame where first two columns are numerical values
    2) Finds the range of each of these columns and divides each range into
        equally spaced intervals; the number of intervals is specified by
        'intervals_num'
    3) Creates new DataFrame with two columns where the rows represent the
        Cartesian product of the equally spaced intervals

    :param df: a Pandas DataFrame where first two columns are numerical values
    :param intervals_num: scalar integer; the number of equally spaced intervals
        to create for each column
    :return:
    """

    intervals_df = df.apply(
        lambda d: np.linspace(start=d.min(), stop=d.max(), num=intervals_num))

    # the following code works much like 'expand.grid' in R, but it handles only
    #   two variables
    cartesian_product = list(
        it_product(intervals_df.iloc[:, 0], intervals_df.iloc[:, 1]))

    product_df = pd.DataFrame.from_records(
        cartesian_product, columns=df.columns)

    return product_df


def main():
    """
    """

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    plt.rcParams.update({'figure.figsize': (16, 10)})

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    src_path = Path.cwd() / 'src'

    output_path = input_path / 'model_s05'
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


    # 'best' manually-fit model
    ##################################################

    # order, AR/p, d, MA/q
    order = (0, 0, 1)
    # order = (1, 0, 0)
    season_period = df[0, 'season_period']
    # assert season_period == 6
    # seasonal order, AR/P, D, MA/Q
    season_pdq = (0, 1, 1)
    seasonal_order = (
        season_pdq[0], season_pdq[1], season_pdq[2], season_period)

    model_result = sarimax.SARIMAX(
        ts_train, order=order, seasonal_order=seasonal_order,
        simple_differencing=False).fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)

    md.append('## Model coefficients')
    md.append('\n')
    md.append(f'{str(model_result.param_terms)}')
    md.append('\n')
    md.append(f'{np.array2string(model_result.params)}')
    md.append('\n')


    # model
    ##################################################

    ts_diff_1 = TimeSeriesDifferencing(
        k_diff=order[1], k_seasonal_diff=seasonal_order[1], 
        seasonal_periods=season_period)
    ts_train_season_diff = ts_diff_1.difference_time_series(ts_train)
    
    # order, AR/p, d, MA/q
    order = (1, 0, 0)
    # order = (1, 0, 0)
    season_period = df[0, 'season_period']
    # assert season_period == 6
    # seasonal order, AR/P, D, MA/Q
    season_pdq = (0, 0, 0)
    seasonal_order = (
        season_pdq[0], season_pdq[1], season_pdq[2], season_period)

    model_result = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order,
        simple_differencing=False).fit()
    assert isinstance(model_result, sarimax.SARIMAXResultsWrapper)
    # model_result.summary()


    # 
    ##################################################


    stan_data = {
        'N': len(ts_train_season_diff), 
        'y': ts_train_season_diff}
    stan_filename = 's05_stan.stan'

    stan_filepath = src_path / stan_filename
    model = CmdStanModel(stan_file=stan_filepath)

    # fit_model = model.sample(
    #     data=stan_data, chains=2, thin=2, seed=22074,
    #     iter_warmup=100, iter_sampling=200, output_dir=output_path)
    fit_model = model.sample(
        data=stan_data, chains=4, thin=2, seed=21520,
        iter_warmup=4000, iter_sampling=8000, output_dir=output_path)
    # fit_model = model.sample(
    #     data=stan_data, chains=4, thin=1, seed=81398,
    #     iter_warmup=1000, iter_sampling=2000, output_dir=output_path)


    # tabular summaries
    ##################################################

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_df = fit_model.summary()
    output_filepath = output_path / 'summary.csv'
    fit_df.to_csv(output_filepath, index=True)


    # all samples for all parameters, predicted values, and diagnostics
    #   number of rows = number of 'iter_sampling' in 'CmdStanModel.sample' call
    draws_df = fit_model.draws_pd()
    output_filepath = output_path / 'draws.csv'
    draws_df.to_csv(output_filepath, index=True)






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

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   maybe also relative to in-sample, i.e., scaled errors

    # pyflux
    # sklearn
    # skforecast
    main()