#! /usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path

import matplotlib.pyplot as plt

import statsmodels.tsa.statespace.sarimax as sarimax


try:
    from src.common import (
        TimeSeriesDifferencing,
        plot_time_series,
        )

except:
    from common import (
        TimeSeriesDifferencing,
        plot_time_series,
        )


def run_model(
    order: tuple[int, int, int],
    season_pdq: tuple[int, int, int],
    run_dir_name: str,
    train_diff: bool = False,
    simple_differencing: bool = False,
    hamilton_representation: bool = False) -> np.ndarray:
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

    prediction_train = model_result.predict(start=0, end=len(train_data)-1)
    if train_diff:
        prediction_train = ts_diff_0.de_difference_time_series(prediction_train)

    plot_srs = np.vstack([ts_train, prediction_train])

    output_filepath = output_path / 'original_and_predictions.png'
    plot_time_series(
        plot_srs, plot_srs.shape[0], output_filepath=output_filepath)

    return prediction_train


def main():

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    predictions_train = []

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
    prediction_train = run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    predictions_train.append(prediction_train)
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
    prediction_train = run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    predictions_train.append(prediction_train)
    # this has poor results, as expected

    # same settings as simple differencing model, except with manual differencing
    order = (0, 0, 1)
    season_pdq = (0, 0, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff'
    train_diff = True
    simple_differencing = True
    hamilton_representation = False
    prediction_train = run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    predictions_train.append(prediction_train)
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
    prediction_train = run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    predictions_train.append(prediction_train)
    # this has poor results, as expected
    # Hamilton representation doesn't seem to matter much

    # same settings as simple differencing model, except with manual differencing
    order = (0, 0, 1)
    season_pdq = (0, 0, 1)
    order_str = ''.join([str(e) for e in (order + season_pdq)])
    run_dir_name = 'run_' + order_str + '_simple_diff_hamilton'
    train_diff = True
    simple_differencing = True
    hamilton_representation = True
    prediction_train = run_model(
        order, season_pdq, run_dir_name, train_diff, simple_differencing, 
        hamilton_representation)
    predictions_train.append(prediction_train)
    # this has decent-looking results, though the de-differencing doesn't fully
    #   capture the original trends
    # Hamilton representation doesn't seem to matter much


    # shows that the two corresponding model pairs with and without the Hamilton
    #   representation produce the same results
    assert np.allclose(predictions_train[1], predictions_train[3])
    assert np.allclose(predictions_train[2], predictions_train[4])


    '''
    Documentation on the Harvey and Hamilton representations:

    https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    This class allows two different underlying representations of ARMA models as 
        state space models: that of Hamilton and that of Harvey. Both are 
        equivalent in the sense that they are analytical representations of the 
        ARMA model, but the state vectors of each have different meanings. For 
        this reason, maximum likelihood does not result in identical parameter 
        estimates and even the same set of parameters will result in different 
        loglikelihoods.

    The Harvey representation is convenient because it allows integrating 
        differencing into the state vector to allow using all observations for 
        estimation.

    In this implementation of differenced models, the Hamilton representation is 
        not able to accommodate differencing in the state vector, so 
        simple_differencing (which performs differencing prior to estimation so 
        that the first d + sD observations are lost) must be used.

    Many other packages use the Hamilton representation, so that tests against 
        Stata and R require using it along with simple differencing (as Stata 
        does).
    '''

if __name__ == '__main__':
    main()
