#! /usr/bin/env python3

import numpy as np
from pandas.core.common import is_empty_slice
import polars as pl
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots

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


@dataclass
class TimeSeriesDifferencing:

    k_diff: int = 1
    k_seasonal_diff: int = 0
    seasonal_periods: int = 1
    original_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))
    seasonal_difference_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))
    final_difference_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))
    prepend_vector: np.ndarray = field(
        default_factory=lambda: np.array([]))


    def difference_time_series_seasonal(
        self, series: np.ndarray, k_seasonal_diff: int, seasonal_periods: int
        ) -> np.ndarray:
        """
        Apply seasonal differencing to given time series vector
        """

        for _ in range(k_seasonal_diff):
            self.prepend_vector = np.append(
                self.prepend_vector, series[:seasonal_periods])
            series = (
                series[seasonal_periods:] - 
                series[:-seasonal_periods])

        return series


    def difference_time_series(self, series: np.ndarray) -> np.ndarray:
        """
        Apply simple and/or seasonal differencing to given time series vector
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert len(series) >= max(self.k_diff, self.k_seasonal_diff) 
        if self.k_seasonal_diff > 0:
            assert self.seasonal_periods >= 1

        series = series.copy()
        series = series.reshape(-1)
        self.original_vector = series

        # seasonal differencing
        series = self.difference_time_series_seasonal(
            series, self.k_seasonal_diff, self.seasonal_periods)
        self.seasonal_difference_vector = series

        # simple/ordinary differencing
        diff_vectors = [
            np.diff(series, k, axis=0) for k in range(self.k_diff+1)]
        series = diff_vectors[-1] 
        prepends = np.array([dfv[0] for dfv in diff_vectors[:-1]])
        self.prepend_vector = np.append(self.prepend_vector, prepends)
        self.final_difference_vector = series

        return series


    def _difference_time_series_2(self, series: np.ndarray) -> np.ndarray:
        """
        Apply simple and/or seasonal differencing to given time series vector
        This function reverses the order of differencing:  simple, then 
            seasonal, instead of seasonal, then simple
        This reversal is used in tests to verify that either order of 
            differencing produces the same result
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert (self.k_diff + self.k_seasonal_diff) <= len(series)
        if self.k_seasonal_diff > 0:
            assert self.seasonal_periods >= 1

        series = series.copy()
        series = series.reshape(-1)
        self.original_vector = series

        # simple/ordinary differencing
        series = np.diff(series, self.k_diff, axis=0)
        self.final_difference_vector = series

        # seasonal differencing
        series = self.difference_time_series_seasonal(
            series, self.k_seasonal_diff, self.seasonal_periods)
        self.seasonal_difference_vector = series

        return series


    def periodic_cumulative_sum(
        self, series: np.ndarray, seasonal_periods: int) -> np.ndarray:
        """
        Calculate cumulative sums in a 1-dimensional array 'series' by position 
            in a period specified by 'seasonal_periods'
        If the length of 'series' is not divisible by 'seasonal_periods', the 
            function assumes that the 'extra', remainder elements are at the
            beginning of the array and should be excluded from the cumulative
            sum
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert len(series) > seasonal_periods

        remainder = len(series) % seasonal_periods
        remainder_elements = series[:remainder]
        cumsum_elements = series[remainder:]
        cum_sums_by_period = [
            np.cumsum(cumsum_elements[i::seasonal_periods]) 
            for i in range(seasonal_periods)]
        periodically_cumulated_array = np.vstack(cum_sums_by_period).flatten('F')
        periodic_cumsum = np.concatenate(
            [remainder_elements, periodically_cumulated_array])

        return periodic_cumsum  


    def de_difference_time_series(
        self, series: np.ndarray=np.array([])) -> np.ndarray:
        """
        "De-difference" given time series vector, i.e., back-transform the 
            series so that the "forward" differencing procedure is reversed
        """

        # INPUT PRE-CHECKS
        ##################################################

        if self.original_vector.size == 0:
            raise ValueError(
                'Original time series vector has not been provided; '
                'run method "difference_time_series" on the vector first')

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1
        # if (np.array(series.shape) > 1).sum() > 1:
        #     raise ValueError('Input series must be a 1-dimensional array')

        if series.size == 0:
            return self.original_vector

        if self.k_diff == 0 and self.k_seasonal_diff == 0:
            return series

        series = series.copy()
        series = series.reshape(-1)
        original_vector = self.original_vector.copy()
        original_vector = original_vector.reshape(-1)

        season_period_diff_len = self.k_seasonal_diff * self.seasonal_periods
        diff_total = self.k_diff + season_period_diff_len 
        assert (len(series) + diff_total) == len(original_vector)


        # DE-DIFFERENCE TIME SERIES
        ##################################################
        # there's probably an elegant recursive algorithm for this
        ##################################################

        # if the given series is the final difference vector, pass original
        #   difference vector along as the combined vector
        if np.allclose(self.final_difference_vector, series):
            combined_vector = series.copy()
        # otherwise, sum the given vector with the final difference vector, 
        #   i.e., the given vector modifies the original differences
        else:
            combined_vector = np.sum(
                [series, self.final_difference_vector], axis=0)

        # simple de-differencing
        p = None
        for p in range(-1, -self.k_diff-1, -1):
            prepend = np.array([self.prepend_vector[p]])
            prepend_vector = np.concatenate([prepend, combined_vector])
            combined_vector = np.cumsum(prepend_vector)

        # remove/"pop" used elements from 'prepend_vector'
        self.prepend_vector = self.prepend_vector[:p] 

        for _ in range(self.k_seasonal_diff):

            # end_idx = self.k_seasonal_diff * self.seasonal_periods
            end_idx = len(self.prepend_vector)
            start_idx = end_idx - self.seasonal_periods
            prepend = self.prepend_vector[start_idx:end_idx]
            prepend_vector = np.concatenate([prepend, combined_vector])
            combined_vector = self.periodic_cumulative_sum(
                prepend_vector, self.seasonal_periods)

            # remove/"pop" used elements from 'prepend_vector'
            self.prepend_vector = self.prepend_vector[:-self.seasonal_periods]

        return combined_vector  


    def _de_difference_time_series_0(
        self, series: np.ndarray=np.array([])) -> np.ndarray:
        """
        "De-difference" given time series vector, i.e., back-transform the 
            series so that the "forward" differencing procedure is reversed
        """

        # INPUT PRE-CHECKS
        ##################################################

        if self.original_vector.size == 0:
            raise ValueError(
                'Original time series vector has not been provided; '
                'run method "difference_time_series" on the vector first')

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1
        # if (np.array(series.shape) > 1).sum() > 1:
        #     raise ValueError('Input series must be a 1-dimensional array')

        if series.size == 0:
            return self.original_vector

        if self.k_diff == 0 and self.k_seasonal_diff == 0:
            return series

        series = series.copy()
        series = series.reshape(-1)
        original_vector = self.original_vector.copy()
        original_vector = original_vector.reshape(-1)

        season_period_diff_len = self.k_seasonal_diff * self.seasonal_periods
        diff_total = self.k_diff + season_period_diff_len 
        assert (len(series) + diff_total) == len(original_vector)


        # DE-DIFFERENCE TIME SERIES
        ##################################################
        # there's probably an elegant recursive algorithm for this
        ##################################################

        # apply simple and seasonal differencing to original vector to get 
        #   original difference vector
        diff_vector_seasonal = self.difference_time_series_seasonal(
            original_vector, self.k_seasonal_diff, self.seasonal_periods)
        diff_vector_0 = np.diff(diff_vector_seasonal, self.k_diff, axis=0)

        # if the given series is the original difference vector, pass original
        #   difference vector along as the combined vector
        if np.allclose(self.final_difference_vector, series):
            combined_vector = diff_vector_0 
        # otherwise, sum the given vector with the original difference vector, 
        #   i.e., the given vector modifies the original differences
        else:
            combined_vector = np.sum([series, diff_vector_0], axis=0)

        # simple de-differencing
        k_total = self.k_diff + self.k_seasonal_diff
        for k in range(k_total, self.k_seasonal_diff, -1):

            if self.k_seasonal_diff > 0:
                diff_vector = self.seasonal_difference_vector
            else:
                diff_vector = np.diff(original_vector, k-1, axis=0)
            prepend = np.array([diff_vector[0]])
            prepend_vector = np.concatenate([prepend, combined_vector])

            combined_vector = np.cumsum(prepend_vector)

        # seasonal de-differencing
        for k in range(self.k_seasonal_diff, 0, -1):

            diff_vector = self.difference_time_series_seasonal(
                original_vector, k_seasonal_diff=k-1, 
                seasonal_periods=self.seasonal_periods)
            prepend = np.array(
                [diff_vector[:self.seasonal_periods]]).reshape(-1)
            prepend_vector = np.concatenate([prepend, combined_vector])

            combined_vector = self.periodic_cumulative_sum(
                prepend_vector, self.seasonal_periods)

        return combined_vector  


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


def stationarity_tests(
    ts_train: np.ndarray, detrend_ts_train: np.ndarray, md: list[str]
    ) -> list[str]:

    # Augmented Dickey-Fuller unit root test')
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
    md.append('\n')
    md.append(
        '"The null hypothesis of the Augmented Dickey-Fuller is that there is '
        'a unit root, with the alternative that there is no unit root."')
    md.append('\n')

    adf = tsa_tools.adfuller(ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
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
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
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


    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    ################################################## 

    md.append('## Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test')
    md.append(
        'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html')
    md.append('\n')

    md.append(
        '"Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the '
        'null hypothesis that x is level or trend stationary."')

    kpss = tsa_tools.kpss(ts_train)
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
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
    md.append('\n')
    md.append('-##--##--##--##--##--##--##--##--##--##-')
    md.append('\n')
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

    return md


def plot_time_series_differenced_and_autocorrelation(
    ts: np.ndarray, output_filepath: Path=Path('plot.png')):
    """

    Adapted from:
        https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    """

    ts_srs = pl.DataFrame(ts)[:, 0]

    plt.rcParams.update({'figure.figsize': (16, 10)})

    fig, axes = plt.subplots(3, 3, sharex=False)

    # Original Series
    axes[0, 0].plot(ts_srs); axes[0, 0].set_title('Original Series')
    tsa_plots.plot_acf(ts_srs, ax=axes[0, 1], lags=len(ts_srs)-1)
    tsa_plots.plot_pacf(ts_srs, ax=axes[0, 2], lags=len(ts_srs)//4)
    # tsa_plots.plot_pacf(ts_srs, ax=axes[0, 2])

    # 1st Differencing
    ts_1diff = ts_srs.diff()
    axes[1, 0].plot(ts_1diff); axes[1, 0].set_title('1st Order Differencing')
    tsa_plots.plot_acf(ts_1diff.drop_nulls(), ax=axes[1, 1])
    tsa_plots.plot_pacf(
        ts_1diff.drop_nulls(), ax=axes[1, 2], lags=len(ts_srs)//4)

    # 2nd Differencing
    ts_2diff = ts_srs.diff()
    axes[2, 0].plot(ts_2diff); axes[2, 0].set_title('2nd Order Differencing')
    tsa_plots.plot_acf(ts_2diff.drop_nulls(), ax=axes[2, 1])
    tsa_plots.plot_pacf(
        ts_2diff.drop_nulls(), ax=axes[2, 2], lags=len(ts_srs)//4)

    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_differencing_and_autocorrelation(
    ts: np.ndarray, output_path: Path, md: list[str]) -> list[str]:

    # plot differencing and (partial) autocorrelation
    ################################################## 

    md.append('## Plot differencing and (partial) autocorrelation')
    md.append('\n')

    output_filepath = output_path / 'time_series_diff_autocorr.png'
    plot_time_series_differenced_and_autocorrelation(ts, output_filepath)
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    plt.rcParams.update({'figure.figsize': (6, 4)})

    # autocorrelation

    output_filepath = output_path / 'ts_1st_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_acf(ts_srs.diff().diff().diff().drop_nulls(), lags=120)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    # partial autocorrelation

    output_filepath = output_path / 'ts_1st_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_2nd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'ts_3rd_diff_p_autocorr.png'
    ts_srs = pl.DataFrame(ts)[:, 0]
    tsa_plots.plot_pacf(ts_srs.diff().diff().diff().drop_nulls(), lags=60)
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    return md


def plot_time_series_autocorrelation(
    ts: list[np.ndarray], output_filepath: Path=Path('plot.png')):
    """

    Adapted from:
        https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    """

    plt.rcParams.update({'figure.figsize': (16, 3*len(ts))})

    fig, axes = plt.subplots(len(ts), 3, sharex=False)

    for i, _ in enumerate(ts):
        axes[i, 0].plot(ts[i]); axes[i, 0].set_title(f'Series #{i}')
        tsa_plots.plot_acf(ts[i], ax=axes[i, 1])
        tsa_plots.plot_pacf(ts[i], ax=axes[i, 2])

    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def exploratory01():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'explore'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'model01.md'
    md = []
    md.append('# Classical ARIMA-style analysis/modeling')
    md.append('\n')


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


    # Augmented Dickey-Fuller unit root test')
    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    ################################################## 

    md = stationarity_tests(ts_train, detrend_ts_train, md)


    # plot differencing and (partial) autocorrelation, full series
    ################################################## 

    md = plot_differencing_and_autocorrelation(ts, output_path, md)


    # detrend and deseason time series train segment
    ################################################## 

    md.append('## Detrend and deseason time series train segment')
    md.append('\n')

    ts_train_srs = pl.DataFrame(ts_train)[:, 0]
    ts_train_1diff = ts_train_srs.diff().drop_nulls()
    adf = tsa_tools.adfuller(ts_train_1diff)
    kpss = tsa_tools.kpss(ts_train_1diff)
    md.append(
        'The 1st-order difference of the series training segment has ADF '
        f'p-value = {adf[1]:.3f} and KPSS p-value = {kpss[1]:.3f}, verifying '
        'a lack of trend')
    md.append('\n')


    write_list_to_text_file(md, md_filepath, True)


def exploratory02():
    # The ACF and PACF plots show the seasonality and the PACF shows notable
    #   down-spikes for several of the first lags
    # Because the data are synthetic, I know that the AR coefficients are
    #   substantially higher than the MA coefficients, but with both in play
    #   this isn't obvious from the plots, and this following advice doesn't 
    #   provide a firm direction

    # https://otexts.com/fpp3/non-seasonal-arima.html
    #
    # The data may follow an ARIMA(p,d, 0) model if the ACF and PACF plots of 
    #   the differenced data show the following patterns:
    #
    # the ACF is exponentially decaying or sinusoidal;
    # there is a significant spike at lag p in the PACF, but none beyond lag p.
    #
    # The data may follow an ARIMA(0,d, q) model if the ACF and PACF plots of 
    #   the differenced data show the following patterns:
    #
    # the PACF is exponentially decaying or sinusoidal;
    # there is a significant spike at lag q in the ACF, but none beyond lag q.

    # Based on the following discussion of seasonal ARMA terms, the PACF shows
    #   exponential decay while the ACF shows spikes at regular intervals, 
    #   suggesting a seasonal MA term is appropriate

    # https://otexts.com/fpp3/seasonal-arima.html
    #
    # The seasonal part of an AR or MA model will be seen in the seasonal lags 
    #   of the PACF and ACF. For example, an ARIMA(0,0,0)(0,0,1)12 model will 
    #   show:
    #
    # a spike at lag 12 in the ACF but no other significant spikes;
    # exponential decay in the seasonal lags of the PACF (i.e., at lags 12, 24, 36, …).
    #
    # Similarly, an ARIMA(0,0,0)(1,0,0)12 model will show:
    #
    # exponential decay in the seasonal lags of the ACF;
    # a single significant spike at lag 12 in the PACF.

    # According to this, one does seasonal differencing first:
    #   https://otexts.com/fpp3/seasonal-arima.html

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing01'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing01.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff_1 = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)
    ts_train_season_diff_2 = sarimax.diff(
        ts_train, k_diff=1, k_seasonal_diff=1, seasonal_periods=6)


    md.append('# Looking at differencing')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(ts_train_season_diff_1, alpha=0.5, color='green')
    plt.plot(ts_train_season_diff_2, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series (blue), with seasonal differencing only (green), and '
        'with seasonal and simple differencing (orange)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff_1, ts_train_season_diff_2]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series (Series #0), with seasonal differencing only '
        '(Series #1), and with seasonal and simple differencing (Series #2)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    md.append(
        'Time series with seasonal differencing only (Series #1) shows best '
        'autocorrelation profile')

    write_list_to_text_file(md, md_filepath, True)


def exploratory03():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing02'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing02.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff_0 = sarimax.diff(
        ts_train, k_diff=1, k_seasonal_diff=0, seasonal_periods=6)
    ts_train_season_diff_1 = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)


    md.append('# Looking at differencing')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train, alpha=0.5, color='blue')
    plt.plot(ts_train_season_diff_0, alpha=0.5, color='green')
    plt.plot(ts_train_season_diff_1, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series (blue), with simple differencing only (green), and '
        'with seasonal differencing only (orange)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [
        ts_train, ts_train_season_diff_0, ts_train_season_diff_1]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series (Series #0), with simple differencing only (Series #1), '
        'and with seasonal differencing only (Series #2)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    md.append(
        'The seasonal differencing only (Series #1) does better than the '
        'simple differencing')
    md.append('\n')

    md.append(
        'The PACF spikes at 6, 12, 18, and 24 (where 6 is the seasonal period) '
        'suggest a seasonal AR term.  The spikes at 18 and 24 are very small, '
        'and the spike at 12 is not large, so I am unsure how large the term '
        'should be.  Probably best to start at 1, see the effects, and then '
        'perhaps increase from there.')
    md.append('\n')
    md.append(
        'Both ACF and PACF show spikes at one, but only ACF spikes at 7. '
        'That might suggest a seasonal MA term of 2, or else a non-seasonal '
        'AR or MA term of 1, but it might be best again to apply one change at '
        'a time.')
    md.append('\n')

    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    write_list_to_text_file(md, md_filepath, True)


def exploratory04():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing03'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing03.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)

    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    fittedvalues = model.fittedvalues
    assert isinstance(fittedvalues, np.ndarray)


    md.append('# Looking at model fit on differenced time series')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(fittedvalues, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series with seasonal differencing only (blue), and fitted values '
        'from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(orange)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [ts_train_season_diff, fittedvalues]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series with seasonal differencing only (Series #0), and fitted '
        'values from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    md.append(
        'Adding the seasonal AR = 1 term with the fitted model nearly '
        'eliminated the seasonal PACF spikes.  Both ACF and PACF show some '
        'small spikes at 6 and 12, so an additional seasonal term of order 2 '
        'might help.  Additionally, both ACF and PACF show large spikes at 1, '
        'so a non-seasonal AR or MA term of order 1 would be useful. ')
    md.append('\n')

    # magnitudes of spike at 1 on ACF and PACF are nearly identical, so I don't
    #   think it matters much whether I put the term on AR or MA
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    write_list_to_text_file(md, md_filepath, True)


def exploratory05():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    output_path = input_path / 'model01' / 'differencing04'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'differencing04.md'
    md = []


    row_idx = 0

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    ts_train = ts[:test_start_idx]

    ts_train_season_diff = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)


    # model 1
    ##################################################
    order = (0, 0, 0)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_1 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    fittedvalues_1 = model_1.fittedvalues
    assert isinstance(fittedvalues_1, np.ndarray)


    # model 2
    ##################################################
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_2 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    fittedvalues_2 = model_2.fittedvalues
    assert isinstance(fittedvalues_2, np.ndarray)


    md.append('# Looking at model fit on differenced time series')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(fittedvalues_1, alpha=0.5, color='green')
    plt.plot(fittedvalues_2, alpha=0.5, color='orange')
    plt.title('Time series and seasonal differencing')
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    md.append(
        'Time series with seasonal differencing only (blue), and fitted values '
        'from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(green), and with p, d, q = 0, 0, 1 and P, D, Q = 1, 1, 0 (orange)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    output_filepath = output_path / 'time_series_season_diff_autocorr.png'
    ts_series_by_row = [ts_train_season_diff, fittedvalues_1, fittedvalues_2]
    plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

    md.append(
        'Time series with seasonal differencing only (Series #0), and fitted '
        'values from SARIMAX model with p, d, q = 0, 0, 0 and P, D, Q = 1, 1, 0 '
        '(Series #1), and with p, d, q = 0, 0, 1 and P, D, Q = 1, 1, 0 (Series '
        '#2)')
    md.append('\n')
    md.append(f'![Image]({output_filepath.name})')
    md.append('\n')

    md.append(
        'Adding the non-seasonal MA = 1 term made the ACF and PACF spikes at 1 '
        'smaller and reversed their direction, but it magnified spikes at 6 '
        'and 11, especially on the PACF. ')
    md.append('\n')

    md.append(
        'Also, the variability of the fitted values for both models is much '
        'smaller than for the differenced series, suggesting that a more '
        'complex model would be needed -- a conclusion supported by the small, '
        'persisting spikes in the ACF and PACF and by the parameters by which '
        'I created the synthetic data.')
    md.append('\n')

    write_list_to_text_file(md, md_filepath, True)


def main():

    input_path = Path.cwd() / 'output'
    input_filepath = input_path / f'time_series.parquet'
    df = pl.read_parquet(input_filepath)

    # df.columns
    # arma_colnames = [e for e in df.columns if 'lag_polynomial_coefficients' in e]

    # for e in df.columns:
    #     if 'constant' not in e and e[:3] != 'ts_':
    #         print(e)

    output_path = input_path / 'model01' / 'sarima02'
    output_path.mkdir(exist_ok=True, parents=True)

    md_filepath = output_path / 'sarima02.md'
    md = []

    row_idx = 0
    # trend = extract_trend_from_time_series_dataframe(df, row_idx)

    train_len = int(df[row_idx, 'time_n'] * 0.6)
    test_start_idx = train_len

    ts_colnames = [e for e in df.columns if e[:3] == 'ts_']
    ts = df[row_idx, ts_colnames].to_numpy().reshape(-1)
    # detrend_ts = ts - trend

    ts_train = ts[:test_start_idx]
    ts_test = ts[test_start_idx:]
    ts_train_season_diff = sarimax.diff(
        ts_train, k_diff=0, k_seasonal_diff=1, seasonal_periods=6)


    # model 2
    ##################################################
    # p, d, q = 0, 0, 1 
    order = (0, 0, 1)
    season_period = df[0, 'season_period']
    assert season_period == 6
    # seasonal, AR = 1, D = 1, MA = 0
    seasonal_order = (1, 1, 0, season_period)

    model_2 = sarimax.SARIMAX(
        ts_train_season_diff, order=order, seasonal_order=seasonal_order).fit()
    fittedvalues_2 = model_2.fittedvalues
    assert isinstance(fittedvalues_2, np.ndarray)

    dir(model_2)
    model_2._params_ar
    model_2._params_ma
    model_2._params_seasonal_ar
    model_2._params_seasonal_ma
    model_2.mse
    model_2.llf
    model_2.maroots
    model_2.seasonalarparams
    model_2.fixed_params
    model_2.get_smoothed_decomposition()
    ts_pred = model_2.forecast(steps=len(ts_test))

    output_filepath = output_path / 'time_series_predictions.png'
    plt.plot(ts_train_season_diff, alpha=0.5, color='blue')
    plt.plot(ts_test, alpha=0.5, color='green')
    plt.plot(ts_pred, alpha=0.5, color='orange')
    plt.title('Time series')
    plt.tight_layout()
    plt.show()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()



if __name__ == '__main__':

    # metrics:  RMSE, MAE, RMdSE, MdAE, 
    #   plus those 4 relative to benchmark (probably naive and seasonal naive) 
    #   maybe also relative to in-sample, i.e., scaled errors

    # sklearn
    # statsmodels
    # skforecast
    # pmdarima

    exploratory01()
    exploratory02()
    exploratory03()
    exploratory04()
    exploratory05()
