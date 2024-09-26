#! /usr/bin/env python3

import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as tsa_plots


@dataclass
class TimeSeriesDifferencing:

    k_diff: int = 1
    k_seasonal_diff: int = 0
    seasonal_periods: int = 1
    original_vector: np.ndarray = field(
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

        return series


    def periodic_cumulative_sum(
        self, series: np.ndarray, seasonal_periods: int) -> np.ndarray:
        """
        Calculate cumulative sums in a 1-dimensional array 'series' by position 
            in a period specified by 'seasonal_periods'
        """

        # input series must be a 1-dimensional array
        assert (np.array(series.shape) > 1).sum() <= 1

        assert len(series) > seasonal_periods

        periods_n = len(series) - seasonal_periods
        periodic_cumsum = series[:seasonal_periods]
        for idx_1 in range(periods_n):
            idx_2 = idx_1 + seasonal_periods
            period_sum = np.array([periodic_cumsum[idx_1] + series[idx_2]])
            periodic_cumsum = np.concatenate([periodic_cumsum, period_sum])

        return periodic_cumsum  


    def de_difference_time_series(
        self, series: np.ndarray=np.array([])) -> np.ndarray:
        """
        "De-difference" given time series vector, i.e., back-transform the 
            series so that the "forward" differencing procedure is reversed
        'De-differencing' is often referred to as 'integrating', though Box, 
            Jenkins, and Reinsel (Time Series Analysis:  Forecasting and 
            Control, 3rd edition, Prentice Hall, Inc., 1994) on page 12 suggest
            that the better term is 'summing'

        NOTE:  as cumulative sums are added from the start to the end of the
            vector, error from floating-point imprecision accumulates
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
            # could return 'original_vector' here for speed, but continuing
            #   through rest of code provides an important debugging scenario
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


def get_git_root_path() -> Path | None:
    """
    Returns the top-level project directory where the Git repository is defined
    """

    try:
        # Run the git command to get the top-level directory
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], 
            stderr=subprocess.STDOUT)
        git_root_path = Path(git_root.decode('utf-8').strip())
        return git_root_path 

    except subprocess.CalledProcessError as e:
        print('Error while trying to find the Git root:', e.output.decode())
        return None


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

