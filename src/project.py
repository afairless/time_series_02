#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.deterministic import DeterministicProcess


def dummy_function01() -> int:
    """
    This is a function
    """

    return 1


def dummy_function02(input: int) -> int:
    """
    This is a function
    """

    assert input > 0

    return input + 1


def randomize_segment_lengths(
    total_n: int, segment_n: int, seed=282840) -> np.ndarray:
    """
    Divide 'total_n' into 'segment_n' segments and randomize the lengths of the
        segments
    """

    segment_even_len = total_n // segment_n
    segment_len_vary = int(np.floor(segment_even_len * 0.9))

    np.random.seed(seed)
    segment_lens = np.random.uniform(
        segment_even_len - segment_len_vary, 
        segment_even_len + segment_len_vary, 
        segment_n)
    np.append(segment_lens, total_n - np.sum(segment_lens))

    assert (segment_lens > 0).all()

    return segment_lens


def main():

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()

    # alterations in series parameters:
    #   use trig function for trend to change slope
    #   arbitrarily flatten trend
    #   shift constant


    n_samples = 1000
    constant = np.zeros(n_samples)

    trend_n = 5
    trend_lens = randomize_segment_lengths(n_samples, trend_n)

    trend_slope = 0.05

    season_period = 500
    sin_amplitude = 10
    cos_amplitude = -10

    time_idx = np.arange(n_samples)
    trend = trend_slope * time_idx
    season_sin = sin_amplitude * np.sin(2 * np.pi * time_idx / season_period)
    season_cos = cos_amplitude * np.cos(2 * np.pi * time_idx / season_period)

    ar = np.array([1, 0.9])
    ma = np.array([1, 1.9])
    arma_process = ArmaProcess(ar, ma)

    np.random.seed(374352)
    arma_noise = arma_process.generate_sample(nsample=n_samples)

    time_series = constant + trend + season_sin + season_cos + arma_noise

    plt.clf()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time_series)
    plt.title('Generated Time Series with Trend, Seasonality, and ARMA Terms')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()




if __name__ == '__main__':
    main()
