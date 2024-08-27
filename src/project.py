#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.deterministic import DeterministicProcess

from pandas import date_range


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


def main():

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()

    n_samples = 1000
    trend_slope = 0.05
    seasonality_period = 500
    sin_amplitude = 10
    cos_amplitude = -10

    time_idx = np.arange(n_samples)
    trend = trend_slope * time_idx
    sin_seasonality = (
        sin_amplitude * np.sin(2 * np.pi * time_idx / seasonality_period))
    cos_seasonality = (
        cos_amplitude * np.cos(2 * np.pi * time_idx / seasonality_period))

    ar = np.array([1, 0.9])
    ma = np.array([1, 1.9])
    arma_process = ArmaProcess(ar, ma)

    np.random.seed(374352)
    arma_noise = arma_process.generate_sample(nsample=n_samples)

    time_series = trend + sin_seasonality + cos_seasonality + arma_noise

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
