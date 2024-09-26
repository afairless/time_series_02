
# +
import os
import subprocess
import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots


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


project_root_path = get_git_root_path()
assert isinstance(project_root_path, Path)
os.chdir(project_root_path)


if __name__ == '__main__':

    from s01_generate_data import (
        TimeSeriesParameters,
        create_arma_coefficients,
        create_time_series,
        plot_time_series,
        )

    from common import (
        # get_git_root_path,
        plot_time_series_autocorrelation,
        )

else:

    from src.s01_generate_data import (
        TimeSeriesParameters,
        create_arma_coefficients,
        create_time_series,
        plot_time_series,
        )

    from src.common import (
        # get_git_root_path,
        plot_time_series_autocorrelation,
        )
# -

# +
def create_time_series_with_params(
    seed: int=761824, series_n: int=1) -> TimeSeriesParameters:
    """
    Generate time series data with specified parameters for trends, seasonality, 
        ARMA error, etc. and return the parameters and series packaged together
        in a dataclass
    """

    # index = date_range('2000-1-1', freq='M', periods=240)
    # dtrm_process = DeterministicProcess(
    #     index=index, constant=True, period=3, order=2, seasonal=True)
    # dtrm_pd = dtrm_process.in_sample()

    time_n = 200
    constant = np.zeros(time_n)

    # trend parameters
    trend_n = 1
    trend_slope_min = 0 
    trend_slope_max = 0
    last_segment_len = -1

    # season parameters
    rng = np.random.default_rng(seed)
    season_period = 1
    sin_amplitude = 0
    cos_amplitude = 0

    # ARMA parameters
    coef_n = 3
    ar_lag_coef = create_arma_coefficients(coef_n, 4, 2, 4, 2, seed+1) 
    ma_lag_coef = create_arma_coefficients(coef_n, 4, 2, 4, 2, seed+2) 
    arma_scale = rng.integers(0, int(8*np.sqrt(time_n)), 1)[0]

    time_series = create_time_series(
        time_n, series_n, constant, 
        trend_n, last_segment_len, 
        trend_slope_min, trend_slope_max, 
        season_period, sin_amplitude, cos_amplitude, 
        ar_lag_coef, ma_lag_coef, arma_scale, seed)

    ts_params = TimeSeriesParameters(
        time_n, series_n, constant, trend_n, trend_slope_min, trend_slope_max,
        season_period, sin_amplitude, cos_amplitude, ar_lag_coef, ma_lag_coef,
        arma_scale, seed, time_series.trend_lengths, time_series.trend_slopes,
        time_series.time_series)

    return ts_params


def plot_time_series_and_model_values(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')):

    train_steps_n = len(model_result.fittedvalues)
    forecast_steps_n = len(original_series) - train_steps_n
    forecast = model_result.forecast(steps=forecast_steps_n)

    plt.plot(original_series, alpha=0.5, color='blue')
    plt.plot(model_result.fittedvalues, alpha=0.5, color='green')

    forecast_idx = range(train_steps_n, train_steps_n + forecast_steps_n)
    plt.plot(forecast_idx, forecast, alpha=0.5, color='orange')

    plt.title('Original series (blue), fitted values (green), forecast (orange)')
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
# -


# ## Set directories

# +
input_path = Path.cwd() / 'output'
input_filepath = input_path / 'time_series.parquet'
df = pl.read_parquet(input_filepath)

output_path = input_path / 'model01' / 'sarima02'
output_path.mkdir(exist_ok=True, parents=True)
# -


# ## Generate time series 

# +
ts_params = create_time_series_with_params(seed=258041)
ts = ts_params.time_series[0]
# -


# ## Pre-process series 

# +
train_len = int(len(ts) * 0.6)
test_start_idx = train_len

ts_train = ts[:test_start_idx]
ts_test = ts[test_start_idx:]
# -


# ## Fit model

# +
# order = p, d, q | AR, difference, MA
order = (0, 0, 1)

# seasonal_order = P, D, Q, period | AR, difference, MA, period
# seasonal_order = (0, 0, 0, 1)

model_1 = sarimax.SARIMAX(ts_train, order=order).fit()
assert isinstance(model_1, sarimax.SARIMAXResultsWrapper)
# -


# ## Plot original series and fitted values

# +
output_filepath = output_path / 'model_fit_and_forecast.png'
plot_time_series_and_model_values(ts, model_1, output_filepath)
output_filepath
# -

# +
Path.cwd()
# -

# + [markdown]
f'''
Here is the image

Here is the image
'''
# - 

coef_str = 'lag_polynomial_coefficients'
for field in fields(ts_params):
    if coef_str in field.name:
        print(f"{field.name}: {getattr(ts_params, field.name)}")

output_filepath = output_path / 'time_series_season_diff_autocorr.png'
ts_series_by_row = [ts_train_season_diff, fittedvalues_1, fittedvalues_2]
plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

dir(model_1)
model_1._params_ar
model_1._params_ma
model_1._params_seasonal_ar
model_1._params_seasonal_ma
model_1.mse
model_1.llf
model_1.maroots
model_1.seasonalarparams
model_1.fixed_params
model_1.get_smoothed_decomposition()

