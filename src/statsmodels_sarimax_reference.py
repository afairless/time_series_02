# + [markdown]
'''
The primary purpose of this notebook is to serve as a convenient reference for 
    the various model internals/attributes/methods available for a SARIMAX model
    (which mostly applies to statsmodels time series state space models in 
    general)
The notebook fits a SARIMAX model to a synthetic time series and then produces
    various plots, metrics, and other print-outs about the model
'''
# - 

# +
import os
import subprocess
import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import fields

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as tsa_tools
import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.graphics.tsaplots as tsa_plots
# -

# +
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


def is_notebook() -> bool:
    """
    Return 'True' if the code is running in a Jupyter notebook; otherwise 
        return 'False'
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False


project_root_path = get_git_root_path()
assert isinstance(project_root_path, Path)
os.chdir(project_root_path)
# -

# +
if is_notebook():
    from IPython.display import display, Image
# - 

# +
# if __name__ == '__main__' and not is_notebook():
if __name__ == '__main__':

    from src.s01_generate_data import (
        TimeSeriesParameters,
        create_arma_coefficients,
        create_time_series,
        plot_time_series,
        )

    from src.common import (
        # get_git_root_path,
        plot_time_series_autocorrelation,
        write_list_to_text_file,
        )

else:

    from s01_generate_data import (
        TimeSeriesParameters,
        create_arma_coefficients,
        create_time_series,
        plot_time_series,
        )

    from common import (
        # get_git_root_path,
        plot_time_series_autocorrelation,
        write_list_to_text_file,
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


def plot_time_series_and_model_values_1(
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


def plot_time_series_and_model_values_2(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')):

    prediction = model_result.predict(start=0, end=len(original_series))

    plt.plot(original_series, alpha=0.5, color='blue')
    plt.plot(prediction, alpha=0.5, color='green')

    plt.title(
        'Original series (blue), model values (green), fit vs. forecast (red)')
    plt.axvline(x=len(model_result.fittedvalues), color='red', linestyle='--')
    plt.tight_layout()

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_time_series_and_model_values_3(
    original_series: np.ndarray, model_result: sarimax.SARIMAXResultsWrapper,
    output_filepath: Path=Path('plot.png')):

    simulations = model_result.simulate(
        nsimulations=len(original_series), repetitions=50).squeeze()

    plt.plot(simulations, alpha=0.1, color='green')
    plt.plot(original_series, alpha=0.9, color='blue')

    plt.title(
        'Original series (blue), simulated values based on model (green)')
    plt.axvline(x=len(model_result.fittedvalues), color='red', linestyle='--')
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
order = (2, 0, 2)

# seasonal_order = P, D, Q, period | AR, difference, MA, period
# seasonal_order = (0, 0, 0, 1)

model_1 = sarimax.SARIMAX(ts_train, order=order).fit()
assert isinstance(model_1, sarimax.SARIMAXResultsWrapper)
# -


# ## Plot original series and fitted values

# +
output_filepath = output_path / 'model_fit_and_forecast_1.png'
plot_time_series_and_model_values_1(ts, model_1, output_filepath)

if is_notebook():
    display(Image(output_filepath._str))
# - 

# +
output_filepath = output_path / 'model_fit_and_forecast_2.png'
plot_time_series_and_model_values_2(ts, model_1, output_filepath)

if is_notebook():
    display(Image(output_filepath._str))
# - 

# +
output_filepath = output_path / 'model_fit_and_forecast_3.png'
plot_time_series_and_model_values_3(ts, model_1, output_filepath)

if is_notebook():
    display(Image(output_filepath._str))
# - 

# +
output_filepath = output_path / 'model_diagnostics.png'
fig = model_1.plot_diagnostics()
plt.savefig(output_filepath)
plt.clf()
plt.close()

if is_notebook():
    display(Image(output_filepath._str))
# - 

# ## Autocorrelation

# + [markdown]
'''
These are the "true" AR and MA parameters used to create the original series
'''
# - 
# +
coef_str = 'lag_polynomial_coefficients'
for field in fields(ts_params):
    if coef_str in field.name:
        print(f"{field.name}: {getattr(ts_params, field.name)}")
# - 

# +
output_filepath = output_path / 'original_series_and_model_fit_autocorr.png'
ts_series_by_row = [ts[:len(model_1.fittedvalues)], model_1.fittedvalues]
plot_time_series_autocorrelation(ts_series_by_row, output_filepath)

if is_notebook():
    display(Image(output_filepath._str))
# -

# ## Summary

# +
model_1.summary()
# -
# +
model_1.specification
# -
# +
model_1._init_kwds
# -
# +
model_1.nobs
# -
# +
model_1.nobs_effective
# -
# +
model_1.nobs_diffuse
# -

# ## Parameters

# + [markdown]
'''
Parameters
'''
# +
model_1.param_terms
# - 
# +
model_1.params
# - 

# + [markdown]
'''
Standard errors of parameter estimates
'''
# +
model_1.bse
# -

# + [markdown]
'''
Confidence intervals of parameter estimates
'''
# +
model_1.conf_int(alpha=0.05)
# -
# +
model_1.params + 1.96 * model_1.bse
# -
# +
model_1.params - 1.96 * model_1.bse
# -

# +
model_1.fixed_params
# -

# + [markdown]
'''
Fitted model AR and MA parameters
'''
# - 
# +
model_1.arparams
# -
# +
model_1.polynomial_ar
# -
# +
model_1._params_ar
# -
# +
model_1.maparams
# -
# +
model_1.polynomial_ma
# -
# +
model_1._params_ma
# -
# +
model_1.seasonalarparams
# -
# +
model_1._params_seasonal_ar
# -
# +
model_1.polynomial_seasonal_ar
# -
# +
model_1.seasonalmaparams
# -
# +
model_1._params_seasonal_ma
# -
# +
model_1.polynomial_seasonal_ma
# -

# + [markdown]
'''
AR and MA roots
'''
# +
model_1.arroots
# -
# +
model_1.maroots
# -

# + [markdown]
'''
AR and MA root frequencies
'''
# +
model_1.arfreq
# -
# +
model_1.mafreq
# -

# + [markdown]
'''
Parameter hypothesis testing
'''
# +
model_1.zvalues
# -
# +
model_1.pvalues
# -
# +
model_1.use_t
# -
# +
model_1.tvalues
# -

# ## Metrics

# + [markdown]
'''
Mean absolute error
'''
# +
model_1.mae
# -
# +
np.abs((ts[:len(model_1.fittedvalues)] - model_1.fittedvalues)).mean()
# -

# + [markdown]
'''
Mean squared error
'''
# +
model_1.mse
# -
# +
model_1._params_variance
# -
# +
((ts[:len(model_1.fittedvalues)] - model_1.fittedvalues)**2).mean()
# -

# + [markdown]
'''
Summed squared error
'''
# +
model_1.sse
# -
# +
((ts[:len(model_1.fittedvalues)] - model_1.fittedvalues)**2).sum()
# -

# + [markdown]
'''
Akaike Information Criterion (AIC)
'''
# +
model_1.aic
# -
# +
model_1.info_criteria('aic', method='standard')
# -
# +
model_1.info_criteria('aic', method='lutkepohl')
# -

# + [markdown]
'''
Akaike Information Criterion (AIC) with small sample correction
'''
# +
model_1.aicc
# -

# + [markdown]
'''
Bayesian Information Criterion (BIC)
'''
# +
model_1.bic
# -
# +
model_1.info_criteria('bic', method='standard')
# -
# +
model_1.info_criteria('bic', method='lutkepohl')
# -

# + [markdown]
'''
Hannan-Quinn Information Criterion (BIC)
'''
# +
model_1.hqic
# -
# +
model_1.info_criteria('hqic', method='standard')
# -
# +
model_1.info_criteria('hqic', method='lutkepohl')
# -

# + [markdown]
'''
Log-Likelihood Function
'''
# +
model_1.llf
# -

# + [markdown]
'''
Log-Likelihood Function at each observation
'''
# +
len(model_1.llf_obs)
# -
# +
model_1.llf_obs
# -

# ## Residuals

# + [markdown]
'''
Residuals
'''
# +
model_1.resid.shape
# -
# +
model_1.resid
# -
# +
model_1.standardized_forecasts_error.shape
# -
# +
model_1.standardized_forecasts_error
# -
# +
model_1.resid / model_1.standardized_forecasts_error
# -

# ## Covariance

# +
model_1.cov_params()
# -
# +
model_1.cov_params_approx
# -
# +
model_1.cov_params_oim
# -
# +
model_1.cov_params_opg
# -
# +
model_1.cov_params_robust
# -
# +
model_1.cov_params_robust_approx
# -
# +
model_1.cov_params_robust_oim
# -
# +
model_1.cov_kwds
# -

# ## Diagnostic hypothesis tests

# +
model_1.test_serial_correlation(method='ljungbox')
# -
# +
model_1.test_serial_correlation(method='boxpierce')
# -
# +
model_1.test_heteroskedasticity(method='breakvar')
# -
# +
model_1.test_normality(method='jarquebera')
# -

# ## Degrees of freedom

# +
model_1.df_model
# -
# +
model_1.df_resid
# -

# ## States

# +
row_n = 4
dir(model_1.states)
# -
# +
model_1.df_resid
# -

# + [markdown]
'''
Filtered states
'''

# +
model_1.states.filtered.shape
# -
# +
model_1.states.filtered[:row_n, :]
# -
# +
model_1.states.filtered_cov.shape
# -
# +
model_1.states.filtered_cov[:row_n, :, :]
# -

# + [markdown]
'''
Predicted states
'''

# +
model_1.states.predicted.shape
# -
# +
model_1.states.predicted[:row_n, :]
# -
# +
model_1.states.predicted_cov.shape
# -
# +
model_1.states.predicted_cov[:row_n, :, :]
# -

# + [markdown]
'''
Smoothed states
'''

# +
model_1.states.smoothed.shape
# -
# +
model_1.states.smoothed[:row_n, :]
# -
# +
model_1.states.smoothed_cov.shape
# -
# +
model_1.states.smoothed_cov[:row_n, :, :]
# -

# + [markdown]
'''
Fitted values
'''

# +
len(model_1.fittedvalues)
# -
# +
model_1.forecasts == model_1.fittedvalues
# -
# +
(model_1.forecasts == model_1.fittedvalues).all()
# -

# ## Smoothed

# + [markdown]
'''
Smoothed state
'''

# +
model_1.states.smoothed.shape
# -
# +
model_1.smoothed_state.shape
# -
# +
(model_1.states.smoothed == model_1.smoothed_state.T).all()
# -
# +

# + [markdown]
'''
Smoothed state covariance
'''

# +
model_1.smoothed_state_cov.shape
# -
# +
model_1.states.smoothed_cov.shape
# -
# +
(model_1.smoothed_state_cov.transpose(2, 0, 1) == model_1.states.smoothed_cov).all()
# -
# +
model_1.scaled_smoothed_estimator.shape
# -
# +
model_1.scaled_smoothed_estimator_cov.shape
# -
# +
model_1.smoothed_measurement_disturbance.shape
# -
# +
model_1.smoothed_measurement_disturbance_cov.shape
# -
# +
model_1.smoothed_state_autocov.shape
# -
# +
model_1.smoothed_state_disturbance.shape
# -
# +
model_1.smoothing_error.shape
# -
# +
smoothed_results = model_1.get_smoothed_decomposition()
smoothed_results[0].shape
# -
# +
smoothed_results[0].head()
# -
# +
smoothed_results[1].shape
# -
# +
smoothed_results[1].head()
# -
# +
smoothed_results[2].shape
# -
# +
smoothed_results[2].head()
# -
# +
smoothed_results[3].shape
# -
# +
smoothed_results[3].head()
# -

# ## Miscellaneous

# +
model_1.k_constant
# -
# +
model_1.scale
# -
# +
model_1.k_diffuse_states
# -
# +
model_1.polynomial_trend
# -
# +
model_1._rank
# -
# +
model_1._init_kwds
# -

# +
# dir(model_1)
# model_1.wald_test()
# model_1.wald_test_terms()
# model_1.t_test()
# -

# ## TSA plots

# + [markdown]
'''
Autocorrelation function
'''
# +
tsa_plots.acf(x=ts)
# -
# +
output_filepath = output_path / 'autocorrelation.png'
fig = tsa_plots.plot_acf(x=ts)
plt.savefig(output_filepath)
plt.clf()
plt.close()

if is_notebook():
    display(Image(output_filepath._str))
# -

# + [markdown]
'''
Partial autocorrelation function
'''
# +
tsa_plots.pacf(x=ts)
# -
# +
output_filepath = output_path / 'partial_autocorrelation.png'
fig = tsa_plots.plot_pacf(x=ts)
plt.savefig(output_filepath)
plt.clf()
plt.close()

if is_notebook():
    display(Image(output_filepath._str))
# -

# + [markdown]
'''
Cross-correlation function, original series with model predictions
'''
# +
tsa_plots.ccf(x=ts, y=model_1.predict(start=0, end=len(ts)-1))
# - 
# +
output_filepath = output_path / 'crosscorrelation_original_and_predictions.png'
fig = tsa_plots.plot_ccf(x=ts, y=model_1.predict(start=0, end=len(ts)-1))
plt.savefig(output_filepath)
plt.clf()
plt.close()

if is_notebook():
    display(Image(output_filepath._str))
# -

# + [markdown]
'''
Cross-correlation function, original series with model residuals
'''
# +
tsa_plots.ccf(x=ts[:len(model_1.resid)], y=model_1.resid)
# - 
# +
output_filepath = output_path / 'crosscorrelation_original_and_residuals.png'
fig = tsa_plots.plot_ccf(x=ts[:len(model_1.resid)], y=model_1.resid)
plt.savefig(output_filepath)
plt.clf()
plt.close()

if is_notebook():
    display(Image(output_filepath._str))
# -

# ## TSA tools

# + [markdown]
'''
List of available tools
'''
# +
dir(tsa_tools)
# -
# +
dir(tsa_tools.stats)
# -
