## import robyn ## Manual, no need
from python import data, inputs, checks, model ## Manual, Added manually
from python import transformation
import numpy as np
import pandas as pd

# Load data
## dt_simulated_weekly = robyn.data("dt_simulated_weekly")
## dt_prophet_holidays = robyn.data("dt_prophet_holidays")

dt_simulated_weekly = data.dt_simulated_weekly()
dt_prophet_holidays = data.dt_prophet_holidays()

# Define input variables
## manually converted to
input_collect = inputs.robyn_inputs(
    dt_input=dt_simulated_weekly,
    dt_holidays=dt_prophet_holidays,
    date_var="DATE",
    dep_var="revenue",
    dep_var_type="revenue",
    prophet_vars=["trend", "season", "holiday"], ## Manually added list
    prophet_country="DE",
    context_vars=["competitor_sales_B", "events"],
    paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
    paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
    organic_vars=["newsletter"], ## Manually converted to list, since R behaves single variable as list as well.
    window_start="2016-01-01",
    window_end="2018-12-31",
    adstock="geometric"
)

# Print input collection
print(input_collect)

# Define and add hyperparameters
hyper_names = inputs.hyper_names(adstock=input_collect['robyn_inputs']['adstock'], all_media=input_collect['robyn_inputs']['all_media'])

## Manually added
##pads_stock1, pads_stock2 = transformation.plot_adstock(plot = False)
transformation.plot_adstock(plot = False)
##psaturation1, psaturation2 =
transformation.plot_saturation(plot = False)

## Manually added
checks.hyper_limits()

# Define hyperparameters ranges
facebook_S_alphas = np.array([0.5, 3])
facebook_S_gammas = np.array([0.3, 1])
facebook_S_thetas = np.array([0, 0.3])
print_S_alphas = np.array([0.5, 3])
print_S_gammas = np.array([0.3, 1])
print_S_thetas = np.array([0.1, 0.4])
tv_S_alphas = np.array([0.5, 3])
tv_S_gammas = np.array([0.3, 1])
tv_S_thetas = np.array([0.3, 0.8])
search_S_alphas = np.array([0.5, 3])
search_S_gammas = np.array([0.3, 1])
search_S_thetas = np.array([0, 0.3])
ooh_S_alphas = np.array([0.5, 3])
ooh_S_gammas = np.array([0.3, 1])
ooh_S_thetas = np.array([0.1, 0.4])
newsletter_alphas = np.array([0.5, 3])
newsletter_gammas = np.array([0.3, 1])
newsletter_thetas = np.array([0.1, 0.4])
train_size = np.array([0.5, 0.8])

hyperparameters = pd.DataFrame({
        'facebook_S_alphas': facebook_S_alphas,
        'facebook_S_gammas': facebook_S_gammas,
        'facebook_S_thetas': facebook_S_thetas,
        'print_S_alphas': print_S_alphas,
        'print_S_gammas': print_S_gammas,
        'print_S_thetas': print_S_thetas,
        'tv_S_alphas': tv_S_alphas,
        'tv_S_gammas': tv_S_gammas,
        'tv_S_thetas': tv_S_thetas,
        'search_S_alphas': search_S_alphas,
        'search_S_gammas': search_S_gammas,
        'search_S_thetas': search_S_thetas,
        'ooh_S_alphas': ooh_S_alphas,
        'ooh_S_gammas': ooh_S_gammas,
        'ooh_S_thetas': ooh_S_thetas,
        'newsletter_alphas': newsletter_alphas,
        'newsletter_gammas': newsletter_gammas,
        'newsletter_thetas': newsletter_thetas,
        'train_size': train_size
    })

# Define InputCollect
## Manually converted, parameters defined wrong.
input_collect = inputs.robyn_inputs(
    InputCollect = input_collect['robyn_inputs'],
    hyperparameters = hyperparameters
)

# Print InputCollect
print(input_collect)

calibration_input = pd.DataFrame({
  # channel name must in paid_media_vars
  "channel": ["facebook_S",  "tv_S", "facebook_S+search_S", "newsletter"],
  # liftStartDate must be within input data range
  "liftStartDate": pd.to_datetime(["2018-05-01", "2018-04-03", "2018-07-01", "2017-12-01"]),
  # liftEndDate must be within input data range
  "liftEndDate": pd.to_datetime(["2018-06-10", "2018-06-03", "2018-07-20", "2017-12-31"]),
  # Provided value must be tested on same campaign level in model and same metric as dep_var_type
  "liftAbs": [400000, 300000, 700000, 200],
  # Spend within experiment: should match within a 10% error your spend on date range for each channel from dt_input
  "spend": [421000, 7100, 350000, 0],
  # Confidence: if frequentist experiment, you may use 1 - pvalue
  "confidence": [0.85, 0.8, 0.99, 0.95],
  # KPI measured: must match your dep_var
  "metric": ["revenue", "revenue", "revenue", "revenue"],
  # Either "immediate" or "total". For experimental inputs like Facebook Lift, "immediate" is recommended.
  "calibration_scope": ["immediate", "immediate", "immediate", "immediate"]
})
input_collect = inputs.robyn_inputs(
    InputCollect = input_collect['robyn_inputs'],
    hyperparameters = hyperparameters,
    calibration_input = calibration_input
)

# Check spend exposure fit if available
if 'exposure_vars' in input_collect.keys() and len(input_collect['exposure_vars']) > 0:
    for plot in input_collect['modNLS']['plots']:
        plot.show()

# Define the input collector
output_models = model.robyn_run(
    InputCollect=input_collect['robyn_inputs'],
    # Feed in all model specification
    ## model_specs=['my_model'],
    # Set to NULL to use all available CPU cores
    cores=None,
    # Run 2000 iterations
    iterations=2000,
    # Run 5 trials
    trials=5,
    # Use 3-way-split time series for NRMSE validation
    ts_validation=True,
    # Add penalty factor for experimental feature
    add_penalty_factor=False
)

print(output_models)

output_models['convergence']['moo_distrb_plot']
output_models['convergence']['moo_cloud_plot']

if output_models['ts_validation']:
    output_models['ts_validation_plot']
