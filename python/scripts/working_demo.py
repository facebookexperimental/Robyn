## import robyn ## Manual, no need
from robyn import data, inputs, checks ## Manual, Added manually
from robyn import transformation
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
