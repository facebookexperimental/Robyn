########################################################################################################################
# DEBUG
# File of variables used to debug functions
import pandas as pd


# todo What do you think about renaming these as df_.. vs dt_ (for dataframe instead of data table)
dt_prophet_holidays = pd.read_csv('util/data/prophet_holidays.csv')
dt_simulated_weekly = pd.read_csv('util/data/simulated_weekly.csv')

dt_input = dt_simulated_weekly
dt_holidays = dt_prophet_holidays

mod = None
dt_modRollWind = None
dt_transform = None
xDecompAggPrev = None
date_var = None
dayInterval = None
intervalType = None
dep_var = None
dep_var_type = None
prophet_vars = None
prophet_signs = None
prophet_country = None
context_vars = None
context_signs = None
paid_media_vars = None
paid_media_signs = None
paid_media_spends = None
organic_vars = None
organic_signs = None
factor_vars = None
cores = 1
window_start = None
window_end = None
rollingWindowStartWhich = None
rollingWindowEndWhich = None
rollingWindowLength = None
refreshAddedStart = None
adstock = None
iterations = 2000
nevergrad_algo = "TwoPointsDE"
trials = 5
hyperparameters = None
calibration_input = None
mediaVarCount = None
exposureVarName = None
local_name = None
all_media = None

#################
# debug robyn_mmm
# prep input param
# hyper_collect = InputCollect$hyperparameters
hyper_collect = None
# iterations = InputCollect$iterations
iterations = None
lambda_n = 100
lambda_control = 1
lambda_fixed = None
refresh = False
# seed = 123L
seed = None
# go into robyn_mmm() line by line

#################
# debug robyn_run
# prep input para
# plot_folder = robyn_object
plot_folder = None
plot_folder_sub = None
pareto_fronts = 1
plot_pareto = True
calibration_constraint = 0.1
lambda_control = 1
refresh = False
dt_hyper_fixed = None
ui = False
csv_out = "pareto"
seed = 123
# go into robyn_run() line by line

#################
# debug robyn_refresh
# robyn_object
plot_folder_sub = None
dt_input = None
# dt_holidays = dt_prophet_holidays
dt_holidays = None
refresh_steps = 14
refresh_mode = "auto"  # "auto", "manual"
refresh_iters = 100
refresh_trials = 2
plot_pareto = True

#################
# debug robyn_allocator
# prep input para
# robyn_object
select_build = 1
InputCollect = None
OutputCollect = None
select_model = None
optim_algo = "SLSQP_AUGLAG"
scenario = "max_historical_response"
expected_spend = None
expected_spend_days = None
# channel_constr_low = rep(0.5, 5)
channel_constr_low = None
# channel_constr_up = rep(2, 5)
channel_constr_up = None
maxeval = 100000
constr_mode = "eq"
ui = False

#################
# debug adstock_weibull
# x = 1:120
x = None
shape = 1
scale = 0.5
windlen = None
# type = "cdf"
type_ = "cdf"  # todo find actual name - 2021.12.09

#################
# debug allocator
InputCollect = InputCollect
OutputCollect = OutputCollect
select_model = select_model
scenario = "max_historical_response"
# channel_constr_low = c(0.7, 0.7, 0.7, 0.7, 0.7)
channel_constr_low = None
# channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5)
channel_constr_up = None
robyn_object = None
select_build = None
optim_algo = "SLSQP_AUGLAG"
scenario = "max_historical_response"
expected_spend = None
expected_spend_days = None
maxeval = 100000
constr_mode = "eq"
ui = False

