########################################################################################################################
# DEBUG
# File of variables used to debug functions
import pandas as pd
import os

# todo What do you think about renaming these as df_.. vs dt_ (for dataframe instead of data table)
dt_prophet_holidays = pd.read_csv('util/Python/data/prophet_holidays.csv')
dt_simulated_weekly = pd.read_csv('util/Python/data/simulated_weekly.csv')

dt_input = dt_simulated_weekly
dt_holidays = dt_prophet_holidays

mod = None
dt_modRollWind = None
dt_transform = None
xDecompAggPrev = None
date_var = 'DATE'  # date format must be "2020-01-01"
dayInterval = None
intervalType = None
dep_var = 'revenue'  # there should be only one dependent variable
dep_var_type = 'revenue'  # "revenue" or "conversion"
prophet_vars = ['trend', 'season', 'holiday']  # "trend","season", "weekday", "holiday" are provided and case-sensitive.
# Recommended to at least keep Trend & Holidays
prophet_signs = ['default', 'default', 'default']  # "default", "positive", and "negative". Recommend as default.
# Must be same length as prophet_vars
prophet_country = 'DE'  # only one country allowed once. Including national holidays for 59 countries, whose list can be
# found on our github guide
context_vars = ['competitor_sales_B', 'events']  # typically competitors, price & promotion, temperature, unemployment
# rate etc
context_signs = ['default', 'default']  # "default", " positive", and "negative", control the signs of coefficients for
# baseline variables
paid_media_vars = ['tv_S', 'ooh_S', 'print_S', 'facebook_I', 'search_clicks_P']  # we recommend to use media exposure
# metrics like impressions, GRP etc for the model. If not applicable, use spend instead
paid_media_signs = ['positive', 'positive', 'positive', 'positive', 'positive']  # "default", "positive", and
# "negative". must have same length as paid_media_vars. Controls the signs of coefficients for media variables
paid_media_spends = ['tv_S', 'ooh_S', 'print_S'	, 'facebook_S', 'search_S']  # spends must have same order and same
# length as paid_media_vars
organic_vars = ['newsletter']
organic_signs = ['positive']  # must have same length as organic_vars
factor_vars = ['events']  # specify which variables in context_vars and organic_vars are factorial

cores = os.cpu_count() - 2  # todo could use multiprocessing.cpu_count() since that is likely what we will be using in
# the future.
# subtracting 2 as to not use all cores

# set rolling window start
window_start = '2016-11-23'
window_end = '2018-08-22'
rollingWindowStartWhich = None
rollingWindowEndWhich = None
rollingWindowLength = None
refreshAddedStart = None

# set model core features
adstock = 'geometric'  # geometric, weibull_cdf or weibull_pdf. Both weibull adstocks are more flexible due to the
# changing decay rate over time, as opposed to the fixed decay rate for geometric. weibull_pdf allows also lagging
# effect. Yet weibull adstocks are two-parametric and thus take longer to run.
iterations = 4  # setting to 4 to not take too much runtime
# number of allowed iterations per trial. For the simulated dataset with 11 independent variables, 2000 is recommended
# for Geometric adsttock, 4000 for weibull_cdf and 6000 for weibull_pdf. The larger the dataset, the more iterations
# required to reach convergence.
nevergrad_algo = "TwoPointsDE"  # recommended algorithm for Nevergrad, the gradient-free optimisation library
# https://facebookresearch.github.io/nevergrad/index.html
trials = 5  # int, number of allowed trials. 5 is recommended without calibration, 10 with calibration.

# Time estimation: with geometric adstock, 2000 iterations * 5 trials and 6 cores, it takes less than 1 hour. Both
# Weibull adstocks take up to twice as much time.

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
dt_input = dt_simulated_weekly
# dt_holidays = dt_prophet_holidays
dt_holidays = dt_prophet_holidays
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
channel_constr_low = [0.5, 5]
channel_constr_up = [2, 5]
maxeval = 100000
constr_mode = "eq"
ui = False

#################
# debug adstock_weibull
x = [int for int in range(1, 121)]  # list of integers from 1 to 120
shape = 1
scale = 0.5
windlen = None
# type = "cdf"
type_ = "cdf"  # todo find actual name of variable - 2021.12.09

#################
# debug allocator
InputCollect = InputCollect
OutputCollect = OutputCollect
select_model = select_model
scenario = "max_historical_response"
channel_constr_low = [0.7, 0.7, 0.7, 0.7, 0.7]
channel_constr_up = [1.2, 1.5, 1.5, 1.5, 1.5]
robyn_object = None
select_build = None
optim_algo = "SLSQP_AUGLAG"
scenario = "max_historical_response"
expected_spend = None
expected_spend_days = None
maxeval = 100000
constr_mode = "eq"
ui = False

