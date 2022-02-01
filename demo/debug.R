devtools::load_all()

## debug robyn_mmm
# prep input param
hyper_collect = InputCollect$hyperparameters
iterations = InputCollect$iterations
lambda.n = 100
lambda_control = 1
lambda_fixed = NULL
refresh = FALSE
seed = 123L
# go into robyn_mmm() line by line

## debug robyn_run
# prep input param

calibration_constraint = 0.1
lambda_control = 1
refresh = FALSE
dt_hyper_fixed = NULL
seed = 123
outputs = FALSE
# go into robyn_run() line by line

## debug robyn_refresh
# robyn_object
plot_folder_sub = NULL
dt_input = dt_input
dt_holidays = dt_prophet_holidays
refresh_steps = 14
refresh_mode = "auto" # "auto", "manual"
refresh_iters = 100
refresh_trials = 2
plot_pareto = TRUE

## debug robyn_allocator
# prep input para

#robyn_object
select_build = 1
InputCollect = NULL
OutputCollect = NULL
select_model = NULL
optim_algo = "SLSQP_AUGLAG"
scenario = "max_historical_response"
expected_spend = NULL
expected_spend_days = NULL
channel_constr_low = rep(0.5, 5)
channel_constr_up = rep(2, 5)
maxeval = 100000
constr_mode = "eq"
ui = FALSE

## debug adstock_weibull
x = 1:120
shape = 1
scale = 0.5
windlen = NULL
type = "cdf"


## debug allocator
#InputCollect = InputCollect
#OutputCollect = OutputCollect
#select_model = select_model
scenario = "max_historical_response"
channel_constr_low = c(0.7, 0.7, 0.7, 0.7, 0.7)
channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5)
robyn_object = NULL
select_build = NULL
optim_algo = "SLSQP_AUGLAG"
scenario = "max_historical_response"
expected_spend = NULL
expected_spend_days = NULL
maxeval = 100000
constr_mode = "eq"
ui = FALSE


## debug robyn_outputs
args(robyn_outputs)
#InputCollect
#OutputModels
pareto_fronts = 1
calibration_constraint = 0.1
plot_folder = robyn_object
plot_folder_sub = NULL
plot_pareto = TRUE
csv_out = "pareto"
clusters = TRUE
selected = "clusters"
ui = FALSE
export = TRUE
quiet = FALSE


## debug robyn_clusters
args(robyn_clusters)
input = OutputCollect
all_media = NULL
k = "auto"
limit = 1
weights = rep(1, 3)
dim_red = "PCA"
quiet = FALSE
export = FALSE
