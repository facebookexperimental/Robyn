devtools::load_all()

## debug robyn_mmm
# prep input param
hyper_collect = InputCollect$hyperparameters
iterations = InputCollect$iterations
lambda.n = 100
lambda_control = 1
lambda_fixed = NULL
refresh = FALSE
# go into robyn_mmm() line by line

## debug robyn_run
# prep input para
plot_folder = robyn_object
pareto_fronts = 1
plot_pareto = TRUE
calibration_constraint = 0.1
lambda_control = 1
refresh = FALSE
dt_hyper_fixed = NULL
ui = FALSE
seed = 123
# go into robyn_run() line by line

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
