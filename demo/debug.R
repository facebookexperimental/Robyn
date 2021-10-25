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
# go into robyn_run() line by line
