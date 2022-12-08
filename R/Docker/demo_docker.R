# install.packages("remotes") # Install remotes first if not already happend
library(Robyn) # remotes::install_github("facebookexperimental/Robyn/R")

Sys.setenv(R_FUTURE_FORK_ENABLE = "true")
options(future.fork.enable = TRUE)

data("dt_simulated_weekly")
head(dt_simulated_weekly)

## Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tip: any events can be added into this table, school break, events etc.
data("dt_prophet_holidays")
head(dt_prophet_holidays)

## Set robyn_object. It must have extension .RDS. The object name can be different than Robyn:
robyn_object <- "/home/guest/MyRobyn.RDS"

plot_adstock(plot = T)
plot_saturation(plot = T)

hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)

# Example hyperparameters for Geometric adstock
hyperparameters <- list(
  facebook_I_alphas = c(0.5, 3),
  facebook_I_gammas = c(0.3, 1),
  facebook_I_thetas = c(0, 0.3),
  print_S_alphas = c(0.5, 3),
  print_S_gammas = c(0.3, 1),
  print_S_thetas = c(0.1, 0.4),
  tv_S_alphas = c(0.5, 3),
  tv_S_gammas = c(0.3, 1),
  tv_S_thetas = c(0.3, 0.8),
  search_clicks_P_alphas = c(0.5, 3),
  search_clicks_P_gammas = c(0.3, 1),
  search_clicks_P_thetas = c(0, 0.3),
  ooh_S_alphas = c(0.5, 3),
  ooh_S_gammas = c(0.3, 1),
  ooh_S_thetas = c(0.1, 0.4),
  newsletter_alphas = c(0.5, 3),
  newsletter_gammas = c(0.3, 1),
  newsletter_thetas = c(0.1, 0.4)
)


InputCollect <- robyn_inputs(
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  date_var = "DATE",
  dep_var = "revenue",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "holiday"),
  prophet_signs = c("default", "default", "default"),
  prophet_country = "DE",
  context_vars = c("competitor_sales_B", "events"),
  context_signs = c("default", "default"),
  paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
  paid_media_signs = c("positive", "positive", "positive", "positive", "positive"),
  paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
  organic_vars = "newsletter",
  organic_signs = "positive",
  factor_vars = "events",
  window_start = "2016-11-23",
  window_end = "2018-08-22",
  adstock = "geometric",
  iterations = 100,
  trials = 2,
  hyperparameters = hyperparameters # as in 2a-2 above
  # ,calibration_input = dt_calibration # as in 2a-4 above
)

OutputCollect <- robyn_run(
  InputCollect = InputCollect, # feed in all model specification
  plot_folder = robyn_object, # plots will be saved in the same folder as robyn_object
  pareto_fronts = 1,
  plot_pareto = TRUE,
)
