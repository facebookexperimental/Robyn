########################################################################################################################
# IMPORTS

##################
# Base Python imports
import pandas as pd

##################
# Python R Import
import rpy2
import rpy2.situation
from rpy2 import robjects
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter, py2rpy
from rpy2.robjects.packages import importr, data

pandas2ri.activate()  # Won't work unless activated

##################
# R imports
# robjects.r['library']('reticulate')
# robjects.r['import']('sys')  # Gives an error
utils = importr("utils")
base = importr('base')

##################
# Import Robyn from R
robyn = importr('Robyn')


########################################################################################################################
# REVIEW

print(rpy2.__version__)

# Setup details
for row in rpy2.situation.iter_info():
    print(row)


########################################################################################################################
# SETTINGS

# Set seed
set_seed = r('set.seed')
set_seed(123)

# Force multicore when using RStudio
# TODO
# robjects.r['import']('sys')  # Gives an error
# base.Sys_getenv(R_FUTURE_FORK_ENABLE="true")
# Sys.setenv(R_FUTURE_FORK_ENABLE="true")

########################################################################################################################
# READ IN DATA

# Check simulated dataset or load your own dataset
# utils.data("dt_simulated_weekly")
r.data('dt_simulated_weekly')
r['dt_simulated_weekly'].head()

# Import data then convert to R data frame
df_simulated = pd.read_csv('util/data/simulated_weekly.csv')  # import as pandas data frame
with localconverter(ro.default_converter + pandas2ri.converter):
  r_df_simulated = ro.conversion.py2rpy(df_simulated)


# Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tip: any events can be added into this table, school break, events et
r.data('dt_prophet_holidays')
r['dt_prophet_holidays'].head()

df_prophet = pd.read_csv('util/data/prophet_holidays.csv')
with localconverter(ro.default_converter + pandas2ri.converter):
  r_df_prophet = ro.conversion.py2rpy(df_prophet)

# Set robyn_object. It must have extension .RDS. The object name can be different than Robyn:
# TODO
# robyn_object = "~/Desktop/MyRobyn.RDS"


########################################################################################################################
# Step 2a: For first time user: Model specification in 4 steps

# 2a-1: First, specify input data & model parameters

# Run ?robyn_inputs to check parameter definition
# TODO

robyn.robyn_inputs(
    # dt_input=df_simulated
    # , dt_holidays=df_prophet
    # dt_input=r['dt_simulated_weekly']
    # , dt_holidays=r['dt_prophet_holidays']
    dt_input=r_df_prophet
    , dt_holidays=r_df_prophet

    # set variables
    , date_var="DATE"
    # # date format must be "2020-01-01"
    # , dep_var="revenue"
    # # there should be only one dependent variable
    # , dep_var_type="revenue"
    # # "revenue" or "conversion"

    # , prophet_vars=["trend", "season", "holiday"]
    # # "trend","season", "weekday", "holiday"
    # # are provided and case-sensitive. Recommended to at least keep Trend & Holidays
    # , prophet_signs=["default", "default", "default"]
    # # c("default", "positive", and "negative").
    # # Recommend as default.Must be same length as prophet_vars
    # , prophet_country="DE"
    # # only one country allowed once. Including national holidays
    # # for 59 countries, whose list can be found on our github guide

    # , context_vars=["competitor_sales_B", "events"]
    # # typically competitors, price &

    # # promotion, temperature, unemployment rate etc
    # , context_signs=["default", "default"]
    # # c("default", " positive", and "negative"),
    # # control the signs of coefficients for baseline variables

    # , paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"]
    # # c("tv_S", "ooh_S", "print_S", "facebook_I", "facebook_S","search_clicks_P", "search_S")
    # # we recommend to use media exposure metrics like impressions, GRP etc for the model.
    # # If not applicable, use spend instead
    # , paid_media_signs=["positive", "positive", "positive", "positive", "positive"]
    # # c("default", "positive", and "negative"). must have same length as paid_media_vars.
    # # Controls the signs of coefficients for media variables
    # , paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
    # # spends must have same order and same length as paid_media_vars

    # , organic_vars=["newsletter"]
    # , organic_signs=["positive"]
    # # must have same length as organic_vars

    # , factor_vars=["events"]
    # # specify which variables in context_vars and organic_vars are factorial

    # # set model parameters

    # # set cores for parallel computing
    # , cores=6
    # # I am using 6 cores from 8 on my local machine. Use future::availableCores() to find out cores

    # # set rolling window start
    # , window_start="2016-11-23"
    # , window_end="2018-08-22"

    # # set model core features
    # , adstock="geometric"
    # # geometric, weibull_cdf or weibull_pdf. Both weibull adstocks are more flexible
    # # due to the changing decay rate over time, as opposed to the fixed decay rate for geometric. weibull_pdf
    # # allows also lagging effect. Yet weibull adstocks are two-parametric and thus take longer to run.
    # , iterations=2
    # # number of allowed iterations per trial. For the simulated dataset with 11 independent
    # # variables, 2000 is recommended for Geometric adstock, 4000 for weibull_cdf and 6000 for weibull_pdf.
    # # The larger the dataset, the more iterations required to reach convergence.

    # , intercept_sign="non_negative"
    # # intercept_sign input must be any of: non_negative, unconstrained
    # , nevergrad_algo="TwoPointsDE"
    # # recommended algorithm for Nevergrad, the gradient-free
    # # optimisation library https://facebookresearch.github.io/nevergrad/index.html
    # , trials=2  # number of allowed trials. 5 is recommended without calibration,
    # # 10 with calibration.

    # # Time estimation: with geometric adstock, 2000 iterations * 5 trials
    # # and 6 cores, it takes less than 1 hour. Both Weibull adstocks take up to twice as much time.
)
###


robyn.plot_adstock(plot=False)