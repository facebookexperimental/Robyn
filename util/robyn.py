# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import os
# os.environ['R_HOME'] = '/usr/local/bin/R'


import rpy2

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')


########################################################################################################################
# RESEARCH
# Remove later - 2022.02.14

# Overview documentation
# https://rpy2.github.io/doc/latest/html/overview.html#
# https://rpy2.github.io/doc/v2.9.x/html/introduction.html

# Installation instructions
# https://rpy2.github.io/doc/latest/html/overview.html#install-installation
# conda install -c r rpy2  # did not work
# conda install -c conda-forge rpy2

# https://docs.anaconda.com/anaconda/user-guide/tasks/using-r-language/
# conda create -n r-environment r-essentials r-base
# conda activate r-environment
# conda install r-essentials r-base


# conda install -c conda-forge r-essentials r-base
# conda install -c conda-forge rpy2
# r-base-4.1.2


# https://anaconda.org/r/r
# conda install -c r r

# python -m rpy2.situation
# python -m rpy2.tests


# Run .libPaths() in R to find path


# Check location
print(f'ryp2 location: {rpy2.__path__}')

# See ryp2 version
print(f'ryp2 version: {rpy2.__version__}')

# Setup details
import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

########################################################################################################################
# Step 1: Setup Environment

# Import rpy2's package module
import rpy2.robjects.packages as rpackages

# Import R's utility package
utils = rpackages.importr('utils')

# Select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list


# INSTALL NEEDED PACKAGES
# R package names
packnames = ['Robyn']

# R vector of strings
from rpy2.robjects.vectors import StrVector

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
print(f'Number of packaged to install: {len(names_to_install)}')
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# robjects.r("version")


# https://thomas-cokelaer.info/blog/2012/01/installing-rpy2-with-different-r-version-already-installed/
# https://stackoverflow.com/questions/64181911/call-r-package-data-using-python-with-rpy2
# https://www.marsja.se/r-from-python-rpy2-tutorial/







###############
# dict_params = {
#     'dt_input': dt_simulated_weekly  # pandas data frame with dates
# }
#
# print(f'Hey, I just want you to see this list; {l_rand}')
#
#
# obj_made = Robyn(dt_obj=,
#                  )


########################################################################################################################
# Build InputCollect R Object
'''
20220214
# Hard coding parameters to easily map to demo.R
# We can refactor to agnostic variables after we get the code running
# Code has not been tested
# Hyperparamerters not included, yet

20220215
General thoughts:
# rename robyn variables
# use robyn variables to intiate robyn collect
# save that inputcollect and check and
# Can we save an R object? Then we don't need to convert

TODO:
1. create python object to import data for models
2. create robyn robject to manage InputCollect robject via robyn_inputs() (see inputs.R) # robjects.r(InputCollect)???
3. pass/save python object to an from robyn object
4. use robyn object to run robyn_run()
'''
import os
import pandas as pd

df_prophet_holidays = pd.read_csv(os.path.join(os.getcwd(), 'util/data/prophet_holidays.csv'))
df_simulated_weekly = pd.read_csv(os.path.join(os.getcwd(), 'util/data/simulated_weekly.csv'))

df_input = df_simulated_weekly
df_holidays = df_prophet_holidays


class InputCollect(object): # rename for python object
    def __init__(self,
                 df_input = pd.DataFrame,
                 date_var_name = None,
                 dep_var_name = None,
                 dep_var_type = None,
                 prophet_vars = None,
                 prophet_signs = None,
                 prophet_country = None,
                 context_var_names = None,
                 context_var_signs = None,
                 paid_media_var_names = None,
                 paid_media_var_signs = None,
                 paid_media_spends = None,
                 organic_var_names = None,
                 organic_var_signs = None,
                 factor_var_names = None,
                 cores = None,
                 window_start = None,
                 window_end = None,
                 adstock = None,
                 iterations = None,
                 intercept_sign = None,
                 nevergrad_algo = None,
                 trials = None
                 ):
                 self.df_input = df_input,
                 self.date_var_name = date_var_name,
                 self.dep_var_name = dep_var_name,
                 self.dep_var_type = dep_var_type,
                 self.prophet_vars = prophet_vars,
                 self.prophet_signs = prophet_signs,
                 self.prophet_country = prophet_country,
                 self.context_var_names = context_var_names,
                 self.context_var_signs = context_var_signs,
                 self.paid_media_var_names = paid_media_var_names,
                 self.paid_media_var_signs = paid_media_var_signs,
                 self.paid_media_spends = paid_media_spends,
                 self.organic_var_names = organic_var_names,
                 self.organic_var_signs = organic_var_signs,
                 self.factor_var_names = factor_var_names,
                 self.cores = cores,
                 self.window_start = window_start,
                 self.window_end = window_end,
                 self.adstock = adstock,
                 self.iterations = iterations,
                 self.intercept_sign = intercept_sign,
                 self.nevergrad_algo = nevergrad_algo,
                 self.trials = trials


test = InputCollect(df_input=df_input)
test = InputCollect(cores=8)

print(test.df_input)
print(test.cores)

test.df_input



########################################################################################################################
# Scratch below
#
#  self.date_var_name = date_var_name
#  self.df_input: pd.DataFrame
#  self.date_var_name ='date' # date format must be "2020-01-01"
#  self.dep_var_name ='revenue' # there should be only one dependent variable
#  self.dep_var_type ='revenue' # "revenue" or "conversion"
#  self.prophet_vars = ('trend', "season", "holiday")
#  self.prophet_signs = ('default','default', 'default')
#  self.prophet_country = 'DE' # only one country allowed once. Including national holidays
#  self.context_var_names= ('competitor_sales_B', 'events') # typically competitors, price &,
#  self.context_var_signs = ('default', 'default') # c("default", " positive", and "negative"),
#  self.paid_media_var_names = ('tv_S', 'ooh_S', 'print_S', 'facebook_I', 'search_clicks_P')
#  self.paid_media_var_signs = ('positive', 'positive','positive', 'positive', 'positive')
#  self.paid_media_spends= ('tv_S','ooh_S','print_S','facebook_S', 'search_S')
#  self.organic_var_names = ('newsletter') # must have same length as organic_vars
#  self.organic_var_signs = ('positive') # specify which variables in context_vars and
#  self.factor_var_names= ('events') # specify which variables in context_vars and
#  self.cores = os.cpu_count() - 2 # I am using 6 cores from 8 on my local machine. Use future::availableCores() to find out cores
#  self.window_start = None
#  self.window_end = None
#  self.adstock = 'geometric' # geometric, weibull_cdf or weibull_pdf. Both weibull adstocks are more flexible
#  self.iterations = 2000 # number of allowed iterations per trial. For the simulated dataset with 11 independent
#  self.intercept_sign = 'non_negative' # intercept_sign input must be any of: non_negative, unconstrained
#  self.nevergrad_algo = 'TwoPointsDE' # recommended algorithm for Nevergrad, the gradient-free
#  self.trials = 5,  # number of allowed trials. 5 is recommended without calibration
#
#  rolling_window_start_which=None,
#  rolling_window_end_which=None,
#  rolling_window_length=None,
#  refresh_added_start=None,
#  cores=os.cpu_count() - 2
#  ):
#
# # Set variables from init
# self.df_input = df_input
# self.dep_var_name = dep_var_name
# self.dep_var_type = dep_var_type
# self.date_var_name = date_var_name
# self.adstock = adstock
# self.iterations = iterations
# self.nevergrad_algo = nevergrad_algo
# self.trials = trials
# self.prophet_vars = prophet_vars
# self.prophet_signs = prophet_signs
# self.prophet_country = prophet_country
# self.context_vars = context_var_names
# self.context_signs = context_var_signs
# self.paid_media_vars = paid_media_var_names
# self.paid_media_signs = paid_media_var_signs
# self.paid_media_spends = paid_media_spends
# self.organic_vars = organic_var_names
# self.organic_signs = organic_var_signs
# self.factor_vars = factor_var_names
# self.window_start = window_start
# self.window_end = window_end
# self.rollingWindowStartWhich = rolling_window_start_which
# self.rollingWindowEndWhich = rolling_window_end_which
# self.rollingWindowLength = rolling_window_length
# self.refreshAddedStart = refresh_added_start
# self.cores = cores
#
# print(f'self.df_input: {self.df_input}')
# print(f'self.dep_var_name: {self.dep_var_name}')
# print(f'self.dep_var_type: {self.dep_var_type}')
# print(f'self.date_var_name: {self.date_var_name}')
# print(f'self.adstock: {self.adstock}')
# print(f'self.iterations: {self.iterations}')
# print(f'self.nevergrad_algo: {self.nevergrad_algo}')
# print(f'self.trials: {self.trials}')
# print(f'self.prophet_vars: {self.prophet_vars}')
# print(f'self.prophet_signs: {self.prophet_signs}')
# print(f'self.prophet_country: {self.prophet_country}')
# print(f'self.context_vars: {self.context_vars}')
# print(f'self.context_signs: {self.context_signs}')
# print(f'self.paid_media_vars: {self.paid_media_vars}')
# print(f'self.paid_media_signs: {self.paid_media_signs}')
# print(f'self.paid_media_spends: {self.paid_media_spends}')
# print(f'self.organic_vars: {self.organic_vars}')
# print(f'self.organic_signs: {self.organic_signs}')
# print(f'self.factor_vars: {self.factor_vars}')
# print(f'self.window_start: {self.window_start}')
# print(f'self.window_end: {self.window_end}')
# print(f'self.rollingWindowStartWhich: {self.rollingWindowStartWhich}')
# print(f'self.rollingWindowEndWhich: {self.rollingWindowEndWhich}')
# print(f'self.rollingWindowLength: {self.rollingWindowLength}')
# print(f'self.refreshAddedStart: {self.refreshAddedStart}')
# print(f'self.cores: {self.cores}')
#
# # Set defaults
# self.df_holidays = pd.read_csv('util/data/prophet_holidays.csv')
#
# # Set in methods
# # todo Organize these variables as process is built out
# """
# :param day_interval:
# Integer
# Number of days detected between last two dates in data set.  Set in check_conditions
# """
# self.day_interval = None  # Get's automatically set in check_conditions
# self.interval_type = None  # Get's automatically set in check_conditions
# self.mod = None
# self.df_modRollWind = None
# self.xDecompAggPrev = None
# self.hyperparameters = None
# self.calibration_input = None
# self.mediaVarCount = None
# self.exposureVarName = None
# self.local_name = None
# self.all_media = None
#
# # Check that data frame and function are properly calibrated
# self.check_conditions(df_input)
# )