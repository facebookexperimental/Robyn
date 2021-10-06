# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS
import pandas as pd
from python.robyn_v02 import robyn as r

# FOR TESTING
import importlib as il
il.reload(r)

########################################################################################################################
# EXAMPLES - HOW TO USE THE CLASS

# Initialize a Robyn object
robyn = r.Robyn(country="DE", dateVarName='Date', depVarName='revenue', mediaVarName=[])

# See a parameter
robyn.test_y_train
robyn.iterations

# See all variables
robyn.__dict__

# See an empty class variable
robyn.mod

# Run something
robyn.refit(x_train=robyn.test_x_train,
            y_train=robyn.test_y_train,
            lambda_=robyn.test_lambda_,
            lower_limits=robyn.test_lower_limits,
            upper_limits=robyn.test_upper_limits)

# See the class variable that was just updated
robyn.mod


########################################################################################################################
# SCRIPT - DEMO OF HOW A USER WOULD USE THE PACKAGE

# Step 1: INITIALIZE OBJECT
# status: working version
robyn = r.Robyn(country="DE", dateVarName='Date', depVarName='revenue'
                , mediaVarName=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"])

# Step 2: IMPORT DATA SET FOR PREDICTIONS
df = pd.read_csv('source/de_simulated_data.csv')

# Step 3: USER SET HYPERPARAMETER BOUNDS
# status: #TODO needs to be implemented
robyn.set_param_bounds()

# Step 4: PREPARE DATA FOR MODELING
# Status: working version
df_mod = robyn.input_wrangling(df)

# Step 5: FIT MODEL
# status #TODO in progress mmm() and fit(), the fit() function is the robyn() function in the R version
robyn.fit(df=df_mod)

# Step 6: BUDGET ALLOCATOR
# status: #TODO needs to be implemented
robyn.allocate_budget(modID="3_10_2",
                      scenario="max_historical_response",
                      channel_constr_low=[0.7, 0.75, 0.60, 0.8, 0.65],
                      channel_constr_up=[1.2, 1.5, 1.5, 2, 1.5])