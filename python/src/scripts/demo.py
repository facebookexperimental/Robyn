# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#############################################################################################
####################         Meta MMM Open Source: Robyn <version> TODO       #######################
####################             Quick demo guide                     #######################
#############################################################################################

# Advanced marketing mix modeling using Meta Open Source project Robyn (Blueprint training)
# TODO: add link to blueprint training.

################################################################
#### Step 0: Setup environment

#TODO: Add setup instructions or point to readme

from src.robyn import transformation
from src.robyn import data, inputs, checks, model, outputs, json, plots, response, allocator ## Manual, Added manually
import numpy as np
import pandas as pd


################################################################
#### Step 1: Load data

## Check simulated dataset or load your own dataset
dt_simulated_weekly = data.dt_simulated_weekly()
print(dt_simulated_weekly.head())

## Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tipp: any events can be added into this table, school break, events etc.
dt_prophet_holidays = data.dt_prophet_holidays()
print(dt_prophet_holidays.head())


# Directory where you want to export results to (will create new folders)
robyn_directory = "~/Desktop"


################################################################
#### Step 2a: For first time user: Model specification in 4 steps

#### 2a-1: First, specify input variables

## All sign control are now automatically provided: "positive" for media & organic
## variables and "default" for all others. User can still customise signs if necessary.

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

#### 2a-2: Second, define and add hyperparameters
## Default media variable for modelling has changed from paid_media_vars to paid_media_spends.
## Also, calibration_input are required to be spend names.
## hyperparameter names are based on paid_media_spends names too. See right hyperparameter names:

hyper_names = inputs.hyper_names(adstock=input_collect['robyn_inputs']['adstock'], all_media=input_collect['robyn_inputs']['all_media'])

## Guide to setup & understand hyperparameters

## Robyn's hyperparameters have four components:
## - Adstock parameters (theta or shape/scale)
## - Saturation parameters (alpha/gamma)
## - Regularisation parameter (lambda). No need to specify manually
## - Time series validation parameter (train_size)

## 1. IMPORTANT: set plot = TRUE to create example plots for adstock & saturation
## hyperparameters and their influence in curve transformation.

transformation.plot_adstock(plot = False)
transformation.plot_saturation(plot = False)

## 3. Hyperparameter interpretation & recommendation:
#TODO: add more details from demo.R
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

#### 2a-3: Third, add hyperparameters into robyn_inputs()
## Manually converted, parameters defined wrong.
#input_collect = inputs.robyn_inputs(
#    InputCollect = input_collect['robyn_inputs'],
#    hyperparameters = hyperparameters
#)

# Print InputCollect
#print(input_collect)

#### 2a-4: Fourth (optional), model calibration / add experimental input

## Guide for calibration

#TODO: add more details from demo.R
################################################################
#### Step 2b: For known model specification, setup in one single step

#TODO: add more details from demo.R

# # Check spend exposure fit if available
# if 'exposure_vars' in input_collect.keys() and len(input_collect['exposure_vars']) > 0:
#     for plot in input_collect['modNLS']['plots']:
#         ##plot.show()
#         print('Skipping plot...')


calibration_input = pd.DataFrame({
  # channel name must in paid_media_vars
  "channel": ["facebook_S",  "tv_S", "facebook_S+search_S", "newsletter"],
  # liftStartDate must be within input data range
  "liftStartDate": pd.to_datetime(["2018-05-01", "2018-04-03", "2018-07-01", "2017-12-01"]),
  # liftEndDate must be within input data range
  "liftEndDate": pd.to_datetime(["2018-06-10", "2018-06-03", "2018-07-20", "2017-12-31"]),
  # Provided value must be tested on same campaign level in model and same metric as dep_var_type
  "liftAbs": [400000, 300000, 700000, 200],
  # Spend within experiment: should match within a 10% error your spend on date range for each channel from dt_input
  "spend": [421000, 7100, 350000, 0],
  # Confidence: if frequentist experiment, you may use 1 - pvalue
  "confidence": [0.85, 0.8, 0.99, 0.95],
  # KPI measured: must match your dep_var
  "metric": ["revenue", "revenue", "revenue", "revenue"],
  # Either "immediate" or "total". For experimental inputs like Facebook Lift, "immediate" is recommended.
  "calibration_scope": ["immediate", "immediate", "immediate", "immediate"]
})

test = calibration_input["channel"]
input_collect = inputs.robyn_inputs(
    InputCollect = input_collect['robyn_inputs'],
    hyperparameters = hyperparameters,
    calibration_input = calibration_input
)

# Check spend exposure fit if available
if 'exposure_vars' in input_collect.keys() and len(input_collect['exposure_vars']) > 0:
    for plot in input_collect['modNLS']['plots']:
        ##plot.show()
        print('Skipping plot...')

################################################################
#### Step 3: Build initial model
## Run all trials and iterations.
output_models = model.robyn_run(
    InputCollect=input_collect['robyn_inputs'],
    # Feed in all model specification
    ## model_specs=['my_model'],
    # Set to NULL to use all available CPU cores
    cores=None,
    # Run 2000 iterations
    iterations=29,
    # Run 5 trials
    trials=4,
    # Use 3-way-split time series for NRMSE validation
    ts_validation=True,
    # Add penalty factor for experimental feature
    add_penalty_factor=False
)

# output_models = model.robyn_run(
#     InputCollect=input_collect['robyn_inputs'],
#     # Feed in all model specification
#     ## model_specs=['my_model'],
#     # Set to NULL to use all available CPU cores
#     cores=None,
#     # Run 2000 iterations
#     iterations=200,
#     # Run 5 trials
#     trials=5,
#     # Use 3-way-split time series for NRMSE validation
#     ts_validation=True,
#     # Add penalty factor for experimental feature
#     add_penalty_factor=False
# )

# Print output models
print(output_models)

## Check MOO (multi-objective optimization) convergence plots
# Read more about convergence rules: See robyn_converge

output_models['convergence']['moo_distrb_plot']
output_models['convergence']['moo_cloud_plot']

## Check time-series validation plot (when ts_validation == TRUE)
# Read more and replicate results: See ts_validation
if output_models['metadata']['ts_validation']:
    output_models['ts_validation_plot']

# Check time-series validation plot
if output_models['metadata']['ts_validation']:
    print(output_models['ts_validation_plot'])

## Calculate Pareto fronts, cluster and export results and plots. See robyn_outputs
output_collect = outputs.robyn_outputs(
    input_collect,
    output_models,
    # Automatically pick how many Pareto fronts to fill
    pareto_fronts='auto',
    # Set to 100 top Pareto models for clustering
    # min_candidates=100,
    # Calibration constraint
    # calibration_constraint=0.1,
    # Export results to CSV files
    csv_out='pareto',
    # Cluster similar models by ROAS
    clusters=True,
    # Create files locally
    export=True,
    # Path for plots exports and files creation
    plot_folder=robyn_directory,
    # Set to FALSE to deactivate plotting and saving model one-pagers
    plot_pareto=False
)

# Print the output collect
print(output_collect)

################################################################
#### Step 4: Select and save the any model
select_model = '1_0_1'
# TODO add below code once plotting in 'robyn_outputs' has been fixed
# exported_model = json.robyn_write(input_collect, output_collect, select_model, export=True)
# print(exported_model)

# # Plot any model's one-pager
# my_one_pager = plots.robyn_onepagers(input_collect, output_collect, select_model, export=False)
# print(my_one_pager)

# # Check each of the one-pager's plots
# my_one_pager.patches.plots[1]
# my_one_pager.patches.plots[2]
# my_one_pager.patches.plots[3]

################################################################
#### Step 5: Get budget allocation based on the selected model above

## Budget allocation result requires further validation. Please use this recommendation with caution.
## Don't interpret budget allocation result if selected model above doesn't meet business expectation.

allocator_collect1 = allocator.robyn_allocator(
    InputCollect=input_collect,
    OutputCollect=output_collect,
    select_model=select_model,
    # Date range for budget allocation
    date_range=None,
    # Total budget for budget allocation
    total_budget=None,
    # Channel constraints
    channel_constr_low=0.7,
    channel_constr_up=[1.2, 1.5, 1.5, 1.5, 1.5],
    # Scenario for budget allocation
    scenario='max_response',
    # Export results to CSV files
    export=True
)

# Print and plot allocator's output
print(allocator_collect1)
# plot(allocator_collector)



# Example 2: maximize response for latest 10 periods with given spend

# allocator_collect2 = allocator.robyn_allocator(
#     InputCollect=input_collect,
#     OutputCollect=output_collect,
#     select_model=select_model,
#     date_range="last_10",
#     total_budget=5000000,
#     channel_constr_low=[0.8, 0.7, 0.7, 0.7, 0.7],
#     channel_constr_up=[1.2, 1.5, 1.5, 1.5, 1.5],
#     channel_constr_multiplier=5,
#     scenario="max_response",
#     export=True
# )

# print(allocator_collect2)

# plot(allocator_collect2)

# Example 3: Use default ROAS target for revenue or CPA target for conversion

allocator_collect3 = allocator.robyn_allocator(
    InputCollect=input_collect,
    OutputCollect=output_collect,
    select_model=select_model,
    #date_range=None,  # Default last month as initial period
    scenario="target_efficiency",
    # target_value=2,  # Customize target ROAS or CPA value
    export=True
)

print(allocator_collect3)

# plot(allocator_collect3)

# Example 4: Customize target_value for ROAS or CPA (using json_file)

json_file = "~/Desktop/Robyn_202302221206_init/RobynModel-1_117_11.json"

allocator_collect4 = allocator.robyn_allocator(
    json_file=json_file,  # Using json file from robyn_write() for allocation
    dt_input=dt_simulated_weekly,
    dt_holidays=dt_prophet_holidays,
    date_range=None,  # Default last month as initial period
    scenario="target_efficiency",
    target_value=2,  # Customize target ROAS or CPA value
    plot_folder="~/Desktop/my_dir",
    plot_folder_sub="my_subdir",
    export=True
)

# A csv is exported into the folder for further usage. Check schema here:
# https://github.com/facebookexperimental/Robyn/blob/main/demo/schema.R

# QA optimal response

select_media = "search_S"  # Pick any media variable: InputCollect$all_media

metric_value = allocator_collect1['dt_optimOut']['optmSpendUnit'][
    allocator_collect1['dt_optimOut']['channels'].index(select_media)
]  # For paid_media_spends set metric_value as your optimal spend

# # For paid_media_vars and organic_vars, manually pick a value
# metric_value = 10000

# Saturation curve for adstocked metric results (example)

response.robyn_response(
    InputCollect=input_collect,
    OutputCollect=output_collect,
    select_model=select_model,
    metric_name=select_media,
    metric_value=metric_value,
    date_range="last_5"
)

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

################################################################
#### Step 6: Model refresh based on selected model and saved results

## Must run robyn_write() (manually or automatically) to export any model first, before refreshing.
## The robyn_refresh() function is suitable for updating within "reasonable periods".
## Two situations are considered better to rebuild model:
## 1. most data is new. If initial model has 100 weeks and 80 weeks new data is added in refresh,
## it might be better to rebuild the model. Rule of thumb: 50% of data or less can be new.
## 2. new variables are added.

# Provide JSON file with your InputCollect and ExportedModel specifications
# It can be any model, initial or a refresh model

with open('~/Desktop/Robyn_202211211853_init/RobynModel-1_100_6.json') as f:
    data = json.load(f)

# Define functions
def robyn_refresh(json_file, dt_input, dt_holidays, refresh_steps, refresh_iters, refresh_trials):
    # Load JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Extract relevant data
    input_collect = data['listRefresh1']['InputCollect']
    output_collect = data['listRefresh1']['OutputCollect']
    select_model = data['listRefresh1']['OutputCollect']['selectID']

    # Split data into training and testing sets
    train_input, test_input, train_output, test_output = train_test_split(input_collect, output_collect, test_size=0.2, random_state=42)

    # Initialize model
    model = LinearRegression()

    # Train model
    for i in range(refresh_steps):
        # Fit model on training data
        model.fit(train_input, train_output)

        # Predict on testing data
        predictions = model.predict(test_input)

        # Calculate mean squared error
        mse = mean_squared_error(test_output, predictions)

        # Print iteration information
        print(f'Refresh Step {i+1}, MSE: {mse:.2f}')

        # Update model
        model.coef_ = np.random.rand(model.coef_.shape[0], 1)

    # Return model
    return model

def robyn_response(input_collect, output_collect, select_model, metric_name, metric_value, date_range):
    # Initialize response
    response = {}

    # Calculate response
    for i in range(len(input_collect)):
        # Extract relevant data
        input_data = input_collect.iloc[i]
        output_data = output_collect.iloc[i]

        # Calculate metric value
        metric_value = calculate_metric(input_data, output_data, select_model, metric_name)

        # Add to response
        response[f'{date_range}'] = metric_value

    # Return response
    return response

def calculate_metric(input_data, output_data, select_model, metric_name):
    # Calculate metric value
    if metric_name == 'facebook_S':
        # Calculate Facebook S metric
        metric_value = (output_data['facebook_S'] - input_data['facebook_S']) / input_data['facebook_S']
    else:
        # Calculate other metrics
        raise NotImplementedError

    return metric_value

# Load JSON file
with open('~/Desktop/Robyn_202208231837_init/Robyn_202208231841_rf1/RobynModel-1_12_5.json') as f:
    data = json.load(f)

# Extract relevant data
input_collect = data['listRefresh1']['InputCollect']
output_collect = data['listRefresh1']['OutputCollect']
select_model = data['listRefresh1']['OutputCollect']['selectID']

# Split data into training and testing sets
train_input, test_input, train_output, test_output = train_test_split(input_collect, output_collect, test_size=0.2, random_state=42)

# Initialize model
model = robyn_refresh(json_file=data, dt_input=dt_simulated_weekly, dt_holidays=dt_prophet_holidays, refresh_steps=7, refresh_iters=1000, refresh_trials=1)

# Train model
model.fit(train_input, train_output)

# Predict on testing data
predictions = model.predict(test_input)

# Calculate mean squared error
mse = mean_squared_error(test_output, predictions)

# Print iteration information
print(f'Refresh Step 1, MSE: {mse:.2f}')

# Define response
response = robyn_response(input_collect, output_collect, select_model, 'facebook_S', 20000, 'last_1')

# Plot response
import matplotlib.pyplot as plt
plt.plot(response['last_1'])
plt.xlabel('Date')
plt.ylabel('Facebook S')
plt.title('Facebook S vs. Date')
plt.show()

# Calculate spend
spend = 20000

# Define response
response1 = robyn_response(input_collect, output_collect, select_model, 'facebook_S', spend, 'last_1')

# Plot response
import matplotlib.pyplot as plt
plt.plot(response1['last_1'])
plt.xlabel('Date')
plt.ylabel('Facebook S')
plt.title('Facebook S vs. Date')
plt.show()

# Calculate spend
spend2 = spend + 100

# Define response
response2 = robyn_response(input_collect, output_collect, select_model, 'facebook_S', spend2, 'last_1')

# Plot response
import matplotlib.pyplot as plt
plt.plot(response2['last_1'])
plt.xlabel('Date')
plt.ylabel('Facebook S')
plt.title('Facebook S vs. Date')
plt.show()

# Calculate difference
difference = (response2['response_total'] - response1['response_total']) / (spend2 - spend)

print(f'Difference: {difference:.2f}')

import python.src as src

# Set up Robyn environment
src.set_env(src.Environment(
    input_collect=src.InputCollect(
        dt_input='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json',
        dt_holidays='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json'
    ),
    output_collect=src.OutputCollect(
        select_model='select_model'
    )
))

# Define budget and date range for Spend3
spend3 = 100000
date_range = 'last_5'

# Create Robyn response object
response3 = src.response(
    InputCollect=src.InputCollect(
        dt_input='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json',
        dt_holidays='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json'
    ),
    OutputCollect=src.OutputCollect(
        select_model='select_model'
    ),
    metric_name='facebook_S',
    metric_value=spend3,
    date_range=date_range
)

# Plot the response
response3.plot()

# Define sendings and create Robyn response object
sendings = 30000
response_sending = robyn_response(
    InputCollect=src.InputCollect(
        dt_input='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json',
        dt_holidays='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json'
    ),
    OutputCollect=src.OutputCollect(
        select_model='select_model'
    ),
    metric_name='newsletter',
    metric_value=sendings
)

# Calculate and print the response total
response_sending.response_total / sendings * 1000
print(response_sending.plot())

# Write Robyn inputs and outputs to files
src.write(src.InputCollect(
    dt_input='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json',
    dt_holidays='~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json'
), '~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json')
src.write(src.OutputCollect(
    select_model='select_model'
), '~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json')

# Read Robyn inputs and outputs from files
json_file = '~/Desktop/Robyn_202208231837_init/RobynModel-1_100_6.json'
json_data = src.read(json_file)
print(json_data)

# Create Robyn inputs and outputs for recreated model
input_collect = src.InputCollect(
    dt_input=json_data['dt_input'],
    dt_holidays=json_data['dt_holidays']
)
output_collect = src.OutputCollect(
    select_model=json_data['select_model']
)

# Recreate Robyn model
robyn_recreate(
    json_file=json_file,
    dt_input=input_collect.dt_input,
    dt_holidays=input_collect.dt_holidays,
    quiet=False
)

# Write Robyn inputs and outputs to files
src.write(input_collect, output_collect, export=False, dir='~/Desktop')
my_model = src.read(json_file)
print(my_model)

# Create one-pagers for Robyn model
plots.robyn_onepagers(input_collect, output_collect, export=False)

# Refresh Robyn model
robyn_refresh(
    json_file=json_file,
    dt_input=input_collect.dt_input,
    dt_holidays=input_collect.dt_holidays,
    refresh_steps=6,
    refresh_mode='manual',
    refresh_iters=1000,
    refresh_trials=1
)

# Create Robyn response object
response = src.response(
    InputCollect=input_collect,
    OutputCollect=output_collect,
    metric_name='newsletter',
    metric_value=50000
)

# Print the response
print(response.plot())
