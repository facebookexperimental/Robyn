# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import time
import pandas as pd
import numpy as np
import nevergrad as ng
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nevergrad.optimization import optimizerlib
import multiprocessing
from functools import partial
from tqdm import tqdm
import re
#from glmnet import glmnet
from sklearn.linear_model import Ridge
import logging
from math import sqrt
import math
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import sys
from collections import defaultdict
from datetime import datetime

## Robyn imports
from .inputs import hyper_names
from .checks import check_hyper_fixed, check_legacy_input, check_run_inputs, check_iteration, check_obj_weight, LEGACY_PARAMS, HYPS_OTHERS, check_adstock, check_parallel, check_init_msg
from .json import robyn_read ## name conflict?
from .outputs import robyn_outputs
from .transformation import run_transformations
from .calibration import robyn_calibrate
from .convergence import robyn_converge
from .plots import ts_validation_fun

## Manually added
from time import strftime, localtime
from scipy.stats import uniform
from itertools import repeat

def robyn_run(InputCollect=None,
              dt_hyper_fixed=None,
              json_file=None,
              ts_validation=False,
              add_penalty_factor=False,
              refresh=False,
              seed=123,
              quiet=False,
              cores=None,
              trials=5,
              iterations=2000,
              rssd_zero_penalty=True,
              objective_weights=None,
              nevergrad_algo="TwoPointsDE",
              intercept=True,
              intercept_sign="non_negative",
              lambda_control=None,
              outputs=False,
              *args,
              **kwargs
              ):

    if outputs:
        OutputModels = robyn_run(
            InputCollect=InputCollect,
            dt_hyper_fixed=dt_hyper_fixed,
            json_file=json_file,
            add_penalty_factor=add_penalty_factor,
            ts_validation=ts_validation,
            refresh=refresh,
            seed=seed,
            quiet=quiet,
            cores=cores,
            trials=trials,
            iterations=iterations,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            lambda_control=lambda_control,
            outputs=False
        )
        OutputCollect = robyn_outputs(InputCollect, OutputModels) ##, *args, **kwargs)

        return {
            "OutputModels": OutputModels,
            "OutputCollect": OutputCollect
        }

    ## t0 = time.time()
    t0 = strftime("%Y-%m-%d %H:%M:%S", localtime())

    # Use previously exported model using json_file
    if json_file is not None:
        # InputCollect <- robyn_inputs(json_file = json_file, dt_input = dt_input, dt_holidays = dt_holidays)
        if InputCollect is None:
            InputCollect = robyn_inputs(json_file=json_file) ##, *args, **kwargs)

        json_data = robyn_read(json_file, step=2, quiet=True)
        dt_hyper_fixed = json_data['ExportedModel']['hyper_values']
        for key, value in json_data['ExportedModel'].items():
            globals()[key] = value
        # Select specific columns from a DataFrame
        bootstrap = DataFrame(json_data['ExportedModel']['summary']).filter(items=['variable', 'boot_mean', 'ci_low', 'ci_up'])
        if seed is None or len(str(seed)) == 0:
            seed = 123
        dt_hyper_fixed['solID'] = json_data['ExportedModel']['select_model']
    else:
        bootstrap = None

    #####################################
    #### Set local environment

    # Check for 'hyperparameters' in InputCollect
    if "hyperparameters" not in InputCollect.keys() or InputCollect['hyperparameters'] is None:
        raise ValueError("Must provide 'hyperparameters' in robyn_inputs()'s output first")

    # Check and warn on legacy inputs
    InputCollect = check_legacy_input(InputCollect, cores, iterations, trials, intercept_sign, nevergrad_algo)
    # Overwrite values imported from InputCollect
    legacyValues = {k: v for k, v in InputCollect.items() if v is not None and k in LEGACY_PARAMS}
    if legacyValues:
        for key, value in InputCollect.items():
            globals()[key] = value

    # Handling cores
    max_cores = max(1, multiprocessing.cpu_count())
    if cores is None:
        cores = max_cores - 1  # Leave at least one core free
    elif cores > max_cores:
        print(f"Warning: Max possible cores in your machine is {max_cores} (your input was {cores})")
        cores = max_cores
    if cores == 0:
        cores = 1

    hyps_fixed = dt_hyper_fixed is not None
    if hyps_fixed:
        trials = iterations = 1
    check_run_inputs(cores, iterations, trials, intercept_sign, nevergrad_algo)
    check_iteration(InputCollect['calibration_input'], iterations, trials, hyps_fixed, refresh)
    init_msgs_run(InputCollect, refresh, quiet, lambda_control=None)
    objective_weights = check_obj_weight(InputCollect['calibration_input'], objective_weights, refresh)

    # Prepare hyper-parameters
    hyper_collect = hyper_collector(
        InputCollect,
        hyper_in=InputCollect['hyperparameters'],
        ts_validation=ts_validation,
        add_penalty_factor=add_penalty_factor,
        cores=cores,
        dt_hyper_fixed=dt_hyper_fixed
    )
    InputCollect['hyper_updated'] = hyper_collect['hyper_list_all']

    # Run robyn_mmm() for each trial
    OutputModels = robyn_train(
        InputCollect, hyper_collect,
        cores=cores, iterations=iterations, trials=trials,
        intercept_sign=intercept_sign, intercept=intercept,
        nevergrad_algo=nevergrad_algo,
        dt_hyper_fixed=dt_hyper_fixed,
        ts_validation=ts_validation,
        add_penalty_factor=add_penalty_factor,
        rssd_zero_penalty=rssd_zero_penalty,
        objective_weights=objective_weights,
        refresh=refresh, seed=seed, quiet=quiet
    )

    OutputModels['metadata']['hyper_fixed'] = hyper_collect['all_fixed']['hyper_fixed']
    OutputModels['metadata']['bootstrap'] = bootstrap
    OutputModels['metadata']['refresh'] = refresh
    OutputModels['metadata']['train_timestamp'] = time.time()
    OutputModels['metadata']['cores'] = cores
    OutputModels['metadata']['iterations'] = iterations
    OutputModels['metadata']['trials'] = trials
    OutputModels['metadata']['intercept'] = intercept
    OutputModels['metadata']['intercept_sign'] = intercept_sign
    OutputModels['metadata']['nevergrad_algo'] = nevergrad_algo
    OutputModels['metadata']['ts_validation'] = ts_validation
    OutputModels['metadata']['add_penalty_factor'] = add_penalty_factor
    OutputModels['metadata']['hyper_updated'] = hyper_collect['hyper_list_all']

    # Handling different output conditions
    if dt_hyper_fixed is None:
        output = OutputModels
    elif not hyper_collect['all_fixed']:
        # Direct output & not all fixed hyperparameters, including refresh mode
        output = robyn_outputs(InputCollect, OutputModels, refresh=refresh, *args, **kwargs)
    else:
        # Direct output & all fixed hyperparameters, thus no cluster
        output = robyn_outputs(InputCollect, OutputModels, clusters=False, *args, **kwargs)

    # Check convergence when more than 1 iteration
    if not hyper_collect['all_fixed']['hyper_fixed']:
        output["convergence"] = robyn_converge(OutputModels, *args, **kwargs)
        output["ts_validation_plot"] = ts_validation_fun(OutputModels, *args, **kwargs)
    else:
        if "solID" in dt_hyper_fixed:
            output["selectID"] = dt_hyper_fixed["solID"]
        else:
            output["selectID"] = OutputModels['trial1']['resultCollect']['resultHypParam']['solID']
        if not quiet:
            print(f"Successfully recreated model ID: {output['selectID']}")

    # Save hyper-parameters list
    output["hyper_updated"] = hyper_collect['hyper_list_all']
    output["seed"] = seed

    # Report total timing
    t0_datetime = datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')
    current_time = datetime.fromtimestamp(time.time())
    time_diff_seconds = (current_time - t0_datetime).total_seconds()
    runTime = round(time_diff_seconds / 60, 2)

    if not quiet and iterations > 1:
        print(f"Total run time: {runTime} mins")

    output['__class__'] = "robyn_models"  # Assuming the need to store class information
    return output

#' @rdname robyn_run
#' @aliases robyn_run
#' @param x \code{robyn_models()} output.
#' @export
def print_robyn_models(x):
    is_fixed = all(len(h) == 1 for h in x['hyper_updated'].values())
    total_iters = f"({nrow(x['trial1']['resultCollect']['resultHypParam'])} real)" if "trial1" in x else "(1 real)"
    iters = ", ".join(map(str, x['convergence']['errors']['cuts'][-2:])) if x.get('convergence') else "1"
    fixed = " (fixed)" if is_fixed else ""
    convergence = "\n  ".join(x['convergence']['conv_msg']) if not is_fixed else "Fixed hyper-parameters"
    hypers = flatten_hyps(x['hyper_updated'])

    print(f"""
    Total trials: {x['trials']}
    Iterations per trial: {x['iterations']} {total_iters}
    Runtime (minutes): {x.get('runTime')}
    Cores: {x['cores']}

    Updated Hyper-parameters{fixed}:
    {hypers}

    Nevergrad Algo: {x['nevergrad_algo']}
    Intercept: {x['intercept']}
    Intercept sign: {x['intercept_sign']}
    Time-series validation: {x['ts_validation']}
    Penalty factor: {x['add_penalty_factor']}
    Refresh: {bool(x.get('refresh'))}

    Convergence on last quantile (iters {iters}):
        {convergence}
    """)

    def nrow(df):
        # Assuming df is a DataFrame or similar structure
        return len(df)

    def flatten_hyps(hyp_dict):
        # This function should transform the hyperparameters dictionary into a string
        # Assuming a simple implementation here; modify as needed
        return '\n  '.join(f'{k}: {v}' for k, v in hyp_dict.items())

    if "robyn_outputs" in x.get('__class__', []):
        clusters_info = ""
        if "clusters" in x:
            clusters_info = f"Clusters (k = {x['clusters']['n_clusters']}): {', '.join(x['clusters']['models']['solID'])}"

        print(f"""
            Plot Folder: {x.get('plot_folder')}
            Calibration Constraint: {x.get('calibration_constraint')}
            Hyper-parameters fixed: {x.get('hyper_fixed')}
            Pareto-front ({x.get('pareto_fronts')}) All solutions ({len(x.get('allSolutions', []))}): {', '.join(x.get('allSolutions', []))}
            {clusters_info}
        """)

####################################################################
#' Train Robyn Models
#'
#' \code{robyn_train()} consumes output from \code{robyn_input()}
#' and runs the \code{robyn_mmm()} on each trial.
#'
#' @inheritParams robyn_run
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @return List. Iteration results to include in \code{robyn_run()} results.
#' @export
def robyn_train(InputCollect, hyper_collect, cores, iterations, trials,
                intercept_sign, intercept, nevergrad_algo, dt_hyper_fixed=None,
                ts_validation=True, add_penalty_factor=False, objective_weights=None,
                rssd_zero_penalty=True, refresh=False, seed=123, quiet=False):

    hyper_fixed = hyper_collect['all_fixed']

    OutputModels = {
        "trials": [],
        "metadata": {}
    }

    if hyper_fixed['hyper_fixed'] == True:
        OutputModels["trials"].append(robyn_mmm(
            InputCollect=InputCollect,
            hyper_collect=hyper_collect,
            iterations=iterations,
            cores=cores,
            nevergrad_algo=nevergrad_algo,
            intercept_sign=intercept_sign,
            intercept=intercept,
            dt_hyper_fixed=dt_hyper_fixed,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            seed=seed,
            quiet=quiet
        ))
        OutputModels["trials"][0]['trial'] = 1

        if "solID" in dt_hyper_fixed:
            these = ["resultHypParam", "xDecompVec", "xDecompAgg", "decompSpendDist"]
            for tab in these:
                OutputModels["trials"][0]['resultCollect'][tab]['solID'] = dt_hyper_fixed['solID']

    else:
        # Run robyn_mmm() for each trial if hyperparameters are not all fixed
        check_init_msg(InputCollect, cores)
        if not quiet:
            calibration_phrase = "with calibration using" if InputCollect['calibration_input'] is not None else "using"
            print(f">>> Starting {trials} trials with {iterations} iterations each {calibration_phrase} {nevergrad_algo} nevergrad algorithm...")

        for ngt in range(1, trials + 1):  # Python uses 0-based indexing, so range is adjusted
            if not quiet:
                print(f"  Running trial {ngt} of {trials}")
            model_output = robyn_mmm(
                InputCollect=InputCollect,
                hyper_collect=hyper_collect,
                iterations=iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept_sign=intercept_sign,
                intercept=intercept,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                rssd_zero_penalty=rssd_zero_penalty,
                objective_weights=objective_weights,
                refresh=refresh,
                trial=ngt,
                seed=seed + ngt,
                quiet=quiet
            )
            check_coef0 = any(value == float('inf') for value in model_output['resultCollect']['decompSpendDist']['decomp.rssd'])
            if check_coef0:
                # Assuming model_output['resultCollect']['decompSpendDist'] is a DataFrame or similar
                num_coef0_mod = len(model_output['resultCollect']['decompSpendDist'][model_output['resultCollect']['decompSpendDist']['decomp.rssd'].apply(lambda x: x == float('inf'))].drop_duplicates(subset=['iterNG', 'iterPar']))
                num_coef0_mod = min(num_coef0_mod, iterations)
                if not quiet:
                    print(f"This trial contains {num_coef0_mod} iterations with all media coefficient = 0. "
                          "Please reconsider your media variable choice if the pareto choices are unreasonable.\n"
                          "   Recommendations:\n"
                          "1. Increase hyperparameter ranges for 0-coef channels to give Robyn more freedom\n"
                          "2. Split media into sub-channels, and/or aggregate similar channels, and/or introduce other media\n"
                          "3. Increase trials to get more samples")
            model_output['trial'] = ngt
            OutputModels["trials"].append(model_output)

    for i in range(len(OutputModels["trials"])):
        OutputModels["trials"][i]['name'] = f"trial{i + 1}"

    return OutputModels

####################################################################
#' Core MMM Function
#'
#' \code{robyn_mmm()} function activates Nevergrad to generate samples of
#' hyperparameters, conducts media transformation within each loop, fits the
#' Ridge regression, calibrates the model optionally, decomposes responses
#' and collects the result. It's an inner function within \code{robyn_run()}.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @param iterations Integer. Number of iterations to run.
#' @param trial Integer. Which trial are we running? Used to ID each model.
#' @return List. MMM results with hyperparameters values.
#' @export
def robyn_mmm(InputCollect,
              hyper_collect,
              iterations,
              cores,
              nevergrad_algo,
              intercept_sign,
              intercept=True,
              ts_validation=True,
              add_penalty_factor=False,
              objective_weights=None,
              dt_hyper_fixed=None,
              rssd_zero_penalty=True,
              refresh=False,
              trial=1,
              seed=123,
              quiet=False):

    ## This is not necessary as nevergrad is being used in R with Interface.
    ## try:
    ##    import nevergrad as ng
    ## except ImportError:
    ##    raise ImportError("You must have the nevergrad python library installed.\n"
    ##                      "Please check the installation instructions: "
    ##                      "https://github.com/facebookexperimental/Robyn/blob/main/demo/install_nevergrad.R")

    #if isinstance(seed, int):
    #    np.random.seed(124)
    #    random.seed(124)

    ################################################
    #### Collect hyperparameters

    ##hypParamSamName = list(hyper_collect['hyper_list_all'].keys())
    hypParamSamName = list(hyper_collect['hyper_list_all'].keys())
    # Optimization hyper-parameters
    hyper_bound_list_updated = hyper_collect['hyper_bound_list_updated']
    hyper_bound_list_updated_name = list(hyper_bound_list_updated.keys())
    hyper_count = len(hyper_bound_list_updated_name)
    # Fixed hyper-parameters
    hyper_bound_list_fixed = hyper_collect['hyper_bound_list_fixed']
    hyper_bound_list_fixed_name = list(hyper_bound_list_fixed.keys())
    hyper_count_fixed = len(hyper_bound_list_fixed_name)
    dt_hyper_fixed_mod = hyper_collect['dt_hyper_fixed_mod']
    hyper_fixed = hyper_collect['all_fixed']

    ################################################
    #### Setup environment

    ##if InputCollect.get('dt_mod') is None:
    if 'dt_mod' not in InputCollect.keys():
        raise ValueError("Run InputCollect['dt_mod'] = robyn_engineering() first to get the dt_mod")

    # Since the condition is always TRUE, we directly assign the variables
    dt_mod = InputCollect['dt_mod']
    xDecompAggPrev = InputCollect['xDecompAggPrev']
    rollingWindowStartWhich = InputCollect.get('rollingWindowStartWhich')
    rollingWindowEndWhich = InputCollect.get('rollingWindowEndWhich')
    refreshAddedStart = InputCollect.get('refreshAddedStart')
    dt_modRollWind = InputCollect.get('dt_modRollWind')
    refresh_steps = InputCollect.get('refresh_steps')
    rollingWindowLength = InputCollect.get('rollingWindowLength')
    paid_media_spends = InputCollect.get('paid_media_spends')
    organic_vars = InputCollect.get('organic_vars')
    context_vars = InputCollect.get('context_vars')
    prophet_vars = InputCollect.get('prophet_vars')
    adstock = InputCollect.get('adstock')
    context_signs = InputCollect.get('context_signs')
    paid_media_signs = InputCollect.get('paid_media_signs')
    prophet_signs = InputCollect.get('prophet_signs')
    organic_signs = InputCollect.get('organic_signs')
    calibration_input = InputCollect.get('calibration_input')
    optimizer_name = nevergrad_algo
    i = None  # For parallel iterations (globalVar)

    ################################################
    #### Get spend share

    dt_inputTrain = InputCollect['dt_input'].iloc[(rollingWindowStartWhich-1):rollingWindowEndWhich]
    temp = dt_inputTrain[paid_media_spends]
    dt_spendShare = pd.DataFrame({
        'rn': paid_media_spends,
        'total_spend': temp.sum(),
        'mean_spend': temp.mean()
    })
    dt_spendShare['spend_share'] = dt_spendShare['total_spend'] / dt_spendShare['total_spend'].sum()

    # When not refreshing, dt_spendShareRF = dt_spendShare
    ## refreshAddedStartWhich = dt_modRollWind[dt_modRollWind['ds'] == refreshAddedStart].index[0]
    ## Return the index which is equal to refreshAddedStart
    refreshAddedStartWhich = dt_modRollWind.index[dt_modRollWind['ds'] == refreshAddedStart].tolist()[0] - (rollingWindowStartWhich - 1)

    temp = dt_inputTrain[paid_media_spends].iloc[(refreshAddedStartWhich):rollingWindowLength]
    dt_spendShareRF = pd.DataFrame({
        'rn': paid_media_spends,
        'total_spend': temp.sum(),
        'mean_spend': temp.mean()
    })
    dt_spendShareRF['spend_share'] = dt_spendShareRF['total_spend'] / dt_spendShareRF['total_spend'].sum()

    # Join both dataframes into a single one
    ##dt_spendShare = dt_spendShare.merge(dt_spendShareRF, on='rn', suffixes=('', '_refresh'))
    dt_spendShare = dt_spendShare.join(dt_spendShareRF, on='rn', how='left', rsuffix='_refresh')

    ################################################
    #### Get lambda

    lambda_min_ratio = 0.0001  # default value from glmnet
    # Assuming dt_mod is a DataFrame and dep_var is the name of the dependent variable column
    ##X = dt_mod[drop(columns=['ds', 'dep_var'])
    select_columns = [col for col in dt_mod.columns.values if col not in ['ds', 'dep_var']]
    X = dt_mod[select_columns].to_numpy()
    y = dt_mod[['dep_var']].to_numpy()
    # Generate a sequence of lambdas for regularization
    lambdas = lasso_path(X, y, eps=lambda_min_ratio)[0]  # lasso_path returns alphas which are equivalent to lambdas
    lambda_max = lambdas.max() * 0.1
    lambda_min = lambda_max * lambda_min_ratio

    # Start Nevergrad loop
    t0 = time.time()

    # Set iterations
    ##if not hyper_fixed:
    if hyper_fixed['hyper_fixed'] == False:
        iterTotal = iterations
        iterPar = cores
        iterNG = int(np.ceil(iterations / cores))  # Sometimes the progress bar may not get to 100%
    else:
        iterTotal = iterPar = iterNG = 1

    # Start Nevergrad optimizer
    ##if not hyper_fixed:
    if hyper_fixed['hyper_fixed'] == False:
        my_tuple = tuple([hyper_count])
        instrumentation = ng.p.Array(shape=my_tuple, lower=0, upper=1)
        optimizer = optimizerlib.registry[optimizer_name](instrumentation, budget=iterTotal, num_workers=cores)
        print(instrumentation)
        print(iterTotal)
        # Set multi-objective dimensions for objective functions (errors)
        if calibration_input is None:
            optimizer.tell(ng.p.MultiobjectiveReference(), tuple([1, 1]))
            if objective_weights is None:
                objective_weights = tuple([1, 1])
            else:
                objective_weights = tuple([objective_weights[0], objective_weights[1]])
            optimizer.set_objective_weights(objective_weights)
        else:
            optimizer.tell(ng.p.MultiobjectiveReference(), tuple([1, 1, 1]))
            if objective_weights is None:
                objective_weights = tuple([1, 1, 1])
            else:
                objective_weights = tuple([objective_weights[0], objective_weights[1], objective_weights[2]])
            optimizer.set_objective_weights(objective_weights)

    result_collect_ng = []
    cnt = 0
    pb = None

    ##if not hyper_fixed and not quiet:
    ## May not be necessary since tqdm is used.
    if hyper_fixed['hyper_fixed'] == False and quiet == False:
        pbar = tqdm(total=iterTotal, desc="Progress", leave=True)
    #    pb = range(iterTotal)

    sys_time_dopar = None

    try:
        sys_time_dopar = time.time()
        for lng in tqdm(range(iterTotal), desc="Optimization Progress"):
            nevergrad_hp = {}
            nevergrad_hp_val = {}
            hypParamSamList = []
            hypParamSamNG = dict()

            ##if not hyper_fixed:
            if hyper_fixed['hyper_fixed'] == False:
                # Setting initial seeds (co = cores)
                ##for co in range(1, iterPar + 1):  # co = 1
                for co in range(0, iterPar):  # co = 1
                    # Get hyperparameter sample with ask (random)
                    nevergrad_hp[co] = optimizer.ask()
                    nevergrad_hp_val[co] = nevergrad_hp[co].value

                    # Scale sample to given bounds using uniform distribution
                    ## [True if var in hyper_bound_list_updated_name else False for var in hyper_bound_list_updated.keys()]
                    for hypNameLoop in hyper_bound_list_updated_name:
                        index = hyper_bound_list_updated_name.index(hypNameLoop)
                        channelBound = hyper_bound_list_updated[hypNameLoop]

                        hyppar_value = round(nevergrad_hp_val[co][index], 10)

                        if len(channelBound) > 1:
                            ##hypParamSamNG[hypNameLoop] = uniform(hyppar_value, min(channelBound), max(channelBound))
                            hypParamSamNG[hypNameLoop] = uniform.ppf(hyppar_value, loc=min(channelBound), scale=max(channelBound) - min(channelBound))
                        else:
                            hypParamSamNG[hypNameLoop] = hyppar_value

                    hypParamSamList.append(pd.DataFrame(hypParamSamNG, index=[0])) ## .T)

                hypParamSamNG = pd.concat(hypParamSamList, ignore_index=True)
                hypParamSamNG.columns = hyper_bound_list_updated_name

                # Add fixed hyperparameters
                if hyper_count_fixed != 0:
                    train_size_value = dt_hyper_fixed_mod[1][0][0]
                    hypParamSamNG['train_size'] = [train_size_value] * len(hypParamSamNG)
                    valid_columns = set(hypParamSamNG.columns).intersection(hypParamSamName)
                    hypParamSamNG = hypParamSamNG[list(valid_columns)]

            else:
                hypParamSamNG = dt_hyper_fixed_mod[hypParamSamName]

            # Initialize lists to collect results
            nrmse_collect = []
            decomp_rssd_collect = []
            mape_lift_collect = []

            ## robyn_iterations()
            # Define a function to run robyn_iterations
            ## def run_robyn_iterations(i):
            ##   return robyn_iterations(i)


            def robyn_iterations(iteration, hypParamSamNG, adstock):  # i=1
                t1 = time.time()
                i = iteration

                # Get hyperparameter sample
                hypParamSam = hypParamSamNG.iloc[i - 1]  # Adjusted for 0-based indexing

                # Check and transform adstock
                adstock = check_adstock(adstock)

                # Transform media for model fitting
                temp = run_transformations(InputCollect, hypParamSam, adstock)
                dt_modSaturated = temp['dt_modSaturated']
                dt_saturatedImmediate = temp['dt_saturatedImmediate']
                dt_saturatedCarryover = temp['dt_saturatedCarryover']

                # Split train & test and prepare data for modeling
                dt_window = dt_modSaturated

                # Contrast matrix because glmnet does not treat categorical variables (one hot encoding)
                y_window = list(dt_window['dep_var'].values)
                select_columns = [var for var in dt_window.columns.values if var != 'dep_var' and var != 'index']
                x_window = pd.get_dummies(dt_window[select_columns]) ##, columns=select_columns) ##TODO: better to use OneHotEncoding or not?
                ## x_window = lares.ohse(dt_window.drop('dep_var', axis=1)).values  # Assuming ohse returns a DataFrame

                y_train = y_val = y_test = y_window
                x_train = x_val = x_test = x_window

                train_size = hypParamSam['train_size'] ##.values[0]
                val_size = test_size = (1 - train_size) / 2
                if train_size < 1:
                    ## train_size_index = int(train_size * len(dt_window))
                    train_size_index = int(np.floor(np.quantile(range(len(dt_window)), [train_size])[0]))
                    ## val_size_index = train_size_index + int(val_size * len(dt_window))
                    val_size_index = train_size_index + int(np.floor(val_size * len(dt_window)))
                    y_train = y_window[:train_size_index]
                    y_val = y_window[train_size_index:val_size_index]
                    y_test = y_window[val_size_index:]
                    x_train = x_window[:train_size_index]
                    x_val = x_window[train_size_index:val_size_index]
                    x_test = x_window[val_size_index:]
                else:
                    y_val = y_test = x_val = x_test = None

                # Define and set sign control
                select_columns = [var for var in dt_window.columns.values if var != 'dep_var' and var != 'index']
                dt_sign = dt_window[select_columns]
                x_sign = prophet_signs + context_signs + paid_media_signs + organic_signs #TODO: why signs are not same as in R??
                check_factor = dt_sign.applymap(lambda x: isinstance(x, pd.CategoricalDtype)) ##TODO: In R all returns False
                lower_limits = [0] * len(prophet_signs)
                upper_limits = [1] * len(prophet_signs)
                for s in range(len(prophet_signs), len(x_sign)):
                    if s in check_factor.index:
                        level_n = len(dt_sign.iloc[:, s].astype('category').cat.categories)
                        if level_n <= 1:
                            raise ValueError("All factor variables must have more than 1 level")
                        lower_vec = [0] * (level_n - 1) if x_sign[s] == "positive" else [-float('inf')] * (level_n - 1)
                        upper_vec = [0] * (level_n - 1) if x_sign[s] == "negative" else [float('inf')] * (level_n - 1)
                        lower_limits.extend(lower_vec)
                        upper_limits.extend(upper_vec)
                    else:
                        lower_limits.append(0 if x_sign[s] == "positive" else -float('inf'))
                        upper_limits.append(0 if x_sign[s] == "negative" else float('inf'))

                # Fit ridge regression with nevergrad's lambda
                lambda_hp = float(hypParamSamNG['lambda'].iloc[i-1])
                ##if not hyper_fixed:
                if hyper_fixed['hyper_fixed'] == False:
                    lambda_scaled = lambda_min + (lambda_max - lambda_min) * lambda_hp
                else:
                    lambda_scaled = lambda_hp

                if add_penalty_factor:
                    penalty_factor = hypParamSamNG.iloc[i, [col.endswith("_penalty") for col in hypParamSamNG.columns]]
                else:
                    penalty_factor = [1] * x_train.shape[1]

                mod_out = model_refit(
                    x_train, y_train,
                    x_val, y_val,
                    x_test, y_test,
                    lambda_scaled,
                    lower_limits,
                    upper_limits,
                    intercept,
                    intercept_sign,
                    penalty_factor
                )

                decompCollect = model_decomp(
                    coefs=mod_out["coefs"],
                    y_pred=mod_out["y_pred"],
                    dt_modSaturated=dt_modSaturated,
                    dt_saturatedImmediate=dt_saturatedImmediate,
                    dt_saturatedCarryover=dt_saturatedCarryover,
                    dt_modRollWind=dt_modRollWind,
                    refreshAddedStart=refreshAddedStart
                )

                nrmse = mod_out["nrmse_val"] if ts_validation else mod_out["nrmse_train"]
                mape = 0
                df_int = mod_out["df_int"]

                # MAPE: Calibration error
                if calibration_input is not None:
                    liftCollect = robyn_calibrate(
                        calibration_input=calibration_input,
                        df_raw=dt_mod,
                        hypParamSam=hypParamSam,
                        wind_start=rollingWindowStartWhich,
                        wind_end=rollingWindowEndWhich,
                        dayInterval=InputCollect["dayInterval"],
                        adstock=adstock,
                        xDecompVec=decompCollect["xDecompVec"],
                        coefs=decompCollect["coefsOutCat"]
                    )
                    mape = liftCollect["mape_lift"].mean()

                df_xDecompAgg = decompCollect["xDecompAgg"][decompCollect["xDecompAgg"].index.isin(paid_media_spends)]
                dt_spendShare_idx = dt_spendShare.set_index('rn')
                dt_decompSpendDist = df_xDecompAgg.join(dt_spendShare_idx, how='inner')

                selected_columns = [
                    'coefs', 'xDecompAgg', 'xDecompPerc', 'xDecompMeanNon0', 'xDecompMeanNon0Perc',
                    'xDecompAggRF', 'xDecompPercRF', 'xDecompMeanNon0RF', 'xDecompMeanNon0PercRF',
                    'pos', 'total_spend', 'mean_spend', 'spend_share', 'mean_spend_refresh'
                ]

                dt_decompSpendDist = dt_decompSpendDist[selected_columns]

                dt_decompSpendDist = dt_decompSpendDist.assign(
                    effect_share=dt_decompSpendDist['xDecompPerc'] / dt_decompSpendDist['xDecompPerc'].sum(),
                    effect_share_refresh=dt_decompSpendDist['xDecompPercRF'] / dt_decompSpendDist['xDecompPercRF'].sum()
                )

                if not refresh:
                    decomp_rssd = sqrt(sum((dt_decompSpendDist["effect_share"] - dt_decompSpendDist["spend_share"])**2))

                    # Penalty for models with more 0-coefficients
                    if rssd_zero_penalty:
                        is_0eff = (dt_decompSpendDist["effect_share"].round(4) == 0)
                        share_0eff = sum(is_0eff) / len(dt_decompSpendDist["effect_share"])
                        decomp_rssd = decomp_rssd * (1 + share_0eff)
                else:
                    dt_decompRF = dt_decompSpendDist[["rn", "effect_share"]].merge(
                        xDecompAggPrev[["rn", "decomp_perc_prev"]],
                        on="rn"
                    )
                    decomp_rssd_media = dt_decompRF[dt_decompRF["rn"].isin(paid_media_spends)]["decomp_perc"].mean()
                    decomp_rssd_nonmedia = dt_decompRF[~dt_decompRF["rn"].isin(paid_media_spends)]["decomp_perc"].mean()
                    decomp_rssd = decomp_rssd_media + decomp_rssd_nonmedia / (1 - refresh_steps / rollingWindowLength)

                # Handle the case when all media in this iteration have 0 coefficients
                if math.isnan(decomp_rssd):
                    decomp_rssd = math.inf
                    dt_decompSpendDist["effect_share"] = 0

                # Initialize resultCollect list
                resultCollect = {}

                # Create a common DataFrame with shared values
                common = pd.DataFrame({
                    "rsq_train": mod_out["rsq_train"],
                    "rsq_val": mod_out["rsq_val"],
                    "rsq_test": mod_out["rsq_test"],
                    "nrmse_train": mod_out["nrmse_train"],
                    "nrmse_val": mod_out["nrmse_val"],
                    "nrmse_test": mod_out["nrmse_test"],
                    "nrmse": nrmse,
                    "decomp.rssd": decomp_rssd,
                    "mape": mape,
                    "lambda": lambda_scaled,
                    "lambda_hp": lambda_hp,
                    "lambda_max": lambda_max,
                    "lambda_min_ratio": lambda_min_ratio,
                    "solID": f"{trial}_{lng}_{i}",
                    "trial": trial,
                    "iterNG": lng,
                    "iterPar": i
                }, index=[0])

                total_common = common.shape[1]
                split_common = common.columns.get_loc("lambda_min_ratio")
                hypParamSam_df = hypParamSam.drop('lambda').to_frame().T

                current_time = time.time()
                Elapsed = current_time - t1
                ElapsedAccum = current_time - t0

                hypParamSam_df_reset = hypParamSam_df.reset_index(drop=True)
                common_reset = common.reset_index(drop=True)

                result_df = hypParamSam_df_reset.join(
                    common_reset.iloc[:, :split_common]
                )

                result_df = result_df.assign(
                    pos=decompCollect['xDecompAgg']['pos'].prod(),  # Adjust according to actual structure
                    Elapsed=Elapsed,
                    ElapsedAccum=ElapsedAccum
                )

                result_df = result_df.join(
                    common_reset.iloc[:, split_common:total_common]
                )

                dresult_hyp_param_df = pd.DataFrame(result_df)
                list_of_dicts = dresult_hyp_param_df.to_dict(orient='records')
                resultCollect["resultHypParam"] = list_of_dicts[0]

                repeated_common_reset = pd.concat([common_reset] * len(decompCollect["xDecompAgg"]), ignore_index=True)
                decompCollect_xDecompAgg_reset = decompCollect["xDecompAgg"].reset_index()
                resultCollect["xDecompAgg"] = pd.concat([decompCollect_xDecompAgg_reset, repeated_common_reset], axis=1)

                if liftCollect is not None:
                    for column in common_reset.columns:
                        liftCollect[column] = common_reset.iloc[0][column]
                    resultCollect["liftCalibration"] = liftCollect

                repeated_common_reset_decompSpendDist = pd.DataFrame([common_reset.iloc[0]] * len(dt_decompSpendDist))
                resultCollect["decompSpendDist"] = pd.concat([dt_decompSpendDist.reset_index(), repeated_common_reset_decompSpendDist.reset_index()], axis=1)
                resultCollect.update(common_reset.to_dict())

                return resultCollect


            cores = 1 ## TODO Remove to test MULTIPROCESSING when Completed Debugging robyn_iterations
            # Parallel processing
            if cores == 1:
                ##dopar_collect = [run_robyn_iterations(i) for i in range(1, iterPar + 1)]
                dopar_collect = [robyn_iterations(i,hypParamSamNG, adstock) for i in range(1, iterPar + 1)]
            else:
                # Create a pool of worker processes
                if check_parallel() and hyper_fixed['hyper_fixed'] == False: ##not hyper_fixed:
                    pool = multiprocessing.Pool(processes=cores)
                else:
                    pool = multiprocessing.Pool(processes=1)

                # Use the pool to run robyn_iterations in parallel
                ##dopar_collect = pool.map(run_robyn_iterations, range(1, iterPar + 1))
                dopar_collect = pool.map(robyn_iterations, zip(range(1, iterPar + 1),
                                                               InputCollect,
                                                               hypParamSamNG,
                                                               prophet_signs,
                                                               context_signs,
                                                               paid_media_signs,
                                                               organic_signs,
                                                               adstock))
                pool.close()
                pool.join()

            # Collect nrmse, decomp.rssd, and mape.lift from the results
            for result in dopar_collect:
                nrmse_collect.append(result['nrmse'])
                decomp_rssd_collect.append(result['decomp.rssd'])
                mape_lift_collect.append(result['mape'])

            # Update optimizer objectives if not hyper_fixed
            ##if not hyper_fixed:
            if hyper_fixed['hyper_fixed'] == False:
                if calibration_input is None:
                    for co in range(1, iterPar + 1):
                        nrmse_collect_value = next(iter(nrmse_collect[co].values()), None)
                        decomp_rssd_collect_value = next(iter(decomp_rssd_collect[co].values()), None)
                        optimizer.tell(nevergrad_hp[co - 1], [nrmse_collect_value, decomp_rssd_collect_value])
                else:
                    for co in range(iterPar):  # iterPar is the total count, range() is 0-based
                        nrmse_collect_value = next(iter(nrmse_collect[co].values()), None)
                        decomp_rssd_collect_value = next(iter(decomp_rssd_collect[co].values()), None)
                        mape_lift_collect_value = next(iter(mape_lift_collect[co].values()), None)
                        optimizer.tell(nevergrad_hp[co], [nrmse_collect_value, decomp_rssd_collect_value, mape_lift_collect_value])

            result_collect_ng.append(dopar_collect)

            if not quiet:
                cnt += iterPar
                if hyper_fixed['hyper_fixed'] == False:
                    pbar.update(1)

    except Exception as err:
        if len(result_collect_ng) > 1:
            msg = "Error while running robyn_mmm(); providing PARTIAL results"
            print("Warning:", msg)
            print("Error:", err)
            sys_time_dopar = [time.time() - t0] * 3
        else:
            raise err

    pbar.close()
    # Stop the cluster to avoid memory leaks
    ## TODO: Is it necessary for Python?
    ## stop_implicit_cluster()
    ## register_do_seq()
    ## get_do_par_workers()

    if hyper_fixed['hyper_fixed'] == False:
        elapsed_time = time.time() - sys_time_dopar
        print("\r", f"\n  Finished in {round(elapsed_time / 60, 2)} mins")
        sys.stdout.flush()

    # Final result collect
    result_collect = {}

    # Construct resultHypParam
    resultHypParamDefaultDict = defaultdict(list)
    for group in result_collect_ng:
        for element in group:
            for key, value in element["resultHypParam"].items():
                resultHypParamDefaultDict[key].append(value)
    resultHypParam = dict(resultHypParamDefaultDict)
    result_collect["resultHypParam"] = pd.DataFrame.from_dict(resultHypParamDefaultDict)

    # Construct xDecompAgg
    rn_list = []
    xDecompAggDefaultDict = defaultdict(list)
    for group in result_collect_ng:
        for element in group:
            df = element['xDecompAgg']
            for _, row in df.iterrows():
                for col in df.columns:
                    xDecompAggDefaultDict[col].append(row[col])
                rn_list.append(row['rn'])

    xDecompAggDefaultDict['rn'] = rn_list
    result_collect["xDecompAgg"] = pd.DataFrame.from_dict(xDecompAggDefaultDict)
    cols = ['rn'] + [col for col in result_collect["xDecompAgg"] if col != 'rn']
    result_collect["xDecompAgg"] = result_collect["xDecompAgg"][cols]

    # Construct liftCalibration
    liftCalibrationDefaultDict = defaultdict(list)
    for group in result_collect_ng:
        for element in group:
            df = element['liftCalibration']
            for col in df.columns:
                liftCalibrationDefaultDict[col].extend(df[col].tolist())
    liftCalibrationDF = pd.DataFrame.from_dict(liftCalibrationDefaultDict)
    sortedLiftCalibrationDF = liftCalibrationDF.sort_values(by=['mape', 'liftMedia', 'liftStart'])
    result_collect["liftCalibration"] = pd.DataFrame.from_dict(sortedLiftCalibrationDF)

    # Construct decompSpendDist
    rn_list = []
    decompSpendDistDefaultDict = defaultdict(list)
    for group in result_collect_ng:
        for element in group:
            df = element['decompSpendDist']
            for _, row in df.iterrows():
                for col in df.columns:
                    decompSpendDistDefaultDict[col].append(row[col])
                rn_list.append(row['rn'])
    decompSpendDistDefaultDict['rn'] = rn_list
    result_collect["decompSpendDist"] = pd.DataFrame.from_dict(decompSpendDistDefaultDict)
    cols = ['rn'] + [col for col in result_collect["decompSpendDist"] if col != 'rn']
    result_collect["decompSpendDist"] = result_collect["decompSpendDist"][cols]

    # Skil following line for now as it looks a bug in the R code to me.
    # ["mape"] is a column of result_collect["liftCalibration"]. In R is always initialized to 0
    #result_collect["iter"] = len(result_collect["mape"])
    result_collect["elapsed.min"] = sys_time_dopar / 60

    # Adjust accumulated time
    result_collect["resultHypParam"]["ElapsedAccum"] = (
        result_collect["resultHypParam"]["ElapsedAccum"]
        - min(result_collect["resultHypParam"]["ElapsedAccum"])
        + result_collect["resultHypParam"]["Elapsed"].iloc[
            result_collect["resultHypParam"]["ElapsedAccum"].idxmin()
        ]
    )

    return {
        "resultCollect": result_collect,
        "hyperBoundNG": hyper_bound_list_updated,
        "hyperBoundFixed": hyper_bound_list_fixed,
    }


def model_decomp(coefs, y_pred, dt_modSaturated, dt_saturatedImmediate,
                 dt_saturatedCarryover, dt_modRollWind, refreshAddedStart):
    # Input for decomp
    y = dt_modSaturated['dep_var']
    x = dt_modSaturated.drop(columns=['dep_var'])

    #print("--------- coefs")
    #print(coefs)
    intercept = coefs.loc['Intercept', 's0']
    #print("--------- intercept")
    #print(intercept)
    x_name = x.columns
    x_factor = [col for col in x_name if isinstance(x[col][0], str)]

    coeffs_series = coefs['s0'].drop('Intercept')
    x_decomp = x.multiply(coeffs_series, axis=1)
    intercept_column = pd.Series([coefs.loc['Intercept', 's0']] * len(x_decomp), index=x_decomp.index)
    x_decomp.insert(0, 'intercept', intercept_column)
    x_decomp.reset_index(drop=True, inplace=True)
    if 'index' in x_decomp.columns:
        x_decomp.drop(columns=['index'], inplace=True)

    y_y_pred_df = pd.DataFrame({'y': y, 'y_pred': y_pred})
    x_decomp_out = pd.concat([dt_modRollWind[['ds']].reset_index(), y_y_pred_df.reset_index(), x_decomp.reset_index()], axis=1).drop(columns=['index'])
    media_features = coefs.index.intersection(dt_saturatedImmediate.columns)
    media_decomp_immediate = dt_saturatedImmediate[media_features].multiply(coefs.loc[media_features, 's0'], axis=1)
    media_decomp_carryover = dt_saturatedCarryover[media_features].multiply(coefs.loc[media_features, 's0'], axis=1)

    # Output decomp
    y_hat = x_decomp.sum(axis=1, skipna=True)
    y_hat_scaled = np.abs(x_decomp).sum(axis=1, skipna=True)
    x_decomp_out_perc_scaled = np.abs(x_decomp).divide(y_hat_scaled, axis=0)
    x_decomp_out_scaled = x_decomp_out_perc_scaled.multiply(y_hat, axis=0)

    existing_cols = ['intercept'] + [col for col in x_name if col in x_decomp_out.columns]
    temp = x_decomp_out[existing_cols]

    x_decomp_out_agg = temp.sum()

    x_decomp_out_agg_perc = x_decomp_out_agg / y_hat.sum()
    x_decomp_out_agg_mean_non0 = temp.apply(lambda x: 0 if np.isnan(np.mean(x[x > 0])) else np.mean(x[x != 0]))
    x_decomp_out_agg_mean_non0[np.isnan(x_decomp_out_agg_mean_non0)] = 0
    numeric_df = x_decomp_out_agg_mean_non0.apply(pd.to_numeric, errors='coerce')
    x_decomp_out_agg_mean_non0 = x_decomp_out_agg_mean_non0.astype(float)
    x_decomp_out_agg_mean_non0_perc = x_decomp_out_agg_mean_non0 / x_decomp_out_agg_mean_non0.sum().sum()

    refresh_added_start_which = x_decomp_out.index[x_decomp_out['ds'] == refreshAddedStart].tolist()[0]
    refresh_added_end = x_decomp_out['ds'].max()
    refresh_added_end_which = x_decomp_out.index[x_decomp_out['ds'] == refresh_added_end].tolist()[0]

    existing_cols = ['intercept'] + [col for col in x_name if col in x_decomp_out.columns]
    temp = x_decomp_out[existing_cols]
    temp = temp.loc[refresh_added_start_which:refresh_added_end_which]
    x_decomp_out_agg_rf = temp.sum()
    y_hat_rf = y_hat.loc[refresh_added_start_which:refresh_added_end_which]
    x_decomp_out_agg_perc_rf = x_decomp_out_agg_rf / y_hat_rf.sum()
    x_decomp_out_agg_mean_non0_rf = temp.apply(lambda x: 0 if np.isnan(np.mean(x[x > 0])) else np.mean(x[x != 0]))
    x_decomp_out_agg_mean_non0_rf[np.isnan(x_decomp_out_agg_mean_non0_rf)] = 0
    x_decomp_out_agg_mean_non0_perc_rf = x_decomp_out_agg_mean_non0_rf / x_decomp_out_agg_mean_non0_rf.sum().sum()

    coefs_out_cat = pd.DataFrame({
        'rn': coefs.index,
        's0': coefs['s0']
    }).reset_index(drop=True)
    coefs_out_cat.set_index('rn', inplace=True)

    if len(x_factor) > 0:
        for factor in x_factor:
            coefs_out_cat['rn'] = coefs_out_cat['rn'].apply(lambda x: re.sub(f"{factor}.*", factor, x))

    x_decomp_out_agg = x_decomp_out_agg.rename({'intercept': 'Intercept'})
    coefs_out = coefs_out_cat.copy()
    coefs_out.rename(columns={'s0': 'coefs'}, inplace=True)

    decomp_out_agg = coefs_out.copy()
    decomp_out_agg['xDecompAgg'] = x_decomp_out_agg.values
    decomp_out_agg['xDecompPerc'] = x_decomp_out_agg_perc.values
    decomp_out_agg['xDecompMeanNon0'] = x_decomp_out_agg_mean_non0.values
    decomp_out_agg['xDecompMeanNon0Perc'] = x_decomp_out_agg_mean_non0_perc.values
    decomp_out_agg['xDecompAggRF'] = x_decomp_out_agg_rf.values
    decomp_out_agg['xDecompPercRF'] = x_decomp_out_agg_perc_rf.values
    decomp_out_agg['xDecompMeanNon0RF'] = x_decomp_out_agg_mean_non0_rf.values
    decomp_out_agg['xDecompMeanNon0PercRF'] = x_decomp_out_agg_mean_non0_perc_rf.values
    decomp_out_agg['pos'] = x_decomp_out_agg >= 0
    decomp_collect = {
        'xDecompVec': x_decomp_out,
        'xDecompVec.scaled': x_decomp_out_scaled,
        'xDecompAgg': decomp_out_agg,
        'coefsOutCat': coefs_out_cat,
        'mediaDecompImmediate': media_decomp_immediate.assign(ds=x_decomp_out['ds'], y=x_decomp_out['y']),
        'mediaDecompCarryover': media_decomp_carryover.assign(ds=x_decomp_out['ds'], y=x_decomp_out['y'])
    }
    return decomp_collect

def model_refit(x_train, y_train, x_val, y_val, x_test, y_test, lambda_, lower_limits, upper_limits, intercept=True, intercept_sign="non_negative", penalty_factor=None, *args, **kwargs):

    ridge = Ridge(alpha=lambda_, fit_intercept=intercept, random_state=42)
    ridge.fit(x_train, y_train)

    if intercept_sign == "non_negative " and ridge.intercept_ < 0:
        ridge = Ridge(alpha=lambda_, fit_intercept=False, random_state=42)
        ridge.fit(x_train, y_train)
        intercept = False

    coefs = np.append(ridge.intercept_, ridge.coef_) if intercept else ridge.coef_
    feature_names = ['Intercept'] + list(x_train.columns) if intercept else x_train.columns

    coefs_df = pd.DataFrame(coefs, index=feature_names, columns=['s0'])

    def get_adjusted_r2(y_true, y_pred, n_features, n_samples):
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - (1-r2)*(n_samples-1)/(n_samples-n_features-1)
        return adjusted_r2

    def get_nrmse(y_true, y_pred):
        y_true = np.array(y_true)  # Convert y_true to numpy array
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        nrmse = rmse / (y_true.max() - y_true.min())  # Now you can use .max() and .min()
        return nrmse

    y_train_pred = ridge.predict(x_train)
    n_samples, n_features = x_train.shape
    adjusted_r2_train = get_adjusted_r2(y_train, y_train_pred, n_features, n_samples)
    nrmse_train = get_nrmse(y_train, y_train_pred)

    results = {'train': (y_train_pred, adjusted_r2_train, nrmse_train)}

    for split, x, y in (('val', x_val, y_val), ('test', x_test, y_test)):
        if x is not None:
            y_pred = ridge.predict(x)
            adjusted_r2 = get_adjusted_r2(y, y_pred, x.shape[1], len(y))
            nrmse = get_nrmse(y, y_pred)
            results[split] = (y_pred, adjusted_r2, nrmse)
        else:
            results[split] = (None, None, None)

    mod_out = {
        "rsq_train": results['train'][1],
        "rsq_val": results['val'][1],
        "rsq_test": results['test'][1],
        "nrmse_train": results['train'][2],
        "nrmse_val": results['val'][2],
        "nrmse_test": results['test'][2],
        "coefs": coefs_df,
        "y_train_pred": results['train'][0],
        "y_val_pred": results['val'][0],
        "y_test_pred": results['test'][0],
        "y_pred": np.concatenate([y for y, _, _ in results.values() if y is not None]),
        "mod": ridge,
        "df_int": intercept
    }

    return mod_out

# def model_refit(x_train, y_train, x_val, y_val, x_test, y_test, lambda_, lower_limits, upper_limits, intercept=True, intercept_sign="non_negative", penalty_factor=None, *args, **kwargs):

#     ridge = Ridge(alpha=lambda_, fit_intercept=intercept, random_state=42)
#     ridge.fit(x_train, y_train)

#     if ridge.fit_intercept:
#         coefs = np.append(ridge.intercept_, ridge.coef_)
#         feature_names = ['Intercept'] + list(x_train.columns)
#     else:
#         coefs = ridge.coef_
#         feature_names = x_train.columns

#     coefs_df = pd.DataFrame(coefs, index=feature_names, columns=['s0'])

#     def get_adjusted_r2(y_true, y_pred, n_features, n_samples):
#         r2 = r2_score(y_true, y_pred)
#         adjusted_r2 = 1 - (1-r2)*(n_samples-1)/(n_samples-n_features-1)
#         return adjusted_r2

#     def get_nrmse(y_true, y_pred):
#         y_true = np.array(y_true)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         nrmse = rmse / (y_true.max() - y_true.min())
#         return nrmse

#     if 'intercept_sign' in locals() or 'intercept_sign' in globals():
#         if intercept_sign == "non_negative" and ridge.intercept_ < 0:
#             ridge_no_intercept = Ridge(alpha=0.1, fit_intercept=False, random_state=42)
#             ridge_no_intercept.fit(x_train, y_train)
#             ridge = ridge_no_intercept

#     y_train_pred = ridge.predict(x_train)
#     n_samples, n_features = x_train.shape
#     adjusted_r2_train = get_adjusted_r2(y_train, y_train_pred, n_features, n_samples)
#     nrmse_train = get_nrmse(y_train, y_train_pred)

#     if x_val is not None:
#         y_val_pred = ridge.predict(x_val)
#         adjusted_r2_val = get_adjusted_r2(y_val, y_val_pred, x_val.shape[1], len(y_val))
#         nrmse_val = get_nrmse(y_val, y_val_pred)
#     else:
#         adjusted_r2_val = nrmse_val = None

#     if x_test is not None:
#         y_test_pred = ridge.predict(x_test)
#         adjusted_r2_test = get_adjusted_r2(y_test, y_test_pred, x_test.shape[1], len(y_test))
#         nrmse_test = get_nrmse(y_test, y_test_pred)
#     else:
#         adjusted_r2_test = nrmse_test = None

#     if x_val is not None and x_test is not None:
#         y_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
#     elif x_val is not None:
#         y_pred = np.concatenate([y_train_pred, y_val_pred])
#     elif x_test is not None:
#         y_pred = np.concatenate([y_train_pred, y_test_pred])
#     else:
#         y_pred = y_train_pred

#     mod_out = {
#         "rsq_train": adjusted_r2_train,
#         "rsq_val": adjusted_r2_val,
#         "rsq_test": adjusted_r2_test,
#         "nrmse_train": nrmse_train,
#         "nrmse_val": nrmse_val,
#         "nrmse_test": nrmse_test,
#         #"coefs": lasso.coef_.reshape(-1, 1),
#         "coefs": coefs_df,
#         "y_train_pred": y_train_pred,
#         "y_val_pred": y_val_pred,
#         "y_test_pred": y_test_pred,
#         "y_pred": y_pred,
#         "mod": ridge,
#         "df_int": intercept
#     }

#     return mod_out


def get_rsq(true, predicted, p, df_int, n_train=None):
    sse = np.sum((true - predicted) ** 2)
    tss = np.sum((true - np.mean(true)) ** 2)

    if n_train is None:
        n = len(true)
    else:
        n = n_train

    rsq = 1 - (sse / (n - df_int - p)) / (tss / (n - 1))
    return rsq


def lambda_seq(x, y, seq_len=100, lambda_min_ratio=0.0001):
    def mysd(y):
        return np.sqrt(np.sum((y - np.mean(y)) ** 2) / len(y))

    # Standardize the features
    scaler = StandardScaler()
    sx = scaler.fit_transform(x)

    # Check for NaN columns and replace with zeros
    check_nan = np.all(np.isnan(sx), axis=0)
    for i, is_nan in enumerate(check_nan):
        if is_nan:
            sx[:, i] = 0

    # Standardize the target variable (assuming it's already centered)
    sy = y

    # Calculate lambda_max
    lambda_max = np.max(np.abs(np.sum(sx * sy, axis=0))) / (0.001 * x.shape[0])

    # Create a logarithmic sequence of lambdas
    lambda_max_log = np.log(lambda_max)
    log_step = (lambda_max_log - np.log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
    log_seq = np.linspace(lambda_max_log, np.log(lambda_max * lambda_min_ratio), num=seq_len)
    lambdas = np.exp(log_seq)

    return lambdas


def hyper_collector(InputCollect, hyper_in, ts_validation, add_penalty_factor, cores, dt_hyper_fixed=None):
    # Fetch hyper-parameters based on media
    hypParamSamName = hyper_names(adstock=InputCollect['adstock'], all_media=InputCollect['all_media'])

    # Manually add other hyper-parameters
    ##hypParamSamName.extend(HYPS_OTHERS)
    hypParamSamName += HYPS_OTHERS

    # Add penalty factor hyper-parameters names
    for_penalty = [col for col in InputCollect['dt_mod'].columns.values if col not in ['ds', 'dep_var']]
    if add_penalty_factor:
        ##for_penalty = InputCollect['dt_mod'].drop(columns=['ds', 'dep_var']).columns.tolist()
        penalty_names = ['penalty_' + name for name in for_penalty]
        hypParamSamName += penalty_names

    # Check hyper_fixed condition + add lambda + penalty factor hyper-parameters names
    all_fixed = check_hyper_fixed(InputCollect, dt_hyper_fixed, add_penalty_factor)
    hypParamSamName = all_fixed['hyp_param_sam_name']
    if not all_fixed['hyper_fixed']:
        hyper_bound_list = {}
        for param_name in hypParamSamName:
            if param_name in hyper_in.keys(): ## Added since R automatically creates an empty list don't raise an error
                hyper_bound_list[param_name] = hyper_in[param_name]
            else:
                hyper_bound_list[param_name] = list()

        # Add unfixed lambda hyperparameter manually
        ##if len(hyper_bound_list.get("lambda", [])) != 1:
        if len(hyper_bound_list["lambda"]) != 1:
            hyper_bound_list["lambda"] = [0, 1]

        # Add unfixed train_size hyperparameter manually
        if ts_validation:
            if "train_size" not in hyper_bound_list.keys():
                hyper_bound_list["train_size"] = [0.5, 0.8]
            print(f"Time-series validation with train_size range of {hyper_bound_list['train_size'][0]*100}% - {hyper_bound_list['train_size'][1]*100}% of the data...")
        else:
            if "train_size" in hyper_bound_list.keys():
                print("Warning: Provided train_size but ts_validation = FALSE. Time series validation inactive.")

            hyper_bound_list["train_size"] = [1]
            print("Fitting time series with all available data...")

        # Add unfixed penalty.factor hyperparameters manually
        ## for_penalty = InputCollect['dt_mod'].drop(columns=['ds', 'dep_var']).columns.tolist()
        penalty_names = [name + "_penalty" for name in for_penalty]
        if add_penalty_factor:
            for penalty in penalty_names:
                ##if len(hyper_bound_list.get(penalty, [])) != 1:
                if len(hyper_bound_list[penalty]) != 1:
                    hyper_bound_list[penalty] = [0, 1]

        # Get hyperparameters for Nevergrad
        ## hyper_bound_list_updated = {k: v for k, v in hyper_bound_list.items() if len(v) == 2}
        hyper_bound_list_updated = dict()
        for key, val in hyper_bound_list.items():
            if len(val) == 2:
                hyper_bound_list_updated[key] = val

        # Get fixed hyperparameters
        ##hyper_bound_list_fixed = {k: v for k, v in hyper_bound_list.items() if len(v) == 1}
        hyper_bound_list_fixed = dict()
        for key, val in hyper_bound_list.items():
            if len(val) == 1:
                hyper_bound_list_fixed[key] = val

        # Combine updated and fixed hyperparameters
        hyper_list_bind = {**hyper_bound_list_updated, **hyper_bound_list_fixed}
        hyper_list_all = dict() ##{}
        for param_name in hypParamSamName:
            if param_name in hyper_list_bind.keys():
                hyper_list_all[param_name] = hyper_list_bind[param_name]
            else:
                hyper_list_all[param_name] = []

        # Prepare a DataFrame for fixed hyperparameters
        ##dt_hyper_fixed_mod = pd.DataFrame({k: [v[0]] * cores for k, v in hyper_bound_list_fixed.items()})
        dt_hyper_fixed_mod = pd.DataFrame(hyper_bound_list_fixed.items())

    else:
        # Initialize hyper_bound_list_fixed
        hyper_bound_list_fixed = dict()
        for param_name in hypParamSamName:
            if param_name in dt_hyper_fixed.columns.values:
                hyper_bound_list_fixed[param_name] = dt_hyper_fixed[param_name].values.to_list()
            else:
                hyper_bound_list_fixed[param_name] = list()
            ##hyper_bound_list_fixed[param_name] = dt_hyper_fixed.get(param_name, [])

        # Update hyper_list_all and hyper_bound_list_updated
        hyper_list_all = hyper_bound_list_fixed ##.copy()
        ##hyper_bound_list_updated = {k: v for k, v in hyper_bound_list_fixed.items() if len(v) == 2}
        hyper_bound_list_updated = dict()
        for key, val in hyper_bound_list.items():
            if len(val) == 2:
                hyper_bound_list_updated[key] = val

        # Set cores to 1
        cores = 1

        # Prepare a DataFrame for fixed hyperparameters
        ## pd.DataFrame({k: v for k, v in hyper_bound_list_fixed.items()}, index=[0])
        dt_hyper_fixed_mod = dt_hyper_fixed_mod = pd.DataFrame(hyper_bound_list_fixed.items())

    return {
        "hyper_list_all": hyper_list_all,
        "hyper_bound_list_updated": hyper_bound_list_updated,
        "hyper_bound_list_fixed": hyper_bound_list_fixed,
        "dt_hyper_fixed_mod": dt_hyper_fixed_mod,
        "all_fixed": all_fixed
    }


def init_msgs_run(InputCollect, refresh, quiet=False, lambda_control=None):
    if lambda_control is not None:
        logging.info("Input 'lambda_control' deprecated in v3.6.0; lambda is now selected by hyperparameter optimization")

    if not quiet:
        # First message
        logging.info(f"Input data has {len(InputCollect['dt_mod'])} {InputCollect['intervalType']}s in total: {InputCollect['dt_mod']['ds'].min()} to {InputCollect['dt_mod']['ds'].max()}")

        # Calculate depth
        if 'refreshDepth' in InputCollect:
            depth = InputCollect['refreshDepth']
        elif 'refreshCounter' in InputCollect:
            depth = InputCollect['refreshCounter']
        else:
            depth = 0

        # Update refresh
        refresh = int(depth) > 0

        # Second message
        model_type = "Initial" if not refresh else f"Refresh #{depth}"
        logging.info(f"{model_type} model is built on rolling window of {InputCollect['rollingWindowLength']} {InputCollect['intervalType']}: {InputCollect['window_start']} to {InputCollect['window_end']}")

    if refresh:
        logging.info(f"Rolling window moving forward: {InputCollect['refresh_steps']} {InputCollect['intervalType']}s")
