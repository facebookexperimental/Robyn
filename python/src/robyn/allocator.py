# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import pandas as pd
import numpy as np
import itertools
import nlopt
import os
from functools import partial
from itertools import chain

from .checks import check_allocator, check_allocator_constrains, check_metric_dates, check_daterange
from .response import robyn_response, which_usecase
from .plots import allocation_plots

ROBYN_TEMP = None

def robyn_allocator(robyn_object=None,
                    select_build=0,
                    InputCollect=None,
                    OutputCollect=None,
                    select_model=None,
                    json_file=None,
                    scenario="max_response",
                    total_budget=None,
                    target_value=None,
                    date_range=None,
                    channel_constr_low=None,
                    channel_constr_up=None,
                    channel_constr_multiplier=3,
                    optim_algo="SLSQP_AUGLAG",
                    maxeval=100000,
                    constr_mode="eq",
                    plots=True,
                    plot_folder=None,
                    plot_folder_sub=None,
                    export=True,
                    quiet=False,
                    ui=False,
                    **kwargs):
    """
    Allocates budget for a given model using the Robyn framework.

    Parameters:
    - robyn_object (object): The Robyn object containing the model.
    - select_build (int): The build number of the model.
    - InputCollect (object): The input collection object.
    - OutputCollect (object): The output collection object.
    - select_model (object): The selected model.
    - json_file (str): The path to the JSON file containing the exported model.
    - scenario (str): The scenario for the budget allocation.
    - total_budget (float): The total budget for the allocation.
    - target_value (float): The target value for the allocation.
    - date_range (str): The date range for the allocation.
    - channel_constr_low (float): The lower constraint for the channels.
    - channel_constr_up (float): The upper constraint for the channels.
    - channel_constr_multiplier (int): The multiplier for the channel constraints.
    - optim_algo (str): The optimization algorithm to use.
    - maxeval (int): The maximum number of evaluations for the optimization.
    - constr_mode (str): The constraint mode for the optimization.
    - plots (bool): Whether to generate plots.
    - plot_folder (str): The folder to save the plots.
    - plot_folder_sub (str): The subfolder to save the plots.
    - export (bool): Whether to export the results.
    - quiet (bool): Whether to suppress the output.
    - ui (bool): Whether to use the user interface.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    # Use previously exported model using json_file
    if not json_file is None:
        if InputCollect is None:
            InputCollect = robyn_inputs(json_file=json_file, quiet=True, **kwargs)
        if OutputCollect is None:
            if plot_folder is None:
                json = robyn_read(json_file, step=2, quiet=True)
                plot_folder = dirname(json.ExportedModel.plot_folder)
                if not plot_folder_sub is None:
                    plot_folder_sub = None
            OutputCollect = robyn_run(json_file=json_file, export=export, plot_folder=plot_folder, plot_folder_sub=plot_folder_sub, **kwargs)
        if select_model is None:
            select_model = OutputCollect.allSolutions

    # Collect inputs
    # if not robyn_object is None and (InputCollect is None or OutputCollect is None or select_model is None):
    #     if "robyn_exported" in robyn_object.__class__.__name__:
    #         imported = robyn_object
    #         robyn_object = imported.robyn_object
    #     else:
    #         imported = robyn_load(robyn_object, select_build, quiet=True)
    #     InputCollect = imported.InputCollect
    #     OutputCollect = imported.OutputCollect
    #     select_model = imported.select_model
    # else:
    #     if select_model is None and len(OutputCollect.allSolutions) == 1:
    #         select_model = OutputCollect.allSolutions
    #     if any(InputCollect is None, OutputCollect is None, select_model is None):
    #         raise ValueError("When 'robyn_object' is not provided, then InputCollect, OutputCollect, select_model must be provided")

    if select_model is None and len(OutputCollect['allSolutions']) == 1:
        select_model = OutputCollect['allSolutions']

    # Check if any of InputCollect, OutputCollect, or select_model is None
    if InputCollect is None or OutputCollect is None or select_model is None:
        raise ValueError("When 'robyn_object' is not provided, then InputCollect, OutputCollect, and select_model must be provided")

    # Check inputs and parameters
    if len(InputCollect["robyn_inputs"]["paid_media_spends"]) <= 1:
        raise ValueError("Must have a valid model with at least two 'paid_media_spends'")

    if not quiet:
        print(f">>> Running budget allocator for model ID {select_model}...")

    # Set local data & params values
    paid_media_spends = InputCollect["robyn_inputs"]["paid_media_spends"]
    media_order = pd.Series(paid_media_spends).sort_values().index
    media_order_list = media_order.tolist()
    mediaSpendSorted = [paid_media_spends[i] for i in media_order_list]
    # mediaSpendSorted = paid_media_spends[media_order]
    dep_var_type = InputCollect["robyn_inputs"]["dep_var_type"]
    if channel_constr_low is None:
        channel_constr_low = 0.5 if scenario == "max_response" else 0.1
    if channel_constr_up is None:
        channel_constr_up = 2 if scenario == "max_response" else np.inf

    if isinstance(channel_constr_low, list) and len(channel_constr_low) == 1:
        channel_constr_low = pd.Series([channel_constr_low[0]] * len(paid_media_spends))
    elif not isinstance(channel_constr_low, list):
        channel_constr_low = pd.Series([channel_constr_low] * len(paid_media_spends))
    else:
        channel_constr_low = pd.Series(channel_constr_low)

    if isinstance(channel_constr_up, list) and len(channel_constr_up) == 1:
        channel_constr_up = pd.Series([channel_constr_up[0]] * len(paid_media_spends))
    elif not isinstance(channel_constr_up, list):
        channel_constr_up = pd.Series([channel_constr_up] * len(paid_media_spends))
    else:
        channel_constr_up = pd.Series(channel_constr_up)

    check_allocator_constrains(channel_constr_low, channel_constr_up)

    # channel_constr_low = pd.Series(channel_constr_low, index=paid_media_spends)
    channel_constr_low.index = paid_media_spends
    channel_constr_low = channel_constr_low.iloc[media_order]
    # channel_constr_up = pd.Series(channel_constr_up, index=paid_media_spends)
    channel_constr_up.index = paid_media_spends
    channel_constr_up = channel_constr_up.iloc[media_order]

    # channel_constr_low.index = paid_media_spends
    # channel_constr_up.index = paid_media_spends
    # channel_constr_low = channel_constr_low[media_order]
    # channel_constr_up = channel_constr_up[media_order]
    dt_hyppar = OutputCollect["resultHypParam"][OutputCollect["resultHypParam"]["solID"] == select_model]
    # dt_bestCoef = OutputCollect["xDecompAgg"][OutputCollect["xDecompAgg"]["solID"] == select_model][OutputCollect["xDecompAgg"].rn.isin(paid_media_spends)]
    dt_bestCoef = OutputCollect['xDecompAgg'][
        (OutputCollect['xDecompAgg']['solID'] == select_model) &
        (OutputCollect['xDecompAgg']['rn'].isin(paid_media_spends))
    ]

    # Check inputs and parameters
    scenario = check_allocator(OutputCollect, select_model, paid_media_spends, scenario, channel_constr_low, channel_constr_up, constr_mode)

    # Sort media
    dt_coef = dt_bestCoef[['rn', 'coefs']].copy()
    get_rn_order = np.argsort(dt_bestCoef['rn'].values)
    dt_coefSorted = dt_coef.iloc[get_rn_order].copy()
    dt_bestCoef.iloc[:] = dt_bestCoef.iloc[get_rn_order]
    coefSelectorSorted = (dt_coefSorted["coefs"] > 0)
    coefSelectorSorted.index = dt_coefSorted["rn"]

    from .checks import hyper_names

    # dt_hyppar = InputCollect.select(hyper_names(InputCollect["robyn_inputs"]["adstock"], mediaSpendSorted))
    # dt_hyppar = dt_hyppar.select(sort(dt_hyppar.columns))
    selected_columns = hyper_names(InputCollect["robyn_inputs"]["adstock"], all_media=mediaSpendSorted)

    # Clean up duplicated columns
    for col in dt_hyppar.columns:
        if col.endswith('_x'):
            base_name = col[:-2]  # remove the last two characters '_x'
            col_y = base_name + '_y'

            if col_y in dt_hyppar.columns:
                dt_hyppar[base_name] = dt_hyppar[col_y].combine_first(dt_hyppar[col])
                dt_hyppar.drop([col, col_y], axis=1, inplace=True)

    dt_hyppar = dt_hyppar[selected_columns]
    dt_hyppar = dt_hyppar[sorted(dt_hyppar.columns)]

    dt_bestCoef = dt_bestCoef.drop_duplicates(subset='rn', keep='first')
    dt_bestCoef = dt_bestCoef[dt_bestCoef['rn'].isin(mediaSpendSorted)]

    channelConstrLowSorted = channel_constr_low[mediaSpendSorted]
    channelConstrUpSorted = channel_constr_up[mediaSpendSorted]

    hills = get_hill_params(InputCollect, OutputCollect, dt_hyppar, dt_coef, mediaSpendSorted, select_model)
    alphas = hills["alphas"].reset_index()
    inflexions = hills["inflexions"].reset_index()
    coefs_sorted = hills["coefs_sorted"]

    start = InputCollect["robyn_inputs"]["rollingWindowStartWhich"]
    end = InputCollect["robyn_inputs"]["rollingWindowEndWhich"]
    window_loc = range(start, end + 1)

    dt_optimCost = InputCollect["robyn_inputs"]['dt_mod'].loc[window_loc]
    new_date_range = check_metric_dates(date_range, dt_optimCost.ds, InputCollect["robyn_inputs"]["dayInterval"], quiet=False, is_allocator=True)
    date_min = new_date_range["date_range_updated"][0]
    date_max = new_date_range["date_range_updated"][-1]
    # check_daterange(date_min, date_max, dt_optimCost["ds"])
    if pd.isna(date_min):
        date_min = dt_optimCost.ds.min()
    if pd.isna(date_max):
        date_max = dt_optimCost.ds.max()
    if date_min < dt_optimCost.ds.min():
        date_min = dt_optimCost.ds.min()
    if date_max > dt_optimCost.ds.max():
        date_max = dt_optimCost.ds.max()
    histFiltered = dt_optimCost.loc[dt_optimCost.ds.between(date_min, date_max), ]

    histSpendAll = dt_optimCost[mediaSpendSorted].sum()
    histSpendAllTotal = histSpendAll.sum()
    histSpendAllUnit = dt_optimCost[mediaSpendSorted].mean()
    histSpendAllUnitTotal = histSpendAllUnit.sum()
    histSpendAllShare = histSpendAllUnit / histSpendAllUnitTotal

    histSpendWindow = histFiltered[mediaSpendSorted].sum()
    histSpendWindowTotal = histSpendWindow.sum()
    initSpendUnit = histSpendWindowUnit = histFiltered[mediaSpendSorted].mean()
    histSpendWindowUnitTotal = initSpendUnit.sum()
    histSpendWindowShare = initSpendUnit / histSpendWindowUnitTotal

    simulation_period = initial_mean_period = [len(histFiltered[x]) for x in mediaSpendSorted]
    # nDates = {x: histFiltered.ds for x in mediaSpendSorted}
    nDates = {x: histFiltered.ds.tolist() for x in mediaSpendSorted}
    if not quiet:
        unique_mean_period = list(set(initial_mean_period))[0]
        print(f"Date Window: {date_min}:{date_max} ({unique_mean_period} {InputCollect['robyn_inputs']['intervalType']}s)")

    zero_spend_channel = [x for x in mediaSpendSorted if histSpendWindow[x] == 0]

    initSpendUnitTotal = initSpendUnit.sum()
    initSpendShare = initSpendUnit / initSpendUnitTotal
    unique_period = np.unique(simulation_period)[0]
    # total_budget_unit = total_budget / unique(simulation_period) if pd.isna(total_budget) else total_budget / unique(simulation_period)
    if pd.isna(total_budget):
        total_budget_unit = initSpendUnitTotal
    else:
        total_budget_unit = total_budget / unique_period
    total_budget_window = total_budget_unit * unique_period

    # Get use case based on inputs
    usecase = which_usecase(initSpendUnit[0], date_range)
    if usecase == "all_historical_vec":
        ndates_loc = np.where(InputCollect["robyn_inputs"]["dt_mod"].ds.isin(histFiltered.ds))[0]
    else:
        ndates_loc = np.arange(len(histFiltered.ds))
    usecase = f"{usecase}+ defined_budget" if not pd.isna(total_budget) else f"{usecase}+ historical_budget"

    # Response values based on date range -> mean spend
    initResponseUnit = []
    initResponseMargUnit = []
    hist_carryover = []
    qa_carryover = []
    for i in range(len(mediaSpendSorted)):
        resp = robyn_response(
            json_file=json_file,
            select_build=select_build,
            select_model=select_model,
            metric_name=mediaSpendSorted[i],
            dt_hyppar=OutputCollect["resultHypParam"],
            dt_coef=OutputCollect["xDecompAgg"],
            InputCollect=InputCollect,
            OutputCollect=OutputCollect,
            quiet=True
        )
        window_loc = range(window_loc.start - 1, window_loc.stop - 1)
        hist_carryover_temp = resp["input_carryover"][window_loc]
        qa_carryover.append(round(resp["input_total"][window_loc]))
        hist_carryover_temp.index = resp["date"][window_loc]
        hist_carryover.append(hist_carryover_temp)
        x_input = initSpendUnit[i]
        resp_simulate = fx_objective(
            x=x_input,
            coeff=coefs_sorted[mediaSpendSorted[i]],
            alpha=alphas[f"{mediaSpendSorted[i]}_alphas"][0],
            inflexion=inflexions[f"{mediaSpendSorted[i]}_gammas"][0],
            x_hist_carryover=np.mean(hist_carryover_temp),
            get_sum=False
        )
        resp_simulate_plus1 = fx_objective(
            x=x_input + 1,
            coeff=coefs_sorted[mediaSpendSorted[i]],
            alpha=alphas[f"{mediaSpendSorted[i]}_alphas"][0],
            inflexion=inflexions[f"{mediaSpendSorted[i]}_gammas"][0],
            x_hist_carryover=np.mean(hist_carryover_temp),
            get_sum=False
        )
        initResponseUnit = np.append(initResponseUnit, resp_simulate)
        initResponseMargUnit = np.append(initResponseMargUnit, resp_simulate_plus1 - resp_simulate)

    qa_carryover = pd.concat(qa_carryover, axis=1)
    qa_carryover = qa_carryover.fillna(0)
    # Assign names to the columns of qa_carryover
    qa_carryover.columns = mediaSpendSorted

    initResponseUnit = pd.DataFrame([initResponseUnit], columns=mediaSpendSorted)

    hist_carryover = pd.concat(hist_carryover, axis=1)
    hist_carryover = hist_carryover.fillna(0)
    # Assign names to the columns of hist_carryover
    hist_carryover.columns = mediaSpendSorted

    # QA adstock: simulated adstock should be identical to model adstock
    # qa_carryover_origin = OutputCollect$mediaVecCollect[
    #   .data$solID == select_model & .data$type == "adstockedMedia",
    #   "mediaSpendSorted"
    # ]
    # qa_carryover == qa_carryover_origin

    if len(zero_spend_channel) > 0 and not quiet:
        print("Media variables with 0 spending during date range:", zero_spend_channel)
        # hist_carryover[zero_spend_channel] = 0

    channelConstrLowSortedExt = np.where(
        1 - (1 - channelConstrLowSorted) * channel_constr_multiplier < 0,
        0, 1 - (1 - channelConstrLowSorted) * channel_constr_multiplier
    )
    channelConstrUpSortedExt = np.where(
        1 + (channelConstrUpSorted - 1) * channel_constr_multiplier < 0,
        channelConstrUpSorted * channel_constr_multiplier,
        1 + (channelConstrUpSorted - 1) * channel_constr_multiplier
    )

    target_value_ext = target_value
    if scenario == "target_efficiency":
        channelConstrLowSortedExt = channelConstrLowSorted
        channelConstrUpSortedExt = channelConstrUpSorted
        if dep_var_type == "conversion":
            if target_value is None:
                target_value = initResponseUnit.sum().sum() / initSpendUnit.sum().sum() * 1.2
            target_value_ext = target_value * 1.5
        else:
            if target_value is None:
                target_value = initResponseUnit.sum().sum() / initSpendUnit.sum().sum() * 0.8
            target_value_ext = 1

    temp_init = temp_init_all = initSpendUnit
    if len(zero_spend_channel) > 0:
        temp_init_all[zero_spend_channel] = histSpendAllUnit[zero_spend_channel]

    temp_ub = temp_ub_all = channelConstrUpSorted
    temp_lb = temp_lb_all = channelConstrLowSorted
    temp_ub_ext = temp_ub_ext_all = channelConstrUpSortedExt
    temp_lb_ext = temp_lb_ext_all = channelConstrLowSortedExt

    x0 = x0_all = lb = lb_all = temp_init_all * temp_lb_all
    ub = ub_all = temp_init_all * temp_ub_all
    x0_ext = x0_ext_all = lb_ext = lb_ext_all = temp_init_all * temp_lb_ext_all
    ub_ext = ub_ext_all = temp_init_all * temp_ub_ext_all

    skip_these = (channel_constr_low == 0) & (channel_constr_up == 0)
    zero_constraint_channel = [channel for channel in mediaSpendSorted if skip_these[channel]]
    if any(skip_these) and not quiet:
        print("Excluded variables (constrained to 0):", zero_constraint_channel)
    zero_coef_channel = []
    if not all(coefSelectorSorted):
        for index, value in coefSelectorSorted.items():
            if not value:  # Check if the value is False
                zero_coef_channel.append(index)
        if not quiet:
            print("Excluded variables (coefficients are 0):", zero_coef_channel)

    channel_to_drop_loc = [channel in (zero_coef_channel + zero_constraint_channel) for channel in mediaSpendSorted]
    channel_for_allocation = [mediaSpendSorted[i] for i in range(len(mediaSpendSorted)) if not channel_to_drop_loc[i]]
    if any(channel_to_drop_loc):

        temp_init = temp_init_all.loc[channel_for_allocation]
        temp_ub = temp_ub_all.loc[channel_for_allocation]
        temp_lb = temp_lb_all.loc[channel_for_allocation]
        x0 = x0_all.loc[channel_for_allocation]
        lb = lb_all.loc[channel_for_allocation]
        ub = ub_all.loc[channel_for_allocation]

        channel_indices = [temp_init_all.index.get_loc(c) for c in channel_for_allocation]
        temp_ub_ext = temp_ub_ext_all[channel_indices]
        temp_lb_ext = temp_lb_ext_all[channel_indices]
        x0_ext = x0_ext_all[channel_indices]
        lb_ext = lb_ext_all[channel_indices]
        ub_ext = ub_ext_all[channel_indices]

    x0 = lb = temp_init * temp_lb
    ub = temp_init * temp_ub
    x0_ext = lb_ext = temp_init * temp_lb_ext
    ub_ext = temp_init * temp_ub_ext

    coefs_eval = coefs_sorted[channel_for_allocation]
    alphas_keys = [f"{channel}_alphas" for channel in channel_for_allocation]
    alphas_eval = {key: alphas[key].iloc[0] for key in alphas_keys}
    gammas_keys = [f"{channel}_gammas" for channel in channel_for_allocation]
    inflexions_eval = {key: inflexions[key].iloc[0] for key in gammas_keys}

    hist_carryover_eval = hist_carryover[channel_for_allocation]

    column_means = hist_carryover_eval.mean(axis=0)
    x_hist_carryover = column_means.values
    # hist_carryover_eval = hist_carryover[channel_for_allocation].iloc[0].values

    eval_list = {
        "coefs_eval": coefs_eval,
        "alphas_eval": alphas_eval,
        "inflexions_eval": inflexions_eval,
        "total_budget": total_budget,
        "total_budget_unit": total_budget_unit,
        "hist_carryover_eval": hist_carryover_eval,
        "target_value": target_value,
        "target_value_ext": target_value_ext,
        "dep_var_type": dep_var_type
    }

    # So we can implicitly use these values within eval_f()
    global ROBYN_TEMP
    ROBYN_TEMP = eval_list

    # Set optim options
    if optim_algo == "MMA_AUGLAG":
        local_optimizer = nlopt.LD_MMA
    else:
        local_optimizer = nlopt.LD_SLSQP

    x0_list = []
    x0_ext_list = []
    lb_list = []
    ub_list = []
    lb_ext_list = []
    ub_ext_list = []

    channels_list = [item for item in mediaSpendSorted if item not in zero_spend_channel]
    for channel in channels_list:
        x0_list.append(x0.loc[channel])
        x0_ext_list.append(x0_ext.loc[channel])
        lb_list.append(lb.loc[channel])
        ub_list.append(ub.loc[channel])
        lb_ext_list.append(lb_ext.loc[channel])
        ub_ext_list.append(ub_ext.loc[channel])

    # Run optim
    if scenario == "max_response":


        ###
        ## nlsMod
        ###
        nlsMod_opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0_list))  # Use the Augmented Lagrangian algorithm
        nlsMod_opt.set_lower_bounds(lb_list)
        nlsMod_opt.set_upper_bounds(ub_list)
        nlsMod_opt.set_min_objective(eval_f)
        nlsMod_opt.set_xtol_rel(1e-10)
        nlsMod_opt.set_maxeval(maxeval)

        if constr_mode == "eq":
            nlsMod_opt.add_equality_constraint(eval_g_eq, 1e-8)
        elif constr_mode == "ineq":
            nlsMod_opt.add_inequality_constraint(eval_g_ineq, 1e-8)

        # Local optimizer options (optional)
        nlsMod_local_opt = nlopt.opt(local_optimizer, len(x0_list))
        nlsMod_local_opt.set_xtol_rel(1e-10)
        nlsMod_opt.set_local_optimizer(nlsMod_local_opt)

        # Perform the optimization
        optmSpendUnit = nlsMod_opt.optimize(x0_list)
        nlsMod_min_f = nlsMod_opt.last_optimum_value()
        optmResponseUnit = calculate_channels(optmSpendUnit)

        ###
        ## nlsModUnbound
        ###
        nlsModUnbound_opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0_ext_list))  # Use the Augmented Lagrangian algorithm
        nlsModUnbound_opt.set_lower_bounds(lb_ext_list)
        nlsModUnbound_opt.set_upper_bounds(ub_ext_list)
        nlsModUnbound_opt.set_min_objective(eval_f)
        nlsModUnbound_opt.set_xtol_rel(1e-10)
        nlsModUnbound_opt.set_maxeval(maxeval)

        if constr_mode == "eq":
            nlsModUnbound_opt.add_equality_constraint(eval_g_eq, 1e-8)
        elif constr_mode == "ineq":
            nlsModUnbound_opt.add_inequality_constraint(eval_g_ineq, 1e-8)

        # Local optimizer options (optional)
        nlsModUnbound_local_opt = nlopt.opt(local_optimizer, len(x0_ext_list))
        nlsModUnbound_local_opt.set_xtol_rel(1e-10)
        nlsModUnbound_opt.set_local_optimizer(nlsModUnbound_local_opt)

        # Perform the optimization
        optmSpendUnitUnbound = nlsModUnbound_opt.optimize(x0_ext_list)
        nlsModUnbound_min_f = nlsModUnbound_opt.last_optimum_value()
        optmResponseUnitUnbound = calculate_channels(optmSpendUnitUnbound)

    # TODO debug else statement following above if structure
    elif scenario == "target_efficiency":

        total_response = OutputCollect["xDecompAgg"]["xDecompAgg"].sum()

        ###
        ## nlsMod
        ###
        nlsMod_opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0_list))
        nlsMod_opt.set_lower_bounds(lb_list)
        nlsMod_opt.set_upper_bounds([total_response] * len(x0_list))
        nlsMod_opt.set_min_objective(eval_f)
        nlsMod_opt.set_xtol_rel(1e-10)
        nlsMod_opt.set_maxeval(maxeval)

        wrapper_function = wrapper_eval_g_eq_effi(target_value)

        if constr_mode == "eq":
            nlsMod_opt.add_equality_constraint(wrapper_function, 1e-8)
        elif constr_mode == "ineq":
            nlsMod_opt.add_inequality_constraint(wrapper_function, 1e-8)

        nlsMod_local_opt = nlopt.opt(local_optimizer, len(x0_list))
        nlsMod_local_opt.set_xtol_rel(1e-10)
        nlsMod_opt.set_local_optimizer(nlsMod_local_opt)

        optmSpendUnit = nlsMod_opt.optimize(x0_list)
        nlsMod_min_f = nlsMod_opt.last_optimum_value()
        optmResponseUnit = calculate_channels(optmSpendUnit)
        # bounded optimisation
        # total_response = sum(OutputCollect.xDecompAgg.xDecompAgg)
        # nlsMod = nlopt.nlopt(
        #     x0=x0,
        #     f=eval_f,
        #     f_eq=eval_g_eq_effi if constr_mode == "eq" else None,
        #     f_ieq=eval_g_eq_effi if constr_mode == "ineq" else None,
        #     lb=lb,
        #     ub=total_response * [1] * len(ub),
        #     opts=[
        #         "algorithm", "NLOPT_LD_AUGLAG",
        #         "xtol_rel", 1.0e-10,
        #         "maxeval", maxeval,
        #         "local_opts", local_opts
        #     ],
        #     target_value=target_value
        # )
        # unbounded optimisation

        ###
        ## nlsModUnbound
        ###
        nlsModUnbound_opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0_list))
        nlsModUnbound_opt.set_lower_bounds(lb_list)
        nlsModUnbound_opt.set_upper_bounds([total_response] * len(x0_list))
        nlsModUnbound_opt.set_min_objective(eval_f)
        nlsModUnbound_opt.set_xtol_rel(1e-10)
        nlsModUnbound_opt.set_maxeval(maxeval)

        wrapper_function = wrapper_eval_g_eq_effi(target_value_ext)

        if constr_mode == "eq":
            nlsModUnbound_opt.add_equality_constraint(wrapper_function, 1e-8)
        elif constr_mode == "ineq":
            nlsModUnbound_opt.add_inequality_constraint(wrapper_function, 1e-8)

        nlsMod_local_opt = nlopt.opt(local_optimizer, len(x0_list))
        nlsMod_local_opt.set_xtol_rel(1e-10)
        nlsModUnbound_opt.set_local_optimizer(nlsMod_local_opt)

        optmSpendUnitUnbound = nlsModUnbound_opt.optimize(x0_list)
        nlsModUnbound_min_f = nlsModUnbound_opt.last_optimum_value()
        optmResponseUnitUnbound = calculate_channels(optmSpendUnit)

        # nlsModUnbound = nlopt.nlopt(
        #     x0=x0,
        #     f=eval_f,
        #     f_eq=eval_g_eq_effi if constr_mode == "eq" else None,
        #     f_ieq=eval_g_eq_effi if constr_mode == "ineq" else None,
        #     lb=lb,
        #     ub=total_response * [1] * len(ub),
        #     opts=[
        #         "algorithm", "NLOPT_LD_AUGLAG",
        #         "xtol_rel", 1.0e-10,
        #         "maxeval", maxeval,
        #         "local_opts", local_opts
        #     ],
        #     target_value=target_value_ext
        # )


    optmResponseMargUnit = np.array(list(map(
        lambda x, coeff, alpha, inflexion, x_hist_carryover: fx_objective(x, coeff, alpha, inflexion, x_hist_carryover),
        optmSpendUnit + 1, coefs_eval, alphas_eval.values(), inflexions_eval.values(), x_hist_carryover
    ))) - optmResponseUnit

    optmResponseMargUnitUnbound = np.array(list(map(
        lambda x, coeff, alpha, inflexion, x_hist_carryover: fx_objective(x, coeff, alpha, inflexion, x_hist_carryover),
        optmSpendUnitUnbound + 1, coefs_eval, alphas_eval.values(), inflexions_eval.values(), x_hist_carryover
    ))) - optmResponseUnitUnbound

    # Collect the output
    names = [channel_for_allocation[i] for i in range(len(channel_for_allocation))]
    mediaSpendSorted = names

    optmSpendUnitOut = np.zeros(len(channel_for_allocation))
    optmResponseUnitOut = np.zeros(len(channel_for_allocation))
    optmResponseMargUnitOut = np.zeros(len(channel_for_allocation))
    optmSpendUnitUnboundOut = np.zeros(len(channel_for_allocation))
    optmResponseUnitUnboundOut = np.zeros(len(channel_for_allocation))
    optmResponseMargUnitUnboundOut = np.zeros(len(channel_for_allocation))

    arrays_to_zero = [optmSpendUnitOut, optmResponseUnitOut, optmResponseMargUnitOut,
                  optmSpendUnitUnboundOut, optmResponseUnitUnboundOut, optmResponseMargUnitUnboundOut]
    for array in arrays_to_zero:
        array[channel_to_drop_loc] = 0

    channel_to_drop_loc_temp = np.array(channel_to_drop_loc, dtype=bool)
    # Apply non-dropped channel values
    optmSpendUnitOut[~channel_to_drop_loc_temp] = optmSpendUnit

    optmResponseUnitOut[~channel_to_drop_loc_temp] = optmResponseUnit
    optmResponseMargUnitOut[~channel_to_drop_loc_temp] = optmResponseMargUnit
    optmSpendUnitUnboundOut[~channel_to_drop_loc_temp] = optmSpendUnitUnbound
    optmResponseUnitUnboundOut[~channel_to_drop_loc_temp] = optmResponseUnitUnbound
    optmResponseMargUnitUnboundOut[~channel_to_drop_loc_temp] = optmResponseMargUnitUnbound

    optmResponseUnitTotal = initResponseUnit.sum(axis=1).values[0]
    initResponseUnitTotal_array = np.full(initResponseUnit.shape[1], optmResponseUnitTotal)

    sum_initResponseUnit = np.sum(initResponseUnit)
    unique_simulation_period = np.unique(simulation_period)
    initResponseTotal = sum_initResponseUnit * unique_simulation_period
    initResponseUnitShare = initResponseUnit / sum_initResponseUnit

    sum_initResponseUnit = np.sum(initResponseUnit)
    periods_list = ["{} {}".format(period, InputCollect['robyn_inputs']['intervalType']) for period in initial_mean_period]

    dt_optimOut = {
        'solID':select_model,
        'dep_var_type':dep_var_type,
        'channels':mediaSpendSorted,
        'date_min':date_min,
        'date_max':date_max,
        'periods':periods_list,
        'constr_low':temp_lb_all,
        'constr_low_abs':lb_all,
        'constr_up':temp_ub_all,
        'constr_up_abs':ub_all,
        'unconstr_mult':channel_constr_multiplier,
        'constr_low_unb':temp_lb_ext_all,
        'constr_low_unb_abs':lb_ext_all,
        'constr_up_unb':temp_ub_ext_all,
        'constr_up_unb_abs':ub_ext_all,
        # Historical spends
        'histSpendAll':histSpendAll,
        'histSpendAllTotal':histSpendAllTotal,
        'histSpendAllUnit':histSpendAllUnit,
        'histSpendAllUnitTotal':histSpendAllUnitTotal,
        'histSpendAllShare':histSpendAllShare,
        'histSpendWindow':histSpendWindow,
        'histSpendWindowTotal':histSpendWindowTotal,
        'histSpendWindowUnit':histSpendWindowUnit,
        'histSpendWindowUnitTotal':histSpendWindowUnitTotal,
        'histSpendWindowShare':histSpendWindowShare,
        # Initial spends for allocation
        'initSpendUnit':initSpendUnit,
        'initSpendUnitTotal':initSpendUnitTotal,
        'initSpendShare':initSpendShare,
        'initSpendTotal':initSpendUnitTotal * np.unique(simulation_period),
        # initSpendUnitRaw=histSpendUnitRaw,
        # adstocked=adstocked,
        # adstocked_start_date=as.Date(ifelse(adstocked, head(resp$date, 1), NA), origin="1970-01-01"),
        # adstocked_end_date=as.Date(ifelse(adstocked, tail(resp$date, 1), NA), origin="1970-01-01"),
        # adstocked_periods=length(resp$date),
        'initResponseUnit':initResponseUnit,
        'initResponseUnitTotal':initResponseUnitTotal_array,
        'initResponseMargUnit':initResponseMargUnit,
        'initResponseTotal':initResponseTotal,
        'initResponseUnitShare':initResponseUnitShare,
        'initRoiUnit':initResponseUnit / initSpendUnit,
        'initCpaUnit':initSpendUnit / initResponseUnit,
        # Budget change
        'total_budget_unit':total_budget_unit,
        'total_budget_unit_delta':total_budget_unit / initSpendUnitTotal - 1,
        # Optimized
        'optmSpendUnit':optmSpendUnitOut,
        'optmSpendUnitDelta':(optmSpendUnitOut / initSpendUnit - 1),
        'optmSpendUnitTotal':sum(optmSpendUnitOut),
        'optmSpendUnitTotalDelta':sum(optmSpendUnitOut) / initSpendUnitTotal - 1,
        'optmSpendShareUnit':optmSpendUnitOut / sum(optmSpendUnitOut),
        'optmSpendTotal':sum(optmSpendUnitOut) * unique_simulation_period,
        'optmSpendUnitUnbound':optmSpendUnitUnboundOut,
        'optmSpendUnitDeltaUnbound':(optmSpendUnitUnboundOut / optmSpendUnit - 1),
        'optmSpendUnitTotalUnbound':sum(optmSpendUnitUnboundOut),
        'optmSpendUnitTotalDeltaUnbound':sum(optmSpendUnitUnboundOut) / initSpendUnitTotal - 1,
        'optmSpendShareUnitUnbound':optmSpendUnitUnboundOut / sum(optmSpendUnitUnboundOut),
        'optmSpendTotalUnbound':sum(optmSpendUnitUnboundOut) * unique_simulation_period,
        'optmResponseUnit':optmResponseUnitOut,
        'optmResponseMargUnit':optmResponseMargUnitOut,
        'optmResponseUnitTotal':sum(optmResponseUnitOut),
        'optmResponseTotal':sum(optmResponseUnitOut) * unique_simulation_period,
        'optmResponseUnitShare':optmResponseUnitOut / sum(optmResponseUnitOut),
        'optmRoiUnit':optmResponseUnitOut / optmSpendUnitOut,
        'optmCpaUnit':optmSpendUnitOut / optmResponseUnitOut,
        'optmResponseUnitLift':(optmResponseUnitOut / initResponseUnit) - 1,
        'optmResponseUnitUnbound':optmResponseUnitUnboundOut,
        'optmResponseMargUnitUnbound':optmResponseMargUnitUnboundOut,
        'optmResponseUnitTotalUnbound':sum(optmResponseUnitUnboundOut),
        'optmResponseTotalUnbound':sum(optmResponseUnitUnboundOut) * unique_simulation_period,
        'optmResponseUnitShareUnbound':optmResponseUnitUnboundOut / sum(optmResponseUnitUnboundOut),
        'optmRoiUnitUnbound':optmResponseUnitUnboundOut / optmSpendUnitUnboundOut,
        'optmCpaUnitUnbound':optmSpendUnitUnboundOut / optmResponseUnitUnboundOut,
        'optmResponseUnitLiftUnbound':(optmResponseUnitUnboundOut / initResponseUnit) - 1
    }

    dt_optimOut["optmResponseUnitTotalLift"] = (dt_optimOut["optmResponseUnitTotal"] / dt_optimOut["initResponseUnitTotal"]) - 1
    dt_optimOut["optmResponseUnitTotalLiftUnbound"] = (dt_optimOut["optmResponseUnitTotalUnbound"] / dt_optimOut["initResponseUnitTotal"]) - 1

    # Calculate curves and main points for each channel
    if scenario == "max_response":
        levs1 = ["Initial", "Bounded", f"Bounded x {channel_constr_multiplier}"]
    else:
        if dep_var_type == "revenue":
            levs1 = ["Initial", f"Hit ROAS {round(target_value, 2)}", f"Hit ROAS {target_value_ext}"]
        else:
            levs1 = ["Initial", f"Hit CPA {round(target_value, 2)}", f"Hit CPA {round(target_value_ext, 2)}"]

    eval_list["levs1"] = levs1

    # Create a list to store the dataframes
    df_list = []

    initResponseUnit_series = dt_optimOut['initResponseUnit'].iloc[0]
    temp_df = pd.DataFrame({
        'channels': dt_optimOut['channels'],
        'spend': dt_optimOut['initSpendUnit'],
        'response': initResponseUnit_series,
        'type': levs1[0]
    })
    df_list.append(temp_df)

    temp_df = pd.DataFrame({
        'channels': dt_optimOut['channels'],
        'spend': dt_optimOut['optmSpendUnit'],
        'response': dt_optimOut['optmResponseUnit'],
        'type': levs1[1]
    })
    df_list.append(temp_df)

    temp_df = pd.DataFrame({
        'channels': dt_optimOut['channels'],
        'spend': dt_optimOut['optmSpendUnitUnbound'],
        'response': dt_optimOut['optmResponseUnitUnbound'],
        'type': levs1[2]
    })
    df_list.append(temp_df)

    # Concatenate the dataframes
    dt_optimOutScurve = pd.concat(df_list)
    # Rename the columns
    dt_optimOutScurve.columns = ["channels", "spend", "response", "type"]
    # Append a new row

    dt_optimOutScurve = pd.concat([
        dt_optimOutScurve,
        pd.DataFrame({"channels": dt_optimOut["channels"], "spend": 0, "response": 0, "type": "Carryover"})
    ], ignore_index=True)

    # Convert the spend and response columns to numeric
    dt_optimOutScurve['spend'] = pd.to_numeric(dt_optimOutScurve['spend'])
    dt_optimOutScurve['response'] = pd.to_numeric(dt_optimOutScurve['response'])
    # Group by channels TODO: groupby below cause lost of "type"
    # dt_optimOutScurve = dt_optimOutScurve.groupby("channels").agg({"spend": "sum", "response": "sum"})

    plotDT_scurve = {}
    for i in channel_for_allocation:
        carryover_vec = eval_list['hist_carryover_eval'][i]
        dt_optimOutScurve = dt_optimOutScurve.assign(
            spend=np.where(
                (dt_optimOutScurve['channels'] == i) & (dt_optimOutScurve['type'].isin(levs1)),
                dt_optimOutScurve['spend'] + np.mean(carryover_vec),
                np.where(
                    (dt_optimOutScurve['channels'] == i) & (dt_optimOutScurve['type'] == 'Carryover'),
                    np.mean(carryover_vec),
                    dt_optimOutScurve['spend']
                )
            )
        )
        get_max_x = max(dt_optimOutScurve.loc[dt_optimOutScurve['channels'] == i, 'spend']) * 1.5
        simulate_spend = np.linspace(0, get_max_x, 100)
        simulate_response = fx_objective(
            x=simulate_spend,
            coeff=eval_list['coefs_eval'][i],
            alpha=eval_list['alphas_eval'][f'{i}_alphas'],
            inflexion=eval_list['inflexions_eval'][f'{i}_gammas'],
            x_hist_carryover=0,
            get_sum=False
        )
        simulate_response_carryover = fx_objective(
            x=np.mean(carryover_vec),
            coeff=eval_list['coefs_eval'][i],
            alpha=eval_list['alphas_eval'][f'{i}_alphas'],
            inflexion=eval_list['inflexions_eval'][f'{i}_gammas'],
            x_hist_carryover=0,
            get_sum=False
        )
        plotDT_scurve[i] = pd.DataFrame({
            'channel': i,
            'spend': simulate_spend,
            'mean_carryover': np.mean(carryover_vec),
            'carryover_response': simulate_response_carryover,
            'total_response': simulate_response
        })
        dt_optimOutScurve = dt_optimOutScurve.assign(
            response=np.where(
                (dt_optimOutScurve['channels'] == i) & (dt_optimOutScurve['type'] == 'Carryover'),
                simulate_response_carryover,
                dt_optimOutScurve['response']
            )
        )

    # Convert plotDT_scurve to a pandas DataFrame
    plotDT_scurve_df = pd.concat([
        pd.DataFrame(v).assign(channel=k) for k, v in plotDT_scurve.items()
    ], ignore_index=True)

    # Rename columns in mainPoints
    mainPoints = dt_optimOutScurve.rename(columns={"response": "response_point", "spend": "spend_point", "channels": "channel"})

    # Filter out Carryover rows from mainPoints
    temp_caov = mainPoints[mainPoints["type"] == "Carryover"]

    mainPoints["mean_spend"] = mainPoints["spend_point"] - mainPoints["channel"].map(temp_caov.set_index("channel")["spend_point"])
    mainPoints["mean_spend"] = np.where(mainPoints["type"] == "Carryover", mainPoints["spend_point"], mainPoints["mean_spend"])
    if levs1[1] == levs1[2]:
        levs1[2] = levs1[2] + "."
    mainPoints["type"] = pd.Categorical(mainPoints["type"], categories=["Carryover"] + levs1)
    mainPoints["roi_mean"] = mainPoints["response_point"] / mainPoints["mean_spend"]

    mresp_caov = mainPoints[mainPoints["type"] == "Carryover"]["response_point"].values
    mresp_init = mainPoints[mainPoints["type"] == levs1[0]]["response_point"].values - mresp_caov
    mresp_b = mainPoints[mainPoints["type"] == levs1[1]]["response_point"].values - mresp_caov
    mresp_unb = mainPoints[mainPoints["type"] == levs1[2]]["response_point"].values - mresp_caov

    mainPoints["marginal_response"] = np.concatenate((mresp_init, mresp_b, mresp_unb, np.zeros(len(mresp_init))))
    mainPoints["roi_marginal"] = mainPoints["marginal_response"] / mainPoints["mean_spend"]
    mainPoints["cpa_marginal"] = mainPoints["mean_spend"] / mainPoints["marginal_response"]

    eval_list["mainPoints"] = mainPoints

    # # Calculate mean spend and ROI for each channel
    # mainPoints["mean_spend"] = mainPoints["spend_point"] - temp_caov["spend_point"]
    # mainPoints.loc[mainPoints["type"] == "Carryover", "mean_spend"] = mainPoints.loc[mainPoints["type"] == "Carryover", "spend_point"]
    # mainPoints["roi_mean"] = mainPoints["response_point"] / mainPoints["mean_spend"]

    # # Calculate marginal response, ROI, and CPA for each channel
    # mresp_caov = mainPoints[mainPoints["type"] == "Carryover"]["response_point"]
    # mresp_init = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[2]]["response_point"] - mresp_caov
    # mresp_b = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[3]]["response_point"] - mresp_caov
    # mresp_unb = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[4]]["response_point"] - mresp_caov
    # mainPoints["marginal_response"] = [mresp_init, mresp_b, mresp_unb] + [0] * len(mresp_init)
    # mainPoints["roi_marginal"] = mainPoints["marginal_response"] / mainPoints["mean_spend"]
    # mainPoints["cpa_marginal"] = mainPoints["mean_spend"] / mainPoints["marginal_response"]

    # Exporting directory
    if export:
        if json_file is None and plot_folder is not None:
            if plot_folder_sub is None:
                plot_folder_sub = os.path.basename(OutputCollect['plot_folder'])
            plot_folder = os.path.join(plot_folder, plot_folder_sub)
        else:
            plot_folder = os.path.join(OutputCollect['plot_folder'])
        if not os.path.exists(plot_folder):
            print(f"Creating directory for allocator: {plot_folder}")
            os.makedirs(plot_folder)
        # Export results into CSV
        export_dt_optimOut = dt_optimOut
        if dep_var_type == "conversion":
            export_dt_optimOut.columns = [col.replace("Roi", "CPA") for col in export_dt_optimOut.columns]

        # Convert all values to lists to ensure consistent length
        max_len = max(len(v) if isinstance(v, (list, np.ndarray, pd.Series)) else 1 for v in export_dt_optimOut.values())
        # Convert constant values to lists
        export_dt_optimOut_c = export_dt_optimOut.copy()
        for k, v in export_dt_optimOut.items():
            if not isinstance(v, (list, np.ndarray, pd.Series, pd.DataFrame)):
                export_dt_optimOut[k] = [v] * max_len
        # Pad ndarrays
        for k, v in export_dt_optimOut.items():
            if isinstance(v, np.ndarray):
                export_dt_optimOut[k] = np.pad(v, (0, max_len - len(v)))
        # Convert Series to lists
        for k, v in export_dt_optimOut.items():
            if isinstance(v, pd.Series):
                export_dt_optimOut[k] = v.tolist()
        # Convert DataFrames to lists
        for k, v in export_dt_optimOut.items():
            if isinstance(v, pd.DataFrame):
                export_dt_optimOut[k] = list(chain.from_iterable(v.values.tolist()))
        # Create the DataFrame
        export_dt_optimOut = pd.DataFrame.from_dict(export_dt_optimOut, orient='columns')

        export_dt_optimOut.to_csv(os.path.join(plot_folder, f"{select_model}_{scenario}_reallocated.csv"), index=False)
    # Plot allocator results
    if plots:
        # TODO: Enable plots at some point and uncomment this
        plots = None
        # plots = allocation_plots(
        #     InputCollect, OutputCollect,
        #     dt_optimOut,
        #     select_model, scenario, eval_list,
        #     export, plot_folder, quiet
        # )
    else:
        plots = None
    output = {
        'dt_optimOut': dt_optimOut,
        'mainPoints': mainPoints,
        'nlsMod': nlsMod_opt,
        'plots': plots,
        'scenario': scenario,
        'usecase': usecase,
        'total_budget': total_budget_window if total_budget is None else total_budget,
        'skipped_coef0': zero_coef_channel,
        'skipped_constr': zero_constraint_channel,
        'no_spend': zero_spend_channel,
        'ui': plots if ui else None
    }
    # output = pd.DataFrame(output)
    return output


# Define the objective function
def fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum=False, SIMPLIFY=True):
    """
    Calculate the objective function value for a given set of parameters.

    Parameters:
    x (float): The input value.
    coeff (float): Coefficient value.
    alpha (float): Exponent value.
    inflexion (float): Inflexion value.
    x_hist_carryover (float): Carryover value.
    get_sum (bool, optional): If True, returns the sum of the objective function values. Defaults to False.
    SIMPLIFY (bool, optional): If True, simplifies the objective function value. Defaults to True.

    Returns:
    float: The objective function value or the sum of objective function values if get_sum is True.
    """
    if get_sum:
        return np.sum(coeff * x**alpha * np.exp(-inflexion * x))
    else:
        return coeff * x**alpha * np.exp(-inflexion * x)

# Define the optimization problem
def optimize(x0, coeff, alpha, inflexion, x_hist_carryover, total_budget, channel_constr_low, channel_constr_up, channel_constr_multiplier, optim_algo, maxeval, constr_mode):
    """
    Optimize the allocation of resources based on the given parameters.

    Args:
        x0 (array_like): Initial guess for the allocation.
        coeff (array_like): Coefficients for the allocation function.
        alpha (float): Exponent for the allocation function.
        inflexion (float): Inflexion parameter for the allocation function.
        x_hist_carryover (array_like): Historical allocation values.
        total_budget (float): Total budget for the allocation.
        channel_constr_low (array_like): Lower bounds for channel constraints.
        channel_constr_up (array_like): Upper bounds for channel constraints.
        channel_constr_multiplier (float): Multiplier for channel constraints.
        optim_algo (str): Optimization algorithm to use.
        maxeval (int): Maximum number of function evaluations.
        constr_mode (str): Constraint mode ('eq' for equality, 'ineq' for inequality).

    Returns:
        array_like: Optimized allocation.

    """
    import scipy.optimize as opt

    def f(x):
        return -np.sum(coeff * x**alpha * np.exp(-inflexion * x))

    def cons(x):
        return x - total_budget

    if constr_mode == "eq":
        cons = (x - total_budget) * channel_constr_multiplier

    if channel_constr_low is not None and channel_constr_up is not None:
        cons = np.concatenate((cons,
                                x - channel_constr_low,
                                channel_constr_up - x))

    res = opt.minimize(f, x0, method=optim_algo, constraints=cons, options={"maxiter": maxeval})

    return res.x


def print_robyn_allocator(x):
    """
    Prints the allocator details for Robyn.

    Args:
        x: The input object containing allocator details.

    Returns:
        None
    """
    temp = x.dt_optimOut[~x.dt_optimOut.optmRoiUnit.isna(), ]
    coef0 = (len(x.skipped_coef0) > 0) * f"Coefficient 0: {v2t(x.skipped_coef0, quotes=False)}"
    constr = (len(x.skipped_constr) > 0) * f"Constrained @0: {v2t(x.skipped_constr, quotes=False)}"
    nospend = (len(x.no_spend) > 0) * f"Spend = 0: {v2t(x.no_spend, quotes=False)}"
    media_skipped = " | ".join([coef0, constr, nospend])
    media_skipped = media_skipped if media_skipped else "None"

    print(f"Model ID: {x.dt_optimOut.solID[0]}\n"
          f"Scenario: {x.scenario}\n"
          f"Use case: {x.usecase}\n"
          f"Window: {x.dt_optimOut.date_min[0]}:{x.dt_optimOut.date_max[0]} ({x.dt_optimOut.periods[0]})\n"
          f"Dep. Variable Type: {temp.dep_var_type[0]}\n"
          f"Media Skipped: {media_skipped}\n"
          f"Relative Spend Increase: {num_abbr(100 * x.dt_optimOut.optmSpendUnitTotalDelta[0], 3)}% ({formatNum(sum(x.dt_optimOut.optmSpendUnitTotal) - sum(x.dt_optimOut.initSpendUnitTotal), abbr=True, sign=True)})"
          f"Total Response Increase (Optimized): {signif(100 * x.dt_optimOut.optmResponseUnitTotalLift[0], 3)}%\n"
          f"Allocation Summary:\n"
         )


def plot_robyn_allocator(x, *args, **kwargs):
    """
    Plot the Robyn allocator.

    Parameters:
    - x: The input data.
    - *args: Additional positional arguments to be passed to the plot function.
    - **kwargs: Additional keyword arguments to be passed to the plot function.
    """
    plots = x.plots
    plots = plots.plots
    plot(plots, *args, **kwargs)

def calculate_channels(X):
    eval_list = ROBYN_TEMP  # This should be defined globally or passed as an argument
    return np.array([
        fx_objective_channel(x, coeff, alpha, inflexion, x_hist)
        for x, coeff, alpha, inflexion, x_hist in zip(
            X,
            eval_list['coefs_eval'],
            eval_list['alphas_eval'].values(),
            eval_list['inflexions_eval'].values(),
            eval_list['hist_carryover_eval'].mean(axis=0)
        )
    ])

def eval_f(X, grad):
    eval_list = ROBYN_TEMP
    results = np.array([
        fx_objective(x, coeff, alpha, inflexion, x_hist)
        for x, coeff, alpha, inflexion, x_hist in zip(X, eval_list['coefs_eval'], eval_list['alphas_eval'].values(), eval_list['inflexions_eval'].values(), eval_list['hist_carryover_eval'].mean(axis=0))
    ])
    objective = -np.sum(results)

    if grad.size > 0:
        grad[:] = np.array([
            fx_gradient(x, coeff, alpha, inflexion, x_hist)
            for x, coeff, alpha, inflexion, x_hist in zip(X, eval_list['coefs_eval'], eval_list['alphas_eval'].values(), eval_list['inflexions_eval'].values(), eval_list['hist_carryover_eval'].mean(axis=0))
        ])

    return objective

def fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum=True):
    """
    Calculate the objective function value for the given parameters.

    Parameters:
    x (array-like): Input values.
    coeff (float): Coefficient value.
    alpha (float): Alpha value.
    inflexion (float): Inflexion value.
    x_hist_carryover (array-like): Historical carryover values.
    get_sum (bool, optional): Flag to determine if the sum of the objective function should be returned. Defaults to True.

    Returns:
    float: Objective function value.
    """
    # Apply Michaelis Menten model to scale spend to exposure
    xScaled = x
    # Adstock scales
    xAdstocked = x + np.mean(x_hist_carryover)
    # Hill transformation
    if get_sum:
        xOut = coeff * np.sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    else:
        xOut = coeff * ((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    return xOut

def fx_gradient(x, coeff, alpha, inflexion, x_hist_carryover):
    """
    Calculate the gradient of the function fx.

    Parameters:
    x (float): The input value.
    coeff (float): Coefficient value.
    alpha (float): Alpha value.
    inflexion (float): Inflexion value.
    x_hist_carryover (float): Carryover value.

    Returns:
    float: The gradient of the function fx.
    """
    # Apply Michaelis Menten model to scale spend to exposure
    xScaled = x
    # Adstock scales
    xAdstocked = x + np.mean(x_hist_carryover)
    xOut = -coeff * np.sum((alpha * (inflexion**alpha) * (xAdstocked**(alpha - 1))) / (xAdstocked**alpha + inflexion**alpha)**2)
    return xOut

def fx_objective_channel(x, coeff, alpha, inflexion, x_hist_carryover):
    # Adstock calculation
    xAdstocked = x + np.mean(x_hist_carryover)
    xOut = -coeff * np.sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    return xOut

# def fx_objective_channel(x, coeff, alpha, inflexion, x_hist_carryover):
#     """
#     Calculate the objective value for a channel allocation.

#     Parameters:
#     x (array-like): The channel allocation.
#     coeff (float): Coefficient used in the objective calculation.
#     alpha (float): Alpha parameter used in the objective calculation.
#     inflexion (float): Inflexion parameter used in the objective calculation.
#     x_hist_carryover (array-like): Historical carryover values.

#     Returns:
#     float: The objective value for the given channel allocation.
#     """
#     # Apply Michaelis Menten model to scale spend to exposure
#     xScaled = x
#     # Adstock scales
#     xAdstocked = x + np.mean(x_hist_carryover)
#     xOut = -coeff * np.sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
#     return xOut

def eval_g_eq(X, grad):

    eval_list = ROBYN_TEMP
    # Assuming 'total_budget_unit' is a scalar value representing the total budget
    constraint_value = np.sum(X) - eval_list["total_budget_unit"]
    if grad.size > 0:
        grad[:] = np.ones(len(X))  # Set gradient of the constraint
    return constraint_value  # Return the scalar constraint value

def eval_g_ineq(X, grad):

    eval_list = ROBYN_TEMP
    constraint_value = np.sum(X) - eval_list["total_budget_unit"]
    if grad.size > 0:
        grad[:] = np.ones(len(X))  # Set gradient of the constraint
    return constraint_value  # This must be non-negative for "ineq" constraints to be satisfied


# def eval_g_eq(X, target_value):
#     """
#     Evaluate the equality constraint function for optimization.

#     Parameters:
#     X (array-like): The decision variables.
#     target_value (float): The target value for the constraint.

#     Returns:
#     dict: A dictionary containing the constraint value and its gradient.
#         The constraint value is the sum of the decision variables minus the total budget unit.
#         The gradient is an array of ones with the same length as X.
#     """
#     global ROBYN_TEMP
#     eval_list = ROBYN_TEMP
#     constr = np.sum(X) - eval_list["total_budget_unit"]
#     grad = np.ones(len(X))
#     return {"constraints": constr, "jacobian": grad}

# def eval_g_ineq(X, target_value):
#     """
#     Evaluate the inequality constraints for the optimization problem.

#     Parameters:
#     - X: A numpy array representing the decision variables.
#     - target_value: The target value for the optimization problem.

#     Returns:
#     - A dictionary containing the constraints and their gradients.
#     """
#     global ROBYN_TEMP
    # eval_list = ROBYN_TEMP
    # constr = np.sum(X) - eval_list["total_budget_unit"]
    # grad = np.ones(len(X))
    # return {"constraints": constr, "jacobian": grad}

def wrapper_eval_g_eq_effi(target_value):
    def eval_g_eq_effi(X, grad):
        """
        Evaluate the equality constraints and their Jacobian for the given input vector X and target value.

        Parameters:
        X (array-like): Input vector.
        target_value (float): Target value for the constraints.

        Returns:
        dict: A dictionary containing the constraints and their Jacobian.
            The constraints are stored under the key 'constraints',
            and the Jacobian is stored under the key 'jacobian'.
        """
        global ROBYN_TEMP
        eval_list = ROBYN_TEMP
        # sum_response = np.sum(np.vectorize(fx_objective)(x=X, coeff=eval_list["coefs_eval"], alpha=eval_list["alphas_eval"].values(), inflexion=eval_list["inflexions_eval"].values(), x_hist_carryover=eval_list["hist_carryover_eval"].mean(axis=0)))
        sum_response = np.sum([
            fx_objective(x, coeff, alpha, inflexion, x_hist)
            for x, coeff, alpha, inflexion, x_hist in zip(X, eval_list['coefs_eval'], eval_list['alphas_eval'].values(), eval_list['inflexions_eval'].values(), eval_list['hist_carryover_eval'].mean(axis=0))
        ])
        if target_value is None or (isinstance(target_value, (list, np.ndarray)) and len(target_value) == 0):
            if eval_list["dep_var_type"] == "conversion":
                constr = np.sum(X) - sum_response * eval_list["target_value"]
            else:
                constr = np.sum(X) - sum_response / eval_list["target_value"]
        else:
            if eval_list["dep_var_type"] == "conversion":
                constr = np.sum(X) - sum_response * target_value
            else:
                constr = np.sum(X) - sum_response / target_value
        # grad = np.ones(len(X)) - np.vectorize(fx_gradient)(x=X, coeff=eval_list["coefs_eval"], alpha=eval_list["alphas_eval"].values(), inflexion=eval_list["inflexions_eval"].values(), x_hist_carryover=eval_list["hist_carryover_eval"].mean(axis=0))
        grads = [
            fx_gradient(x, coeff, alpha, inflexion, x_hist)
            for x, coeff, alpha, inflexion, x_hist in zip(
                X,
                eval_list["coefs_eval"],
                eval_list["alphas_eval"].values(),
                eval_list["inflexions_eval"].values(),
                eval_list["hist_carryover_eval"].mean(axis=0)
            )
        ]
        grad = np.ones(len(X)) - np.array(grads)
        return constr
    return eval_g_eq_effi

def eval_g_eq_effi(X, target_value):
    """
    Evaluate the equality constraints for the efficiency function.

    Parameters:
    X (array-like): The decision variables.
    target_value (float): The target value for the efficiency function.

    Returns:
    dict: A dictionary containing the constraints and the jacobian.
    """
    global ROBYN_TEMP
    eval_list = ROBYN_TEMP
    # sum_response = np.sum(np.vectorize(fx_objective)(X, eval_list["coefs_eval"], eval_list["alphas_eval"].values(), eval_list["inflexions_eval"].values(), eval_list["hist_carryover_eval"].mean(axis=0)))
    sum_response = np.sum([
        fx_objective(x, coeff, alpha, inflexion, x_hist)
        for x, coeff, alpha, inflexion, x_hist in zip(X, eval_list['coefs_eval'], eval_list['alphas_eval'].values(), eval_list['inflexions_eval'].values(), eval_list['hist_carryover_eval'].mean(axis=0))
    ])
    if target_value is None or (isinstance(target_value, (list, np.ndarray)) and len(target_value) == 0):

        if eval_list["dep_var_type"] == "conversion":
            constr = np.sum(X) - sum_response * eval_list["target_value"]
        else:
            constr = np.sum(X) - sum_response / eval_list["target_value"]
    else:
        if eval_list["dep_var_type"] == "conversion":
            constr = np.sum(X) - sum_response * target_value
        else:
            constr = np.sum(X) - sum_response / target_value
    # grad = np.ones(len(X)) - np.vectorize(fx_gradient)(X, eval_list["coefs_eval"], eval_list["alphas_eval"].values(), eval_list["inflexions_eval"].values(), eval_list["hist_carryover_eval"].mean(axis=0))
    grads = [
        fx_gradient(x, coeff, alpha, inflexion, x_hist)
        for x, coeff, alpha, inflexion, x_hist in zip(
            X,
            eval_list["coefs_eval"],
            eval_list["alphas_eval"].values(),
            eval_list["inflexions_eval"].values(),
            eval_list["hist_carryover_eval"].mean(axis=0)
        )
    ]
    grad = np.ones(len(X)) - np.array(grads)
    return constr

def get_adstock_params(InputCollect, dt_hyppar):
    """
    Retrieves the adstock parameters based on the adstock type specified in InputCollect.

    Parameters:
    InputCollect (object): The input collection object.
    dt_hyppar (DataFrame): The DataFrame containing the adstock hyperparameters.

    Returns:
    DataFrame: The adstock hyperparameters based on the adstock type.

    """
    if InputCollect.adstock == "geometric":
        getAdstockHypPar = dt_hyppar.loc[dt_hyppar.columns.str.extract(".*_thetas", expand=False)]
    else:
        getAdstockHypPar = dt_hyppar.loc[dt_hyppar.columns.str.extract(".*_shapes|.*_scales", expand=False)]
    return getAdstockHypPar


def get_hill_params(InputCollect, OutputCollect, dt_hyppar, dt_coef, mediaSpendSorted, select_model, chnAdstocked=None):
    """
    Calculate the hill parameters for the given inputs.

    Args:
        InputCollect (type): Description of InputCollect.
        OutputCollect (type): Description of OutputCollect.
        dt_hyppar (type): Description of dt_hyppar.
        dt_coef (type): Description of dt_coef.
        mediaSpendSorted (type): Description of mediaSpendSorted.
        select_model (type): Description of select_model.
        chnAdstocked (type, optional): Description of chnAdstocked. Defaults to None.

    Returns:
        dict: A dictionary containing the calculated hill parameters:
            - "alphas": The alphas values.
            - "inflexions": The inflexions values.
            - "coefs_sorted": The sorted coefficients.
    """
    hillHypParVec = dt_hyppar.loc[:, dt_hyppar.columns.str.contains("_alphas|_gammas")]
    if isinstance(mediaSpendSorted, list):
        columns_to_select_alpha = [s + "_alphas" for s in mediaSpendSorted]
    else:
        columns_to_select_alpha = mediaSpendSorted + "_alphas"
    alphas = hillHypParVec.loc[:, columns_to_select_alpha]
    # alphas = hillHypParVec.loc[:, mediaSpendSorted + "_alphas"]
    if isinstance(mediaSpendSorted, list):
        columns_to_select_gammas = [s + "_gammas" for s in mediaSpendSorted]
    else:
        columns_to_select_gammas = mediaSpendSorted + "_gammas"
    gammas = hillHypParVec.loc[:, columns_to_select_gammas]

    if chnAdstocked is None:
        mask = (OutputCollect['mediaVecCollect']['type'] == 'adstockedMedia') & (OutputCollect['mediaVecCollect']['solID'] == select_model)
        filtered_df = OutputCollect['mediaVecCollect'].loc[mask]

        selected_df = filtered_df.loc[:, mediaSpendSorted]
        chnAdstocked = selected_df.iloc[InputCollect['robyn_inputs']['rollingWindowStartWhich']:InputCollect['robyn_inputs']['rollingWindowEndWhich'] + 1]  # +1 if end index should be inclusive

    if isinstance(gammas, pd.DataFrame):
        gammas_cycled = itertools.cycle(gammas.values.flatten())
        # Initialize inflexions as the first row of chnAdstocked
        inflexions = chnAdstocked.iloc[0:1].copy()
        # Apply a function to each column of chnAdstocked
        for i in range(chnAdstocked.shape[1]):
            max_val = np.max(chnAdstocked.iloc[:, i])
            min_val = np.min(chnAdstocked.iloc[:, i])
            # Ensure gamma_value is a scalar by using float() or accessing the first element if it's a single-element array
            gamma_value = float(next(gammas_cycled))  # This assumes gammas is now a flat array or list
            inflexion = (max_val - min_val) * (1 - gamma_value + gamma_value)
            inflexions.iloc[0, i] = inflexion
        # Set the column names of inflexions to match gammas
        if isinstance(gammas, pd.DataFrame):
            inflexions.columns = gammas.columns
    else:
        gammas = np.array([gammas])
        # Create a list to store the inflexions
        inflexions = []
        for i in range(chnAdstocked.shape[1]):
            max_val = np.max(chnAdstocked.iloc[:, i])
            min_val = np.min(chnAdstocked.iloc[:, i])
            # Ensure gamma_value is a scalar by using float() or accessing the first element if it's a single-element array
            gamma_value = float(gammas[i % len(gammas)])  # This assumes gammas is already a flat array or list
            inflexion = (max_val - min_val) * (1 - gamma_value + gamma_value)
            inflexions.append(inflexion)

    coefs = dt_coef.set_index('rn').loc[mediaSpendSorted]['coefs']
    if not isinstance(coefs, float):
        coefs = coefs.drop_duplicates()

    if isinstance(dt_coef, pd.DataFrame):
        coefs = dt_coef['coefs']
    else:
        coefs = pd.Series(dt_coef)
    # Assign names to the elements of coefs
    coefs.index = dt_coef['rn']
    # Sort coefs according to mediaSpendSorted
    coefs_sorted = coefs[mediaSpendSorted]

    # Removing old logic as it might not be needed
    # if isinstance(coefs, pd.DataFrame):
    #     coefs_sorted = pd.Series(coefs.iloc[:, 0].values, index=coefs.index, name='DataFrame Column')
    # elif isinstance(coefs, pd.Series):
    #     coefs_sorted = pd.Series(coefs.values, index=coefs.index, name=coefs.name)
    # elif isinstance(coefs, np.ndarray):
    #     coefs_sorted = pd.Series(coefs, index=range(len(coefs)), name='Array')
    # elif isinstance(coefs, list):
    #     coefs_sorted = pd.Series(coefs, index=range(len(coefs)), name='List')
    # else:
    #     coefs_sorted = pd.Series(coefs, index=dt_coef.index, name=dt_coef['rn'].values[0])

    return {'alphas': alphas, 'inflexions': inflexions, 'coefs_sorted': coefs_sorted}
