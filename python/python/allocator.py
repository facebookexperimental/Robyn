# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import pandas as pd
import numpy as np

from .checks import check_allocator, check_allocator_constrains, check_metric_dates, check_daterange
from .response import robyn_response, which_usecase

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
    alphas = hills["alphas"]
    inflexions = hills["inflexions"]
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
    initSpendUnit = histFiltered[mediaSpendSorted].mean()
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
    initResponseUnit = None
    initResponseMargUnit = None
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
            alpha=alphas[f"{mediaSpendSorted[i]}_alphas"],
            inflexion=inflexions[f"{mediaSpendSorted[i]}_gammas"],
            x_hist_carryover=np.mean(hist_carryover_temp),
            get_sum=False
        )
        resp_simulate_plus1 = fx_objective(
            x=x_input + 1,
            coeff=coefs_sorted[mediaSpendSorted[i]],
            alpha=alphas[f"{mediaSpendSorted[i]}_alphas"],
            inflexion=inflexions[f"{mediaSpendSorted[i]}_gammas"],
            x_hist_carryover=np.mean(hist_carryover_temp),
            get_sum=False
        )
        initResponseUnit = np.append(initResponseUnit, resp_simulate)
        initResponseMargUnit = np.append(initResponseMargUnit, resp_simulate_plus1 - resp_simulate)

    qa_carryover = np.c_[qa_carryover, np.zeros(len(qa_carryover))]
    initResponseUnit.columns = hist_carryover.columns = qa_carryover.columns = mediaSpendSorted

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
                target_value = sum(initSpendUnit) / sum(initResponseUnit) * 1.2
            target_value_ext = target_value * 1.5
        else:
            if target_value is None:
                target_value = sum(initResponseUnit) / sum(initSpendUnit) * 0.8
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
    zero_constraint_channel = mediaSpendSorted[skip_these]
    if any(skip_these) and not quiet:
        print("Excluded variables (constrained to 0):", zero_constraint_channel)
    if not all(coefSelectorSorted):
        zero_coef_channel = set(names(coefSelectorSorted)) - set(mediaSpendSorted[coefSelectorSorted])
        if not quiet:
            print("Excluded variables (coefficients are 0):", zero_coef_channel)
    else:
        zero_coef_channel = []
    channel_to_drop_loc = mediaSpendSorted in (zero_coef_channel + zero_constraint_channel)
    channel_for_allocation = mediaSpendSorted[~channel_to_drop_loc]
    if any(channel_to_drop_loc):
        temp_init = temp_init_all[channel_for_allocation]
        temp_ub = temp_ub_all[channel_for_allocation]
        temp_lb = temp_lb_all[channel_for_allocation]
        temp_ub_ext = temp_ub_ext_all[channel_for_allocation]
        temp_lb_ext = temp_lb_ext_all[channel_for_allocation]
        x0 = x0_all[channel_for_allocation]
        lb = lb_all[channel_for_allocation]
        ub = ub_all[channel_for_allocation]
        x0_ext = x0_ext_all[channel_for_allocation]
        lb_ext = lb_ext_all[channel_for_allocation]
        ub_ext = ub_ext_all[channel_for_allocation]

    x0 = lb = temp_init * temp_lb
    ub = temp_init * temp_ub
    x0_ext = lb_ext = temp_init * temp_lb_ext
    ub_ext = temp_init * temp_ub_ext

    coefs_eval = coefs_sorted[channel_for_allocation]
    alphas_eval = alphas[f"{channel_for_allocation}_alphas"]
    inflexions_eval = inflexions[f"{channel_for_allocation}_gammas"]
    hist_carryover_eval = hist_carryover[channel_for_allocation]

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
    options["ROBYN_TEMP"] = eval_list

    # Set optim options
    if optim_algo == "MMA_AUGLAG":
        local_opts = [
            "algorithm", "NLOPT_LD_MMA",
            "xtol_rel", 1.0e-10
        ]
    else:
        local_opts = [
            "algorithm", "NLOPT_LD_SLSQP",
            "xtol_rel", 1.0e-10
        ]

    # Run optim
    x_hist_carryover = [mean(hist_carryover_eval) for hist_carryover_eval in x_hist_carryover]
    if scenario == "max_response":
        # bounded optimisation
        nlsMod = nlopt.nlopt(
            x0=x0,
            f=eval_f,
            f_eq=eval_g_eq if constr_mode == "eq" else None,
            f_ieq=eval_g_ineq if constr_mode == "ineq" else None,
            lb=lb,
            ub=ub,
            opts=[
                "algorithm", "NLOPT_LD_AUGLAG",
                "xtol_rel", 1.0e-10,
                "maxeval", maxeval,
                "local_opts", local_opts
            ],
            target_value=None
        )
        # unbounded optimisation
        nlsModUnbound = nlopt.nlopt(
            x0=x0_ext,
            f=eval_f,
            f_eq=eval_g_eq if constr_mode == "eq" else None,
            f_ieq=eval_g_ineq if constr_mode == "ineq" else None,
            lb=lb_ext,
            ub=ub_ext,
            opts=[
                "algorithm", "NLOPT_LD_AUGLAG",
                "xtol_rel", 1.0e-10,
                "maxeval", maxeval,
                "local_opts", local_opts
            ],
            target_value=None
        )
    else:
        # bounded optimisation
        total_response = sum(OutputCollect.xDecompAgg.xDecompAgg)
        nlsMod = nlopt.nlopt(
            x0=x0,
            f=eval_f,
            f_eq=eval_g_eq_effi if constr_mode == "eq" else None,
            f_ieq=eval_g_eq_effi if constr_mode == "ineq" else None,
            lb=lb,
            ub=total_response * [1] * len(ub),
            opts=[
                "algorithm", "NLOPT_LD_AUGLAG",
                "xtol_rel", 1.0e-10,
                "maxeval", maxeval,
                "local_opts", local_opts
            ],
            target_value=target_value
        )
        # unbounded optimisation
        nlsModUnbound = nlopt.nlopt(
            x0=x0,
            f=eval_f,
            f_eq=eval_g_eq_effi if constr_mode == "eq" else None,
            f_ieq=eval_g_eq_effi if constr_mode == "ineq" else None,
            lb=lb,
            ub=total_response * [1] * len(ub),
            opts=[
                "algorithm", "NLOPT_LD_AUGLAG",
                "xtol_rel", 1.0e-10,
                "maxeval", maxeval,
                "local_opts", local_opts
            ],
            target_value=target_value_ext
        )

    # get marginal
    optmSpendUnit = nlsMod.solution
    optmResponseUnit = -eval_f(optmSpendUnit)["objective.channel"]
    optmSpendUnitUnbound = nlsModUnbound.solution
    optmResponseUnitUnbound = -eval_f(optmSpendUnitUnbound)["objective.channel"]

    # Get the input data
    if InputCollect is None:
        InputCollect = robyn_object.InputCollect

    if OutputCollect is None:
        OutputCollect = robyn_object.OutputCollect

    if select_model is None:
        select_model = robyn_object.select_model

    if json_file is None:
        json_file = robyn_object.json_file

    if total_budget is None:
        total_budget = robyn_object.total_budget

    if target_value is None:
        target_value = robyn_object.target_value

    if date_range is None:
        date_range = robyn_object.date_range

    if channel_constr_low is None:
        channel_constr_low = robyn_object.channel_constr_low

    if channel_constr_up is None:
        channel_constr_up = robyn_object.channel_constr_up

    if channel_constr_multiplier is None:
        channel_constr_multiplier = robyn_object.channel_constr_multiplier

    if optim_algo is None:
        optim_algo = robyn_object.optim_algo

    if maxeval is None:
        maxeval = robyn_object.maxeval

    if constr_mode is None:
        constr_mode = robyn_object.constr_mode

    if plots is None:
        plots = robyn_object.plots

    if plot_folder is None:
        plot_folder = robyn_object.plot_folder

    if plot_folder_sub is None:
        plot_folder_sub = robyn_object.plot_folder_sub

    if export is None:
        export = robyn_object.export

    if quiet is None:
        quiet = robyn_object.quiet

    if ui is None:
        ui = robyn_object.ui

    # Get the channel names
    channel_for_allocation = InputCollect.columns

    # Get the coefficients and inflexion points
    coefs_eval = OutputCollect.coefs_eval
    alphas_eval = OutputCollect.alphas_eval
    inflexions_eval = OutputCollect.inflexions_eval

    # Get the historical spend and response data
    x_hist = InputCollect.x_hist
    y_hist = InputCollect.y_hist

    # Get the initial spend and response data
    initSpendUnit = InputCollect.initSpendUnit
    initResponseUnit = InputCollect.initResponseUnit

    # Get the channel to drop
    channel_to_drop = OutputCollect.channel_to_drop
    channel_to_drop_loc = channel_for_allocation.get_loc(channel_to_drop)

    # Get the optimization bounds
    optmSpendUnit = np.array([0] * len(channel_for_allocation))
    optmResponseUnit = np.array([0] * len(channel_for_allocation))
    optmSpendUnitUnbound = np.array([0] * len(channel_for_allocation))
    optmResponseUnitUnbound = np.array([0] * len(channel_for_allocation))

    # Optimize the spend and response for each channel
    optmSpendUnit = optimize(optmSpendUnit, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, total_budget, channel_constr_low, channel_constr_up, channel_constr_multiplier, optim_algo, maxeval, constr_mode)
    optmResponseUnit = fx_objective(optmSpendUnit, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, get_sum=True, SIMPLIFY=True)
    optmSpendUnitUnbound = optimize(optmSpendUnitUnbound, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, total_budget, channel_constr_low, channel_constr_up, channel_constr_multiplier, optim_algo, maxeval, constr_mode)
    optmResponseUnitUnbound = fx_objective(optmSpendUnitUnbound, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, get_sum=True, SIMPLIFY=True)

    # Calculate the marginal response for each channel
    optmResponseMargUnit = fx_objective(optmSpendUnit + 1, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, get_sum=False, SIMPLIFY=True) - optmResponseUnit
    optmResponseMargUnitUnbound = fx_objective(optmSpendUnitUnbound + 1, coefs_eval, alphas_eval, inflexions_eval, x_hist_carryover, get_sum=False, SIMPLIFY=True) - optmResponseUnitUnbound

    # Collect the output
    names = [channel_for_allocation[i] for i in range(len(channel_for_allocation))]
    mediaSpendSorted = names

    optmSpendUnitOut = np.zeros(len(channel_for_allocation))
    optmResponseUnitOut = np.zeros(len(channel_for_allocation))
    optmResponseMargUnitOut = np.zeros(len(channel_for_allocation))
    optmSpendUnitUnboundOut = np.zeros(len(channel_for_allocation))
    optmResponseUnitUnboundOut = np.zeros(len(channel_for_allocation))
    optmResponseMargUnitUnboundOut = np.zeros(len(channel_for_allocation))

    optmSpendUnitOut[channel_to_drop_loc] = optmResponseUnitOut[channel_to_drop_loc] = optmResponseMargUnitOut[channel_to_drop_loc] = optmSpendUnitUnboundOut[channel_to_drop_loc] = optmResponseUnitUnboundOut[channel_to_drop_loc] = optmResponseMargUnitUnboundOut[channel_to_drop_loc] = 0

    optmSpendUnitOut[~channel_to_drop_loc] = optmSpendUnit
    optmResponseUnitOut[~channel_to_drop_loc] = optmResponseUnit
    optmResponseMargUnitOut[~channel_to_drop_loc] = optmResponseMargUnit
    optmSpendUnitUnboundOut[~channel_to_drop_loc] = optmSpendUnitUnbound
    optmResponseUnitUnboundOut[~channel_to_drop_loc] = optmResponseUnitUnbound
    optmResponseMargUnitUnboundOut[~channel_to_drop_loc] = optmResponseMargUnitUnbound

    dt_optimOut = pd.DataFrame(
        solID=select_model,
        dep_var_type=dep_var_type,
        channels=mediaSpendSorted,
        date_min=date_min,
        date_max=date_max,
        periods=f"{initial_mean_period} {InputCollect.intervalType}",
        constr_low=temp_lb_all,
        constr_low_abs=lb_all,
        constr_up=temp_ub_all,
        constr_up_abs=ub_all,
        unconstr_mult=channel_constr_multiplier,
        constr_low_unb=temp_lb_ext_all,
        constr_low_unb_abs=lb_ext_all,
        constr_up_unb=temp_ub_ext_all,
        constr_up_unb_abs=ub_ext_all,
        # Historical spends
        histSpendAll=histSpendAll,
        histSpendAllTotal=histSpendAllTotal,
        histSpendAllUnit=histSpendAllUnit,
        histSpendAllUnitTotal=histSpendAllUnitTotal,
        histSpendAllShare=histSpendAllShare,
        histSpendWindow=histSpendWindow,
        histSpendWindowTotal=histSpendWindowTotal,
        histSpendWindowUnit=initSpendUnit,
        histSpendWindowUnitTotal=histSpendWindowUnitTotal,
        histSpendWindowShare=histSpendWindowShare,
        # Initial spends for allocation
        initSpendUnit=initSpendUnit,
        initSpendUnitTotal=initSpendUnitTotal,
        initSpendShare=initSpendShare,
        initSpendTotal=initSpendUnitTotal * len(simulation_period),
        # initSpendUnitRaw=histSpendUnitRaw,
        # adstocked=adstocked,
        # adstocked_start_date=as.Date(ifelse(adstocked, head(resp$date, 1), NA), origin="1970-01-01"),
        # adstocked_end_date=as.Date(ifelse(adstocked, tail(resp$date, 1), NA), origin="1970-01-01"),
        # adstocked_periods=length(resp$date),
        initResponseUnit=initResponseUnit,
        initResponseUnitTotal=sum(initResponseUnit),
        initResponseMargUnit=initResponseMargUnit,
        initResponseTotal=sum(initResponseUnit) * len(simulation_period),
        initResponseUnitShare=initResponseUnit / sum(initResponseUnit),
        initRoiUnit=initResponseUnit / initSpendUnit,
        initCpaUnit=initSpendUnit / initResponseUnit,
        # Budget change
        total_budget_unit=total_budget_unit,
        total_budget_unit_delta=total_budget_unit / initSpendUnitTotal - 1,
        # Optimized
        optmSpendUnit=optmSpendUnitOut,
        optmSpendUnitDelta=(optmSpendUnitOut / initSpendUnit - 1),
        optmSpendUnitTotal=sum(optmSpendUnitOut),
        optmSpendUnitTotalDelta=sum(optmSpendUnitOut) / initSpendUnitTotal - 1,
        optmSpendShareUnit=optmSpendUnitOut / sum(optmSpendUnitOut),
        optmSpendTotal=sum(optmSpendUnitOut) * len(simulation_period),
        optmSpendUnitUnbound=optmSpendUnitUnboundOut,
        optmSpendUnitDeltaUnbound=(optmSpendUnitUnboundOut / initSpendUnit - 1),
        optmSpendUnitTotalUnbound=sum(optmSpendUnitUnboundOut),
        optmSpendUnitTotalDeltaUnbound=sum(optmSpendUnitUnboundOut) / initSpendUnitTotal - 1,
        optmSpendShareUnitUnbound=optmSpendUnitUnboundOut / sum(optmSpendUnitUnboundOut),
        optmSpendTotalUnbound=sum(optmSpendUnitUnboundOut) * len(simulation_period),
        optmResponseUnit=optmResponseUnitOut,
        optmResponseMargUnit=optmResponseMargUnitOut,
        optmResponseUnitTotal=sum(optmResponseUnitOut),
        optmResponseTotal=sum(optmResponseUnitOut) * len(simulation_period),
        optmResponseUnitShare=optmResponseUnitOut / sum(optmResponseUnitOut),
        optmRoiUnit=optmResponseUnitOut / optmSpendUnitOut,
        optmCpaUnit=optmSpendUnitOut / optmResponseUnitOut,
        optmResponseUnitLift=(optmResponseUnitOut / initResponseUnit) - 1,
        optmResponseUnitUnbound=optmResponseUnitUnboundOut,
        optmResponseMargUnitUnbound=optmResponseMargUnitUnboundOut,
        optmResponseUnitTotalUnbound=sum(optmResponseUnitUnboundOut),
        optmResponseTotalUnbound=sum(optmResponseUnitUnboundOut) * len(simulation_period),
        optmResponseUnitShareUnbound=optmResponseUnitUnboundOut / sum(optmResponseUnitUnboundOut),
        optmRoiUnitUnbound=optmResponseUnitUnboundOut / optmSpendUnitUnboundOut,
        optmCpaUnitUnbound=optmSpendUnitUnboundOut / optmResponseUnitUnboundOut,
        optmResponseUnitLiftUnbound=(optmResponseUnitUnboundOut / initResponseUnit) - 1
    )

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

    dt_optimOutScurve = pd.concat([
        robyn_object.dt_optimOut[["channels", "initSpendUnit", "initResponseUnit"]].rename(columns={"initSpendUnit": "spend", "initResponseUnit": "response"}).assign(x=levs1[0]).to_numpy(),
        robyn_object.dt_optimOut[["channels", "optmSpendUnit", "optmResponseUnit"]].rename(columns={"optmSpendUnit": "spend", "optmResponseUnit": "response"}).assign(x=levs1[1]).to_numpy(),
        robyn_object.dt_optimOut[["channels", "optmSpendUnitUnbound", "optmResponseUnitUnbound"]].rename(columns={"optmSpendUnitUnbound": "spend", "optmResponseUnitUnbound": "response"}).assign(x=levs1[2]).to_numpy()
    ], ignore_index=True)

    dt_optimOutScurve = pd.concat([
        dt_optimOutScurve,
        pd.DataFrame({"channels": robyn_object.dt_optimOut["channels"], "spend": 0, "response": 0, "type": "Carryover"})
    ], ignore_index=True)

    dt_optimOutScurve = dt_optimOutScurve.groupby("channels").agg({"spend": "sum", "response": "sum"})

    plotDT_scurve = []

    for i in channel_for_allocation:
        carryover_vec = eval_list["hist_carryover_eval"][i]

        dt_optimOutScurve.loc[i, "spend"] = dt_optimOutScurve.loc[i, "spend"] + carryover_vec.mean()

        get_max_x = dt_optimOutScurve.loc[i, "spend"].max() * 1.5

        simulate_spend = np.arange(0, get_max_x, 100)

        simulate_response = robyn_object.fx_objective(
            x=simulate_spend,
            coeff=eval_list["coefs_eval"][i],
            alpha=eval_list["alphas_eval"][f"{i}_alphas"],
            inflexion=eval_list["inflexions_eval"][f"{i}_gammas"],
            x_hist_carryover=0,
            get_sum=False
        )

        simulate_response_carryover = robyn_object.fx_objective(
            x=carryover_vec.mean(),
            coeff=eval_list["coefs_eval"][i],
            alpha=eval_list["alphas_eval"][f"{i}_alphas"],
            inflexion=eval_list["inflexions_eval"][f"{i}_gammas"],
            x_hist_carryover=0,
            get_sum=False
        )

        plotDT_scurve.append({
            "channel": i,
            "spend": simulate_spend,
            "mean_carryover": carryover_vec.mean(),
            "carryover_response": simulate_response_carryover,
            "total_response": simulate_response
        })

        dt_optimOutScurve.loc[i, "response"] = dt_optimOutScurve.loc[i, "response"] + simulate_response_carryover

    # Convert plotDT_scurve to a pandas DataFrame
    plotDT_scurve = pd.DataFrame(plotDT_scurve)

    # Rename columns in mainPoints
    mainPoints = dt_optimOutScurve.rename(columns={"response": "response_point", "spend": "spend_point", "channels": "channel"})

    # Filter out Carryover rows from mainPoints
    temp_caov = mainPoints[mainPoints["type"] == "Carryover"]

    # Calculate mean spend and ROI for each channel
    mainPoints["mean_spend"] = mainPoints["spend_point"] - temp_caov["spend_point"]
    mainPoints.loc[mainPoints["type"] == "Carryover", "mean_spend"] = mainPoints.loc[mainPoints["type"] == "Carryover", "spend_point"]
    mainPoints["roi_mean"] = mainPoints["response_point"] / mainPoints["mean_spend"]

    # Calculate marginal response, ROI, and CPA for each channel
    mresp_caov = mainPoints[mainPoints["type"] == "Carryover"]["response_point"]
    mresp_init = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[2]]["response_point"] - mresp_caov
    mresp_b = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[3]]["response_point"] - mresp_caov
    mresp_unb = mainPoints[mainPoints["type"] == levels(mainPoints["type"])[4]]["response_point"] - mresp_caov
    mainPoints["marginal_response"] = [mresp_init, mresp_b, mresp_unb] + [0] * len(mresp_init)
    mainPoints["roi_marginal"] = mainPoints["marginal_response"] / mainPoints["mean_spend"]
    mainPoints["cpa_marginal"] = mainPoints["mean_spend"] / mainPoints["marginal_response"]

    # Set export directory
    if export:
        if json_file is None and plot_folder is not None:
            if plot_folder_sub is None:
                plot_folder_sub = basename(OutputCollect["plot_folder"])
            plot_folder = gsub("//+", "/", paste0(plot_folder, "/", plot_folder_sub, "/"))
        else:
            plot_folder = gsub("//+", "/", paste0(OutputCollect["plot_folder"], "/"))

        if not dir.exists(plot_folder):
            print("Creating directory for allocator: ", plot_folder)
            dir.create(plot_folder)

        # Export results into CSV
        export_dt_optimOut = dt_optimOut
        if dep_var_type == "conversion":
            export_dt_optimOut.columns = gsub("Roi", "CPA", export_dt_optimOut.columns)
        write.csv(export_dt_optimOut, paste0(plot_folder, select_model, "_reallocated.csv"))

    # Generate plots
    if plots:
        plots = allocation_plots(InputCollect, OutputCollect,
                                  dt_optimOut,
                                  # filter(dt_optimOut, .data$channels %in% channel_for_allocation),
                                  select_model, scenario, eval_list,
                                  export, plot_folder, quiet)
    else:
        plots = None

    # Return output
    output = [dt_optimOut, mainPoints, nlsMod, plots, scenario, usecase, total_budget, zero_coef_channel, zero_constraint_channel, zero_spend_channel, ui]
    output = pd.DataFrame(output)
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
          ## f"{paste(sprintf(\n"
          ## f"- %s:\n"
          ## f"  Optimizable bound: [%s%%, %s%%],\n"
          ## f"  Initial spend share: %s%% -> Optimized bounded: %s%%\n"
          ## f"  Initial response share: %s%% -> Optimized bounded: %s%%\n"
          ## f"  Initial abs. mean spend: %s -> Optimized: %s [Delta = %s%%]",
          ## temp.channels,
          ##100 * temp.constr_low - 100,
          ##100 * temp.constr_up - 100,
          ##signif(100 * temp.initSpendShare, 3),
          ##signif(100 * temp.optmSpendShareUnit, 3),
          ##signif(100 * temp.initResponseUnitShare, 3),
          ##signif(100 * temp.optmResponseUnitShare, 3),
          ##formatNum(temp.initSpendUnit, 3, abbr=True),
          ##formatNum(temp.optmSpendUnit, 3, abbr=True),
          ##formatNum(100 * temp.optmSpendUnitDelta, signif=2)),
          ##collapse="\n  ")}
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

def eval_f(X, target_value):
    """
    Evaluate the objective function, gradient, and objective channel for optimization.

    Args:
        X (array-like): Input values.
        target_value: Target value.

    Returns:
        dict: Dictionary containing the objective, gradient, and objective channel.
    """
    eval_list = getOption("ROBYN_TEMP")
    coefs_eval = eval_list["coefs_eval"]
    alphas_eval = eval_list["alphas_eval"]
    inflexions_eval = eval_list["inflexions_eval"]
    hist_carryover_eval = eval_list["hist_carryover_eval"]

    """ iLLAMA wasn't aware all these functions implemented, so commented out
    def fx_objective(x, coeff, alpha, inflexion, x_hist_carryover):
        # Implement the objective function here
        pass

    def fx_gradient(x, coeff, alpha, inflexion, x_hist_carryover):
        # Implement the gradient function here
        pass

    def fx_objective_channel(x, coeff, alpha, inflexion, x_hist_carryover):
        # Implement the objective channel function here
        pass
    """
    objective = -np.sum(np.ma.apply(
        fx_objective,
        x=X,
        coeff=coefs_eval,
        alpha=alphas_eval,
        inflexion=inflexions_eval,
        x_hist_carryover=hist_carryover_eval,
        SIMPLIFY=True
    ))

    gradient = np.ma.apply(
        fx_gradient,
        x=X,
        coeff=coefs_eval,
        alpha=alphas_eval,
        inflexion=inflexions_eval,
        x_hist_carryover=hist_carryover_eval,
        SIMPLIFY=True
    )

    objective_channel = np.ma.apply(
        fx_objective_channel,
        x=X,
        coeff=coefs_eval,
        alpha=alphas_eval,
        inflexion=inflexions_eval,
        x_hist_carryover=hist_carryover_eval,
        SIMPLIFY=True
    )

    optm = {"objective": objective, "gradient": gradient, "objective_channel": objective_channel}

    return optm


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
    """
    Calculate the objective value for a channel allocation.

    Parameters:
    x (array-like): The channel allocation.
    coeff (float): Coefficient used in the objective calculation.
    alpha (float): Alpha parameter used in the objective calculation.
    inflexion (float): Inflexion parameter used in the objective calculation.
    x_hist_carryover (array-like): Historical carryover values.

    Returns:
    float: The objective value for the given channel allocation.
    """
    # Apply Michaelis Menten model to scale spend to exposure
    xScaled = x
    # Adstock scales
    xAdstocked = x + np.mean(x_hist_carryover)
    xOut = -coeff * np.sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
    return xOut

def eval_g_eq(X, target_value):
    """
    Evaluate the equality constraint function for optimization.

    Parameters:
    X (array-like): The decision variables.
    target_value (float): The target value for the constraint.

    Returns:
    dict: A dictionary containing the constraint value and its gradient.
        The constraint value is the sum of the decision variables minus the total budget unit.
        The gradient is an array of ones with the same length as X.
    """
    eval_list = getOption("ROBYN_TEMP")
    constr = np.sum(X) - eval_list.total_budget_unit
    grad = np.ones(len(X))
    return {"constraints": constr, "jacobian": grad}

def eval_g_ineq(X, target_value):
    """
    Evaluate the inequality constraints for the optimization problem.

    Parameters:
    - X: A numpy array representing the decision variables.
    - target_value: The target value for the optimization problem.

    Returns:
    - A dictionary containing the constraints and their gradients.
    """
    eval_list = getOption("ROBYN_TEMP")
    constr = np.sum(X) - eval_list.total_budget_unit
    grad = np.ones(len(X))
    return {"constraints": constr, "jacobian": grad}

def eval_g_eq_effi(X, target_value):
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
    eval_list = getOption("ROBYN_TEMP")
    sum_response = np.sum(np.vectorize(fx_objective)(x=X, coeff=eval_list.coefs_eval, alpha=eval_list.alphas_eval, inflexion=eval_list.inflexions_eval, x_hist_carryover=eval_list.hist_carryover_eval))
    if target_value is None:
        if eval_list.dep_var_type == "conversion":
            constr = np.sum(X) - sum_response * eval_list.target_value
        else:
            constr = np.sum(X) - sum_response / eval_list.target_value
    else:
        if eval_list.dep_var_type == "conversion":
            constr = np.sum(X) - sum_response * target_value
        else:
            constr = np.sum(X) - sum_response / target_value
    grad = np.ones(len(X)) - np.vectorize(fx_gradient)(x=X, coeff=eval_list.coefs_eval, alpha=eval_list.alphas_eval, inflexion=eval_list.inflexions_eval, x_hist_carryover=eval_list.hist_carryover_eval)
    return {"constraints": constr, "jacobian": grad}


def eval_g_eq_effi(X, target_value):
    """
    Evaluate the equality constraints for the efficiency function.

    Parameters:
    X (array-like): The decision variables.
    target_value (float): The target value for the efficiency function.

    Returns:
    dict: A dictionary containing the constraints and the jacobian.
    """
    eval_list = getOption("ROBYN_TEMP")
    sum_response = np.sum(np.vectorize(fx_objective)(X, eval_list.coefs_eval, eval_list.alphas_eval, eval_list.inflexions_eval, eval_list.hist_carryover_eval))
    if target_value is None:
        if eval_list.dep_var_type == "conversion":
            constr = np.sum(X) - sum_response * eval_list.target_value
        else:
            constr = np.sum(X) - sum_response / eval_list.target_value
    else:
        if eval_list.dep_var_type == "conversion":
            constr = np.sum(X) - sum_response * target_value
        else:
            constr = np.sum(X) - sum_response / target_value
    grad = np.ones(len(X)) - np.vectorize(fx_gradient)(X, eval_list.coefs_eval, eval_list.alphas_eval, eval_list.inflexions_eval, eval_list.hist_carryover_eval)
    return {"constraints": constr, "jacobian": grad}


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
        gammas = gammas.values.flatten()
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

    inflexions_array = np.array(inflexions)

    coefs = dt_coef.set_index('rn').loc[mediaSpendSorted]['coefs']
    if not isinstance(coefs, float):
        coefs = coefs.drop_duplicates()

    if isinstance(coefs, pd.DataFrame):
        coefs_sorted = pd.Series(coefs.iloc[:, 0].values, index=coefs.index, name='DataFrame Column')
    elif isinstance(coefs, pd.Series):
        coefs_sorted = pd.Series(coefs.values, index=coefs.index, name=coefs.name)
    elif isinstance(coefs, np.ndarray):
        coefs_sorted = pd.Series(coefs, index=range(len(coefs)), name='Array')
    elif isinstance(coefs, list):
        coefs_sorted = pd.Series(coefs, index=range(len(coefs)), name='List')
    else:
        coefs_sorted = pd.Series(coefs, index=dt_coef.index, name=dt_coef['rn'].values[0])

    return {'alphas': alphas, 'inflexions': inflexions, 'coefs_sorted': coefs_sorted}
