import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def robyn_response(InputCollect=None,
                   OutputCollect=None,
                   json_file=None,
                   robyn_object=None,
                   select_build=None,
                   select_model=None,
                   metric_name=None,
                   metric_value=None,
                   date_range=None,
                   dt_hyppar=None,
                   dt_coef=None,
                   quiet=False):
    # Get input
    if json_file:
        # Use previously exported model using json_file
        if not InputCollect:
            InputCollect = robyn_inputs(json_file=json_file, ...)
        if not OutputCollect:
            OutputCollect = robyn_run(InputCollect=InputCollect, json_file=json_file, export=False, quiet=quiet, ...)
        dt_hyppar = OutputCollect.resultHypParam
        dt_coef = OutputCollect.xDecompAgg
    else:
        # Try to get some pre-filled values
        if not dt_hyppar:
            dt_hyppar = OutputCollect.resultHypParam
        if not dt_coef:
            dt_coef = OutputCollect.xDecompAgg
        if any(dt_hyppar is None, dt_coef is None, InputCollect is None, OutputCollect is None):
            raise ValueError("When 'robyn_object' is not provided, 'InputCollect' & 'OutputCollect' must be provided")

    # Prep environment
    if True:
        dt_input = InputCollect.dt_input
        startRW = InputCollect.rollingWindowStartWhich
        endRW = InputCollect.rollingWindowEndWhich
        adstock = InputCollect.adstock
        spendExpoMod = InputCollect.modNLS.results
        paid_media_vars = InputCollect.paid_media_vars
        paid_media_spends = InputCollect.paid_media_spends
        exposure_vars = InputCollect.exposure_vars
        organic_vars = InputCollect.organic_vars
        allSolutions = unique(dt_hyppar.solID)
        dayInterval = InputCollect.dayInterval

    # Check select_model
    if not select_model or select_model not in allSolutions:
        raise ValueError(f"Input 'select_model' must be one of these values: {', '.join(allSolutions)}")

    # Get use case based on inputs
    usecase = which_usecase(metric_value, date_range)

    # Check inputs with usecases
    metric_type = check_metric_type(metric_name, paid_media_spends, paid_media_vars, exposure_vars, organic_vars)
    all_dates = pull(dt_input, InputCollect$date_var)
    all_values = pull(dt_input, metric_name)

    if usecase == "all_historical_vec":
        # Calculate dates and values for all historical data
        ds_list = check_metric_dates(date_range="all", all_dates[1:endRW], dayInterval, quiet, ...)
        metric_value = None
        val_list = check_metric_value(metric_value, metric_name, all_values, ds_list$metric_loc)
    elif usecase == "unit_metric_default_last_n":
        # Calculate dates and values for last n days
        ds_list = check_metric_dates(date_range=paste0("last_", length(metric_value)), all_dates[1:endRW], dayInterval, quiet, ...)
        val_list = check_metric_value(metric_value, metric_name, all_values, ds_list$metric_loc)
    else:
        # Calculate dates and values for specified date range
        ds_list = check_metric_dates(date_range, all_dates[1:endRW], dayInterval, quiet, ...)

    # Transform exposure to spend when necessary
    if metric_type == "exposure":
        get_spend_name = paid_media_spends[which(paid_media_vars == metric_name)]
        expo_vec = dt_input[, metric_name][[1]]
        # Use non-0 mean as marginal level if metric_value not provided
        if is.null(metric_value):
            metric_value = mean(expo_vec[startRW:endRW][expo_vec[startRW:endRW] > 0])
            if not quiet:
                print("Input 'metric_value' not provided. Using mean of ", metric_name, " instead")

        # Fit spend to exposure
        spend_vec = dt_input[, get_spend_name][[1]]
        temp = filter(spendExpoMod, .data$channel == metric_name)
        nls_select = temp$rsq_nls > temp$rsq_lm
        if nls_select:
            Vmax = spendExpoMod$Vmax[spendExpoMod$channel == metric_name]
            Km = spendExpoMod$Km[spendExpoMod$channel == metric_name]
            input_immediate = mic_men(x=metric_value_updated, Vmax=Vmax, Km=Km, reverse=True)
        else:
            coef_lm = spendExpoMod$coef_lm[spendExpoMod$channel == metric_name]
            input_immediate = metric_value_updated / coef_lm

        all_values_updated[ds_list$metric_loc] = input_immediate
        hpm_name = get_spend_name

    # Adstocking original
    media_vec_origin = dt_input[, metric_name][[1]]
    theta = scale = shape = NULL
    if adstock == "geometric":
        theta = dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_thetas")]][[1]]
    elif grepl("weibull", adstock):
        shape = dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]][[1]]
        scale = dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]][[1]]

    x_list = transform_adstock(media_vec_origin, adstock, theta=theta, shape=shape, scale=scale)
    m_adstocked = x_list$x_decayed
    # net_carryover_ref = m_adstocked - media_vec_origin

    # Adstocking simulation
    x_list_sim = transform_adstock(all_values_updated, adstock, theta=theta, shape=shape, scale=scale)
    media_vec_sim = x_list_sim$x_decayed
    media_vec_sim_imme = if adstock == "weibull_pdf" else x_list_sim$x
    input_total = media_vec_sim[ds_list$metric_loc]
    input_immediate = media_vec_sim_imme[ds_list$metric_loc]
    input_carryover = input_total - input_immediate

    # Saturation
    m_adstockedRW = m_adstocked[startRW:endRW]
    alpha = head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_alphas")]], 1)
    gamma = head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_gammas")]], 1)
    if usecase == "all_historical_vec":
        metric_saturated_total = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
        metric_saturated_carryover = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
    else:
        metric_saturated_total = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma, x_marginal=input_total)
        metric_saturated_carryover = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma, x_marginal=input_carryover)

    metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

    # Decomp
    coeff = dt_coef[dt_coef['solID'] == select_model & dt_coef['rn'] == hpm_name, ][['coef']]
    m_saturated = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
    m_resposne = m_saturated * coeff
    response_total = np.numeric(metric_saturated_total * coeff)
    response_carryover = np.numeric(metric_saturated_carryover * coeff)
    response_immediate = response_total - response_carryover

    dt_line = pd.DataFrame({'metric': m_adstockedRW, 'response': m_resposne, 'channel': metric_name})

    if usecase == "all_historical_vec":
        dt_point = pd.DataFrame({'input': input_total[startRW:endRW], 'output': response_total, 'ds': date_range_updated[startRW:endRW]})
        dt_point_caov = pd.DataFrame({'input': input_carryover[startRW:endRW], 'output': response_carryover})
        dt_point_imme = pd.DataFrame({'input': input_immediate[startRW:endRW], 'output': response_immediate})
    else:
        dt_point = pd.DataFrame({'input': input_total, 'output': response_total, 'ds': date_range_updated})
        dt_point_caov = pd.DataFrame({'input': input_carryover, 'output': response_carryover})
        dt_point_imme = pd.DataFrame({'input': input_immediate, 'output': response_immediate})

    # Plot optimal response
    p_res = plt.figure(figsize=(12, 6))
    sns.lineplot(x='metric', y='response', data=dt_line)
    sns.pointplot(x='input', y='output', data=dt_point, size=3)
    sns.pointplot(x='input', y='output', data=dt_point_caov, size=3, shape=8)
    sns.pointplot(x='input', y='output', data=dt_point_imme, size=3)
    plt.title(f"Saturation curve of {metric_name}")
    plt.subtitle(f"Carryover* Response: {response_carryover} @ Input {input_carryover}")
    plt.subtitle(f"Immediate Response: {response_immediate} @ Input {input_immediate}")
    plt.subtitle(f"Total (C+I) Response: {response_total} @ Input {input_total}")
    plt.xlabel('Input')
    plt.ylabel('Response')
    plt.caption(f"Response period: {date_range_updated}")
    plt.show()

    ret = {
        'metric_name': metric_name,
        'date': date_range_updated,
        'input_total': input_total,
        'input_carryover': input_carryover,
        'input_immediate': input_immediate,
        'response_total': response_total,
        'response_carryover': response_carryover,
        'response_immediate': response_immediate,
        'usecase': usecase,
        'plot': p_res
    }
    return ret



def which_usecase(metric_value, date_range):
    usecase = None

    if pd.isnull(metric_value) and pd.isnull(date_range):
        usecase = "all_historical_vec"
    elif pd.isnull(metric_value) and not pd.isnull(date_range):
        usecase = "selected_historical_vec"
    elif len(metric_value) == 1 and pd.isnull(date_range):
        usecase = "total_metric_default_range"
    elif len(metric_value) == 1 and not pd.isnull(date_range):
        usecase = "total_metric_selected_range"
    elif len(metric_value) > 1 and pd.isnull(date_range):
        usecase = "unit_metric_default_last_n"
    elif len(metric_value) > 1 and not pd.isnull(date_range):
        usecase = "unit_metric_selected_dates"

    if not pd.isnull(date_range):
        if len(date_range) == 1 and date_range[0] == "all":
            usecase = "all_historical_vec"

    return usecase
