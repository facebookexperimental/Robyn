# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from .inputs import robyn_inputs
#from .model import robyn_run
from .transformation import saturation_hill, transform_adstock

from .checks import check_metric_dates, check_metric_type, check_metric_value
import seaborn as sns

plt.ioff()

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
        if InputCollect is None:
            InputCollect = robyn_inputs(json_file=json_file)
        if OutputCollect is None:
            OutputCollect = robyn_run(InputCollect=InputCollect, json_file=json_file, export=False, quiet=quiet)
        if dt_hyppar is None:
            dt_hyppar = OutputCollect.resultHypParam
        if dt_coef is None:
            dt_coef = OutputCollect.xDecompAgg
    else:
        if robyn_object:
            if not os.path.exists(robyn_object):
                raise FileNotFoundError(f"File does not exist or is somewhere else. Check: {robyn_object}")
            else:
                Robyn = readRDS(robyn_object)  # Assume readRDS is a function you have defined to read RDS files in Python
                objectPath = os.path.dirname(robyn_object)
                objectName = re.sub(r'\..*$', '', os.path.basename(robyn_object))

            select_build_all = range(len(Robyn))
            if select_build is None:
                select_build = max(select_build_all)
                if not quiet and len(select_build_all) > 1:
                    print(f"Using latest model: {'initial model' if select_build == 0 else f'refresh model #{select_build}'} for the response function. Use parameter 'select_build' to specify which run to use")

            if select_build not in select_build_all or not isinstance(select_build, int):
                raise ValueError(f"'select_build' must be one value of {', '.join(map(str, select_build_all))}")

            listName = "listInit" if select_build == 0 else f"listRefresh{select_build}"
            InputCollect = Robyn[listName]["InputCollect"]
            OutputCollect = Robyn[listName]["OutputCollect"]
            dt_hyppar = OutputCollect.resultHypParam
            dt_coef = OutputCollect.xDecompAgg
        else:
            # Try to get some pre-filled values
            if dt_hyppar is None:
                dt_hyppar = OutputCollect['resultHypParam']
            if dt_coef is None:
                dt_coef = OutputCollect['xDecompAgg']
            if any(x is None for x in [dt_hyppar, dt_coef, InputCollect, OutputCollect]):
                raise ValueError("When 'robyn_object' is not provided, 'InputCollect' & 'OutputCollect' must be provided")

    # Prep environment
    if True:
        dt_input = InputCollect["robyn_inputs"]["dt_input"]
        startRW = InputCollect["robyn_inputs"]["rollingWindowStartWhich"]
        endRW = InputCollect["robyn_inputs"]["rollingWindowEndWhich"]
        adstock = InputCollect["robyn_inputs"]["adstock"]
        spendExpoMod = InputCollect["robyn_inputs"]["modNLS"]["results"]
        paid_media_vars = InputCollect["robyn_inputs"]["paid_media_vars"]
        paid_media_spends = InputCollect["robyn_inputs"]["paid_media_spends"]
        exposure_vars = InputCollect["robyn_inputs"]["exposure_vars"]
        organic_vars = InputCollect["robyn_inputs"]["organic_vars"]
        allSolutions = dt_hyppar['solID'].unique()
        dayInterval = InputCollect["robyn_inputs"]["dayInterval"]

    # Check select_model
    if not select_model or select_model not in allSolutions:
        raise ValueError(f"Input 'select_model' must be one of these values: {', '.join(allSolutions)}")

    # Get use case based on inputs
    usecase = which_usecase(metric_value, date_range)

    # Check inputs with usecases
    metric_type = check_metric_type(metric_name, paid_media_spends, paid_media_vars, exposure_vars, organic_vars)
    all_dates = dt_input['DATE'].tolist()
    all_values = dt_input[metric_name].tolist()

    if usecase == "all_historical_vec":
        # Calculate dates and values for all historical data
        ds_list = check_metric_dates("all", all_dates[0:endRW], dayInterval, quiet)
        metric_value = None
    elif usecase == "unit_metric_default_last_n":
        # Calculate dates and values for last n days
        ds_list = check_metric_dates("last_{}".format(len(metric_value)), all_dates[0:endRW], dayInterval, quiet)
    else:
        # Calculate dates and values for specified date range
        ds_list = check_metric_dates(date_range, all_dates[0:endRW], dayInterval, quiet)

    val_list = check_metric_value(metric_value, metric_name, all_values, ds_list['metric_loc'])
    date_range_updated = ds_list['date_range_updated']
    metric_value_updated = val_list['metric_value_updated']
    all_values_updated = val_list['all_values_updated']

    # Transform exposure to spend when necessary
    if metric_type == "exposure":
        get_spend_name = paid_media_spends[np.where(paid_media_vars == metric_name)]
        expo_vec = dt_input[metric_name][[1]]
        # Use non-0 mean as marginal level if metric_value not provided
        if metric_value is None:
            metric_value = np.mean(expo_vec[startRW:endRW][expo_vec[startRW:endRW] > 0])
            if not quiet:
                print("Input 'metric_value' not provided. Using mean of ", metric_name, " instead")

        # Fit spend to exposure
        spend_vec = dt_input[get_spend_name][[1]]
        temp = filter(spendExpoMod, dt_input['channel'] == metric_name)
        nls_select = temp['rsq_nls'] > temp['rsq_lm']
        if nls_select:
            Vmax = spendExpoMod['Vmax'][spendExpoMod['channel'] == metric_name]
            Km = spendExpoMod['Km'][spendExpoMod['channel'] == metric_name]
            input_immediate = mic_men(x=metric_value_updated, Vmax=Vmax, Km=Km, reverse=True)
        else:
            coef_lm = spendExpoMod['coef_lm'][spendExpoMod['channel'] == metric_name]
            input_immediate = metric_value_updated / coef_lm

        all_values_updated[ds_list['metric_loc']] = input_immediate
        hpm_name = get_spend_name
    else:
        input_immediate = metric_value_updated
        hpm_name = metric_name

    # Adstocking original
    # media_vec_origin = dt_input[metric_name][[1]]
    media_vec_origin = dt_input[metric_name].tolist()

    dt_hyppar = sanitize_suffixes(dt_hyppar)

    theta = scale = shape = None
    if adstock == "geometric":
        theta_column_name = f"{hpm_name}_thetas"
        theta = dt_hyppar[dt_hyppar['solID'] == select_model][theta_column_name].iloc[0]
        # theta = dt_hyppar[dt_hyppar['solID'] == select_model][["{}{}".format(hpm_name, "_thetas")]][[1]]
    elif re.search("weibull", adstock):
        shape_column_name = f"{hpm_name}_shapes"
        shape = dt_hyppar[dt_hyppar['solID'] == select_model][shape_column_name].iloc[0]

        scale_column_name = f"{hpm_name}_scales"
        scale = dt_hyppar[dt_hyppar['solID'] == select_model][scale_column_name].iloc[0]

    x_list = transform_adstock(media_vec_origin, adstock, theta=theta, shape=shape, scale=scale)
    m_adstocked = x_list['x_decayed']

    # Adstocking simulation
    x_list_sim = transform_adstock(all_values_updated, adstock, theta=theta, shape=shape, scale=scale)
    media_vec_sim = x_list_sim['x_decayed']
    media_vec_sim_imme = True if adstock == "weibull_pdf" else x_list_sim['x']
    input_total = media_vec_sim[ds_list['metric_loc']]
    input_immediate = media_vec_sim_imme[ds_list['metric_loc']]
    input_carryover = input_total - input_immediate

    # Saturation
    m_adstockedRW = m_adstocked[startRW:endRW]
    alpha_column_name = f"{hpm_name}_alphas"
    alpha = dt_hyppar[dt_hyppar['solID'] == select_model][alpha_column_name].iloc[0]

    gamma_column_name = f"{hpm_name}_gammas"
    gamma = dt_hyppar[dt_hyppar['solID'] == select_model][gamma_column_name].iloc[0]
    # alpha = head(dt_hyppar[dt_hyppar['solID'] == select_model, ][["{}{}".format(hpm_name, "_alphas")]], 1)
    # gamma = head(dt_hyppar[dt_hyppar['solID'] == select_model, ][["{}{}".format(hpm_name, "_gammas")]], 1)
    if usecase == "all_historical_vec":
        metric_saturated_total = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
        metric_saturated_carryover = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
    else:
        metric_saturated_total = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma, x_marginal=input_total)
        metric_saturated_carryover = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma, x_marginal=input_carryover)

    metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

    # Decomp
    coeff = dt_coef[(dt_coef['solID'] == select_model) & (dt_coef['rn'] == hpm_name)][['coefs']]

    # metric_saturated_total = metric_saturated_total.reset_index(drop=True)


    coeff_value = coeff.iloc[0]['coefs']
    m_saturated = saturation_hill(x=m_adstockedRW, alpha=alpha, gamma=gamma)
    m_resposne = m_saturated * coeff_value

    response_total = metric_saturated_total * coeff_value
    response_carryover = metric_saturated_carryover * coeff_value
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
    # p_res = plt.figure(figsize=(12, 6))
    # sns.lineplot(x='metric', y='response', data=dt_line, color="steelblue")
    # sns.scatterplot(x='input', y='output', data=dt_point, size=3)
    # sns.scatterplot(x='input', y='output', data=dt_point_caov, size=3, marker=8)
    # sns.scatterplot(x='input', y='output', data=dt_point_imme, size=3)
    # plt.title(f"Saturation curve of {metric_name}")
    # plt.text(0.5, 0.95, f"Carryover* Response: {response_carryover} @ Input {input_carryover} \nImmediate Response: {response_immediate} @ Input {input_immediate} \n Total (C+I) Response: {response_total} @ Input {input_total}")
    # plt.xlabel('Input')
    # plt.ylabel('Response')
    # plt.text(0.5, 0.05, f"Response period: {date_range_updated[0]} to {date_range_updated[-1]} [{len(date_range_updated)} periods]")
    # plt.show()

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
        # 'plot': p_res
        'plot': None
    }
    return ret

def sanitize_suffixes(df):
    columns = df.columns
    to_drop = []
    rename_map = {}

    for col in columns:
        if col.endswith('_x'):
            base_name = col[:-2]  # Remove '_x'
            y_col = base_name + '_y'
            if y_col in columns:
                to_drop.append(y_col)
            rename_map[col] = base_name

    df = df.drop(columns=to_drop)
    df = df.rename(columns=rename_map)
    return df

def which_usecase(metric_value, date_range):
    usecase = None

    if pd.isnull(metric_value) and pd.isnull(date_range):
        usecase = "all_historical_vec"
    elif pd.isnull(metric_value) and not pd.isnull(date_range):
        usecase = "selected_historical_vec"
    elif (isinstance(metric_value, str) and pd.isnull(date_range)) or (isinstance(metric_value, list) and len(metric_value) == 1 and pd.isnull(date_range)):
        usecase = "total_metric_default_range"
    elif (isinstance(metric_value, str) and not pd.isnull(date_range)) or (isinstance(metric_value, list) and len(metric_value) == 1 and not pd.isnull(date_range)):
        usecase = "total_metric_selected_range"
    elif isinstance(metric_value, list) and len(metric_value) > 1 and pd.isnull(date_range):
        usecase = "unit_metric_default_last_n"
    elif isinstance(metric_value, list) and len(metric_value) > 1 and not pd.isnull(date_range):
        usecase = "unit_metric_selected_dates"

    if date_range is not None and date_range == "all":
        usecase = "all_historical_vec"

    return usecase
