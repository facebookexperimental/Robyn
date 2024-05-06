# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import matplotlib as plt

import pandas as pd
import numpy as np

def robyn_refresh(json_file=None, robyn_object=None, dt_input=None, dt_holidays=None, refresh_steps=4, refresh_mode='manual', refresh_iters=1000, refresh_trials=3, plot_folder=None, plot_pareto=True, version_prompt=False, export=True, calibration_input=None, objective_weights=None, **kwargs):
    refresh_control = True
    while refresh_control:
        # Check for NA values
        check_nas(dt_input)
        check_nas(dt_holidays)

        # Load initial model
        if not json_file:
            robyn = list()
            json = robyn_read(json_file, step=2, quiet=True)
            if not plot_folder:
                plot_folder = json['ExportedModel']['plot_folder']
            list_init = robyn_recreate(
                json_file=json_file,
                dt_input=dt_input,
                dt_holidays=dt_holidays,
                plot_folder=plot_folder,
                quiet=False,
                **kwargs
            )
            list_init['InputCollect']['refreshSourceID'] = json['ExportedModel']['select_model']
            chain_data = robyn_chain(json_file)
            list_init['InputCollect']['refreshChain'] = attr(chain_data, 'chain')
            list_init['InputCollect']['refreshDepth'] = refresh_depth = len(attr(chain_data, 'chain'))
            list_init['OutputCollect']['hyper_updated'] = json['ExportedModel']['hyper_updated']
            robyn = [list_init]
            refresh_counter = 1  # Dummy for now (legacy)
        else:
            robyn_imported = robyn_load(robyn_object)
            robyn = robyn_imported['Robyn']
            plot_folder = robyn_imported['objectPath']
            robyn_object = robyn_imported['robyn_object']
            refresh_counter = len(robyn) - sum(names(robyn) == 'refresh')
            refresh_depth = None  # Dummy for now (legacy)

        depth = refresh_counter if refresh_counter == 1 else c('listInit', paste0('listRefresh', 1, refresh_counter - 1))

        # Check rule of thumb: 50% of data shouldn't be new
        check_refresh_data(robyn, dt_input)

        # Get previous data
        if refresh_counter == 1:
            input_collect_rf = robyn['listInit']['InputCollect']
            list_output_prev = robyn['listInit']['OutputCollect']
            input_collect_rf['xDecompAggPrev'] = list_output_prev['xDecompAgg']
            if len(unique(robyn['listInit']['OutputCollect']['resultHypParam']['solID'])) > 1:
                stop("Run robyn_write() first to select and export any Robyn model")
        else:
            list_name = paste0('listRefresh', refresh_counter - 1)
            input_collect_rf = robyn[list_name]['InputCollect']
            list_output_prev = robyn[list_name]['OutputCollect']
            list_report_prev = robyn[list_name]['ReportCollect']
            # Model selection from previous build (new normalization range for error_score)
            if not 'error_score' in names(list_output_prev['resultHypParam']):
                list_output_prev['resultHypParam'] = pd.DataFrame(list_output_prev['resultHypParam'])
                list_output_prev['resultHypParam']['error_score'] = errors_scores(list_output_prev['resultHypParam'], ts_validation=list_output_prev['OutputModels']['ts_validation'], **kwargs)
            which_best_mod_rf = which.min(list_output_prev['resultHypParam']['error_score'])[1]
            list_output_prev['resultHypParam'] = list_output_prev['resultHypParam'][which_best_mod_rf, ]
            list_output_prev['xDecompAgg'] = list_output_prev['xDecompAgg'][which_best_mod_rf, ]
            list_output_prev['mediaVecCollect'] = list_output_prev['mediaVecCollect'][which_best_mod_rf, ]
            list_output_prev['xDecompVecCollect'] = list_output_prev['xDecompVecCollect'][which_best_mod_rf, ]

        # Update refresh model parameters
        input_collect_rf['refreshCounter'] = refresh_counter
        input_collect_rf['refresh_steps'] = refresh_steps
        if True:
            dt_input = pd.DataFrame(dt_input)
            date_input = check_datevar(dt_input, input_collect_rf['date_var'])
            dt_input = date_input['dt_input']  # sort date by ascending
            input_collect_rf['dt_input'] = dt_input
            dt_holidays = pd.DataFrame(dt_holidays)
            input_collect_rf['dt_holidays'] = dt_holidays

        # Load new data
        if True:
            dt_input = pd.DataFrame(dt_input)
            date_input = check_datevar(dt_input, input_collect_rf['date_var'])
            dt_input = date_input['dt_input']  # sort date by ascending
            input_collect_rf['dt_input'] = dt_input
            dt_holidays = pd.DataFrame(dt_holidays)
            input_collect_rf['dt_holidays'] = dt_holidays

        # Refresh rolling window
        if True:
            input_collect_rf['refreshAddedStart'] = pd.DataFrame(input_collect_rf['window_end']) + input_collect_rf['dayInterval']
            total_dates = pd.DataFrame(dt_input[input_collect_rf['date_var']])
            refresh_start = input_collect_rf['window_start'] = pd.DataFrame(input_collect_rf['window_start']) + input_collect_rf['dayInterval'] * refresh_steps
            refresh_start_which = input_collect_rf['rollingWindowStartWhich'] = which.min(abs(pd.DataFrame(total_dates - refresh_start)))
            refresh_end = input_collect_rf['window_end'] = pd.DataFrame(input_collect_rf['window_end']) + input_collect_rf['dayInterval'] * refresh_steps
            refresh_end_which = input_collect_rf['rollingWindowEndWhich'] = which.min(abs(pd.DataFrame(total_dates - refresh_end)))
            input_collect_rf['rollingWindowLength'] = refresh_end_which - refresh_start_which + 1

        if refresh_end > max(total_dates):
            raise ValueError("Not enough data for this refresh. Input data from date {} or later required".format(refresh_end))

        if json_file is not None and refresh_mode == "auto":
            print("Input 'refresh_mode' = 'auto' has been deprecated. Changed to 'manual'")
            refresh_mode = "manual"

        if refresh_mode == "manual":
            refresh_looper = 1
            print("Building refresh model #{} in {} mode".format(depth, refresh_mode))
            refresh_control = False
        else:
            refresh_looper = int(np.floor(np.abs(difftime(max(total_dates), refresh_end, units="days")) / (InputCollectRF.day_interval / refresh_steps)))
            print("Building refresh model #{} in {} mode. {} more to go...".format(depth, refresh_mode, refresh_looper))

        # Update refresh model parameters
        if calibration_input is not None:
            calibration_input = pd.concat([InputCollectRF.calibration_input, calibration_input], ignore_index=True)
            calibration_input = check_calibration(dt_input=InputCollectRF.dt_input, date_var=InputCollectRF.date_var, calibration_input=calibration_input, day_interval=InputCollectRF.day_interval, dep_var=InputCollectRF.dep_var, window_start=InputCollectRF.window_start, window_end=InputCollectRF.window_end, paid_media_spends=InputCollectRF.paid_media_spends, organic_vars=InputCollectRF.organic_vars)
            InputCollectRF.calibration_input = calibration_input

        # Refresh hyperparameter bounds
        InputCollectRF.hyperparameters = refresh_hyps(init_bounds=Robyn.list_init.OutputCollect.hyper_updated, list_output_prev, refresh_steps, rolling_window_length=InputCollectRF.rolling_window_length)

        # Feature engineering for refreshed data
        InputCollectRF = robyn_engineering(InputCollectRF, **kwargs)

        # Refresh model with adjusted decomp.rssd
        OutputModelsRF = robyn_run(InputCollect=InputCollectRF, iterations=refresh_iters, trials=refresh_trials, refresh=True, add_penalty_factor=list_output_prev["add_penalty_factor"], **kwargs)

        OutputCollectRF = robyn_outputs(InputCollectRF, OutputModelsRF, plot_folder=plot_folder, calibration_constraint=rf_cal_constr, export=export, plot_pareto=plot_pareto, objective_weights=objective_weights, **kwargs)

        # Select winner model for current refresh
        OutputCollectRF.result_hyp_param = OutputCollectRF.result_hyp_param.sort_values(by="error_score", ascending=False)

        best_mod = OutputCollectRF.result_hyp_param.iloc[0, 0]
        select_id = None

        while select_id is None or not select_id.isin(OutputCollectRF.all_solutions):
            if version_prompt:
                select_id = input("Input model ID to use for the refresh: ")
                print("Selected model ID: {} for refresh model #{} based on your input".format(select_id, depth))
                if not select_id.isin(OutputCollectRF.all_solutions):
                    print("Selected model ({}) NOT valid. Choose any of: {}".format(select_id, v2t(OutputCollectRF.all_solutions)))
            else:
                select_id = best_mod
                print("Selected model ID: {} for refresh model #{} based on the smallest combined normalized errors".format(select_id, depth))
        OutputCollectRF.select_id = select_id
        # Result collect & save
        these = ["result_hyp_param", "x_decomp_agg", "media_vec_collect", "x_decomp_vec_collect"]
        for tb in these:
            OutputCollectRF[tb] = OutputCollectRF[tb].assign(refresh_status=refresh_counter, best_mod_rf=select_id.isin(best_mod))


        # Create bestModRF and refreshStatus columns in listOutputPrev data.frames
        if refresh_counter == 1:
            for tb in these:
                list_output_prev[tb] = pd.concat([
                    list_output_prev[tb],
                    pd.DataFrame({'bestModRF': True, 'refreshStatus': 0})
                ])
                list_report_prev[tb] = pd.DataFrame({'mediaVecReport': list_output_prev[tb]['mediaVecCollect'], 'xDecompVecReport': list_output_prev[tb]['xDecompVecCollect']})
                names(list_report_prev[tb]) = [f'{name}Report' for name in names(list_report_prev[tb])]

        # Filter and bind rows for listReportPrev and listReportPrev$mediaVecReport
        list_report_prev['resultHypParamReport'] = pd.concat([
            list_report_prev['resultHypParamReport'],
            pd.DataFrame(filter(OutputCollectRF['resultHypParam'], bestModRF==True), columns=['solID', 'refreshStatus']) ##), one more ) added by ai, commented out
        ])
        list_report_prev['xDecompAggReport'] = pd.concat([
            list_report_prev['xDecompAggReport'],
            pd.DataFrame(filter(OutputCollectRF['xDecompAgg'], bestModRF==True), columns=['solID', 'refreshStatus']) ##), one more ) added by ai, commented out
        ])
        list_report_prev['mediaVecReport'] = pd.concat([
            list_report_prev['mediaVecReport'],
            pd.DataFrame(filter(OutputCollectRF['mediaVecCollect'], bestModRF==True), columns=['ds', 'refreshStatus']) ##), one more ) added by ai, commented out
        ])
        list_report_prev['xDecompVecReport'] = pd.concat([
            list_report_prev['xDecompVecReport'],
            pd.DataFrame(filter(OutputCollectRF['xDecompVecCollect'], bestModRF==True), columns=['ds', 'refreshStatus']) ##), one more ) added by ai, commented out
        ])

        # Update listNameUpdate and Robyn with new data
        list_name_update = f'listRefresh{refresh_counter}'
        Robyn[list_name_update] = pd.DataFrame({
            'InputCollect': InputCollectRF,
            'OutputCollect': OutputCollectRF,
            'ReportCollect': list_report_prev
        })

        # Plotting
        if json_file is not None:
            json_temp = robyn_write(InputCollectRF, OutputCollectRF, select_model=selectID, export=True, quiet=True)
            plots = refresh_plots_json(OutputCollectRF, json_file=attr(json_temp, 'json_file'), export=True)
        else:
            plots = refresh_plots(InputCollectRF, OutputCollectRF, ReportCollect, export=True)

        # Export data
        if export:
            message(f'>> Exporting refresh CSVs into directory...')
            pd.write_csv(resultHypParamReport, f'{plot_folder}report_hyperparameters.csv')
            pd.write_csv(xDecompAggReport, f'{plot_folder}report_aggregated.csv')
            pd.write_csv(mediaVecReport, f'{plot_folder}report_media_transform_matrix.csv')
            pd.write_csv(xDecompVecReport, f'{plot_folder}report_alldecomp_matrix.csv')

        if refresh_counter == 0:
            refresh_control = False
            message(f'Reached maximum available date. No further refresh possible')


    ## Indentation was wrong, manually corrected this part.
    # Save some parameters to print
    """
    robyn['refresh'] = list(
        selectIDs=report_collect['selectIDs'],
        refresh_steps=refresh_steps,
        refresh_mode=refresh_mode,
        refresh_trials=refresh_trials,
        refresh_iters=refresh_iters,
        plots=plots
    )
        # Save Robyn object and print parameters
    Robyn['refresh'] = pd.DataFrame({
        'selectIDs': ReportCollect['selectIDs'],
        'refresh_steps': refresh_steps,
        'refresh_mode': refresh_mode,
        'refresh_trials': refresh_trials,
        'refresh_iters': refresh_iters,
        'plots': plots
    })
    """

    ##Partially wrong interpretation since, saving R models is different in Python
    # Save Robyn object locally
    robyn = robyn[robyn.keys()]
    ## class robyn(robyn_refresh, class robyn):
    ##    pass

    ##    if not json_file:
    ##        message('>> Exporting results: ', robyn_object)
    ##        saveRDS(robyn, file=robyn_object)
    ##    else:
    ##        robyn_write(input_collect_rf, output_collect_rf, select_model=selectID, **kwargs)

    return(invisible(robyn))


##MetaMate
def print_robyn_refresh(x, *args):
    top_models = x.refresh.selectIDs
    top_models = [f"{id} ({i})" for i, id in enumerate(top_models)]
    print("Refresh Models: {}".format(len(top_models)))
    print("Mode: {}".format(x.refresh.refresh_mode))
    print("Steps: {}".format(x.refresh.refresh_steps))
    print("Trials: {}".format(x.refresh.refresh_trials))
    print("Iterations: {}".format(x.refresh.refresh_iters))
    print("Models (IDs):\n{}".format(", ".join(top_models)))


##MetaMate
def plot_robyn_refresh(x, *args):
    plt.plot((x.refresh.plots[0] / x.refresh.plots[1]), *args)

##MetaMate
def refresh_hyps(initBounds, listOutputPrev, refresh_steps, rollingWindowLength):
    initBoundsDis = [x[1] - x[0] if len(x) == 2 else 0 for x in initBounds]
    newBoundsFreedom = refresh_steps / rollingWindowLength
    print(">>> New bounds freedom:", round(newBoundsFreedom * 100, 2), "%")
    hyper_updated_prev = listOutputPrev.hyper_updated
    hypNames = listOutputPrev.resultHypParam.columns
    resultHypParam = pd.DataFrame(listOutputPrev.resultHypParam)
    for h in range(len(hypNames)):
        hn = hypNames[h]
        getHyp = resultHypParam[hn].values[0]
        getDis = initBoundsDis[hn]
        if hn == "lambda":
            lambda_max = resultHypParam["lambda_max"].unique()
            lambda_min = lambda_max * 0.0001
            getHyp = getHyp / (lambda_max - lambda_min)
        getRange = initBounds[hn][0]
        if len(getRange) == 2:
            newLowB = getHyp - getDis * newBoundsFreedom
            if newLowB < getRange[0]:
                newLowB = getRange[0]
            newUpB = getHyp + getDis * newBoundsFreedom
            if newUpB > getRange[1]:
                newUpB = getRange[1]
            newBounds = [newLowB, newUpB]
            hyper_updated_prev[hn][0] = newBounds
        else:
            hyper_updated_prev[hn][0] = getRange
    return hyper_updated_prev
