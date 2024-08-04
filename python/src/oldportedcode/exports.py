# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import warnings
import pandas as pd
import numpy as np
# from the second method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os



def robyn_save(input_collect, output_collect, robyn_object=None, select_model=None, dir=None, quiet=False):
    warnings.warn("Function robyn_save() is not supported anymore. Please migrate to robyn_write() and robyn_read()")
    check_robyn_name(robyn_object, quiet)
    if select_model is None:
        select_model = output_collect['selectID']
    if not select_model in output_collect['allSolutions']:
        raise ValueError(f"Input 'select_model' must be one of these values: {', '.join(output_collect['allSolutions'])}")

    # Export as JSON file
    json = robyn_write(input_collect, output_collect, select_model, quiet=quiet)

    # Summarize results
    summary = filter(output_collect['xDecompAgg'], solID==select_model) \
        .select(variable=['rn', 'coef', 'decomp', 'total_spend', 'mean_non0_spend']) \
        .rename(columns={'variable': 'hyperparameter'}) \
        .drop(columns=['solID'])

    # Nice and tidy table format for hyper-parameters
    hyps = filter(output_collect['resultHypParam'], solID==select_model) \
        .select(contains(HYPS_NAMES)) \
        .gather() \
        .separate(channel=['channel', 'none'], sep=r'^.*_', remove=FALSE) \
        .mutate(hyperparameter=lambda x: re.sub(r'^.*_', '', x.channel)) \
        .select(channel, hyperparameter, value) \
        .spread(key='hyperparameter', value='value')

    # Collect other outputs
    values = output_collect.drop(columns=['allSolutions', 'hyper_fixed', 'plot_folder'])
    values['robyn_object'] = robyn_object
    values['select_model'] = select_model
    values['summary'] = summary
    values['errors'] = json['ExportedModel']['errors']
    values['hyper_df'] = hyps
    values['hyper_updated'] = output_collect['hyper_updated']
    values['window'] = [input_collect['window_start'], input_collect['window_end']]
    values['periods'] = input_collect['rollingWindowLength']
    values['interval'] = input_collect['intervalType']
    values['adstock'] = input_collect['adstock']
    values['plot'] = robyn_onepagers(input_collect, output_collect, select_model, quiet=quiet, export=FALSE)

    # Append other outputs to the list
    output = [values]

    # Rename columns
    if input_collect['dep_var_type'] == 'conversion':
        colnames(output[0]['summary']) = re.sub(r'roi_', 'cpa_', colnames(output[0]['summary']))

    # Set class
    class_(output) <- c('robyn_save', class(output))

    # Overwrite existing file if necessary
    if robyn_object is not None:
        if file.exists(robyn_object):
            if not quiet:
                answer = askYesNo(f'{robyn_object} already exists. Are you certain to overwrite it?')
            else:
                answer = True
            if answer is False or is.na(answer):
                message(f'Stopped export to avoid overwriting {robyn_object}')
                return output
            else:
                saveRDS(output, file=robyn_object)
                if not quiet: message(f'Exported results: {robyn_object}')
        else:
            saveRDS(output, file=robyn_object)
            if not quiet: message(f'Exported results: {robyn_object}')

    return output



def robyn_save(x, *args):
    print("Exported file: {x['robyn_object']}")
    print("Exported model: {x['select_model']}")
    print("Window: {x['window'][1]} to {x['window'][2]} ({x['periods']} {x['interval']}s)")

    errors = pd.DataFrame({'errors': [f"R2 ({x['ExportedModel']['ts_validation']}): {x['errors']['rsq_train']}, {x['errors']['rsq_test']}"
                             "| NRMSE = {x['errors']['nrmse']}"
                             "| DECOMP.RSSD = {x['errors']['decomp.rssd']}"
                             "| MAPE = {x['errors']['mape']}"
                            ]})
    print(errors)

    print("Summary Values on Selected Model:")
    print(x['summary'].mutate(decomp=lambda x: 100*x['decomp']).replace(np.inf, 0).replace(np.nan, "-").astype({'decomp': float}))

    print("Hyper-parameters:")
    print(pd.DataFrame({'Adstock': x['adstock']}))
    print(x['hyper_df'])



def plot_robyn_save(x, *args, **kwargs):
    plt.plot(x['plot'][0], *args, **kwargs)


def robyn_load(robyn_object, select_build=None, quiet=False):
    """
    Load a Robyn model from a saved RDS file.
    """
    if isinstance(robyn_object, str) or isinstance(robyn_object, list):
        # If the input is a string or a list, we assume it's a file path or a list of Robyn objects
        Robyn = read_rds(robyn_object)
        object_path = os.path.dirname(robyn_object)
    else:
        # If the input is not a string or a list, we assume it's a Robyn object
        Robyn = robyn_object
        object_path = None

    select_build_all = range(len(Robyn))
    if select_build is None:
        select_build = max(select_build_all)
        if not quiet:
            print(f"Loaded Model: {select_build}")

    if select_build not in select_build_all or len(select_build) != 1:
        raise ValueError(f"Input 'select_build' must be one value of {select_build_all}")

    list_name = "listInit" if select_build == 0 else f"listRefresh{select_build}"
    InputCollect = Robyn[list_name]["InputCollect"]
    OutputCollect = Robyn[list_name]["OutputCollect"]
    select_model = OutputCollect["selectID"]

    output = {
        "Robyn": Robyn,
        "InputCollect": InputCollect,
        "OutputCollect": OutputCollect,
        "select_model": select_model,
        "objectPath": object_path,
        "robyn_object": robyn_object
    }
    return output
