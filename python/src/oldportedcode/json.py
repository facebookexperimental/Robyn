# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import os
import json
import pandas as pd
import numpy as np
import re
import warnings

def robyn_write(InputCollect, OutputCollect=None, select_model=None, dir=None, export=True, quiet=False, pareto_df=None):
    """
    A function that takes in various inputs and exports a Robyn model.

    Parameters:
    - InputCollect: A robyn_inputs object.
    - OutputCollect: A robyn_outputs object (optional).
    - select_model: A string representing the selected model (optional).
    - dir: A valid directory path (optional).
    - export: A boolean indicating whether to export the model (default is True).
    - quiet: A boolean indicating whether to print export messages (default is False).
    - pareto_df: A DataFrame containing 'solID' and 'cluster' columns (optional).

    Returns:
    - ret: A dictionary containing the exported model data.
    """
    # Checks
    if not isinstance(InputCollect, robyn_inputs):
        raise ValueError("InputCollect must be a robyn_inputs object")
    if OutputCollect and not isinstance(OutputCollect, robyn_outputs):
        raise ValueError("OutputCollect must be a robyn_outputs object")
    if select_model and not isinstance(select_model, str):
        raise ValueError("select_model must be a string")
    if dir and not os.path.isdir(dir):
        raise ValueError("dir must be a valid directory path")
    if export and not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    # InputCollect JSON
    ## ret = {}
    ret = dict()
    skip = [x for x in range(len(InputCollect)) if isinstance(InputCollect[x], (list, np.ndarray)) or (isinstance(InputCollect[x], str) and InputCollect[x].startswith("robyn_"))]
    ret["InputCollect"] = InputCollect[:len(InputCollect)-len(skip)]

    # ExportedModel JSON
    if OutputCollect:
        ##collect = {}
        collect = dict()
        collect["ts_validation"] = OutputCollect.ts_validation
        collect["train_timestamp"] = OutputCollect.train_timestamp
        collect["export_timestamp"] = pd.Timestamp.now()
        collect["run_time"] = f"{OutputCollect.run_time} min"
        collect["outputs_time"] = f"{OutputCollect.outputs_time} min"
        collect["total_time"] = f"{OutputCollect.run_time + OutputCollect.outputs_time} min"
        collect["total_iters"] = OutputCollect.iterations * OutputCollect.trials
        collect["conv_msg"] = re.sub(r":.*", "", OutputCollect.convergence.conv_msg)
        if "clusters" in OutputCollect:
            collect["clusters"] = OutputCollect.clusters.n_clusters
        skip = [x for x in range(len(OutputCollect)) if isinstance(OutputCollect[x], (list, np.ndarray)) or (isinstance(OutputCollect[x], str) and OutputCollect[x].startswith("robyn_"))]
        collect = {k: v for k, v in collect.items() if k not in skip}
        ret["ModelsCollect"] = collect

    # Model associated data
    if select_model:
        outputs = {}
        outputs["select_model"] = select_model
        outputs["summary"] = filter(OutputCollect.xDecompAgg, solID=select_model)
        outputs["errors"] = filter(OutputCollect.resultHypParam, solID=select_model)
        outputs["hyper_values"] = OutputCollect.resultHypParam.loc[select_model]
        outputs["hyper_updated"] = OutputCollect.hyper_updated
        ret["ExportedModel"] = outputs
    else:
        select_model = "models"

    if not dir.exists(dir) and export:
        dir.create(dir, recursive=True)

    filename = f"{dir}/RobynModel-{select_model}.json"
    filename = re.sub(r"//", "/", filename)
    ## ret.class = ["robyn_write", ret.class]
    ##ret.attr("json_file") = filename
    if export is not None:
        if quiet is False:
            print(f">> Exported model {select_model} as {filename}")

        if pareto_df is not None:
            if not all([x in pareto_df.columns for x in ("solID", "cluster")]):
                warnings.warn("Input 'pareto_df' is not a valid data.frame; must contain 'solID' and 'cluster' columns.")
            else:
                all_c = set(pareto_df['cluster'])
                pareto_df = [pareto_df[pareto_df.cluster == x] for x in all_c]
                ## names(pareto_df) = paste0("cluster", all_c)
                pareto_df.rename(columns={"cluster": "cluster"}, inplace=True)
                ret["OutputCollect"]["all_sols"] = pareto_df

    write_json(ret, filename, pretty=True, digits=10)
    return ret

def print_robyn_write(x, *args, **kwargs):
    """
    Print various information related to the exported model and its performance.

    Args:
        x: The exported model object.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    # Print exported directory, model, and window information
    print("Exported directory: {x.ExportedModel.plot_folder}")
    print("Exported model: {x.ExportedModel.select_model}")
    print("Window: {start} to {end} ({periods} {type}s)")

    # Print time series validation information
    val = x.ExportedModel.ts_validation
    print("Time Series Validation: {val} (train size = {val_detail})")

    # Print model performance and errors
    errors = x.ExportedModel.errors
    print("Model's Performance and Errors:")
    print("    {errors}")

    # Print summary values on selected model
    print("Summary Values on Selected Model:")
    print(x.ExportedModel.summary.drop(columns=["boot", "ci_"]).rename(columns={"performance": "ROI"}).mutate(decompPer=lambda x: format_num(100*x, pos="%")).replace(to_replace="NA", value="-").to_dataframe())

    # Print hyper-parameters
    print("Hyper-parameters:")
    print("    Adstock: {x.InputCollect.adstock}")

    # Print nice and tidy table format for hyper-parameters
    hyper_df = x.ExportedModel.hyper_values.drop(columns=["lambda", "penalty"]).gather().separate(key="key", into=["channel", "none"], sep=r"_", remove=False).mutate(hyperparameter=lambda x: gsub("^.*_", "", x.key)).select(channel=["channel", "hyperparameter"], value=["value"]).spread(key="hyperparameter", value="value")
    print(hyper_df)

def robyn_read(json_file=None, step=1, quiet=False):
    """
    Reads a JSON file and performs some modifications on the data.

    Args:
        json_file (str): The path to the JSON file to be read.
        step (int): The step value.
        quiet (bool): If True, suppresses the print statement.

    Returns:
        dict or None: The modified JSON data if `json_file` is provided, otherwise returns `json_file`.
    
    Raises:
        ValueError: If `json_file` is not a string or does not end with ".json".
        FileNotFoundError: If the specified `json_file` does not exist.
    """
    if json_file is not None:
        if not isinstance(json_file, str):
            raise ValueError("JSON file must be a string")
        if not json_file.endswith(".json"):
            raise ValueError("JSON file must be a valid .json file")
        if not os.path.exists(json_file):
            raise FileNotFoundError("JSON file can't be imported: {}".format(json_file))
        json = json.loads(open(json_file, "r").read())
        json["InputCollect"] = [x for x in json["InputCollect"] if len(x) > 0]
        json["ExportedModel"] = json["ModelsCollect"] + [json["ExportedModel"]]
        if not quiet:
            print("Imported JSON file successfully: {}".format(json_file))
        return json
    return json_file

def robyn_read(x, *args, **kwargs):
    # Extract input collect
    a = x['InputCollect']

    # Create a string to print
    str_to_print = """

########### InputCollect ############



Date: {a['date_var']}

Dependent: {a['dep_var']} [{a['dep_var_type']}]

Paid Media: {', '.join(a['paid_media_vars'])}

Paid Media Spend: {', '.join(a['paid_media_spends'])}

Context: {', '.join(a['context_vars'])}

Organic: {', '.join(a['organic_vars'])}

Prophet (Auto-generated): {a['prophet']}

Unused variables: {a['unused_vars']}

Model Window: {', '.join(a['window_start'], a['window_end'], sep=':')} ({a['rollingWindowEndWhich'] - a['rollingWindowStartWhich'] + 1} {a['intervalType']}s)

With Calibration: {not a['calibration_input'].isnull()}

Custom parameters: {a['custom_params']}



Adstock: {a['adstock']}

{hyps}

""".format(**a)

    # Print the string
    print(str_to_print)

    # Check if there's an exported model
    if x['ExportedModel'] is not None:
        # Create a new dataframe with the exported model
        temp = x.copy()

        # Set the class of the dataframe to "robyn_write"
        temp.attrs['class'] = 'robyn_write'

        # Print a blank line
        print()

        # Print the exported model
        print(temp)

    # Return an invisible object
    return pd.Series([], index=[0])

def robyn_recreate(json_file, quiet=False, *args, **kwargs):
    """
    Recreates a model based on the provided JSON file.

    Parameters:
    - json_file (str): The path to the JSON file.
    - quiet (bool): Whether to suppress console output. Default is False.
    - *args: Additional positional arguments.
    - **kwargs: Additional keyword arguments.

    Returns:
    - list: A list containing the InputCollect and OutputCollect objects.
    """
    # Read JSON file
    json = json.load(open(json_file, 'r'))

    # Extract model name
    model_name = json['ExportedModel']['select_model']
    print(f">>> Recreating model {model_name}")

    # Create list of arguments
    args = list(args)

    # Check if InputCollect is provided
    if 'InputCollect' not in args:
        # If not, create it using robyn_inputs
        InputCollect = robyn_inputs(json_file, quiet=quiet, *args, **kwargs)
        # Run model using robyn_run
        OutputCollect = robyn_run(InputCollect, json_file, export=False, quiet=quiet, *args, **kwargs)
    else:
        # If InputCollect is provided, use it
        InputCollect = args.pop('InputCollect')
        OutputCollect = robyn_run(InputCollect, json_file, export=False, quiet=quiet, *args, **kwargs)

    # Return list of InputCollect and OutputCollect
    return [InputCollect, OutputCollect]

def robyn_chain(json_file):
    """
    Extracts chain data from a JSON file and returns it as a dictionary.

    Parameters:
    json_file (str): The path to the JSON file.

    Returns:
    dict: A dictionary containing the extracted chain data.
    """
    # Read JSON data from file
    json_data = json.load(open(json_file, 'r'))

    # Extract IDs from JSON data
    ids = [json_data['InputCollect']['refreshChain'], json_data['ExportedModel']['select_model']]

    # Extract plot folder from JSON data
    plot_folder = json_data['ExportedModel']['plot_folder']

    # Split plot folder into parts
    temp = re.split('/', plot_folder)[1]

    # Extract chain from plot folder
    chain = temp[re.search('Robyn_', temp).start():]

    # If chain is empty, use the last part of the plot folder
    if not chain:
        chain = temp[temp != ''].pop()

    # Remove chain from plot folder
    base_dir = re.sub(f'/{chain}', '', plot_folder)

    # Initialize list to store chain data
    chain_data = dict()

    # Iterate over chain and read JSON files
    for i in range(len(chain)):
        if i == len(chain) - 1:
            json_new = json_data
        else:
            file = f'RobynModel-{json_new["InputCollect"]["refreshSourceID"]}.json'
            filename = os.path.join(base_dir, *chain[1:i], file)
            json_new = json.load(open(filename, 'r'))

        # Add JSON data to chain data
        chain_data[json_new['ExportedModel']['select_model']] = json_new

    # Reverse chain data
    chain_data = chain_data[::-1]

    # Extract plot folders from chain data
    dirs = [json_new['ExportedModel']['plot_folder'] for json_new in chain_data]

    # Create JSON file names
    json_files = [os.path.join(dir, f'RobynModel-{name}.json') for dir, name in zip(dirs, chain_data)]

    # Add JSON file names to chain data
    chain_data['json_files'] = json_files

    # Add chain to chain data
    chain_data['chain'] = ids

    # Check if chain and chain data match
    if len(ids) != len(chain_data):
        warnings.warn('Can\'t replicate chain-like results if you don\'t follow Robyn\'s chain structure')

    return chain_data
