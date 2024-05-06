# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import os
import time
import pandas as pd
import re
from datetime import datetime

from .json import robyn_write
from .cluster import robyn_clusters
from .checks import check_dir, check_calibconstr
from .pareto import robyn_pareto

def robyn_outputs(input_collect,
                  output_models,
                  pareto_fronts="auto",
                  calibration_constraint=0.1,
                  plot_folder=None,
                  plot_folder_sub=None,
                  plot_pareto=True,
                  csv_out="pareto",
                  clusters=True,
                  select_model="clusters",
                  ui=False,
                  export=True,
                  all_sol_json=False,
                  quiet=False,
                  refresh=False):
    """
    Runs the Robyn Pareto algorithm on the given output models and collects the results.

    Args:
        input_collect (dict): The input collection.
        output_models (object): The output models.
        pareto_fronts (str, optional): The number of Pareto fronts to calculate. Defaults to "auto".
        calibration_constraint (float, optional): The calibration constraint. Defaults to 0.1.
        plot_folder (str, optional): The folder to save the plots. Defaults to None.
        plot_folder_sub (str, optional): The subfolder within the plot folder. Defaults to None.
        plot_pareto (bool, optional): Whether to plot the Pareto fronts. Defaults to True.
        csv_out (str, optional): The type of CSV output. Defaults to "pareto".
        clusters (bool, optional): Whether to calculate clusters for model selection. Defaults to True.
        select_model (str, optional): The model selection method. Defaults to "clusters".
        ui (bool, optional): Whether to enable the user interface. Defaults to False.
        export (bool, optional): Whether to export the results. Defaults to True.
        all_sol_json (bool, optional): Whether to export all solutions as JSON. Defaults to False.
        quiet (bool, optional): Whether to suppress console output. Defaults to False.
        refresh (bool, optional): Whether to refresh the results. Defaults to False.

    Returns:
        dict: The collected output results.
    """

    t0 = time.time()

    if plot_folder is None:
        plot_folder = os.getcwd()

    plot_folder = check_dir(plot_folder)

    # Check calibration constrains
    calibrated = 'calibration_input' in input_collect and input_collect['calibration_input'] is not None
    all_fixed = len(output_models["trials"][0]["hyperBoundFixed"]) == len(output_models["hyper_updated"])
    if not all_fixed:
        calibration_constraint = check_calibconstr(calibration_constraint, output_models["metadata"]["iterations"], output_models["metadata"]["trials"], input_collect["robyn_inputs"]["calibration_input"], refresh=refresh)

    #####################################
    #### Run robyn_pareto on OutputModels

    total_models = output_models["metadata"]["iterations"] * output_models["metadata"]["trials"]
    if not isinstance(output_models["metadata"]["hyper_fixed"], bool):
        print(f"Running Pareto calculations for {total_models} models on {pareto_fronts} fronts...")

    pareto_results = robyn_pareto(input_collect, output_models, pareto_fronts=pareto_fronts, calibration_constraint=calibration_constraint, quiet=quiet, calibrated=calibrated, refresh=refresh)
    pareto_fronts = pareto_results["pareto_fronts"]
    all_solutions = pareto_results["pareto_solutions"]

    #####################################
    #### Gather the results into output object

    all_pareto = {
        "resultHypParam": pareto_results["resultHypParam"],
        "xDecompAgg": pareto_results["xDecompAgg"],
        "resultCalibration": pareto_results["resultCalibration"],
        "plotDataCollect": pareto_results["plotDataCollect"],
        "df_caov_pct": pareto_results["df_caov_pct_all"]
    }

    # Set folder to save outputs
    depth = 0 if "refreshDepth" not in input_collect["robyn_inputs"] else input_collect["robyn_inputs"]["refreshDepth"]
    folder_var = "init" if not int(depth) > 0 else "rf" + str(depth)

    if plot_folder_sub is None:
        plot_folder_sub = "Robyn_" + datetime.now().strftime("%Y%m%d%H%M") + "_" + folder_var

    plot_folder = re.sub("//+", "/", f"{plot_folder}/{plot_folder_sub}/")

    if not os.path.exists(plot_folder) and export:
        print(f"Creating directory for outputs: {plot_folder}")
        os.makedirs(plot_folder)

    # Final results object
    OutputCollect = {
        'resultHypParam': pareto_results['resultHypParam'][pareto_results['resultHypParam']['solID'].isin(all_solutions)],
        'xDecompAgg': pareto_results['xDecompAgg'][pareto_results['xDecompAgg']['solID'].isin(all_solutions)],

        #"xDecompAgg": pareto_results["xDecompAgg"].loc[pareto_results["solID"].isin(allSolutions)],
        "mediaVecCollect": pareto_results["mediaVecCollect"],
        "xDecompVecCollect": pareto_results["xDecompVecCollect"],
        #"resultCalibration": None if not calibrated else pareto_results["resultCalibration"].loc[pareto_results["solID"].isin(allSolutions)],
        "resultCalibration": pareto_results["resultCalibration"][pareto_results["resultCalibration"]["solID"].isin(all_solutions)] if calibrated else None,
        "allSolutions": all_solutions,
        "allPareto": all_pareto,
        "calibration_constraint": calibration_constraint,
        "OutputModels": output_models,
        "cores": output_models["metadata"]["cores"],
        "iterations": output_models["metadata"]["iterations"],
        "trials": output_models["trials"],
        "intercept_sign": output_models["metadata"]["intercept_sign"],
        "nevergrad_algo": output_models["metadata"]["nevergrad_algo"],
        "add_penalty_factor": output_models["metadata"]["add_penalty_factor"],
        "seed": output_models["seed"],
        "UI": None,
        "pareto_fronts": pareto_fronts,
        'hyper_fixed': output_models["metadata"]['hyper_fixed'],
        "plot_folder": plot_folder
    }
    OutputCollect.keys()

    # Cluster results and amend cluster output
    if clusters:
        if not quiet:
            print(">>> Calculating clusters for model selection using Pareto fronts...")
        clusterCollect = robyn_clusters(
            OutputCollect,
            dep_var_type=input_collect["robyn_inputs"]["dep_var_type"],
            quiet=quiet,
            export=export
        )
        OutputCollect["resultHypParam"] = pd.merge(
            OutputCollect["resultHypParam"],
            clusterCollect["data"].loc[clusterCollect["data"]["solID"].isin(all_solutions)],
            on="solID"
        )
        OutputCollect["xDecompAgg"] = pd.merge(
            OutputCollect["xDecompAgg"],
            clusterCollect["data"].loc[clusterCollect["data"]["solID"].isin(all_solutions)],
            on="solID"
        )
        OutputCollect["mediaVecCollect"] = pd.merge(
            OutputCollect["mediaVecCollect"],
            clusterCollect["data"].loc[clusterCollect["data"]["solID"].isin(all_solutions)],
            on="solID"
        )
        OutputCollect["xDecompVecCollect"] = pd.merge(
            OutputCollect["xDecompVecCollect"],
            clusterCollect["data"].loc[clusterCollect["data"]["solID"].isin(all_solutions)],
            on="solID"
        )
        if calibrated:
            OutputCollect["resultCalibration"] = pd.merge(
                OutputCollect["resultCalibration"],
                clusterCollect["data"].loc[clusterCollect["data"]["solID"].isin(all_solutions)],
                on="solID"
            )
        OutputCollect["clusters"] = clusterCollect

    # TODO Add export code to enable plotting
    # if export:
    #     try:
    #         message(">>> Collecting {} pareto-optimum results into: {}".format(len(all_solutions), plot_folder))
    #         all_plots = robyn_plots(input_collect, output_collect, export=export, quiet=quiet)
    #         message(">> Exporting general plots into directory...")
    #         if csv_out in ["all", "pareto"]:
    #             message(">> Exporting {} results as CSVs into directory...".format(csv_out))
    #             robyn_csv(input_collect, output_collect, csv_out, export=export, calibrated=calibrated)
    #         if plot_pareto:
    #             message(">>> Exporting pareto one-pagers into directory...")
    #             select_model = select_model if not clusters or output_collect["clusters"] is None else None
    #             pareto_onepagers = robyn_onepagers(input_collect, output_collect, select_model=select_model, quiet=quiet, export=export)
    #         if all_sol_json:
    #             pareto_df = output_collect["resultHypParam"].filter(pandas.notnull(pandas.Series(["cluster"]))).select(["solID", "cluster", "top_sol"]).sort_values(by=["cluster", "top_sol"], ascending=False).drop(columns=["solID"])
    #         else:
    #             pareto_df = None
    #         ##attr(output_collect, "runTime") = round(difftime(sys.time(), t0, units="mins"), 2)
    #         output_collect["runTime"] = round(difftime(sys.time(), t0, units="mins"), 2)
    #         robyn_write(input_collect, output_collect, dir=plot_folder, quiet=quiet, pareto_df=pareto_df, export=export)
    #         if ui and plot_pareto:
    #             output_collect["UI"] = {"pareto_onepagers": pareto_onepagers}
    #         output_collect["UI"] = output_collect.get("UI", pandas.DataFrame()) if ui else None
    #     except Exception as e:
    #         message("Failed exporting results, but returned model results anyways: {}".format(e))
    ##if not is.null(output_models["hyper_updated"]):
    if output_models["hyper_updated"] is not None:
        OutputCollect["hyper_updated"] = output_models["hyper_updated"]
    ##attr(output_collect, "runTime") = round(difftime(sys.time(), t0, units="mins"), 2)
    # OutputCollect["runTime"] = round(difftime(sys.time(), t0, units="mins"), 2)
    ##class(output_collect) = ["robyn_outputs", class(output_collect)]
    ##??output_collect["robyn_outputs"] = output_collect
    ##return(invisible(output_collect))
    return OutputCollect

def print_robyn_outputs(x, *args, **kwargs):
    """
    Print various outputs related to Robyn.

    Parameters:
    - x: Robyn object
    - *args: Additional positional arguments
    - **kwargs: Additional keyword arguments
    """
    print("Plot Folder: {x.plot_folder}")
    print("Calibration Constraint: {x.calibration_constraint}")
    print("Hyper-parameters fixed: {x.hyper_fixed}")
    print("Pareto-front ({x.pareto_fronts}) All solutions ({len(x.allSolutions)}): {', '.join(x.allSolutions)}")
    if "clusters" in x.keys():
        print("Clusters (k = {x.clusters.n_clusters}): {', '.join(x.clusters.models.solID)}")
    else:
        print("")

def robyn_csv(input_collect, output_collect, csv_out=None, export=True, calibrated=False):
    """
    Export data from Robyn outputs to CSV files.

    Args:
        input_collect (InputCollect): The input collection object.
        output_collect (robyn_outputs): The output collection object.
        csv_out (str or None, optional): The type of CSV files to export. Defaults to None.
        export (bool, optional): Whether to export the data to CSV files. Defaults to True.
        calibrated (bool, optional): Whether the data is calibrated. Defaults to False.
    """
    if export:
        # Check that OutputCollect has the correct class
        assert isinstance(output_collect, robyn_outputs)

        # Get the temp all dataframe
        temp_all = output_collect.allPareto

        # Get the plot folder
        plot_folder = output_collect.plot_folder

        # Write the pareto hyperparameters and aggregated data to CSV
        if "pareto" in csv_out:
            pd.write_csv(output_collect.resultHypParam, os.path.join(plot_folder, "pareto_hyperparameters.csv"))
            pd.write_csv(output_collect.xDecompAgg, os.path.join(plot_folder, "pareto_aggregated.csv"))
            if calibrated:
                pd.write_csv(output_collect.resultCalibration, os.path.join(plot_folder, "pareto_calibration.csv"))

        # Write the all hyperparameters and aggregated data to CSV
        if "all" in csv_out:
            pd.write_csv(temp_all.resultHypParam, os.path.join(plot_folder, "all_hyperparameters.csv"))
            pd.write_csv(temp_all.xDecompAgg, os.path.join(plot_folder, "all_aggregated.csv"))
            if calibrated:
                pd.write_csv(temp_all.resultCalibration, os.path.join(plot_folder, "all_calibration.csv"))

        # Write the raw data and transformation matrices to CSV
        ## if not is.null(csv_out):
        if csv_out is not None:
            pd.write_csv(input_collect.dt_input, os.path.join(plot_folder, "raw_data.csv"))
            pd.write_csv(output_collect.mediaVecCollect, os.path.join(plot_folder, "pareto_media_transform_matrix.csv"))
            pd.write_csv(output_collect.xDecompVecCollect, os.path.join(plot_folder, "pareto_alldecomp_matrix.csv"))
