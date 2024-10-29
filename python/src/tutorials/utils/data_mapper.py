# data_mapper.py
import json
import sys
from typing import Any, Dict

import pandas as pd
from robyn.data.entities.enums import (
    AdstockType,
    CalibrationScope,
    ContextSigns,
    DependentVarType,
    OrganicSigns,
    PaidMediaSigns,
    ProphetSigns,
    ProphetVariableType,
    SaturationType,
)
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.convergence import Convergence
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.feature_engineering import FeaturizedMMMData


def export_data(
    InputCollect: Dict[str, Any],
    OutputModels: Dict[str, Any],
    outputsArgs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Export data from the R/Python script to a JSON format.
    """
    data = {
        "InputCollect": InputCollect,
        "OutputModels": OutputModels,
        "outputsArgs": outputsArgs,
    }
    return data


def import_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Import data from the JSON format and initialize relevant Python classes.
    """
    # Extract adstock and ensure it's not a list
    adstock_type = data["InputCollect"].get("adstock", "geometric")
    if isinstance(adstock_type, list):
        adstock_type = adstock_type[0]  # Take the first element if it's a list
    # Extract dep_var_type and ensure it's not a list
    dep_var_type = data["InputCollect"].get("dep_var_type")
    if isinstance(dep_var_type, list):
        dep_var_type = dep_var_type[0]  # Take the first element if it's a list
    mmm_data_spec_args = {
        "dep_var": data["InputCollect"].get("dep_var"),
        "dep_var_type": DependentVarType(dep_var_type),
        "date_var": data["InputCollect"].get("date_var"),
        "paid_media_spends": data["InputCollect"].get("paid_media_spends", []),
        "paid_media_vars": data["InputCollect"].get("paid_media_vars", []),
        "organic_vars": data["InputCollect"].get("organic_vars", []),
        "context_vars": data["InputCollect"].get("context_vars", []),
        "factor_vars": data["InputCollect"].get("factor_vars", []),
        "window_start": data["InputCollect"].get("window_start")[0],
        "window_end": data["InputCollect"].get("window_end")[0],
        "rolling_window_length": data["InputCollect"].get("rollingWindowLength")[0],
        "rolling_window_start_which": data["InputCollect"].get("rollingWindowStartWhich")[0],
        "all_media": data["InputCollect"].get("all_media", []),
        "rolling_window_end_which": data["InputCollect"].get("rollingWindowEndWhich", 0)[0],
    }
    mmm_data = MMMData(
        data=pd.DataFrame(data["InputCollect"]["dt_input"]),
        mmmdata_spec=MMMData.MMMDataSpec(**mmm_data_spec_args),
    )
    holidays_data = HolidaysData(
        dt_holidays=pd.DataFrame(data["InputCollect"].get("dt_holidays", {})),
        prophet_vars=[ProphetVariableType(v) for v in data["InputCollect"].get("prophet_vars", [])],
        prophet_country=data["InputCollect"].get("prophet_country"),
        prophet_signs=[ProphetSigns(s) for s in data["InputCollect"].get("prophet_signs", [])],
    )
    hyperparameters = Hyperparameters(
        hyperparameters=data["InputCollect"].get("hyperparameters", {}),
        adstock=AdstockType(adstock_type),
        lambda_=data["InputCollect"].get("lambda_", 0.0),
        train_size=data["InputCollect"].get("train_size", (0.5, 0.8)),
    )
    featurized_mmm_data = FeaturizedMMMData(
        dt_mod=pd.DataFrame(data["InputCollect"].get("dt_mod", {})),
        dt_modRollWind=pd.DataFrame(data["InputCollect"].get("dt_modRollWind", {})),
        modNLS=data["InputCollect"].get("modNLS", {}),
    )
    # Other initializations...
    trials = []
    convergence_data = None
    hyper_bound_ng = pd.DataFrame()
    hyper_bound_fixed = pd.DataFrame()
    hyper_updated = {}

    for trial_key, trial_data in data["OutputModels"].items():
        if trial_key == "convergence":
            convergence_data = Convergence.from_dict(trial_data)
        elif trial_key == "hyper_updated":
            hyper_updated = trial_data
        elif trial_key == "hyper_fixed":
            hyper_fixed = trial_data[0]
        elif trial_key == "train_timestamp":
            train_timestamp = trial_data[0]
        elif trial_key == "cores":
            cores = trial_data[0]
        elif trial_key == "iterations":
            iterations = trial_data[0]
        elif trial_key == "intercept":
            intercept = trial_data[0]
        elif trial_key == "intercept_sign":
            intercept_sign = trial_data[0]
        elif trial_key == "nevergrad_algo":
            nevergrad_algo = trial_data[0]
        elif trial_key == "ts_validation":
            ts_validation = trial_data[0]
        elif trial_key == "add_penalty_factor":
            add_penalty_factor = trial_data[0]
        elif isinstance(trial_data, dict):
            result_collect = trial_data.get("resultCollect", {})
            result_hyp_param = pd.DataFrame(result_collect.get("resultHypParam", []))
            x_decomp_agg = pd.DataFrame(result_collect.get("xDecompAgg", []))
            lift_calibration = pd.DataFrame(result_collect.get("liftCalibration", []))
            decomp_spend_dist = pd.DataFrame(result_collect.get("decompSpendDist", []))
            hyper_bound_ng = pd.DataFrame(trial_data.get("hyperBoundNG", {}))
            hyper_bound_fixed = pd.DataFrame(trial_data.get("hyperBoundFixed", []))
            trial = Trial(
                result_hyp_param=result_hyp_param,
                x_decomp_agg=x_decomp_agg,
                lift_calibration=lift_calibration,
                decomp_spend_dist=decomp_spend_dist,
                nrmse=0,
                decomp_rssd=0,
                mape=0,
                rsq_train=0,
                rsq_val=0,
                rsq_test=0,
                lambda_=0,
                lambda_hp=0,
                lambda_max=0,
                lambda_min_ratio=0,
                pos=0,
                elapsed=0,
                elapsed_accum=0,
                trial=trial_data.get("trial", 0),
                iter_ng=0,
                iter_par=0,
                train_size=0,
                sol_id="",
            )
            trials.append(trial)
    model_outputs = ModelOutputs(
        trials=trials,
        train_timestamp=train_timestamp,
        cores=cores,
        iterations=iterations,
        intercept=intercept,
        intercept_sign=intercept_sign,
        nevergrad_algo=nevergrad_algo,
        ts_validation=ts_validation,
        add_penalty_factor=add_penalty_factor,
        hyper_fixed=hyper_fixed,
        hyper_updated=hyper_updated,
        convergence=convergence_data,
        select_id=data["OutputModels"].get("select_id", ""),
        seed=data["OutputModels"].get("seed", 0),
        hyper_bound_ng=hyper_bound_ng,
        hyper_bound_fixed=hyper_bound_fixed,
        ts_validation_plot=data["OutputModels"].get("ts_validation_plot"),  # Add this line
    )

    # Extract and process pareto optimization results
    output_collect = data.get("OutputCollect", {})

    # Create ParetoResult
    try:
        pareto_result = ParetoResult(
            pareto_solutions=output_collect.get("allSolutions", []),
            pareto_fronts=output_collect.get("pareto_fronts", 0),
            result_hyp_param=pd.DataFrame(output_collect.get("resultHypParam", {})),
            x_decomp_agg=pd.DataFrame(output_collect.get("xDecompAgg", {})),
            result_calibration=(
                pd.DataFrame(output_collect.get("resultCalibration", {}))
                if output_collect.get("resultCalibration")
                else None
            ),
            media_vec_collect=pd.DataFrame(output_collect.get("mediaVecCollect", {})),
            x_decomp_vec_collect=pd.DataFrame(output_collect.get("xDecompVecCollect", {})),
            plot_data_collect=_convert_plot_data(output_collect.get("allPareto", {}).get("plotDataCollect", {})),
            df_caov_pct_all=pd.DataFrame(output_collect.get("allPareto", {}).get("df_caov_pct", {})),
        )
    except Exception as e:
        print(f"Warning: Error creating ParetoResult: {str(e)}")
        pareto_result = None

    # Create ParetoData
    try:
        pareto_data = ParetoData(
            decomp_spend_dist=pd.DataFrame(output_collect.get("decomp_spend_dist", {})),
            result_hyp_param=pd.DataFrame(output_collect.get("resultHypParam", {})),
            x_decomp_agg=pd.DataFrame(output_collect.get("xDecompAgg", {})),
            pareto_fronts=[output_collect.get("pareto_fronts", 0)],  # Convert to list as per class definition
        )
    except Exception as e:
        print(f"Warning: Error creating ParetoData: {str(e)}")
        pareto_data = None

    # Process cluster data if available
    cluster_data = None
    if output_collect.get("clusters"):
        try:
            cluster_data = {
                "n_clusters": output_collect["clusters"].get("n_clusters"),
                "models": output_collect["clusters"].get("models", {}),
                "data": pd.DataFrame(output_collect["clusters"].get("data", {})),
                "df_cluster_ci": pd.DataFrame(output_collect["clusters"].get("df_cluster_ci", {})),
            }
        except Exception as e:
            print(f"Warning: Error processing cluster data: {str(e)}")

    # Update model_outputs with any cluster information
    if cluster_data:
        try:
            # Add cluster information to relevant DataFrames
            for df_name in ["result_hyp_param", "x_decomp_agg", "media_vec_collect", "x_decomp_vec_collect"]:
                if df_name in output_collect and "data" in cluster_data:
                    df = pd.DataFrame(output_collect[df_name])
                    cluster_info = cluster_data["data"][["solID", "cluster", "top_sol"]]
                    df = pd.merge(df, cluster_info, on="solID", how="left")
                    output_collect[df_name] = df.to_dict()
        except Exception as e:
            print(f"Warning: Error adding cluster information to DataFrames: {str(e)}")

    return {
        "mmm_data": mmm_data,
        "holidays_data": holidays_data,
        "hyperparameters": hyperparameters,
        "featurized_mmm_data": featurized_mmm_data,
        "model_outputs": model_outputs,
        "pareto_result": pareto_result,
        "pareto_data": pareto_data,
        "cluster_data": cluster_data,
    }


def save_data_to_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save the exported data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f)


def load_data_from_json(filename: str) -> Dict[str, Any]:
    """
    Load the exported data from a JSON file.
    """
    with open(filename, "r") as f:
        return json.load(f)


# Example usage
if __name__ == "__main__":
    # Assuming you have InputCollect, OutputModels, and outputsArgs from the R/Python script
    # exported_data = export_data(InputCollect, OutputModels, outputsArgs)
    # save_data_to_json(
    #     exported_data,
    #     "/Users/yijuilee/project_robyn/robynpy_interfaces/Robyn/python/src/tutorials/data/R/exported_data.json",
    # )

    # Later, in your Python/Python script
    loaded_data = load_data_from_json(
        "/Users/yijuilee/project_robyn/robynpy_interfaces/Robyn/python/src/tutorials/data/R/exported_data.json"
    )
    imported_data = import_data(loaded_data)

    # Now you can use the imported_data dictionary to access the initialized Python classes
    mmm_data = imported_data["mmm_data"]
    holidays_data = imported_data["holidays_data"]
    hyperparameters = imported_data["hyperparameters"]
    featurized_mmm_data = imported_data["featurized_mmm_data"]
    model_outputs = imported_data["model_outputs"]

    # You can now compare these with the outputs from your Python/Python script
