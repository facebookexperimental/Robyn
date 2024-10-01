import json
import pandas as pd
from typing import Dict, Any

from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.enums import (
    DependentVarType,
    AdstockType,
    SaturationType,
    ProphetVariableType,
    PaidMediaSigns,
    OrganicSigns,
    ContextSigns,
    ProphetSigns,
    CalibrationScope,
)


def export_data(
    InputCollect: Dict[str, Any], OutputModels: Dict[str, Any], outputsArgs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Export data from the R/Python script to a JSON format.
    """
    data = {"InputCollect": InputCollect, "OutputModels": OutputModels, "outputsArgs": outputsArgs}
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
        "window_start": data["InputCollect"].get("window_start"),
        "window_end": data["InputCollect"].get("window_end"),
    }
    mmm_data = MMMData(
        data=pd.DataFrame(data["InputCollect"]["dt_input"]), mmmdata_spec=MMMData.MMMDataSpec(**mmm_data_spec_args)
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
    for trial_data in data["OutputModels"].get("trials", []):
        # Check if trial_data is a dictionary and contains the expected keys
        if isinstance(trial_data, dict) and "result_hyp_param" in trial_data:
            result_hyp_param = trial_data.get("result_hyp_param")
            if isinstance(result_hyp_param, dict):
                result_hyp_param = pd.DataFrame(result_hyp_param)
            else:
                result_hyp_param = pd.DataFrame()  # Default to empty DataFrame if not a dict
            x_decomp_agg = trial_data.get("x_decomp_agg")
            if isinstance(x_decomp_agg, dict):
                x_decomp_agg = pd.DataFrame(x_decomp_agg)
            else:
                x_decomp_agg = pd.DataFrame()  # Default to empty DataFrame if not a dict
            lift_calibration = trial_data.get("lift_calibration")
            if isinstance(lift_calibration, dict):
                lift_calibration = pd.DataFrame(lift_calibration)
            else:
                lift_calibration = pd.DataFrame()  # Default to empty DataFrame if not a dict
            decomp_spend_dist = trial_data.get("decomp_spend_dist")
            if isinstance(decomp_spend_dist, dict):
                decomp_spend_dist = pd.DataFrame(decomp_spend_dist)
            else:
                decomp_spend_dist = pd.DataFrame()  # Default to empty DataFrame if not a dict
            trial = Trial(
                result_hyp_param=result_hyp_param,
                x_decomp_agg=x_decomp_agg,
                lift_calibration=lift_calibration,
                decomp_spend_dist=decomp_spend_dist,
                nrmse=trial_data.get("nrmse", 0),
                decomp_rssd=trial_data.get("decomp_rssd", 0),
                mape=trial_data.get("mape", 0),
            )
            trials.append(trial)
        else:
            # Handle case where trial_data is not a dictionary or does not contain expected keys
            print("Warning: Skipping a trial entry due to unexpected data format.")

    model_outputs = ModelOutputs(
        trials=trials,
        train_timestamp=data["OutputModels"].get("train_timestamp", ""),
        cores=data["OutputModels"].get("cores", 1),
        iterations=data["OutputModels"].get("iterations", 0),
        intercept=data["OutputModels"].get("intercept", True),
        intercept_sign=data["OutputModels"].get("intercept_sign", ""),
        nevergrad_algo=data["OutputModels"].get("nevergrad_algo", ""),
        ts_validation=data["OutputModels"].get("ts_validation", False),
        add_penalty_factor=data["OutputModels"].get("add_penalty_factor", False),
        hyper_updated=data["OutputModels"].get("hyper_updated", {}),
        hyper_fixed=data["OutputModels"].get("hyper_fixed", False),
        convergence=data["OutputModels"].get("convergence", {}),
        ts_validation_plot=data["OutputModels"].get("ts_validation_plot", None),
        select_id=data["OutputModels"].get("select_id", ""),
        seed=data["OutputModels"].get("seed", 0),
    )

    return {
        "mmm_data": mmm_data,
        "holidays_data": holidays_data,
        "hyperparameters": hyperparameters,
        "featurized_mmm_data": featurized_mmm_data,
        "model_outputs": model_outputs,
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
