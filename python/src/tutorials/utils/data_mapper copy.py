import json
import pandas as pd
from typing import Dict, Any
import sys
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
    attributes = {}  # Initialize an empty dictionary to store attributes
    print("Keys of OutputModels:", data["OutputModels"].keys())
    for trial_key, trial_data in data["OutputModels"].items():
        print("====================================")
        print("Name of trial key:", trial_key)
        print("Type of trial key:", type(trial_key))
        try:
            print("Type of trial data:", type(trial_data))
            print("Keys of trial data:", trial_data.keys())
        except:
            print("Not a dictionary")
        if isinstance(trial_data, dict):
            # This is a trial entry
            result_collect = trial_data.get("resultCollect", {})
            result_hyp_param = pd.DataFrame(result_collect.get("resultHypParam", []))
            print("Result Hyp Param:", result_hyp_param)
            x_decomp_agg = pd.DataFrame(result_collect.get("xDecompAgg", []))
            lift_calibration = pd.DataFrame(result_collect.get("liftCalibration", []))
            decomp_spend_dist = pd.DataFrame(result_collect.get("decompSpendDist", []))
            first_hyp_param = result_hyp_param.iloc[0] if not result_hyp_param.empty else {}
            trial = Trial(
                result_hyp_param=result_hyp_param,
                x_decomp_agg=x_decomp_agg,
                lift_calibration=lift_calibration,
                decomp_spend_dist=decomp_spend_dist,
                nrmse=first_hyp_param.get("NRMSE", 0),
                decomp_rssd=first_hyp_param.get("RSSD", 0),
                mape=first_hyp_param.get("MAPE", 0),
                rsq_train=first_hyp_param.get("rsq_train", 0),
                rsq_val=first_hyp_param.get("rsq_val", 0),
                rsq_test=first_hyp_param.get("rsq_test", 0),
                lambda_=first_hyp_param.get("lambda", 0),
                lambda_hp=first_hyp_param.get("lambda_hp", 0),
                lambda_max=first_hyp_param.get("lambda_max", 0),
                lambda_min_ratio=first_hyp_param.get("lambda_min_ratio", 0),
                pos=first_hyp_param.get("pos", 0),
                elapsed=first_hyp_param.get("elapsed", 0),
                elapsed_accum=first_hyp_param.get("elapsed_accum", 0),
                trial=first_hyp_param.get("trial", 0),
                iter_ng=first_hyp_param.get("iterNG", 0),
                iter_par=first_hyp_param.get("iterPar", 0),
                train_size=first_hyp_param.get("train_size", 0),
                sol_id=first_hyp_param.get("solID", ""),
            )
            trials.append(trial)
        else:
            # This is likely an attribute entry
            attributes[trial_key] = trial_data
    model_outputs = ModelOutputs(
        trials=trials,
        train_timestamp=attributes.get("train_timestamp", ""),
        cores=attributes.get("cores", 1),
        iterations=attributes.get("iterations", 0),
        intercept=attributes.get("intercept", True),
        intercept_sign=attributes.get("intercept_sign", ""),
        nevergrad_algo=attributes.get("nevergrad_algo", ""),
        ts_validation=attributes.get("ts_validation", False),
        add_penalty_factor=attributes.get("add_penalty_factor", False),
        hyper_fixed=attributes.get("hyper_fixed", False),
        hyper_updated={},  # Update this if necessary
        convergence={},  # Update this if necessary
        select_id="",  # Update this if necessary
        seed=0,  # Update this if necessary
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

    print(model_outputs.trials)
    # You can now compare these with the outputs from your Python/Python script
