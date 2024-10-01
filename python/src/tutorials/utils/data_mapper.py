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
    # Initialize MMMData
    mmm_data = MMMData(
        data=pd.DataFrame(data["InputCollect"]["dt_input"]),
        mmmdata_spec=MMMData.MMMDataSpec(**data["InputCollect"]["mmmdata_spec"]),
    )

    # Initialize HolidaysData
    holidays_data = HolidaysData(
        dt_holidays=pd.DataFrame(data["InputCollect"]["dt_holidays"]),
        prophet_vars=[ProphetVariableType(v) for v in data["InputCollect"]["prophet_vars"]],
        prophet_country=data["InputCollect"]["prophet_country"],
        prophet_signs=[ProphetSigns(s) for s in data["InputCollect"]["prophet_signs"]],
    )

    # Initialize Hyperparameters
    hyperparameters = Hyperparameters(
        hyperparameters=data["InputCollect"]["hyperparameters"],
        adstock=AdstockType(data["InputCollect"]["adstock"]),
        lambda_=data["InputCollect"].get("lambda_", 0.0),
        train_size=data["InputCollect"].get("train_size", (0.5, 0.8)),
    )

    # Initialize FeaturizedMMMData
    featurized_mmm_data = FeaturizedMMMData(
        dt_mod=pd.DataFrame(data["InputCollect"]["dt_mod"]),
        dt_modRollWind=pd.DataFrame(data["InputCollect"]["dt_modRollWind"]),
        modNLS=data["InputCollect"]["modNLS"],
    )

    # Initialize ModelOutputs
    trials = []
    for trial_data in data["OutputModels"]["trials"]:
        trial = Trial(
            result_hyp_param=pd.DataFrame(trial_data["result_hyp_param"]),
            x_decomp_agg=pd.DataFrame(trial_data["x_decomp_agg"]),
            lift_calibration=pd.DataFrame(trial_data["lift_calibration"]),
            decomp_spend_dist=pd.DataFrame(trial_data["decomp_spend_dist"]),
            nrmse=trial_data["nrmse"],
            decomp_rssd=trial_data["decomp_rssd"],
            mape=trial_data["mape"],
        )
        trials.append(trial)

    model_outputs = ModelOutputs(
        trials=trials,
        train_timestamp=data["OutputModels"]["train_timestamp"],
        cores=data["OutputModels"]["cores"],
        iterations=data["OutputModels"]["iterations"],
        intercept=data["OutputModels"]["intercept"],
        intercept_sign=data["OutputModels"]["intercept_sign"],
        nevergrad_algo=data["OutputModels"]["nevergrad_algo"],
        ts_validation=data["OutputModels"]["ts_validation"],
        add_penalty_factor=data["OutputModels"]["add_penalty_factor"],
        hyper_updated=data["OutputModels"]["hyper_updated"],
        hyper_fixed=data["OutputModels"]["hyper_fixed"],
        convergence=data["OutputModels"]["convergence"],
        ts_validation_plot=data["OutputModels"]["ts_validation_plot"],
        select_id=data["OutputModels"]["select_id"],
        seed=data["OutputModels"]["seed"],
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
