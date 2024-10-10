# data_mapper.py
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
from robyn.modeling.entities.convergence import Convergence


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
    attributes = {}
    convergence_data = None
    hyper_bound_ng = {}
    hyper_bound_fixed = {}

    for trial_key, trial_data in data["OutputModels"].items():
        if trial_key == "attributes":
            attributes = trial_data
        elif trial_key == "convergence":
            convergence_data = Convergence.from_dict(trial_data)
        elif isinstance(trial_data, dict):
            result_collect = trial_data.get("resultCollect", {})
            result_hyp_param = pd.DataFrame(result_collect.get("resultHypParam", []))
            x_decomp_agg = pd.DataFrame(result_collect.get("xDecompAgg", []))
            lift_calibration = pd.DataFrame(result_collect.get("liftCalibration", []))
            decomp_spend_dist = pd.DataFrame(result_collect.get("decompSpendDist", []))
        elif trial_key == "hyperBoundNG":
            hyper_bound_ng = trial_data
        elif trial_key == "hyperBoundFixed":
            hyper_bound_fixed = trial_data
        for i, row in result_hyp_param.iterrows():
            trial = Trial(
                result_hyp_param=result_hyp_param,
                x_decomp_agg=x_decomp_agg,
                lift_calibration=lift_calibration,
                decomp_spend_dist=decomp_spend_dist,
                nrmse=row.get("NRMSE", 0),
                decomp_rssd=row.get("decomp.rssd", 0),
                mape=row.get("MAPE", 0),
                rsq_train=row.get("rsq_train", 0),
                rsq_val=row.get("rsq_val", 0),
                rsq_test=row.get("rsq_test", 0),
                lambda_=row.get("lambda", 0),
                lambda_hp=row.get("lambda_hp", 0),
                lambda_max=row.get("lambda_max", 0),
                lambda_min_ratio=row.get("lambda_min_ratio", 0),
                pos=row.get("pos", 0),
                elapsed=row.get("Elapsed", 0),
                elapsed_accum=row.get("ElapsedAccum", 0),
                trial=row.get("trial", 0),
                iter_ng=row.get("iterNG", 0),
                iter_par=row.get("iterPar", 0),
                train_size=row.get("train_size", 0),
                sol_id=row.get("solID", ""),
            )
            trials.append(trial)

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
        hyper_updated=attributes.get("hyper_updated", {}),
        convergence=convergence_data,
        select_id=attributes.get("select_id", ""),
        seed=attributes.get("seed", 0),
        hyper_bound_ng=hyper_bound_ng,
        hyper_bound_fixed=hyper_bound_fixed,
        ts_validation_plot=data["OutputModels"].get("ts_validation_plot"),  # Add this line
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
