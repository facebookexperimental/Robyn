# data_mapper.py
import json
import pandas as pd
from typing import Any, Dict
from robyn.data.entities.enums import (
    AdstockType,
    DependentVarType,
    ProphetVariableType,
    ProphetSigns,
)
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.convergence_data import ConvergenceData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import (
    ClusteredResult,
    ClusterPlotResults,
    ClusterConfidenceIntervals,
    DimentionalityReductionResults,
)


def import_input_collect(data: Dict[str, Any]) -> Dict[str, Any]:
    adstock_type = data.get("adstock", "geometric")
    if isinstance(adstock_type, list):
        adstock_type = adstock_type[0]
    dep_var_type = data.get("dep_var_type")[0]
    if isinstance(dep_var_type, list):
        dep_var_type = dep_var_type[0]
    mmm_data_spec_args = {
        "dep_var": data.get("dep_var")[0],
        "dep_var_type": DependentVarType(dep_var_type),
        "date_var": data.get("date_var")[0],
        "paid_media_spends": data.get("paid_media_spends", []),
        "paid_media_vars": data.get("paid_media_vars", []),
        "organic_vars": data.get("organic_vars", []),
        "context_vars": data.get("context_vars", []),
        "factor_vars": data.get("factor_vars", []),
        "window_start": data.get("window_start")[0],
        "window_end": data.get("window_end")[0],
        "rolling_window_length": data.get("rollingWindowLength")[0],
        "rolling_window_start_which": data.get("rollingWindowStartWhich")[0],
        "all_media": data.get("all_media", []),
        "rolling_window_end_which": data.get("rollingWindowEndWhich", 0)[0],
    }
    mmm_data = MMMData(
        data=pd.DataFrame(data["dt_input"]),
        mmmdata_spec=MMMData.MMMDataSpec(**mmm_data_spec_args),
    )
    holidays_data = HolidaysData(
        dt_holidays=pd.DataFrame(data.get("dt_holidays", {})),
        prophet_vars=[ProphetVariableType(v) for v in data.get("prophet_vars", [])],
        prophet_country=data.get("prophet_country"),
        prophet_signs=[ProphetSigns(s) for s in data.get("prophet_signs", [])],
    )

    print("Raw hyperparameters data:", data["hyperparameters"])
    # Group parameters by channel
    channel_params: Dict[str, Dict[str, list]] = {}
    for key, value in data["hyperparameters"].items():
        if key == "train_size":
            continue  # Skip train_size as it's not a channel parameter
        channel, param_type = key.rsplit("_", 1)
        if channel not in channel_params:
            channel_params[channel] = {}
        channel_params[channel][param_type] = value

    # Convert grouped parameters to ChannelHyperparameters objects
    hyperparameters_dict: Dict[str, ChannelHyperparameters] = {
        channel: ChannelHyperparameters(
            thetas=params.get("thetas"),
            alphas=params.get("alphas"),
            gammas=params.get("gammas"),
            # Add other parameters as needed
        )
        for channel, params in channel_params.items()
    }

    # Initialize the Hyperparameters instance
    hyperparameters = Hyperparameters(
        hyperparameters=hyperparameters_dict,
        adstock=AdstockType(adstock_type),  # Ensure adstock_type is defined
        lambda_=data.get("lambda_", 0.0),
        train_size=data.get("train_size", [0.5, 0.8]),
    )
    featurized_mmm_data = FeaturizedMMMData(
        dt_mod=pd.DataFrame(data.get("dt_mod", {})),
        dt_modRollWind=pd.DataFrame(data.get("dt_modRollWind", {})),
        modNLS=data.get("modNLS", {}),
    )
    return {
        "mmm_data": mmm_data,
        "holidays_data": holidays_data,
        "hyperparameters": hyperparameters,
        "featurized_mmm_data": featurized_mmm_data,
    }


def import_output_collect(output_collect: Dict[str, Any]) -> Dict[str, Any]:
    """
    Import and process OutputCollect data separately.

    Args:
        output_collect (Dict[str, Any]): The OutputCollect data dictionary

    Returns:
        Dict[str, Any]: Dictionary containing processed pareto and cluster data
    """
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
            x_decomp_vec_collect=pd.DataFrame(
                output_collect.get("xDecompVecCollect", {})
            ),
            plot_data_collect=_convert_plot_data(
                output_collect.get("allPareto", {}).get("plotDataCollect", {})
            ),
            df_caov_pct_all=pd.DataFrame(
                output_collect.get("allPareto", {}).get("df_caov_pct", {})
            ),
        )
    except Exception as e:
        print(f"Warning: Error creating ParetoResult: {str(e)}")
        pareto_result = None

    # Process cluster data if available
    cluster_data = None
    if output_collect.get("clusters"):
        try:
            clusters_dict = output_collect["clusters"]

            # Process PCA results if available
            pca_results = None
            pca_data = clusters_dict.get("clusters_PCA")
            if pca_data:
                pca_results = DimentionalityReductionResults(
                    explained_variance=pd.Series(pca_data.get("pca_explained", [])),
                    df=pd.DataFrame(pca_data.get("pcadf", [])),
                    plot_explained=(
                        pd.DataFrame(pca_data.get("plot_explained", []))
                        if pca_data.get("plot_explained")
                        else None
                    ),
                    plot=pca_data.get("plot"),
                )

            # Create plot results
            plot_results = ClusterPlotResults(
                top_solutions_errors_plot=pd.DataFrame(
                    clusters_dict.get("plot_models_errors", [])
                ),
                top_solutions_rois_plot=pd.DataFrame(
                    clusters_dict.get("plot_models_rois", [])
                ),
            )

            # Create confidence intervals
            cluster_ci = ClusterConfidenceIntervals(
                cluster_confidence_interval_df=pd.DataFrame(
                    clusters_dict.get("df_cluster_ci", [])
                ),
                boot_n=clusters_dict.get("boot_n", [0])[0],
                sim_n=clusters_dict.get("sim_n", [0])[0],
                clusters_confidence_interval_plot=pd.DataFrame(
                    clusters_dict.get("plot_clusters_ci", [])
                ),
            )

            # Create final clustered result
            cluster_data = ClusteredResult(
                cluster_data=pd.DataFrame(clusters_dict.get("data", [])),
                top_solutions=pd.DataFrame(clusters_dict.get("models", [])),
                cluster_ci=cluster_ci,
                n_clusters=clusters_dict.get("n_clusters", [0])[0],
                errors_weights=clusters_dict.get("errors_weights", []),
                clusters_means=pd.DataFrame(clusters_dict.get("clusters_means", [])),
                wss=pd.DataFrame(clusters_dict.get("wss", [])),
                correlations=pd.DataFrame(clusters_dict.get("corrs", [])),
                clusters_pca=pca_results,
                plots=plot_results,
            )

        except Exception as e:
            print(f"Warning: Error processing cluster data: {str(e)}")
            import traceback

            traceback.print_exc()
            cluster_data = None

    return {
        "pareto_result": pareto_result,
        "cluster_data": cluster_data,
    }


def _convert_plot_data(plot_data_collect: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert plot data collections to DataFrames while maintaining R's structure.

    Args:
        plot_data_collect: Dictionary containing plot data from R

    Returns:
        Dictionary of converted pandas DataFrames
    """
    converted_data = {}

    for model_id, model_data in plot_data_collect.items():
        try:
            converted_data[model_id] = {}

            # Convert each plot type (plot1data, plot2data, etc)
            for plot_type, plot_content in model_data.items():
                if plot_type == "plot7data":
                    converted_data[model_id][plot_type] = pd.DataFrame(plot_content)

                if not isinstance(plot_content, dict):
                    continue

                converted_data[model_id][plot_type] = {}

                # Convert each component within the plot type
                for component_name, component_data in plot_content.items():
                    try:
                        if isinstance(component_data, pd.DataFrame):
                            converted_data[model_id][plot_type][
                                component_name
                            ] = component_data
                        elif isinstance(component_data, (list, dict)):
                            converted_data[model_id][plot_type][component_name] = (
                                pd.DataFrame(component_data)
                            )
                        else:
                            # For scalar values, store as is
                            converted_data[model_id][plot_type][
                                component_name
                            ] = component_data
                    except Exception as e:
                        print(
                            f"Warning: Error converting {component_name} in {plot_type} for {model_id}: {str(e)}"
                        )

        except Exception as e:
            print(f"Warning: Error converting plot data for {model_id}: {str(e)}")

    return converted_data


def import_output_models(data: Dict[str, Any]) -> ModelOutputs:
    # Debug: Print the keys and a brief summary of the data
    print("Debug: R output data keys:", list(data.keys())[:5])
    # Print the shape or length for each key
    for key in list(data.keys())[:5]:
        if isinstance(data[key], dict):
            print(f"Data for {key}: Keys = {list(data[key].keys())[:3]}")
            # Check the content of resultCollect
            result_collect = data[key].get("resultCollect", {})
            print(f"Debug: resultCollect keys for {key}: {list(result_collect.keys())}")
            print(
                f"Debug: Sample resultHypParam for {key}: {result_collect.get('resultHypParam', [])[:3]}"
            )
    trials = []
    convergence_data = None
    hyper_bound_ng = pd.DataFrame()
    hyper_bound_fixed = pd.DataFrame()
    hyper_updated = {}
    for trial_key, trial_data in data.items():
        if trial_key == "convergence":
            convergence_data = ConvergenceData.from_dict(trial_data)
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
            print("Shape of result_hyp_param:")
            print(result_hyp_param.shape)
            print(f"Debug: result_hyp_param DataFrame for {trial_key}:")
            print(result_hyp_param.head())  # Print a few rows
            print("Data types in result_hyp_param:")
            print(result_hyp_param.dtypes)  # Check data types

            result_hyp_param.rename(columns={"solID": "sol_id"}, inplace=True)
            x_decomp_agg = pd.DataFrame(result_collect.get("xDecompAgg", []))
            print(f"Debug: x_decomp_agg DataFrame shape: {x_decomp_agg.shape}")
            x_decomp_agg.rename(columns={"solID": "sol_id"}, inplace=True)
            lift_calibration = pd.DataFrame(result_collect.get("liftCalibration", []))
            decomp_spend_dist = pd.DataFrame(result_collect.get("decompSpendDist", []))
            decomp_spend_dist.rename(columns={"solID": "sol_id"}, inplace=True)
            hyper_bound_ng = pd.DataFrame(trial_data.get("hyperBoundNG", {}))
            hyper_bound_fixed = pd.DataFrame(trial_data.get("hyperBoundFixed", []))
            trial = Trial(
                result_hyp_param=result_hyp_param,
                x_decomp_agg=x_decomp_agg,
                lift_calibration=lift_calibration,
                decomp_spend_dist=decomp_spend_dist,
                nrmse=result_hyp_param.get("nrmse", 0),  # Newly added
                decomp_rssd=result_hyp_param.get("decomp.rssd", 0),  # Newly added
                mape=result_hyp_param.get("mape", 0),  # Newly added
                rsq_train=result_hyp_param.get("rsq_train", 0),  # Newly added
                rsq_val=result_hyp_param.get("rsq_val", 0),  # Newly added
                rsq_test=result_hyp_param.get("rsq_test", 0),  # Newly added
                lambda_=result_hyp_param.get("lambda", 0),  # Newly added
                lambda_hp=result_hyp_param.get("lambda_hp", 0),  # Newly added
                lambda_max=result_hyp_param.get("lambda_max", 0),  # Newly added
                lambda_min_ratio=result_hyp_param.get(
                    "lambda_min_ratio", 0
                ),  # Newly added
                pos=result_hyp_param.get("pos", 0),  # Newly added
                elapsed=result_hyp_param.get("Elapsed", 0),  # Newly added
                elapsed_accum=result_hyp_param.get("ElapsedAccum", 0),  # Newly added
                trial=result_hyp_param.get("trial", 0),  # Newly added
                iter_ng=result_hyp_param.get("iterNG", 0),  # Newly added
                iter_par=result_hyp_param.get("iterPar", 0),  # Newly added
                train_size=result_hyp_param.get("train_size", 0),  # Newly added
                sol_id=result_hyp_param.get("solID", ""),  # Newly added
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
        select_id=data.get("select_id", ""),
        seed=data.get("seed", 0),
        hyper_bound_ng=hyper_bound_ng,
        hyper_bound_fixed=hyper_bound_fixed,
        ts_validation_plot=data.get("ts_validation_plot"),
        all_result_hyp_param=pd.concat([trial.result_hyp_param for trial in trials]),
        all_x_decomp_agg=pd.concat([trial.x_decomp_agg for trial in trials]),
        all_decomp_spend_dist=pd.concat([trial.decomp_spend_dist for trial in trials]),
    )
    return model_outputs


def load_data_from_json(filename: str) -> Dict[str, Any]:
    """
    Load the exported data from a JSON file.
    """
    with open(filename, "r") as f:
        return json.load(f)
