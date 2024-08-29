# modeling.py

import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.append("/Users/yijuilee/project_robyn/modelling/Robyn/python/src")

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.enums import AdstockType, DependentVarType, PaidMediaSigns
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.mmmdata_collection import (
    IntermediateData,
    MMMDataCollection,
    ModelParameters,
    TimeWindow,
)
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.mmm_model_executor import MMMModelExecutor
from robyn.modeling.mmm_pareto_optimizer import ParetoOptimizer
from robyn.modeling.model_clusters_analyzer import ModelClustersAnalyzer
from robyn.modeling.model_evaluation import ModelEvaluator
from robyn.modeling.model_refresh import ModelRefresh


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration purposes."""
    date_range = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "date": date_range,
            "sales": np.random.normal(1000, 100, len(date_range)),
            "tv_spend": np.random.uniform(500, 1500, len(date_range)),
            "radio_spend": np.random.uniform(200, 800, len(date_range)),
            "online_spend": np.random.uniform(300, 1000, len(date_range)),
        }
    )

    # Add some seasonality and trend to sales
    data["sales"] += (
        np.sin(np.arange(len(data)) * 2 * np.pi / 365) * 200
        + np.arange(len(data)) * 0.5
    )

    return data


def prepare_mmmdata_collection(data: pd.DataFrame) -> MMMDataCollection:
    """Prepare MMMDataCollection from the input data."""
    mmmdata_spec = MMMData.MMMDataSpec(
        dep_var="sales",
        dep_var_type=DependentVarType.REVENUE,
        date_var="date",
        paid_media_spends=["tv_spend", "radio_spend", "online_spend"],
        paid_media_vars=["tv_spend", "radio_spend", "online_spend"],
        paid_media_signs=[
            PaidMediaSigns.POSITIVE,
            PaidMediaSigns.POSITIVE,
            PaidMediaSigns.POSITIVE,
        ],
        organic_vars=[],
        organic_signs=[],
        context_vars=[],
        context_signs=[],
        factor_vars=[],
    )

    mmmdata = MMMData(data=data, mmmdata_spec=mmmdata_spec)

    holiday_data = None  # Placeholder for holiday data
    adstock = AdstockType.GEOMETRIC  # Example adstock type
    hyperparameters = Hyperparameters()  # Initialize with default values
    calibration_input = None  # Placeholder for calibration input
    intermediate_data = IntermediateData(
        dt_mod=None, dt_modRollWind=None, xDecompAggPrev=None
    )
    model_parameters = ModelParameters(
        dayInterval=1,
        intervalType="day",
        mediaVarCount=3,
        exposure_vars=[],
        all_media=["tv_spend", "radio_spend", "online_spend"],
        all_ind_vars=["tv_spend", "radio_spend", "online_spend"],
        factor_vars=[],
        unused_vars=[],
    )
    time_window = TimeWindow(
        window_start=data["date"].min(),
        window_end=data["date"].max(),
        rollingWindowStartWhich=0,
        rollingWindowEndWhich=len(data) - 1,
        rollingWindowLength=len(data),
        totalObservations=len(data),
        refreshAddedStart=data["date"].min(),
    )
    custom_params = {}  # Placeholder for custom parameters

    mmm_data = MMMDataCollection(
        mmmdata=mmmdata,
        holiday_data=holiday_data,
        adstock=adstock,
        hyperparameters=hyperparameters,
        calibration_input=calibration_input,
        intermediate_data=intermediate_data,
        model_parameters=model_parameters,
        time_window=time_window,
        custom_params=custom_params,
    )

    return mmm_data


def run_model(mmm_data: MMMDataCollection) -> ModelOutputCollection:
    """Run the MMM model and return the output collection."""
    model_executor = MMMModelExecutor()
    trials_config = TrialsConfig(
        num_trials=5,
        num_iterations_per_trial=1000,
        timeseries_validation=True,
        add_penalty_factor=False,
    )

    model_output = model_executor.model_run(
        mmmdata_collection=mmm_data, trials_config=trials_config, seed=42, quiet=True
    )
    return model_output


def optimize_pareto(
    mmm_data: MMMDataCollection, model_output: ModelOutputCollection
) -> Dict[str, Any]:
    """Run Pareto optimization on the model output."""
    pareto_optimizer = ParetoOptimizer()
    pareto_results = pareto_optimizer.pareto_optimize(
        mmmdata_collection=mmm_data,
        modeloutput=model_output,
        pareto_fronts="auto",
        min_candidates=50,
    )
    return pareto_results


def analyze_clusters(model_output: ModelOutputCollection) -> Dict[str, Any]:
    """Analyze model clusters."""
    cluster_analyzer = ModelClustersAnalyzer()
    cluster_results = cluster_analyzer.model_clusters_analyze(
        input=model_output.model_output,
        dep_var_type="continuous",
        cluster_by="hyperparameters",
        k="auto",
        quiet=True,
    )
    return cluster_results


def evaluate_model(model_output: ModelOutputCollection) -> Dict[str, float]:
    """Evaluate the model and return key metrics."""
    evaluator = ModelEvaluator()
    metrics = {
        "rsquared": evaluator.calculate_rsquared(model_output.model_output),
        "mae": evaluator.calculate_mae(model_output.model_output),
        "mape": evaluator.calculate_mape(model_output.model_output),
    }
    return metrics


def refresh_model(
    mmm_data: MMMDataCollection, model_output: ModelOutputCollection
) -> ModelOutputCollection:
    """Refresh the model with new data."""
    model_refresher = ModelRefresh()
    refresh_config = ModelRefreshConfig(
        refresh_steps=2, refresh_mode="manual", refresh_iters=500, refresh_trials=2
    )
    refreshed_output = model_refresher.model_refresh(
        mmmdata_collection=mmm_data,
        model_output_collection=model_output,
        refresh_config=refresh_config,
    )
    return refreshed_output


def main():
    print("Starting MMM Modeling Demo")

    # Generate and prepare data
    print("Generating sample data...")
    data = generate_sample_data()
    mmm_data = prepare_mmmdata_collection(data)
    print("Data prepared successfully.")

    # Run the model
    print("=========================")
    print("Running MMM model...")
    model_output = run_model(mmm_data)
    print("=========================")
    print("=========================")
    print("Model Output after run_model at main.py: ", model_output)
    print("Model run completed.")
    print("=========================")

    # Optimize Pareto front
    print("Optimizing Pareto front...")
    pareto_results = optimize_pareto(mmm_data, model_output)
    print(
        f"Number of Pareto-optimal solutions: {len(pareto_results['pareto_solutions'])}"
    )

    # Analyze clusters
    print("Analyzing model clusters...")
    cluster_results = analyze_clusters(model_output)
    print(f"Number of clusters: {cluster_results['n_clusters']}")

    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluate_model(model_output)
    print(f"Model metrics: {metrics}")

    # Refresh model
    print("Refreshing model...")
    refreshed_output = refresh_model(mmm_data, model_output)
    print("Model refreshed successfully.")

    print("MMM Modeling Demo completed.")


if __name__ == "__main__":
    main()
