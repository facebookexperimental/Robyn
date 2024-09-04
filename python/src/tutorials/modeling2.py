import sys
from typing import Any, Dict

import numpy as np
import pandas as pd


sys.path.append("/Users/yijuilee/project_robyn/modelling/Robyn/python/src")

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.enums import (
    AdstockType,
    DependentVarType,
    PaidMediaSigns,
    ProphetSigns,
    ProphetVariableType,
)
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.mmmdata_collection import (
    IntermediateData,
    MMMDataCollection,
    ModelParameters,
    TimeWindow,
)
from robyn.modeling.allocator import RobynAllocator
from robyn.modeling.convergence import ModelConvergence
from robyn.modeling.entities.convergence_result import ConvergenceResult
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.mmm_model_executor import MMMModelExecutor
from robyn.modeling.mmm_pareto_optimizer import ParetoOptimizer
from robyn.modeling.model_clusters_analyzer import ModelClustersAnalyzer
from robyn.modeling.model_evaluation import ModelEvaluator
from robyn.modeling.model_refresh import ModelRefresh, ModelRefreshConfig
from robyn.modeling.transformation import saturation_hill, transform_adstock


def apply_adstock_transformations(mmm_data: MMMDataCollection) -> pd.DataFrame:
    """
    Apply adstock transformations to the media variables in the dataset.
    """
    data = mmm_data.intermediate_data.dt_mod.copy()
    all_media = mmm_data.model_parameters.all_media
    adstock_type = mmm_data.adstock
    hyperparameters = mmm_data.hyperparameters.hyperparameters

    for media in all_media:
        if adstock_type == AdstockType.GEOMETRIC:
            theta = hyperparameters[media].thetas[
                0
            ]  # Using the first value of the range
            transformed = transform_adstock(
                data[media].values, "geometric", theta=theta
            )
        elif adstock_type == AdstockType.WEIBULL_CDF:
            shape = hyperparameters[media].shapes[0]
            scale = hyperparameters[media].scales[0]
            transformed = transform_adstock(
                data[media].values, "weibull_cdf", shape=shape, scale=scale
            )
        elif adstock_type == AdstockType.WEIBULL_PDF:
            shape = hyperparameters[media].shapes[0]
            scale = hyperparameters[media].scales[0]
            transformed = transform_adstock(
                data[media].values, "weibull_pdf", shape=shape, scale=scale
            )
        else:
            raise ValueError(f"Unsupported adstock type: {adstock_type}")

        # Apply saturation
        alpha = hyperparameters[media].alphas[0]
        gamma = hyperparameters[media].gammas[0]
        saturated = saturation_hill(transformed["x_decayed"], alpha, gamma)

        # Update the dataframe with transformed and saturated values
        data[f"{media}_adstocked"] = transformed["x_decayed"]
        data[f"{media}_saturated"] = saturated

    return data


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration purposes."""
    date_range = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "DATE": date_range,
            "revenue": np.random.normal(1000, 100, len(date_range)),
            "tv_S": np.random.uniform(500, 1500, len(date_range)),
            "ooh_S": np.random.uniform(200, 800, len(date_range)),
            "print_S": np.random.uniform(300, 1000, len(date_range)),
            "facebook_S": np.random.uniform(400, 1200, len(date_range)),
            "search_S": np.random.uniform(300, 900, len(date_range)),
            "newsletter": np.random.uniform(0, 1, len(date_range)),
        }
    )

    # Add some seasonality and trend to revenue
    data["revenue"] += (
        np.sin(np.arange(len(data)) * 2 * np.pi / 365) * 200
        + np.arange(len(data)) * 0.5
    )
    print("Generated data shape:", data.shape)
    print("Generated data head:")
    print(data.head())
    return data


def prepare_mmmdata_collection(data: pd.DataFrame) -> MMMDataCollection:
    """Prepare MMMDataCollection from the input data."""
    print("Input data shape:", data.shape)
    print("Input data head:")
    print(data.head())

    mmmdata_spec = MMMData.MMMDataSpec(
        dep_var="revenue",
        dep_var_type=DependentVarType.REVENUE,
        date_var="DATE",
        paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
        paid_media_signs=[PaidMediaSigns.POSITIVE] * 5,
        organic_vars=["newsletter"],
        organic_signs=[PaidMediaSigns.POSITIVE],
        context_vars=[],
        context_signs=[],
        factor_vars=[],
        prophet_vars=["trend", "season", "holiday"],
        prophet_country="DE",
    )

    mmmdata = MMMData(data=data, mmmdata_spec=mmmdata_spec)

    print("MMMData created. Data shape:", mmmdata.data.shape)
    print("MMMData head:")
    print(mmmdata.data.head())

    # Create a simple holidays DataFrame
    holidays_df = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=data["DATE"].min(), end=data["DATE"].max(), freq="D"
            ),
            "holiday": "No Holiday",  # You might want to replace this with actual holiday data
            "country": "DE",
        }
    )

    # Initialize HolidaysData correctly
    holiday_data = HolidaysData(
        dt_holidays=holidays_df,
        prophet_vars=[ProphetVariableType.HOLIDAY],  # Assuming you have this enum
        prophet_signs=[ProphetSigns.POSITIVE],  # Assuming you have this enum
        prophet_country="DE",
        day_interval=1,  # Assuming daily data
    )

    adstock = AdstockType.GEOMETRIC

    # Initialize hyperparameters
    channel_hyperparameters = {
        "tv_S": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.3, 0.8]
        ),
        "ooh_S": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]
        ),
        "print_S": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]
        ),
        "facebook_S": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0, 0.3]
        ),
        "search_S": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0, 0.3]
        ),
        "newsletter": ChannelHyperparameters(
            alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]
        ),
    }

    hyperparameters = Hyperparameters(hyperparameters=channel_hyperparameters)
    # TODO: Add actual calibration input if available
    calibration_input = None

    intermediate_data = IntermediateData(
        dt_mod=data.copy(),  # Use a copy of the original data as a starting point
        dt_modRollWind=data.copy(),  # Same as above
        xDecompAggPrev=None,  # This might be populated later in the process
    )

    model_parameters = ModelParameters(
        dayInterval=1,
        intervalType="day",
        mediaVarCount=5,
        exposure_vars=[],
        all_media=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S", "newsletter"],
        all_ind_vars=[
            "tv_S",
            "ooh_S",
            "print_S",
            "facebook_S",
            "search_S",
            "newsletter",
        ],
        factor_vars=[],
        unused_vars=[],
    )

    time_window = TimeWindow(
        window_start=data["DATE"].min(),
        window_end=data["DATE"].max(),
        rollingWindowStartWhich=0,
        rollingWindowEndWhich=len(data) - 1,
        rollingWindowLength=len(data),
        totalObservations=len(data),
        refreshAddedStart=data["DATE"].min(),
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

    print("MMMDataCollection created.")
    print("MMMDataCollection attributes:")
    for attr, value in mmm_data.__dict__.items():
        print(f"{attr}: {type(value)}")

    print("Applying adstock transformations...")
    transformed_data = apply_adstock_transformations(mmm_data)
    # Update the MMMDataCollection with the new transformed data
    updated_mmm_data = MMMDataCollection.update(obj=mmm_data, dt_mod=transformed_data)

    print("MMMDataCollection updated.")
    print("MMMDataCollection attributes:")
    for attr, value in updated_mmm_data.__dict__.items():
        print(f"{attr}: {type(value)}")
    return updated_mmm_data


def main():
    print("Starting MMM Modeling Demo")

    # Step 1: Load data
    print("Generating sample data...")
    data = generate_sample_data()
    print("Data prepared successfully.")

    # Step 2: Model specification
    print("Preparing MMM data collection...")
    mmm_data = prepare_mmmdata_collection(data)
    print("MMM data collection prepared.")

    # You can add some print statements here to show the effects of the transformations
    print("Sample of transformed data:")
    print(mmm_data.intermediate_data.dt_mod.head())

    # Step 3: Build initial model
    print("Running MMM model...")
    model_executor = MMMModelExecutor()
    trials_config = TrialsConfig(
        num_trials=5,
        num_iterations_per_trial=2000,
        timeseries_validation=True,
        add_penalty_factor=False,
    )
    model_output = model_executor.model_run(
        mmmdata_collection=mmm_data, trials_config=trials_config, seed=42, quiet=False
    )
    print("Model run completed.")
    print("=========================")
    print("Model Output after run_model:")
    for attr, value in model_output.__dict__.items():
        print(f"{attr}: {type(value)}")
    if hasattr(model_output, "model_output"):
        print("model_output.model_output:")
        for attr, value in model_output.model_output.__dict__.items():
            print(f"  {attr}: {type(value)}")
        if hasattr(model_output.model_output, "trials"):
            print("Number of trials:", len(model_output.model_output.trials))
            print("Trial data:")
            for i, trial in enumerate(model_output.model_output.trials):
                print(f"  Trial {i}:")
                for trial_attr, trial_value in trial.__dict__.items():
                    print(f"    {trial_attr}: {type(trial_value)}")
    print("=========================")

    # Step 4: Analyze results
    print("Optimizing Pareto front...")
    pareto_optimizer = ParetoOptimizer()
    pareto_results = pareto_optimizer.pareto_optimize(
        mmmdata_collection=mmm_data,
        modeloutput=model_output,
        pareto_fronts="auto",
        min_candidates=100,
    )
    print(
        f"Number of Pareto-optimal solutions: {len(pareto_results['pareto_solutions'])}"
    )

    print("Analyzing model clusters...")
    cluster_analyzer = ModelClustersAnalyzer()

    # Debug print statements
    print("Model output type:", type(model_output.model_output))
    print("Model output attributes:")
    for attr, value in model_output.model_output.__dict__.items():
        print(f"  {attr}: {type(value)}")

    if (
        hasattr(model_output.model_output, "trials")
        and model_output.model_output.trials
    ):
        print("Number of trials:", len(model_output.model_output.trials))
        print("First trial data:")
        for attr, value in vars(model_output.model_output.trials[0]).items():
            print(f"  {attr}: {value}")
    else:
        print("No trials data found in model output.")

    cluster_results = cluster_analyzer.model_clusters_analyze(
        input_data=model_output.model_output,
        dep_var_type="continuous",
        cluster_by="hyperparameters",
        k="auto",
        quiet=False,  # Set to False to see more output
    )
    if cluster_results is not None:
        print("=========================")
        print("Cluster results: ", cluster_results)
        print("=========================")
        print(f"Number of clusters: {len(cluster_results)}")
    else:
        print("Clustering could not be performed.")

    print("Evaluating model performance...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model_output.model_output)

    print("Average metrics across all trials:")
    for metric, value in metrics["average_metrics"].items():
        print(f"  {metric}: {value:.6f}")

    print("\nMetrics for each trial:")
    for trial_id, trial_metrics in metrics["per_trial_metrics"].items():
        print(f"  Trial {trial_id}:")
        for metric, value in trial_metrics.items():
            print(f"    {metric}: {value:.6f}")

    # Step 5: Model refresh (optional)
    print("Refreshing model...")
    model_refresher = ModelRefresh()
    refresh_config = ModelRefreshConfig(
        refresh_steps=2, refresh_mode="manual", refresh_iters=500, refresh_trials=2
    )
    refreshed_output = model_refresher.model_refresh(
        mmmdata_collection=mmm_data,
        model_output_collection=model_output,
        refresh_config=refresh_config,
    )
    print("Model refreshed successfully.")

    print("Analyzing model convergence...")
    model_convergence = ModelConvergence()
    convergence_result: ConvergenceResult = model_convergence.converge(
        model_output=model_output.model_output,
        n_cuts=20,
        sd_qtref=3,
        med_lowb=2,
        nrmse_win=(0, 0.998),
    )
    print("Convergence analysis completed.")
    print("Convergence messages:")
    print(convergence_result["conv_msg"])
    # # Step 5: Run allocator
    # print("Running budget allocator...")

    # print("MMM Data Collection:", mmm_data)
    # allocator = RobynAllocator(
    #     mmm_data, model_output, select_model=model_output.model_output.trials[0].solID
    # )
    # allocator_result = allocator.allocate(
    #     scenario="max_response",
    #     total_budget=None,  # Use historical budget
    #     date_range="all",
    #     channel_constr_low=0.7,
    #     channel_constr_up=1.2,
    #     channel_constr_multiplier=3,
    #     plots=True,
    #     export=True,
    #     quiet=False,
    # )
    # print("Budget allocation completed.")
    # print("Allocation Summary:")
    # print(allocator_result["allocation_results"])

    # if allocator_result["plots"]:
    #     print("Allocation plots generated. Check the output folder.")

    print("MMM Modeling Demo completed.")


if __name__ == "__main__":
    main()
