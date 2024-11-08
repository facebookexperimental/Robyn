# pyre-strict

# TODO This needs to be rewritten to match the new structure of the codebase
# TODO Add separate methods if state is loaded from robyn_object or json_file for each method

import logging
import pandas as pd

from robyn.allocator.entities.allocation_config import AllocationConfig
from robyn.allocator.entities.allocation_constraints import AllocationConstraints
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData
from robyn.modeling.model_executor import ModelExecutor
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation
from robyn.data.validation.mmmdata_validation import MMMDataValidation
from robyn.visualization.feature_visualization import FeaturePlotter
import matplotlib.pyplot as plt


class Robyn:
    def __init__(self, working_dir: str):
        """
        Initializes the Robyn object with a working directory.

        Args:
            working_dir (str): The path to the working directory.
        """
        self.working_dir = working_dir
        self.logger = logging.getLogger()
        self.logger.info("Robyn initialized with working directory: %s", working_dir)
        self.mmm_data: MMMData = None
        self.holidays_data: HolidaysData = None
        self.hyperparameters: Hyperparameters = None
        self.calibration_input: CalibrationInput = None

    # Load input data for the first time and validates
    def initialize(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
    ) -> None:
        """
        Loads input data for the first time and validates it.
        Calls validate from MMMDataValidation, HolidaysDataValidation, HyperparametersValidation, and CalibrationInputValidation.

        Args:
            mmm_data (MMMData): The MMM data object.
            holidays_data (HolidaysData): The holidays data object.
            hyperparameters (HyperParametersConfig): The hyperparameters configuration object.
            calibration_input (CalibrationInputConfig): The calibration input configuration object.
        """
        mmm_data_validation = MMMDataValidation(mmm_data)
        holidays_data_validation = HolidaysDataValidation(holidays_data)
        hyperparameters_validation = HyperparametersValidation(hyperparameters)

        mmm_data_validation.validate()
        holidays_data_validation.validate()
        hyperparameters_validation.validate()

        self.mmm_data = mmm_data
        self.holidays_data = holidays_data
        self.hyperparameters = hyperparameters

        print("Validation complete")

    def feature_engineering(self, plot=True) -> FeaturizedMMMData:
        """
        Perform feature engineering on the data.

        This method processes the data to create new features that can be used in
        the marketing mix model (MMM). The exact transformations and feature
        creation steps are not specified in this placeholder method.

        Args:
            plot (bool): If True, generate and display plots for the engineered features.

        Returns:
            FeaturizedMMMData: The data with new features added, ready for use in the MMM.
        """
        feature_engineering = FeatureEngineering(self.mmm_data, self.hyperparameters, self.holidays_data)
        featurized_mmm_data = feature_engineering.perform_feature_engineering()
        if plot:
            feature_plotter = FeaturePlotter(self.mmm_data, self.hyperparameters)
            for channel in self.mmm_data.mmmdata_spec.paid_media_spends:
                try:
                    fig = feature_plotter.plot_spend_exposure(featurized_mmm_data, channel)
                    plt.show()
                except ValueError as e:
                    print(f"Skipping {channel}: {str(e)}")
        return featurized_mmm_data

    def model_run(
        self,
        feature_plots=True,
        trials_config=trials_config,
        ts_validation=False,
        add_penalty_factor=False,
        rssd_zero_penalty=True,
        cores=16,  # max of cores available or user provided
        nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
        intercept=True,
        intercept_sign="non_negative",
        model_name=Models.RIDGE,
        plot=True,
        export=True,
        run_calibration=False,
        calibration_input=None,
        model_output_plot=True,
        pareto_fronts="auto",
        min_candidates=100,
        run_cluster=True,
        cluster_config: ClusteringConfig = None,
    ):
        """
        Runs the model with the specified configuration and parameters.

        Args:
            feature_plots (bool): Whether to plot feature engineering results. Default is True.
            trials_config (TrialsConfig): Configuration for the trials. Default is trials_config.
            ts_validation (bool): Whether to perform time series validation. Default is False.
            add_penalty_factor (bool): Whether to add a penalty factor. Default is False.
            rssd_zero_penalty (bool): Whether to apply zero penalty for RSSD. Default is True.
            cores (int): Number of cores to use. Default is 16.
            nevergrad_algo (NevergradAlgorithm): Algorithm to use for optimization. Default is NevergradAlgorithm.TWO_POINTS_DE.
            intercept (bool): Whether to include an intercept in the model. Default is True.
            intercept_sign (str): Sign constraint for the intercept. Default is "non_negative".
            model_name (Models): Name of the model to use. Default is Models.RIDGE.
            plot (bool): Whether to plot the model results. Default is True.
            export (bool): Whether to export the model results. Default is True.
            run_calibration (bool): Whether to run calibration. Default is False.
            calibration_input: Input data for calibration. Default is None.
            model_output_plot (bool): Whether to plot the model output. Default is True.
            pareto_fronts (str): Configuration for Pareto fronts. Default is "auto".
            min_candidates (int): Minimum number of candidates. Default is 100.
            run_cluster (bool): Whether to run clustering. Default is True.
            cluster_config (ClusteringConfig): Configuration for clustering. Default is None.

        Returns:
            None
        """
        feature_engineering = FeatureEngineering(self.mmm_data, self.hyperparameters, self.holidays_data)
        featurized_mmm_data = feature_engineering.perform_feature_engineering()

        if feature_plots:
            # Create a FeaturePlotter instance
            feature_plotter = FeaturePlotter(self.mmm_data, self.hyperparameters)

            # Plot spend-exposure relationship for each channel
            for channel in self.mmm_data.mmmdata_spec.paid_media_spends:
                try:
                    fig = feature_plotter.plot_spend_exposure(featurized_mmm_data, channel)
                    plt.show()
                except ValueError as e:
                    print(f"Skipping {channel}: {str(e)}")

        # Setup ModelExecutor
        model_executor = ModelExecutor(
            mmmdata=self.mmm_data,
            holidays_data=self.holidays_data,
            hyperparameters=self.hyperparameters,
            calibration_input=None,  # Add calibration input if available
            featurized_mmm_data=featurized_mmm_data,
        )

        # Setup TrialsConfig
        trials_config = TrialsConfig(iterations=2000, trials=5)  # Set to the number of cores you want to use

        print(
            f">>> Starting {trials_config.trials} trials with {trials_config.iterations} iterations each using {NevergradAlgorithm.TWO_POINTS_DE.value} nevergrad algorithm on x cores..."
        )

        # Run the model
        model_outputs = model_executor.model_run(
            trials_config=trials_config,
            ts_validation=False,  # changed from True to False -> deactivate
            add_penalty_factor=False,
            rssd_zero_penalty=True,
            cores=8,
            nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
            intercept=True,
            intercept_sign="non_negative",
            model_name=Models.RIDGE,
        )
        print("Model training complete.")

    def build_models(
        self,
        trials_config: TrialsConfig = trials_config,
        ts_validation=False,
        add_penalty_factor=False,
        rssd_zero_penalty=True,
        cores=16,  # max of cores available or user provided
        nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
        intercept=True,
        intercept_sign="non_negative",
        model_name=Models.RIDGE,
        plot=True,
        export=True,
        run_calibration=False,
        calibration_input=None,
    ) -> ModelOutputs:
        pass

    def evaluate_models(
        self, pareto_fronts="auto", min_candidates=100, run_cluster=True, cluster_config: ClusteringConfig = None
    ) -> None:
        pass

    def budget_allocator(
        self,
        select_model,
        allocation_contstraints: AllocationConstraints,
        allocator_config: AllocationConfig,
        plot=True,
        export_allocation_result=False,
    ) -> AllocationResult:
        pass
