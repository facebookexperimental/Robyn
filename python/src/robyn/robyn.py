# pyre-strict

# TODO This needs to be rewritten to match the new structure of the codebase
# TODO Add separate methods if state is loaded from robyn_object or json_file for each method

import logging
import pandas as pd
from robyn.data.entities.enums import DependentVarType

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
from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.clustering.cluster_builder import ClusterBuilder
from robyn.modeling.clustering.clustering_config import ClusterBy, ClusteringConfig
from robyn.allocator.budget_allocator import BudgetAllocator


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

    def feature_engineering(self, plot: bool = True, export: bool = False) -> FeaturizedMMMData:
        if not all([self.mmm_data, self.hyperparameters, self.holidays_data]):
            raise ValueError("Please initialize Robyn with mmm_data, hyperparameters, and holidays_data first")
        
        try:
            # Initialize FeatureEngineering and process data
            feature_engineering = FeatureEngineering(self.mmm_data, self.hyperparameters, self.holidays_data)
            featurized_mmm_data = feature_engineering.perform_feature_engineering()
            
            # Plot if requested
            if plot or export:
                feature_plotter = FeaturePlotter(self.mmm_data, self.hyperparameters)
                
                # Get channels with spend-exposure data from featurized results
                channels_with_data = []
                if hasattr(featurized_mmm_data, 'modNLS') and 'results' in featurized_mmm_data.modNLS:
                    channels_with_data = [result['channel'] for result in featurized_mmm_data.modNLS['results']]
                
                # Create plots directory if exporting
                if export:
                    import os
                    plots_dir = os.path.join(self.working_dir, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                
                for channel in channels_with_data:
                    try:
                        fig = feature_plotter.plot_spend_exposure(featurized_mmm_data, channel)
                        
                        # Save plot if export is True
                        if export:
                            plot_path = os.path.join(plots_dir, f'spend_exposure_{channel}.png')
                            plt.savefig(plot_path)
                        
                        # Display plot if requested
                        if plot:
                            plt.show()
                        else:
                            plt.close()
                            
                    except ValueError as e:
                        logging.debug(f"Skipping {channel}: {str(e)}")
                    except Exception as e:
                        logging.error(f"Error plotting {channel}: {str(e)}")
            
            return featurized_mmm_data
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise

    def model_e2e_run(
        self,
        trials_config=TrialsConfig(iterations=54, trials=5),
        ts_validation=True,
        add_penalty_factor=False,
        rssd_zero_penalty=True,
        cores=8,
        nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
        intercept=True,
        intercept_sign="non_negative",
        model_name=Models.RIDGE,
        plot=True,
        export=True,
        run_calibration=False,
        calibration_input=None,
        pareto_fronts="auto",
        min_candidates=100,
        run_cluster=False,
        cluster_config: ClusteringConfig = None,
    ):
        """
        Runs the model end-to-end with the specified configuration and parameters.
        """
        # Step 1: Feature Engineering
        featurized_mmm_data = self.feature_engineering(plot=plot)
        self.featurized_mmm_data = featurized_mmm_data

        # Step 2: Build Models
        model_outputs = self.build_models(
            trials_config=trials_config,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            rssd_zero_penalty=rssd_zero_penalty,
            cores=cores,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            model_name=model_name,
            plot=plot,
            export=export,
            run_calibration=run_calibration,
            calibration_input=calibration_input,
        )
        self.model_outputs = model_outputs  # Store for later use
        # Step 3: Evaluate Models
        self.evaluate_models(
            pareto_fronts=pareto_fronts,
            min_candidates=min_candidates,
            run_cluster=run_cluster,
            cluster_config=cluster_config,
            plot=plot,
            export=export,
        )
        print("Model training and evaluation complete.")

    def build_models(
        self,
        trials_config: TrialsConfig,
        ts_validation=True,
        add_penalty_factor=False,
        rssd_zero_penalty=True,
        cores=16,
        nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
        intercept=True,
        intercept_sign="non_negative",
        model_name=Models.RIDGE,
        plot=True,
        export=True,
        run_calibration=False,
        calibration_input=None,
    ) -> ModelOutputs:
        """
        Builds models using the specified configuration and parameters.
        """
        # Initialize the ModelExecutor with necessary data
        model_executor = ModelExecutor(
            mmmdata=self.mmm_data,
            holidays_data=self.holidays_data,
            hyperparameters=self.hyperparameters,
            calibration_input=calibration_input,
            featurized_mmm_data=self.featurized_mmm_data,
        )
        # Run the model
        model_outputs = model_executor.model_run(
            trials_config=trials_config,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            rssd_zero_penalty=rssd_zero_penalty,
            cores=cores,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            model_name=model_name,
        )

        # if(plot)
            #viz_build_models()
        return model_outputs

    def visualize_outputs(self, plot=True):
        # Add logic to visualize model outputs
        pass

    def export_outputs(self, export=True):
        # Add logic to export model outputs
        pass

    def evaluate_models(
        self,
        pareto_fronts="auto",
        min_candidates=100,
        run_cluster=True,
        cluster_config: ClusteringConfig = None,
        plot=False,
        export=False,
    ) -> None:
        # Perform Pareto optimization
        pareto_result = self.pareto_optimization(pareto_fronts, min_candidates, plot, export)
        self.pareto_result = pareto_result
        if run_cluster:
            # Perform clustering on the Pareto-optimized results
            cluster_results = self.cluster_models(pareto_result, cluster_config, plot, export)
        print("Model evaluation complete.")
        # if plot == True:
            # instantiate pareto plotter and cluster plotter and plot the graphs

    def pareto_optimization(self, pareto_fronts: str, min_candidates: int, plot: bool, export: bool) -> ParetoResult:
        # Create ParetoOptimizer instance
        pareto_optimizer = ParetoOptimizer(
            mmm_data=self.mmm_data,
            model_outputs=self.model_outputs,
            hyperparameter=self.hyperparameters,
            featurized_mmm_data=self.featurized_mmm_data,
            holidays_data=self.holidays_data,
        )
        # Run optimize function
        pareto_result = pareto_optimizer.optimize(pareto_fronts=pareto_fronts, min_candidates=min_candidates)

        # Visualize and/or export results if required
        if plot:
            self.visualize_outputs(pareto_result)  # Remove plot=plot
        if export:
            self.export_outputs(pareto_result)  # Remove export=export

        return pareto_result

    def cluster_models(self, pareto_result: ParetoResult, cluster_config: ClusteringConfig, plot: bool, export: bool):
        # Instantiate ClusterBuilder with the Pareto result
        cluster_builder = ClusterBuilder(pareto_result=pareto_result)
        # Use provided cluster_config or create a default one
        if not cluster_config:
            cluster_config = ClusteringConfig(
                dep_var_type=DependentVarType(self.mmm_data.mmmdata_spec.dep_var_type),
                cluster_by=ClusterBy.HYPERPARAMETERS,
                max_clusters=30,
                min_clusters=3,
                weights=[1.0, 1.0, 1.0],
            )
        # Perform clustering
        cluster_results = cluster_builder.cluster_models(cluster_config)
        # Visualize and/or export clustering results if required
        if plot:
            self.visualize_outputs(cluster_results, plot=plot)
        if export:
            self.export_outputs(cluster_results, export=export)
        return cluster_results

    def budget_allocator(
        self,
        select_model,
        allocation_constraints: AllocationConstraints,
        allocator_config: AllocationConfig,
        plot=True,
        export=False,
    ) -> AllocationResult:

        if not select_model:
            select_model = self.model_outputs.select_id

        # Initialize budget allocator
        allocator = BudgetAllocator(
            mmm_data=self.mmm_data,
            featurized_mmm_data=self.featurized_mmm_data,
            model_outputs=self.model_outputs,
            pareto_result=self.pareto_result,
            select_model=select_model,
        )
        # Run optimization
        result = allocator.allocate(allocator_config)
        # Print results
        print(
            f"""
            Model ID: {select_model}
            Scenario: {allocator_config.scenario}
            Use case: {result.metrics.get('use_case', '')}
            Window: {result.metrics.get('date_range_start')}:{result.metrics.get('date_range_end')} ({result.metrics.get('n_periods')} {self.mmm_data.mmmdata_spec.interval_type})
            Dep. Variable Type: {self.mmm_data.mmmdata_spec.dep_var_type}
            Media Skipped: {result.metrics.get('skipped_channels', 'None')}
            Relative Spend Increase: {result.metrics.get('spend_lift_pct', 0):.1f}% ({result.metrics.get('spend_lift_abs', 0):+.0f}K)
            Total Response Increase (Optimized): {result.metrics.get('response_lift', 0)*100:.1f}%
            Allocation Summary:
            """
        )
        # Print channel-level results
        for channel in self.mmm_data.mmmdata_spec.paid_media_spends:
            current = result.optimal_allocations[result.optimal_allocations["channel"] == channel].iloc[0]
            print(
                f"""
                - {channel}:
                  Optimizable bound: [{(current['constr_low']-1)*100:.0f}%, {(current['constr_up']-1)*100:.0f}%],
                  Initial spend share: {current['current_spend_share']*100:.2f}% -> Optimized bounded: {current['optimal_spend_share']*100:.2f}%
                  Initial response share: {current['current_response_share']*100:.2f}% -> Optimized bounded: {current['optimal_response_share']*100:.2f}%
                  Initial abs. mean spend: {current['current_spend']/1000:.3f}K -> Optimized: {current['optimal_spend']/1000:.3f}K [Delta = {(current['optimal_spend']/current['current_spend']-1)*100:.0f}%]
                """
            )
        # Visualize and/or export allocation results if required
        if plot:
            self.visualize_allocation(result, plot=plot)
        if export:
            self.export_allocation(result, export=export)
        return result
