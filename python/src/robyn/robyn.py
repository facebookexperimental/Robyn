# pyre-strict

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
import copy
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.validation.mmmdata_validation import MMMDataValidation
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation

from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.model_executor import ModelExecutor

from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
from robyn.modeling.clustering.cluster_builder import ClusterBuilder
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData

from robyn.allocator.optimizer import BudgetAllocator
from robyn.allocator.entities.allocation_params import AllocatorParams
from robyn.allocator.entities.allocation_result import AllocationResult

from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.reporting.onepager_reporting import OnePager
from robyn.visualization.allocator_visualizer import AllocatorPlotter
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from robyn.visualization.feature_visualization import FeaturePlotter
from robyn.visualization.model_convergence_visualizer import ModelConvergenceVisualizer
from robyn.visualization.pareto_visualizer import ParetoVisualizer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Robyn:
    """Client interface for the Robyn Marketing Mix Modeling framework."""

    def __init__(
        self,
        working_dir: str,
        console_log_level: Optional[str] = "INFO",
        file_log_level: Optional[str] = "DEBUG",
    ):
        """
        Initialize Robyn.

        Args:
            working_dir: Working directory for logs and outputs
            console_level: Logging level for console output
            file_level: Logging level for file output
            working_dir: Working directory for logs and outputs
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging(console_log_level, file_log_level)
        logger.info("Initialized Robyn in %s", working_dir)

        # Core components initialized to None
        self.mmm_data = None
        self.holidays_data = None
        self.hyperparameters = None
        self.featurized_mmm_data = None
        self.model_outputs = None
        self.pareto_result = None
        self.cluster_result = None

    def initialize(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
    ) -> None:
        """
        Initialize and validate input data.

        Args:
            mmm_data: Marketing mix modeling data
            holidays_data: Holiday calendar data
            hyperparameters: Model hyperparameters

        Raises:
            ValidationError: If any validation fails
        """
        logger.info("Validating input data")
        try:
            # Run validations
            MMMDataValidation(mmm_data).validate()
            HolidaysDataValidation(holidays_data).validate()
            HyperparametersValidation(hyperparameters).validate()

            # Store validated data
            self.mmm_data = mmm_data
            self.holidays_data = holidays_data
            self.hyperparameters = hyperparameters

            # # Feature engineering
            # self.featurized_mmm_data = self.feature_engineering()
            logger.info("Data initialization complete")

        except Exception as e:
            logger.error("Initialization failed: %s", str(e))
            raise

    def feature_engineering(
        self,
        display_plots: bool = True,
        export_plots: bool = False,
    ) -> FeaturizedMMMData:
        """
        Perform feature engineering on the MMM data.

        This method initializes the FeatureEngineering class with the provided MMM data,
        hyperparameters, and holidays data, and performs feature engineering. Optionally,
        it can display and/or export plots of the engineered features.

        Args:
            display_plots (bool): If True, display plots of the engineered features. Default is True.
            export_plots (bool): If True, export plots of the engineered features. Default is False.

        Returns:
            FeaturizedMMMData: The featurized MMM data.

        Raises:
            ValueError: If the MMM data, hyperparameters, or holidays data are not initialized.
            Exception: If an error occurs during feature engineering.
        """
        logger.info("Performing feature engineering")
        if not all([self.mmm_data, self.hyperparameters, self.holidays_data]):
            raise ValueError(
                "Please initialize Robyn with mmm_data, hyperparameters, and holidays_data first"
            )

        try:
            # Initialize FeatureEngineering and process data
            feature_engineering = FeatureEngineering(
                self.mmm_data, self.hyperparameters, self.holidays_data
            )
            self.featurized_mmm_data = feature_engineering.perform_feature_engineering()

            if display_plots or export_plots:
                feature_plotter = FeaturePlotter(
                    self.mmm_data, self.hyperparameters, self.featurized_mmm_data
                )
                feature_plotter.plot_all(display_plots, self.working_dir)

            return self.featurized_mmm_data

        except Exception as e:
            logging.error("Error in feature engineering: %s", str(e))
            raise

    def train_models(
        self,
        trials_config: Optional[TrialsConfig] = None,
        ts_validation: Optional[bool] = False,
        add_penalty_factor: Optional[bool] = False,
        rssd_zero_penalty: Optional[bool] = True,
        nevergrad_algo: Optional[str] = NevergradAlgorithm.TWO_POINTS_DE,
        model_name: Optional[str] = Models.RIDGE,
        cores: Optional[int] = 16,
        display_plots: bool = True,
        export_plots: bool = False,
    ) -> ModelOutputs:
        """
        Trains the specified models using the provided configuration and parameters.

        Args:
            trials_config (Optional[TrialsConfig]): Configuration for the trials, including the number of trials and iterations. Defaults to None.
            ts_validation (Optional[bool]): Whether to perform time series validation. Defaults to False.
            add_penalty_factor (Optional[bool]): Whether to add a penalty factor during training. Defaults to False.
            nevergrad_algo (Optional[str]): The Nevergrad algorithm to use for optimization. Defaults to NevergradAlgorithm.TWO_POINTS_DE.
            model_name (Optional[str]): The name of the model to train. Defaults to Models.RIDGE.
            cores (Optional[int]): The number of CPU cores to use for training. Defaults to 16.
            display_plots (bool): Whether to display plots after training. Defaults to True.
            export_plots (bool): Whether to export plots after training. Defaults to False.

        Returns:
            ModelOutputs: The outputs of the trained models.

        Raises:
            Exception: If training the models fails.
        """

        self._validate_initialization()

        try:
            logger.info("Training models")
            trials_config = trials_config or TrialsConfig(trials=5, iterations=2000)
            model_executor = ModelExecutor(
                mmmdata=self.mmm_data,
                holidays_data=self.holidays_data,
                hyperparameters=self.hyperparameters,
                calibration_input=None,
                featurized_mmm_data=self.featurized_mmm_data,
            )

            self.model_outputs = model_executor.model_run(
                trials_config=trials_config,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                rssd_zero_penalty=rssd_zero_penalty,
                nevergrad_algo=nevergrad_algo,
                model_name=model_name,
                cores=cores,
            )
        except Exception as e:
            logger.error("Training models failed: %s", str(e))
            raise

        if display_plots or export_plots:
            visualizer = ModelConvergenceVisualizer(self.model_outputs)
            visualizer.plot_all(display_plots, self.working_dir)

    def evaluate_models(
        self,
        pareto_config: Optional[Dict] = None,
        cluster_config: Optional[ClusteringConfig] = None,
        display_plots: bool = True,
        export_plots: bool = False,
    ) -> None:
        """
        Evaluates the trained models using Pareto optimization and optional clustering.
        Parameters:
        pareto_config (Optional[Dict]): Configuration for Pareto optimization. If not provided, default settings will be used.
        cluster_config (Optional[ClusteringConfig]): Configuration for clustering the models. If not provided, clustering will be skipped.
        display_plots (bool): If True, plots will be displayed. Default is True.
        export_plots (bool): If True, plots will be exported. Default is False.
        Raises:
        ValueError: If models have not been trained before evaluation.
        Exception: If any error occurs during the evaluation process.
        Returns:
        None
        """
        if not self.model_outputs:
            raise ValueError("Must train models before evaluation")

        try:
            logger.info("Evaluating models")

            # Pareto optimization
            pareto_config = pareto_config or {
                "pareto_fronts": "auto",
                "min_candidates": 100,
            }
            pareto_optimizer = ParetoOptimizer(
                mmm_data=self.mmm_data,
                model_outputs=self.model_outputs,
                hyperparameter=self.hyperparameters,
                featurized_mmm_data=self.featurized_mmm_data,
                holidays_data=self.holidays_data,
            )
            self.pareto_result = pareto_optimizer.optimize(**pareto_config)
            unfiltered_pareto_result = copy.deepcopy(self.pareto_result)

            # Optional clustering
            is_clustered = False
            if cluster_config:
                cluster_builder = ClusterBuilder(self.pareto_result)
                self.cluster_result = cluster_builder.cluster_models(cluster_config)
                is_clustered = True

            self.pareto_result = ParetoUtils().process_pareto_clustered_results(
                self.pareto_result, self.cluster_result, is_clustered
            )
            if display_plots or export_plots:
                pareto_visualizer = ParetoVisualizer(
                    pareto_result=self.pareto_result,
                    mmm_data=self.mmm_data,
                    holiday_data=self.holidays_data,
                    hyperparameter=self.hyperparameters,
                    featurized_mmm_data=self.featurized_mmm_data,
                    unfiltered_pareto_result=unfiltered_pareto_result,
                    model_outputs=self.model_outputs,
                )
                pareto_visualizer.plot_all(display_plots, self.working_dir)
                if self.cluster_result:
                    cluster_visualizer = ClusterVisualizer(
                        self.pareto_result,
                        self.cluster_result,
                        self.mmm_data,
                    )
                    cluster_visualizer.plot_all(display_plots, self.working_dir)
                    plt.close("all")
            logger.info("Model evaluation complete")

        except Exception as e:
            logger.error("Model evaluation failed: %s", str(e))
            raise

    def train_and_evaluate_models(
        self,
        trials_config: Optional[TrialsConfig] = None,
        ts_validation: Optional[bool] = False,
        add_penalty_factor: Optional[bool] = False,
        rssd_zero_penalty: Optional[bool] = True,
        nevergrad_algo: Optional[str] = NevergradAlgorithm.TWO_POINTS_DE,
        model_name: Optional[str] = Models.RIDGE,
        cores: Optional[int] = 16,
        pareto_config: Optional[Dict] = None,
        cluster_config: Optional[ClusteringConfig] = None,
        display_plots: bool = True,
        export_plots: bool = False,
    ) -> None:
        """
        Trains and evaluates models based on the provided configurations.
        Args:
            trials_config (Optional[TrialsConfig]): Configuration for the trials during model training.
            model_config (Optional[Dict]): Configuration for the models to be trained.
            pareto_config (Optional[Dict]): Configuration for Pareto front evaluation.
            cluster_config (Optional[ClusteringConfig]): Configuration for clustering during evaluation.
            display_plots (bool): Whether to display plots during training and evaluation. Defaults to True.
            export_plots (bool): Whether to export plots during training and evaluation. Defaults to False.
        Returns:
            None
        """
        self.train_models(
            trials_config,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            nevergrad_algo,
            model_name,
            cores,
            display_plots,
            export_plots,
        )
        self.evaluate_models(pareto_config, cluster_config, display_plots, export_plots)

    def optimize_budget(
        self,
        allocator_params: AllocatorParams,
        select_model: Optional[str] = None,
        display_plots: bool = True,
        export_plots: bool = False,
    ) -> AllocationResult:
        """
        Optimize the budget allocation based on the given configuration.
        Args:
            allocation_config (AllocationConfig): Configuration for budget allocation.
            select_model (Optional[str], optional): Specific model to use for allocation. Defaults to None.
            display_plots (bool, optional): Whether to display plots. Defaults to True.
            export_plots (bool, optional): Whether to export plots. Defaults to False.
        Returns:
            AllocationResult: The result of the budget allocation.
        Raises:
            ValueError: If models have not been evaluated before budget optimization.
            Exception: If budget optimization fails.
        """

        if not self.pareto_result:
            raise ValueError("Must evaluate models before budget optimization")

        try:
            logger.info("Optimizing budget allocation")

            if select_model is None:
                pareto_solutions = self.pareto_result.pareto_solutions
                if (
                    pareto_solutions
                    and pareto_solutions[0] is not None
                    and pareto_solutions[0] != ""
                ):
                    select_model = pareto_solutions[0]
                elif (
                    len(pareto_solutions) > 1
                    and pareto_solutions[1] is not None
                    and pareto_solutions[1] != ""
                ):
                    select_model = pareto_solutions[1]

            logger.info("Selected model for budget optimization: %s", select_model)

            allocator = BudgetAllocator(
                mmm_data=self.mmm_data,
                featurized_mmm_data=self.featurized_mmm_data,
                hyperparameters=self.hyperparameters,
                pareto_result=self.pareto_result,
                select_model=select_model,
                params=allocator_params,
            )

            allocation_result = allocator.optimize()

            if display_plots or export_plots:
                allocator_visualizer = AllocatorPlotter(
                    allocation_result=allocation_result, budget_allocator=allocator
                )
                allocator_visualizer.plot_all(display_plots, self.working_dir)

            logger.info("Budget optimization complete")
            return allocation_result

        except Exception as e:
            logger.error("Budget optimization failed: %s", str(e))
            raise

    def generate_one_pager(self, solution_id: Optional[str] = None) -> None:
        """
        Generate one-page summary report.

        Args:
            plots: Optional list of specific plots to include
            solution_id: Optional specific solution ID to plot
        """
        try:
            onepager = OnePager(
                pareto_result=self.pareto_result,
                clustered_result=self.cluster_result,
                hyperparameter=self.hyperparameters,
                mmm_data=self.mmm_data,
                holidays_data=self.holidays_data,
            )

            # Set top_pareto based on whether solution_id is provided
            top_pareto = solution_id is None

            figures = onepager.generate_one_pager(
                solution_ids=solution_id if solution_id else "all",
                top_pareto=top_pareto,
            )
            return figures

        except Exception as e:
            logging.error("One-pager generation failed: %s", str(e))
            raise

    def _setup_logging(self, console_level: str, file_level: str) -> None:
        """Configure logging with console and file handlers."""

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, console_level))
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        # File handler
        #        log_file = self.working_dir / "logs/robynpy_%(asctime)s.log"
        log_file = self.working_dir / "logs/robynpy.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(getattr(logging, file_level))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(console)
        logger.addHandler(file_handler)

    def _validate_initialization(self) -> None:
        """Validate that required components are initialized."""
        if not all(
            [
                self.mmm_data,
                self.holidays_data,
                self.hyperparameters,
                self.featurized_mmm_data,
            ]
        ):
            raise ValueError("Must call initialize() first")
