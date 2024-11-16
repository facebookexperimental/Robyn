# pyre-strict

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, List

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.validation.mmmdata_validation import MMMDataValidation
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation

from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.model_executor import ModelExecutor

from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
from robyn.modeling.clustering.cluster_builder import ClusterBuilder
from robyn.modeling.clustering.clustering_config import ClusteringConfig
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData

from robyn.allocator.budget_allocator import BudgetAllocator
from robyn.allocator.entities.allocation_config import AllocationConfig
from robyn.allocator.entities.allocation_results import AllocationResult

from robyn.reporting.onepager_reporting import OnePager
from robyn.visualization.feature_visualization import FeaturePlotter

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

            # if display_plots or export_plots: #TODO
            #     feature_plotter = FeaturePlotter(
            #         self.mmm_data, self.hyperparameters, self.featurized_mmm_data
            #     )
            #     feature_plotter.plot_all(display_plots, self.working_dir)

            return self.featurized_mmm_data

        except Exception as e:
            logging.error("Error in feature engineering: %s", str(e))
            raise

    def train_models(
        self,
        trials_config: Optional[TrialsConfig] = None,
        ts_validation: Optional[bool] = False,
        add_penalty_factor: Optional[bool] = False,
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
                nevergrad_algo=nevergrad_algo,
                model_name=model_name,
                cores=cores,
            )
        except Exception as e:
            logger.error("Training models failed: %s", str(e))
            raise

        # if display_plots or export_plots: #TODO
        #     # call plot_all from model convergence visualizer

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

            # Optional clustering
            if cluster_config:
                cluster_builder = ClusterBuilder(self.pareto_result)
                self.cluster_result = cluster_builder.cluster_models(cluster_config)

            # if display_plots or export_plots: #TODO
            #     # call plot_all from pareteo and cluster visualizer
            logger.info("Model evaluation complete")

        except Exception as e:
            logger.error("Model evaluation failed: %s", str(e))
            raise

    def train_and_evaluate_models(
        self,
        trials_config: Optional[TrialsConfig] = None,
        ts_validation: Optional[bool] = False,
        add_penalty_factor: Optional[bool] = False,
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
            nevergrad_algo,
            model_name,
            cores,
            display_plots,
            export_plots,
        )
        self.evaluate_models(pareto_config, cluster_config, display_plots, export_plots)

    def optimize_budget(
        self,
        allocation_config: AllocationConfig,
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

            select_model = select_model or self.model_outputs.select_id

            allocator = BudgetAllocator(
                mmm_data=self.mmm_data,
                featurized_mmm_data=self.featurized_mmm_data,
                pareto_result=self.pareto_result,
                select_model=select_model,
            )

            allocation_result = allocator.allocate(allocation_config)

            # if display_plots or export_plots: #TODO
            #     # call plot_all from allocator visualizer

            logger.info("Budget optimization complete")
            return allocation_result

        except Exception as e:
            logger.error("Budget optimization failed: %s", str(e))
            raise

    @staticmethod
    def generate_one_pager(
        pareto_result: ParetoResult,
        cluster_result: Optional[ClusteringConfig] = None,
        mmm_data: Optional[MMMData] = None,
        plots: Optional[List[str]] = None,
    ) -> None:
        """
        Generate one-page summary report.

        Args:
            pareto_result: Pareto optimization results
            cluster_result: Optional clustering results
            mmm_data: Optional MMM data for additional context
            plots: Optional list of specific plots to include
        """
        try:
            onepager = OnePager(
                pareto_result=pareto_result,
                clustered_result=cluster_result,
                mmm_data=mmm_data,
            )
            onepager.generate_one_pager(plots=plots)

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
        log_file = self.working_dir / "robyn.log"
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
