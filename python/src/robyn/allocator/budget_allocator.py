from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.pareto.hill_calculator import HillCalculator

from robyn.allocator.entities.allocation_config import AllocationConfig, DateRange
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.allocator.entities.enums import OptimizationScenario
from robyn.allocator.allocation_optimizer import AllocationOptimizer
from robyn.allocator.response_calculator import ResponseCurveCalculator


class BudgetAllocator:
    """Main class for optimizing marketing budget allocations."""

    def __init__(
        self, mmm_data: MMMData, model_outputs: ModelOutputs, hyperparameter: Hyperparameters, select_model: str
    ):
        """Initialize the BudgetAllocator."""
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.hyperparameter = hyperparameter
        self.select_model = select_model
        self.hill_calculator = HillCalculator(mmm_data, model_outputs, hyperparameter)
        self.response_calculator = ResponseCurveCalculator()
        self.optimizer = AllocationOptimizer()

    def allocate(self, config: AllocationConfig) -> AllocationResult:
        """
        Perform budget allocation optimization.

        Implementation follows logic from robyn_allocator() in allocator.R
        """
        # Validate inputs
        self._validate_inputs(config)

        # Get paid media data
        paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends
        media_order = np.argsort(paid_media_spends)
        media_spend_sorted = np.array(paid_media_spends)[media_order]

        # Process date range
        date_range = self._process_date_range(config.date_range)

        # Get model parameters
        dt_hyppar = self._get_hyperparameters()
        dt_bestCoef = self._get_model_coefficients()

        # Get hill parameters
        hills = self._get_hill_params(dt_hyppar, dt_bestCoef, media_spend_sorted)
        alphas = hills["alphas"]
        inflexions = hills["inflexions"]
        coefs = hills["coefs"]

        # Calculate initial metrics and spend statistics
        initial_metrics = self._calculate_initial_metrics(date_range, media_spend_sorted, config.total_budget)

        # Run optimization based on scenario
        if config.scenario == OptimizationScenario.MAX_RESPONSE:
            optimal_result = self._optimize_max_response(initial_metrics, alphas, inflexions, coefs, config)
        else:
            optimal_result = self._optimize_target_efficiency(initial_metrics, alphas, inflexions, coefs, config)

        # Calculate final metrics and prepare results
        final_metrics = self._calculate_final_metrics(optimal_result, initial_metrics, config)

        # Generate plots if requested
        plots = {}
        if config.plots:
            plots = self._generate_plots(optimal_result, initial_metrics, final_metrics, config)

        # Prepare complete result
        result = AllocationResult(
            optimal_allocations=optimal_result["allocations"],
            predicted_responses=optimal_result["responses"],
            response_curves=optimal_result["curves"],
            metrics=final_metrics,
            plots=plots,
        )

        # Export if requested
        if config.export:
            result.export(self.model_outputs.plot_folder)

        return result
