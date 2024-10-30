from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.allocator.media_response import MediaResponseParameters, MediaResponseParamsCalculator

from robyn.allocator.entities.allocation_config import AllocationConfig, DateRange
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.allocator.entities.enums import OptimizationScenario
from robyn.allocator.allocation_optimizer import AllocationOptimizer
from robyn.allocator.response_calculator import ResponseCurveCalculator

from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.allocator.entities.allocation_config import AllocationConfig, DateRange
from robyn.allocator.entities.enums import OptimizationScenario, ConstrMode, UseCase
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.allocator.entities.allocation_constraints import AllocationConstraints


class BudgetAllocator:
    """Main class for optimizing marketing budget allocations."""

    def __init__(
        self,
        mmm_data: MMMData,
        model_outputs: ModelOutputs,
        pareto_result: ParetoResult,
        select_model: str,
    ):
        """Initialize the BudgetAllocator."""
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.pareto_result = pareto_result
        self.select_model = select_model

        # Initialize calculators
        self.media_params_calculator = MediaResponseParamsCalculator(
            mmm_data=mmm_data, pareto_result=pareto_result, select_model=select_model
        )
        self.response_calculator = ResponseCurveCalculator()
        self.optimizer = AllocationOptimizer()

        # Calculate media response parameters
        self.media_params = self.media_params_calculator.calculate_parameters()

    def _process_date_range(self, date_range: Union[str, List[str], datetime]) -> DateRange:
        """Process and validate date range for allocation calculations.

        Args:
            date_range: Can be "all", "last", "last_n", specific date, or date range

        Returns:
            DateRange object with processed dates and indices
        """
        dt_mod = self.mmm_data.mmmdata_spec.dt_mod
        date_col = self.mmm_data.mmmdata_spec.date_var

        dates = pd.to_datetime(dt_mod[date_col])

        if isinstance(date_range, str):
            if date_range == "all":
                start_date = dates.min()
                end_date = dates.max()
            elif date_range == "last":
                end_date = dates.max()
                start_date = end_date
            elif date_range.startswith("last_"):
                n = int(date_range.split("_")[1])
                end_date = dates.max()
                start_date = dates.iloc[-n]
            else:
                start_date = end_date = pd.to_datetime(date_range)
        else:
            if isinstance(date_range, list):
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
            else:
                start_date = end_date = pd.to_datetime(date_range)

        # Get indices
        start_idx = dates[dates >= start_date].index[0]
        end_idx = dates[dates <= end_date].index[-1]
        n_periods = end_idx - start_idx + 1

        return DateRange(
            start_date=start_date,
            end_date=end_date,
            start_index=start_idx,
            end_index=end_idx,
            n_periods=n_periods,
            interval_type=self.mmm_data.mmmdata_spec.intervalType,
        )

    def _calculate_initial_metrics(
        self, date_range: DateRange, media_spend_sorted: np.ndarray, total_budget: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate initial metrics for optimization.

        Args:
            date_range: Processed date range information
            media_spend_sorted: Sorted array of media spend column names
            total_budget: Optional total budget constraint

        Returns:
            Dictionary containing initial metrics including spends and responses
        """
        dt_mod = self.mmm_data.mmmdata_spec.dt_mod

        # Get historical spend data
        hist_spend = dt_mod.loc[date_range.start_index : date_range.end_index, media_spend_sorted]

        # Calculate spend statistics
        hist_spend_total = hist_spend.sum()
        hist_spend_mean = hist_spend.mean()
        hist_spend_share = hist_spend_mean / hist_spend_mean.sum()

        # Calculate initial responses using media parameters
        init_responses = {}
        response_margins = {}

        for channel in media_spend_sorted:
            spend = hist_spend_mean[channel]
            params = self.media_params

            response = self.response_calculator.calculate_response(
                spend=spend,
                coef=params.coefficients[channel],
                alpha=params.alphas[channel],
                inflexion=params.inflexions[channel],
            )

            margin = self.response_calculator.calculate_gradient(
                spend=spend,
                coef=params.coefficients[channel],
                alpha=params.alphas[channel],
                inflexion=params.inflexions[channel],
            )

            init_responses[channel] = response
            response_margins[channel] = margin

        # Calculate totals
        init_spend_total = hist_spend_mean.sum()
        init_response_total = sum(init_responses.values())

        # Handle total budget
        if total_budget is not None:
            budget_unit = total_budget / date_range.n_periods
        else:
            budget_unit = init_spend_total

        return {
            "hist_spend_total": hist_spend_total,
            "hist_spend_mean": hist_spend_mean,
            "hist_spend_share": hist_spend_share,
            "init_responses": init_responses,
            "response_margins": response_margins,
            "init_spend_total": init_spend_total,
            "init_response_total": init_response_total,
            "budget_unit": budget_unit,
            "date_range": date_range,
        }

    def _optimize_max_response(self, initial_metrics: Dict[str, Any], config: AllocationConfig) -> AllocationResult:
        """Optimize for maximum response while respecting budget constraint.

        Args:
            initial_metrics: Dictionary of initial metrics
            config: Allocation configuration

        Returns:
            AllocationResult containing optimization results
        """
        # Set up optimization problem
        x0 = initial_metrics["hist_spend_mean"].values

        # Get bounds from constraints
        bounds = config.constraints.get_bounds(initial_metrics["hist_spend_mean"])

        # Define objective function for maximizing response
        def objective(x):
            total_response = 0
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=x[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                total_response += response
            return -total_response  # Negative because we're minimizing

        # Define budget constraint
        def budget_constraint(x):
            return np.sum(x) - initial_metrics["budget_unit"]

        constraints = [
            {"type": "eq" if config.constr_mode == ConstrMode.EQUALITY else "ineq", "fun": budget_constraint}
        ]

        # Run optimization
        result = self.optimizer.optimize(
            objective_func=objective,
            bounds=bounds,
            constraints=constraints,
            initial_guess=x0,
            method=config.optim_algo,
            maxeval=config.maxeval,
        )

        # Calculate optimized responses
        optimal_spend = result.x
        optimal_responses = {}
        for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
            response = self.response_calculator.calculate_response(
                spend=optimal_spend[i],
                coef=self.media_params.coefficients[channel],
                alpha=self.media_params.alphas[channel],
                inflexion=self.media_params.inflexions[channel],
            )
            optimal_responses[channel] = response

        # Prepare result
        return AllocationResult(
            optimal_allocations=pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                }
            ),
            predicted_responses=pd.DataFrame(
                {"channel": self.mmm_data.mmmdata_spec.paid_media_spends, "response": list(optimal_responses.values())}
            ),
            response_curves=self._generate_response_curves(
                optimal_spend=optimal_spend, current_spend=initial_metrics["hist_spend_mean"]
            ),
            metrics={
                "total_budget": initial_metrics["budget_unit"] * initial_metrics["date_range"].n_periods,
                "response_lift": (sum(optimal_responses.values()) / initial_metrics["init_response_total"]) - 1,
                "optimization_iterations": result.nit,
                "optimization_status": result.success,
            },
        )

    def _optimize_target_efficiency(
        self, initial_metrics: Dict[str, Any], config: AllocationConfig
    ) -> AllocationResult:
        """Optimize for target efficiency (ROAS/CPA).

        Args:
            initial_metrics: Dictionary of initial metrics
            config: Allocation configuration

        Returns:
            AllocationResult containing optimization results
        """
        # Set default target value if not provided
        if config.target_value is None:
            if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                config.target_value = (
                    initial_metrics["init_response_total"] / initial_metrics["init_spend_total"]
                ) * 0.8
            else:  # conversion
                config.target_value = (
                    initial_metrics["init_spend_total"] / initial_metrics["init_response_total"]
                ) * 1.2

        # Setup optimization similar to max_response but with efficiency constraint
        x0 = initial_metrics["hist_spend_mean"].values
        bounds = config.constraints.get_bounds(initial_metrics["hist_spend_mean"])

        def objective(x):
            total_response = 0
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=x[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                total_response += response
            return -total_response

        def efficiency_constraint(x):
            total_response = -objective(x)
            total_spend = np.sum(x)

            if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                return total_response / total_spend - config.target_value
            else:  # conversion
                return total_spend / total_response - config.target_value

        constraints = [
            {"type": "eq" if config.constr_mode == ConstrMode.EQUALITY else "ineq", "fun": efficiency_constraint}
        ]

        # Run optimization
        result = self.optimizer.optimize(
            objective_func=objective,
            bounds=bounds,
            constraints=constraints,
            initial_guess=x0,
            method=config.optim_algo,
            maxeval=config.maxeval,
        )

        # Calculate results similarly to max_response
        optimal_spend = result.x
        optimal_responses = {}
        for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
            response = self.response_calculator.calculate_response(
                spend=optimal_spend[i],
                coef=self.media_params.coefficients[channel],
                alpha=self.media_params.alphas[channel],
                inflexion=self.media_params.inflexions[channel],
            )
            optimal_responses[channel] = response

        return AllocationResult(
            optimal_allocations=pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                }
            ),
            predicted_responses=pd.DataFrame(
                {"channel": self.mmm_data.mmmdata_spec.paid_media_spends, "response": list(optimal_responses.values())}
            ),
            response_curves=self._generate_response_curves(
                optimal_spend=optimal_spend, current_spend=initial_metrics["hist_spend_mean"]
            ),
            metrics={
                "total_spend": sum(optimal_spend) * initial_metrics["date_range"].n_periods,
                "response_lift": (sum(optimal_responses.values()) / initial_metrics["init_response_total"]) - 1,
                "achieved_efficiency": (
                    sum(optimal_responses.values()) / sum(optimal_spend)
                    if self.mmm_data.mmmdata_spec.dep_var_type == "revenue"
                    else sum(optimal_spend) / sum(optimal_responses.values())
                ),
                "target_efficiency": config.target_value,
                "optimization_iterations": result.nit,
                "optimization_status": result.success,
            },
        )
