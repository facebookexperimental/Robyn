from typing import List, Dict, Optional, Tuple, Any, Union
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
from robyn.allocator.response_calculator import ResponseCalculator

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.allocator.entities.allocation_config import AllocationConfig, DateRange
from robyn.allocator.entities.enums import OptimizationScenario, ConstrMode, AllocatorUseCase
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.allocator.entities.allocation_constraints import AllocationConstraints
from robyn.modeling.feature_engineering import FeaturizedMMMData


class BudgetAllocator:
    """Main class for optimizing marketing budget allocations."""

    def __init__(
        self,
        mmm_data: MMMData,
        featurized_mmm_data: FeaturizedMMMData,
        model_outputs: ModelOutputs,
        pareto_result: ParetoResult,
        select_model: str,
    ):
        """Initialize the BudgetAllocator."""
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.model_outputs = model_outputs
        self.pareto_result = pareto_result
        self.select_model = select_model

        # Validate date data
        self._validate_date_data()

        # Initialize calculators
        self.media_params_calculator = MediaResponseParamsCalculator(
            mmm_data=mmm_data, pareto_result=pareto_result, select_model=select_model
        )
        self.response_calculator = ResponseCalculator()
        self.optimizer = AllocationOptimizer()

        # Calculate media response parameters
        self.media_params = self.media_params_calculator.calculate_parameters()

    def _process_date_range(self, date_range: Union[str, List[str], datetime]) -> DateRange:
        """Process and validate date range for allocation calculations."""
        # Get correct date column name
        date_col = self._get_date_column_name()

        # Get dates from data
        raw_dates = self.mmm_data.data[date_col]
        if not pd.api.types.is_datetime64_any_dtype(raw_dates):
            dates = pd.to_datetime(raw_dates, format=None)
        else:
            dates = raw_dates

        # Process the date range
        try:
            if isinstance(date_range, str):
                if date_range == "all":
                    start_date = dates.min()
                    end_date = dates.max()
                elif date_range == "last":
                    end_date = dates.max()
                    start_date = end_date
                elif date_range.startswith("last_"):
                    n = int(date_range.split("_")[1])
                    if n > len(dates):
                        raise ValueError(f"Requested last_{n} dates but only {len(dates)} dates available")
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

            # Validate dates are within range
            if start_date < dates.min() or end_date > dates.max():
                raise ValueError(
                    f"Date range {start_date} to {end_date} outside available data range "
                    f"{dates.min()} to {dates.max()}"
                )

            # Get indices
            start_idx = dates[dates >= start_date].index[0]
            end_idx = dates[dates <= end_date].index[-1]
            n_periods = end_idx - start_idx + 1

            if n_periods < 1:
                raise ValueError(f"Invalid date range: {start_date} to {end_date}")

            return DateRange(
                start_date=start_date,
                end_date=end_date,
                start_index=start_idx,
                end_index=end_idx,
                n_periods=n_periods,
                interval_type=self.mmm_data.mmmdata_spec.interval_type,
            )

        except Exception as e:
            raise ValueError(f"Error processing date range {date_range}: {str(e)}")

    def _calculate_initial_metrics(
        self, date_range: DateRange, media_spend_sorted: np.ndarray, total_budget: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate initial metrics for optimization.

        Args:
            date_range: Date range information
            media_spend_sorted: Array of media channel names
            total_budget: Optional total budget constraint

        Returns:
            Dictionary containing initial metrics
        """
        # Get data from mmm_data
        dt_mod = self.mmm_data.data

        # Get historical spend data
        hist_spend = dt_mod.loc[date_range.start_index : date_range.end_index, media_spend_sorted]

        # Calculate spend statistics
        hist_spend_total = hist_spend.sum()
        hist_spend_mean = hist_spend.mean()

        # Replace any zero means with small positive value
        zero_mask = hist_spend_mean == 0
        if zero_mask.any():
            min_nonzero = hist_spend_mean[hist_spend_mean > 0].min()
            hist_spend_mean[zero_mask] = min_nonzero * 0.1

        hist_spend_share = hist_spend_mean / hist_spend_mean.sum()

        # Calculate initial responses
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
            "model_id": self.select_model,
            "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
            "interval_type": self.mmm_data.mmmdata_spec.interval_type,
        }

    def _optimize_max_response(self, initial_metrics: Dict[str, Any], config: AllocationConfig) -> AllocationResult:
        """Optimize for maximum response while respecting budget constraint."""
        try:
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

            if not result["success"]:
                print(f"Warning: {result['message']}")

            # Get optimal allocations
            optimal_spend = result["x"]
            optimal_responses = {}

            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=optimal_spend[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                optimal_responses[channel] = response

            # Create allocation results DataFrame with constraints
            optimal_allocations = pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                    # Add constraint bounds
                    "constr_low": [
                        config.constraints.channel_constr_low[ch]
                        for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "constr_up": [
                        config.constraints.channel_constr_up[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                }
            )

            # Prepare result
            return AllocationResult(
                optimal_allocations=optimal_allocations,
                predicted_responses=pd.DataFrame(
                    {
                        "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                        "response": list(optimal_responses.values()),
                    }
                ),
                response_curves=self._generate_response_curves(
                    optimal_spend=optimal_spend, current_spend=initial_metrics["hist_spend_mean"]
                ),
                metrics={
                    "total_budget": initial_metrics["budget_unit"] * initial_metrics["date_range"].n_periods,
                    "response_lift": (sum(optimal_responses.values()) / initial_metrics["init_response_total"]) - 1,
                    "optimization_iterations": result["nit"],
                    "optimization_status": result["success"],
                    "model_id": self.select_model,
                    "scenario": str(config.scenario),
                    "use_case": initial_metrics.get("use_case", ""),
                    "date_range_start": initial_metrics["date_range"].start_date,
                    "date_range_end": initial_metrics["date_range"].end_date,
                    "n_periods": initial_metrics["date_range"].n_periods,
                    "interval_type": initial_metrics["interval_type"],
                    "dep_var_type": initial_metrics["dep_var_type"],
                },
            )

        except Exception as e:
            raise ValueError(f"Max response optimization failed: {str(e)}")

    def _optimize_target_efficiency(
        self, initial_metrics: Dict[str, Any], config: AllocationConfig
    ) -> AllocationResult:
        """Optimize for target efficiency (ROAS/CPA).

        Args:
            initial_metrics: Dictionary containing initial metrics including:
                - hist_spend_mean: Current spend by channel
                - init_responses: Initial response by channel
                - init_response_total: Total initial response
                - init_spend_total: Total initial spend
                - budget_unit: Budget per period
                - date_range: DateRange object
            config: AllocationConfig object containing:
                - target_value: Target ROAS/CPA value
                - constraints: Spend constraints
                - optim_algo: Optimization algorithm
                - maxeval: Maximum evaluations
                - constr_mode: Constraint mode

        Returns:
            AllocationResult containing optimization results
        """
        try:
            # Set default target value if not provided
            if config.target_value is None:
                if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                    # For revenue, target is 80% of initial ROAS
                    config.target_value = (
                        initial_metrics["init_response_total"] / initial_metrics["hist_spend_mean"].sum()
                    ) * 0.8
                else:
                    # For conversion, target is 120% of initial CPA
                    config.target_value = (
                        initial_metrics["hist_spend_mean"].sum() / initial_metrics["init_response_total"]
                    ) * 1.2

            # Setup optimization
            x0 = initial_metrics["hist_spend_mean"].values
            bounds = config.constraints.get_bounds(initial_metrics["hist_spend_mean"])

            def objective(x):
                """Objective function to maximize total response."""
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

            def efficiency_constraint(x):
                """Constraint function to maintain target efficiency."""
                total_response = -objective(x)  # Negative because objective returns negative
                total_spend = np.sum(x)

                if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                    # For revenue, maintain ROAS >= target
                    return total_response / total_spend - config.target_value
                else:
                    # For conversion, maintain CPA <= target
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

            if not result["success"]:
                print(f"Warning: Optimization may not have converged. Message: {result['message']}")

            # Get optimal allocations
            optimal_spend = result["x"]
            optimal_responses = {}

            # Calculate responses for optimal spend
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=optimal_spend[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                optimal_responses[channel] = response

            # Calculate total metrics
            total_spend = sum(optimal_spend)
            total_response = sum(optimal_responses.values())

            # Calculate achieved efficiency
            if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                achieved_efficiency = total_response / total_spend
            else:
                achieved_efficiency = total_spend / total_response

            # Create allocation results DataFrame with constraints
            optimal_allocations = pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                    # Add constraint bounds
                    "constr_low": [
                        config.constraints.channel_constr_low[ch]
                        for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "constr_up": [
                        config.constraints.channel_constr_up[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                }
            )

            return AllocationResult(
                optimal_allocations=optimal_allocations,
                predicted_responses=pd.DataFrame(
                    {
                        "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                        "response": list(optimal_responses.values()),
                    }
                ),
                response_curves=self._generate_response_curves(
                    optimal_spend=optimal_spend, current_spend=initial_metrics["hist_spend_mean"].values
                ),
                metrics={
                    "total_budget": total_spend * initial_metrics["date_range"].n_periods,
                    "response_lift": (total_response / initial_metrics["init_response_total"]) - 1,
                    "achieved_efficiency": achieved_efficiency,
                    "target_efficiency": config.target_value,
                    "optimization_iterations": result["nit"],
                    "optimization_status": result["success"],
                    "model_id": self.select_model,
                    "scenario": str(config.scenario),
                    "use_case": "all_historical_vec + historical_budget",
                    "date_range_start": initial_metrics["date_range"].start_date,
                    "date_range_end": initial_metrics["date_range"].end_date,
                    "n_periods": initial_metrics["date_range"].n_periods,
                    "interval_type": initial_metrics["interval_type"],
                    "dep_var_type": initial_metrics["dep_var_type"],
                },
            )

        except Exception as e:
            raise ValueError(f"Target efficiency optimization failed: {str(e)}")

    def _generate_response_curves(
        self, optimal_spend: np.ndarray, current_spend: pd.Series, n_points: int = 100
    ) -> pd.DataFrame:
        """Generate response curves for visualization.

        Args:
            optimal_spend: Array of optimal spend values
            current_spend: Series of current spend values
            n_points: Number of points to generate for each curve

        Returns:
            DataFrame containing response curves for each channel
        """
        curves_data = []

        # Convert spend arrays to numpy for efficient computation
        optimal_spend_np = np.asarray(optimal_spend)
        current_spend_np = current_spend

        for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
            # Generate spend range
            max_spend = max(optimal_spend_np[i], current_spend_np[i]) * 1.5
            spend_range = np.linspace(1e-10, max_spend, n_points)  # Add small epsilon to avoid zero

            # Calculate response values
            response_values = np.array(
                [
                    self.response_calculator.calculate_response(
                        spend=spend,
                        coef=self.media_params.coefficients[channel],
                        alpha=self.media_params.alphas[channel],
                        inflexion=self.media_params.inflexions[channel],
                    )
                    for spend in spend_range
                ]
            )

            # Calculate marginal response (response per unit spend)
            marginal_response = np.gradient(response_values, spend_range)

            # Calculate ROI (avoid division by zero)
            mask = spend_range > 1e-10
            roi = np.zeros_like(spend_range)
            roi[mask] = response_values[mask] / spend_range[mask]

            # Create curve data points
            for j in range(n_points):
                curves_data.append(
                    {
                        "channel": channel,
                        "spend": spend_range[j],
                        "response": response_values[j],
                        "marginal_response": marginal_response[j],
                        "roi": roi[j],
                        "is_current": np.isclose(spend_range[j], current_spend_np[i], rtol=1e-3),
                        "is_optimal": np.isclose(spend_range[j], optimal_spend_np[i], rtol=1e-3),
                    }
                )

        # Convert to DataFrame
        curve_df = pd.DataFrame(curves_data)

        # Add metadata for visualization
        metadata = {
            "channel": list(self.mmm_data.mmmdata_spec.paid_media_spends),
            "current_spend": current_spend_np,
            "optimal_spend": optimal_spend_np,
            "current_response": [
                self.response_calculator.calculate_response(
                    spend=spend,
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                for spend, channel in zip(current_spend_np, self.mmm_data.mmmdata_spec.paid_media_spends)
            ],
            "optimal_response": [
                self.response_calculator.calculate_response(
                    spend=spend,
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                for spend, channel in zip(optimal_spend_np, self.mmm_data.mmmdata_spec.paid_media_spends)
            ],
        }

        # Add summary statistics
        summary_stats = {
            "total_current_spend": current_spend_np.sum(),
            "total_optimal_spend": optimal_spend_np.sum(),
            "total_current_response": sum(metadata["current_response"]),
            "total_optimal_response": sum(metadata["optimal_response"]),
            "response_lift_pct": (sum(metadata["optimal_response"]) / sum(metadata["current_response"]) - 1) * 100,
        }

        # Add to curve DataFrame as attributes
        curve_df.attrs["metadata"] = metadata
        curve_df.attrs["summary_stats"] = summary_stats

        return curve_df

    def allocate(self, config: AllocationConfig) -> AllocationResult:
        """Run budget allocation optimization.

        Args:
            config: Allocation configuration

        Returns:
            AllocationResult containing optimization results
        """
        try:
            # Get date range
            date_range = self._process_date_range(config.date_range)

            # Get sorted media spends
            media_spend_sorted = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)

            # Calculate initial metrics
            initial_metrics = self._calculate_initial_metrics(
                date_range=date_range, media_spend_sorted=media_spend_sorted, total_budget=config.total_budget
            )

            # Run optimization based on scenario
            if config.scenario == OptimizationScenario.MAX_RESPONSE:
                result = self._optimize_max_response(initial_metrics=initial_metrics, config=config)
            else:
                result = self._optimize_target_efficiency(initial_metrics=initial_metrics, config=config)

            # Generate response curves if plots requested
            if config.plots:
                result.response_curves = self._generate_response_curves(
                    optimal_spend=result.optimal_allocations["optimal_spend"].values,
                    current_spend=result.optimal_allocations["current_spend"].values,
                )

            return result

        except Exception as e:
            print(f"Error during allocation: {str(e)}")
            raise

    def _validate_date_data(self) -> None:
        """Validate date data during initialization."""
        try:
            date_col = self._get_date_column_name()
            if date_col not in self.mmm_data.data.columns:
                raise ValueError(f"Date column '{date_col}' not found in data")

            # Try converting to datetime
            dates = pd.to_datetime(self.mmm_data.data[date_col], format=None)

            # Ensure dates are sorted
            if not dates.is_monotonic_increasing:
                raise ValueError("Dates must be in ascending order")

            # Check for missing dates
            if dates.isna().any():
                raise ValueError("Date column contains missing values")

        except Exception as e:
            raise ValueError(f"Invalid date data: {str(e)}")

    def _get_date_column_name(self) -> str:
        """Get the date column name, handling cases where it might be a list."""
        date_var = self.mmm_data.mmmdata_spec.date_var
        if isinstance(date_var, list):
            return date_var[0]
        return date_var
