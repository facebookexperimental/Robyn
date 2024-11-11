from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.allocator.media_response import MediaResponseParamsCalculator
from robyn.allocator.entities.allocation_config import AllocationConfig, DateRange
from robyn.allocator.entities.allocation_results import AllocationResult
from robyn.allocator.entities.enums import OptimizationScenario, ConstrMode
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.allocator.allocation_optimizer import AllocationOptimizer
from robyn.allocator.response_calculator import ResponseCalculator

logger = logging.getLogger(__name__)

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
        logger.info("Initializing BudgetAllocator")
        logger.debug("Input parameters: mmm_data=%s, model_outputs=%s, pareto_result=%s, select_model=%s",
                    mmm_data, model_outputs, pareto_result, select_model)

        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.model_outputs = model_outputs
        self.pareto_result = pareto_result
        self.select_model = select_model

        try:
            # Validate date data
            logger.debug("Validating date data")
            self._validate_date_data()

            # Initialize calculators
            logger.debug("Initializing media parameters calculator")
            self.media_params_calculator = MediaResponseParamsCalculator(
                mmm_data=mmm_data, pareto_result=pareto_result, select_model=select_model
            )
            self.response_calculator = ResponseCalculator()
            self.optimizer = AllocationOptimizer()

            # Calculate media response parameters
            logger.debug("Calculating media response parameters")
            self.media_params = self.media_params_calculator.calculate_parameters()
            logger.info("BudgetAllocator initialization completed successfully")

        except Exception as e:
            logger.error("Failed to initialize BudgetAllocator: %s", str(e))
            raise

    def _process_date_range(self, date_range: Union[str, List[str], datetime]) -> DateRange:
        """Process and validate date range for allocation calculations."""
        logger.debug("Processing date range: %s", date_range)
        
        try:
            date_col = self._get_date_column_name()
            raw_dates = self.mmm_data.data[date_col]
            
            if not pd.api.types.is_datetime64_any_dtype(raw_dates):
                logger.debug("Converting dates to datetime format")
                dates = pd.to_datetime(raw_dates, format=None)
            else:
                dates = raw_dates

            if isinstance(date_range, str):
                logger.debug("Processing string date range: %s", date_range)
                if date_range == "all":
                    start_date = dates.min()
                    end_date = dates.max()
                elif date_range == "last":
                    end_date = dates.max()
                    start_date = end_date
                elif date_range.startswith("last_"):
                    n = int(date_range.split("_")[1])
                    if n > len(dates):
                        logger.error("Requested last_%d dates but only %d dates available", n, len(dates))
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

            if start_date < dates.min() or end_date > dates.max():
                logger.error("Date range %s to %s outside available data range %s to %s",
                           start_date, end_date, dates.min(), dates.max())
                raise ValueError(
                    f"Date range {start_date} to {end_date} outside available data range "
                    f"{dates.min()} to {dates.max()}"
                )

            start_idx = dates[dates >= start_date].index[0]
            end_idx = dates[dates <= end_date].index[-1]
            n_periods = end_idx - start_idx + 1

            if n_periods < 1:
                logger.error("Invalid date range: %s to %s", start_date, end_date)
                raise ValueError(f"Invalid date range: {start_date} to {end_date}")

            date_range_obj = DateRange(
                start_date=start_date,
                end_date=end_date,
                start_index=start_idx,
                end_index=end_idx,
                n_periods=n_periods,
                interval_type=self.mmm_data.mmmdata_spec.interval_type,
            )
            
            logger.debug("Successfully processed date range: %s", date_range_obj)
            return date_range_obj

        except Exception as e:
            logger.error("Error processing date range %s: %s", date_range, str(e))
            raise ValueError(f"Error processing date range {date_range}: {str(e)}")

    def _calculate_initial_metrics(
        self, date_range: DateRange, media_spend_sorted: np.ndarray, total_budget: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate initial metrics for optimization."""
        logger.debug("Calculating initial metrics with date_range=%s, total_budget=%s", date_range, total_budget)

        try:
            dt_mod = self.mmm_data.data
            hist_spend = dt_mod.loc[date_range.start_index : date_range.end_index, media_spend_sorted]
            
            logger.debug("Historical spend statistics: total=%s, mean=%s", 
                        hist_spend.sum(), hist_spend.mean())

            hist_spend_total = hist_spend.sum()
            hist_spend_mean = hist_spend.mean()

            zero_mask = hist_spend_mean == 0
            if zero_mask.any():
                logger.warning("Found zero mean spend for channels: %s", 
                             media_spend_sorted[zero_mask])
                min_nonzero = hist_spend_mean[hist_spend_mean > 0].min()
                hist_spend_mean[zero_mask] = min_nonzero * 0.1

            hist_spend_share = hist_spend_mean / hist_spend_mean.sum()

            init_responses = {}
            response_margins = {}

            logger.debug("Calculating initial responses and margins for each channel")
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

            init_spend_total = hist_spend_mean.sum()
            init_response_total = sum(init_responses.values())

            budget_unit = total_budget / date_range.n_periods if total_budget is not None else init_spend_total

            metrics = {
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

            logger.info("Initial metrics calculated successfully")
            logger.debug("Initial metrics summary: total_spend=%.2f, total_response=%.2f, budget_unit=%.2f",
                        init_spend_total, init_response_total, budget_unit)
            
            return metrics

        except Exception as e:
            logger.error("Failed to calculate initial metrics: %s", str(e))
            raise

    def _optimize_max_response(self, initial_metrics: Dict[str, Any], config: AllocationConfig) -> AllocationResult:
        """Optimize for maximum response while respecting budget constraint."""
        logger.info("Starting maximum response optimization")
        logger.debug("Optimization config: %s", config)

        try:
            x0 = initial_metrics["hist_spend_mean"].values
            bounds = config.constraints.get_bounds(initial_metrics["hist_spend_mean"])
            
            logger.debug("Initial guess: %s", x0)
            logger.debug("Optimization bounds: %s", bounds)

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

            def budget_constraint(x):
                return np.sum(x) - initial_metrics["budget_unit"]

            constraints = [
                {"type": "eq" if config.constr_mode == ConstrMode.EQUALITY else "ineq", "fun": budget_constraint}
            ]

            logger.debug("Running optimization with method=%s, maxeval=%d", config.optim_algo, config.maxeval)
            result = self.optimizer.optimize(
                objective_func=objective,
                bounds=bounds,
                constraints=constraints,
                initial_guess=x0,
                method=config.optim_algo,
                maxeval=config.maxeval,
            )

            if not result["success"]:
                logger.warning("Optimization warning: %s", result["message"])

            optimal_spend = result["x"]
            optimal_responses = {}

            logger.debug("Calculating optimal responses for each channel")
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=optimal_spend[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                optimal_responses[channel] = response

            # Create results DataFrame
            optimal_allocations = pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                    "constr_low": [
                        config.constraints.channel_constr_low[ch]
                        for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "constr_up": [
                        config.constraints.channel_constr_up[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                }
            )

            logger.info("Maximum response optimization completed successfully")
            logger.debug("Optimization results: iterations=%d, success=%s", 
                        result["nit"], result["success"])

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
            logger.error("Max response optimization failed: %s", str(e))
            raise ValueError(f"Max response optimization failed: {str(e)}")


    def _optimize_target_efficiency(
        self, initial_metrics: Dict[str, Any], config: AllocationConfig
    ) -> AllocationResult:
        """Optimize for target efficiency (ROAS/CPA)."""
        logger.info("Starting target efficiency optimization")
        logger.debug("Initial metrics: %s", initial_metrics)
        logger.debug("Optimization config: %s", config)

        try:
            if config.target_value is None:
                if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                    config.target_value = (
                        initial_metrics["init_response_total"] / initial_metrics["hist_spend_mean"].sum()
                    ) * 0.8
                    logger.debug("Set default ROAS target: %.2f", config.target_value)
                else:
                    config.target_value = (
                        initial_metrics["hist_spend_mean"].sum() / initial_metrics["init_response_total"]
                    ) * 1.2
                    logger.debug("Set default CPA target: %.2f", config.target_value)

            x0 = initial_metrics["hist_spend_mean"].values
            bounds = config.constraints.get_bounds(initial_metrics["hist_spend_mean"])
            
            logger.debug("Initial guess: %s", x0)
            logger.debug("Optimization bounds: %s", bounds)

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
                else:
                    return total_spend / total_response - config.target_value

            constraints = [
                {"type": "eq" if config.constr_mode == ConstrMode.EQUALITY else "ineq", "fun": efficiency_constraint}
            ]

            logger.debug("Running optimization with method=%s, maxeval=%d", config.optim_algo, config.maxeval)
            result = self.optimizer.optimize(
                objective_func=objective,
                bounds=bounds,
                constraints=constraints,
                initial_guess=x0,
                method=config.optim_algo,
                maxeval=config.maxeval,
            )

            if not result["success"]:
                logger.warning("Optimization warning: %s", result["message"])

            optimal_spend = result["x"]
            optimal_responses = {}

            logger.debug("Calculating optimal responses for each channel")
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                response = self.response_calculator.calculate_response(
                    spend=optimal_spend[i],
                    coef=self.media_params.coefficients[channel],
                    alpha=self.media_params.alphas[channel],
                    inflexion=self.media_params.inflexions[channel],
                )
                optimal_responses[channel] = response

            total_spend = sum(optimal_spend)
            total_response = sum(optimal_responses.values())

            achieved_efficiency = (
                total_response / total_spend
                if self.mmm_data.mmmdata_spec.dep_var_type == "revenue"
                else total_spend / total_response
            )

            logger.debug("Achieved efficiency: %.2f (target: %.2f)", achieved_efficiency, config.target_value)

            optimal_allocations = pd.DataFrame(
                {
                    "channel": self.mmm_data.mmmdata_spec.paid_media_spends,
                    "current_spend": initial_metrics["hist_spend_mean"],
                    "optimal_spend": optimal_spend,
                    "current_response": [
                        initial_metrics["init_responses"][ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "optimal_response": [optimal_responses[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends],
                    "constr_low": [
                        config.constraints.channel_constr_low[ch]
                        for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                    "constr_up": [
                        config.constraints.channel_constr_up[ch] for ch in self.mmm_data.mmmdata_spec.paid_media_spends
                    ],
                }
            )

            logger.info("Target efficiency optimization completed successfully")
            logger.debug("Optimization results: iterations=%d, success=%s", 
                        result["nit"], result["success"])

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
            logger.error("Target efficiency optimization failed: %s", str(e))
            raise ValueError(f"Target efficiency optimization failed: {str(e)}")

    def _generate_response_curves(
        self, optimal_spend: np.ndarray, current_spend: pd.Series, n_points: int = 100
    ) -> pd.DataFrame:
        """Generate response curves for visualization."""
        logger.debug("Generating response curves with n_points=%d", n_points)
        logger.debug("Input parameters: optimal_spend=%s, current_spend=%s", optimal_spend, current_spend)

        try:
            curves_data = []
            optimal_spend_np = np.asarray(optimal_spend)
            current_spend_np = current_spend

            logger.debug("Generating curves for each channel")
            for i, channel in enumerate(self.mmm_data.mmmdata_spec.paid_media_spends):
                max_spend = max(optimal_spend_np[i], current_spend_np[i]) * 1.5
                spend_range = np.linspace(1e-10, max_spend, n_points)

                logger.debug("Calculating response values for channel %s", channel)
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

                marginal_response = np.gradient(response_values, spend_range)

                mask = spend_range > 1e-10
                roi = np.zeros_like(spend_range)
                roi[mask] = response_values[mask] / spend_range[mask]

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

            curve_df = pd.DataFrame(curves_data)

            # Add metadata
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

            summary_stats = {
                "total_current_spend": current_spend_np.sum(),
                "total_optimal_spend": optimal_spend_np.sum(),
                "total_current_response": sum(metadata["current_response"]),
                "total_optimal_response": sum(metadata["optimal_response"]),
                "response_lift_pct": (sum(metadata["optimal_response"]) / sum(metadata["current_response"]) - 1) * 100,
            }

            logger.debug("Response curves summary stats: %s", summary_stats)
            
            curve_df.attrs["metadata"] = metadata
            curve_df.attrs["summary_stats"] = summary_stats

            logger.info("Response curves generated successfully")
            return curve_df

        except Exception as e:
            logger.error("Failed to generate response curves: %s", str(e))
            raise

    def allocate(self, config: AllocationConfig) -> AllocationResult:
        """Run budget allocation optimization."""
        logger.info("Starting budget allocation optimization")
        logger.debug("Allocation config: %s", config)

        try:
            logger.debug("Processing date range")
            date_range = self._process_date_range(config.date_range)

            logger.debug("Getting sorted media spends")
            media_spend_sorted = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)

            logger.debug("Calculating initial metrics")
            initial_metrics = self._calculate_initial_metrics(
                date_range=date_range,
                media_spend_sorted=media_spend_sorted,
                total_budget=config.total_budget
            )

            logger.info("Running optimization for scenario: %s", config.scenario)
            if config.scenario == OptimizationScenario.MAX_RESPONSE:
                result = self._optimize_max_response(initial_metrics=initial_metrics, config=config)
            else:
                result = self._optimize_target_efficiency(initial_metrics=initial_metrics, config=config)

            if config.plots:
                logger.debug("Generating response curves for visualization")
                result.response_curves = self._generate_response_curves(
                    optimal_spend=result.optimal_allocations["optimal_spend"].values,
                    current_spend=result.optimal_allocations["current_spend"].values,
                )

            logger.info("Budget allocation optimization completed successfully")
            return result

        except Exception as e:
            logger.error("Budget allocation failed: %s", str(e))
            raise

    def _validate_date_data(self) -> None:
            """Validate date data during initialization."""
            logger.debug("Validating date data")
            
            try:
                date_col = self._get_date_column_name()
                if date_col not in self.mmm_data.data.columns:
                    logger.error("Date column '%s' not found in data", date_col)
                    raise ValueError(f"Date column '{date_col}' not found in data")

                dates = pd.to_datetime(self.mmm_data.data[date_col], format=None)

                if not dates.is_monotonic_increasing:
                    logger.error("Dates are not in ascending order")
                    raise ValueError("Dates must be in ascending order")

                if dates.isna().any():
                    logger.error("Date column contains missing values")
                    raise ValueError("Date column contains missing values")

                logger.debug("Date validation completed successfully")

            except Exception as e:
                logger.error("Date validation failed: %s", str(e))
                raise ValueError(f"Invalid date data: {str(e)}")

    def _get_date_column_name(self) -> str:
        """Get the date column name, handling cases where it might be a list."""
        logger.debug("Getting date column name from mmmdata_spec")
        date_var = self.mmm_data.mmmdata_spec.date_var
        if isinstance(date_var, list):
            logger.debug("Date variable is a list, using first element: %s", date_var[0])
            return date_var[0]
        logger.debug("Using date variable: %s", date_var)
        return date_var        