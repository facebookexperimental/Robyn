# pyre-strict

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from dataclasses import asdict


from robyn.data.entities.enums import AdstockType
from .optimization.optimizer import Optimizer
from .optimization.objective_function import ObjectiveFunction
from .entities.optimization_result import OptimizationResult, AllocationPlots
from .entities.optimization_spec import OptimizationSpec
from .utils.validation_utils import ValidationUtils
from .utils.transformation_utils import TransformationUtils
from .utils.math_utils import MathUtils
from ..data.entities.mmmdata import MMMData
from ..modeling.entities.modeloutputs import ModelOutputs
from ..data.entities.hyperparameters import Hyperparameters
from .allocation_plotter import AllocationPlotter

logger = logging.getLogger(__name__)


class BudgetAllocator:
    """Main class for budget allocation optimization."""

    def __init__(
        self,
        input_collect: MMMData,
        output_collect: ModelOutputs,
        select_model: str,
        hyperparameters: Hyperparameters,  # Add this parameter
    ) -> None:
        """Initialize budget allocator.

        Args:
            input_collect: MMM input data and parameters
            output_collect: MMM model outputs
            select_model: Selected model ID
            hyperparameters: Model hyperparameters
        """
        self.input_collect = input_collect
        self.output_collect = output_collect
        self.select_model = select_model
        self.hyperparameters = hyperparameters  # Store hyperparameters
        self.paid_media_vars = input_collect.mmmdata_spec.paid_media_spends

        # Validate inputs
        self._validate_inputs()
        logger.info(f"Initialized BudgetAllocator with model {select_model}")

    def _validate_inputs(self) -> None:
        """Validates input data and model selection."""
        if not self.paid_media_vars:
            raise ValueError("No paid media variables specified")

        if self.select_model not in self.output_collect.all_result_hyp_param.sol_id.unique():
            raise ValueError(f"Model {self.select_model} not found in results")

        ValidationUtils.validate_spend_data(self.input_collect.data, self.paid_media_vars)

    def _get_model_params(self) -> Dict:
        """Extracts model parameters for selected model."""
        print("Extracting model parameters...")
        model_params = (
            self.output_collect.all_result_hyp_param[
                self.output_collect.all_result_hyp_param.sol_id == self.select_model
            ]
            .iloc[0]
            .to_dict()
        )

        # Add debug information
        print("\nModel parameters:")
        print(f"Total parameters: {len(model_params)}")
        print("Parameter names:", list(model_params.keys()))
        print("\nPaid media variables:", self.paid_media_vars)

        # Detect adstock type from parameters
        sample_channel = self.paid_media_vars[0]
        if f"{sample_channel}_thetas" in model_params:
            actual_adstock_type = "geometric"
        elif f"{sample_channel}_S_shapes" in model_params:
            actual_adstock_type = "weibull"
        else:
            actual_adstock_type = str(self.hyperparameters.adstock).lower()

        print(f"\nDetected adstock type: {actual_adstock_type}")

        # Validate parameters exist for each channel
        for channel in self.paid_media_vars:
            if actual_adstock_type == "geometric":
                param_name = f"{channel}_thetas"
                required_params = [param_name]
            else:  # weibull
                param_names = [f"{channel}_S_shapes", f"{channel}_S_scales"]
                required_params = param_names

            missing_params = [p for p in required_params if p not in model_params]

            if missing_params:
                raise KeyError(
                    f"Missing required parameters for channel {channel}: {missing_params}"
                    f"\nAvailable parameters: {list(model_params.keys())}"
                )

        # Store detected adstock type for use in other methods
        self._actual_adstock_type = actual_adstock_type
        return model_params

    def _get_response_coefficients(self) -> Dict[str, float]:
        """Gets response coefficients for each channel."""
        coef_data = self.output_collect.all_x_decomp_agg[
            (self.output_collect.all_x_decomp_agg.sol_id == self.select_model)
            & (self.output_collect.all_x_decomp_agg.rn.isin(self.paid_media_vars))
        ]
        return dict(zip(coef_data.rn, coef_data.coef))

    def _prepare_optimization_inputs(
        self,
        optimization_spec: OptimizationSpec,
    ) -> Dict:
        """Prepares inputs for optimization."""
        print("\nPreparing optimization inputs with spec:")
        print(optimization_spec)

        # Get date range indices
        start_idx, end_idx = TransformationUtils.get_date_range_indices(
            self.input_collect.data[self.input_collect.mmmdata_spec.date_var], optimization_spec.date_range
        )
        print(f"\nDate range indices: {start_idx} to {end_idx}")

        # Get historical spend data
        hist_spend = self.input_collect.data.iloc[start_idx : end_idx + 1][self.paid_media_vars]
        print("\nHistorical spend head:")
        print(hist_spend.head())

        # Calculate initial spend values (mean spend for each channel)
        initial_spend = hist_spend.mean().values
        print(f"\nInitial spend values:\n{initial_spend}")

        # If total_budget is not provided, use historical total spend
        if optimization_spec.total_budget is None:
            total_budget = hist_spend.sum().sum()
            logger.info(f"Using historical total spend as budget: {total_budget}")
        else:
            total_budget = optimization_spec.total_budget

        # Get model parameters
        model_params = self._get_model_params()
        print(f"\nAdstock type: {self.hyperparameters.adstock}")

        # Get response coefficients
        coef_dict = self._get_response_coefficients()
        print("\nResponse coefficients:")
        print(coef_dict)

        # Calculate constraint bounds
        lower_bounds, upper_bounds = self._calculate_constraint_bounds(
            initial_spend=initial_spend,
            constraints_low=optimization_spec.channel_constraints_low,
            constraints_up=optimization_spec.channel_constraints_up,
            multiplier=optimization_spec.channel_constraint_multiplier,
        )
        print("\nConstraint bounds:")
        print(f"Lower: {lower_bounds}")
        print(f"Upper: {upper_bounds}")

        # Calculate historical carryover effects
        hist_carryover = self._calculate_historical_carryover(hist_spend, model_params)

        return {
            "initial_spend": initial_spend,
            "total_budget": total_budget,  # Add this line
            "model_params": model_params,
            "coef_dict": coef_dict,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "hist_carryover": hist_carryover,
            "hist_spend": hist_spend,
        }

    def _calculate_historical_carryover(
        self,
        hist_spend: pd.DataFrame,
        model_params: Dict,
    ) -> Dict[str, np.ndarray]:
        """Calculates historical carryover effects."""
        print("\nCalculating historical carryover...")
        hist_carryover = {}
        adstock_type = getattr(self, "_actual_adstock_type", "geometric")  # Default to geometric if not set
        print(f"Using adstock type: {adstock_type}")

        for channel in self.paid_media_vars:
            print(f"\nProcessing channel: {channel}")
            try:
                if adstock_type == "geometric":
                    theta = model_params[f"{channel}_thetas"]
                    print(f"Using theta value: {theta}")
                    carryover = TransformationUtils.calculate_adstock(
                        hist_spend[channel].values, theta=theta, adstock_type="geometric"
                    )
                else:  # weibull
                    shape = model_params[f"{channel}_S_shapes"]
                    scale = model_params[f"{channel}_S_scales"]

                    carryover = TransformationUtils.calculate_adstock(
                        hist_spend[channel].values,
                        theta=0.0,  # dummy value for non-geometric
                        adstock_type="weibull",
                        shape=shape,
                        scale=scale,
                    )
                hist_carryover[channel] = carryover
                print(f"Successfully calculated carryover for {channel}")

            except KeyError as e:
                print(f"Error: Missing parameter for channel {channel}: {str(e)}")
                print(f"Available parameters: {list(model_params.keys())}")
                raise

        return hist_carryover

    def run_allocation(
        self,
        scenario: str = "max_response",
        total_budget: Optional[float] = None,
        date_range: str = "all",
        channel_constraints_low: Union[float, List[float]] = 0.7,
        channel_constraints_up: Union[float, List[float]] = 1.2,
        channel_constraint_multiplier: float = 3.0,
        max_eval: int = 100000,
        constr_mode: str = "eq",
        target_value: Optional[float] = None,
    ) -> OptimizationResult:
        """Runs budget allocation optimization."""
        logger.info(f"Starting budget allocation with scenario: {scenario}")

        print("\nRunning budget allocation")
        print(f"Scenario: {scenario}")
        print(f"Channel constraints: [{channel_constraints_low}, {channel_constraints_up}]")

        # Input validation
        if len(self.paid_media_vars) < 2:
            raise ValueError("Need at least 2 paid media channels")

        if total_budget is not None and total_budget <= 0:
            raise ValueError("Total budget must be positive")

        # Create optimization specification
        optimization_spec = OptimizationSpec(
            scenario=scenario,
            total_budget=total_budget,
            date_range=date_range,
            channel_constraints_low=channel_constraints_low,
            channel_constraints_up=channel_constraints_up,
            channel_constraint_multiplier=channel_constraint_multiplier,
            max_eval=max_eval,
            constr_mode=constr_mode,
            target_value=target_value,
        )

        print("\nPreparing optimization inputs with spec:")
        print(optimization_spec)

        # Prepare optimization inputs
        optim_inputs = self._prepare_optimization_inputs(optimization_spec)

        # Use historical total spend if no budget specified
        if total_budget is None:
            total_budget = np.sum(optim_inputs["hist_spend"])
            logger.info(f"Using historical total spend as budget: {total_budget}")

        # Create objective function
        self.objective_function = ObjectiveFunction(
            coef_dict=optim_inputs["coef_dict"],
            alphas_dict=optim_inputs["model_params"],
            gammas_dict=optim_inputs["model_params"],
            hist_carryover_dict=optim_inputs["hist_carryover"],
        )

        # Calculate initial response for comparison
        initial_response = self.objective_function.evaluate_total_response(
            optim_inputs["initial_spend"], self.paid_media_vars
        )[0]

        # Create and run optimizer
        optimizer = Optimizer(
            objective_function=self.objective_function,
            optimization_spec=optimization_spec,
            channel_names=self.paid_media_vars,
            initial_spend=optim_inputs["initial_spend"],
            lower_bounds=optim_inputs["lower_bounds"],
            upper_bounds=optim_inputs["upper_bounds"],
        )

        optimal_spend, optimization_result, channel_responses = optimizer.optimize()

        # Validate optimization results
        final_response = self.objective_function.evaluate_total_response(optimal_spend, self.paid_media_vars)[0]
        improvement = (final_response - initial_response) / abs(initial_response)

        if improvement < 0.001:  # Less than 0.1% improvement
            logger.warning(
                f"Very small improvement in optimization: {improvement:.2%}. "
                "Consider adjusting constraints or optimization parameters."
            )

        # Create results DataFrame
        dt_optim_out = pd.DataFrame(
            {
                "channels": self.paid_media_vars,
                "initSpendUnit": optim_inputs["initial_spend"],
                "optmSpendUnit": optimal_spend,
                "optmResponseUnit": channel_responses,
                "optmSpendShareUnit": optimal_spend / optimal_spend.sum(),
                "optmResponseShareUnit": channel_responses / channel_responses.sum(),
                "initSpendShare": optim_inputs["initial_spend"] / optim_inputs["initial_spend"].sum(),
                "initResponseUnit": self.objective_function.evaluate_total_response(
                    optim_inputs["initial_spend"], self.paid_media_vars
                )[1],
            }
        )

        # Calculate total metrics
        dt_optim_out["optmSpendUnitTotal"] = dt_optim_out["optmSpendUnit"].sum()
        dt_optim_out["optmResponseUnitTotal"] = dt_optim_out["optmResponseUnit"].sum()
        dt_optim_out["initResponseUnitTotal"] = dt_optim_out["initResponseUnit"].sum()
        dt_optim_out["optmResponseUnitTotalLift"] = (
            dt_optim_out["optmResponseUnitTotal"] / dt_optim_out["initResponseUnitTotal"] - 1
        )

        # Calculate ROI/CPA metrics
        dep_var_type = self.input_collect.mmmdata_spec.dep_var_type
        if dep_var_type == "revenue":
            dt_optim_out["optmRoiUnit"] = dt_optim_out["optmResponseUnit"] / dt_optim_out["optmSpendUnit"]
            dt_optim_out["initRoiUnit"] = dt_optim_out["initResponseUnit"] / dt_optim_out["initSpendUnit"]
        else:  # conversion/CPA case
            dt_optim_out["optmCpaUnit"] = dt_optim_out["optmSpendUnit"] / dt_optim_out["optmResponseUnit"]
            dt_optim_out["initCpaUnit"] = dt_optim_out["initSpendUnit"] / dt_optim_out["initResponseUnit"]

        # Print optimization results
        print("\nOptimization Results:")
        print(f"Initial total spend: {optim_inputs['initial_spend'].sum():.2f}")
        print(f"Optimal total spend: {optimal_spend.sum():.2f}")
        print(f"Initial total response: {initial_response:.2f}")
        print(f"Final total response: {final_response:.2f}")
        print(f"Improvement: {improvement:.2%}")

        # Calculate final budget value properly handling pandas DataFrame
        if total_budget is None:
            # Sum each row, then sum all rows to get total spend
            row_sums = optim_inputs["hist_spend"].sum(axis=1)  # Sum across columns for each row
            final_budget = float(row_sums.sum())  # Sum all row totals
            print(f"\nCalculated budget from historical spend: {final_budget}")
        else:
            # Handle the case where total_budget might be a pandas Series
            if isinstance(total_budget, pd.Series):
                final_budget = float(total_budget.iloc[0])
            else:
                final_budget = float(total_budget)
            print(f"\nUsing provided budget: {final_budget}")

        # Create result object with properly typed budget value
        result = OptimizationResult(
            dt_optim_out=dt_optim_out,
            main_points={},  # Can be expanded if needed
            nls_mod=optimization_result,
            plots=AllocationPlots(),
            scenario=scenario,
            usecase=self._determine_usecase(date_range),
            total_budget=final_budget,  # Use the properly typed value
            skipped_coef0=[],  # Add if implementing coefficient filtering
            skipped_constr=[],  # Add if implementing constraint filtering
            no_spend=[],  # Add if implementing zero spend detection
        )

        logger.info("Budget allocation completed successfully")
        return result

    def _create_allocation_plots(
        self,
        metrics: Dict,
        optimization_spec: OptimizationSpec,
        hist_spend: pd.DataFrame,
    ) -> AllocationPlots:
        """Creates visualization plots for allocation results."""
        plotter = AllocationPlotter()

        # Create response curves data
        plot_data = self._prepare_response_curves_data(
            self.paid_media_vars, metrics["allocation_df"], self.objective_function
        )

        # Create and save one-pager
        fig = plotter.create_onepager(
            dt_optim_out=metrics["allocation_df"],
            plot_data=plot_data,
            scenario=optimization_spec.scenario,
            date_range=(hist_spend.index[0], hist_spend.index[-1]),
            interval_type="Week",  # Or from your input parameters
        )

        if self.export:
            plotter.save_plot(fig, f"{self.output_dir}/allocation_onepager.png")

        return AllocationPlots(onepager_plot=fig)

    def _prepare_response_curves_data(
        self, channels: List[str], allocation_df: pd.DataFrame, objective_function: ObjectiveFunction
    ) -> Dict:
        """Prepares data for response curves plotting.

        Args:
            channels: List of media channel names
            allocation_df: DataFrame containing allocation results
            objective_function: Configured ObjectiveFunction instance

        Returns:
            Dictionary containing spend and response curves data for each channel
        """
        plot_data = {}

        for channel in channels:
            # Get channel's current and optimal spend
            current_spend = allocation_df[allocation_df["channels"] == channel]["initSpendUnit"].iloc[0]
            optimal_spend = allocation_df[allocation_df["channels"] == channel]["optmSpendUnit"].iloc[0]

            # Create spend range for curve
            max_spend = max(current_spend, optimal_spend) * 1.5
            spend_range = np.linspace(0, max_spend, 100)

            # Calculate responses
            responses = [
                objective_function.calculate_response(x=np.array([spend]), channel_name=channel)
                for spend in spend_range
            ]

            plot_data[channel] = {
                "spend": spend_range,
                "response": responses,
                "current_point": (
                    current_spend,
                    objective_function.calculate_response(x=np.array([current_spend]), channel_name=channel),
                ),
                "optimal_point": (
                    optimal_spend,
                    objective_function.calculate_response(x=np.array([optimal_spend]), channel_name=channel),
                ),
            }

        return plot_data

    def _calculate_allocation_metrics(
        self,
        optimal_spend: np.ndarray,
        initial_spend: np.ndarray,
        channel_responses: np.ndarray,
        hist_spend: pd.DataFrame,
    ) -> Dict:
        """Calculates metrics for optimization results."""
        # Get date range info
        date_range = pd.date_range(
            hist_spend.index.min(), hist_spend.index.max(), freq=self.input_collect.mmmdata_spec.date_frequency
        )
        periods = len(date_range)

        # Calculate historical metrics
        hist_spend_means = hist_spend[self.paid_media_vars].mean()
        hist_spend_total = hist_spend_means.sum()

        # Calculate allocation metrics matching R output
        allocation_df = pd.DataFrame(
            {
                "channels": self.paid_media_vars,
                "solID": [self.select_model] * len(self.paid_media_vars),
                "dep_var_type": [self.input_collect.mmmdata_spec.dep_var_type] * len(self.paid_media_vars),
                "date_min": [hist_spend.index.min()] * len(self.paid_media_vars),
                "date_max": [hist_spend.index.max()] * len(self.paid_media_vars),
                "periods": [f"{periods} {self.input_collect.mmmdata_spec.date_frequency}s"]
                * len(self.paid_media_vars),
                # Historical spend metrics
                "histSpendAll": hist_spend_means * periods,
                "histSpendAllTotal": hist_spend_total * periods,
                "histSpendAllUnit": hist_spend_means,
                "histSpendAllUnitTotal": hist_spend_total,
                "histSpendAllShare": hist_spend_means / hist_spend_total,
                # Current metrics
                "initSpendUnit": initial_spend,
                "initSpendUnitTotal": initial_spend.sum(),
                "initSpendShare": initial_spend / initial_spend.sum(),
                "initSpendTotal": initial_spend.sum() * periods,
                # Response metrics
                "initResponseUnit": channel_responses,
                "initResponseUnitTotal": channel_responses.sum(),
                "initResponseTotal": channel_responses.sum() * periods,
                # Optimized metrics
                "optmSpendUnit": optimal_spend,
                "optmSpendUnitTotal": optimal_spend.sum(),
                "optmSpendShareUnit": optimal_spend / optimal_spend.sum(),
                "optmSpendTotal": optimal_spend.sum() * periods,
                "optmResponseUnit": channel_responses,
                "optmResponseUnitTotal": channel_responses.sum(),
                "optmResponseTotal": channel_responses.sum() * periods,
            }
        )

        # Calculate ROI/CPA metrics
        if self.input_collect.mmmdata_spec.dep_var_type == "revenue":
            allocation_df["optmRoiUnit"] = allocation_df["optmResponseUnit"] / allocation_df["optmSpendUnit"]
            allocation_df["initRoiUnit"] = allocation_df["initResponseUnit"] / allocation_df["initSpendUnit"]
        else:
            allocation_df["optmCpaUnit"] = allocation_df["optmSpendUnit"] / allocation_df["optmResponseUnit"]
            allocation_df["initCpaUnit"] = allocation_df["initSpendUnit"] / allocation_df["initResponseUnit"]

        print("\nAllocation DataFrame:")
        print(allocation_df.head())
        return allocation_df

    def _determine_usecase(self, date_range: str) -> str:
        """Determines optimization use case based on date range.

        Args:
            date_range: Date range specification

        Returns:
            Use case description
        """
        if date_range == "all":
            return "all_historical"
        elif date_range.startswith("last"):
            return f"last_{date_range.split('_')[1]}_periods"
        else:
            return "custom_date_range"

    def _calculate_constraint_bounds(
        self,
        initial_spend: np.ndarray,
        constraints_low: Union[float, List[float]],
        constraints_up: Union[float, List[float]],
        multiplier: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates lower and upper bounds for constraints.

        Args:
            initial_spend: Initial spend values
            constraints_low: Lower constraint percentages
            constraints_up: Upper constraint percentages
            multiplier: Multiplier for constraint ranges

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        logger.debug("Calculating constraint bounds")

        # Convert inputs to numpy arrays
        initial_spend = np.array(initial_spend)

        # Convert single values to arrays if needed
        if isinstance(constraints_low, (int, float)):
            constraints_low = np.full_like(initial_spend, constraints_low, dtype=float)
        else:
            constraints_low = np.array(constraints_low, dtype=float)

        if isinstance(constraints_up, (int, float)):
            constraints_up = np.full_like(initial_spend, constraints_up, dtype=float)
        else:
            constraints_up = np.array(constraints_up, dtype=float)

        # Calculate bounds
        lower_bounds = initial_spend * constraints_low
        upper_bounds = initial_spend * constraints_up

        logger.debug(f"Initial bounds: {lower_bounds} to {upper_bounds}")
        logger.debug(f"Using multiplier: {multiplier}")

        # Apply constraint multiplier
        lower_extension = 1 - (1 - constraints_low) * multiplier
        lower_bounds_ext = np.where(lower_extension < 0, np.zeros_like(initial_spend), initial_spend * lower_extension)

        upper_extension = 1 + (constraints_up - 1) * multiplier
        upper_bounds_ext = np.where(
            upper_extension < 0, initial_spend * constraints_up * multiplier, initial_spend * upper_extension
        )

        return lower_bounds_ext, upper_bounds_ext
