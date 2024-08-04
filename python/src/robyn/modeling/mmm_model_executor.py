#pyre-strict

from typing import Optional, Dict, Any

class MMMModelExecutor:
    def model_run(
        self,
        mmmdata_collection: MMMDataCollection = None,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        json_file: Optional[str] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: int = 123,
        quiet: bool = False,
        cores: Optional[int] = None,
        trials: int = 5,
        iterations: int = 2000,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: str = "TwoPointsDE",
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        lambda_control: Optional[float] = None,
        outputs: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Run the Robyn model with the specified parameters.

        Args:
            InputCollect: Input data collection.
            dt_hyper_fixed: Fixed hyperparameters.
            json_file: JSON file path.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            refresh: Refresh the model.
            seed: Random seed.
            quiet: Suppress output.
            cores: Number of cores to use.
            trials: Number of trials.
            iterations: Number of iterations.
            rssd_zero_penalty: Enable RSSD zero penalty.
            objective_weights: Objective weights.
            nevergrad_algo: Nevergrad algorithm to use.
            intercept: Include intercept term.
            intercept_sign: Sign of the intercept term.
            lambda_control: Lambda control value.
            outputs: Output results.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def print_robyn_models(self, x: Any) -> None:
        """
        Print the Robyn models.

        Args:
            x: Model data.
        """

    def model_train(
        self,
        InputCollect: Dict[str, Any],
        hyper_collect: Dict[str, Any],
        cores: int,
        iterations: int,
        trials: int,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: str,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        rssd_zero_penalty: bool = True,
        refresh: bool = False,
        seed: int = 123,
        quiet: bool = False
    ) -> None:
        """
        Train the Robyn model.

        Args:
            InputCollect: Input data collection.
            hyper_collect: Hyperparameter collection.
            cores: Number of cores to use.
            iterations: Number of iterations.
            trials: Number of trials.
            intercept_sign: Sign of the intercept term.
            intercept: Include intercept term.
            nevergrad_algo: Nevergrad algorithm to use.
            dt_hyper_fixed: Fixed hyperparameters.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            objective_weights: Objective weights.
            rssd_zero_penalty: Enable RSSD zero penalty.
            refresh: Refresh the model.
            seed: Random seed.
            quiet: Suppress output.
        """

    #model.R robyn_mmm
    def run_nevergrad_optimization(
        self,
        InputCollect: Dict[str, Any],
        hyper_collect: Dict[str, Any],
        iterations: int,
        cores: int,
        nevergrad_algo: str,
        intercept_sign: str,
        intercept: bool = True,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        rssd_zero_penalty: bool = True,
        refresh: bool = False,
        trial: int = 1,
        seed: int = 123,
        quiet: bool = False
    ) -> None:
        """
        Run the nevergrad optimization.

        Args:
            InputCollect: Input data collection.
            hyper_collect: Hyperparameter collection.
            iterations: Number of iterations.
            cores: Number of cores to use.
            nevergrad_algo: Nevergrad algorithm to use.
            intercept_sign: Sign of the intercept term.
            intercept: Include intercept term.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            objective_weights: Objective weights.
            dt_hyper_fixed: Fixed hyperparameters.
            rssd_zero_penalty: Enable RSSD zero penalty.
            refresh: Refresh the model.
            trial: Trial number.
            seed: Random seed.
            quiet: Suppress output.
        """
        
        def model_fit_iteration(iteration: int, *args: Any, **kwargs: Any) -> None:
            """
            Fit the model.

            Args:
                iteration: Iteration number.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.
            """

    def model_decomp(
        self,
        coefs: Any,
        y_pred: Any,
        dt_modSaturated: Any,
        dt_saturatedImmediate: Any,
        dt_saturatedCarryover: Any,
        dt_modRollWind: Any,
        refreshAddedStart: Any
    ) -> None:
        """
        Decompose the model.

        Args:
            coefs: Model coefficients.
            y_pred: Predicted values.
            dt_modSaturated: Saturated model data.
            dt_saturatedImmediate: Saturated immediate data.
            dt_saturatedCarryover: Saturated carryover data.
            dt_modRollWind: Rolling window data.
            refreshAddedStart: Refresh added start data.
        """

    def model_refit(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any,
        y_val: Any,
        x_test: Any,
        y_test: Any,
        lambda_: float,
        lower_limits: Any,
        upper_limits: Any,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        penalty_factor: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Refit the model.

        Args:
            x_train: Training data.
            y_train: Training targets.
            x_val: Validation data.
            y_val: Validation targets.
            x_test: Testing data.
            y_test: Testing targets.
            lambda_: Lambda value.
            lower_limits: Lower limits.
            upper_limits: Upper limits.
            intercept: Include intercept term.
            intercept_sign: Sign of the intercept term.
            penalty_factor: Penalty factor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def __get_rsq(
        self,
        true: Any,
        predicted: Any,
        p: int,
        df_int: int,
        n_train: Optional[int] = None
    ) -> float:
        """
        Calculate the R-squared value.

        Args:
            true: True values.
            predicted: Predicted values.
            p: Number of parameters.
            df_int: Degrees of freedom.
            n_train: Number of training samples.

        Returns:
            R-squared value.
        """

    def __lambda_seq(
        self,
        x: Any,
        y: Any,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001
    ) -> Any:
        """
        Generate a sequence of lambda values.

        Args:
            x: Input data.
            y: Target data.
            seq_len: Sequence length.
            lambda_min_ratio: Minimum lambda ratio.

        Returns:
            Sequence of lambda values.
        """

    def __hyper_collector(
        self,
        InputCollect: Dict[str, Any],
        hyper_in: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        cores: int,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Collect hyperparameters.

        Args:
            InputCollect: Input data collection.
            hyper_in: Hyperparameter input.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            cores: Number of cores to use.
            dt_hyper_fixed: Fixed hyperparameters.
        """

    def __init_msgs_run(
        self,
        InputCollect: Dict[str, Any],
        refresh: bool,
        quiet: bool = False,
        lambda_control: Optional[float] = None
    ) -> None:
        """
        Initialize the model run.

        Args:
            InputCollect: Input data collection.
            refresh: Refresh the model.
            quiet: Suppress output.
            lambda_control: Lambda control value.
        """
