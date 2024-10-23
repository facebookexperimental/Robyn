# CLASS
## ModelRefitOutput
* This class is a data container for storing the results of refitting a Ridge regression model.
* It uses the `dataclass` decorator for automatic generation of special methods such as `__init__`.
* Stores metrics and predictions for training, validation, and test datasets.
* Contains model coefficients and parameters related to the Ridge regression process.

## RidgeModelBuilder
* This class is responsible for building and training Ridge regression models using marketing mix modeling data.
* It manages data preparation, model training, hyperparameter optimization, and model selection.
* It utilizes several entities such as `MMMData`, `HolidaysData`, `CalibrationInput`, `Hyperparameters`, and `FeaturizedMMMData`.
* Employs specific algorithms and techniques for optimization and model evaluation.

# CONSTRUCTORS
## RidgeModelBuilder `(mmm_data: MMMData, holiday_data: HolidaysData, calibration_input: CalibrationInput, hyperparameters: Hyperparameters, featurized_mmm_data: FeaturizedMMMData)`
* `mmm_data`: An instance of `MMMData` containing the main data for modeling.
* `holiday_data`: An instance of `HolidaysData` for incorporating holiday effects.
* `calibration_input`: An instance of `CalibrationInput` for model calibration.
* `hyperparameters`: An instance of `Hyperparameters` containing the hyperparameters for the model.
* `featurized_mmm_data`: An instance of `FeaturizedMMMData` containing preprocessed data features.
### USAGE
* Use this constructor to initialize the `RidgeModelBuilder` with the necessary data and configurations required for building Ridge regression models.
### IMPL
* Initializes the class attributes with the provided data and configuration objects, setting up the environment for the model-building process.

# METHODS

## `build_models(trials_config: TrialsConfig, dt_hyper_fixed: Optional[pd.DataFrame] = None, ts_validation: bool = False, add_penalty_factor: bool = False, seed: int = 123, rssd_zero_penalty: bool = True, objective_weights: Optional[List[float]] = None, nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE, intercept: bool = True, intercept_sign: str = "non_negative", cores: int = 2) -> ModelOutputs`
### USAGE
* `trials_config`: Configuration for the number of trials and iterations.
* `dt_hyper_fixed`: A DataFrame to fix certain hyperparameters (if any).
* `ts_validation`: Boolean flag for time-series validation.
* `add_penalty_factor`: Boolean flag to add a penalty factor during training.
* `seed`: Random seed for reproducibility.
* `rssd_zero_penalty`: Boolean flag to apply zero-coefficient penalty.
* `objective_weights`: List of weights for different optimization objectives.
* `nevergrad_algo`: Algorithm choice for Nevergrad optimization.
* `intercept`: Boolean flag for including intercept in the model.
* `intercept_sign`: Specifies the sign constraint for the intercept.
* `cores`: Number of CPU cores to use for parallel processing.
### IMPL
* The method begins by recording the start time for performance tracking.
* Calculates the interval type and rolling window length based on the date range in `mmm_data`.
* Displays information about the data intervals and rolling window setup.
* Displays a message if time-series validation is enabled.
* Collects hyperparameters for optimization using `_hyper_collector`.
* Displays the number of hyperparameters and the algorithm being used for optimization.
* Initiates multiple trials using `_model_train` with the specified optimization configurations.
* Records the end time and calculates the total run time.
* Calculates convergence metrics for the trials using a `Convergence` object.
* Constructs a `ModelOutputs` instance containing the results of the modeling process, including convergence information.
* Prints the convergence messages and returns the `ModelOutputs` instance.

## `_select_best_model(output_models: List[Trial]) -> str`
### USAGE
* `output_models`: A list of `Trial` objects representing different model runs.
### IMPL
* Extracts and normalizes relevant metrics (`nrmse` and `decomp_rssd`) from the trials.
* Calculates a combined score to evaluate models based on normalized metrics.
* Identifies and returns the solution ID of the best model with the lowest combined score.

## `_model_train(hyper_collect: Dict[str, Any], trials_config: TrialsConfig, intercept_sign: str, intercept: bool, nevergrad_algo: NevergradAlgorithm, dt_hyper_fixed: Optional[pd.DataFrame], ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], rssd_zero_penalty: bool, seed: int, cores: int) -> List[Trial]`
### USAGE
* `hyper_collect`: Collected hyperparameters for optimization.
* `trials_config`: Configuration for the number of trials and iterations.
* Additional parameters for model training and optimization.
### IMPL
* Iterates over the number of trials specified in `trials_config`.
* For each trial, calls `_run_nevergrad_optimization` to perform optimization.
* Collects and returns the results of each trial in a list of `Trial` objects.

## `_run_nevergrad_optimization(hyper_collect: Dict[str, Any], iterations: int, cores: int, nevergrad_algo: NevergradAlgorithm, intercept: bool, intercept_sign: str, ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], dt_hyper_fixed: Optional[pd.DataFrame], rssd_zero_penalty: bool, trial: int, seed: int, total_trials: int) -> Trial`
### USAGE
* `hyper_collect`: Collected hyperparameters for optimization.
* `iterations`: Number of optimization iterations.
* `cores`: Number of CPU cores to use.
* Additional parameters for optimization settings.
### IMPL
* Sets warning filters to ignore specific warnings during optimization.
* Initializes the Nevergrad optimizer with the specified algorithm and budget.
* Iterates over the specified number of iterations, updating the best model parameters and metrics.
* Tracks the best performing model parameters and metrics.
* Returns a `Trial` object representing the best model from the trial.

## `_prepare_data(params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]`
### USAGE
* `params`: Dictionary containing model parameters.
### IMPL
* Prepares the feature matrix `X` and target vector `y` for model training.
* Converts date columns to numeric and one-hot encodes categorical variables.
* Applies transformations like adstock and hill transformation based on parameters.
* Handles NaN and infinite values, ensuring numerical stability.
* Returns the processed feature matrix and target vector.

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* `x`: A Pandas Series representing media spend data.
* `theta`: The adstock decay parameter.
### IMPL
* Applies geometric adstock transformation to the input series.
* Modifies the series to include the effect of previous values weighted by `theta`.
* Returns the transformed series.

## `_hill_transformation(x: pd.Series, alpha: float, gamma: float) -> pd.Series`
### USAGE
* `x`: A Pandas Series representing media spend data.
* `alpha`: Slope parameter for the Hill transformation.
* `gamma`: Inflection point parameter for the Hill transformation.
### IMPL
* Scales the input series and applies the Hill transformation.
* Returns the transformed series based on the provided parameters.

## `_calculate_rssd(coefs: np.ndarray, rssd_zero_penalty: bool) -> float`
### USAGE
* `coefs`: Numpy array of model coefficients.
* `rssd_zero_penalty`: Boolean flag for applying zero-coefficient penalty.
### IMPL
* Calculates the Root Sum of Squared Differences (RSSD) for the coefficients.
* Applies a penalty for zero coefficients if `rssd_zero_penalty` is True.
* Returns the RSSD value.

## `_calculate_mape(model: Ridge) -> float`
### USAGE
* `model`: A trained Ridge regression model.
### IMPL
* Calculates the Mean Absolute Percentage Error (MAPE) for the model.
* Uses calibration data to estimate lift and compute MAPE.
* Returns the average MAPE across calibration points.

## `_evaluate_model(params: Dict[str, float], ts_validation: bool, add_penalty_factor: bool, rssd_zero_penalty: bool, objective_weights: Optional[List[float]]) -> Tuple[float, float, float, float, Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame, float, float, float, float, float, float, float, int]`
### USAGE
* `params`: Dictionary containing model parameters.
* Additional flags and settings for model evaluation.
### IMPL
* Prepares training and validation data splits.
* Fits a Ridge regression model and evaluates its performance.
* Computes various metrics like NRMSE, RSSD, MAPE, and R-squared.
* Returns a tuple containing the evaluation results and metrics.

## `_hyper_collector(hyperparameters: Hyperparameters, ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> Dict[str, Any]`
### USAGE
* `hyperparameters`: An instance of `Hyperparameters`.
* Additional flags and settings for hyperparameter collection.
### IMPL
* Collects and organizes hyperparameters for optimization.
* Differentiates between fixed and variable hyperparameters.
* Prepares a dictionary of hyperparameter bounds and settings for optimization.
* Returns the collected hyperparameter configuration as a dictionary.

## `_model_refit(x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, lambda_: float = 1.0, lower_limits: Optional[List[float]] = None, upper_limits: Optional[List[float]] = None, intercept: bool = True, intercept_sign: str = "non_negative") -> ModelRefitOutput`
### USAGE
* `x_train`: Training feature matrix.
* `y_train`: Training target vector.
* `x_val`, `y_val`: Optional validation features and targets.
* `x_test`, `y_test`: Optional test features and targets.
* `lambda_`: Regularization parameter for Ridge regression.
* Additional parameters for intercept handling and constraints.
### IMPL
* Refits a Ridge regression model to the training data.
* Predicts target values for training, validation, and test data.
* Computes R-squared and NRMSE metrics for each data split.
* Returns a `ModelRefitOutput` instance containing refit results.

## `_lambda_seq(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> np.ndarray`
### USAGE
* `x`: Feature matrix.
* `y`: Target vector.
* `seq_len`: Length of the lambda sequence.
* `lambda_min_ratio`: Minimum ratio of lambda values.
### IMPL
* Calculates a sequence of lambda values for Ridge regression.
* Uses logspace generation based on maximum lambda and ratio constraints.
* Returns an array of lambda values.