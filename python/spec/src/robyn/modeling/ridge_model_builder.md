# CLASS
## ModelRefitOutput
* This is a data class used to encapsulate the output of a model refitting process.
* It contains various metrics and model predictions for training, validation, and testing datasets.
* Additionally, it stores model coefficients and hyperparameter information.
* Belongs to a package that handles model evaluation and performance metrics.

# CONSTRUCTORS
## ModelRefitOutput `(rsq_train: float, rsq_val: Optional[float], rsq_test: Optional[float], nrmse_train: float, nrmse_val: Optional[float], nrmse_test: Optional[float], coefs: np.ndarray, y_train_pred: np.ndarray, y_val_pred: Optional[np.ndarray], y_test_pred: Optional[np.ndarray], y_pred: np.ndarray, mod: Ridge, df_int: int, lambda_: float, lambda_hp: float, lambda_max: float, lambda_min_ratio: float)`
* `rsq_train`: R-squared value for the training dataset, indicates the proportion of variance explained by the model.
* `rsq_val`: R-squared value for the validation dataset, can be `None` if not applicable.
* `rsq_test`: R-squared value for the testing dataset, can be `None` if not applicable.
* `nrmse_train`: Normalized Root Mean Square Error for training data, measures the model's predictive accuracy.
* `nrmse_val`: Normalized Root Mean Square Error for validation data, can be `None`.
* `nrmse_test`: Normalized Root Mean Square Error for testing data, can be `None`.
* `coefs`: Coefficients of the trained model as a numpy array.
* `y_train_pred`: Predicted values for the training dataset as a numpy array.
* `y_val_pred`: Predicted values for the validation dataset, can be `None`.
* `y_test_pred`: Predicted values for the testing dataset, can be `None`.
* `y_pred`: All predicted values combined into a single numpy array.
* `mod`: The Ridge model instance used for training.
* `df_int`: Integer indicating whether an intercept was used (1 if true, otherwise 0).
* `lambda_`: The lambda value for regularization, impacts model complexity.
* `lambda_hp`: Hyperparameter for lambda, used during optimization.
* `lambda_max`: Maximum lambda value, important for determining the range of regularization.
* `lambda_min_ratio`: Minimum ratio for lambda values, defines the scope of lambda adjustments.

### USAGE
* Utilized to store and access results of a model refit operation, providing comprehensive performance metrics and predictions.

### IMPL
* Implements a data structure using Python's `@dataclass` to automatically generate initialization, representation, and equality methods.

# CLASS
## RidgeModelBuilder
* This class is responsible for building and evaluating Ridge regression models based on provided data and hyperparameters.
* It handles model training, validation, and optimization using various techniques, including Nevergrad for hyperparameter optimization.

# CONSTRUCTORS
## RidgeModelBuilder `(mmm_data: MMMData, holiday_data: HolidaysData, calibration_input: CalibrationInput, hyperparameters: Hyperparameters, featurized_mmm_data: FeaturizedMMMData)`
* `mmm_data`: Instance of `MMMData` containing marketing mix modeling data.
* `holiday_data`: Instance of `HolidaysData` holding holiday-related information affecting the model.
* `calibration_input`: Instance of `CalibrationInput` for calibrating the model.
* `hyperparameters`: Instance of `Hyperparameters` detailing the parameters used for model training.
* `featurized_mmm_data`: Instance of `FeaturizedMMMData` which contains featurized marketing mix modeling data.

### USAGE
* Instantiate this class to prepare and build Ridge regression models using comprehensive marketing data.

### IMPL
* Initializes internal variables with provided data and sets up a logging configuration to track the model building process.

# METHODS
## `build_models(trials_config: TrialsConfig, dt_hyper_fixed: Optional[pd.DataFrame] = None, ts_validation: bool = False, add_penalty_factor: bool = False, seed: int = 123, rssd_zero_penalty: bool = True, objective_weights: Optional[List[float]] = None, nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE, intercept: bool = True, intercept_sign: str = "non_negative", cores: int = 2) -> ModelOutputs`
### USAGE
* `trials_config`: Configuration details for model training trials, includes number of trials and iterations.
* `dt_hyper_fixed`: Fixed hyperparameters for the trials, if any.
* `ts_validation`: Boolean flag to indicate whether time series validation is enabled.
* `add_penalty_factor`: Boolean flag to add a penalty factor during model training.
* `seed`: Random seed for ensuring reproducibility of results.
* `rssd_zero_penalty`: Boolean flag to apply zero penalty for RSSD.
* `objective_weights`: List of weights for components of the objective function.
* `nevergrad_algo`: Choice of optimization algorithm from Nevergrad.
* `intercept`: Boolean flag to include an intercept in the model.
* `intercept_sign`: Constraint for the sign of the intercept.
* `cores`: Number of CPU cores to utilize for parallel processing.

### IMPL
* Starts by initializing hyperparameters and setting up objective weights, including calibration if available.
* Executes multiple trials using the `_run_nevergrad_optimization` method, each trial adapting to the specified parameters.
* Aggregates results from all trials and calculates convergence metrics.
* Constructs a `ModelOutputs` object with aggregated results, including metrics and model predictions, and returns it.

## `_select_best_model(output_models: List[Trial]) -> str`
### USAGE
* `output_models`: List of model trials from which the best model is selected based on performance metrics.

### IMPL
* Extracts NRMSE and decomp RSSD values from each trial.
* Normalizes the extracted metrics for fair comparison.
* Computes a combined score for each model, assuming equal weights for each metric.
* Identifies and returns the ID of the model with the lowest combined score, signifying the best performance.

## `_run_nevergrad_optimization(hyper_collect: Dict[str, Any], iterations: int, cores: int, nevergrad_algo: NevergradAlgorithm, intercept: bool, intercept_sign: str, ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], dt_hyper_fixed: Optional[pd.DataFrame], rssd_zero_penalty: bool, trial: int, seed: int, total_trials: int) -> Trial`
### USAGE
* `hyper_collect`: Dictionary of collected hyperparameters for optimization.
* `iterations`: Number of iterations to run the optimization process.
* `cores`: Number of processing cores to use.
* `nevergrad_algo`: Selected algorithm from Nevergrad for optimization.
* `intercept`: Boolean to determine if an intercept is included.
* `intercept_sign`: Constraint on the sign of the intercept.
* `ts_validation`: Boolean to execute time series validation.
* `add_penalty_factor`: Boolean to incorporate penalty factor in model training.
* `objective_weights`: Weights for different components of the objective function.
* `dt_hyper_fixed`: DataFrame containing fixed hyperparameters.
* `rssd_zero_penalty`: Boolean to apply zero penalty for RSSD.
* `trial`: Current trial number in the sequence.
* `seed`: Random seed for reproducibility.
* `total_trials`: Total number of trials planned.

### IMPL
* Suppresses specific warnings to maintain a clean output during operations.
* Initializes random seed and parameter bounds based on collected hyperparameters.
* Configures the optimizer using Nevergrad, specifying algorithm and budget.
* Iteratively asks for candidate solutions, evaluates them, and updates the optimizer with results.
* Aggregates results from all iterations and identifies the best solution using defined metrics.
* Returns a `Trial` object encapsulating the best results from the optimization process.

## `_calculate_decomp_spend_dist(model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> pd.DataFrame`
### USAGE
* `model`: Trained Ridge regression model instance.
* `X`: DataFrame containing feature data.
* `y`: Series containing target data.
* `params`: Additional parameters needed for calculations.

### IMPL
* Computes the decomposed spend distribution for specified paid media columns in the dataset.
* Calculates and returns a DataFrame with metrics related to spend, model effect, and performance indicators.

## `_prepare_data(params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]`
### USAGE
* `params`: Dictionary of parameters for preparing the data, including any transformations required.

### IMPL
* Renames the dependent variable column and handles date and categorical columns.
* Applies necessary transformations, such as geometric adstock and hill transformation, based on provided parameters.
* Returns a tuple of prepared feature (X) and target (y) data, ready for model training.

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* `x`: Series of input data to be transformed.
* `theta`: Adstock decay parameter to apply to the series.

### IMPL
* Implements geometric adstock transformation on the input series, modifying values based on decay parameter.
* Returns the transformed series reflecting adstock effect.

## `_hill_transformation(x: pd.Series, alpha: float, gamma: float) -> pd.Series`
### USAGE
* `x`: Series of input data to undergo hill transformation.
* `alpha`: Parameter influencing the slope of the transformation.
* `gamma`: Parameter affecting the asymptote of the transformation.

### IMPL
* Normalizes input series and applies the Hill transformation using provided parameters.
* Returns the transformed series adjusted by the Hill function.

## `_evaluate_model(params: Dict[str, float], ts_validation: bool, add_penalty_factor: bool, rssd_zero_penalty: bool, objective_weights: Optional[List[float]], start_time: float, iter_ng: int, trial: int) -> Dict[str, Any]`
### USAGE
* `params`: Dictionary containing model parameters.
* `ts_validation`: Boolean flag for time series validation.
* `add_penalty_factor`: Boolean flag to apply a penalty factor.
* `rssd_zero_penalty`: Boolean flag to apply RSSD zero penalty.
* `objective_weights`: List of weights for evaluation metrics.
* `start_time`: Timestamp indicating when the evaluation started.
* `iter_ng`: Current iteration number for Nevergrad optimization.
* `trial`: Trial number in the sequence.

### IMPL
* Prepares data and performs train-validation split if validation is enabled.
* Trains a Ridge regression model and computes relevant performance metrics.
* Evaluates RSSD, MAPE, and calibration if applicable, and returns a dictionary with results, including loss and metrics.

## `_hyper_collector(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> Dict[str, Any]`
### USAGE
* `hyperparameters_dict`: Dictionary of hyperparameters for collection.
* `ts_validation`: Boolean indicating time series validation.
* `add_penalty_factor`: Boolean for adding a penalty factor.
* `dt_hyper_fixed`: Fixed hyperparameter data if any.
* `cores`: Number of cores for parallel processing.

### IMPL
* Logs information about hyperparameters being collected.
* Collects, organizes, and returns a dictionary of hyperparameters prepared for optimization, handling fixed hyperparameters and conditions.

## `_model_refit(x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, lambda_: float = 1.0, lower_limits: Optional[List[float]] = None, upper_limits: Optional[List[float]] = None, intercept: bool = True, intercept_sign: str = "non_negative") -> ModelRefitOutput`
### USAGE
* `x_train`: Array of training feature data.
* `y_train`: Array of training target data.
* `x_val`: Array of validation feature data, optional.
* `y_val`: Array of validation target data, optional.
* `x_test`: Array of testing feature data, optional.
* `y_test`: Array of testing target data, optional.
* `lambda_`: Regularization parameter for the model.
* `lower_limits`: List of lower limits for coefficients, optional.
* `upper_limits`: List of upper limits for coefficients, optional.
* `intercept`: Boolean for intercept inclusion.
* `intercept_sign`: Constraint for intercept sign.

### IMPL
* Trains a Ridge model using provided feature and target data.
* Predicts outcomes for train, validation, and test datasets.
* Computes R-squared and NRMSE for each dataset.
* Returns a `ModelRefitOutput` object containing all metrics and predictions.

## `_lambda_seq(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> np.ndarray`
### USAGE
* `x`: Array of feature data.
* `y`: Array of target data.
* `seq_len`: Length of the lambda sequence to generate.
* `lambda_min_ratio`: Minimum ratio for lambda values.

### IMPL
* Calculates the maximum lambda value based on input data.
* Generates a logarithmic sequence of lambda values.
* Returns an array of lambda values suitable for regularization parameter selection.