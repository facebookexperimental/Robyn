# CLASS
## ModelRefitOutput
* This class is a data container using the `dataclass` decorator.
* It holds the output of a model refitting process, containing statistical metrics and model parameters.
* It provides a structured way to store and access model evaluation results.
* Fields are as follows:
  - `rsq_train`: R-squared on the training data.
  - `rsq_val`: Optional R-squared on the validation data.
  - `rsq_test`: Optional R-squared on the test data.
  - `nrmse_train`: Normalized RMSE on the training data.
  - `nrmse_val`: Optional normalized RMSE on the validation data.
  - `nrmse_test`: Optional normalized RMSE on the test data.
  - `coefs`: Coefficients of the trained model.
  - `y_train_pred`: Predicted values for the training data.
  - `y_val_pred`: Optional predicted values for the validation data.
  - `y_test_pred`: Optional predicted values for the test data.
  - `y_pred`: Combined predicted values.
  - `mod`: Trained Ridge model instance.
  - `df_int`: Degrees of freedom used in the model.
  - `lambda_`: Regularization parameter used.
  - `lambda_hp`: Hyperparameter value for lambda.
  - `lambda_max`: Maximum lambda value calculated.
  - `lambda_min_ratio`: Minimum ratio for lambda.

# CLASS
## RidgeModelBuilder
* This class is responsible for building and training Ridge regression models.
* It utilizes various external libraries like `scikit-learn`, `nevergrad`, and `pandas` for model optimization and data manipulation.
* It integrates with other parts of the system, taking in data entities such as `MMMData`, `HolidaysData`, and more.

# CONSTRUCTORS
## `__init__(self, mmm_data: MMMData, holiday_data: HolidaysData, calibration_input: CalibrationInput, hyperparameters: Hyperparameters, featurized_mmm_data: FeaturizedMMMData)`
* Initializes the `RidgeModelBuilder` with necessary data components.
* Parameters:
  - `mmm_data`: Instance of `MMMData`, containing marketing mix modeling data.
  - `holiday_data`: Instance of `HolidaysData`, containing holiday-specific data.
  - `calibration_input`: Instance of `CalibrationInput`, used for calibration in modeling.
  - `hyperparameters`: Instance of `Hyperparameters`, specifying parameters for model optimization.
  - `featurized_mmm_data`: Instance of `FeaturizedMMMData`, containing engineered features for modeling.

### USAGE
* Use this constructor when you have all the required data and are ready to set up a new Ridge model builder.

### IMPL
* Initializes instance variables to store the provided data and a logger for logging purposes.
* Sets up logging using `logging.getLogger(__name__)`.

# METHODS
## `build_models(self, trials_config: TrialsConfig, dt_hyper_fixed: Optional[pd.DataFrame] = None, ts_validation: bool = False, add_penalty_factor: bool = False, seed: int = 123, rssd_zero_penalty: bool = True, objective_weights: Optional[List[float]] = None, nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE, intercept: bool = True, intercept_sign: str = "non_negative", cores: int = 2) -> ModelOutputs`
### USAGE
* Builds Ridge regression models using specified configurations and hyperparameters.
* Parameters:
  - `trials_config`: Configuration for the number of trials and iterations.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `ts_validation`: Boolean flag for time-series validation.
  - `add_penalty_factor`: Boolean flag to add penalty factors.
  - `seed`: Seed for random number generation.
  - `rssd_zero_penalty`: Boolean flag for zero penalty in RSSD calculation.
  - `objective_weights`: List of weights for different objectives.
  - `nevergrad_algo`: Algorithm from Nevergrad for optimization.
  - `intercept`: Boolean flag to include intercept in the model.
  - `intercept_sign`: Sign of the intercept.
  - `cores`: Number of CPU cores to use for parallel processing.

### IMPL
* Records start time using `time.time()`.
* Extracts date information from `mmm_data` to determine intervals and window lengths.
* Logs input data details and model configuration using the logger.
* Collects hyperparameters using the `_hyper_collector` method.
* Logs the start of model trials with configured settings.
* Trains models through the `_model_train` method.
* Measures total run time and logs it.
* Uses the `Convergence` class to assess model convergence.
* Compiles and returns results in a `ModelOutputs` instance after aggregating trial results.

## `_select_best_model(self, output_models: List[Trial]) -> str`
### USAGE
* Selects the best model from a list of trials based on combined scores.
* Parameter:
  - `output_models`: List of `Trial` instances representing model output from each trial.

### IMPL
* Extracts NRMSE and decomp RSSD values from each trial in `output_models`.
* Normalizes these metrics to ensure comparability.
* Calculates a combined score by summing normalized metrics.
* Identifies and returns the solution ID of the trial with the lowest combined score.

## `_model_train(self, hyper_collect: Dict[str, Any], trials_config: TrialsConfig, intercept_sign: str, intercept: bool, nevergrad_algo: NevergradAlgorithm, dt_hyper_fixed: Optional[pd.DataFrame], ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], rssd_zero_penalty: bool, seed: int, cores: int) -> List[Trial]`
### USAGE
* Trains multiple models over several trials and iterations using specified configurations.
* Parameters:
  - `hyper_collect`: Dictionary of hyperparameters collected for optimization.
  - `trials_config`: Configuration for trials and iterations.
  - `intercept_sign`: Sign of the intercept for the model.
  - `intercept`: Boolean flag to include intercept in the model.
  - `nevergrad_algo`: Optimization algorithm from Nevergrad.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `ts_validation`: Boolean flag for time-series validation.
  - `add_penalty_factor`: Boolean flag to add penalty factors.
  - `objective_weights`: List of weights for different objectives.
  - `rssd_zero_penalty`: Boolean for zero penalty in RSSD calculation.
  - `seed`: Seed for random number generation.
  - `cores`: Number of CPU cores for parallel processing.

### IMPL
* Initiates an empty list `trials` to store trial results.
* Iterates through each specified trial, invoking `_run_nevergrad_optimization` to conduct optimization.
* Collects results of each trial in the `trials` list.
* Returns the list of completed `Trial` objects.

## `_run_nevergrad_optimization(self, hyper_collect: Dict[str, Any], iterations: int, cores: int, nevergrad_algo: NevergradAlgorithm, intercept: bool, intercept_sign: str, ts_validation: bool, add_penalty_factor: bool, objective_weights: Optional[List[float]], dt_hyper_fixed: Optional[pd.DataFrame], rssd_zero_penalty: bool, trial: int, seed: int, total_trials: int) -> Trial`
### USAGE
* Executes the optimization process using Nevergrad over a specified number of iterations.
* Parameters:
  - `hyper_collect`: Dictionary of hyperparameters to optimize.
  - `iterations`: Number of iterations per trial.
  - `cores`: Number of CPU cores for parallel processing.
  - `nevergrad_algo`: Algorithm from Nevergrad for optimization.
  - `intercept`: Boolean flag to include intercept.
  - `intercept_sign`: Sign of the intercept.
  - `ts_validation`: Boolean flag for time-series validation.
  - `add_penalty_factor`: Boolean flag for adding penalty factors.
  - `objective_weights`: List of weights for different objectives.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `rssd_zero_penalty`: Boolean for zero penalty in RSSD calculation.
  - `trial`: Current trial number.
  - `seed`: Random seed.
  - `total_trials`: Total number of trials.

### IMPL
* Ignores specific warnings during optimization setup using `filterwarnings`.
* Initializes the random seed for reproducibility.
* Constructs parameter bounds for optimization using `hyper_collect`.
* Configures the Nevergrad optimizer with specified settings.
* Executes the optimization loop, using `_evaluate_model` to assess each candidate.
* Updates optimizer with results and progress bar for visual feedback.
* Aggregates results and identifies the best trial outcome based on loss.
* Returns a `Trial` instance encapsulating optimization results.

## `_calculate_decomp_spend_dist(self, model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> pd.DataFrame`
### USAGE
* Calculates decomposition of spend distribution for a given model.
* Parameters:
  - `model`: Trained Ridge model.
  - `X`: Feature data used in training.
  - `y`: Target variable.
  - `params`: Dictionary of model parameters.

### IMPL
* Identifies and processes columns related to paid media spends.
* Computes metrics such as decomposition proportions and means.
* Constructs and returns a DataFrame containing decomposition results and model metrics.

## `_calculate_x_decomp_agg(self, model: Ridge, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> pd.DataFrame`
### USAGE
* Calculates aggregated decomposition for each feature in the model.
* Parameters:
  - `model`: Trained Ridge model.
  - `X`: Feature data used in training.
  - `y`: Target variable.
  - `params`: Dictionary of model parameters.

### IMPL
* Derives decomposition for each feature using model coefficients.
* Returns a DataFrame with aggregated decomposition data and additional performance metrics.

## `_prepare_data(self, params: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series]`
### USAGE
* Prepares data for model training by applying transformations and feature engineering.
* Parameters:
  - `params`: Dictionary of hyperparameters for transformations.

### IMPL
* Renames dependent variable and converts dates to numeric format.
* Applies one-hot encoding to categorical variables.
* Implements transformations like geometric adstock and Hill transformation.
* Cleans data by handling NaN and infinite values.
* Returns prepared features and target variable as a tuple.

## `_geometric_adstock(self, x: pd.Series, theta: float) -> pd.Series`
### USAGE
* Applies geometric adstock transformation to a series.
* Parameters:
  - `x`: Input series to transform.
  - `theta`: Adstock decay parameter.

### IMPL
* Iteratively applies adstock transformation across the series.
* Returns the transformed series.

## `_hill_transformation(self, x: pd.Series, alpha: float, gamma: float) -> pd.Series`
### USAGE
* Applies Hill transformation to a series.
* Parameters:
  - `x`: Input series to transform.
  - `alpha`: Hill curve parameter for scaling.
  - `gamma`: Hill curve parameter for shape.

### IMPL
* Normalizes, scales, and applies Hill transformation to the input.
* Returns the transformed series.

## `_calculate_rssd(self, coefs: np.ndarray, rssd_zero_penalty: bool) -> float`
### USAGE
* Calculates Root Sum of Squared Differences (RSSD) for model coefficients.
* Parameters:
  - `coefs`: Model coefficients.
  - `rssd_zero_penalty`: Boolean flag to apply zero penalty.

### IMPL
* Computes RSSD and adjusts for zero coefficients if specified.
* Returns the computed RSSD value.

## `_calculate_mape(self, model: Ridge) -> float`
### USAGE
* Calculates Mean Absolute Percentage Error (MAPE) for the model based on calibration input.
* Parameter:
  - `model`: Trained Ridge model.

### IMPL
* Iterates over calibration data to compute actual and predicted lifts.
* Calculates mean MAPE and returns it.

## `_evaluate_model(self, params: Dict[str, float], ts_validation: bool, add_penalty_factor: bool, rssd_zero_penalty: bool, objective_weights: Optional[List[float]], start_time: float, iter_ng: int, trial: int) -> Dict[str, Any]`
### USAGE
* Evaluates a model based on provided parameters and returns performance metrics.
* Parameters:
  - `params`: Dictionary of model hyperparameters.
  - `ts_validation`: Boolean flag for time-series validation.
  - `add_penalty_factor`: Boolean flag to add penalty factors.
  - `rssd_zero_penalty`: Boolean flag for zero penalty in RSSD.
  - `objective_weights`: List of objective weights.
  - `start_time`: Start time of the evaluation.
  - `iter_ng`: Current iteration in Nevergrad optimization.
  - `trial`: Current trial number.

### IMPL
* Prepares and splits data for training and validation as needed.
* Fits a Ridge model and computes various performance metrics.
* Calculates decomposition metrics and aggregates results.
* Returns a dictionary with model evaluation metrics and results.

## `_hyper_collector(hyperparameters_dict: Dict[str, Any], ts_validation: bool, add_penalty_factor: bool, dt_hyper_fixed: Optional[pd.DataFrame], cores: int) -> Dict[str, Any]`
### USAGE
* Collects and organizes hyperparameters for model optimization.
* Parameters:
  - `hyperparameters_dict`: Dictionary of prepared hyperparameters.
  - `ts_validation`: Boolean flag for time-series validation.
  - `add_penalty_factor`: Boolean flag to add penalty factors.
  - `dt_hyper_fixed`: Optional fixed hyperparameter DataFrame.
  - `cores`: Number of CPU cores for parallel processing.

### IMPL
* Logs hyperparameter collection process.
* Differentiates between fixed and optimized hyperparameters.
* Returns a dictionary containing collected hyperparameters and related data.

## `_model_refit(x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None, lambda_: float = 1.0, lower_limits: Optional[List[float]] = None, upper_limits: Optional[List[float]] = None, intercept: bool = True, intercept_sign: str = "non_negative") -> ModelRefitOutput`
### USAGE
* Refits a model on given training data and optionally on validation and test data.
* Parameters:
  - `x_train`: Training feature matrix.
  - `y_train`: Training target vector.
  - `x_val`: Optional validation feature matrix.
  - `y_val`: Optional validation target vector.
  - `x_test`: Optional test feature matrix.
  - `y_test`: Optional test target vector.
  - `lambda_`: Regularization strength.
  - `lower_limits`: Optional lower limits for coefficients.
  - `upper_limits`: Optional upper limits for coefficients.
  - `intercept`: Boolean flag for fitting intercept.
  - `intercept_sign`: Sign of the intercept.

### IMPL
* Fits a Ridge model to training data.
* Predicts outcomes for train, validation, and test sets as applicable.
* Calculates R-squared and NRMSE metrics for each dataset.
* Returns a `ModelRefitOutput` object containing results and model parameters.

## `_lambda_seq(x: np.ndarray, y: np.ndarray, seq_len: int = 100, lambda_min_ratio: float = 0.0001) -> np.ndarray`
### USAGE
* Generates a sequence of lambda values for regularization.
* Parameters:
  - `x`: Feature matrix.
  - `y`: Target vector.
  - `seq_len`: Length of the lambda sequence.
  - `lambda_min_ratio`: Minimum ratio for lambda sequence.

### IMPL
* Computes the maximum lambda based on input data.
* Creates and returns a logarithmically spaced sequence of lambda values.