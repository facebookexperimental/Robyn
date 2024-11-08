# CLASS
## ParetoResult
* Holds the results of Pareto optimization for marketing mix models.
* It is a data class that stores several attributes related to Pareto-optimal solutions.
* Attributes:
  - `pareto_solutions (List[str])`: A list of solution IDs that are Pareto-optimal.
  - `pareto_fronts (int)`: The number of Pareto fronts considered in the optimization.
  - `result_hyp_param (pd.DataFrame)`: Hyperparameters of Pareto-optimal solutions.
  - `x_decomp_agg (pd.DataFrame)`: Aggregated decomposition results for Pareto-optimal solutions.
  - `result_calibration (Optional[pd.DataFrame])`: Calibration results, if calibration was performed.
  - `media_vec_collect (pd.DataFrame)`: Collected media vectors for all Pareto-optimal solutions.
  - `x_decomp_vec_collect (pd.DataFrame)`: Collected decomposition vectors for all Pareto-optimal solutions.
  - `plot_data_collect (Dict[str, pd.DataFrame])`: Data for various plots, keyed by plot type.
  - `df_caov_pct_all (pd.DataFrame)`: Carryover percentage data for all channels and Pareto-optimal solutions.

## ParetoData
* This class holds data necessary for Pareto optimization in marketing mix models.
* It is a data class that contains attributes for handling decomposition of spending and hyperparameters.
* Attributes:
  - `decomp_spend_dist (pd.DataFrame)`: Decomposed spending distribution.
  - `result_hyp_param (pd.DataFrame)`: Result hyperparameters.
  - `x_decomp_agg (pd.DataFrame)`: Aggregated decomposition results.
  - `pareto_fronts (List[int])`: List of Pareto fronts.

## ParetoOptimizer
* Performs Pareto optimization on marketing mix models.
* This class orchestrates the Pareto optimization process, including data aggregation, Pareto front calculation, response curve calculation, and plot data preparation.
* Attributes:
  - `mmm_data (MMMData)`: Input data for the marketing mix model.
  - `model_outputs (ModelOutputs)`: Output data from the model runs.
  - `response_calculator (ResponseCurveCalculator)`: Calculator for response curves.
  - `carryover_calculator (ImmediateCarryoverCalculator)`: Calculator for immediate and carryover effects.
  - `pareto_utils (ParetoUtils)`: Utility functions for Pareto-related calculations.

# CONSTRUCTORS
## ParetoOptimizer `(mmm_data: MMMData, model_outputs: ModelOutputs, hyper_parameter: Hyperparameters, featurized_mmm_data: FeaturizedMMMData, holidays_data: HolidaysData)`
* Initializes the ParetoOptimizer with the necessary data for optimization.

### USAGE
* Use this constructor when you need to perform Pareto optimization on marketing mix models using the specified inputs.

### IMPL
* Initializes the following attributes:
  - `self.mmm_data`: Stores the input marketing mix model data.
  - `self.model_outputs`: Stores the output data from model runs.
  - `self.hyper_parameter`: Stores the hyperparameters for the model runs.
  - `self.featurized_mmm_data`: Stores the featurized marketing mix model data.
  - `self.holidays_data`: Stores the holidays data.
  - `self.transformer`: Initialized as a `Transformation` instance using `mmm_data`.

# METHODS
## `optimize(pareto_fronts: str = "auto", min_candidates: int = 100, calibration_constraint: float = 0.1, calibrated: bool = False) -> ParetoResult`
### USAGE
* Parameters:
  - `pareto_fronts (str)`: Number of Pareto fronts to consider or "auto" for automatic selection.
  - `min_candidates (int)`: Minimum number of candidates to consider when using "auto" Pareto fronts.
  - `calibration_constraint (float)`: Constraint for calibration, used if models are calibrated.
  - `calibrated (bool)`: Whether the models have undergone calibration.
* Use this method to perform the entire Pareto optimization process.

### IMPL
* Calls `_aggregate_model_data` to aggregate model data based on calibration status.
* Computes Pareto fronts using `_compute_pareto_fronts` with aggregated data and specified parameters.
* Prepares Pareto data using `prepare_pareto_data` with aggregated data and specified parameters.
* Computes response curves with `_compute_response_curves` using Pareto data and aggregated data.
* Generates plot data with `_generate_plot_data` using aggregated data and Pareto data.
* Returns a `ParetoResult` containing the optimization results.

## `_aggregate_model_data(calibrated: bool) -> Dict[str, pd.DataFrame]`
### USAGE
* Parameter:
  - `calibrated (bool)`: Indicates whether calibration was performed.
* Aggregates and prepares data from model outputs for Pareto optimization.

### IMPL
* Extracts hyperparameters, decomposition results, and calibration data from model outputs.
* Concatenates data into DataFrames using `pd.concat`.
* Adds iteration numbers based on whether hyperparameters are fixed.
* Merges bootstrap results with `xDecompAgg` if available.
* Returns a dictionary containing aggregated data, including 'result_hyp_param', 'x_decomp_agg', and 'result_calibration'.

## `_compute_pareto_fronts(aggregated_data: Dict[str, pd.DataFrame], pareto_fronts: str, min_candidates: int, calibration_constraint: float) -> pd.DataFrame`
### USAGE
* Parameters:
  - `aggregated_data (Dict[str, pd.DataFrame])`: Aggregated model data.
  - `pareto_fronts (str)`: Number of Pareto fronts to compute or "auto".
  - `min_candidates (int)`: Minimum number of candidates for automatic selection.
  - `calibration_constraint (float)`: Calibration constraint.
* Calculates Pareto fronts from aggregated data.

### IMPL
* Filters and groups data to calculate coefficients and quantiles.
* Identifies Pareto-optimal solutions based on NRMSE, DECOMP.RSSD, and MAPE criteria.
* Assigns solutions to Pareto fronts.
* Computes combined weighted error scores using `ParetoUtils.calculate_errors_scores`.
* Returns a DataFrame of Pareto-optimal solutions with their corresponding front numbers.

## `prepare_pareto_data(aggregated_data: Dict[str, pd.DataFrame], pareto_fronts: str, min_candidates: int, calibrated: bool) -> ParetoData`
### USAGE
* Parameters:
  - `aggregated_data (Dict[str, pd.DataFrame])`: Aggregated data for processing.
  - `pareto_fronts (str)`: Pareto fronts configuration.
  - `min_candidates (int)`: Minimum candidates for selection.
  - `calibrated (bool)`: Indicates calibration status.
* Prepares data for Pareto optimization.

### IMPL
* Merges decomposed spend distribution with result hyperparameters.
* Handles automatic Pareto front selection and filtering based on configuration.
* Returns a `ParetoData` instance with filtered data for selected Pareto fronts.

## `run_dt_resp(row: pd.Series, paretoData: ParetoData) -> pd.Series`
### USAGE
* Parameters:
  - `row (pd.Series)`: A row of Pareto data.
  - `paretoData (ParetoData)`: Pareto data instance.
* Calculates response curves for a given row, used for parallel processing.

### IMPL
* Calculates response curves using `ResponseCurveCalculator`.
* Computes mean spend adstocked and carryover values.
* Returns computed values for response, spend, and carryover in a pandas Series.

## `_compute_response_curves(pareto_data: ParetoData, aggregated_data: Dict[str, pd.DataFrame]) -> ParetoData`
### USAGE
* Parameters:
  - `pareto_data (ParetoData)`: Pareto data for processing.
  - `aggregated_data (Dict[str, pd.DataFrame])`: Aggregated data for reference.
* Computes response curves for Pareto-optimal solutions.

### IMPL
* Utilizes parallel processing to compute response curves for media variables.
* Merges computed response curves into `ParetoData`.
* Calculates ROI and CPA metrics.
* Returns updated `ParetoData` with computed response curves.

## `_generate_plot_data(aggregated_data: Dict[str, pd.DataFrame], pareto_data: ParetoData) -> Dict[str, pd.DataFrame]`
### USAGE
* Parameters:
  - `aggregated_data (Dict[str, pd.DataFrame])`: Aggregated data for plotting.
  - `pareto_data (ParetoData)`: Pareto data for visualization.
* Prepares data for various plots used in Pareto analysis.

### IMPL
* Iterates over Pareto fronts, generating plot data for spend vs. effect share, waterfall plots, adstock rates, and spend response curves.
* Collects media vectors, decomposition vectors, and plot data into dictionaries.
* Returns a dictionary containing plot data for visualization.

## `robyn_immcarr(pareto_data: ParetoData, result_hyp_param: pd.DataFrame, solID=None, start_date=None, end_date=None)`
### USAGE
* Parameters:
  - `pareto_data (ParetoData)`: Pareto data for analysis.
  - `result_hyp_param (pd.DataFrame)`: Result hyperparameters.
  - `solID`: Optional solution ID.
  - `start_date`: Optional start date for analysis.
  - `end_date`: Optional end date for analysis.
* Analyzes immediate and carryover response, calculating percentages.

### IMPL
* Extracts and processes hyperparameters for the specified solution.
* Runs transformations and decompositions on the data.
* Calculates media decomposition and carryover percentages.
* Returns a DataFrame with immediate and carryover response percentages.

## `_extract_hyperparameter(hypParamSam: pd.DataFrame) -> Hyperparameters`
### USAGE
* Parameter:
  - `hypParamSam (pd.DataFrame)`: DataFrame containing hyperparameters.
* Extracts hyperparameters from the provided DataFrame.

### IMPL
* Iterates over media channels, extracting relevant hyperparameters such as alphas, gammas, thetas, shapes, and scales.
* Constructs and returns a `Hyperparameters` instance with extracted values.

## `_model_decomp(inputs) -> Dict[str, pd.DataFrame]`
### USAGE
* Parameter:
  - `inputs (dict)`: Dictionary containing decomposition inputs.
* Decomposes model outputs into immediate and carryover responses.

### IMPL
* Extracts coefficients and performs decomposition on model data.
* Computes immediate and carryover responses using coefficients.
* Returns a dictionary with decomposition results, including `xDecompVec`, `mediaDecompImmediate`, and `mediaDecompCarryover`.