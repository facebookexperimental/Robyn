# CLASS
## FeaturizedMMMData
* This class is a data structure for storing the results of feature engineering in Marketing Mix Modeling (MMM) data.
* It uses the `dataclass` decorator, providing an easy-to-use container for data attributes.
* Package: `dataclasses`

## Attributes
- `dt_mod`: A `pandas.DataFrame` representing the transformed dataset, including date and dependent variable transformations.
- `dt_modRollWind`: A `pandas.DataFrame` representing the dataset with rolling window transformations applied.
- `modNLS`: A dictionary of type `Dict[str, Any]` containing results of model runs, including predictions and plots.

# CLASS
## FeatureEngineering
* A class used to perform feature engineering for Marketing Mix Modeling (MMM) data.
* It manages the transformation and preparation of data for further analysis and modeling.
* Package: Custom implementation in Python using libraries like pandas, Prophet, and others.

# CONSTRUCTORS
## `__init__`(mmm_data: MMMData, hyperparameters: Hyperparameters, holidays_data: Optional[HolidaysData] = None)
* Initializes the `FeatureEngineering` class with necessary data and parameters.

### USAGE
* `mmm_data`: An instance of `MMMData` containing the dataset and specifications related to the marketing mix model.
* `hyperparameters`: An instance of `Hyperparameters` which includes parameters used for feature engineering processes.
* `holidays_data`: An optional instance of `HolidaysData` used specifically for incorporating holiday effects using Prophet decomposition.

### IMPL
* The constructor sets up the `mmm_data`, `hyperparameters`, and `holidays_data` as instance variables.
* A logger is initialized for the class to manage logging of information and warnings during the feature engineering process.

# METHODS
## `perform_feature_engineering(quiet: bool = False) -> FeaturizedMMMData`
### USAGE
* Executes the feature engineering process, transforming the data and preparing it for modeling.
* `quiet`: A boolean flag indicating whether to suppress logging messages during the process.

### IMPL
* Calls `_prepare_data()` to apply initial data transformations.
* Checks if Prophet decomposition is required based on `holidays_data`, and performs it if necessary.
* Constructs a comprehensive list of independent variables from different data specifications.
* Generates a rolling window dataset using `_create_rolling_window_data()`.
* Calculates a media cost factor with `_calculate_media_cost_factor()`.
* Executes model fitting on the rolling window data using `_run_models()`.
* Filters out columns that are not present in both `dt_mod` and `dt_modRollWind`.
* Logs the completion of feature engineering if `quiet` is set to False.
* Returns a `FeaturizedMMMData` instance containing the processed data.

## `_prepare_data() -> pd.DataFrame`
### USAGE
* Prepares the initial dataset by transforming date and dependent variable columns.

### IMPL
* Creates a copy of the original MMM data.
* Converts the date variable to a standardized format and assigns it to a new column `ds`.
* Maps the specified dependent variable to the column `dep_var`.
* Converts `competitor_sales_B` to integer type for consistency.

## `_create_rolling_window_data(dt_transform: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Constructs a rolling window dataset based on specified start and end dates.

### IMPL
* Filters data using `window_start` and `window_end` constraints.
* Raises a `ValueError` if the specified window constraints are invalid or contradictory.

## `_calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* Computes the media cost factor for the provided rolling window dataset.

### IMPL
* Calculates the total media spend.
* Returns a pandas Series representing the media cost factor for each paid media spend.

## `_run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]`
### USAGE
* Executes model runs for each paid media variable and returns the results.

### IMPL
* Initializes `modNLS` dictionary to store results, predictions, and plots.
* Iterates through paid media spends and fits the spend-exposure model using `_fit_spend_exposure()`.
* Aggregates results and plots into `modNLS`.

## `_fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict[str, Any]`
### USAGE
* Fits the spend-exposure model for a given paid media variable.

### IMPL
* Logs the progress for the current `paid_media_var`.
* Defines a Michaelis-Menten model and fits it to spend and exposure data.
* Calculates R-squared values for both Michaelis-Menten and linear models.
* Selects the model with the higher R-squared value and returns the result along with plot data.
* Handles exceptions by falling back to a linear model if needed.

## `_hill_function(x, alpha, gamma)`
### USAGE
* Static method to apply the Hill function transformation.

### IMPL
* Computes the Hill function using provided parameters `alpha` and `gamma`.

## `_prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Applies Prophet decomposition to the dataset and returns the transformed data.

### IMPL
* Prepares data and holidays for Prophet modeling.
* Configures and fits a Prophet model with specified parameters, including seasonalities and regressors.
* Computes forecasts and integrates them into the dataset.

## `_set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* Integrates holidays into the dataset based on the specified interval type.

### IMPL
* Converts `ds` columns to datetime format.
* Adjusts holidays according to interval type ('day', 'week', or 'month').
* Aggregates and returns the modified holidays dataframe.

## `_apply_transformations(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies adstock and saturation transformations to the provided series.

### IMPL
* Utilizes `_apply_adstock()` and `_apply_saturation()` methods to perform transformations and returns the modified series.

## `_apply_adstock(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies the specified adstock transformation to the given series.

### IMPL
* Identifies the adstock type and applies the corresponding transformation (`_geometric_adstock()` or `_weibull_adstock()`).

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* Static method to execute geometric adstock transformation.

### IMPL
* Utilizes an exponentially weighted moving average to compute the adstock effect.

## `_weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* Static method to apply Weibull adstock transformation.

### IMPL
* Calculates Weibull PDF weights and performs convolution on the series.

## `_apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Static method to apply saturation transformation.

### IMPL
* Applies the Hill function for saturation using parameters `alpha` and `gamma`.