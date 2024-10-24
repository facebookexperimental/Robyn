# CLASS
## FeaturizedMMMData
* This class is a data container specifically used to store the results of feature engineering for Marketing Mix Modeling (MMM) data.
* It holds the modulated data, rolling window modulated data, and the results from non-linear models.
* This class is located in the main file for feature engineering.

# CONSTRUCTORS
## FeaturizedMMMData `(dt_mod: pd.DataFrame, dt_modRollWind: pd.DataFrame, modNLS: Dict[str, Any])`
* **dt_mod**: A pandas DataFrame that contains the modulated data after feature engineering.
* **dt_modRollWind**: A pandas DataFrame representing the rolling window modulated data.
* **modNLS**: A dictionary that holds the results of non-linear model fitting, with keys as model names and values as model outcomes.

### USAGE
* The constructor is used to instantiate a `FeaturizedMMMData` object, encapsulating the results of feature engineering for further analysis or modeling.

### IMPL
* The `@dataclass` decorator is employed, which automatically provides an `__init__` method that initializes class attributes based on the provided parameters.

# CLASS
## FeatureEngineering
* This class is designed to carry out feature engineering specifically for Marketing Mix Modeling (MMM) data.
* It incorporates external data such as holidays and utilizes statistical models to transform and prepare data for analysis.
* The class is located in the main file dedicated to feature engineering tasks.

# CONSTRUCTORS
## FeatureEngineering `(mmm_data: MMMData, hyperparameters: Hyperparameters, holidays_data: Optional[HolidaysData] = None)`
* **mmm_data**: An instance of `MMMData` that includes the dataset and specifications required for MMM.
* **hyperparameters**: An instance of `Hyperparameters` that configures the feature engineering process.
* **holidays_data**: An optional instance of `HolidaysData`, used to include holiday effects via Prophet decomposition.

### USAGE
* Instantiate this class when there is a need to perform feature engineering on MMM data using specific hyperparameters, with the option to factor in holidays data.

### IMPL
* The constructor initializes class variables such as `mmm_data`, `hyperparameters`, `holidays_data`, and a logger instance.
* The logger is set up using Python's `logging` library to log information and warnings throughout the feature engineering process.

# METHODS
## `perform_feature_engineering(quiet: bool = False) -> FeaturizedMMMData`
### USAGE
* **quiet**: A boolean flag to indicate whether logging output should be suppressed. Defaults to `False`.
* This method orchestrates the entire feature engineering process and returns a `FeaturizedMMMData` object containing the results.

### IMPL
* The method begins by preparing the initial dataset through `_prepare_data()`.
* It checks for the presence of Prophet variables and performs decomposition if required, logging the process unless `quiet` is set to `True`.
* Collects all relevant independent variables and transforms the dataset.
* Generates rolling window data and computes the media cost factor.
* Runs models using `_run_models()`.
* Filters columns to retain necessary data in the resulting DataFrames and addresses any missing values.
* Logs the completion of feature engineering if `quiet` is `False`.
* Finally, returns an instance of `FeaturizedMMMData` containing the processed data and model results.

## `_prepare_data() -> pd.DataFrame`
### USAGE
* Prepares the dataset by transforming the date and dependent variable columns for further processing.

### IMPL
* Copies the original data to avoid modifying the input directly.
* Converts the date column to a standardized `YYYY-MM-DD` format.
* Sets the dependent variable column for easier access and transformations.
* Ensures specific variable types, such as converting `competitor_sales_B` to `int64`.

## `_create_rolling_window_data(dt_transform: pd.DataFrame) -> pd.DataFrame`
### USAGE
* **dt_transform**: A pandas DataFrame representing the transformed dataset.
* Creates a rolling window dataset based on specified start and end dates for analysis.

### IMPL
* Filters the dataset according to the window start and end specifications provided in `mmm_data`.
* Raises a `ValueError` if the window specifications are inconsistent with the dataset, ensuring logical integrity.

## `_calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* **dt_input_roll_wind**: A pandas DataFrame of the rolling window input data.
* Calculates the media cost factor for the given rolling window dataset.

### IMPL
* Computes the total spend from the specified paid media spends.
* Returns the media cost factor as a pandas Series, representing the proportion of spend for each media type.

## `_run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]`
### USAGE
* **dt_modRollWind**: A pandas DataFrame containing rolling window modulated data.
* **media_cost_factor**: A float representing the media cost factor.
* Runs statistical models for each paid media variable and returns the results.

### IMPL
* Initializes a dictionary `modNLS` to store model results, yhat predictions, and plots.
* Iterates over each paid media variable, calling `_fit_spend_exposure()` to fit models.
* Aggregates model results into `modNLS` and returns it.

## `_fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict[str, Any]`
### USAGE
* **dt_modRollWind**: A pandas DataFrame of rolling window modulated data.
* **paid_media_var**: A string representing the paid media variable.
* **media_cost_factor**: A float representing the media cost factor.
* Fits spend-exposure models for a given paid media variable and returns the results.

### IMPL
* Logs the processing of the paid media variable.
* Attempts to fit data using the Michaelis-Menten and linear regression models.
* Computes R-squared values to assess model fit and selects the better-performing model.
* Handles exceptions by defaulting to a linear model and logs warnings if necessary.
* Returns a dictionary containing model results, plots, and predicted values.

## `_hill_function(x, alpha, gamma)`
### USAGE
* Static method to apply the Hill function transformation to a dataset.

### IMPL
* Computes the Hill function using the mathematical formula: `x^alpha / (x^alpha + gamma^alpha)`.
* This transformation is used in the feature engineering process to model certain types of relationships.

## `_prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* **dt_mod**: A pandas DataFrame representing the modulated data.
* Performs Prophet decomposition on the dataset and returns the transformed data with additional features.

### IMPL
* Configures and fits a Prophet model using available holiday and seasonal data.
* Incorporates custom parameters if available and manages multiple regressors.
* Logs warnings for known Prophet issues to prevent unexpected errors.
* Updates the dataset with trends, seasonalities, and holidays information.

## `_set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* **dt_transform**: A pandas DataFrame representing the transformed dataset.
* **dt_holidays**: A pandas DataFrame containing holiday data.
* **interval_type**: A string indicating the data interval type ("day", "week", or "month").
* Sets holidays in the dataset based on the specified interval type.

### IMPL
* Ensures date columns are in datetime format for consistency.
* Adjusts holidays according to the interval type and raises a `ValueError` for invalid types.
* Handles aggregation for weekly and monthly intervals to ensure accurate holiday representation.

## `_apply_transformations(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* **x**: A pandas Series representing the data to be transformed.
* **params**: An instance of `ChannelHyperparameters` that contains transformation parameters.
* Applies adstock and saturation transformations to the given series.

### IMPL
* Calls `_apply_adstock()` and `_apply_saturation()` to perform the necessary transformations.
* Returns the transformed series for further analysis or modeling.

## `_apply_adstock(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* **x**: A pandas Series representing the data to be adstocked.
* **params**: An instance of `ChannelHyperparameters` containing adstock parameters.
* Applies the specified adstock transformation to the given series.

### IMPL
* Selects the appropriate adstock function based on the hyperparameter type.
* Supports both geometric and Weibull adstock types.
* Raises a `ValueError` for unsupported adstock types to ensure proper error handling.

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* Static method to apply geometric adstock transformation to a dataset.

### IMPL
* Utilizes an exponential weighted moving average with a specified `theta`.
* Computes the geometric adstocked series for the provided data.

## `_weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* Static method to apply Weibull adstock transformation.

### IMPL
* Computes the Weibull probability density function to generate weights for the transformation.
* Convolves the weights with the input series to produce the adstocked series, capturing delayed effects.

## `_apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Static method to apply saturation transformation to a dataset.

### IMPL
* Computes the saturation transformation using a formula involving `alpha` and `gamma`.
* Returns the transformed series, modeling diminishing returns or saturation effects.