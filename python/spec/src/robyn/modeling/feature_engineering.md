# CLASS
## FeaturizedMMMData
* This class is a data structure defined using the `dataclass` decorator, tailored to encapsulate the results of feature engineering within Marketing Mix Modeling (MMM) data.
* It serves as a container for:
  - `dt_mod`: A `pandas.DataFrame` that represents the dataset after modifications.
  - `dt_modRollWind`: A `pandas.DataFrame` containing data with rolling window transformations.
  - `modNLS`: A dictionary that holds results from non-linear model fits and related data.

# CLASS
## FeatureEngineering
* The `FeatureEngineering` class is designed to conduct feature engineering tasks on marketing mix modeling (MMM) data.
* It processes and augments data to improve model performance using various methods and techniques.
* This class handles data preparation, transformation, and model fitting.

# CONSTRUCTORS
## FeatureEngineering `(mmm_data: MMMData, hyperparameters: Hyperparameters, holidays_data: Optional[HolidaysData] = None)`
* This constructor initializes an instance of the `FeatureEngineering` class.
* **Parameters:**
  - `mmm_data`: An instance of `MMMData` containing the relevant data and specifications for the marketing mix model.
  - `hyperparameters`: An instance of `Hyperparameters` that configures the parameters for modeling.
  - `holidays_data`: An optional instance of `HolidaysData` that provides holiday-related information for Prophet model adjustments.
### USAGE
* Use this constructor to set up feature engineering for MMM data, applying specified hyperparameters and optionally incorporating holiday data.
### IMPL
* The constructor assigns the provided `mmm_data`, `hyperparameters`, and `holidays_data` to the instance variables `self.mmm_data`, `self.hyperparameters`, and `self.holidays_data`, thereby setting up the internal state of the `FeatureEngineering` object.

# METHODS
## `perform_feature_engineering(quiet: bool = False) -> FeaturizedMMMData`
### USAGE
* Executes the full feature engineering process on the provided MMM data.
* **Parameters:**
  - `quiet`: A boolean flag indicating whether to suppress print statements during execution. Default is `False`.
### IMPL
* Initiates the data preparation process by calling `_prepare_data()`.
* Checks if Prophet decomposition is necessary by examining the presence of specific variables in `holidays_data.prophet_vars`.
* Performs Prophet decomposition if required, and outputs a completion message unless `quiet` is `True`.
* Collects all independent variables, including Prophet variables, context variables, paid media spends, and organic variables.
* Calls `_create_rolling_window_data()` to generate rolling window data.
* Computes the media cost factor using `_calculate_media_cost_factor()`.
* Executes model fitting through `_run_models()`, storing results in `modNLS`.
* Filters the dataframes `dt_mod` and `dt_modRollWind` to retain only necessary columns by checking column existence in both frames.
* Returns a `FeaturizedMMMData` object encapsulating `dt_mod`, `dt_modRollWind`, and `modNLS`.

## `_prepare_data() -> pd.DataFrame`
### USAGE
* Prepares and transforms the MMM data to make it suitable for further processing.
### IMPL
* Creates a copy of the MMM data from `mmm_data.data`.
* Converts the date variable specified in `mmm_data.mmmdata_spec.date_var` to a standardized date format (`"%Y-%m-%d"`) and assigns it to a new column `ds`.
* Maps the dependent variable specified in `mmm_data.mmmdata_spec.dep_var` to the `dep_var` column.
* Changes the data type of `competitor_sales_B` to `int64` for subsequent calculations.
* Returns the transformed `pandas.DataFrame`.

## `_create_rolling_window_data(dt_transform: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Creates a rolling window subset of the data based on specified start and end dates within the MMM data specifications.
* **Parameters:**
  - `dt_transform`: A `pandas.DataFrame` containing the pre-transformed data.
### IMPL
* Retrieves `window_start` and `window_end` from `mmm_data.mmmdata_spec`.
* Filters the data according to the presence of these window boundaries, allowing for:
  - Data up to `window_end` if `window_start` is `None`.
  - Data from `window_start` onward if `window_end` is `None`.
  - Data within both start and end dates if both are specified.
* Returns the filtered `pandas.DataFrame`.

## `_calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* Calculates the media cost factor for paid media spends within the rolling window data.
* **Parameters:**
  - `dt_input_roll_wind`: A `pandas.DataFrame` containing rolling window data.
### IMPL
* Computes the total spend across all paid media variables using `sum()`.
* Divides the sum of each paid media spend by the total spend to determine the media cost factor.
* Returns a `pandas.Series` with the calculated cost factors.

## `_run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]`
### USAGE
* Fits models to the rolling window data using both Michaelis-Menten and linear models.
* **Parameters:**
  - `dt_modRollWind`: A `pandas.DataFrame` containing rolling window data.
  - `media_cost_factor`: A float value representing the media cost factor.
### IMPL
* Initializes a dictionary `modNLS` to store model results, predictions (`yhat`), and plots.
* Iterates over each paid media variable in `mmm_data.mmmdata_spec.paid_media_spends`.
* Calls `_fit_spend_exposure()` for each variable to fit the data and retrieve model results.
* Appends results to `modNLS`, including concatenating prediction plots.
* Returns the `modNLS` dictionary containing model fit results.

## `_fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict[str, Any]`
### USAGE
* Fits spend-exposure models for each paid media variable using Michaelis-Menten and linear models.
* **Parameters:**
  - `dt_modRollWind`: A `pandas.DataFrame` with rolling window data.
  - `paid_media_var`: A string representing the name of the paid media variable being analyzed.
  - `media_cost_factor`: A float indicating the media cost factor.
### IMPL
* Defines the `michaelis_menten` function for model fitting.
* Retrieves spend and exposure data from `dt_modRollWind`.
* Attempts to fit the Michaelis-Menten model using `curve_fit`, handling exceptions to default to a linear model if errors occur.
* Calculates R-squared values for both models to assess fit quality.
* Selects the model with the higher R-squared value, storing details in a results dictionary.
* Constructs a DataFrame for plotting model predictions against actual data.
* Returns a dictionary containing model results and plot data.

## `_hill_function(x, alpha, gamma) -> float`
### USAGE
* Represents the Hill function used for saturation transformation.
* **Parameters:**
  - `x`: The input data series.
  - `alpha`: A parameter indicating the sensitivity of the transformation.
  - `gamma`: A parameter representing the threshold of the transformation.
### IMPL
* Computes the Hill function transformation using the formula: \(\frac{x^\alpha}{x^\alpha + \gamma^\alpha}\).
* Returns the transformed series as a float.

## `_prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Applies Prophet decomposition to extract trend, seasonality, and holiday effects from the data.
* **Parameters:**
  - `dt_mod`: A `pandas.DataFrame` with the modified data.
### IMPL
* Sets up Prophet parameters using available holiday data and specified variables.
* Fits the Prophet model to the data and predicts components such as trend, seasonality, and holidays.
* Returns a `pandas.DataFrame` with the decomposed data including added columns for extracted components.

## `_set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* Configures and processes holiday data according to the specified interval type.
* **Parameters:**
  - `dt_transform`: A `pandas.DataFrame` with transformed data.
  - `dt_holidays`: A `pandas.DataFrame` containing holiday information.
  - `interval_type`: A string specifying the interval type ('day', 'week', or 'month').
### IMPL
* Ensures `ds` columns in both dataframes are of datetime type.
* Processes holiday dates based on the interval type, adjusting dates to align with the start of weeks or months.
* Returns the processed holidays data as a `pandas.DataFrame`.

## `_apply_transformations(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies adstock and saturation transformations to a given data series.
* **Parameters:**
  - `x`: A `pandas.Series` representing the data to be transformed.
  - `params`: An instance of `ChannelHyperparameters` containing transformation parameters.
### IMPL
* Applies adstock transformations based on specified parameters using `_apply_adstock()`.
* Applies saturation transformations using `_apply_saturation()`.
* Returns the fully transformed `pandas.Series`.

## `_apply_adstock(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies adstock transformation according to the specified adstock type in the hyperparameters.
* **Parameters:**
  - `x`: A `pandas.Series` representing the data to be adstocked.
  - `params`: An instance of `ChannelHyperparameters` specifying adstock parameters.
### IMPL
* Determines the adstock type from `hyperparameters.adstock`.
* Applies the appropriate adstock transformation (geometric or Weibull) to the series.
* Returns the adstocked `pandas.Series`.

## `_geometric_adstock(x: pd.Series, theta: float) -> pd.Series`
### USAGE
* Applies a geometric adstock transformation to a data series.
* **Parameters:**
  - `x`: A `pandas.Series` representing the input data.
  - `theta`: A float parameter controlling the decay rate of the adstock effect.
### IMPL
* Uses exponential weighted mean to compute adstocked values.
* Returns the adstocked data as a `pandas.Series`.

## `_weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* Applies a Weibull adstock transformation to a data series.
* **Parameters:**
  - `x`: A `pandas.Series` representing the data.
  - `shape`: A float indicating the shape parameter of the Weibull distribution.
  - `scale`: A float indicating the scale parameter of the Weibull distribution.
### IMPL
* Calculates Weibull probability density function weights.
* Convolves these weights with the input series to apply the transformation.
* Returns the transformed series as a `pandas.Series`.

## `_apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies saturation transformation using the Hill function.
* **Parameters:**
  - `x`: A `pandas.Series` to be transformed.
  - `params`: An instance of `ChannelHyperparameters` containing saturation parameters.
### IMPL
* Utilizes the Hill function with parameters `alpha` and `gamma` to transform the input series.
* Returns the saturated `pandas.Series`.