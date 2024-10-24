# CLASS
## FeatureEngineering
* A class used to perform feature engineering for Marketing Mix Modeling (MMM) data.
* Part of the "robyn" package, handling data transformations and model executions.
* Involves methods for data preparation, adstock transformations, decomposition using Prophet, and model fitting.

# CONSTRUCTORS
## `__init__(mmm_data: MMMData, hyperparameters: Hyperparameters, holidays_data: Optional[HolidaysData] = None)`
* Initializes the FeatureEngineering class with the given MMM data, hyperparameters, and optional holidays data.
### USAGE
* Use this constructor to create an instance of FeatureEngineering with the necessary data and configuration for feature engineering.
### IMPL
* Ensure that the MMMData, Hyperparameters, and optional HolidaysData inputs are correctly assigned to the class attributes.
* Verify that a logger is properly initialized for logging information and warnings.

# METHODS
## `perform_feature_engineering(quiet: bool = False) -> FeaturizedMMMData`
### USAGE
* Initiates the feature engineering process on the provided data, with optional logging based on the quiet parameter.
### IMPL
* Mock the dependencies `_prepare_data`, `_prophet_decomposition`, `_create_rolling_window_data`, `_calculate_media_cost_factor`, and `_run_models` in sequence.
* Validate the transformation of the initial dataset into a featurized form with rolling window adjustments, media cost factors, and model results.
* Execute assertions to confirm that the result is an instance of FeaturizedMMMData with expected DataFrame attributes.

## `_prepare_data() -> pd.DataFrame`
### USAGE
* Transforms the initial dataset by converting date and dependent variable columns to the required format.
### IMPL
* Mock the MMMData dependency to return a copy of the data with specified columns.
* Assert the transformation of date_var to a string format and the presence of dep_var and competitor_sales_B with correct types.

## `_create_rolling_window_data(dt_transform: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Creates a dataset with rolling window constraints based on provided start and end dates.
### IMPL
* Mock the MMMData to return window_start and window_end values.
* Assert that the resulting dataset respects the defined window boundaries, raising exceptions if constraints are violated.

## `_calculate_media_cost_factor(dt_input_roll_wind: pd.DataFrame) -> pd.Series`
### USAGE
* Computes the media cost factor for the given rolling window dataset.
### IMPL
* Test with various scenarios of media spends (valid, zero, missing, negative) to confirm correct calculation of cost factors.
* Ensure handling of division by zero or missing data without raising unexpected exceptions.

## `_run_models(dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]`
### USAGE
* Executes model fitting for each paid media variable and returns results.
### IMPL
* Mock the `_fit_spend_exposure` method to simulate model fitting for each media variable.
* Verify the output dictionary structure, confirming that results and plots are populated as expected based on input data.

## `_fit_spend_exposure(dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float) -> Dict[str, Any]`
### USAGE
* Fits a model to relate media spend to exposure, selecting between nonlinear and linear models based on R-squared values.
### IMPL
* Mock the MMMData to provide media spend and exposure data.
* Simulate curve fitting and linear model fitting, comparing R-squared values to determine the best model.
* Assert correct handling of empty data or exceptions, ensuring proper fallbacks and logging of warnings.

## `_hill_function(x, alpha, gamma)`
### USAGE
* Applies the Hill function transformation to a given input value.
### IMPL
* Test various input values for x, alpha, and gamma, confirming the mathematical correctness of the transformation.
* Include edge cases such as zero or negative values to verify the function handles them without errors.

## `_prophet_decomposition(dt_mod: pd.DataFrame) -> pd.DataFrame`
### USAGE
* Performs decomposition using Prophet, adding trend, seasonality, and holiday effects to the dataset.
### IMPL
* Mock the HolidaysData to provide prophet variables and holiday data.
* Test different prophet configurations, ensuring the resulting DataFrame includes the expected decomposition components.

## `_set_holidays(dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str) -> pd.DataFrame`
### USAGE
* Integrates holidays into the dataset according to the specified interval type.
### IMPL
* Confirm the proper setting of holidays for daily, weekly, and monthly intervals, raising exceptions for invalid types.
* Test with different holiday datasets to ensure correct grouping and aggregation based on interval type.

## `_apply_adstock(x: pd.Series, params: ChannelHyperparameters) -> pd.Series`
### USAGE
* Applies adstock transformation using specified parameters to the input series.
### IMPL
* Test for each adstock type (GEOMETRIC, WEIBULL_CDF, WEIBULL_PDF) to confirm transformation correctness.
* Validate that unsupported adstock types raise the appropriate exceptions with clear messages.

## `_weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series`
### USAGE
* Applies Weibull adstock transformation to the input series based on shape and scale parameters.
### IMPL
* Test with positive, zero, and negative values for shape and scale, confirming expected behavior and error handling.
* Use series of various lengths to ensure the method scales correctly with input size.