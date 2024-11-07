# CLASS
## MetricDateInfo
* A data class used to store information about metric locations and updated date ranges.
* This class is part of a module related to metrics handling.

# CONSTRUCTORS
## MetricDateInfo `(metric_loc: pd.Series, date_range_updated: pd.Series)`
* `metric_loc`: A pandas Series that contains the location of metrics.
* `date_range_updated`: A pandas Series that contains the updated date range for the metrics.
### USAGE
* Use this constructor to create an instance of `MetricDateInfo` with the specified metric location and date range.
### IMPL
* Initializes the data class with two main attributes: `metric_loc` and `date_range_updated`.

---

# CLASS
## MetricValueInfo
* A data class used to store updated metric values and all updated values.
* This class is part of a module related to metrics handling.

# CONSTRUCTORS
## MetricValueInfo `(metric_value_updated: np.ndarray, all_values_updated: pd.Series)`
* `metric_value_updated`: A numpy array that contains updated metric values.
* `all_values_updated`: A pandas Series that contains all updated values.
### USAGE
* Use this constructor to create an instance of `MetricValueInfo` with the specified updated metric values.
### IMPL
* Initializes the data class with two main attributes: `metric_value_updated` and `all_values_updated`.

---

# CLASS
## ResponseOutput
* A data class that encapsulates the results of a response calculation, including metric names, dates, inputs, responses, and a plot.
* This class is part of a module related to response handling.

# CONSTRUCTORS
## ResponseOutput `(metric_name: str, date: pd.Series, input_total: np.ndarray, input_carryover: np.ndarray, input_immediate: np.ndarray, response_total: np.ndarray, response_carryover: np.ndarray, response_immediate: np.ndarray, usecase: str, plot: plt.Figure)`
* `metric_name`: A string representing the name of the metric.
* `date`: A pandas Series representing the date range.
* `input_total`: A numpy array representing the total input values.
* `input_carryover`: A numpy array representing the carryover input values.
* `input_immediate`: A numpy array representing the immediate input values.
* `response_total`: A numpy array representing the total response values.
* `response_carryover`: A numpy array representing the carryover response values.
* `response_immediate`: A numpy array representing the immediate response values.
* `usecase`: A string representing the use case.
* `plot`: A matplotlib Figure object representing the plot of the response.
### USAGE
* Use this constructor to create an instance of `ResponseOutput` with the specified attributes related to metric response.
### IMPL
* Initializes the data class with attributes to store details of the response calculation.

---

# CLASS
## UseCase
* This class is an ENUM.
* Represents different types of use cases for response calculation.
* Enum fields:
  - `ALL_HISTORICAL_VEC`: Represents all historical vectors.
  - `SELECTED_HISTORICAL_VEC`: Represents selected historical vectors.
  - `TOTAL_METRIC_DEFAULT_RANGE`: Represents total metric with default range.
  - `TOTAL_METRIC_SELECTED_RANGE`: Represents total metric with selected range.
  - `UNIT_METRIC_DEFAULT_LAST_N`: Represents unit metric with default last N.
  - `UNIT_METRIC_SELECTED_DATES`: Represents unit metric with selected dates.

---

# CLASS
## ResponseCurveCalculator
* This class is responsible for calculating response curves based on media mix modeling (MMM) data, model outputs, and hyperparameters.
* Part of a larger module that handles data transformations and response calculations.

# CONSTRUCTORS
## ResponseCurveCalculator `(mmm_data: MMMData, model_outputs: ModelOutputs, hyperparameter: Hyperparameters)`
* `mmm_data`: An instance of `MMMData` containing media mix modeling data.
* `model_outputs`: An instance of `ModelOutputs` containing model output data.
* `hyperparameter`: An instance of `Hyperparameters` containing model hyperparameters.
### USAGE
* Use this constructor to create an instance of `ResponseCurveCalculator` with the necessary data and hyperparameters to compute response curves.
### IMPL
* Initializes the class with instances of `MMMData`, `ModelOutputs`, and `Hyperparameters`.
* Initializes a transformation object based on the provided `MMMData`.

# METHODS
## `calculate_response(select_model: str, metric_name: str, metric_value: Optional[Union[float, list[float]]] = None, date_range: Optional[str] = None, quiet: bool = False, dt_hyppar: pd.DataFrame = pd.DataFrame(), dt_coef: pd.DataFrame = pd.DataFrame()) -> ResponseOutput`
### USAGE
* `select_model`: A string specifying which model to use for the calculation.
* `metric_name`: A string specifying the name of the metric to analyze.
* `metric_value`: Optional argument specifying a float or list of floats for metric values.
* `date_range`: Optional string specifying the date range to consider.
* `quiet`: Boolean flag to suppress output.
* `dt_hyppar`: A pandas DataFrame containing hyperparameter values.
* `dt_coef`: A pandas DataFrame containing coefficient values.
* Returns an instance of `ResponseOutput`.
### IMPL
* Determines the use case based on provided metric value and date range.
* Checks the type of metric (spend, exposure, organic) using `_check_metric_type`.
* Processes date ranges and metric values using helper functions `_check_metric_dates` and `_check_metric_value`.
* Transforms exposure to spend if necessary using `_transform_exposure_to_spend`.
* Applies adstock transformation using `_get_channel_hyperparams` and `transform_adstock`.
* Applies saturation transformation using `_get_saturation_params` and `saturation_hill`.
* Calculates final response values using model coefficients.
* Returns a `ResponseOutput` with calculated response data.

## `_which_usecase(metric_value: Optional[Union[float, list[float]]], date_range: Optional[str]) -> UseCase`
### USAGE
* Determines the specific use case based on metric value and date range.
* Returns an appropriate `UseCase` enum member.
### IMPL
* Uses conditional logic to determine the use case based on the presence and type of `metric_value` and `date_range`.

## `_check_metric_type(metric_name: str) -> Literal["spend", "exposure", "organic"]`
### USAGE
* Determines the type of metric based on its name.
* Returns one of the literals: "spend", "exposure", or "organic".
### IMPL
* Checks the metric name against predefined lists in `MMMData` to identify its type.
* Raises a `ValueError` if the metric type is unknown.

## `_check_metric_dates(date_range: Optional[str], all_dates: pd.Series, quiet: bool) -> MetricDateInfo`
### USAGE
* Processes the given date range and returns the corresponding metric location and updated date range.
* Returns an instance of `MetricDateInfo`.
### IMPL
* Uses conditional logic to handle different date range formats.
* Updates the date range based on the provided or default rolling window values.
* Prints the updated date range if `quiet` is False.

## `_check_metric_value(metric_value: Optional[Union[float, list[float]]], metric_name: str, all_values: pd.Series, metric_loc: Union[slice, pd.Series]) -> MetricValueInfo`
### USAGE
* Processes the given metric value and returns updated metric values.
* Returns an instance of `MetricValueInfo`.
### IMPL
* Updates metric values based on the provided input or uses default values.
* Raises a `ValueError` if the length of `metric_value` does not match the selected date range.

## `_transform_exposure_to_spend(metric_name: str, metric_value_updated: np.ndarray, all_values_updated: pd.Series, metric_loc: Union[slice, pd.Series]) -> pd.Series`
### USAGE
* Transforms exposure values to spend values based on predefined models.
* Returns a pandas Series of updated values.
### IMPL
* Determines the appropriate model (non-linear or linear) based on R-squared values.
* Calculates and updates spend values using model parameters.

## `_get_spend_name(metric_name: str) -> str`
### USAGE
* Retrieves the spend name associated with a given metric name.
* Returns the corresponding spend name as a string.
### IMPL
* Uses the index of the metric name in `paid_media_vars` to find the corresponding spend name.

## `_get_channel_hyperparams(select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> ChannelHyperparameters`
### USAGE
* Retrieves hyperparameters for the specified channel and model.
* Returns an instance of `ChannelHyperparameters`.
### IMPL
* Extracts hyperparameter values from `dt_hyppar` based on the selected model and adstock type.

## `_get_saturation_params(select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> ChannelHyperparameters`
### USAGE
* Retrieves saturation parameters for the specified channel and model.
* Returns an instance of `ChannelHyperparameters`.
### IMPL
* Extracts saturation parameter values from `dt_hyppar` based on the selected model.

## `_create_response_plot(m_adstockedRW: np.ndarray, m_response: np.ndarray, input_total: np.ndarray, response_total: np.ndarray, input_carryover: np.ndarray, response_carryover: np.ndarray, input_immediate: np.ndarray, response_immediate: np.ndarray, metric_name: str, metric_type: Literal["spend", "exposure", "organic"], date_range_updated: pd.Series) -> plt.Figure`
### USAGE
* Creates a plot visualizing the response curve based on inputs and responses.
* Returns a matplotlib Figure object.
### IMPL
* Plots the response curve and scatter plots for total, carryover, and immediate responses.
* Configures plot labels, titles, and legends based on metric type and data.
* Optionally includes a detailed subtitle if the input total is unique.
* Adds a text description of the response period at the bottom of the plot.