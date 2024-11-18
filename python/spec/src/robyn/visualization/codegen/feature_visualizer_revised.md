# CLASS
## FeaturePlotter
* Extends the `BaseVisualizer` class from `robyn.visualization.codegen.base_visualizer_revised`.
* Designed to create various plots related to feature engineering in the Robyn framework.
* Utilizes `matplotlib` and `seaborn` for plotting.
* Logs information and debug messages using Python's `logging` module.
* Provides visual insights into different aspects of feature engineering such as adstock, saturation, spend-exposure relationships, feature importance, and response curves.

# CONSTRUCTORS
## `__init__(mmm_data: MMMData, hyperparameters: Hyperparameters)`
* Initializes the `FeaturePlotter` class with the necessary data and hyperparameters.
* **Parameters:**
  * `mmm_data`: The input data and specifications for the Marketing Mix Modeling (MMM).
  * `hyperparameters`: The hyperparameters for the model.

### USAGE
* This constructor is called when creating an instance of `FeaturePlotter`.
* It requires `MMMData` and `Hyperparameters` objects.

### IMPL
* Initializes the `mmm_data` attribute with the `MMMData` instance.
* Initializes the `hyperparameters` attribute with the `Hyperparameters` instance.
* Logs an informational message indicating the initialization of the `FeaturePlotter`.
* Logs detailed debug messages containing the provided `mmm_data` and `hyperparameters`.

# METHODS
## `plot_adstock(channel: str) -> plt.Figure`
### USAGE
* `channel`: The name of the channel to plot the adstock transformation for.
* This method returns a matplotlib Figure object containing the adstock plot.

### IMPL
* Logs an informational message indicating the start of adstock plot generation for the specified channel.
* Logs a debug message for processing the adstock transformation for the channel.
* Contains a placeholder for the actual implementation of the adstock plot.
* Logs a warning indicating that the method is not yet implemented.
* Catches exceptions during plot generation, logs an error message, and re-raises the exception.

## `plot_saturation(channel: str) -> plt.Figure`
### USAGE
* `channel`: The name of the channel to plot the saturation curve transformation for.
* This method returns a matplotlib Figure object containing the saturation curves plot.

### IMPL
* Logs an informational message indicating the start of saturation plot generation for the specified channel.
* Logs a debug message for processing the saturation curve transformation for the channel.
* Contains a placeholder for the actual implementation of the saturation plot.
* Logs a warning indicating that the method is not yet implemented.
* Catches exceptions during plot generation, logs an error message, and re-raises the exception.

## `plot_spend_exposure(featurized_data: FeaturizedMMMData, channel: str) -> plt.Figure`
### USAGE
* `featurized_data`: The featurized data after feature engineering.
* `channel`: The name of the channel to plot spend-exposure data for.
* This method returns a matplotlib Figure object containing the spend-exposure plot.

### IMPL
* Logs an informational message indicating the start of spend-exposure plot generation for the specified channel.
* Logs a debug message about processing the specified `featurized_data`.
* Searches for the channel's results in `featurized_data.modNLS["results"]` and logs an error if not found, raising a `ValueError`.
* Retrieves the plot data for the channel from `featurized_data.modNLS["plots"]`, logs an error if not found, and raises a `ValueError`.
* Logs debug messages with retrieved model results and plot data shape.
* Creates a figure and axis using `matplotlib.pyplot.subplots`.
* Plots actual data as a scatter plot and the fitted data as a line plot using `seaborn`.
* Sets labels for the axes and a title for the plot.
* Adds model information (type, R-squared, Vmax, Km for NLS model, or coefficient for linear model) as text on the plot.
* Adds a legend and adjusts the layout for the plot.
* Returns the generated matplotlib `Figure` object.
* Catches exceptions, logs detailed error information, and re-raises the exception.

## `plot_feature_importance(feature_importance: Dict[str, float]) -> plt.Figure`
### USAGE
* `feature_importance`: A dictionary where keys are feature names and values are their respective importance scores.
* This method returns a matplotlib Figure object containing the feature importance plot.
### IMPL
* Logs an informational message indicating the start of feature importance plot generation.
* Logs a debug message with the provided `feature_importance` data.
* Contains a placeholder for the actual implementation of the feature importance plot.
* Logs a warning indicating that the method is not yet implemented.
* Catches exceptions during plot generation, logs an error message, and re-raises the exception.
## `plot_response_curves(featurized_data: FeaturizedMMMData) -> Dict[str, plt.Figure]`
### USAGE
* `featurized_data`: The featurized data after feature engineering.
* This method returns a dictionary mapping channel names to their respective response curve plots.
### IMPL
* Logs an informational message indicating the start of response curve generation.
* Logs a debug message about processing the provided `featurized_data`.
* Contains a placeholder for the rest of the method implementation.
* Logs a warning indicating that the method is not fully implemented yet.
* Catches exceptions, logs an error message, and re-raises the exception.