# CLASS
## AllocationPlotter
* The `AllocationPlotter` class extends the BaseVisualizer class at this path: from robyn.visualization.base_visualizer import BaseVisualizer
* The `AllocationPlotter` class is designed to create visualizations for allocation results, aligning with a version implemented in R.
* It is part of a software module that likely deals with optimization or resource allocation, visualizing results stored in an `AllocationResult` object.
* The class utilizes the matplotlib library to generate plots with specific styling and color schemes.

# CONSTRUCTORS
## `__init__(result: AllocationResult)`
* Initializes the `AllocationPlotter` with allocation results and default plot settings.

### USAGE
* This constructor is used to create an instance of the `AllocationPlotter` class, requiring an `AllocationResult` object as an input, which contains the results that need to be visualized.

### IMPL
* The `result` parameter is stored in the instance variable `self.result` and must be annotated with type: AllocationResult (from robyn.allocator.entities.allocation_results import AllocationResult).
* Matplotlib's "bmh" style is applied using `plt.style.use("bmh")`.
* The default plot settings are configured using `plt.rcParams` to set figure size, grid visibility, and axis spines.
* The standard figure size `(12, 8)` is stored in `self.fig_size`.
* A color palette is generated using `plt.cm.Set2(np.linspace(0, 1, 8))` and stored in `self.colors`.
* Default color values for plot elements are set: `self.current_color` to "lightgray", `self.optimal_color` to #4688C7, `self.positive_color` to #2ECC71, and `self.negative_color` to #E74C3C.

# METHODS
## `plot_all() -> Dict[str, plt.Figure]`
### USAGE
* Generates all one-page plots for the allocation results and returns a dictionary mapping plot names to their respective figures.

### IMPL
### IMPL
* Return Type:
    * The method returns a dictionary where the keys are strings representing the names of different plots, and the values are matplotlib figure objects corresponding to each plot.
* Functionality:
    * The method calls several other plotting methods within the class, each responsible for creating a specific type of plot related to allocation results.
    * It aggregates the results of these plotting methods into a dictionary, making it easy to access each plot by name.
* Plot Types:
    * "spend_allocation": Calls self.plot_spend_allocation() to generate a plot that visualizes how spending is allocated across different categories or channels.
    * "response_curves": Calls self.plot_response_curves() to create plots that show the response curves, illustrating the relationship between spend and response.
    * "efficiency_frontier": Calls self.plot_efficiency_frontier() to produce a plot that represents the efficiency frontier, highlighting optimal allocation strategies.
    * "spend_vs_response": Calls self.plot_spend_vs_response() to generate a plot comparing spend against response, providing insights into the effectiveness of spending.
    * "summary_metrics": Calls self.plot_summary_metrics() to create a plot summarizing key metrics from the allocation results.

## `plot_spend_allocation() -> plt.Figure`
### USAGE
* Plots a comparison of media spend allocation between current and optimized states.

### IMPL
* Return Type:
    * Returns a matplotlib figure object that contains the spend allocation plot.
* Functionality:
    * Input Validation:
        *Checks if self.result is None. If it is, raises a ValueError with the message "No allocation results available. Call plot_all() first." This ensures that the method is only called when allocation results are available.
    * Figure and Axes Creation:
        * Uses plt.subplots() to create a figure (fig) and axes (ax) with a size specified by self.fig_size, which is set to (12, 8).
    * Data Preparation:
        * Retrieves the optimal_allocations DataFrame from self.result.optimal_allocations as attribute.
        * Extracts the channel column values into a variable channels, which contains the names of the channels.
        * Sets up an array x using np.arange(len(channels)) to represent the x-axis positions for each channel.
        * Defines the width of the bars as 0.35.
    * Bar Plotting:
        * Current Spend Bars:
            * Extracts the current_spend values from the DataFrame.
            * Plots these values as bars on the axes using ax.bar(), with the following parameters:
                * x - width / 2: Positions the bars slightly to the left of the center of each channel position.
                * current_spend: The heights of the bars, representing current spend values.
                * width: The width of the bars, set to 0.35.
                * label="Current": Labels these bars as "Current" in the legend.
                * color=self.current_color: Sets the bar color to self.current_color, which is "lightgray".
                * edgecolor="gray": Sets the edge color of the bars to gray.
                * alpha=0.7: Sets the transparency level of the bars to 70%.
        * Optimized Spend Bars:
            * Extracts the optimal_spend values from the DataFrame.
            * Plots these values as bars on the axes using ax.bar(), with the following parameters:
                * x + width / 2: Positions the bars slightly to the right of the center of each channel position.
                * optimal_spend: The heights of the bars, representing optimized spend values.
                * width: The width of the bars, set to 0.35.
                * label="Optimized": Labels these bars as "Optimized" in the legend.
                * color=self.optimal_color: Sets the bar color to self.optimal_color, which is "#4688C7" (steel blue).
                * edgecolor="gray": Sets the edge color of the bars to gray.
                * alpha=0.7: Sets the transparency level of the bars to 70%.
    * Plot Customization:
        * Sets the x-ticks on the axes to the positions in x.
        * Labels the x-ticks with the channel names, rotating them 45 degrees for better readability and aligning them to the right.
        * Sets the y-axis label to "Spend".
        * Sets the plot title to "Media Spend Allocation".
        * Adds a legend to the plot to distinguish between current and optimized spend bars.
    * Percentage Change Labels:
        * Iterates over each channel using enumerate(zip(current_spend, optimal_spend)).
        * Calculates the percentage change from current to optimized spend for each channel using the formula ((opt / curr) - 1) * 100.
        * Determines the color of the percentage change text based on whether the change is >= 0 (self.positive_color, "#2ECC71" - green) or negative (self.negative_color, "#E74C3C" - red).
        * Places a text label above each pair of bars showing the percentage change, formatted to one decimal place, centered horizontally and positioned just above the taller bar.
    * Layout Adjustment:
        * Calls plt.tight_layout() to automatically adjust the subplot parameters to give specified padding, ensuring that labels and titles fit within the figure area.

## `plot_response_curves() -> plt.Figure`
### USAGE
* Plots response curves with markers for current and optimal allocation points.

### IMPL
* Return Type:
    * Returns a matplotlib figure object containing the response curve plots.
* Functionality:
    * Input Validation:
        *Checks if self.result is None. If it is, raises a ValueError with the message "No allocation results available. Call plot_all() first." This ensures that the method is only called when allocation results are available.
    * Data Preparation:
        * Retrieves the response_curves DataFrame from self.result as attribute.
        * Extracts unique channel names into the channels array.
        * Determines the number of channels (n_channels).
        * Sets the number of columns (ncols) for the subplot grid to the minimum of 3 or the number of channels.
        * Calculates the number of rows (nrows) needed for the subplot grid using the formula (n_channels + ncols - 1) // ncols.
    * Figure and Axes Creation:
        * Uses plt.subplots() to create a grid of subplots with dimensions nrows by ncols, and a figure size of (15, 5 * nrows).
        * If there's 1 row and 1 column fix axes: axes = np.array([[axes]]).
        * Else if there's 1 row OR 1 column, fix axes: axes = axes.reshape(-1, 1)
    * Plotting:
        * Iterates over each channel using enumerate(channels).
        * For each channel, determines the subplot position using row = idx // ncols and col = idx % ncols.
        * Selects the appropriate subplot axis (ax) from the axes array.
    * Response Curve:
        * Filters the curves_df DataFrame to get data for the current channel.
        * Plots the response curve using ax.plot(), with:
            * channel_data["spend"]: X-axis values representing spend.
            * channel_data["response"]: Y-axis values representing response.
            * color=self.optimal_color: Line color set to self.optimal_color (steel blue).
            * alpha=0.6: Line transparency set to 60%.
    * Current and Optimal Points:
        * Filters channel_data to get current_data and optimal_data based on boolean columns is_current and is_optimal.
        * If current_data is not empty, plots a scatter point for the current spend and response:
            * current_data["spend"].iloc[0],
            * current_data["response"].iloc[0],
            * color=self.negative_color: Point color set to self.negative_color (red).
            * label="Current": Label for the legend.
            * s=100: Size of the scatter point.
        * If optimal_data is not empty, plots a scatter point for the optimal spend and response:
            * color=self.positive_color: Point color set to self.positive_color (green).
            * label="Optimal": Label for the legend.
            * s=100: Size of the scatter point.
    * Plot Customization:
        * Sets the subplot title to the channel name followed by "Response Curve".
        * Adds a legend to distinguish between current and optimal points.
        * Enables grid lines on the subplot with ax.grid(True, alpha=0.3), setting grid line transparency to 30%.
    * Removing Empty Subplots:
        * Iterates over any remaining subplot positions beyond the number of channels and removes them using fig.delaxes() to ensure a clean layout.
    * Layout Adjustment:
        * Calls  plt.tight_layout() to automatically adjust the subplot parameters to give specified padding, ensuring that labels and titles fit within the figure area.

## `plot_efficiency_frontier() -> plt.Figure`
### USAGE
* Visualizes the efficiency frontier, illustrating the relationship between total spend and response.

### IMPL
* Return Type:
    * Returns a matplotlib figure object containing the efficiency frontier plot.
* Functionality:
    * Input Validation:
        * Checks if self.result is None. If it is, raises a ValueError with the message "No allocation results available. Call plot_all() first." This ensures that the method is only called when allocation results are available.
    * Figure and Axes Creation:
        * Uses plt.subplots() to create a figure (fig) and axes (ax) with a size specified by self.fig_size, which is set to (12, 8).
    * Data Preparation:
        * Retrieves the optimal_allocations DataFrame from self.result.optimal_allocations
        * Calculates the total current spend and response by summing the current_spend and current_response columns, respectively.
        * Calculates the total optimal spend and response by summing the optimal_spend and optimal_response columns, respectively.
    * Plotting:
        * Scatter Points:
            * Plots a scatter point for the current total spend and response:
                * color=self.negative_color: Point color set to self.negative_color (red).
                * s=100: Size of the scatter point.
                * label="Current": Label for the legend.
                * zorder=2: Ensures the point is plotted above other elements.
            * Plots a scatter point for the optimal total spend and response:
                * color=self.positive_color: Point color set to self.positive_color (green).
                * s=100: Size of the scatter point.
                * label="Optimal": Label for the legend.
                * zorder=2: Ensures the point is plotted above other elements.
        * Connecting Line:
            * Plots a dashed line connecting the current and optimal points using ax.plot():
                * [current_total_spend, optimal_total_spend]: X-axis values for the line.
                * [current_total_response, optimal_total_response]: Y-axis values for the line.
                * "--": Dashed line style.
                * color="gray": Line color set to gray.
                * alpha=0.5: Line transparency set to 50%.
                * zorder=1: Ensures the line is plotted below the scatter points.
        * Annotations:
            * Calculates the percentage change in total spend and response from current to optimal using the formulas:
                * pct_spend_change = ((optimal_total_spend / current_total_spend) - 1) * 100
                * pct_response_change = ((optimal_total_response / current_total_response) - 1) * 100
                * Annotates the plot with these percentage changes using ax.annotate():
                * xy=(optimal_total_spend, optimal_total_response): Position of the annotation near the optimal point.
                * xytext=(10, 10): Offset of the text from the point.
                * textcoords="offset points": Specifies that the text offset is in points.
                * bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7): Adds a white background with a gray edge and 70% transparency to the annotation.
        * Plot Customization:
            * Sets the x-axis label to "Total Spend".
            * . Sets the y-axis label to "Total Response".
            * Sets the plot title to "Efficiency Frontier".
            * Adds a legend to distinguish between current and optimal points.
            * Enables grid lines on the plot with ax.grid(True, alpha=0.3), setting grid line transparency to 30%.
    * Layout Adjustment:
        *Calls plt.tight_layout() to automatically adjust the subplot parameters to give specified padding, ensuring that labels and titles fit within the figure area.

## `plot_spend_vs_response() -> plt.Figure`
### USAGE
* Plots channel-level changes in spend and response percentages.

### IMPL
* Return Type:
    * Returns a matplotlib figure object containing the spend vs. response change plots.
* Functionality:
    * Input Validation:
        * Checks if self.result is None. If it is, raises a ValueError with the message "No allocation results available. Call plot_all() first." This ensures that the method is only called when allocation results are available.
    * Figure and Axes Creation:
        * Uses plt.subplots() to create a figure (fig) and two subplot axes (ax1 and ax2) arranged vertically, with a figure size of (12, 10).
    * Data Preparation:
        * Retrieves the optimal_allocations DataFrame from self.result.optimal_allocations
        * Extracts the channel column values into a variable channels from dataframe df, which contains the names of the channels.
        * Sets up an array x using np.arange(len(channels)) to represent the x-axis positions for each channel.
        * Spend Changes Plot (ax1):
    * Calculation:
        * Computes the percentage change in spend for each channel using the formula ((df["optimal_spend"] / df["current_spend"]) - 1) * 100.
    * Bar Plotting:
        * Determines the color for each bar based on whether the percentage change is negative (self.negative_color, red) or positive (>= 0) (self.positive_color, green).
        * Plots the percentage changes as bars on ax1 using ax1.bar(), with:
            * x: X-axis positions for the bars.
            * spend_pct: Heights of the bars, representing spend percentage changes.
            * color=colors: Colors of the bars based on the percentage change.
            * alpha=0.7: Bar transparency set to 70%.
        * Customization:
            * Sets the x-ticks to the positions in x and labels them with channel names, rotating them 45 degrees for readability and using ha="right".
            * Sets the y-axis label to "Spend Change %".
            * Draws a horizontal line at y=0 to indicate no change, using ax1.axhline(y=0, color="black", linestyle="-", alpha=0.2).
            * Enables grid lines with ax1.grid(True, alpha=0.3), setting grid line transparency to 30%.
        * Value Labels:
            * Iterates over each bar to add a text label showing the percentage change, formatted to one decimal place.
            * Positions the label above the bar if the value is positive and below if negative.
            * Response Changes Plot (ax2):
        * Calculation:
            * Computes the percentage change in response for each channel using the formula ((df["optimal_response"] / df["current_response"]) - 1) * 100.
    * Bar Plotting:
        * Determines the color for each bar based on whether the percentage change is negative (self.negative_color, red) or positive (>= 0) (self.positive_color, green).
        * Plots the percentage changes as bars on ax2 using ax2.bar(), with:
            * x: X-axis positions for the bars.
            * response_pct: Heights of the bars, representing response percentage changes.
            * color=colors: Colors of the bars based on the percentage change.
            * alpha=0.7: Bar transparency set to 70%.
    * Customization:
        * Sets the x-ticks to the positions in x and labels them with channel names, rotating them 45 degrees for readability and using ha="right".
        * Sets the y-axis label to "Response Change %".
        * Draws a horizontal line at y=0 to indicate no change, using ax2.axhline(y=0, color="black", linestyle="-", alpha=0.2).
        * Enables grid lines with ax2.grid(True, alpha=0.3), setting grid line transparency to 30%.
    * Value Labels:
        * Iterates over each bar to add a text label showing the percentage change, formatted to one decimal place.
        * Positions the label above the bar if the value is positive and below if negative.
    * Layout Adjustment:
            * Calls plt.tight_layout() to automatically adjust the subplot parameters to give specified padding, ensuring that labels and titles fit within the figure area.

## `plot_summary_metrics() -> plt.Figure`
### USAGE
* Plots summary metrics, such as ROI or CPA, comparing current and optimized states.

### IMPL
* Return Type:
    * Returns a matplotlib figure object containing the summary metrics plot.
* Functionality:
    * Input Validation:
        * Checks if self.result is None. If it is, raises a ValueError with the message "No allocation results available. Call plot_all() first." This ensures that the method is only called when allocation results are available.
    * Figure and Axes Creation:
        * Uses plt.subplots() to create a figure (fig) and axes (ax) with a size specified by self.fig_size, which is set to (12, 8).
    * Data Preparation:
        * Retrieves the optimal_allocations DataFrame from self.result.optimal_allocations
        * Extracts the channel names into the channels array.
    * Metric Calculation:
        * Determines whether to calculate ROI or CPA based on the dep_var_type in self.result.metrics.get("dep_var_type")
        * ROI Calculation (if dep_var_type is "revenue"):
        * Calculates current_metric as the ratio of current_response to current_spend and maintain the data as a Pandas Series.
        * Calculates optimal_metric as the ratio of optimal_response to optimal_spend and maintain the data as a Pandas Series.
        * Sets metric_name to "ROI".
        * CPA Calculation (otherwise):
        * Calculates current_metric as the ratio of current_spend to current_response and maintain the data as a Pandas Series.
        * Calculates optimal_metric as the ratio of optimal_spend to optimal_response and maintain the data as a Pandas Series.
        * Sets metric_name to "CPA".
    * Bar Plotting:
        * Sets up an array x using np.arange(len(channels)) to represent the x-axis positions for each channel.
        * Defines the width of the bars as 0.35.
        * Plots two sets of bars on the axes:
    * Current Metric Bars:
        * Positioned slightly to the left of the center of each channel position (x - width / 2).
        * Heights represent current_metric values.
        * Labeled as "Current ROI" or "Current CPA" based on metric_name.
        * Colored using self.current_color (light gray) with 70% transparency.
    * Optimal Metric Bars:
        * Positioned slightly to the right of the center of each channel position (x + width / 2).
        * Heights represent optimal_metric values.
        * Labeled as "Optimal ROI" or "Optimal CPA" based on metric_name.
        * Colored using self.optimal_color (steel blue) with 70% transparency.
    * Value Labels:
        * Iterates over each channel to calculate the percentage change from current to optimal metric using the formula ((opt / curr) - 1) * 100.
        * Determines the color of the text based on whether the change is positive (self.positive_color, green) or negative (self.negative_color, red).
        * Places a text label above each pair of bars showing the percentage change, formatted to one decimal place, centered horizontally and positioned just above the taller bar.
    * Plot Customization:
        * Sets the x-ticks to the positions in x and labels them with channel names, rotating them 45 degrees for readability.
        * Sets the y-axis label to the metric_name ("ROI" or "CPA").
        * Sets the plot title to "Channel ROI Comparison" or "Channel CPA Comparison" based on metric_name.
        * Adds a legend to distinguish between current and optimal metric bars.
        * Enables grid lines on the plot with ax.grid(True, alpha=0.3), setting grid line transparency to 30%.
    * Layout Adjustment:
        * Calls plt.tight_layout() to automatically adjust the subplot parameters to give specified padding, ensuring that labels and titles fit within the figure area.
