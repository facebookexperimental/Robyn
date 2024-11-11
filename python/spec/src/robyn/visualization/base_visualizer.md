# CLASS
## BaseVisualizer
* This class serves as the base for all Robyn visualization components, offering common plotting functionalities and styling options.
* It leverages the `matplotlib` and `seaborn` libraries to facilitate plotting and provides methods for configuring plot styles, saving plots, and managing resources effectively.

# CONSTRUCTORS
## `__init__(style: str = "bmh")`
* Initializes the `BaseVisualizer` with a specified or default plot style, along with other common plot settings.
* **Parameters:**
  * `style`: A string that specifies the `matplotlib` style to be applied. Defaults to `"bmh"`.

### USAGE
* Use this constructor to instantiate a `BaseVisualizer` object with a specific plot style.
* Example:
  python
  visualizer = BaseVisualizer(style="ggplot")
  

### IMPL
* Assigns the `style` attribute with the given `style` parameter or defaults to `"bmh"`.
* Sets `default_figsize` to `(12, 8)` for default plot dimensions.
* Creates a `colors` dictionary to store color schemes for different visual elements such as 'primary', 'secondary', 'positive', 'negative', and 'neutral'.
* Establishes a `font_sizes` dictionary to define font sizes for various text elements including 'title', 'label', 'tick', and 'annotation'.
* Initializes `current_figure` and `current_axes` as `None` to keep track of the current plot being worked on.
* Invokes the `_setup_plot_style()` method to apply the chosen plot style configurations.

# METHODS
## `_setup_plot_style() -> None`
### USAGE
* This method requires no parameters.
* It configures the default plotting style settings for the visualizer based on the `style` attribute.

### IMPL
* Calls `plt.style.use(self.style)` to set the plot style as defined in the `style` attribute.
* Utilizes `plt.rcParams.update()` to configure default plot parameters such as figure size, grid visibility, spine visibility, and font size for labels.

## `save_plot(filename: Union[str, Path], dpi: int = 300, cleanup: bool = True) -> None`
### USAGE
* Saves the current plot to a specified file path.
* **Parameters:**
  * `filename`: A string or `Path` object specifying where the plot should be saved.
  * `dpi`: An integer representing the resolution of the saved plot. Defaults to 300.
  * `cleanup`: A boolean indicator that determines whether the plot should be closed after saving. Defaults to `True`.

### IMPL
* Checks if `self.current_figure` is `None` and raises a `ValueError` if there is no figure to save.
* Converts `filename` into a `Path` object and ensures that the parent directory exists, creating it if necessary.
* Saves the current figure using `self.current_figure.savefig()`, applying `dpi` resolution and `bbox_inches='tight'` to ensure layout is compact.
* If `cleanup` is `True`, it invokes the `cleanup()` method to release resources.

## `cleanup() -> None`
### USAGE
* Closes the current plot, freeing up `matplotlib` resources to manage memory efficiently.

### IMPL
* Verifies that `self.current_figure` is not `None` before proceeding.
* Closes the current figure using `plt.close(self.current_figure)`.
* Resets `self.current_figure` and `self.current_axes` to `None` to clear plot tracking.