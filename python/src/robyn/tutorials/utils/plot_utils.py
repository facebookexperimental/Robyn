import io
import binascii
from PIL import Image
from IPython.display import display
import warnings


def plot_outputgraphs(model_outputs, graph_type, max_size=(1000, 1500)):
    """
    Plots the output graphs for the given model outputs.
    Args:
        model_outputs: ModelOutputs object containing the output data for the graphs.
        graph_type: The type of graph to plot.
        max_size: Optional. The maximum size of the rendered images. Defaults to (1000, 1500).
    Returns:
        None. The function renders the plots and displays them using the `display()` function from IPython.
    """
    convergence = model_outputs.convergence

    if graph_type in ["moo_distrb_plot", "moo_cloud_plot"]:
        image_data = binascii.unhexlify("".join(getattr(convergence, graph_type)))
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        display(image)
    elif graph_type == "ts_validation_plot":
        if hasattr(model_outputs, "ts_validation_plot"):
            image_data = binascii.unhexlify("".join(model_outputs.ts_validation_plot))
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display(image)
        else:
            warnings.warn("ts_validation_plot not available in model outputs")
    else:
        warnings.warn(f"Graph type '{graph_type}' is not supported")


# Example usage
if __name__ == "__main__":
    from data_mapper import load_data_from_json, import_data

    # Load the data
    loaded_data = load_data_from_json("path/to/your/exported_data.json")
    imported_data = import_data(loaded_data)

    # Get the model_outputs
    model_outputs = imported_data["model_outputs"]

    # Plot the graphs
    plot_outputgraphs(model_outputs, "moo_distrb_plot")
    plot_outputgraphs(model_outputs, "moo_cloud_plot")
    plot_outputgraphs(model_outputs, "ts_validation_plot")
