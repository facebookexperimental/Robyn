import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any
from robyn.modeling.entities.modeloutputs import ModelOutputs


def plot_moo_distrb(model_outputs: ModelOutputs):
    """
    Plot the MOO distribution plot.
    """
    data = pd.DataFrame([{"nrmse": trial.nrmse, "decomp_rssd": trial.decomp_rssd} for trial in model_outputs.trials])

    plt.figure(figsize=(10, 6))
    plt.scatter(data["nrmse"], data["decomp_rssd"], alpha=0.6)
    plt.xlabel("NRMSE")
    plt.ylabel("Decomp RSSD")
    plt.title("MOO Distribution Plot")
    plt.grid(True)
    plt.show()

    print(f"Number of datapoints plotted: {len(data)}")


def plot_moo_cloud(model_outputs: ModelOutputs):
    """
    Plot the MOO cloud plot.
    """
    # Assuming the data is in the trials of model_outputs
    data = pd.DataFrame([vars(trial) for trial in model_outputs.trials])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(data["nrmse"], data["decomp_rssd"], data["rsq_test"], alpha=0.6)
    ax.set_xlabel("NRMSE")
    ax.set_ylabel("Decomp RSSD")
    ax.set_zlabel("R-squared (Test)")
    ax.set_title("MOO Cloud Plot")
    plt.show()


def plot_ts_validation(model_outputs: ModelOutputs):
    """
    Plot the time series validation plot.
    """
    # Assuming we have actual and predicted values in the trials
    # You might need to adjust this based on your actual data structure
    best_trial = min(model_outputs.trials, key=lambda x: x.nrmse)

    plt.figure(figsize=(12, 6))
    plt.plot(best_trial.x_decomp_agg.index, best_trial.x_decomp_agg["actual"], label="Actual")
    plt.plot(best_trial.x_decomp_agg.index, best_trial.x_decomp_agg["predicted"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Time Series Validation Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_allocator(allocation_data: pd.DataFrame):
    """
    Plot the allocator graph.
    """
    plt.figure(figsize=(10, 6))
    allocation_data.plot(kind="bar", stacked=True)
    plt.xlabel("Scenario")
    plt.ylabel("Allocation")
    plt.title("Media Allocation")
    plt.legend(title="Media Channels", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_outputgraphs(model_outputs: ModelOutputs, graph_type: str, allocation_data: pd.DataFrame = None):
    """
    Plot the output graphs based on the graph type.

    Args:
        model_outputs: ModelOutputs object containing the model output data.
        graph_type: The type of graph to plot.
        allocation_data: Optional. DataFrame containing allocation data for the allocator plot.
    """
    if graph_type == "moo_distrb_plot":
        plot_moo_distrb(model_outputs)
    elif graph_type == "moo_cloud_plot":
        plot_moo_cloud(model_outputs)
    elif graph_type == "ts_validation_plot":
        plot_ts_validation(model_outputs)
    elif graph_type == "allocator":
        if allocation_data is not None:
            plot_allocator(allocation_data)
        else:
            print("Allocation data is required for the allocator plot.")
    else:
        print(f"Graph type '{graph_type}' is not supported.")


# Example usage
if __name__ == "__main__":
    from data_mapper import load_data_from_json, import_data

    # Load the data
    loaded_data = load_data_from_json(
        "/Users/yijuilee/project_robyn/robynpy_interfaces/Robyn/python/src/tutorials/data/R/exported_data.json"
    )
    imported_data = import_data(loaded_data)

    # Get the model_outputs
    model_outputs = imported_data["model_outputs"]

    # Plot the graphs
    plot_outputgraphs(model_outputs, "moo_distrb_plot")
    plot_outputgraphs(model_outputs, "moo_cloud_plot")
    plot_outputgraphs(model_outputs, "ts_validation_plot")

    # For the allocator plot, you'll need to prepare the allocation data
    # This is just an example, you'll need to adjust it based on your actual data
    allocation_data = pd.DataFrame(
        {"TV": [0.3, 0.4, 0.2], "Radio": [0.2, 0.3, 0.4], "Online": [0.5, 0.3, 0.4]},
        index=["Scenario 1", "Scenario 2", "Scenario 3"],
    )

    plot_outputgraphs(model_outputs, "allocator", allocation_data)
