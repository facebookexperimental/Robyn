import binascii
import io
import json
import os
import warnings
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs


def asSerialisedFeather(modelData):
    """
    Serializes given model data to a Feather-formatted hex string.
    """
    modelDataFeather = io.BytesIO()
    pd.DataFrame(modelData).to_feather(modelDataFeather)
    modelDataFeather.seek(0)
    modelDataBinary = modelDataFeather.read()
    return binascii.hexlify(modelDataBinary).decode()


def pandas_builder(jsondata):
    """
    Builds a pandas DataFrame from JSON data.
    """
    return pd.DataFrame(jsondata)


def robyn_api(argument, payload=0, api="http://127.0.0.1:9999/{}"):
    """
    Calls the Robyn API with the specified argument and payload.
    """
    if payload == 0:
        response = requests.get(api.format(argument))
    else:
        response = requests.post(api.format(argument), data=payload)
    return json.loads(response.content.decode("utf-8"))


def render_spendexposure(mmmdata: MMMData, max_size=(1000, 1500)):
    """
    Renders the exposure plots for the given MMMData.
    """
    print("MMMdata exposure_vars: ", mmmdata.mmmdata_spec.paid_media_vars)

    if len(mmmdata.mmmdata_spec.paid_media_vars) > 0:
        print("Rendering Spend vs Exposure Plots")
        for var in mmmdata.mmmdata_spec.paid_media_vars:
            # Assuming the plot data is stored in the MMMData object
            # You may need to adjust this based on your actual implementation
            if hasattr(mmmdata, "plots") and var in mmmdata.plots:
                plot_data = mmmdata.plots[var]
                img = Image.open(io.BytesIO(plot_data))
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.show()
            else:
                print(f"No plot data found for {var}")
    else:
        print("No exposure variables found.")


def plot_outputgraphs(model_outputs: ModelOutputs, graphtype: str, max_size=(1000, 1500)):
    """
    Plots the output graphs for the given ModelOutputs.
    """
    if graphtype in ["moo_distrb_plot", "moo_cloud_plot"]:
        # Assuming these plots are stored in the convergence attribute
        if hasattr(model_outputs, "convergence") and graphtype in model_outputs.convergence:
            image_data = binascii.unhexlify(model_outputs.convergence[graphtype])
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.show()
    elif graphtype == "ts_validation_plot":
        if model_outputs.ts_validation_plot:
            image_data = binascii.unhexlify(model_outputs.ts_validation_plot)
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.show()
    else:
        warnings.warn("Graphtype does not exist")


def load_modeldata(sol_id: str, mmmdata: MMMData, model_outputs: ModelOutputs):
    """
    Loads the model data for the given solution ID.
    """
    # This function might need significant changes based on your new implementation
    # You may need to create a new function in your ModelExecutor or RidgeModelBuilder
    # to generate the one-pager data
    pass


def create_robyn_directory(path="~/RobynOutcomes"):
    """
    Creates a directory for Robyn output files.
    """
    if path == "~":
        path = path.replace("~", os.path.expanduser("~")) + "/RobynOutcomes"
    elif "~" in path:
        path = path.replace("~", os.path.expanduser("~"))

    if path == os.path.expanduser("~") + "/RobynOutcomes":
        print("No path specified. Using default arguments")
    else:
        print("Using specified path")

    if "/" != path[-1:]:
        path = path + "/"

    if not os.path.exists(path):
        os.makedirs(path)
        print("Path did not exist. Creating path:", path)
    else:
        print("Path exists: ", path)

    return path


def load_onepager(
    mmmdata: MMMData,
    model_outputs: ModelOutputs,
    path: str,
    sol: str = "all",
    top_pareto: bool = False,
    write: bool = False,
    max_size=(1000, 1500),
):
    """
    Loads the one-page summary for the given solution ID.
    """
    if top_pareto and sol == "all":
        print("Fetching one pager data for top models")
        for trial in model_outputs.trials:
            sol_id = trial.result_hyp_param["solID"].values[0]
            onepager = load_modeldata(sol_id, mmmdata, model_outputs)
            if onepager:
                image_data = binascii.unhexlify(onepager)
                if write:
                    writefile(datset=image_data, path=path, sol_id=sol_id)
                image = Image.open(io.BytesIO(image_data))
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                image.show()
    elif not top_pareto and sol == "all":
        warnings.warn(
            "Too many one pagers to load, please either select top_pareto=True or just specify a solution id"
        )
    elif not top_pareto and sol != "all":
        if any(trial.result_hyp_param["solID"].values[0] == sol for trial in model_outputs.trials):
            print("Fetching one pager for specified solution id")
            onepager = load_modeldata(sol, mmmdata, model_outputs)
            if onepager:
                image_data = binascii.unhexlify(onepager)
                if write:
                    writefile(datset=image_data, path=path, sol_id=sol)
                image = Image.open(io.BytesIO(image_data))
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                image.show()
        else:
            warnings.warn("Specified solution id does not exist. Please check again")


# Other functions like write_robynmodel might need similar updates
# based on your new code structure
