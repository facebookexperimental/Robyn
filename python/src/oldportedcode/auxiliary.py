# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_rsq(true, predicted, p=None, df_int=None, n_train=None):
    """
    Calculate R-squared.

    Parameters
    ----------
    true : array-like
        True values.
    predicted : array-like
        Predicted values.
    p : int, optional
        Number of parameters.
    df_int : int, optional
        Degrees of freedom for intercept.
    n_train : int, optional
        Number of training examples.

    Returns
    -------
    rsq : float
        R-squared value.
    """
    sse = np.sum((predicted - true)**2)
    sst = np.sum((true - np.mean(true))**2)
    rsq = 1 - sse / sst

    if p is not None and df_int is not None:
        n = n_train if n_train is not None else len(true)
        rdf = n - p - 1
        rsq_adj = 1 - (1 - rsq) * (n - df_int) / rdf
        rsq = rsq_adj

    return rsq

def robyn_palette():
    """
    Robyn colors.

    Returns
    -------
    pal : list
        Color palette.
    """
    pal = [
        "#21130d", "#351904", "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",
        "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#043F43", "#04272D"
    ]
    repeated = 4
    return {
        "fill": rep(pal, repeated),
        "colour": rep(c(rep("#FFFFFF", 4), rep("#000000", 7), rep("#FFFFFF", 3)), repeated)
    }

def flatten_hyps(x):
    """
    Flatten hypothesis.

    Parameters
    ----------
    x : array-like
        Hypothesis.

    Returns
    -------
    str
        Flattened hypothesis.
    """
    if x is None:
        return x
    temp = np.array(lapply(x, lambda x: f"{x:.6f}".format(x)))
    return np.array(temp).reshape(-1, 1)

def robyn_update(dev=True, *args):
    """
    Update Robyn version.

    Parameters
    ----------
    dev : bool, optional
        Dev version? If not, CRAN version.
    *args : tuple
        Parameters to pass to install_github or install.packages.

    Returns
    -------
    None
    """
    if dev:
        try:
            import remotes
            remotes.install_github("facebookexperimental/Robyn/R", *args)
        except:
            pass
    else:
        import utils
        utils.install.packages("Robyn", *args)
