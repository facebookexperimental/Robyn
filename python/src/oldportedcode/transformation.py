# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import FuncFormatter
## from scipy.special import weibull
## Manually added for ggplot
from plotnine import ggplot, aes, labs, geom_point, geom_line, theme_gray, geom_hline, geom_text, facet_grid
import sklearn

from .checks import check_adstock

def mic_men(x, Vmax, Km, reverse=False):
    """
    Calculate the Michaelis-Menten transformation.

    Args:
        x (float): The input value.
        Vmax (float): The maximum rate of the transformation.
        Km (float): The Michaelis constant.
        reverse (bool, optional): Whether to perform the reverse transformation. Defaults to False.

    Returns:
        float: The transformed value.
    """
    if not reverse:
        mm_out = Vmax * x / (Km + x)
    else:
        mm_out = x * Km / (Vmax - x)
    return mm_out

def adstock_geometric(x, theta):
    """
    Adstock geometric function

    Parameters:
    x (list): The input values.
    theta (float): The decay factor.

    Returns:
    pandas.DataFrame: A DataFrame containing the original input values, the decayed values, the cumulative decay factors, and the inflation total.
    """
    ##if len(theta) != 1:
    if theta is None:
        ##raise ValueError("Length of theta should be 1")
        raise ValueError("Theta can not be Null")

    if len(x) > 1:
        x_decayed = [x[0]]
        for i in range(1, len(x)):
            decayed_value = x[i] + theta * x_decayed[i - 1]
            if not np.isscalar(decayed_value):
                decayed_value = decayed_value.iloc[0]
            x_decayed.append(decayed_value)
        x_decayed = np.array(x_decayed)

        if not np.isscalar(theta):
            theta = theta.iloc[0]
        thetaVecCum = [theta]
        for i in range(1, len(x)):
            cum_value = thetaVecCum[i - 1] * theta
            if not np.isscalar(cum_value):
                cum_value = cum_value.iloc[0]
            thetaVecCum.append(cum_value)
        thetaVecCum = np.array(thetaVecCum)

    else:
        # x_decayed = [val[0] for val in x]
        x_decayed = [val[0] if isinstance(val, (list, tuple)) else val for val in x]
        ##thetaVecCum = np.array([theta])
        thetaVecCum = list()
        thetaVecCum.append(theta)

    inflation_total = np.sum(x_decayed) / np.sum(x)

    return pd.DataFrame(
        {
            "x": x,
            "x_decayed": x_decayed,
            "thetaVecCum": thetaVecCum,
            "inflation_total": inflation_total,
        }
    )

## def adstock_weibull(x, shape, scale, windlen=len(x), type="cdf"):
## using stype
def adstock_weibull(x, shape, scale, windlen=None, stype="cdf"):
    """
    Adstock Weibull function

    Calculates the adstock transformation using the Weibull function.

    Parameters:
    - x: array-like
        The input time series data.
    - shape: float
        The shape parameter of the Weibull distribution.
    - scale: float
        The scale parameter of the Weibull distribution.
    - windlen: int, optional
        The length of the adstock window. If not provided, it defaults to the length of x.
    - stype: str, optional
        The type of adstock transformation to perform. Valid options are "cdf" (default) and "pdf".

    Returns:
    - dict:
        A dictionary containing the following keys:
        - "x": array-like
            The input time series data.
        - "x_decayed": array-like
            The adstock transformed data.
        - "thetaVecCum": array-like
            The cumulative adstock weights.
        - "inflation_total": float
            The total inflation factor.
        - "x_imme": array-like
            The immediate adstock transformed data.
    """
    ## Added manually since Python function signature fails getting len of x
    if windlen is None:
        windlen = len(x)
    ## if len(shape) != 1:
    if shape is None:
        ##raise ValueError("Length of shape should be 1")
        raise ValueError("Shape should be a number")
    ## if len(scale) != 1:
    if scale is None:
        ##raise ValueError("Length of scale should be 1")
        raise ValueError("Scale should be a number")
    if len(x) > 1:
        ## check_opts(stype.lower(), ["cdf", "pdf"])
        if stype.lower() not in ["cdf", "pdf"]:
            warnings.warn("Not valid type")

        ## x_bin = np.arange(1, windlen + 1)
        x_bin = np.arange(1, windlen + 1)
        scaleTrans = np.round(np.quantile(x_bin, scale), 0)
        if shape == 0 or scale == 0:
            x_decayed = x
            thetaVecCum = np.zeros(windlen)
            x_imme = None
        else:
            ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html
            if stype.lower() == "cdf":
                ##[1]
                ##1 - scipy.stats.weibull.cdf(x_bin, shape, scaleTrans)
                ##for _ in range(1, windlen) ##]
                thetaVec = np.append([1], 1 - scipy.stats.exponweib.cdf(x_bin[:len(x_bin)-1], shape, scaleTrans))
                thetaVecCum = np.cumprod(thetaVec)
            elif stype.lower() == "pdf":
                thetaVecCum = scipy.stats.exponweib.pdf(x_bin, shape, scaleTrans)
                ##np.array(
                    ##[1]
                    ##+ [
                        ## scipy.stats.weibull.pdf(x_bin, shape, scaleTrans)
                        ##for _ in range(1, windlen)
                    ##]
                ##)
                thetaVecCum = sklearn.preprocessing.normalize(thetaVecCum[:,np.newaxis], axis=0).ravel()
            ##else:
            ##    raise ValueError("Invalid type")
            ## x_pos = np.arange(1, 100)
            x_pos = np.arange(1, len(x) + 1)
            x_val = x
            x_decayed = list()
            ## Manually implemented mapply of R but can be better , may need to optimize
            for i, val in enumerate(x_val):
                x_pos_repeated = np.repeat(0, x_pos[i] - 1)
                x_pos_repeated_reverse = np.repeat(x_val[i], windlen - x_pos[i] + 1)
                x_vec = np.append(x_pos_repeated, x_pos_repeated_reverse)
                thetaVecCumLag = np.roll(thetaVecCum, x_pos[i]- 1)
                x_decayed_row = np.multiply(x_vec, thetaVecCumLag)
                x_decayed.append(x_decayed_row)

            x_decayed = np.asmatrix(x_decayed)
            x_imme = np.diag(x_decayed)
            x_decayed = np.sum(x_decayed, axis=1)
    else:
        x_imme = x
        x_decayed = x_imme
        ## manually added
        thetaVecCum = 1 ##np.array([1])

    inflation_total = np.sum(x_decayed) / np.sum(x)
    ## return pd.DataFrame
    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": thetaVecCum,
        "inflation_total": inflation_total,
        "x_imme": x_imme
        }

def transform_adstock(x, adstock, theta=None, shape=None, scale=None, windlen=None):
    """
    Transforms the input data using the adstock model.

    Parameters:
    - x: The input data to be transformed.
    - adstock: The type of adstock model to be applied. Possible values are "geometric", "weibull_cdf", and "weibull_pdf".
    - theta: The decay factor for the geometric adstock model. Only applicable if adstock is "geometric".
    - shape: The shape parameter for the Weibull adstock model. Only applicable if adstock is "weibull_cdf" or "weibull_pdf".
    - scale: The scale parameter for the Weibull adstock model. Only applicable if adstock is "weibull_cdf" or "weibull_pdf".
    - windlen: The length of the adstock window. If not provided, it defaults to the length of the input data.

    Returns:
    - x_list_sim: The transformed data based on the adstock model.

    """
    ## Added manually since Python function signature fails getting len of x
    if windlen != None:
        windlen = len(x)

    ## Added manually, LLaMa didn't get this one
    check_adstock(adstock)

    x_list_sim = None

    if adstock == "geometric":
        x_list_sim = adstock_geometric(x = x, theta = theta)
    elif adstock == "weibull_cdf":
        x_list_sim = adstock_weibull(x, shape, scale, windlen, "cdf")
    elif adstock == "weibull_pdf":
        x_list_sim = adstock_weibull(x, shape, scale, windlen, "pdf")
    return x_list_sim

## TODO: diff and range?
def normalize(x):
    """
    Normalizes the input data.

    Parameters:
    x (array-like): The input data to be normalized.

    Returns:
    array-like: The normalized data.
    """
    if np.diff(np.range(x)) == 0:
        return np.array([1, np.zeros(len(x) - 1)])
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def saturation_hill(x, alpha, gamma, x_marginal=None):
    """
    Implements the saturation hill function.

    Parameters:
    - x: Input values.
    - alpha: Exponent parameter.
    - gamma: Weighting parameter.
    - x_marginal: Optional marginal values.

    Returns:
    - x_scurve: Output values computed using the saturation hill function.
    """
    ## No need to length check for alpha and gamma since they are numbers not like lists in R
    ## linear interpolation by dot product
    ##inflexion <- c(range(x) %*% c(1 - gamma, gamma)) # linear interpolation by dot product
    ##np.repeat(x)
    if alpha is None or gamma is None:
        raise ValueError("Alpha and Gamma cannot be None")

    if not np.isscalar(gamma):
        gamma = gamma.iloc[0]
    if not np.isscalar(alpha):
        alpha = alpha.iloc[0]
    inflexion = np.dot(np.array([1 - gamma, gamma]), np.array([np.min(x), np.max(x)]))
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + np.power(inflexion, alpha))
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + np.power(inflexion, alpha))

    return x_scurve

def plot_adstock(plot=True):
    """
    Plots the adstock models.

    Parameters:
        plot (bool): If True, plots the adstock models. Default is True.

    Returns:
        p1 (ggplot): The plot of the geometric adstock model.
        p2 (ggplot): The plot of the Weibull adstock model.
    """
    if plot:
        # Plot geometric
        geomCollect = []
        thetaVec = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9])
        for v in range(len(thetaVec)):
            ## thetaVecCum = np.power(np.array([1, np.inf]), thetaVec[v])
            thetaVecCum = [0] * 100
            thetaVecCum[0] = 1
            ## Manually added
            for t in range(1, 100):
                thetaVecCum[t] = thetaVecCum[t-1] * thetaVec[v]

            dt_geom = pd.DataFrame(
                {
                    "x": np.arange(0, 100),
                    "decay_accumulated": thetaVecCum,
                    "theta": thetaVec[v],
                }
            )
            ## Changed
            ## dt_geom["halflife"] = np.where(dt_geom["decay_accumulated"] == 0.5, 1, 0)
            dt_geom["halflife"] = np.argmin(abs(dt_geom["decay_accumulated"] - 0.5))
            geomCollect.append(dt_geom)

        geomCollect = pd.concat(geomCollect)
        ## Added astype for correction
        geomCollect["theta_halflife"] = (
            geomCollect["theta"].astype(str) + "_" + geomCollect["halflife"].astype(str)
        )

        ## Used plotline to use ggplot almost as is from R
        p1 = (
            ggplot(geomCollect, aes(x="x", y="decay_accumulated"))
            + geom_line(aes(color="theta_halflife"))
            + geom_hline(yintercept=0.5, linetype="dashed", color="gray")
            ##+ geom_text(aes(x = max("x"), y = 0.5, vjust = -0.5, hjust = 1, label = "halflife"), colour = "gray")
            + geom_text(aes(x = max("x"), y = 0.5, label = "halflife"), colour = "gray")
            + labs(
                    title = "Geometric Adstock\n(Fixed decay rate)",
                    subtitle = "Halflife = time until effect reduces to 50%",
                    x = "Time unit",
                    y = "Media decay accumulated"
                )
            + theme_gray()
        )

        # Plot weibull
        weibullCollect = []
        shapeVec = np.array([0.5, 1, 2, 9])
        scaleVec = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.5])
        types = ["CDF", "PDF"]
        for t in range(len(types)):
            for v1 in range(len(shapeVec)):
                for v2 in range(len(scaleVec)):
                    dt_weibull = pd.DataFrame(
                        {
                            "x": np.arange(1, 101),
                            "decay_accumulated": adstock_weibull(
                                np.arange(1, 101), shapeVec[v1], scaleVec[v2], stype=types[t].lower()
                            )["thetaVecCum"],
                            "shape": f"shape={shapeVec[v1]}",
                            "scale": scaleVec[v2],
                            "type": types[t],
                        }
                    )
                    ## Manually changed
                    ## dt_weibull["halflife"] = np.where(dt_weibull["decay_accumulated"] == 0.5, 1, 0)
                    dt_weibull["halflife"] = np.argmin(abs(dt_weibull["decay_accumulated"] - 0.5))
                    weibullCollect.append(dt_weibull)
        weibullCollect = pd.concat(weibullCollect)

        ## Using plotline to use ggplot almost as is from R
        p2 = (
            ggplot(weibullCollect, aes(x="x", y="decay_accumulated"))
            + geom_line(aes(color="scale"))
            + facet_grid("shape ~ type")
            + geom_hline(yintercept=0.5, linetype="dashed", color="gray")
            ##+ geom_text(aes(x = max("x"), y = 0.5, vjust = -0.5, hjust = "center", label = "halflife"), colour = "gray")
            + geom_text(aes(x = max("x"), y = 0.5, label = "halflife"), colour = "gray")
            + labs(
                    title = "Weibull Adstock CDF vs PDF\n(Flexible decay rate)",
                    subtitle = "Halflife = time until effect reduces to 50%",
                    x = "Time unit",
                    y = "Media decay accumulated"
                )
            + theme_gray()
        )

        # Create plots
        """ Manually commented out to use ggplot as in R code
        p1 = plt.figure(figsize=(10, 6))
        p1.plot(geomCollect["x"], geomCollect["decay_accumulated"], label="Geometric")
        p1.set_xlabel("Time unit")
        p1.set_ylabel("Media decay accumulated")
        p1.legend()
        p1.title("Geometric Adstock (Fixed decay rate)")

        p2 = plt.figure(figsize=(10, 6))
        p2.plot(
            weibullCollect["x"], weibullCollect["decay_accumulated"], label="Weibull"
        )
        p2.set_xlabel("Time unit")
        p2.set_ylabel("Media decay accumulated")
        p2.legend()
        p2.title("Weibull Adstock (Flexible decay rate)")
        """

        return p1, p2


def plot_saturation(plot=True):
    """
    Plots the saturation response using the hill function.

    Parameters:
    - plot (bool): If True, the plot will be displayed. Default is True.

    Returns:
    - p1 (ggplot object): The plot of the saturation response with varying alpha values.
    - p2 (ggplot object): The plot of the saturation response with varying gamma values.
    """
    ## Too wrong porting
    # Create a sample dataset
    ## Manually corrected the for loops
    if plot:
        x_sample = np.arange(1, 100, 1)
        alpha_sample = np.array([0.1, 0.5, 1, 2, 3])
        gamma_sample = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        hillAlphaCollect = list()
        for i in range(len(alpha_sample)):

            # Create a dataframe with the sample data
            df = pd.DataFrame({
                "x": x_sample,
                "y": x_sample**alpha_sample[i] / (x_sample**alpha_sample[i] + (0.5 * 100)**alpha_sample[i]),
                "alpha": alpha_sample[i]
                })
            hillAlphaCollect.append(df)

        hillAlphaCollect = pd.concat(hillAlphaCollect)

        """ Manually commented out to use ggplot as in R code
        plt.plot(hillAlphaCollect["x"], hillAlphaCollect["y"])
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Saturation Response")
        plt.show()
        """

        p1 = (
            ggplot(hillAlphaCollect, aes(x = "x", y = "y", color = "alpha"))
            + geom_line()
            + labs(title = "Cost response with hill function", subtitle = "Alpha changes while gamma = 0.5")
            + theme_gray(background = "white", pal = 2)
        )

        hillAlphaCollect = list()
        for i in range(len(gamma_sample)):
            # Create a dataframe with the sample data
            df = pd.DataFrame({
                "x": x_sample,
                "y": x_sample**2 / (x_sample**2 + (gamma_sample[i] * 100)**2),
                "gamma": gamma_sample[i]
                })
            hillAlphaCollect.append(df)

        hillAlphaCollect = pd.concat(hillAlphaCollect)

        p2 = (
            ggplot(hillGammaCollect, aes(x = "x", y = "y", color = "gamma"))
            + geom_line()
            + labs(
                title = "Cost response with hill function",
                subtitle = "Gamma changes while alpha = 2"
            )
            + theme_gray(background = "white", pal = 2)
        )

        """ Manually commented out to use ggplot as in R code
        plt.plot(hillAlphaCollect["x"], hillAlphaCollect["y"])
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Cost response with hill function")
        plt.show()
        """

        return p1, p2


def run_transformations(input_collect, hyp_param_sam, adstock):
    """
    Run transformations on the input data.

    Args:
        input_collect (pd.DataFrame): The input data frame.
        hyp_param_sam (dict): The dictionary containing the hyperparameters.
        adstock (str): The type of adstocking to be applied.

    Returns:
        dict: A dictionary containing the transformed data frames.
            - dt_modSaturated (pd.DataFrame): The saturated data frame.
            - dt_saturatedImmediate (pd.DataFrame): The saturated immediate data frame.
            - dt_saturatedCarryover (pd.DataFrame): The saturated carryover data frame.
    """

    if "robyn_inputs" in input_collect:
        input_collect = input_collect["robyn_inputs"]

    # Extract the media names from the input collect dataframe
    all_media = input_collect["all_media"]

    # Extract the rolling window start and end indices
    rolling_window_start_which = input_collect["rollingWindowStartWhich"]
    rolling_window_end_which = input_collect["rollingWindowEndWhich"]

    select_columns = [column for column in input_collect['dt_mod'].columns if column != 'ds']
    dt_modAdstocked = input_collect['dt_mod'][select_columns]

    # Create a list to store the media adstocked data
    media_adstocked = dict()

    # Create a list to store the media immediate data
    media_immediate = dict()

    # Create a list to store the media carryover data
    media_carryover = dict()

    # Create a list to store the media cumulative data
    media_vec_cum = dict()

    # Create a list to store the media saturated data
    media_saturated = dict()

    # Create a list to store the media saturated immediate data
    media_saturated_immediate = dict()

    # Create a list to store the media saturated carryover data
    media_saturated_carryover = dict()

    # Iterate over each media name
    for v in range(len(all_media)):
        # Extract the media name
        media = all_media[v]

        m = list(dt_modAdstocked[[media]].values)
        m = np.array(m).reshape(len(m),)
        theta = shape = scale = None
        # Extract the adstocking parameters for this media
        if adstock == "geometric":
            theta = hyp_param_sam[f"{media}_thetas"]##[0]

        if adstock.startswith('weibull'):
            shape = hyp_param_sam[f"{media}_shapes"]##[0]
            scale = hyp_param_sam[f"{media}_scales"]##[0]

        # Calculate the adstocked response
        x_list = transform_adstock(
            m, adstock, theta=theta, shape=shape, scale=scale
        )

        m_adstocked = x_list["x_decayed"]
        m_adstocked = np.array(m_adstocked).reshape(len(m_adstocked),)
        # Store the adstocked data for this media
        ##media_adstocked.append(m_adstocked)
        media_adstocked[media] = m_adstocked

        # Calculate the immediate response
        if adstock == "weibull_pdf":
            m_imme = x_list["x_imme"]
            m_imme = np.array(m_imme).reshape(len(m_imme),)
        else:
            m_imme = m

        # Calculate the carryover response
        ## m_carryover = x_list["x_decayed"] - m_imme
        m_carryover = m_adstocked - m_imme

        # Store the immediate and carryover data for this media
        ##media_immediate.append(m_imme)
        media_immediate[media] = m_imme
        ##media_carryover.append(m_carryover)
        media_carryover[media] = m_carryover
        ##media_vec_cum.append(x_list["thetaVecCum"])
        media_vec_cum[media] = x_list["thetaVecCum"]

        m_adstockedRollWind = m_adstocked[(rolling_window_start_which-1):(rolling_window_end_which)]
        m_carryoverRollWind = m_carryover[(rolling_window_start_which-1):(rolling_window_end_which)]

        # Calculate the saturated response
        alpha = hyp_param_sam[f"{media}_alphas"] ##[0]
        gamma = hyp_param_sam[f"{media}_gammas"] ##[0]
        ##media_saturated.append(saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma))
        media_saturated[media] = saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma)

        # Calculate the saturated carryover response
        media_saturated_carryover[media] = saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma, x_marginal = m_carryoverRollWind)

        # Calculate the saturated immediate response
        ##media_saturated_immediate.append(media_saturated[v] - media_saturated_carryover[v])
        media_saturated_immediate[media] = media_saturated[media] - media_saturated_carryover[media]

    select_columns = [column for column in dt_modAdstocked.columns if column not in all_media]
    media_adstocked = pd.DataFrame(media_adstocked)
    dt_modAdstockedTemp = dt_modAdstocked[select_columns]
    dt_modAdstocked = pd.concat([dt_modAdstockedTemp.reset_index(), media_adstocked], axis=1)

    dt_mediaImmediate = pd.DataFrame(media_immediate)
    dt_mediaCarryover = pd.DataFrame(media_carryover)
    mediaVecCum = pd.DataFrame(media_vec_cum)

    mediaSaturated = pd.DataFrame(media_saturated)
    dt_modSaturatedTemp = dt_modAdstocked.loc[(rolling_window_start_which-1):(rolling_window_end_which-1)][select_columns]
    dt_modSaturated = pd.concat([dt_modSaturatedTemp.reset_index(), mediaSaturated], axis=1)

    dt_saturatedImmediate = pd.DataFrame(media_saturated_immediate)
    dt_saturatedImmediate.fillna(0, inplace=True)
    dt_saturatedCarryover = pd.DataFrame(media_saturated_carryover)
    dt_saturatedCarryover.fillna(0, inplace=True)

    # Create a dataframe with the media data
    ## media_df = pd.DataFrame(
    ##    {
    ##        "media": all_media,
    ##        "adstocked": media_adstocked,
    ##        "immediate": media_immediate,
    ##        "carryover": media_carryover,
    ##        "saturated": media_saturated,
    ##        "saturated_immediate": media_saturated_immediate,
    ##        "saturated_carryover": media_saturated_carryover,
    ##    }
    ##)

    return {
        "dt_modSaturated": dt_modSaturated,
        "dt_saturatedImmediate": dt_saturatedImmediate,
        "dt_saturatedCarryover": dt_saturatedCarryover
    }
