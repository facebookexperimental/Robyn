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
    Mic_men function
    """
    if not reverse:
        mm_out = Vmax * x / (Km + x)
    else:
        mm_out = x * Km / (Vmax - x)
    return mm_out


def adstock_geometric(x, theta):
    """
    Adstock geometric function
    """
    ##if len(theta) != 1:
    if theta is None:
        ##raise ValueError("Length of theta should be 1")
        raise ValueError("Theta can not be Null")

    if len(x) > 1:
        ## x_decayed = np.array([x[0], 0] * (len(x) - 1))
        x_decayed = list()
        x_decayed.append(x[0])
        x_decayed.extend(np.repeat(0, len(x) - 1))
        for i in range(1, len(x)):
            x_decayed[i] = x[i][0] + theta * x_decayed[i - 1]
        thetaVecCum = list()
        thetaVecCum.append(theta)
        for i in range(1, len(x)):
            thetaVecCum.append(thetaVecCum[i - 1] * theta)
    else:
        x_decayed = [val[0] for val in x]
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
def adstock_weibull(x, shape, scale, windlen = None, stype="cdf"):
    """
    Adstock Weibull function
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
    """
    ## Added manually since Python function signature fails getting len of x
    if windlen is None:
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
    """
    if np.diff(np.range(x)) == 0:
        return np.array([1, np.zeros(len(x) - 1)])
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def saturation_hill(x, alpha, gamma, x_marginal=None):
    """
    Implements the saturation hill function.
    """
    ## No need to length check for alpha and gamma since they are numbers not like lists in R
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + np.power(np.inf, alpha))
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + np.power(np.inf, alpha))
    return x_scurve


def plot_adstock(plot=True):
    """
    Plots the adstock models.
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
    # Extract the media names from the input collect dataframe
    all_media = input_collect["all_media"]

    # Extract the rolling window start and end indices
    rolling_window_start_which = input_collect["rollingWindowStartWhich"]
    rolling_window_end_which = input_collect["rollingWindowEndWhich"]

    select_columns = [column for column in input_collect['dt_mod'].columns if column != 'ds']
    dt_modAdstocked = input_collect['dt_mod'][select_columns]

    # Create a list to store the media adstocked data
    media_adstocked = []

    # Create a list to store the media immediate data
    media_immediate = []

    # Create a list to store the media carryover data
    media_carryover = []

    # Create a list to store the media cumulative data
    media_vec_cum = []

    # Create a list to store the media saturated data
    media_saturated = []

    # Create a list to store the media saturated immediate data
    media_saturated_immediate = []

    # Create a list to store the media saturated carryover data
    media_saturated_carryover = []

    # Iterate over each media name
    for v in range(len(all_media)):
        # Extract the media name
        media = all_media[v]

        m = list(dt_modAdstocked[[media]].values)
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
        # Store the adstocked data for this media
        media_adstocked.append(m_adstocked)

        # Calculate the immediate response
        if adstock == "weibull_pdf":
            m_imme = x_list["x_imme"]
        else:
            m_imme = m

        # Calculate the carryover response
        m_carryover = x_list["x_decayed"] - m_imme

        # Store the immediate and carryover data for this media
        media_immediate.append(m_imme)
        media_carryover.append(m_carryover)
        media_vec_cum.append(x_list["thetaVecCum"])

        m_adstockedRollWind <- m_adstocked[(rollingWindowStartWhich-1):(rollingWindowEndWhich-1)]
        m_carryoverRollWind <- m_carryover[(rollingWindowStartWhich-1):(rollingWindowEndWhich-1)]

        # Calculate the saturated response
        alpha = hyp_param_sam[f"{media}_alphas"] ##[0]
        gamma = hyp_param_sam[f"{media}_gammas"] ##[0]
        media_saturated.append(saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma))

        # Calculate the saturated carryover response
        media_saturated_carryover.append(saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma, x_marginal = m_carryoverRollWind))

        # Calculate the saturated immediate response
        media_saturated_immediate.append(media_saturated[v] - media_saturated_carryover[v])

    ## TODO:

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
