
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line, labs, theme_gray, geom_hline, geom_text, facet_grid
from robyn.analysis.transformation import AdstockSaturationTransformation

class AdstockSaturationPlots:
    def plot_adstock_models(self, theta_values: np.ndarray, shape_values: np.ndarray, scale_values: np.ndarray, plot: bool = True) -> tuple:
        """
        Plots the adstock models.

        Parameters:
            theta_values (np.ndarray): Array of theta values for the geometric adstock model.
            shape_values (np.ndarray): Array of shape values for the Weibull adstock model.
            scale_values (np.ndarray): Array of scale values for the Weibull adstock model.
            plot (bool): If True, plots the adstock models. Default is True.

        Returns:
            tuple: A tuple containing the geometric adstock plot and the Weibull adstock plot.
        """
        if plot:
            transform = AdstockSaturationTransformation()
            # Plot geometric
            geom_collect = []
            for theta in theta_values:
                theta_vec_cum = [0] * 100
                theta_vec_cum[0] = 1
                for t in range(1, 100):
                    theta_vec_cum[t] = theta_vec_cum[t-1] * theta

                dt_geom = pd.DataFrame({
                    "x": np.arange(0, 100),
                    "decay_accumulated": theta_vec_cum,
                    "theta": theta,
                })
                dt_geom["halflife"] = np.argmin(abs(dt_geom["decay_accumulated"] - 0.5))
                geom_collect.append(dt_geom)

            geom_collect = pd.concat(geom_collect)
            geom_collect["theta_halflife"] = geom_collect["theta"].astype(str) + "_" + geom_collect["halflife"].astype(str)

            p1 = (
                ggplot(geom_collect, aes(x="x", y="decay_accumulated"))
                + geom_line(aes(color="theta_halflife"))
                + geom_hline(yintercept=0.5, linetype="dashed", color="gray")
                + geom_text(aes(x=max("x"), y=0.5, label="halflife"), colour="gray")
                + labs(
                    title="Geometric Adstock\n(Fixed decay rate)",
                    subtitle="Halflife = time until effect reduces to 50%",
                    x="Time unit",
                    y="Media decay accumulated"
                )
                + theme_gray()
            )

            # Plot Weibull
            weibull_collect = []
            types = ["CDF", "PDF"]
            for t in range(len(types)):
                for shape in shape_values:
                    for scale in scale_values:
                        dt_weibull = pd.DataFrame({
                            "x": np.arange(1, 101),
                            "decay_accumulated": transform.adstock_weibull(np.arange(1, 101), shape, scale, stype=types[t].lower())["thetaVecCum"],
                            "shape": f"shape={shape}",
                            "scale": scale,
                            "type": types[t],
                        })
                        dt_weibull["halflife"] = np.argmin(abs(dt_weibull["decay_accumulated"] - 0.5))
                        weibull_collect.append(dt_weibull)

            weibull_collect = pd.concat(weibull_collect)

            p2 = (
                ggplot(weibull_collect, aes(x="x", y="decay_accumulated"))
                + geom_line(aes(color="scale"))
                + facet_grid("shape ~ type")
                + geom_hline(yintercept=0.5, linetype="dashed", color="gray")
                + geom_text(aes(x=max("x"), y=0.5, label="halflife"), colour="gray")
                + labs(
                    title="Weibull Adstock CDF vs PDF\n(Flexible decay rate)",
                    subtitle="Halflife = time until effect reduces to 50%",
                    x="Time unit",
                    y="Media decay accumulated"
                )
                + theme_gray()
            )

            return p1, p2

    def plot_saturation_response(self, alpha_values: np.ndarray, gamma_values: np.ndarray, plot: bool = True) -> tuple:
        """
        Plots the saturation response using the hill function.

        Parameters:
            alpha_values (np.ndarray): Array of alpha values for the hill function.
            gamma_values (np.ndarray): Array of gamma values for the hill function.
            plot (bool): If True, the plot will be displayed. Default is True.

        Returns:
            tuple: A tuple containing the saturation response plot with varying alpha values and the saturation response plot with varying gamma values.
        """
        if plot:
            x_sample = np.arange(1, 100, 1)

            hill_alpha_collect = []
            for alpha in alpha_values:
                df = pd.DataFrame({
                    "x": x_sample,
                    "y": x_sample**alpha / (x_sample**alpha + (0.5 * 100)**alpha),
                    "alpha": alpha
                })
                hill_alpha_collect.append(df)

            hill_alpha_collect = pd.concat(hill_alpha_collect)

            p1 = (
                ggplot(hill_alpha_collect, aes(x="x", y="y", color="alpha"))
                + geom_line()
                + labs(title="Cost response with hill function", subtitle="Alpha changes while gamma = 0.5")
                + theme_gray(background="white", pal=2)
            )

            hill_gamma_collect = []
            for gamma in gamma_values:
                df = pd.DataFrame({
                    "x": x_sample,
                    "y": x_sample**2 / (x_sample**2 + (gamma * 100)**2),
                    "gamma": gamma
                })
                hill_gamma_collect.append(df)

            hill_gamma_collect = pd.concat(hill_gamma_collect)

            p2 = (
                ggplot(hill_gamma_collect, aes(x="x", y="y", color="gamma"))
                + geom_line()
                + labs(
                    title="Cost response with hill function",
                    subtitle="Gamma changes while alpha = 2"
                )
                + theme_gray(background="white", pal=2)
            )

            return p1, p2
