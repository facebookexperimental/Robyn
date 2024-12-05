from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any
import logging
from robyn.modeling.entities.modeloutputs import Trial
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ModelConvergenceVisualizer(BaseVisualizer):
    def __init__(
        self,
        n_cuts: Optional[int] = None,
        nrmse_win: Optional[List[float]] = None,
        ts_validation_plot: Optional[List[Trial]] = None,
        moo_cloud_plot: Optional[pd.DataFrame] = None,
        moo_distrb_plot: Optional[pd.DataFrame] = None,
    ):
        super().__init__()  # Initialize BaseVisualizer
        self.n_cuts = n_cuts
        self.nrmse_win = nrmse_win
        self.ts_validation_plot = ts_validation_plot
        self.moo_cloud_plot = moo_cloud_plot
        self.moo_distrb_plot = moo_distrb_plot
        logger.info("Initialized ModelConvergenceVisualizer")

    def create_moo_distrb_plot(
        self, dt_objfunc_cvg: pd.DataFrame, conv_msg: List[str]
    ) -> Dict[str, plt.Figure]:
        logger.debug(
            "Starting moo distribution plot creation with data shape: %s",
            dt_objfunc_cvg.shape,
        )
        try:
            # Convert 'cuts' to categorical with a specific order
            dt_objfunc_cvg["id"] = dt_objfunc_cvg["cuts"].astype(int)
            dt_objfunc_cvg["cuts"] = pd.Categorical(
                dt_objfunc_cvg["cuts"],
                categories=sorted(dt_objfunc_cvg["cuts"].unique(), reverse=True),
            )
            
            # Clip values based on quantiles
            for error_type in dt_objfunc_cvg["error_type"].unique():
                mask = dt_objfunc_cvg["error_type"] == error_type
                original_values = dt_objfunc_cvg.loc[mask, "value"]
                quantiles = np.quantile(original_values, self.nrmse_win)
                dt_objfunc_cvg.loc[mask, "value"] = np.clip(original_values, *quantiles)
                logger.debug(
                    "Clipped values for error_type %s: min=%f, max=%f",
                    error_type,
                    quantiles[0],
                    quantiles[1],
                )

            # Create figure using base visualizer methods
            fig, ax = self.create_figure(figsize=self.figure_sizes["default"])

            # Create the violin plot
            sns.violinplot(
                data=dt_objfunc_cvg,
                x="value",
                y="cuts",
                hue="error_type",
                split=True,
                inner="quartile",
                ax=ax,
            )

            # Set labels and styling using base visualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="Objective functions",
                ylabel="Iterations [#]",
                title="Objective convergence by iterations quantiles"
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)
            self._add_standardized_legend(ax, loc='lower right')

            # Add convergence messages
            plt.figtext(
                0.98,
                0.02,
                "\n".join(conv_msg),
                ha="right",
                va="bottom",
                fontsize=self.fonts["sizes"]["small"],
                wrap=True,
            )

            self.finalize_figure(tight_layout=True)
            logger.info("Successfully created moo distribution plot")
            return {"moo_distribution": fig}

        except Exception as e:
            logger.error(
                "Failed to create moo distribution plot: %s", str(e), exc_info=True
            )
            raise

    def create_moo_cloud_plot(
        self, df: pd.DataFrame, conv_msg: List[str], calibrated: bool
    ) -> Dict[str, plt.Figure]:
        logger.debug(
            "Starting moo cloud plot creation with data shape: %s, calibrated=%s",
            df.shape,
            calibrated,
        )

        try:
            # Clip NRMSE values based on quantiles
            original_nrmse = df["nrmse"]
            quantiles = np.quantile(original_nrmse, self.nrmse_win)
            df["nrmse"] = np.clip(original_nrmse, *quantiles)

            # Create figure using base visualizer methods
            fig, ax = self.create_figure(figsize=self.figure_sizes["default"])

            # Create scatter plot
            scatter = ax.scatter(
                df["nrmse"],
                df["decomp.rssd"],
                c=df["ElapsedAccum"],
                cmap="viridis",
                alpha=self.alpha["primary"]
            )

            if calibrated and "mape" in df.columns:
                logger.debug("Adding calibrated MAPE visualization")
                sizes = (df["mape"] - df["mape"].min()) / (
                    df["mape"].max() - df["mape"].min()
                )
                sizes = sizes * 100 + 10
                ax.scatter(
                    df["nrmse"],
                    df["decomp.rssd"],
                    s=sizes,
                    alpha=self.alpha["secondary"],
                    edgecolor="w",
                    linewidth=0.5,
                )

            # Add colorbar
            plt.colorbar(scatter, label="Time [s]")

            # Set labels and styling using base visualizer methods
            self._set_standardized_labels(
                ax,
                xlabel="NRMSE",
                ylabel="DECOMP.RSSD",
                title="Multi-objective evolutionary performance"
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)

            # Add convergence messages
            plt.figtext(
                0.98,
                0.02,
                "\n".join(conv_msg),
                ha="right",
                va="bottom",
                fontsize=self.fonts["sizes"]["small"],
                wrap=True,
            )

            self.finalize_figure(tight_layout=True)
            logger.info("Successfully created moo cloud plot")
            return {"moo_cloud": fig}

        except Exception as e:
            logger.error("Failed to create moo cloud plot: %s", str(e), exc_info=True)
            raise

    def create_ts_validation_plot(self, trials: List[Trial]) -> Dict[str, plt.Figure]:
        logger.debug(
            "Starting time-series validation plot creation with %d trials", len(trials)
        )
        try:
            # Prepare data
            result_hyp_param = pd.concat(
                [trial.result_hyp_param for trial in trials], ignore_index=True
            )
            result_hyp_param["trial"] = (
                result_hyp_param.groupby("sol_id").cumcount() + 1
            )
            result_hyp_param["iteration"] = result_hyp_param.index + 1

            # Process metrics
            result_hyp_param_long = result_hyp_param.melt(
                id_vars=["sol_id", "trial", "train_size", "iteration"],
                value_vars=[
                    "rsq_train", "rsq_val", "rsq_test",
                    "nrmse_train", "nrmse_val", "nrmse_test"
                ],
                var_name="metric",
                value_name="value"
            )

            # Extract dataset and metric type
            result_hyp_param_long["dataset"] = result_hyp_param_long["metric"].str.split("_").str[-1]
            result_hyp_param_long["metric_type"] = result_hyp_param_long["metric"].str.split("_").str[0]

            # Winsorize the data
            result_hyp_param_long["value"] = result_hyp_param_long.groupby("metric_type")["value"].transform(
                lambda x: np.clip(
                    x,
                    np.percentile(x, self.nrmse_win[0] * 100),
                    np.percentile(x, self.nrmse_win[1] * 100),
                )
            )

            # Create figure using base visualizer methods
            num_trials = result_hyp_param["trial"].nunique()
            fig = plt.figure(figsize=self.figure_sizes["default"])
            
            # Create grid for subplots
            gs = fig.add_gridspec(num_trials + 1, 1, height_ratios=[3] * num_trials + [1])
            
            # NRMSE plots for each trial
            for i, trial in enumerate(result_hyp_param["trial"].unique()):
                ax = fig.add_subplot(gs[i])
                nrmse_data = result_hyp_param_long[
                    (result_hyp_param_long["metric_type"] == "nrmse")
                    & (result_hyp_param_long["trial"] == trial)
                ]

                # Create plots
                sns.scatterplot(
                    data=nrmse_data,
                    x="iteration",
                    y="value",
                    hue="dataset",
                    style="dataset",
                    markers=["o", "s", "D"],
                    ax=ax,
                    alpha=self.alpha["primary"]
                )
                
                sns.lineplot(
                    data=nrmse_data,
                    x="iteration",
                    y="value",
                    hue="dataset",
                    ax=ax,
                    legend=False,
                    linewidth=1
                )

                # Style the subplot
                self._set_standardized_labels(
                    ax,
                    ylabel=f"NRMSE [Trial {trial}]",
                    xlabel="Iteration" if i == num_trials - 1 else ""
                )
                self._add_standardized_grid(ax)
                self._set_standardized_spines(ax)
                self._add_standardized_legend(ax, loc='lower right')

                # Only show x-label on bottom plot
                if i < num_trials - 1:
                    ax.set_xlabel("")

            # Train Size plot
            ax = fig.add_subplot(gs[-1])
            sns.scatterplot(
                data=result_hyp_param,
                x="iteration",
                y="train_size",
                hue="trial",
                ax=ax,
                legend=False,
            )

            # Style the train size plot
            self._set_standardized_labels(
                ax,
                xlabel="Iteration",
                ylabel="Train Size"
            )
            self._add_standardized_grid(ax)
            self._set_standardized_spines(ax)
            
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
            )

            # Set overall title
            fig.suptitle(
                "Time-series validation & Convergence",
                fontsize=self.fonts["sizes"]["title"],
                fontweight="bold",
                y=1.02
            )

            self.finalize_figure(tight_layout=True)
            logger.info("Successfully created time-series validation plot")
            return {"ts_validation": fig}

        except Exception as e:
            logger.error(
                "Failed to create time-series validation plot: %s",
                str(e),
                exc_info=True,
            )
            raise
        
    def _convert_plot_to_base64(self, fig: plt.Figure) -> str:
        logger.debug("Converting plot to base64")
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            logger.debug("Successfully converted plot to base64")
            return graphic.decode("utf-8")
        except Exception as e:
            logger.error("Failed to convert plot to base64: %s", str(e), exc_info=True)
            raise

    def display_convergence_plots(self, plots_dict: Dict[str, Any]) -> None:
        """
        Display all convergence plots from a dictionary.
        """
        logger.info("Displaying convergence plots")
        try:
            if 'moo_distrb_plot' in plots_dict and plots_dict['moo_distrb_plot']:
                logger.info("Displaying MOO distribution plot")
                for name, fig in plots_dict['moo_distrb_plot'].items():
                    plt.figure(fig.number)
                    plt.show()

            if 'moo_cloud_plot' in plots_dict and plots_dict['moo_cloud_plot']:
                logger.info("Displaying MOO cloud plot")
                for name, fig in plots_dict['moo_cloud_plot'].items():
                    plt.figure(fig.number)
                    plt.show()

            if 'ts_validation_plot' in plots_dict and plots_dict['ts_validation_plot']:
                logger.info("Displaying time series validation plot")
                for name, fig in plots_dict['ts_validation_plot'].items():
                    plt.figure(fig.number)
                    plt.show()
        except Exception as e:
            logger.error(f"Error displaying plots: {str(e)}")
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate all available plots.
        """
        logger.info("Generating all plots")
        plot_collect: Dict[str, plt.Figure] = {}

        try:
            # Generate plots if data is available
            if self.moo_distrb_plot is not None:
                logger.info("Creating moo distribution plot")
                plot_collect.update(self.create_moo_distrb_plot(self.moo_distrb_plot, []))

            if self.moo_cloud_plot is not None:
                logger.info("Creating moo cloud plot")
                plot_collect.update(self.create_moo_cloud_plot(self.moo_cloud_plot, [], False))

            if self.ts_validation_plot is not None:
                logger.info("Creating time series validation plot")
                plot_collect.update(self.create_ts_validation_plot(self.ts_validation_plot))

            if display_plots:
                logger.info(f"Displaying plots: {list(plot_collect.keys())}")
                for plot_name, fig in plot_collect.items():
                    plt.figure(fig.number)
                    plt.show()

            if export_location:
                logger.info(f"Exporting plots to: {export_location}")
                self.export_plots_fig(export_location, plot_collect)

            return plot_collect

        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise

    def __del__(self):
        """Cleanup when the visualizer is destroyed."""
        self.cleanup()