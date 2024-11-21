from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import Image, display

matplotlib.use("Agg")
import seaborn as sns
from typing import List, Optional, Union
import io
import base64
import logging
from robyn.modeling.entities.modeloutputs import Trial

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ModelConvergenceVisualizer:
    def __init__(
        self,
        n_cuts: Optional[int] = None,
        nrmse_win: Optional[List[float]] = None,
        ts_validation_plot: Optional[List[Trial]] = None,
        moo_cloud_plot: Optional[pd.DataFrame] = None,
        moo_distrb_plot: Optional[pd.DataFrame] = None,
    ):
        self.n_cuts = n_cuts
        self.nrmse_win = nrmse_win
        self.ts_validation_plot = ts_validation_plot
        self.moo_cloud_plot = moo_cloud_plot
        self.moo_distrb_plot = moo_distrb_plot
        logger.info("Initialized ModelConvergenceVisualizer")

    def create_moo_distrb_plot(
        self, dt_objfunc_cvg: pd.DataFrame, conv_msg: List[str]
    ) -> str:
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
            logger.debug(
                "Processing error types: %s", dt_objfunc_cvg["error_type"].unique()
            )
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
            # Set the style and color palette
            sns.set_style("whitegrid")
            sns.set_palette("Set2")
            # Create the violin plot with a larger figure size
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.violinplot(
                data=dt_objfunc_cvg,
                x="value",
                y="cuts",
                hue="error_type",
                split=True,
                inner="quartile",
                ax=ax,
            )
            ax.set_xlabel("Objective functions", fontsize=12, ha="left", x=0)
            ax.set_ylabel("Iterations [#]", fontsize=12)
            ax.set_title(
                "Objective convergence by iterations quantiles",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, linestyle="--", linewidth=0.5)
            # Adjust layout to make room for figtext on the bottom right
            plt.subplots_adjust(right=0.75, bottom=0.15)
            # Add text annotations on the bottom right
            plt.figtext(
                0.98,
                0,
                "\n".join(conv_msg),
                ha="right",
                va="bottom",
                fontsize=8,
                wrap=True,
            )
            plt.tight_layout()
            logger.info("Successfully created moo distribution plot")
            return self._convert_plot_to_base64(fig)
        except Exception as e:
            logger.error(
                "Failed to create moo distribution plot: %s", str(e), exc_info=True
            )
            raise

    def create_moo_cloud_plot(
        self, df: pd.DataFrame, conv_msg: List[str], calibrated: bool
    ) -> str:
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
            logger.debug(
                "Clipped NRMSE values: min=%f, max=%f", quantiles[0], quantiles[1]
            )
            # Set the style and color palette
            sns.set_style("whitegrid")
            sns.set_palette("Set2")
            # Create the scatter plot
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                df["nrmse"],
                df["decomp.rssd"],
                c=df["ElapsedAccum"],
                cmap="viridis",
                alpha=0.7,
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
                    alpha=0.5,
                    edgecolor="w",
                    linewidth=0.5,
                )
            plt.colorbar(scatter, label="Time [s]")
            ax.set_xlabel("NRMSE", fontsize=12, ha="left", x=0)
            ax.set_ylabel("DECOMP.RSSD", fontsize=12)
            ax.set_title(
                "Multi-objective evolutionary performance",
                fontsize=14,
                fontweight="bold",
            )
            # Add text annotations on the bottom right
            plt.figtext(
                0.98,
                0,
                "\n".join(conv_msg),
                ha="right",
                va="bottom",
                fontsize=8,
                wrap=True,
            )
            plt.tight_layout()

            logger.info("Successfully created moo cloud plot")
            return self._convert_plot_to_base64(fig)

        except Exception as e:
            logger.error("Failed to create moo cloud plot: %s", str(e), exc_info=True)
            raise

    def create_ts_validation_plot(self, trials: List[Trial]) -> str:
        logger.debug(
            "Starting time-series validation plot creation with %d trials", len(trials)
        )
        try:
            # Concatenate trial data
            result_hyp_param = pd.concat(
                [trial.result_hyp_param for trial in trials], ignore_index=True
            )
            result_hyp_param["trial"] = (
                result_hyp_param.groupby("sol_id").cumcount() + 1
            )
            result_hyp_param["iteration"] = result_hyp_param.index + 1
            logger.debug("Processing metrics for validation plot")
            result_hyp_param_long = result_hyp_param.melt(
                id_vars=["sol_id", "trial", "train_size", "iteration"],
                value_vars=[
                    "rsq_train",
                    "rsq_val",
                    "rsq_test",
                    "nrmse_train",
                    "nrmse_val",
                    "nrmse_test",
                ],
                var_name="metric",
                value_name="value",
            )
            # Extract dataset and metric type
            result_hyp_param_long["dataset"] = (
                result_hyp_param_long["metric"].str.split("_").str[-1]
            )
            result_hyp_param_long["metric_type"] = (
                result_hyp_param_long["metric"].str.split("_").str[0]
            )
            # Winsorize the data
            logger.debug("Winsorizing metric values")
            result_hyp_param_long["value"] = result_hyp_param_long.groupby(
                "metric_type"
            )["value"].transform(
                lambda x: np.clip(
                    x,
                    np.percentile(x, self.nrmse_win[0] * 100),
                    np.percentile(x, self.nrmse_win[1] * 100),
                )
            )
            # Set the style and color palette
            sns.set_style("whitegrid")
            sns.set_palette("Set2")
            # Determine the number of trials
            num_trials = result_hyp_param["trial"].nunique()
            # Create subplots
            fig, axes = plt.subplots(
                num_trials + 1,
                1,
                figsize=(12, 5 * (num_trials + 1)),
                gridspec_kw={"height_ratios": [3] * num_trials + [1]},
            )
            # NRMSE plots for each trial
            for i, (trial, ax) in enumerate(
                zip(result_hyp_param["trial"].unique(), axes[:-1])
            ):
                nrmse_data = result_hyp_param_long[
                    (result_hyp_param_long["metric_type"] == "nrmse")
                    & (result_hyp_param_long["trial"] == trial)
                ]
                sns.scatterplot(
                    data=nrmse_data,
                    x="iteration",
                    y="value",
                    hue="dataset",
                    style="dataset",
                    markers=["o", "s", "D"],  # Different markers for train, val, test
                    ax=ax,
                    alpha=0.6,
                )
                sns.lineplot(
                    data=nrmse_data,
                    x="iteration",
                    y="value",
                    hue="dataset",
                    ax=ax,
                    legend=False,
                    linewidth=1,
                )
                ax.set_ylabel(f"NRMSE [Trial {trial}]", fontsize=12, fontweight="bold")
                ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
                ax.legend(title="Dataset", loc="upper right")
            # Train Size plot
            sns.scatterplot(
                data=result_hyp_param,
                x="iteration",
                y="train_size",
                hue="trial",
                ax=axes[-1],
                legend=False,
            )
            axes[-1].set_ylabel("Train Size", fontsize=12, fontweight="bold")
            axes[-1].set_xlabel("Iteration", fontsize=12, fontweight="bold")
            axes[-1].set_ylim(0, 1)
            axes[-1].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
            )
            # Set the overall title
            plt.suptitle(
                "Time-series validation & Convergence", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            logger.info("Successfully created time-series validation plot")
            return self._convert_plot_to_base64(fig)
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

    def display_moo_distrb_plot(self):
        """Display the MOO Distribution Plot."""
        self._display_base64_image(self.moo_distrb_plot)

    def display_moo_cloud_plot(self):
        """Display the MOO Cloud Plot."""
        self._display_base64_image(self.moo_cloud_plot)

    def display_ts_validation_plot(self):
        """Display the Time-Series Validation Plot."""
        self._display_base64_image(self.ts_validation_plot)

    def _display_base64_image(self, base64_image: str):
        """Helper method to display a base64-encoded image."""
        display(Image(data=base64.b64decode(base64_image)))

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:

        logger.warning("this method is not yet implemented")
