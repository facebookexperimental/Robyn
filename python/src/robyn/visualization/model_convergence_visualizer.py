import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import Image, display

matplotlib.use("Agg")
import seaborn as sns
from typing import List, Optional
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

    def create_moo_distrb_plot(self, dt_objfunc_cvg: pd.DataFrame, conv_msg: List[str]) -> str:
        logger.debug("Starting moo distribution plot creation with data shape: %s", dt_objfunc_cvg.shape)

        try:
            dt_objfunc_cvg["id"] = dt_objfunc_cvg["cuts"].astype(int)
            dt_objfunc_cvg["cuts"] = pd.Categorical(
                dt_objfunc_cvg["cuts"], categories=sorted(dt_objfunc_cvg["cuts"].unique(), reverse=True)
            )

            logger.debug("Processing error types: %s", dt_objfunc_cvg["error_type"].unique())
            for error_type in dt_objfunc_cvg["error_type"].unique():
                mask = dt_objfunc_cvg["error_type"] == error_type
                original_values = dt_objfunc_cvg.loc[mask, "value"]
                quantiles = np.quantile(original_values, self.nrmse_win)
                dt_objfunc_cvg.loc[mask, "value"] = np.clip(original_values, *quantiles)
                logger.debug(
                    "Clipped values for error_type %s: min=%f, max=%f", error_type, quantiles[0], quantiles[1]
                )

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.violinplot(
                data=dt_objfunc_cvg, x="value", y="cuts", hue="error_type", split=True, inner="quartile", ax=ax
            )
            ax.set_xlabel("Objective functions")
            ax.set_ylabel("Iterations [#]")
            ax.set_title("Objective convergence by iterations quantiles")
            plt.tight_layout()

            plt.figtext(0.5, 0.01, "\n".join(conv_msg), ha="center", fontsize=8, wrap=True)

            logger.info("Successfully created moo distribution plot")
            return self._convert_plot_to_base64(fig)

        except Exception as e:
            logger.error("Failed to create moo distribution plot: %s", str(e), exc_info=True)
            raise

    def create_moo_cloud_plot(self, df: pd.DataFrame, conv_msg: List[str], calibrated: bool) -> str:
        logger.debug("Starting moo cloud plot creation with data shape: %s, calibrated=%s", df.shape, calibrated)

        try:
            original_nrmse = df["nrmse"]
            quantiles = np.quantile(original_nrmse, self.nrmse_win)
            df["nrmse"] = np.clip(original_nrmse, *quantiles)
            logger.debug("Clipped NRMSE values: min=%f, max=%f", quantiles[0], quantiles[1])

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(df["nrmse"], df["decomp.rssd"], c=df["ElapsedAccum"], cmap="viridis")

            if calibrated and "mape" in df.columns:
                logger.debug("Adding calibrated MAPE visualization")
                sizes = (df["mape"] - df["mape"].min()) / (df["mape"].max() - df["mape"].min())
                sizes = sizes * 100 + 10
                ax.scatter(df["nrmse"], df["decomp.rssd"], s=sizes, alpha=0.5)

            plt.colorbar(scatter, label="Time [s]")
            ax.set_xlabel("NRMSE")
            ax.set_ylabel("DECOMP.RSSD")
            ax.set_title("Multi-objective evolutionary performance")

            plt.figtext(0.5, 0.01, "\n".join(conv_msg), ha="center", fontsize=8, wrap=True)
            plt.tight_layout()

            logger.info("Successfully created moo cloud plot")
            return self._convert_plot_to_base64(fig)

        except Exception as e:
            logger.error("Failed to create moo cloud plot: %s", str(e), exc_info=True)
            raise

    def create_ts_validation_plot(self, trials: List[Trial]) -> str:
        logger.debug("Starting time-series validation plot creation with %d trials", len(trials))

        try:
            result_hyp_param = pd.concat([trial.result_hyp_param for trial in trials], ignore_index=True)
            result_hyp_param["trial"] = result_hyp_param.groupby("solID").cumcount() + 1
            result_hyp_param["iteration"] = result_hyp_param.index + 1

            logger.debug("Processing metrics for validation plot")
            result_hyp_param_long = result_hyp_param.melt(
                id_vars=["solID", "trial", "train_size", "iteration"],
                value_vars=["rsq_train", "rsq_val", "rsq_test", "nrmse_train", "nrmse_val", "nrmse_test"],
                var_name="metric",
                value_name="value",
            )

            result_hyp_param_long["dataset"] = result_hyp_param_long["metric"].str.split("_").str[-1]
            result_hyp_param_long["metric_type"] = result_hyp_param_long["metric"].str.split("_").str[0]

            # Winsorize the data
            logger.debug("Winsorizing metric values")
            result_hyp_param_long["value"] = result_hyp_param_long.groupby("metric_type")["value"].transform(
                lambda x: np.clip(
                    x, np.percentile(x, self.nrmse_win[0] * 100), np.percentile(x, self.nrmse_win[1] * 100)
                )
            )

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])

            # NRMSE plot
            sns.scatterplot(
                data=result_hyp_param_long[result_hyp_param_long["metric_type"] == "nrmse"],
                x="iteration",
                y="value",
                hue="dataset",
                style="trial",
                alpha=0.5,
                ax=ax1,
            )
            sns.lineplot(
                data=result_hyp_param_long[result_hyp_param_long["metric_type"] == "nrmse"],
                x="iteration",
                y="value",
                hue="dataset",
                ax=ax1,
            )
            ax1.set_ylabel("NRMSE [Winsorized]")
            ax1.set_xlabel("Iteration")
            ax1.legend(title="Dataset")

            # Train Size plot
            sns.scatterplot(data=result_hyp_param, x="iteration", y="train_size", hue="trial", ax=ax2, legend=False)
            ax2.set_ylabel("Train Size")
            ax2.set_xlabel("Iteration")
            ax2.set_ylim(0, 1)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

            plt.suptitle("Time-series validation & Convergence")
            plt.tight_layout()

            logger.info("Successfully created time-series validation plot")
            return self._convert_plot_to_base64(fig)

        except Exception as e:
            logger.error("Failed to create time-series validation plot: %s", str(e), exc_info=True)
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
