import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import io
import base64
from robyn.modeling.entities.modeloutputs import Trial


class ModelConvergenceVisualizer:
    def __init__(self, n_cuts: int, nrmse_win: List[float]):
        self.n_cuts = n_cuts
        self.nrmse_win = nrmse_win

    def create_moo_distrb_plot(self, dt_objfunc_cvg: pd.DataFrame, conv_msg: List[str]) -> str:
        dt_objfunc_cvg["id"] = dt_objfunc_cvg["cuts"].astype(int)
        dt_objfunc_cvg["cuts"] = pd.Categorical(
            dt_objfunc_cvg["cuts"], categories=sorted(dt_objfunc_cvg["cuts"].unique(), reverse=True)
        )

        for error_type in dt_objfunc_cvg["error_type"].unique():
            mask = dt_objfunc_cvg["error_type"] == error_type
            dt_objfunc_cvg.loc[mask, "value"] = np.clip(
                dt_objfunc_cvg.loc[mask, "value"], *np.quantile(dt_objfunc_cvg.loc[mask, "value"], self.nrmse_win)
            )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(data=dt_objfunc_cvg, x="value", y="cuts", hue="error_type", split=True, inner="quartile", ax=ax)
        ax.set_xlabel("Objective functions")
        ax.set_ylabel("Iterations [#]")
        ax.set_title("Objective convergence by iterations quantiles")
        plt.tight_layout()

        # Add convergence messages as caption
        plt.figtext(0.5, 0.01, "\n".join(conv_msg), ha="center", fontsize=8, wrap=True)

        return self._convert_plot_to_base64(fig)

    def create_moo_cloud_plot(self, df: pd.DataFrame, conv_msg: List[str], calibrated: bool) -> str:
        df["nrmse"] = np.clip(df["nrmse"], *np.quantile(df["nrmse"], self.nrmse_win))

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df["nrmse"], df["decomp.rssd"], c=df["ElapsedAccum"], cmap="viridis")

        if calibrated and "mape" in df.columns:
            sizes = (df["mape"] - df["mape"].min()) / (df["mape"].max() - df["mape"].min())
            sizes = sizes * 100 + 10  # Scale sizes
            ax.scatter(df["nrmse"], df["decomp.rssd"], s=sizes, alpha=0.5)

        plt.colorbar(scatter, label="Time [s]")
        ax.set_xlabel("NRMSE")
        ax.set_ylabel("DECOMP.RSSD")
        ax.set_title("Multi-objective evolutionary performance")

        # Add convergence messages as caption
        plt.figtext(0.5, 0.01, "\n".join(conv_msg), ha="center", fontsize=8, wrap=True)

        plt.tight_layout()

        return self._convert_plot_to_base64(fig)

    @staticmethod
    def _convert_plot_to_base64(fig: plt.Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png)
        return graphic.decode("utf-8")

    def create_ts_validation_plot(self, output_models: List[Trial]) -> str:
        result_hyp_param = pd.concat([trial.result_hyp_param for trial in output_models], ignore_index=True)

        result_hyp_param_long = result_hyp_param.melt(
            id_vars=["solID", "trial", "train_size"],
            value_vars=["rsq_train", "rsq_val", "rsq_test"],
            var_name="dataset",
            value_name="rsq",
        )

        nrmse_data = result_hyp_param.melt(
            id_vars=["solID"],
            value_vars=["nrmse_train", "nrmse_val", "nrmse_test"],
            var_name="nrmse_dataset",
            value_name="nrmse_value",
        )

        result_hyp_param_long = result_hyp_param_long.merge(
            nrmse_data, left_on=["solID", "dataset"], right_on=["solID", "nrmse_dataset"], how="left"
        )

        result_hyp_param_long["dataset"] = result_hyp_param_long["dataset"].str.replace("rsq_", "")
        result_hyp_param_long["rsq"] = np.clip(result_hyp_param_long["rsq"], 0.01, 0.99)
        result_hyp_param_long["nrmse_value"] = np.clip(
            result_hyp_param_long["nrmse_value"], 0, np.percentile(result_hyp_param_long["nrmse_value"], 99)
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[3, 1])

        # NRMSE plot
        sns.scatterplot(data=result_hyp_param_long, x="trial", y="nrmse_value", hue="dataset", alpha=0.2, ax=ax1)
        sns.lineplot(data=result_hyp_param_long, x="trial", y="nrmse_value", hue="dataset", ax=ax1)
        ax1.set_ylabel("NRMSE [Upper 1% Winsorized]")
        ax1.set_xlabel("Trial")
        ax1.legend(title="Dataset")

        # Train Size plot
        sns.scatterplot(data=result_hyp_param, x="trial", y="train_size", ax=ax2, color="black", alpha=0.5)
        ax2.set_ylabel("Train Size")
        ax2.set_xlabel("Trial")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

        plt.suptitle("Time-series validation & Convergence")
        plt.tight_layout()
