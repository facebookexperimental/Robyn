# pyre-strict

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from robyn.modeling.entities.convergence_result import ConvergenceResult
from robyn.modeling.entities.modeloutput import ModelOutput, ResultHypParam


class ModelConvergence:
    """
    ModelConvergence class to analyze the convergence of model outputs.

    Methods:
        converge: Main method to run convergence analysis.
    """

    def __init__(self) -> None:
        pass

    def converge(
        self,
        model_output: ModelOutput,
        n_cuts: int = 20,
        sd_qtref: int = 3,
        med_lowb: int = 2,
        nrmse_win: Tuple[float, float] = (0, 0.998),
        **kwargs: Any,
    ) -> ConvergenceResult:
        """
        Main method to run convergence analysis.

        :param model_output: ModelOutput object containing the model outputs.
        :param n_cuts: Number of cuts for convergence analysis.
        :param sd_qtref: Standard deviation quantile reference.
        :param med_lowb: Median lower bound.
        :param nrmse_win: Normalized RMSE window.
        :param kwargs: Additional arguments for convergence analysis.
        :return: ConvergenceResult containing convergence analysis results.
        """
        assert (
            n_cuts > min(sd_qtref, med_lowb) + 1
        ), "n_cuts must be greater than min(sd_qtref, med_lowb) + 1"

        # Gather all trials
        print("Gathering all trials...", model_output.trials)
        print(type(model_output))
        print(type(model_output.metadata))
        print(model_output.metadata)
        df = pd.DataFrame([vars(trial) for trial in model_output.trials])
        calibrated = df["mape"].sum() > 0

        # Calculate deciles
        dt_objfunc_cvg = df.melt(
            id_vars=["ElapsedAccum", "trial"],
            value_vars=["nrmse", "decomp_rssd", "mape"],
            var_name="error_type",
            value_name="value",
        )
        dt_objfunc_cvg = dt_objfunc_cvg[
            (dt_objfunc_cvg["value"] > 0) & (np.isfinite(dt_objfunc_cvg["value"]))
        ]
        dt_objfunc_cvg["error_type"] = dt_objfunc_cvg["error_type"].str.upper()
        dt_objfunc_cvg = dt_objfunc_cvg.sort_values(["trial", "ElapsedAccum"])
        dt_objfunc_cvg["iter"] = (
            dt_objfunc_cvg.groupby(["error_type", "trial"]).cumcount() + 1
        )
        dt_objfunc_cvg["cuts"] = pd.cut(
            dt_objfunc_cvg["iter"], bins=n_cuts, labels=False
        )

        # Calculate standard deviations and absolute medians on each cut
        errors = (
            dt_objfunc_cvg.groupby(["error_type", "cuts"])
            .agg({"value": ["count", "median", "std"]})
            .reset_index()
        )
        errors.columns = ["error_type", "cuts", "n", "median", "std"]

        errors = errors.sort_values(["error_type", "cuts"])
        errors["med_var_P"] = (
            errors.groupby("error_type")["median"].pct_change().abs() * 100
        )

        errors["first_med"] = (
            errors.groupby("error_type")["median"].transform("first").abs()
        )
        errors["first_med_avg"] = (
            errors.groupby("error_type")["median"]
            .transform(lambda x: x.head(sd_qtref).mean())
            .abs()
        )
        errors["last_med"] = (
            errors.groupby("error_type")["median"].transform("last").abs()
        )
        errors["first_sd"] = errors.groupby("error_type")["std"].transform("first")
        errors["first_sd_avg"] = errors.groupby("error_type")["std"].transform(
            lambda x: x.head(sd_qtref).mean()
        )
        errors["last_sd"] = errors.groupby("error_type")["std"].transform("last")

        errors["med_thres"] = errors["first_med"] - med_lowb * errors["first_sd_avg"]
        errors["flag_med"] = errors["median"].abs() < errors["med_thres"]
        errors["flag_sd"] = errors["std"] < errors["first_sd_avg"]

        # Generate convergence messages
        conv_msg: List[str] = []
        for obj_fun in errors["error_type"].unique():
            temp_df = errors[errors["error_type"] == obj_fun].copy()
            temp_df["median"] = temp_df["median"].round(2)
            last_qt = temp_df.iloc[-1]
            did_converge = (
                "converged"
                if last_qt["flag_sd"] and last_qt["flag_med"]
                else "NOT converged"
            )
            symb_sd = "<=" if last_qt["flag_sd"] else ">"
            symb_med = "<=" if last_qt["flag_med"] else ">"
            msg = (
                f"{obj_fun} {did_converge}: sd@qt.{n_cuts} {last_qt['last_sd']:.2f} {symb_sd} {last_qt['first_sd_avg']:.2f} & "
                f"|med@qt.{n_cuts}| {last_qt['last_med']:.2f} {symb_med} {last_qt['med_thres']:.2f}"
            )
            conv_msg.append(msg)
            print(f"- {msg}")

        # Generate plots
        moo_distrb_plot = self._generate_moo_distrb_plot(
            dt_objfunc_cvg, n_cuts, model_output, nrmse_win, conv_msg
        )
        moo_cloud_plot = self._generate_moo_cloud_plot(
            df, model_output, nrmse_win, calibrated, conv_msg
        )

        return ConvergenceResult(
            moo_distrb_plot=moo_distrb_plot,
            moo_cloud_plot=moo_cloud_plot,
            errors=errors,
            conv_msg="\n".join(conv_msg),  # Join the list of messages into a single string
            sd_qtref=sd_qtref,
            med_lowb=med_lowb
        )

    def _generate_moo_distrb_plot(
        self,
        dt_objfunc_cvg: pd.DataFrame,
        n_cuts: int,
        model_output: ModelOutput,
        nrmse_win: Tuple[float, float],
        conv_msg: List[str],
    ) -> plt.Figure:
        print("debug", type(model_output.metadata))
        print("debug", model_output.metadata)

        # Use default values if metadata doesn't contain the required information
        trials = model_output.metadata.get("trials", len(model_output.trials))
        iterations = model_output.metadata.get("iterations", "unknown")
        nevergrad_algo = model_output.metadata.get(
            "nevergrad_algo", "unknown algorithm"
        )

        subtitle = (
            f"{trials} trials with {iterations} iterations each using {nevergrad_algo}"
        )

        # Convert 'cuts' to numeric type
        dt_objfunc_cvg["cuts"] = pd.to_numeric(dt_objfunc_cvg["cuts"])

        fig, axes = plt.subplots(
            1, len(dt_objfunc_cvg["error_type"].unique()), figsize=(15, 5)
        )
        for i, error_type in enumerate(sorted(dt_objfunc_cvg["error_type"].unique())):
            data = dt_objfunc_cvg[dt_objfunc_cvg["error_type"] == error_type]
            sns.kdeplot(
                data=data,
                x="value",
                y="cuts",
                fill=True,
                cmap="YlGnBu",
                clip=nrmse_win,
                ax=axes[i],
            )
            axes[i].set_title(error_type)
            axes[i].set_xlabel("Objective function")
            axes[i].set_ylabel("Iterations [#]")

        plt.suptitle("Objective convergence by iterations quantiles")
        plt.tight_layout()
        fig.text(0.5, 0.01, subtitle, ha="center")
        fig.text(0.5, -0.05, "\n".join(conv_msg), ha="center", va="center", fontsize=8)

        return fig

    def _generate_moo_cloud_plot(
        self,
        df: pd.DataFrame,
        model_output: ModelOutput,
        nrmse_win: Tuple[float, float],
        calibrated: bool,
        conv_msg: List[str],
    ) -> plt.Figure:
        print("debug", type(model_output.metadata))

        # Use default values if metadata doesn't contain the required information
        trials = model_output.metadata.get("trials", len(model_output.trials))
        iterations = model_output.metadata.get("iterations", "unknown")
        nevergrad_algo = model_output.metadata.get(
            "nevergrad_algo", "unknown algorithm"
        )

        subtitle = (
            f"{trials} trials with {iterations} iterations each using {nevergrad_algo}"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df["nrmse"],
            df["decomp_rssd"],
            c=df["ElapsedAccum"],
            cmap="viridis",
            alpha=0.6,
        )

        if calibrated:
            scatter.set_sizes(df["mape"] * 10)  # Adjust size based on MAPE

        plt.colorbar(scatter, label="Time [s]")
        ax.set_xlabel(f"NRMSE [Winsorized {nrmse_win[0]}-{nrmse_win[1]}]")
        ax.set_ylabel("DECOMP.RSSD")
        ax.set_title(
            "Multi-objective evolutionary performance"
            + (" with calibration" if calibrated else "")
        )
        plt.suptitle(subtitle)
        fig.text(0.5, -0.05, "\n".join(conv_msg), ha="center", va="center", fontsize=8)

        return fig
