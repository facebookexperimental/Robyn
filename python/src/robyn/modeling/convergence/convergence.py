# convergence.py
# pyre-strict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import io
import base64


class Convergence:
    def __init__(self, n_cuts: int = 20, sd_qtref: int = 3, med_lowb: int = 2, nrmse_win: List[float] = [0, 0.998]):
        self.n_cuts = n_cuts
        self.sd_qtref = sd_qtref
        self.med_lowb = med_lowb
        self.nrmse_win = nrmse_win

    def calculate_convergence(self, trials: List[Any]) -> Dict[str, Any]:
        df = pd.concat([trial.result_hyp_param for trial in trials])
        calibrated = sum(df["mape"]) > 0

        dt_objfunc_cvg = self._prepare_data(df)
        errors = self._calculate_errors(dt_objfunc_cvg)
        conv_msg = self._generate_convergence_messages(errors)

        moo_distrb_plot = self._create_moo_distrb_plot(dt_objfunc_cvg, conv_msg)
        moo_cloud_plot = self._create_moo_cloud_plot(df, conv_msg, calibrated)

        return {
            "moo_distrb_plot": moo_distrb_plot,
            "moo_cloud_plot": moo_cloud_plot,
            "errors": errors,
            "conv_msg": conv_msg,
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_objfunc_cvg = df.melt(
            id_vars=["ElapsedAccum", "trial"],
            value_vars=["nrmse", "decomp.rssd", "mape"],
            var_name="error_type",
            value_name="value",
        )
        dt_objfunc_cvg = dt_objfunc_cvg[dt_objfunc_cvg["value"] > 0]
        dt_objfunc_cvg = dt_objfunc_cvg[np.isfinite(dt_objfunc_cvg["value"])]
        dt_objfunc_cvg["error_type"] = dt_objfunc_cvg["error_type"].str.upper()
        dt_objfunc_cvg = dt_objfunc_cvg.sort_values(["trial", "ElapsedAccum"])
        dt_objfunc_cvg["iter"] = dt_objfunc_cvg.groupby(["error_type", "trial"]).cumcount() + 1

        max_iter = dt_objfunc_cvg["iter"].max()
        bins = pd.cut(
            dt_objfunc_cvg["iter"],
            bins=np.linspace(0, max_iter, self.n_cuts + 1),
            labels=np.round(np.linspace(max_iter / self.n_cuts, max_iter, self.n_cuts)),
            include_lowest=True,
        )
        dt_objfunc_cvg["cuts"] = bins

        return dt_objfunc_cvg

    def _calculate_errors(self, dt_objfunc_cvg: pd.DataFrame) -> pd.DataFrame:
        errors = (
            dt_objfunc_cvg.groupby(["error_type", "cuts"]).agg({"value": ["count", "median", "std"]}).reset_index()
        )
        errors.columns = ["error_type", "cuts", "n", "median", "std"]

        errors["med_var_P"] = errors.groupby("error_type")["median"].pct_change().abs() * 100
        errors["med_var_P"] = errors["med_var_P"].fillna(0)

        errors = errors.merge(
            errors.groupby("error_type").agg(
                {
                    "median": [
                        ("first_med", "first"),
                        ("first_med_avg", lambda x: x.head(self.sd_qtref).mean()),
                        ("last_med", "last"),
                    ],
                    "std": [
                        ("first_sd", "first"),
                        ("first_sd_avg", lambda x: x.head(self.sd_qtref).mean()),
                        ("last_sd", "last"),
                    ],
                }
            ),
            on="error_type",
        )

        errors["med_thres"] = errors["median"]["first_med"] - self.med_lowb * errors["std"]["first_sd_avg"]
        errors["flag_med"] = errors["median"].abs() < errors["med_thres"]
        errors["flag_sd"] = errors["std"] < errors["std"]["first_sd_avg"]

        return errors

    def _generate_convergence_messages(self, errors: pd.DataFrame) -> List[str]:
        conv_msg = []
        for obj_fun in errors["error_type"].unique():
            temp_df = errors[errors["error_type"] == obj_fun].copy()
            temp_df["median"] = temp_df["median"].round(2)
            last_qt = temp_df.iloc[-1]

            did_converge = "converged" if last_qt["flag_sd"] and last_qt["flag_med"] else "NOT converged"
            symb_sd = "<=" if last_qt["flag_sd"] else ">"
            symb_med = "<=" if last_qt["flag_med"] else ">"

            msg = (
                f"{obj_fun} {did_converge}: "
                f"sd@qt.{self.n_cuts} {last_qt['std']:.2f} {symb_sd} {last_qt['std']['first_sd_avg']:.2f} & "
                f"|med@qt.{self.n_cuts}| {abs(last_qt['median']):.2f} {symb_med} {last_qt['med_thres']:.2f}"
            )
            conv_msg.append(msg)

        return conv_msg

    def _create_moo_distrb_plot(self, dt_objfunc_cvg: pd.DataFrame, conv_msg: List[str]) -> str:
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

        return self._convert_plot_to_base64(fig)

    def _create_moo_cloud_plot(self, df: pd.DataFrame, conv_msg: List[str], calibrated: bool) -> str:
        df["nrmse"] = np.clip(df["nrmse"], *np.quantile(df["nrmse"], self.nrmse_win))

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df["nrmse"], df["decomp.rssd"], c=df["ElapsedAccum"], cmap="viridis")

        if calibrated:
            sizes = (df["mape"] - df["mape"].min()) / (df["mape"].max() - df["mape"].min())
            sizes = sizes * 100 + 10  # Scale sizes
            ax.scatter(df["nrmse"], df["decomp.rssd"], s=sizes, alpha=0.5)

        plt.colorbar(scatter, label="Time [s]")
        ax.set_xlabel("NRMSE")
        ax.set_ylabel("DECOMP.RSSD")
        ax.set_title("Multi-objective evolutionary performance")
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
