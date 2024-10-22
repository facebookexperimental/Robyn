# pyre-strict
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from robyn.modeling.entities.modeloutputs import Trial
from robyn.visualization.model_convergence_visualizer import ModelConvergenceVisualizer


class Convergence:
    def __init__(self, n_cuts: int = 20, sd_qtref: int = 3, med_lowb: int = 2, nrmse_win: List[float] = [0, 0.998]):
        self.n_cuts = n_cuts
        self.sd_qtref = sd_qtref
        self.med_lowb = med_lowb
        self.nrmse_win = nrmse_win
        self.visualizer = ModelConvergenceVisualizer(n_cuts, nrmse_win)
        self.logger = logging.getLogger(__name__)

    def calculate_convergence(self, trials: List[Trial]) -> Dict[str, Any]:
        df = pd.concat([trial.result_hyp_param for trial in trials], ignore_index=True)

        required_columns = ["ElapsedAccum", "trial", "nrmse", "decomp.rssd", "mape"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in result_hyp_param: {missing_columns}")

        calibrated = "mape" in df.columns and df["mape"].sum() > 0
        if not calibrated:
            self.logger.warning("'mape' column not found or all zeros. Assuming model is not calibrated.")

        dt_objfunc_cvg = self._prepare_data(df)
        errors = self._calculate_errors(dt_objfunc_cvg)
        conv_msg = self._generate_convergence_messages(errors)

        moo_distrb_plot = self.visualizer.create_moo_distrb_plot(dt_objfunc_cvg, conv_msg)
        moo_cloud_plot = self.visualizer.create_moo_cloud_plot(df, conv_msg, calibrated)
        ts_validation_plot = self.visualizer.create_ts_validation_plot(trials)

        return {
            "moo_distrb_plot": moo_distrb_plot,
            "moo_cloud_plot": moo_cloud_plot,
            "ts_validation_plot": ts_validation_plot,
            "errors": errors,
            "conv_msg": conv_msg,
        }

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        value_vars = ["nrmse", "decomp.rssd"]
        if "mape" in df.columns and df["mape"].sum() > 0:
            value_vars.append("mape")

        dt_objfunc_cvg = df.melt(
            id_vars=["ElapsedAccum", "trial"], value_vars=value_vars, var_name="error_type", value_name="value"
        )
        dt_objfunc_cvg = dt_objfunc_cvg[dt_objfunc_cvg["value"] > 0]
        dt_objfunc_cvg = dt_objfunc_cvg[np.isfinite(dt_objfunc_cvg["value"])]
        dt_objfunc_cvg["error_type"] = dt_objfunc_cvg["error_type"].str.upper()
        dt_objfunc_cvg = dt_objfunc_cvg.sort_values(["trial", "ElapsedAccum"])
        dt_objfunc_cvg["iter"] = dt_objfunc_cvg.groupby(["error_type", "trial"]).cumcount() + 1

        max_iter = dt_objfunc_cvg["iter"].max()
        bins = np.linspace(0, max_iter, self.n_cuts + 1)
        labels = np.round(np.linspace(max_iter / self.n_cuts, max_iter, self.n_cuts)).astype(int)

        dt_objfunc_cvg["cuts"] = pd.cut(
            dt_objfunc_cvg["iter"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            ordered=False,
        )
        dt_objfunc_cvg["cuts"] = dt_objfunc_cvg["cuts"].astype(float)

        return dt_objfunc_cvg

    def _calculate_errors(self, dt_objfunc_cvg: pd.DataFrame) -> pd.DataFrame:
        errors = (
            dt_objfunc_cvg.groupby(["error_type", "cuts"], observed=True)
            .agg({"value": ["count", "median", "std"]})
            .reset_index()
        )
        errors.columns = ["error_type", "cuts", "n", "median", "std"]

        errors["med_var_P"] = (
            errors.groupby("error_type", observed=True)["median"].pct_change(fill_method=None).abs() * 100
        )
        errors["med_var_P"] = errors["med_var_P"].fillna(0)

        agg_errors = errors.groupby("error_type").agg(
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
        )
        agg_errors.columns = ["_".join(col).strip() for col in agg_errors.columns.values]
        agg_errors = agg_errors.reset_index()

        errors = errors.merge(agg_errors, on="error_type", how="left")

        errors["med_thres"] = errors["median_first_med"] - self.med_lowb * errors["std_first_sd_avg"]
        errors["flag_med"] = errors["median"].abs() < errors["med_thres"]
        errors["flag_sd"] = errors["std"] < errors["std_first_sd_avg"]

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
                f"sd@qt.{self.n_cuts} {last_qt['std']:.3f} {symb_sd} {last_qt['std_first_sd_avg']:.3f} & "
                f"|med@qt.{self.n_cuts}| {abs(last_qt['median']):.2f} {symb_med} {last_qt['med_thres']:.2f}"
            )
            conv_msg.append(msg)

        return conv_msg
