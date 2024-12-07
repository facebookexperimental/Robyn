# pyre-strict
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from robyn.modeling.entities.modeloutputs import Trial
from robyn.visualization.model_convergence_visualizer import ModelConvergenceVisualizer
import matplotlib.pyplot as plt


class Convergence:
    def __init__(
        self,
        n_cuts: int = 20,
        sd_qtref: int = 3,
        med_lowb: int = 2,
        nrmse_win: List[float] = [0, 0.998],
    ):
        self.n_cuts = n_cuts
        self.sd_qtref = sd_qtref
        self.med_lowb = med_lowb
        self.nrmse_win = nrmse_win
        self.visualizer = ModelConvergenceVisualizer(n_cuts, nrmse_win)
        self.logger = logging.getLogger(__name__)

    def calculate_convergence(self, trials: List[Trial]) -> Dict[str, Any]:
        self.logger.info("Starting convergence calculation")
        self.logger.debug(f"Processing {len(trials)} trials")

        try:
            # Concatenate trial results
            self.logger.debug("Concatenating trial results")
            df = pd.concat(
                [trial.result_hyp_param for trial in trials], ignore_index=True
            )
            self.logger.debug(f"Combined dataframe shape: {df.shape}")

            # Validate required columns
            required_columns = ["ElapsedAccum", "trial", "nrmse", "decomp.rssd", "mape"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing columns in result_hyp_param: {missing_columns}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Check calibration status
            calibrated = "mape" in df.columns and df["mape"].sum() > 0
            if not calibrated:
                self.logger.warning(
                    "'mape' column not found or all zeros. Assuming model is not calibrated."
                )
            else:
                self.logger.info("Model is calibrated")

            # Process data and calculate errors
            self.logger.debug("Preparing data for convergence analysis")
            dt_objfunc_cvg = self._prepare_data(df)
            self.logger.debug(f"Prepared data shape: {dt_objfunc_cvg.shape}")

            self.logger.debug("Calculating convergence errors")
            errors = self._calculate_errors(dt_objfunc_cvg)
            self.logger.debug(
                f"Generated errors for {len(errors['error_type'].unique())} error types"
            )

            # Generate convergence messages
            self.logger.debug("Generating convergence messages")
            conv_msg = self._generate_convergence_messages(errors)

            # Create visualization plots

            moo_distrb_plot = self.visualizer.create_moo_distrb_plot(
                dt_objfunc_cvg, conv_msg
            )
            moo_cloud_plot = self.visualizer.create_moo_cloud_plot(
                df, conv_msg, calibrated
            )
            ts_validation_plot = None  # self.visualizer.create_ts_validation_plot(trials) #Disabled for testing. #Sandeep

            self.logger.info("Convergence calculation completed successfully")
            return {
                "moo_distrb_plot": moo_distrb_plot,
                "moo_cloud_plot": moo_cloud_plot,
                "ts_validation_plot": ts_validation_plot,
                "errors": errors,
                "conv_msg": conv_msg,
            }

        except Exception as e:
            self.logger.error(
                f"Error during convergence calculation: {str(e)}", exc_info=True
            )
            raise

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Starting data preparation")

        # Determine value variables
        value_vars = ["nrmse", "decomp.rssd"]
        if "mape" in df.columns and df["mape"].sum() > 0:
            value_vars.append("mape")
            self.logger.debug("Including MAPE in value variables")

        self.logger.debug(f"Melting dataframe with value variables: {value_vars}")
        dt_objfunc_cvg = df.melt(
            id_vars=["ElapsedAccum", "trial"],
            value_vars=value_vars,
            var_name="error_type",
            value_name="value",
        )

        # Filter and process data
        initial_rows = len(dt_objfunc_cvg)
        dt_objfunc_cvg = dt_objfunc_cvg[dt_objfunc_cvg["value"] > 0]
        dt_objfunc_cvg = dt_objfunc_cvg[np.isfinite(dt_objfunc_cvg["value"])]
        filtered_rows = len(dt_objfunc_cvg)
        self.logger.debug(
            f"Filtered out {initial_rows - filtered_rows} rows with zero or non-finite values"
        )

        # Process error types and iterations
        dt_objfunc_cvg["error_type"] = dt_objfunc_cvg["error_type"].str.upper()
        dt_objfunc_cvg = dt_objfunc_cvg.sort_values(["trial", "ElapsedAccum"])
        dt_objfunc_cvg["iter"] = (
            dt_objfunc_cvg.groupby(["error_type", "trial"]).cumcount() + 1
        )

        # Create cuts
        max_iter = dt_objfunc_cvg["iter"].max()
        self.logger.debug(
            f"Creating {self.n_cuts} cuts for maximum iteration {max_iter}"
        )
        bins = np.linspace(0, max_iter, self.n_cuts + 1)
        labels = np.round(
            np.linspace(max_iter / self.n_cuts, max_iter, self.n_cuts)
        ).astype(int)

        dt_objfunc_cvg["cuts"] = pd.cut(
            dt_objfunc_cvg["iter"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            ordered=False,
        )
        dt_objfunc_cvg["cuts"] = dt_objfunc_cvg["cuts"].astype(float)

        self.logger.debug("Data preparation completed")
        return dt_objfunc_cvg

    def _calculate_errors(self, dt_objfunc_cvg: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Starting error calculations")

        # Calculate basic statistics
        self.logger.debug("Calculating error statistics by group")
        errors = (
            dt_objfunc_cvg.groupby(["error_type", "cuts"], observed=True)
            .agg({"value": ["count", "median", "std"]})
            .reset_index()
        )
        errors.columns = ["error_type", "cuts", "n", "median", "std"]

        # Calculate median variation percentage
        self.logger.debug("Calculating median variation percentage")
        errors["med_var_P"] = (
            errors.groupby("error_type", observed=True)["median"]
            .pct_change(fill_method=None)
            .abs()
            * 100
        )
        errors["med_var_P"] = errors["med_var_P"].fillna(0)

        # Calculate aggregate statistics
        self.logger.debug(
            f"Calculating aggregate statistics with sd_qtref={self.sd_qtref}"
        )
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
        agg_errors.columns = [
            "_".join(col).strip() for col in agg_errors.columns.values
        ]
        agg_errors = agg_errors.reset_index()

        # Merge and calculate flags
        self.logger.debug(
            "Merging aggregate statistics and calculating threshold flags"
        )
        errors = errors.merge(agg_errors, on="error_type", how="left")
        errors["med_thres"] = (
            errors["median_first_med"] - self.med_lowb * errors["std_first_sd_avg"]
        )
        errors["flag_med"] = errors["median"].abs() < errors["med_thres"]
        errors["flag_sd"] = errors["std"] < errors["std_first_sd_avg"]

        self.logger.debug("Error calculations completed")
        return errors

    def _generate_convergence_messages(self, errors: pd.DataFrame) -> List[str]:
        self.logger.debug("Starting convergence message generation")
        conv_msg = []

        for obj_fun in errors["error_type"].unique():
            self.logger.debug(f"Processing convergence message for {obj_fun}")
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
                f"{obj_fun} {did_converge}: "
                f"sd@qt.{self.n_cuts} {last_qt['std']:.3f} {symb_sd} {last_qt['std_first_sd_avg']:.3f} & "
                f"|med@qt.{self.n_cuts}| {abs(last_qt['median']):.2f} {symb_med} {last_qt['med_thres']:.2f}"
            )
            conv_msg.append(msg)

            # Log convergence status
            log_level = logging.INFO if did_converge == "converged" else logging.WARNING
            self.logger.log(
                log_level, f"Convergence status for {obj_fun}: {did_converge}"
            )
            self.logger.info(msg)  # Log the message as INFO
        self.logger.debug("Convergence message generation completed")
        return conv_msg
