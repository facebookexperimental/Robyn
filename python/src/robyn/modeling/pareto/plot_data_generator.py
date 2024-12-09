# pyre-strict
import logging
from typing import Dict, Union

import pandas as pd
import numpy as np
from robyn.common.logger import RobynLogger
from robyn.data.entities.enums import AdstockType, DependentVarType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData
from robyn.modeling.entities.pareto_data import ParetoData
from robyn.modeling.transformations.transformations import Transformation
from tqdm import tqdm


class PlotDataGenerator:

    def __init__(
        self,
        mmm_data: MMMData,
        hyperparameter: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
        holidays_data: HolidaysData,
    ):
        self.mmm_data = mmm_data
        self.hyperparameter = hyperparameter
        self.featurized_mmm_data = featurized_mmm_data
        self.holidays_data = holidays_data
        self.transformer = Transformation(mmm_data)

        # Setup logger with a single handler
        self.logger = logging.getLogger(__name__)

    def generate_plot_data(
        self,
        aggregated_data: Dict[str, pd.DataFrame],
        pareto_data: ParetoData,
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for various plots used in the Pareto analysis.
        """
        mediaVecCollect = pd.DataFrame()
        xDecompVecCollect = pd.DataFrame()
        plotDataCollect = {}
        df_caov_pct_all = pd.DataFrame()

        xDecompAgg = pareto_data.x_decomp_agg
        dt_mod = self.featurized_mmm_data.dt_mod
        dt_modRollWind = self.featurized_mmm_data.dt_modRollWind
        rw_start_loc = self.mmm_data.mmmdata_spec.rolling_window_start_which
        rw_end_loc = self.mmm_data.mmmdata_spec.rolling_window_end_which

        self.logger.info("Starting plot data generation...")
        self.logger.debug(
            f"Available columns in xDecompAgg: {xDecompAgg.columns.tolist()}"
        )

        pareto_fronts_vec = pareto_data.pareto_fronts

        for pf in pareto_fronts_vec:
            self.logger.info(f"Processing Pareto front {pf}")
            plotMediaShare = xDecompAgg[
                (xDecompAgg["robynPareto"] == pf)
                & (xDecompAgg["rn"].isin(self.mmm_data.mmmdata_spec.paid_media_spends))
            ]
            self.logger.debug(f"Shape of plotMediaShare: {plotMediaShare.shape}")

            uniqueSol = plotMediaShare["sol_id"].unique()
            plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]

            self.logger.info(f"Pareto-Front: {pf} [{len(uniqueSol)} models]")

            for sid in tqdm(uniqueSol, desc="Processing Solutions", unit="solution"):
                try:
                    plot_results = self._process_single_solution(
                        sid,
                        plotMediaShare,
                        plotWaterfall,
                        pareto_data,
                        aggregated_data,
                        dt_mod,
                        dt_modRollWind,
                        rw_start_loc,
                        rw_end_loc,
                    )

                    mediaVecCollect = pd.concat(
                        [mediaVecCollect, plot_results["mediaVecCollect"]],
                        ignore_index=True,
                    )
                    xDecompVecCollect = pd.concat(
                        [xDecompVecCollect, plot_results["xDecompVec"]],
                        ignore_index=True,
                    )
                    plotDataCollect[sid] = plot_results["plotData"]
                    df_caov_pct_all = pd.concat(
                        [df_caov_pct_all, plot_results["plot7data"]]
                    )

                except Exception as e:
                    self.logger.error(f"Error processing solution {sid}: {str(e)}")
                    raise e

        pareto_solutions = set()
        if "sol_id" in xDecompVecCollect.columns:
            pareto_solutions.update(set(xDecompVecCollect["sol_id"].unique()))

        self.logger.debug(
            f"Found ({len(pareto_solutions)}) Pareto Solutions - {pareto_solutions}"
        )
        RobynLogger.log_df(self.logger, mediaVecCollect)
        RobynLogger.log_df(self.logger, xDecompVecCollect)
        RobynLogger.log_df(self.logger, df_caov_pct_all)

        self.logger.info("Plot data generated Successfully.")
        return {
            "pareto_solutions": list(pareto_solutions),
            "mediaVecCollect": mediaVecCollect,
            "xDecompVecCollect": xDecompVecCollect,
            "plotDataCollect": plotDataCollect,
            "df_caov_pct_all": df_caov_pct_all,
        }

    def _generate_spend_effect_data(self, plotMediaShare: pd.DataFrame, sid: str):
        temp = plotMediaShare[plotMediaShare["sol_id"] == sid].melt(
            id_vars=["rn", "nrmse", "decomp.rssd", "rsq_train"],
            value_vars=["spend_share", "effect_share", "roi_total", "cpa_total"],
            var_name="variable",
            value_name="value",
        )
        temp["rn"] = pd.Categorical(
            temp["rn"],
            categories=sorted(self.mmm_data.mmmdata_spec.paid_media_spends),
            ordered=True,
        )

        plotMediaShareLoopBar = temp[
            temp["variable"].isin(["spend_share", "effect_share"])
        ]
        plotMediaShareLoopLine = temp[
            temp["variable"]
            == (
                "cpa_total"
                if self.mmm_data.mmmdata_spec.dep_var_type
                == DependentVarType.CONVERSION
                else "roi_total"
            )
        ]

        line_rm_inf = ~np.isinf(plotMediaShareLoopLine["value"])
        ySecScale = (
            max(plotMediaShareLoopLine["value"][line_rm_inf])
            / max(plotMediaShareLoopBar["value"])
            * 1.1
        )

        return {
            "plotMediaShareLoopBar": plotMediaShareLoopBar,
            "plotMediaShareLoopLine": plotMediaShareLoopLine,
            "ySecScale": ySecScale,
        }

    def _process_single_solution(
        self,
        sid: str,
        plotMediaShare: pd.DataFrame,
        plotWaterfall: pd.DataFrame,
        pareto_data: ParetoData,
        aggregated_data: Dict[str, pd.DataFrame],
        dt_mod: pd.DataFrame,
        dt_modRollWind: pd.DataFrame,
        rw_start_loc: int,
        rw_end_loc: int,
    ) -> Dict:
        # 1. Spend x effect share comparison
        plot1data = self._generate_spend_effect_data(plotMediaShare, sid)
        self.logger.debug(f"Generated plot1data, spend vs effect data, for sid: {sid}")

        # 2. Waterfall
        plotWaterfallLoop = self._generate_waterfall_data(plotWaterfall, sid)
        plot2data = {"plotWaterfallLoop": plotWaterfallLoop}
        self.logger.debug(f"Generated plot2data, waterfall data, for sid: {sid}")

        # 3. Adstock rate
        plot3data = self._generate_adstock_data(sid, pareto_data)
        self.logger.debug(f"Generated plot3data, adstock plot data, for sid: {sid}")

        # 4. Spend response curve
        plot4data = self._generate_response_data(
            sid, dt_mod, plotWaterfall, plotWaterfallLoop, rw_start_loc, rw_end_loc
        )

        dt_transformPlot = plot4data["dt_transformPlot"]
        dt_transformSpend = plot4data["dt_transformSpend"]
        dt_transformSpendMod = plot4data["dt_transformSpendMod"]
        dt_transformAdstock = plot4data["dt_transformAdstock"]
        dt_transformSaturationSpendReverse = plot4data[
            "dt_transformSaturationSpendReverse"
        ]
        dt_transformSaturationDecomp = plot4data["dt_transformSaturationDecomp"]

        plot4data = {
            "dt_scurvePlot": plot4data["dt_scurvePlot"],
            "dt_scurvePlotMean": plot4data["dt_scurvePlotMean"],
        }
        self.logger.debug(f"Generated plot4data, scurve plot data, for sid: {sid}")

        # 5. Fitted vs actual
        col_order = (
            ["ds", "dep_var"]
            + self.mmm_data.mmmdata_spec.all_media
            + [
                self._get_prophet_var_values(var)
                for var in self.holidays_data.prophet_vars
            ]
            + self.mmm_data.mmmdata_spec.context_vars
        )

        selected_columns = (
            ["ds", "dep_var"]
            + [
                self._get_prophet_var_values(var)
                for var in self.holidays_data.prophet_vars
            ]
            + self.mmm_data.mmmdata_spec.context_vars
        )

        dt_transformDecomp = dt_modRollWind[selected_columns]
        dt_transformDecomp = pd.concat(
            [
                dt_transformDecomp,
                self.dt_transformSaturation[self.mmm_data.mmmdata_spec.all_media],
            ],
            axis=1,
        )
        dt_transformDecomp = dt_transformDecomp[col_order]

        xDecompVec = (
            plotWaterfall[plotWaterfall["sol_id"] == sid][["sol_id", "rn", "coef"]]
            .pivot(index="sol_id", columns="rn", values="coef")
            .reset_index()
        )

        if "(Intercept)" not in xDecompVec.columns:
            xDecompVec["(Intercept)"] = 0

        xDecompVec = xDecompVec[
            ["sol_id", "(Intercept)"]
            + [col for col in col_order if col not in ["ds", "dep_var"]]
        ]
        intercept = xDecompVec["(Intercept)"].values[0]

        scurved = dt_transformDecomp.drop(columns=["ds", "dep_var"])
        coefs = xDecompVec.drop(columns=["sol_id", "(Intercept)"])

        scurved = scurved.apply(pd.to_numeric, errors="coerce")
        coefs = coefs.apply(pd.to_numeric, errors="coerce")

        xDecompVec = pd.DataFrame(
            np.multiply(scurved.values, coefs.values),
            columns=coefs.columns,
        )

        xDecompVec["intercept"] = intercept
        xDecompVec["depVarHat"] = xDecompVec.sum(axis=1) + intercept
        xDecompVec["sol_id"] = sid

        xDecompVec = pd.concat(
            [dt_transformDecomp[["ds", "dep_var"]], xDecompVec], axis=1
        )

        xDecompVecPlot = xDecompVec[["ds", "dep_var", "depVarHat"]].rename(
            columns={"dep_var": "actual", "depVarHat": "predicted"}
        )
        xDecompVecPlotMelted = xDecompVecPlot.melt(
            id_vars="ds", var_name="variable", value_name="value"
        )

        rsq = plotWaterfall[plotWaterfall["sol_id"] == sid]["rsq_train"].values[0]
        plot5data = {
            "xDecompVecPlotMelted": xDecompVecPlotMelted,
            "rsq": rsq,
        }
        self.logger.debug(f"Generated plot5data, fitted vs actual, for sid: {sid}")

        # 6. Diagnostic: fitted vs residual
        plot6data = {"xDecompVecPlot": xDecompVecPlot}
        self.logger.debug(f"Generated plot6data, fitted vs residual, for sid: {sid}")

        # 7. Immediate vs carryover response
        plot7data = self.robyn_immcarr(
            pareto_data, aggregated_data["result_hyp_param"], sid
        )
        self.logger.debug(
            f"Generated plot7data, immediate vs carryover, for sid: {sid}"
        )
        mediaVecCollect = pd.concat(
            [
                dt_transformPlot.assign(type="rawMedia", sol_id=sid),
                dt_transformSpend.assign(type="rawSpend", sol_id=sid),
                dt_transformSpendMod.assign(type="predictedExposure", sol_id=sid),
                dt_transformAdstock.assign(type="adstockedMedia", sol_id=sid),
                self.dt_transformSaturation.assign(type="saturatedMedia", sol_id=sid),
                dt_transformSaturationSpendReverse.assign(
                    type="saturatedSpendReversed", sol_id=sid
                ),
                dt_transformSaturationDecomp.assign(type="decompMedia", sol_id=sid),
            ],
            ignore_index=True,
        )
        return {
            "mediaVecCollect": mediaVecCollect,
            "xDecompVec": xDecompVec,
            "plotData": {
                "plot1data": plot1data,
                "plot2data": plot2data,
                "plot3data": plot3data,
                "plot4data": plot4data,
                "plot5data": plot5data,
                "plot6data": plot6data,
                "plot7data": plot7data,
            },
            "plot7data": plot7data,
        }

    def _generate_waterfall_data(
        self, plotWaterfall: pd.DataFrame, sid: str
    ) -> pd.DataFrame:
        plotWaterfallLoop = plotWaterfall[plotWaterfall["sol_id"] == sid].sort_values(
            "xDecompPerc"
        )
        plotWaterfallLoop["end"] = 1 - plotWaterfallLoop["xDecompPerc"].cumsum()
        plotWaterfallLoop["start"] = plotWaterfallLoop["end"].shift(1).fillna(1)
        plotWaterfallLoop["id"] = range(1, len(plotWaterfallLoop) + 1)
        plotWaterfallLoop["rn"] = pd.Categorical(plotWaterfallLoop["rn"])
        plotWaterfallLoop["sign"] = pd.Categorical(
            np.where(plotWaterfallLoop["xDecompPerc"] >= 0, "Positive", "Negative")
        )

        return plotWaterfallLoop[
            ["id", "rn", "coef", "xDecompAgg", "xDecompPerc", "start", "end", "sign"]
        ].reset_index()

    def _generate_adstock_data(
        self, sid: str, pareto_data: ParetoData
    ) -> Dict[str, Union[pd.DataFrame, AdstockType]]:
        dt_geometric = None
        weibullCollect = None
        resultHypParamLoop = pareto_data.result_hyp_param[
            pareto_data.result_hyp_param["sol_id"] == sid
        ]
        get_hp_names = []
        for media in self.mmm_data.mmmdata_spec.all_media:
            if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
                get_hp_names.extend(
                    [f"{media}_alphas", f"{media}_gammas", f"{media}_thetas"]
                )
            else:
                get_hp_names.extend(
                    [
                        f"{media}_alphas",
                        f"{media}_gammas",
                        f"{media}_shapes",
                        f"{media}_scales",
                    ]
                )

        self.hypParam = resultHypParamLoop[get_hp_names]

        adstock_type = self.hyperparameter.adstock
        if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
            hypParam_thetas = [
                self.hypParam[f"{media}_thetas"].iloc[0]
                for media in self.mmm_data.mmmdata_spec.all_media
            ]
            dt_geometric = pd.DataFrame(
                {
                    "channels": self.mmm_data.mmmdata_spec.all_media,
                    "thetas": hypParam_thetas,
                }
            )
        elif self.hyperparameter.adstock in [
            AdstockType.WEIBULL_CDF,
            AdstockType.WEIBULL_PDF,
        ]:
            weibullCollect = self._process_weibull_collect(self.hypParam, adstock_type)

        return {
            "dt_geometric": dt_geometric,
            "weibullCollect": weibullCollect,
            "wb_type": adstock_type,
        }

    def _process_weibull_collect(
        self, hypParam: pd.DataFrame, adstock_type: AdstockType
    ) -> pd.DataFrame:
        shapeVec = [
            hypParam[f"{media}_shapes"].iloc[0]
            for media in self.mmm_data.mmmdata_spec.all_media
        ]
        scaleVec = [
            hypParam[f"{media}_scales"].iloc[0]
            for media in self.mmm_data.mmmdata_spec.all_media
        ]
        weibullCollect = []
        for v1 in range(len(self.mmm_data.mmmdata_spec.all_media)):
            dt_weibull = pd.DataFrame(
                {
                    "x": range(1, self.mmm_data.mmmdata_spec.rolling_window_length + 1),
                    "decay_accumulated": self.transformer.adstock_weibull(
                        range(1, self.mmm_data.mmmdata_spec.rolling_window_length + 1),
                        shape=shapeVec[v1],
                        scale=scaleVec[v1],
                        adstockType=adstock_type,
                    )["thetaVecCum"],
                    "adstockType": adstock_type,
                    "channel": self.mmm_data.mmmdata_spec.all_media[v1],
                }
            )
            dt_weibull["halflife"] = (
                (dt_weibull["decay_accumulated"] - 0.5).abs().idxmin()
            )
            max_non0 = (dt_weibull["decay_accumulated"] > 0.001).argmax()
            dt_weibull["cut_time"] = (
                max_non0 * 2 if max_non0 <= 5 else int(max_non0 + max_non0 / 3)
            )
            weibullCollect.append(dt_weibull)

        weibullCollect = pd.concat(weibullCollect)
        return weibullCollect[weibullCollect["x"] <= weibullCollect["cut_time"].max()]

    def _generate_response_data(
        self,
        sid: str,
        dt_mod: pd.DataFrame,
        plotWaterfall: pd.DataFrame,
        plotWaterfallLoop: pd.DataFrame,
        rw_start_loc: int,
        rw_end_loc: int,
    ) -> Dict[str, pd.DataFrame]:
        dt_transformPlot = dt_mod[["ds"] + self.mmm_data.mmmdata_spec.all_media]
        dt_transformSpend = pd.concat(
            [
                dt_transformPlot[["ds"]],
                self.mmm_data.data[self.mmm_data.mmmdata_spec.paid_media_spends],
            ],
            axis=1,
        )
        dt_transformSpendMod = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]
        dt_transformAdstock = dt_transformPlot.copy()
        self.dt_transformSaturation = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]

        all_media_channels = self.mmm_data.mmmdata_spec.all_media
        for med in range(len(all_media_channels)):
            med_select = all_media_channels[med]
            m = pd.Series(dt_transformPlot[med_select].values)
            adstock = self.hyperparameter.adstock
            if adstock == AdstockType.GEOMETRIC:
                thetas = self.hypParam[f"{all_media_channels[med]}_thetas"].values
                channelHyperparam = ChannelHyperparameters(thetas=thetas)
            elif adstock in [
                AdstockType.WEIBULL_PDF,
                AdstockType.WEIBULL_CDF,
            ]:
                shapes = self.hypParam[f"{all_media_channels[med]}_shapes"].values
                scales = self.hypParam[f"{all_media_channels[med]}_scales"].values
                channelHyperparam = ChannelHyperparameters(shapes=shapes, scales=scales)

            x_list = self.transformer.transform_adstock(m, adstock, channelHyperparam)
            m_adstocked = x_list.x_decayed
            dt_transformAdstock[med_select] = m_adstocked
            m_adstockedRollWind = m_adstocked[rw_start_loc:rw_end_loc]

            # Saturation
            alpha = self.hypParam[f"{all_media_channels[med]}_alphas"].iloc[0]
            gamma = self.hypParam[f"{all_media_channels[med]}_gammas"].iloc[0]
            self.dt_transformSaturation.loc[:, med_select] = (
                self.transformer.saturation_hill(
                    x=m_adstockedRollWind, alpha=alpha, gamma=gamma
                )
            )

        dt_transformSaturationDecomp = self.dt_transformSaturation.copy()
        for i in range(len(all_media_channels)):
            coef = plotWaterfallLoop["coef"][
                plotWaterfallLoop["rn"] == all_media_channels[i]
            ].values[0]
            dt_transformSaturationDecomp[all_media_channels[i]] *= coef

        dt_transformSaturationSpendReverse = dt_transformAdstock.iloc[
            rw_start_loc:rw_end_loc
        ]

        # Spend response curve
        dt_scurvePlot = dt_transformSaturationDecomp.melt(
            id_vars=["ds"],
            value_vars=all_media_channels,
            var_name="channel",
            value_name="response",
        )

        # Gather spend data and merge it into dt_scurvePlot
        spend_data = dt_transformSaturationSpendReverse.melt(
            id_vars=["ds"],
            value_vars=all_media_channels,
            var_name="channel",
            value_name="spend",
        )

        # Merge spend data into dt_scurvePlot based on the 'channel' and 'ds' columns
        dt_scurvePlot = dt_scurvePlot.merge(
            spend_data[["ds", "channel", "spend"]],
            on=["ds", "channel"],
            how="left",
        )

        # Remove outlier introduced by MM nls fitting
        dt_scurvePlot = dt_scurvePlot[dt_scurvePlot["spend"] >= 0]

        # Calculate dt_scurvePlotMean
        dt_scurvePlotMean = (
            plotWaterfall[
                (plotWaterfall["sol_id"] == sid) & (~plotWaterfall["mean_spend"].isna())
            ][
                [
                    "rn",
                    "mean_spend",
                    "mean_spend_adstocked",
                    "mean_carryover",
                    "mean_response",
                    "sol_id",
                ]
            ]
            .rename(columns={"rn": "channel"})
            .reset_index()
        )

        return {
            "dt_scurvePlot": dt_scurvePlot,
            "dt_scurvePlotMean": dt_scurvePlotMean,
            "dt_transformPlot": dt_transformPlot,
            "dt_transformSpend": dt_transformSpend,
            "dt_transformSpendMod": dt_transformSpendMod,
            "dt_transformAdstock": dt_transformAdstock,
            "dt_transformSaturationSpendReverse": dt_transformSaturationSpendReverse,
            "dt_transformSaturationDecomp": dt_transformSaturationDecomp,
        }

    def _get_prophet_var_values(self, var) -> str:
        """Helper function to handle both string and enum prophet variables."""
        try:
            return var.value if hasattr(var, "value") else var
        except AttributeError:
            return var

    def robyn_immcarr(
        self,
        pareto_data: ParetoData,
        result_hyp_param: pd.DataFrame,
        sol_id=None,
        start_date=None,
        end_date=None,
    ):
        """Calculate immediate and carryover effects."""
        # Define default values when not provided
        if sol_id is None:
            sol_id = result_hyp_param["sol_id"].iloc[0]
        if start_date is None:
            start_date = self.mmm_data.mmmdata_spec.window_start
        if end_date is None:
            end_date = self.mmm_data.mmmdata_spec.window_end

        # Convert to datetime series and reset index
        dt_modRollWind = pd.to_datetime(
            self.featurized_mmm_data.dt_modRollWind["ds"]
        ).reset_index(drop=True)
        dt_modRollWind = dt_modRollWind.dropna()

        # Convert inputs to datetime
        if isinstance(start_date, (list, pd.Series)):
            start_date = start_date[0]
        start_date = pd.to_datetime(start_date)

        if isinstance(end_date, (list, pd.Series)):
            end_date = end_date[0]
        end_date = pd.to_datetime(end_date)

        # Find closest dates using absolute difference
        start_idx = (dt_modRollWind - start_date).abs().argmin()
        end_idx = (dt_modRollWind - end_date).abs().argmin()

        start_date = dt_modRollWind[start_idx]
        end_date = dt_modRollWind[end_idx]

        # Use boolean indexing instead of value matching
        rollingWindow = range(start_idx, end_idx + 1)

        # Rest of your function remains the same
        self.logger.info(
            "Calculating saturated dataframes with carryover and immediate parts"
        )
        hypParamSam = result_hyp_param[result_hyp_param["sol_id"] == sol_id]
        hyperparameter = self._extract_hyperparameter(hypParamSam)

        dt_saturated_dfs = self.transformer.run_transformations(
            self.featurized_mmm_data,
            hyperparameter,
            hyperparameter.adstock,
        )
        # Calculate decomposition
        coefs = pareto_data.x_decomp_agg.loc[
            pareto_data.x_decomp_agg["sol_id"] == sol_id, "coef"
        ].values
        coefs_names = pareto_data.x_decomp_agg.loc[
            pareto_data.x_decomp_agg["sol_id"] == sol_id, "rn"
        ].values

        # Create a DataFrame to hold coefficients and their names
        coefs_df = pd.DataFrame({"name": coefs_names, "coefficient": coefs})

        self.logger.debug("Computing decomposition")
        # Check if 'revenue' exists in the columns
        if "revenue" in dt_saturated_dfs.dt_modSaturated.columns:
            # Rename 'revenue' to 'dep_var'
            dt_saturated_dfs.dt_modSaturated = dt_saturated_dfs.dt_modSaturated.rename(
                columns={"revenue": "dep_var"}
            )
            # print("Column 'revenue' renamed to 'dep_var'.")
        else:
            # print("Column 'revenue' does not exist.")
            pass
        decompCollect = self._model_decomp(
            inputs={
                "coefs": coefs_df,
                "y_pred": dt_saturated_dfs.dt_modSaturated["dep_var"].iloc[
                    rollingWindow
                ],
                "dt_modSaturated": dt_saturated_dfs.dt_modSaturated.iloc[rollingWindow],
                "dt_saturatedImmediate": dt_saturated_dfs.dt_saturatedImmediate.iloc[
                    rollingWindow
                ],
                "dt_saturatedCarryover": dt_saturated_dfs.dt_saturatedCarryover.iloc[
                    rollingWindow
                ],
                "dt_modRollWind": self.featurized_mmm_data.dt_modRollWind.iloc[
                    rollingWindow
                ],
                "refreshAddedStart": start_date,
            }
        )

        # Media decomposition
        mediaDecompImmediate = decompCollect["mediaDecompImmediate"].drop(
            columns=["ds", "y"], errors="ignore"
        )
        mediaDecompImmediate.columns = [
            f"{col}_MDI" for col in mediaDecompImmediate.columns
        ]

        mediaDecompCarryover = decompCollect["mediaDecompCarryover"].drop(
            columns=["ds", "y"], errors="ignore"
        )
        mediaDecompCarryover.columns = [
            f"{col}_MDC" for col in mediaDecompCarryover.columns
        ]

        # Combine results
        temp = pd.concat(
            [decompCollect["xDecompVec"], mediaDecompImmediate, mediaDecompCarryover],
            axis=1,
        )
        temp["sol_id"] = sol_id

        # Create vector collections
        vec_collect = {
            "xDecompVec": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDI")
                    | temp.columns.str.endswith("_MDC")
                ]
            ),
            "xDecompVecImmediate": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDC")
                    | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
            "xDecompVecCarryover": temp.drop(
                columns=temp.columns[
                    temp.columns.str.endswith("_MDI")
                    | temp.columns.isin(self.mmm_data.mmmdata_spec.all_media)
                ]
            ),
        }

        # Rename columns
        this = vec_collect["xDecompVecImmediate"].columns.str.replace(
            "_MDI", "", regex=False
        )
        vec_collect["xDecompVecImmediate"].columns = this
        vec_collect["xDecompVecCarryover"].columns = this

        # Convert datetime64[ns] to object: You can convert the ds column to a string format, which will change its type to object.
        vec_collect["xDecompVecCarryover"]["ds"] = vec_collect["xDecompVecCarryover"][
            "ds"
        ].astype(str)
        vec_collect["xDecompVec"]["ds"] = vec_collect["xDecompVec"]["ds"].astype(str)
        # Calculate carryover percentages
        df_caov = (
            vec_collect["xDecompVecCarryover"].groupby("sol_id").sum().reset_index()
        ).drop(columns="ds")
        df_total = (
            vec_collect["xDecompVec"]
            .groupby("sol_id")
            .sum()
            .reset_index()
            .drop(columns="ds")
        )
        df_caov_pct = df_caov.copy()
        df_caov_pct.loc[:, df_caov_pct.columns[1:]] = (
            df_caov_pct.loc[:, df_caov_pct.columns[1:]]
            .div(df_total.iloc[:, 1:].values)
            .astype("float64")
        )
        df_caov_pct = df_caov_pct.melt(
            id_vars="sol_id", var_name="rn", value_name="carryover_pct"
        ).fillna(0)

        # Gather everything in an aggregated format
        self.logger.debug(
            "Aggregating final results from decomposition carryover and immediate parts"
        )
        xDecompVecImmeCaov = (
            pd.concat(
                [
                    vec_collect["xDecompVecImmediate"].assign(type="Immediate"),
                    vec_collect["xDecompVecCarryover"].assign(type="Carryover"),
                ],
                axis=0,
            )
            .melt(
                id_vars=["sol_id", "type"],
                value_vars=self.mmm_data.mmmdata_spec.all_media,
                var_name="rn",
                value_name="value",
            )
            .assign(start_date=start_date, end_date=end_date)
        )

        # Grouping and aggregating the data
        xDecompVecImmeCaov = (
            xDecompVecImmeCaov.groupby(
                ["sol_id", "start_date", "end_date", "rn", "type"]
            )
            .agg(response=("value", "sum"))
            .reset_index()
        )

        xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov[
            "response"
        ] / xDecompVecImmeCaov.groupby(["sol_id", "start_date", "end_date", "type"])[
            "response"
        ].transform(
            "sum"
        )
        xDecompVecImmeCaov.fillna(0, inplace=True)

        # Join with carryover percentages
        xDecompVecImmeCaov = xDecompVecImmeCaov.merge(
            df_caov_pct, on=["sol_id", "rn"], how="left"
        )

        return xDecompVecImmeCaov

    def _extract_hyperparameter(self, hypParamSam: pd.DataFrame) -> Hyperparameters:
        """
        This function extracts hyperparameters from a given DataFrame.

        Parameters:
        hypParamSam (DataFrame): A DataFrame containing hyperparameters.

        Returns:
        hyperparameter (dict): A dictionary of hyperparameters.
        """
        channelHyperparams: dict[str, ChannelHyperparameters] = {}
        for med in self.mmm_data.mmmdata_spec.all_media:
            alphas = hypParamSam[f"{med}_alphas"].values
            gammas = hypParamSam[f"{med}_gammas"].values
            if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
                thetas = hypParamSam[f"{med}_thetas"].values
                channelHyperparams[med] = ChannelHyperparameters(
                    thetas=thetas,
                    alphas=alphas,
                    gammas=gammas,
                )
            elif self.hyperparameter.adstock in [
                AdstockType.WEIBULL_CDF,
                AdstockType.WEIBULL_PDF,
            ]:
                shapes = hypParamSam[f"{med}_shapes"].values
                scales = hypParamSam[f"{med}_scales"].values
                channelHyperparams[med] = ChannelHyperparameters(
                    shapes=shapes,
                    scales=scales,
                    alphas=alphas,
                    gammas=gammas,
                )

        return Hyperparameters(
            adstock=self.hyperparameter.adstock, hyperparameters=channelHyperparams
        )

    def _model_decomp(self, inputs) -> Dict[str, pd.DataFrame]:
        # Extracting inputs from the dictionary
        coefs = inputs["coefs"]
        y_pred = inputs["y_pred"]
        dt_modSaturated = inputs["dt_modSaturated"]
        dt_saturatedImmediate = inputs["dt_saturatedImmediate"]
        dt_saturatedCarryover = inputs["dt_saturatedCarryover"]
        dt_modRollWind = inputs["dt_modRollWind"]
        # Input for decomp
        y = dt_modSaturated["dep_var"]
        # Select all columns except 'dep_var'
        x = dt_modSaturated.drop(columns=["dep_var"])
        # Convert 'events' column to numeric if it exists
        if "events" in x.columns:
            x["events"] = pd.to_numeric(x["events"], errors="coerce")
            # x["events"].fillna(0, inplace=True)  # Replace NaN values with 0
            x.loc[:, "events"] = x["events"].fillna(0)
        intercept = coefs["coefficient"].iloc[0]
        # Decomp x
        # Create an empty DataFrame for xDecomp
        xDecomp = pd.DataFrame()
        # Multiply each regressor by its corresponding coefficient
        for name in x.columns:
            # Get the corresponding coefficient for the regressor
            coefficient_value = coefs.loc[coefs["name"] == name, "coefficient"].values
            xDecomp[name] = x[name] * (
                coefficient_value if len(coefficient_value) > 0 else 0
            )
        # Add intercept as the first column
        xDecomp.insert(0, "intercept", intercept)
        xDecompOut = pd.concat(
            [
                pd.DataFrame({"ds": dt_modRollWind["ds"], "y": y, "y_pred": y_pred}),
                xDecomp,
            ],
            axis=1,
        )
        # Decomp immediate & carryover response
        sel_coef = coefs["name"].isin(
            dt_saturatedImmediate.columns
        )  # Check if coefficient names are in the immediate DataFrame
        coefs_media = coefs[sel_coef].set_index("name")[
            "coefficient"
        ]  # Set names for coefs_media
        mediaDecompImmediate = pd.DataFrame(
            {
                name: dt_saturatedImmediate[name] * coefs_media[name]
                for name in coefs_media.index
            }
        )
        mediaDecompCarryover = pd.DataFrame(
            {
                name: dt_saturatedCarryover[name] * coefs_media[name]
                for name in coefs_media.index
            }
        )
        return {
            "xDecompVec": xDecompOut,
            "mediaDecompImmediate": mediaDecompImmediate,
            "mediaDecompCarryover": mediaDecompCarryover,
        }
