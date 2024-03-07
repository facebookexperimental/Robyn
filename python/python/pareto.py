# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from .allocator import get_hill_params
from .cluster import errors_scores

def robyn_pareto(InputCollect, OutputModels, pareto_fronts="auto", min_candidates=100, calibration_constraint=0.1, quiet=False, calibrated=False, **kwargs):
    hyper_fixed = OutputModels["metadata"]["hyper_fixed"]
    OutModels = [trial for trial in OutputModels["trials"] if "resultCollect" in trial]
    df_list = [item['resultCollect']['resultHypParam'].assign(trial=item['trial']) for item in OutModels]

    resultHypParam = pd.concat(df_list, ignore_index=True)

    # xDecompAgg = pd.concat([x.resultCollect.xDecompAgg.assign(trial=x.trial) for x in OutModels])
    xDecompAgg = pd.concat([
        x['resultCollect']['xDecompAgg'].assign(trial=x['trial'])
        for x in OutModels
    ])

    if calibrated:
        resultCalibration = pd.concat([x['resultCollect']['liftCalibration'].assign(trial=x['trial']).rename(columns={"liftMedia": "rn"}) for x in OutModels])
    else:
        resultCalibration = None

    if not hyper_fixed:
        df_names = ["resultHypParam", "xDecompAgg"] if not calibrated else ["resultHypParam", "xDecompAgg", "resultCalibration"]

        for df_name in df_names:
            concatenated_dfs = pd.concat([
                x["resultCollect"][df_name].assign(iterations=(x["resultCollect"]["resultHypParam"]['iterNG'] - 1) * OutputModels['metadata']['cores'] + x["resultCollect"]["resultHypParam"]['iterPar']) for x in OutModels
            ], ignore_index=True)
            globals()[df_name] = concatenated_dfs

    elif hyper_fixed and calibrated:
        df_names = ["resultCalibration"]

        for df_name in df_names:
            concatenated_dfs = pd.concat([
                x["resultCollect"][df_name].assign(iterations=(x["resultCollect"]["resultHypParam"]['iterNG'] - 1) * OutputModels['metadata']['cores'] + x["resultCollect"]["resultHypParam"]['iterPar'])
                for x in OutModels
            ], ignore_index=True)

            globals()[df_name] = concatenated_dfs

    # If recreated model, inherit bootstrap results
    if len(xDecompAgg.solID.unique()) == 1 and "boot_mean" not in xDecompAgg.columns:
        bootstrap = OutputModels.bootstrap
        if bootstrap is not None:
            xDecompAgg = xDecompAgg.merge(bootstrap, on=["rn", "variable"])

    xDecompAggCoef0 = xDecompAgg.loc[
        xDecompAgg.rn.isin(InputCollect["robyn_inputs"]["paid_media_spends"])
    ].groupby("solID").agg({"coefs": lambda x: (x.min() == 0).any()})

    if not hyper_fixed:
        mape_lift_quantile10 = np.quantile(resultHypParam["mape"], calibration_constraint, overwrite_input=True)
        nrmse_quantile90 = np.quantile(resultHypParam["nrmse"], 0.90, overwrite_input=True)
        decomprssd_quantile90 = np.quantile(resultHypParam["decomp.rssd"], 0.90, overwrite_input=True)

        resultHypParam = pd.merge(resultHypParam, xDecompAggCoef0, on="solID")
        resultHypParam["mape.qt10"] = (resultHypParam["mape"] <= mape_lift_quantile10) & (resultHypParam["nrmse"] <= nrmse_quantile90) & (resultHypParam["decomp.rssd"] <= decomprssd_quantile90)

        resultHypParamPareto = resultHypParam[resultHypParam["mape.qt10"] == True]
        fronts = float('inf') if 'auto' in pareto_fronts else pareto_fronts

        paretoResults = pareto_front(resultHypParamPareto["nrmse"], resultHypParamPareto["decomp.rssd"], fronts=fronts, sort=False)

        # resultHypParamPareto = pd.merge(resultHypParamPareto, paretoResults, on=["nrmse", "decomp.rssd"])
        resultHypParamPareto = pd.merge(resultHypParamPareto, paretoResults,
                                left_on=["nrmse", "decomp.rssd"],
                                right_on=["x", "y"],
                                how="left")
        resultHypParamPareto = resultHypParamPareto.rename(columns={"pareto_front": "robynPareto"})
        resultHypParamPareto = resultHypParamPareto.sort_values(["iterNG", "iterPar", "nrmse"]).reset_index(drop=True)
        resultHypParamPareto = resultHypParamPareto[["solID", "robynPareto"]]
        resultHypParamPareto = resultHypParamPareto.groupby("solID", as_index=False).first()



        resultHypParam = pd.merge(resultHypParam, resultHypParamPareto, on="solID")

    else:
        resultHypParam["mape.qt10"] = True
        resultHypParam["robynPareto"] = 1
        resultHypParam["coef0"] = np.nan

    resultHypParam["error_score"] = errors_scores(resultHypParam, ts_validation=OutputModels["metadata"]["ts_validation"], **kwargs)

    xDecompAgg = pd.merge(xDecompAgg, resultHypParam.filter(['robynPareto','solID'], axis=1), on='solID')
    # decompSpendDist = pd.concat([pd.DataFrame({'decompSpendDist': x["resultCollect"]["decompSpendDist"], 'trial': x["trial"]}) for x in OutModels])
    # decompSpendDist = pd.concat([
    #     pd.DataFrame({
    #         'decompSpendDist': [x["resultCollect"]["decompSpendDist"]],
    #         'trial': [x["trial"]]
    #     }) for x in OutModels
    # ])
    decompSpendDistList = []
    for x in OutModels:
        # Assuming x["resultCollect"]["decompSpendDist"] is already a DataFrame
        df_temp = x["resultCollect"]["decompSpendDist"].copy()
        df_temp['trial'] = x["trial"]  # Add 'trial' as a new column to each DataFrame
        decompSpendDistList.append(df_temp)

    # Concatenate all the DataFrames to stack them vertically
    decompSpendDist = pd.concat(decompSpendDistList, ignore_index=True)

    if not hyper_fixed:
        decompSpendDist['solID'] = decompSpendDist['trial'].astype(str) + '_' + \
                               decompSpendDist['iterNG'].astype(str) + '_' + \
                               decompSpendDist['iterPar'].astype(str)

    decompSpendDist = pd.merge(decompSpendDist, resultHypParam[['robynPareto', 'solID']], on='solID', how='left')

    if True:
        # TODO: Handle parallel execution
        # if OutputModels["metadata"]["cores"] > 1:
        #     from concurrent.futures import ProcessPoolExecutor
        #     with ProcessPoolExecutor(max_workers=OutputModels["metadata"]["cores"]) as executor:
        #         pareto_fronts = list(executor.map(get_pareto_fronts, pareto_fronts))
        # else:
        #     pareto_fronts = get_pareto_fronts(pareto_fronts)

        if hyper_fixed:
            pareto_fronts = 1

        if pareto_fronts == 'auto':
            n_pareto = resultHypParam[~resultHypParam.robynPareto.isna()].shape[0]
            if n_pareto <= min_candidates and resultHypParam.shape[0] > 1 and not calibrated:
                raise ValueError(f"Less than {min_candidates} candidates in pareto fronts. Increase iterations to get more model candidates or decrease min_candidates in robyn_output()")
            # auto_pareto = resultHypParam[~resultHypParam.robynPareto.isna()].groupby('robynPareto').agg({'solID': 'nunique'}).assign(n_cum=lambda df: df['solID'].cumsum()).query(f'n_cum >= {min_candidates}').iloc[[0]]
            auto_pareto = resultHypParam[~resultHypParam['robynPareto'].isna()] \
                .groupby('robynPareto', as_index=False) \
                .agg({'solID': 'nunique'}) \
                .rename(columns={'solID': 'n'}) \
                .assign(n_cum=lambda df: df['n'].cumsum()) \
                .query(f'n_cum >= {min_candidates}') \
                .iloc[0]
            print(auto_pareto)
            # print(f">> Automatically selected {auto_pareto.robynPareto.values[0]} Pareto-fronts to contain at least {min_candidates} pareto-optimal models ({auto_pareto.solID.values[0]})")
            print(f">> Automatically selected {auto_pareto['robynPareto']} Pareto-fronts to contain at least {min_candidates} pareto-optimal models ({auto_pareto['n_cum']})")

            pareto_fronts = int(auto_pareto['robynPareto'])


        pareto_fronts_vec = np.arange(1, pareto_fronts+1)

        decompSpendDistPar = decompSpendDist[decompSpendDist['robynPareto'].isin(pareto_fronts_vec)]
        resultHypParamPar = resultHypParam[resultHypParam['robynPareto'].isin(pareto_fronts_vec)]
        xDecompAggPar = xDecompAgg[xDecompAgg['robynPareto'].isin(pareto_fronts_vec)]
        respN = None

    if not quiet:
        print(f">>> Calculating response curves for all models' media variables ({decompSpendDistPar.shape[0]})...")

    if OutputModels["cores"] > 1:
        resp_collect = pd.concat(
            [run_dt_resp(respN, InputCollect, OutputModels, decompSpendDistPar, resultHypParamPar, xDecompAggPar, **kwargs) for respN in range(len(decompSpendDistPar["rn"]))]
        )
        stopImplicitCluster()
        registerDoSEQ()
        getDoParWorkers()
    else:
        resp_collect = pd.concat(
            [run_dt_resp(respN, InputCollect, OutputModels, decompSpendDistPar, resultHypParamPar, xDecompAggPar, **kwargs) for respN in range(len(decompSpendDistPar["rn"]))]
        )

    decompSpendDist = pd.merge(
        decompSpendDist,
        resp_collect,
        on=["solID", "rn"]
    )
    decompSpendDist["roi_mean"] = decompSpendDist["mean_response"] / decompSpendDist["mean_spend"]
    decompSpendDist["roi_total"] = decompSpendDist["xDecompAgg"] / decompSpendDist["total_spend"]
    decompSpendDist["cpa_mean"] = decompSpendDist["mean_spend"] / decompSpendDist["mean_response"]
    decompSpendDist["cpa_total"] = decompSpendDist["total_spend"] / decompSpendDist["xDecompAgg"]

    xDecompAgg = pd.merge(
        xDecompAgg,
        decompSpendDist[["rn", "solID", "total_spend", "mean_spend", "mean_spend_adstocked", "mean_carryover", "mean_response", "spend_share", "effect_share", "roi_mean", "roi_total", "cpa_total"]],
        on=["solID", "rn"]
    )

    mediaVecCollect = []
    xDecompVecCollect = []
    plotDataCollect = []
    df_caov_pct_all = pd.DataFrame()
    dt_mod = InputCollect["dt_mod"]
    dt_modRollWind = InputCollect["dt_modRollWind"]
    rw_start_loc = InputCollect["rollingWindowStartWhich"]
    rw_end_loc = InputCollect["rollingWindowEndWhich"]

    for pf in pareto_fronts_vec:
        plotMediaShare = InputCollect[
            (InputCollect["robynPareto"] == pf) & (InputCollect["rn"].isin(OutputModels["paid_media_spends"]))
        ]
        uniqueSol = plotMediaShare["solID"].unique()
        plotWaterfall = OutputModels[OutputModels["robynPareto"] == pf]
        if not quiet and len(uniqueSol) > 1:
            print(f">> Pareto-Front: {pf} [{len(uniqueSol)} models]")

        # Calculations for pareto AND pareto plots
        for sid in uniqueSol:
            if not quiet and len(uniqueSol) > 1:
                print(f"Processing solution {sid} of {len(uniqueSol)}", end="\r")

            ## 1. Spend x effect share comparison
            temp = plotMediaShare[plotMediaShare["solID"] == sid].melt(
                id_vars=["rn", "nrmse", "decomp.rssd", "rsq_train"],
                var_name="variable",
                value_name="value",
            )
            plotMediaShareLoopBar = temp[temp["variable"].isin(["spend_share", "effect_share"])]
            plotMediaShareLoopLine = temp[temp["variable"] == "roi_total"]
            line_rm_inf = ~temp["value"].isin([np.inf, -np.inf])
            ySecScale = max(plotMediaShareLoopLine["value"][line_rm_inf]) / max(plotMediaShareLoopBar["value"]) * 1.1
            plot1data = {
                "plotMediaShareLoopBar": plotMediaShareLoopBar,
                "plotMediaShareLoopLine": plotMediaShareLoopLine,
                "ySecScale": ySecScale,
            }

            ## 2. Waterfall
            plotWaterfallLoop = plotWaterfall[plotWaterfall["solID"] == sid].sort_values("xDecompPerc")
            plotWaterfallLoop["end"] = 1 - plotWaterfallLoop["xDecompPerc"].cumsum()
            plotWaterfallLoop["start"] = plotWaterfallLoop["end"].shift(1)
            plotWaterfallLoop.loc[0, "start"] = 1
            plotWaterfallLoop["id"] = range(len(plotWaterfallLoop))
            plotWaterfallLoop["rn"] = plotWaterfallLoop["rn"].astype("category")
            plotWaterfallLoop["sign"] = plotWaterfallLoop["xDecompPerc"].apply(lambda x: "Positive" if x >= 0 else "Negative")
            plot2data = {"plotWaterfallLoop": plotWaterfallLoop}

            ## 3. Adstock rate
            dt_geometric = weibullCollect = wb_type = None
            resultHypParamLoop = OutputModels[OutputModels["solID"] == sid]
            get_hp_names = [name for name in InputCollect["hyperparameters"].keys() if not name.endswith("_penalty")]
            hypParam = resultHypParamLoop[get_hp_names]
            if InputCollect["adstock"] == "geometric":
                hypParam_thetas = hypParam[InputCollect["all_media"] + "_thetas"].values.tolist()
                dt_geometric = pd.DataFrame({"channels": InputCollect["all_media"], "thetas": hypParam_thetas})

            if InputCollect.adstock in ["weibull_cdf", "weibull_pdf"]:
                shapeVec = np.array([hypParam[f"{media}_shapes"] for media in InputCollect.all_media])
                scaleVec = np.array([hypParam[f"{media}_scales"] for media in InputCollect.all_media])
                wb_type = InputCollect.adstock[9:11]
                weibullCollect = []
                n = 1
                for v1 in range(len(InputCollect.all_media)):
                    dt_weibull = pd.DataFrame(
                        {
                            "x": list(range(1, InputCollect.rollingWindowLength + 1)),
                            "decay_accumulated": adstock_weibull(
                                list(range(1, InputCollect.rollingWindowLength + 1)),
                                shape=shapeVec[v1],
                                scale=scaleVec[v1],
                                type=wb_type
                            ).thetaVecCum,
                            "type": wb_type,
                            "channel": InputCollect.all_media[v1]
                        }
                    )
                    dt_weibull["halflife"] = np.argmin(np.abs(dt_weibull.decay_accumulated - 0.5))
                    max_non0 = np.max(np.where(dt_weibull.decay_accumulated > 0.001, dt_weibull.x, np.nan))
                    dt_weibull["cut_time"] = np.where(max_non0 <= 5, 2 * max_non0, np.floor(max_non0 + max_non0 / 3))
                    weibullCollect.append(dt_weibull)
                    n += 1
                weibullCollect = pd.concat(weibullCollect)
                weibullCollect = weibullCollect.loc[weibullCollect.x <= weibullCollect.cut_time.max()]
            plot3data = {
                "dt_geometric": dt_geometric,
                "weibullCollect": weibullCollect,
                "wb_type": wb_type.upper()
            }
            dt_transformAdstock = dt_transformPlot
            dt_transformSaturation = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]

            m_decayRate = []
            for med in range(len(InputCollect.all_media)):
                med_select = InputCollect.all_media[med]
                m = dt_transformPlot.loc[:, med_select].iloc[0]
                # Adstocking
                adstock = InputCollect.adstock
                if adstock == "geometric":
                    theta = hypParam[f"{med_select}_thetas"][0]
                if adstock.startswith("weibull"):
                    shape = hypParam[f"{med_select}_shapes"][0]
                    scale = hypParam[f"{med_select}_scales"][0]
                x_list = transform_adstock(m, adstock, theta=theta, shape=shape, scale=scale)
                m_adstocked = x_list.x_decayed
                dt_transformAdstock.loc[:, med_select] = m_adstocked
                m_adstockedRollWind = m_adstocked.loc[rw_start_loc:rw_end_loc]
                ## Saturation
                alpha = hypParam[f"{med_select}_alphas"][0]
                gamma = hypParam[f"{med_select}_gammas"][0]
                dt_transformSaturation.loc[:, med_select] = saturation_hill(
                    x=m_adstockedRollWind, alpha=alpha, gamma=gamma
                )
            dt_transformSaturationDecomp = dt_transformSaturation
            for i in range(InputCollect.mediaVarCount):
                coef = plotWaterfall.loc[plotWaterfall.rn == InputCollect.all_media[i], "coef"].values[0]
                dt_transformSaturationDecomp.loc[:, InputCollect.all_media[i]] = coef * dt_transformSaturationDecomp.loc[:, InputCollect.all_media[i]]
            dt_transformSaturationSpendReverse = dt_transformAdstock.loc[rw_start_loc:rw_end_loc, :]

            dt_scurvePlot = pd.melt(
                dt_transformSaturationDecomp,
                id_vars=["channel"],
                var_name="response",
                value_name="spend"
            ).assign(spend=lambda df: dt_transformSaturationSpendReverse.loc[:, df.channel].values)

            # Remove outlier introduced by MM nls fitting
            dt_scurvePlot = dt_scurvePlot.loc[dt_scurvePlot.spend >= 0, :]
            dt_scurvePlotMean = plotWaterfall.loc[
                (plotWaterfall.solID == sid) & (~pd.isna(plotWaterfall.mean_spend)),
                ["channel", "mean_spend", "mean_spend_adstocked", "mean_carryover", "mean_response", "solID"]
            ].rename(columns={"channel": "rn"})

            # Exposure response curve
            plot4data = {
                "dt_scurvePlot": dt_scurvePlot,
                "dt_scurvePlotMean": dt_scurvePlotMean
            }

            # 5. Fitted vs actual
            col_order = ["ds", "dep_var"] + InputCollect.all_ind_vars
            dt_transformDecomp = dt_modRollWind.merge(
                dt_transformSaturation[InputCollect.all_media],
                left_on="ds",
                right_on="ds",
                how="left"
            )
            xDecompVec = xDecompAgg[xDecompAgg.solID == sid][["solID", "rn", "coef"]].set_index("rn")
            if "(Intercept)" not in xDecompVec.columns:
                xDecompVec["(Intercept)"] = 0
            xDecompVec = xDecompVec.loc[col_order[~col_order.isin(["ds", "dep_var"])]].reset_index()
            intercept = xDecompVec.iloc[0]["(Intercept)"]
            xDecompVec = dt_transformDecomp.merge(
                xDecompVec,
                left_on=["ds"] + col_order[~col_order.isin(["ds", "dep_var"])],
                right_on=["ds"] + col_order[~col_order.isin(["ds", "dep_var"])],
                how="left"
            )
            xDecompVec["intercept"] = intercept
            xDecompVec["depVarHat"] = xDecompVec.sum(axis=1) + intercept
            xDecompVec["solID"] = sid
            xDecompVecPlot = xDecompVec[["ds", "dep_var", "depVarHat"]]
            xDecompVecPlotMelted = pd.melt(
                xDecompVecPlot,
                id_vars=["ds"],
                value_vars=["actual", "predicted"],
                var_name="variable",
                value_name="value"
            )
            rsq = xDecompAgg[xDecompAgg.solID == sid]["rsq_train"].iloc[0]
            plot5data = {"xDecompVecPlotMelted": xDecompVecPlotMelted, "rsq": rsq}

            # 6. Diagnostic: fitted vs residual
            plot6data = {"xDecompVecPlot": xDecompVecPlot}

            # 7. Immediate vs carryover response
            # temp = xDecompVecImmCarr[xDecompVecImmCarr.solID == sid]
            hypParamSam = resultHypParam[resultHypParam.solID == sid]
            dt_saturated_dfs = run_transformations(InputCollect, hypParamSam, adstock)
            coefs = xDecompAgg['coef'][xDecompAgg.solID == sid]
            ##names(coefs) = xDecompAgg['rn'][xDecompAgg['solID'] == sid]
            coefs.columns = xDecompAgg['rn'][xDecompAgg['solID'] == sid]
            decompCollect = model_decomp(
                coefs=coefs,
                y_pred=dt_saturated_dfs.dt_modSaturated.dep_var,
                dt_modSaturated=dt_saturated_dfs.dt_modSaturated,
                dt_saturatedImmediate=dt_saturated_dfs.dt_saturatedImmediate,
                dt_saturatedCarryover=dt_saturated_dfs.dt_saturatedCarryover,
                dt_modRollWind=dt_modRollWind,
                refreshAddedStart=InputCollect.refreshAddedStart
            )
            mediaDecompImmediate = decompCollect.mediaDecompImmediate.drop(
                columns=["ds", "y"]
            )
            mediaDecompImmediate.columns = [f"{col}_MDI" for col in colnames(mediaDecompImmediate)]
            mediaDecompCarryover = decompCollect.mediaDecompCarryover.drop(
                columns=["ds", "y"]
            )
            mediaDecompCarryover.columns = [f"{col}_MDC" for col in colnames(mediaDecompCarryover)]
            temp = pd.concat(
                [
                    decompCollect.xDecompVec,
                    mediaDecompImmediate,
                    mediaDecompCarryover
                ],
                axis=1
            ).assign(solID=sid)
            vec_collect = {
                "xDecompVec": temp.drop(columns=["_MDI", "_MDC"]),
                "xDecompVecImmediate": temp.drop(columns=["_MDC"]).drop(columns=InputCollect.all_media),
                "xDecompVecCarryover": temp.drop(columns=["_MDI"]).drop(columns=InputCollect.all_media)
            }
            this = [col for col in colnames(vec_collect["xDecompVecImmediate"]) if not col.endswith("_MDI")]
            vec_collect["xDecompVecImmediate"].columns = [f"{col}_MDI" for col in this]
            vec_collect["xDecompVecCarryover"].columns = [f"{col}_MDC" for col in this]
            df_caov = vec_collect["xDecompVecCarryover"].groupby("solID").sum().reset_index()
            df_total = vec_collect["xDecompVec"].groupby("solID").sum().reset_index()
            df_caov_pct = df_caov.merge(
                df_total,
                on="solID",
                suffixes=("", "_total")
            )
            df_caov_pct["carryover_pct"] = df_caov_pct.apply(
                lambda row: row[f"{row.rn}_MDC"] / row[f"{row.rn}_total"],
                axis=1
            )
            df_caov_pct_all = pd.concat([df_caov_pct_all, df_caov_pct], ignore_index=True)
            # Gather everything in an aggregated format
            xDecompVecImmeCaov = pd.concat(
                [
                    vec_collect["xDecompVecImmediate"].assign(type="Immediate").drop(columns=["ds"]),
                    vec_collect["xDecompVecCarryover"].assign(type="Carryover").drop(columns=["ds"])
                ],
                ignore_index=True
            ).melt(
                id_vars=["solID", "type", "rn"],
                var_name="variable",
                value_name="value"
            ).groupby(["solID", "rn", "type"]).sum().reset_index()
            xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov.groupby(["solID", "rn"]).transform(
                lambda x: x["value"] / x["value"].sum()
            )
            xDecompVecImmeCaov.fillna(0, inplace=True)
            xDecompVecImmeCaov = xDecompVecImmeCaov.merge(
                df_caov_pct_all,
                on=["solID", "rn"]
            )
            if len(xDecompAgg.solID.unique()) == 1:
                xDecompVecImmeCaov["solID"] = OutputModels.trial1.resultCollect.resultHypParam.solID
            plot7data = xDecompVecImmeCaov

            # 8. Bootstrapped ROI/CPA with CIs
            # plot8data = "Empty"  # Filled when running robyn_onepagers() with clustering data

            # Gather all results
            mediaVecCollect = pd.concat(
                [
                    dt_transformPlot.assign(type="rawMedia", solID=sid),
                    dt_transformSpend.assign(type="rawSpend", solID=sid),
                    dt_transformSpendMod.assign(type="predictedExposure", solID=sid),
                    dt_transformAdstock.assign(type="adstockedMedia", solID=sid),
                    dt_transformSaturation.assign(type="saturatedMedia", solID=sid),
                    dt_transformSaturationSpendReverse.assign(type="saturatedSpendReversed", solID=sid),
                    dt_transformSaturationDecomp.assign(type="decompMedia", solID=sid)
                ],
                ignore_index=True
            )
            xDecompVecCollect = pd.concat([xDecompVecCollect, xDecompVec], ignore_index=True)
            plotDataCollect[sid] = {
                "plot1data": plot1data,
                "plot2data": plot2data,
                "plot3data": plot3data,
                "plot4data": plot4data,
                "plot5data": plot5data,
                "plot6data": plot6data,
                "plot7data": plot7data
                # "plot8data": plot8data
            }

    ## Manually added some data to following dict since some of them printed as None.
    pareto_results = {
            "pareto_solutions": list(xDecompVecCollect["solID"].unique()),
            "pareto_fronts": pareto_fronts,
            "resultHypParam": resultHypParam,
            "xDecompAgg": xDecompAgg,
            "resultCalibration": resultCalibration,
            "mediaVecCollect": mediaVecCollect,
            "xDecompVecCollect": xDecompVecCollect,
            "plotDataCollect": plotDataCollect,
            "df_caov_pct_all": df_caov_pct_all
    }

    return pareto_results

def pareto_front(x, y, fronts=1, sort=True):
    if len(x) != len(y):
        raise ValueError("Length of x and y must be equal")

    d = pd.DataFrame({'x': x, 'y': y})

    if sort:
        D = d.sort_values(by=['x', 'y'], ascending=[True, True])
    else:
        D = d.copy()

    Dtemp = D.copy()
    df = pd.DataFrame(columns=['x', 'y', 'pareto_front'])

    i = 1
    while len(Dtemp) >= 1 and i <= max(fronts, 1):
        Dtemp['cummin'] = Dtemp['y'].cummin()
        these = Dtemp[Dtemp['y'] == Dtemp['cummin']].copy()
        these['pareto_front'] = i
        df = pd.concat([df, these], ignore_index=True)
        Dtemp = Dtemp[~Dtemp.index.isin(these.index)]
        i += 1

    ret = pd.merge(d, df[['x', 'y', 'pareto_front']], on=['x', 'y'], how='left', sort=sort)
    return ret

def get_pareto_fronts(pareto_fronts):
    if pareto_fronts == 'auto':
        return 1
    else:
        return pareto_fronts


def run_dt_resp(respN, InputCollect, OutputModels, decompSpendDistPar, resultHypParamPar, xDecompAggPar, **kwargs):
    get_solID = decompSpendDistPar.solID[respN]
    get_spendname = decompSpendDistPar.rn[respN]
    startRW = InputCollect.rollingWindowStartWhich
    endRW = InputCollect.rollingWindowEndWhich

    get_resp = robyn_response(
        select_model=get_solID,
        metric_name=get_spendname,
        date_range="all",
        dt_hyppar=resultHypParamPar,
        dt_coef=xDecompAggPar,
        InputCollect=InputCollect,
        OutputCollect=OutputModels,
        quiet=True,
        **kwargs
    )

    mean_spend_adstocked = np.mean(get_resp.input_total[startRW:endRW])
    mean_carryover = np.mean(get_resp.input_carryover[startRW:endRW])
    dt_hyppar = resultHypParamPar[resultHypParamPar.solID == get_solID]
    chnAdstocked = pd.DataFrame({get_spendname: get_resp.input_total[startRW:endRW]})
    dt_coef = xDecompAggPar[xDecompAggPar.solID == get_solID & xDecompAggPar.rn == get_spendname][["rn", "coef"]]
    hills = get_hill_params(
        InputCollect, None, dt_hyppar, dt_coef,
        mediaSpendSorted=get_spendname,
        select_model=get_solID,
        chnAdstocked=chnAdstocked
    )
    mean_response = fx_objective(
        x=decompSpendDistPar.mean_spend[respN],
        coeff=hills.coefs_sorted,
        alpha=hills.alphas,
        inflexion=hills.inflexions,
        x_hist_carryover=mean_carryover,
        get_sum=False
    )
    dt_resp = pd.DataFrame({
        "mean_response": mean_response,
        "mean_spend_adstocked": mean_spend_adstocked,
        "mean_carryover": mean_carryover,
        "rn": decompSpendDistPar.rn[respN],
        "solID": decompSpendDistPar.solID[respN]
    })
    return dt_resp
