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

from .allocator import fx_objective, get_hill_params
from .cluster import errors_scores
from .response import robyn_response
from .transformation import adstock_weibull, saturation_hill, transform_adstock, run_transformations
#from .model import model_decomp


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

    if OutputModels["metadata"]["cores"] > 1:
        resp_collect = pd.concat(
            [run_dt_resp(respN, InputCollect, OutputModels, decompSpendDistPar, resultHypParamPar, xDecompAggPar, **kwargs) for respN in range(len(decompSpendDistPar["rn"]))]
        )
    else:
        resp_collect = pd.concat(
            [run_dt_resp(respN, InputCollect, OutputModels, decompSpendDistPar, resultHypParamPar, xDecompAggPar, **kwargs) for respN in range(len(decompSpendDistPar["rn"]))]
        )

    decompSpendDist = pd.merge(
        decompSpendDist,
        resp_collect,
        on=["solID", "rn"],
        how='left'
    )
    decompSpendDist["roi_mean"] = decompSpendDist["mean_response"] / decompSpendDist["mean_spend"]
    decompSpendDist["roi_total"] = decompSpendDist["xDecompAgg"] / decompSpendDist["total_spend"]
    decompSpendDist["cpa_mean"] = decompSpendDist["mean_spend"] / decompSpendDist["mean_response"]
    decompSpendDist["cpa_total"] = decompSpendDist["total_spend"] / decompSpendDist["xDecompAgg"]

    xDecompAgg = pd.merge(
        xDecompAgg,
        decompSpendDist[["rn", "solID", "total_spend", "mean_spend", "mean_spend_adstocked", "mean_carryover", "mean_response", "spend_share", "effect_share", "roi_mean", "roi_total", "cpa_total"]],
        on=["solID", "rn"],
        how='left'
    )

    # mediaVecCollect = []
    mediaVecCollect = pd.DataFrame()
    xDecompVecCollect = []
    plotDataCollect = {}
    df_caov_pct_all = pd.DataFrame()
    dt_mod = InputCollect["robyn_inputs"]["dt_mod"]
    dt_modRollWind = InputCollect["robyn_inputs"]["dt_modRollWind"]
    rw_start_loc = InputCollect["robyn_inputs"]["rollingWindowStartWhich"]
    rw_end_loc = InputCollect["robyn_inputs"]["rollingWindowEndWhich"]

    for pf in pareto_fronts_vec:
        plotMediaShare = xDecompAgg[
            (xDecompAgg["robynPareto"] == pf) & (xDecompAgg["rn"].isin(InputCollect["robyn_inputs"]["paid_media_spends"]))
        ]
        uniqueSol = plotMediaShare["solID"].unique()
        plotWaterfall = xDecompAgg[xDecompAgg["robynPareto"] == pf]
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
            resultHypParamLoop = resultHypParam[resultHypParam["solID"] == sid]
            get_hp_names = [name for name in InputCollect["robyn_inputs"]["hyperparameters"].keys() if not name.endswith("_penalty")]
            hypParam = resultHypParamLoop[get_hp_names]
            if InputCollect["robyn_inputs"]["adstock"] == "geometric":
                hypParam_thetas = np.array([hypParam[f"{media}_thetas"] for media in InputCollect["robyn_inputs"]["all_media"]])
                dt_geometric = pd.DataFrame({"channels": InputCollect["robyn_inputs"]["all_media"], "thetas": hypParam_thetas.flatten()})

            if InputCollect["robyn_inputs"]["adstock"] in ["weibull_cdf", "weibull_pdf"]:
                shapeVec = np.array([hypParam[f"{media}_shapes"] for media in InputCollect["robyn_inputs"]["all_media"]])
                scaleVec = np.array([hypParam[f"{media}_scales"] for media in InputCollect["robyn_inputs"]["all_media"]])
                wb_type = InputCollect["robyn_inputs"]["adstock"][9:11]
                weibullCollect = []
                n = 1
                for v1 in range(len(InputCollect["robyn_inputs"]["all_media"])):
                    dt_weibull = pd.DataFrame(
                        {
                            "x": list(range(1, InputCollect["robyn_inputs"][rollingWindowLength + 1])),
                            "decay_accumulated": adstock_weibull(
                                list(range(1, InputCollect.rollingWindowLength + 1)),
                                shape=shapeVec[v1],
                                scale=scaleVec[v1],
                                type=wb_type
                            ).thetaVecCum,
                            "type": wb_type,
                            "channel": InputCollect["robyn_inputs"]["all_media"][v1]
                        }
                    )
                    dt_weibull["halflife"] = np.argmin(np.abs(dt_weibull.decay_accumulated - 0.5))
                    max_non0 = np.max(np.where(dt_weibull.decay_accumulated > 0.001, dt_weibull.x, np.nan))
                    dt_weibull["cut_time"] = np.where(max_non0 <= 5, 2 * max_non0, np.floor(max_non0 + max_non0 / 3))
                    weibullCollect.append(dt_weibull)
                    n += 1
                weibullCollect = pd.concat(weibullCollect)
                weibullCollect = weibullCollect.loc[weibullCollect.x <= weibullCollect.cut_time.max()]
                wb_type = wb_type.upper()
            plot3data = {
                "dt_geometric": dt_geometric,
                "weibullCollect": weibullCollect,
                "wb_type": wb_type
            }
            ## 4. Spend response curve
            # Select columns from dt_mod for independent variables
            dt_transformPlot = dt_mod[["ds"] + InputCollect["robyn_inputs"]["all_media"]]
            # Add paid media spends to dt_transformPlot
            dt_transformSpend = pd.concat([dt_transformPlot[["ds"]], InputCollect["robyn_inputs"]["dt_input"][InputCollect["robyn_inputs"]["paid_media_spends"]]], axis=1)
            # Select rows within rolling window
            dt_transformSpendMod = dt_transformPlot.iloc[rw_start_loc:rw_end_loc, :]

            dt_transformAdstock = dt_transformPlot
            dt_transformSaturation = dt_transformPlot.iloc[rw_start_loc:rw_end_loc]

            m_decayRate = []
            for med in range(len(InputCollect["robyn_inputs"]["all_media"])):
                med_select = InputCollect["robyn_inputs"]["all_media"][med]
                m = dt_transformPlot[med_select]
                # Adstocking
                theta, shape, scale = None, None, None
                adstock = InputCollect["robyn_inputs"]["adstock"]
                if adstock == "geometric":
                    theta = hypParam[f"{med_select}_thetas"].iloc[0]
                if adstock.startswith("weibull"):
                    shape = hypParam[f"{med_select}_shapes"].iloc[0]
                    scale = hypParam[f"{med_select}_scales"].iloc[0]
                x_list = transform_adstock(m, adstock, theta=theta, shape=shape, scale=scale)
                m_adstocked = x_list.x_decayed
                dt_transformAdstock.loc[:, med_select] = m_adstocked
                m_adstockedRollWind = m_adstocked.loc[rw_start_loc:rw_end_loc]
                ## Saturation
                alpha = hypParam[f"{med_select}_alphas"].iloc[0]
                gamma = hypParam[f"{med_select}_gammas"].iloc[0]
                dt_transformSaturation.loc[:, med_select] = saturation_hill(
                    x=m_adstockedRollWind, alpha=alpha, gamma=gamma
                )
            dt_transformSaturationDecomp = dt_transformSaturation
            for i in range(InputCollect["robyn_inputs"]["mediaVarCount"]):
                coefs = plotWaterfall.loc[plotWaterfall.rn == InputCollect["robyn_inputs"]["all_media"][i], "coefs"].values[0]
                dt_transformSaturationDecomp.loc[:, InputCollect["robyn_inputs"]["all_media"][i]] = coefs * dt_transformSaturationDecomp.loc[:, InputCollect["robyn_inputs"]["all_media"][i]]
            dt_transformSaturationSpendReverse = dt_transformAdstock.loc[rw_start_loc:rw_end_loc, :]

            dt_scurvePlot = pd.melt(dt_transformSaturationDecomp, id_vars='ds', var_name='channel', value_name='response')

            dt_spend = pd.melt(dt_transformSaturationSpendReverse, id_vars='ds', var_name='channel', value_name='spend')
            dt_scurvePlot['channel'] = dt_scurvePlot['channel'].apply(lambda x: x.replace('_S', ''))
            dt_spend['channel'] = dt_spend['channel'].apply(lambda x: x.replace('_S', ''))
            dt_scurvePlot = pd.merge(dt_scurvePlot, dt_spend, on=['ds', 'channel'], how='left')

            # Remove outlier introduced by MM nls fitting
            dt_scurvePlot = dt_scurvePlot.loc[dt_scurvePlot.spend >= 0, :]
            dt_scurvePlotMean = plotWaterfall.loc[(plotWaterfall.solID == sid) & (~pd.isna(plotWaterfall.mean_spend)),
                ["rn", "mean_spend", "mean_spend_adstocked", "mean_carryover", "mean_response", "solID"]
            ].rename(columns={"rn": "channel"})

            # Exposure response curve
            plot4data = {
                "dt_scurvePlot": dt_scurvePlot,
                "dt_scurvePlotMean": dt_scurvePlotMean
            }

            # 5. Fitted vs actual
            selected_columns = ['ds', 'dep_var'] + InputCollect['robyn_inputs']['prophet_vars'] + InputCollect['robyn_inputs']['context_vars']
            media_columns = InputCollect['robyn_inputs']['all_media']
            dt_transformDecomp = pd.concat([
                dt_modRollWind[selected_columns],
                dt_transformSaturation[media_columns]
            ], axis=1)

            col_order = ["ds", "dep_var"] + InputCollect['robyn_inputs']['all_ind_vars']
            dt_transformDecomp = dt_transformDecomp[col_order]
            #dt_transformDecomp = dt_transformDecomp.loc[:, ~dt_transformDecomp.columns.duplicated()]

            filtered_df = xDecompAgg[xDecompAgg['solID'] == sid][['solID', 'rn', 'coefs']]
            filtered_df_agg = filtered_df.groupby(['solID', 'rn']).agg({'coefs': 'sum'}).reset_index()
            xDecompVec = filtered_df_agg.pivot(index='solID', columns='rn', values='coefs').reset_index()
            xDecompVec['(Intercept)'] = xDecompVec.get('(Intercept)', 0)
            relevant_cols = ['solID', '(Intercept)'] + [col for col in col_order if col not in ['ds', 'dep_var', 'solID', '(Intercept)'] and col in xDecompVec.columns]
            xDecompVec = xDecompVec[relevant_cols]
            intercept = xDecompVec['(Intercept)'].values[0]
            for col in relevant_cols[2:]:  # Skipping 'solID' and '(Intercept)'
                dt_transformDecomp[col] = dt_transformDecomp[col] * xDecompVec[col].values[0]

            dt_transformDecomp['intercept'] = intercept
            numeric_cols = dt_transformDecomp.select_dtypes(include=[np.number])
            dt_transformDecomp['depVarHat'] = numeric_cols.sum(axis=1)
            xDecompVec = pd.concat([dt_transformDecomp[['ds', 'dep_var', 'depVarHat', 'intercept']], dt_transformDecomp.drop(columns=['ds', 'dep_var', 'intercept'])], axis=1)
            xDecompVec['solID'] = sid
            xDecompVec = xDecompVec.loc[:, ~xDecompVec.columns.duplicated()]

            xDecompVecPlot = dt_transformDecomp[['ds', 'dep_var', 'depVarHat']].rename(columns={"dep_var": "actual", "depVarHat": "predicted"})
            xDecompVecPlot = xDecompVecPlot.loc[:, ~xDecompVecPlot.columns.duplicated()]
            xDecompVecPlot = xDecompVecPlot.rename(columns={"dep_var": "actual", "depVarHat": "predicted"})
            xDecompVecPlotMelted = pd.melt(xDecompVecPlot, id_vars=["ds"], value_vars=["actual", "predicted"], var_name="variable", value_name="value")
            rsq = xDecompAgg[xDecompAgg['solID'] == sid]['rsq_train'].iloc[0]
            plot5data = {"xDecompVecPlotMelted": xDecompVecPlotMelted, "rsq": rsq}

            # 6. Diagnostic: fitted vs residual
            plot6data = {"xDecompVecPlot": xDecompVecPlot}

            # 7. Immediate vs carryover response
            hypParamSam = resultHypParam[resultHypParam.solID == sid]
            dt_saturated_dfs = run_transformations(InputCollect, hypParamSam, adstock)
            coefs = xDecompAgg['coefs'][xDecompAgg["solID"] == sid]
            coefs.index = xDecompAgg['rn'][xDecompAgg['solID'] == sid].values
            coefs = coefs.rename('s0').to_frame()

            df_reset = coefs.reset_index()
            df_deduped = df_reset.drop_duplicates(subset=['index'], keep='first')
            coefs = df_deduped.set_index('index')
            coefs.index.name = 'rn'

            from .model import model_decomp

            decompCollect = model_decomp(
                coefs=coefs,
                y_pred=dt_saturated_dfs["dt_modSaturated"]["dep_var"],
                dt_modSaturated=dt_saturated_dfs["dt_modSaturated"],
                dt_saturatedImmediate=dt_saturated_dfs["dt_saturatedImmediate"],
                dt_saturatedCarryover=dt_saturated_dfs["dt_saturatedCarryover"],
                dt_modRollWind=dt_modRollWind,
                refreshAddedStart=InputCollect["robyn_inputs"]["refreshAddedStart"]
            )
            decompCollectCopy = decompCollect.copy()
            mediaDecompImmediate = decompCollect["mediaDecompImmediate"].drop(
                columns=["ds", "y"]
            )
            mediaDecompImmediate.columns = [f"{col}_MDI" for col in mediaDecompImmediate.columns]
            mediaDecompCarryover = decompCollect["mediaDecompCarryover"].drop(
                columns=["ds", "y"]
            )
            mediaDecompCarryover.columns = [f"{col}_MDC" for col in mediaDecompCarryover.columns]
            temp = pd.concat(
                [
                    decompCollect["xDecompVec"],
                    mediaDecompImmediate,
                    mediaDecompCarryover
                ],
                axis=1
            ).assign(solID=sid)
            vec_collect = {
                "xDecompVec": temp.drop(columns=temp.iloc[:, temp.columns.str.endswith('_MDI') | temp.columns.str.endswith('_MDC')], axis=1),
                "xDecompVecImmediate": temp.drop(columns=temp.iloc[:, temp.columns.str.endswith('_MDC')], axis=1)
                                           .drop(columns=(column for column in temp.columns if any(column == name for name in InputCollect["robyn_inputs"]["all_media"])), axis=1),
                "xDecompVecCarryover": temp.drop(columns=temp.iloc[:, temp.columns.str.endswith('_MDI')], axis=1)
                                           .drop(columns=(column for column in temp.columns if any(column == name for name in InputCollect["robyn_inputs"]["all_media"])), axis=1)
            }
            this = vec_collect["xDecompVecImmediate"].columns.str.replace("_MDI", "")
            vec_collect["xDecompVecImmediate"].columns = [col for col in this]
            vec_collect["xDecompVecCarryover"].columns = [col for col in this]
            df_caov = vec_collect["xDecompVecCarryover"][InputCollect["robyn_inputs"]["all_media"] + ["solID"]].groupby("solID").sum().reset_index()
            df_total = vec_collect["xDecompVec"][InputCollect["robyn_inputs"]["all_media"] + ["solID"]].groupby("solID").sum().reset_index()
            df_caov_pct = pd.concat([df_caov[["solID"]], df_caov[InputCollect["robyn_inputs"]["all_media"]].div(df_total[InputCollect["robyn_inputs"]["all_media"]], axis=0)], axis=1)
            df_caov_pct = df_caov_pct.melt(id_vars="solID", var_name="rn", value_name="carryover_pct")
            df_caov_pct.replace(pd.NA, 0, inplace=True)

            df_caov_pct_all = pd.concat([df_caov_pct_all, df_caov_pct], ignore_index=True)
            # Gather everything in an aggregated format
            xDecompVecImmeCaov = pd.concat(
                [
                    vec_collect["xDecompVecImmediate"][InputCollect["robyn_inputs"]["all_media"]+ ["solID"]].assign(type="Immediate"),
                    vec_collect["xDecompVecCarryover"][InputCollect["robyn_inputs"]["all_media"]+ ["solID"]].assign(type="Carryover")
                ],
                ignore_index=True
            ).melt(
                id_vars=["solID", "type"],
                var_name="rn",
                value_name="value"
            ).groupby(["solID", "rn", "type"]).sum().reset_index()
            xDecompVecImmeCaov["percentage"] = xDecompVecImmeCaov.groupby(["solID", "rn"])["value"].transform(
                lambda x: x / x.sum()
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
            # mediaVecCollect = pd.concat(
            #     [
            #         dt_transformAdstock.assign(type="adstockedMedia", solID=sid),
            #         dt_transformPlot.assign(type="rawMedia", solID=sid),
            #         dt_transformSpend.assign(type="rawSpend", solID=sid),
            #         dt_transformSpendMod.assign(type="predictedExposure", solID=sid),
            #         dt_transformSaturation.assign(type="saturatedMedia", solID=sid),
            #         dt_transformSaturationSpendReverse.assign(type="saturatedSpendReversed", solID=sid),
            #         dt_transformSaturationDecomp.assign(type="decompMedia", solID=sid)
            #     ],
            #     ignore_index=True
            # )
            new_data = pd.concat(
                [
                    dt_transformAdstock.assign(type="adstockedMedia", solID=sid),
                    dt_transformPlot.assign(type="rawMedia", solID=sid),
                    dt_transformSpend.assign(type="rawSpend", solID=sid),
                    dt_transformSpendMod.assign(type="predictedExposure", solID=sid),
                    dt_transformSaturation.assign(type="saturatedMedia", solID=sid),
                    dt_transformSaturationSpendReverse.assign(type="saturatedSpendReversed", solID=sid),
                    dt_transformSaturationDecomp.assign(type="decompMedia", solID=sid)
                ],
                ignore_index=True
            )
            mediaVecCollect = pd.concat([mediaVecCollect, new_data], ignore_index=True)
            xDecompVecCollect.append(xDecompVec)
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

    xDecompVecCollect = pd.concat(xDecompVecCollect, ignore_index=True)


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
    get_solID = decompSpendDistPar.solID.values[respN]
    get_spendname = decompSpendDistPar.rn.values[respN]
    startRW = InputCollect["robyn_inputs"]["rollingWindowStartWhich"]
    endRW = InputCollect["robyn_inputs"]["rollingWindowEndWhich"]

    get_resp = robyn_response(
        select_model=get_solID,
        metric_name=get_spendname,
        date_range="all",
        dt_hyppar=resultHypParamPar,
        dt_coef=xDecompAggPar,
        InputCollect=InputCollect,
        OutputCollect=OutputModels,
        quiet=True
    )

    mean_spend_adstocked = np.mean(get_resp['input_total'][startRW:endRW])
    mean_carryover = np.mean(get_resp['input_carryover'][startRW:endRW])
    dt_hyppar = resultHypParamPar[resultHypParamPar.solID == get_solID]
    chnAdstocked = pd.DataFrame({get_spendname: get_resp['input_total'][startRW:endRW]})
    dt_coef = xDecompAggPar[(xDecompAggPar.solID == get_solID) & (xDecompAggPar.rn == get_spendname)][["rn", "coefs"]]
    hills = get_hill_params(
        InputCollect, None, dt_hyppar, dt_coef,
        mediaSpendSorted=get_spendname,
        select_model=get_solID,
        chnAdstocked=chnAdstocked
    )
    mean_response = fx_objective(
        x=decompSpendDistPar.mean_spend.values[respN],
        coeff=hills['coefs_sorted'],
        alpha=hills['alphas'],
        inflexion=hills['inflexions'],
        x_hist_carryover=mean_carryover,
        get_sum=False
    )
    dt_resp = pd.DataFrame({
        "mean_response": mean_response,
        "mean_spend_adstocked": mean_spend_adstocked,
        "mean_carryover": mean_carryover,
        "rn": decompSpendDistPar.rn.values[respN],
        "solID": decompSpendDistPar.solID.values[respN]
    })
    return dt_resp
