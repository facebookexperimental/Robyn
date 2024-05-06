# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import multiprocessing
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
import seaborn as sns
from .auxiliary import robyn_palette
from tqdm import tqdm
from scipy.stats.mstats import winsorize
import json

# Plotting using plotnine
from plotnine import (
    aes,
    element_blank,
    facet_grid,
    geom_bar,
    geom_text,
    ggplot,
    labs,
    scale_fill_manual,
    scale_y_continuous,
    theme,
)


def robyn_plots(InputCollect, OutputCollect, export=True, plot_folder=None, **kwargs):
    def check_class(class_name, obj):
        # You should implement the check_class function according to your needs
        pass

    pareto_fronts = OutputCollect["pareto_fronts"]
    hyper_fixed = OutputCollect["hyper_fixed"]
    temp_all = OutputCollect["allPareto"]
    all_plots = {}

    if not hyper_fixed:
        # Prophet
        if (
            "prophet_vars" in InputCollect and len(InputCollect["prophet_vars"]) > 0
        ) or ("factor_vars" in InputCollect and len(InputCollect["factor_vars"]) > 0):
            dt_plotProphet = InputCollect["dt_mod"][
                ["ds", "dep_var"]
                + InputCollect["prophet_vars"]
                + InputCollect["factor_vars"]
            ]
            dt_plotProphet = dt_plotProphet.melt(
                id_vars=["ds"], var_name="variable", value_name="value"
            )
            dt_plotProphet["ds"] = pd.to_datetime(
                dt_plotProphet["ds"], origin="1970-01-01"
            )
            all_plots["pProphet"] = pProphet = sns.lineplot(
                data=dt_plotProphet,
                x="ds",
                y="value",
                hue="variable",
                palette="steelblue",
            )
            pProphet.set(title="Prophet decomposition", xlabel=None, ylabel=None)
            pProphet = pProphet + sns.scale_y_log()

            if export:
                pProphet.figure.savefig(
                    f"{plot_folder}prophet_decomp.png",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

        # Spend exposure model (Code for this part is commented out in the original R code)

        # Hyperparameter sampling distribution
        if len(temp_all) > 0:
            resultHypParam = temp_all["resultHypParam"]
            hpnames_updated = [
                re.sub("lambda", "lambda_hp", col)
                for col in InputCollect["hyperparameters"]
            ]
            resultHypParam_melted = pd.melt(resultHypParam[hpnames_updated])
            resultHypParam_melted["variable"] = np.where(
                resultHypParam_melted["variable"] == "lambda_hp",
                "lambda",
                resultHypParam_melted["variable"],
            )
            resultHypParam_melted["type"] = [
                col.split("_")[-1] for col in resultHypParam_melted["variable"]
            ]
            resultHypParam_melted["channel"] = [
                col.replace(f'_{resultHypParam_melted["type"].iloc[i]}', "")
                for i, col in enumerate(resultHypParam_melted["variable"])
            ]
            resultHypParam_melted["type"] = pd.Categorical(
                resultHypParam_melted["type"],
                categories=np.unique(resultHypParam_melted["type"]),
            )
            resultHypParam_melted["channel"] = pd.Categorical(
                resultHypParam_melted["channel"],
                categories=np.unique(resultHypParam_melted["channel"]),
            )

            all_plots["pSamp"] = pSamp = sns.violinplot(
                data=resultHypParam_melted,
                x="value",
                y="channel",
                hue="channel",
                palette="viridis",
                inner=None,
                scale="width",
            )
            pSamp.set(
                title="Hyperparameters Optimization Distributions",
                xlabel="Hyperparameter space",
                ylabel=None,
            )

            if export:
                pSamp.figure.savefig(
                    f"{plot_folder}hypersampling.png",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

        # Pareto front
        if len(temp_all) > 0:
            pareto_fronts_vec = list(range(1, pareto_fronts + 1))
            resultHypParam = temp_all["resultHypParam"]

            if (
                "calibration_input" in InputCollect
                and InputCollect["calibration_input"] is not None
            ):
                resultHypParam["iterations"] = np.where(
                    resultHypParam["robynPareto"].isna(),
                    np.nan,
                    resultHypParam["iterations"],
                )
                resultHypParam = resultHypParam[~resultHypParam["robynPareto"].isna()]
                resultHypParam = resultHypParam.sort_values(
                    by=["robynPareto"], ascending=False
                )

            calibrated = (
                "calibration_input" in InputCollect
                and InputCollect["calibration_input"] is not None
            )

            pParFront = sns.scatterplot(
                data=resultHypParam,
                x="nrmse",
                y="decomp.rssd",
                hue="iterations",
                palette="coolwarm",
                alpha=0.7,
                size="mape",
            )
            pParFront.set(
                title="Multi-objective Evolutionary Performance with Calibration"
                if calibrated
                else "Multi-objective Evolutionary Performance",
                xlabel="NRMSE",
                ylabel="DECOMP.RSSD",
            )

            for pfs in range(1, max(pareto_fronts_vec) + 1):
                pf_color = "coral2" if pfs == 2 else "coral" if pfs == 3 else "coral"
                temp = resultHypParam[resultHypParam["robynPareto"] == pfs]
                if len(temp) > 1:
                    pParFront.plot(
                        temp["nrmse"],
                        temp["decomp.rssd"],
                        color=pf_color,
                        linestyle="-",
                    )

            all_plots["pParFront"] = pParFront

            if export:
                pParFront.figure.savefig(
                    f"{plot_folder}pareto_front.png",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

        # Ridgeline model convergence (Code for this part is commented out in the original R code)

    get_height = int(np.ceil(12 * OutputCollect["OutputModels"]["trials"] / 3))

    if OutputCollect["OutputModels"]["ts_validation"]:
        # You should implement the ts_validation function according to your needs
        ts_validation_plot = ts_validation(
            OutputCollect["OutputModels"], quiet=True, **kwargs
        )
        ts_validation_plot.figure.savefig(
            f"{plot_folder}ts_validation.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

    return all_plots


def robyn_onepagers(
    InputCollect,
    OutputCollect,
    select_model=None,
    quiet=False,
    export=True,
    plot_folder=None,
    baseline_level=0,
    *args,
    **kwargs,
):
    def isTRUE(val):
        return val

    check_class("robyn_outputs", OutputCollect)

    if True:
        pareto_fronts = OutputCollect["pareto_fronts"]
        hyper_fixed = OutputCollect["hyper_fixed"]
        resultHypParam = pd.DataFrame(OutputCollect["resultHypParam"])
        xDecompAgg = pd.DataFrame(OutputCollect["xDecompAgg"])
        val = isTRUE(OutputCollect["OutputModels"]["ts_validation"])
        sid = None  # for parallel loops

    if select_model is not None:
        if "clusters" in select_model:
            select_model = OutputCollect["clusters"]["models"]["solID"]
        resultHypParam = resultHypParam[resultHypParam["solID"].isin(select_model)]
        xDecompAgg = xDecompAgg[xDecompAgg["solID"].isin(select_model)]
        if not quiet and resultHypParam.shape[0] > 1:
            print(
                ">> Generating only cluster results one-pagers (",
                resultHypParam.shape[0],
                ")...",
            )

    # Baseline variables
    bvars = baseline_vars(InputCollect, baseline_level)

    # Prepare for parallel plotting
    if check_parallel_plot() and OutputCollect["cores"] > 1:
        registerDoParallel(OutputCollect["cores"])
    else:
        registerDoSEQ()

    if not hyper_fixed:
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))
        count_mod_out = resultHypParam[
            resultHypParam["robynPareto"].isin(pareto_fronts_vec)
        ].shape[0]
    else:
        pareto_fronts_vec = [1]
        count_mod_out = resultHypParam.shape[0]

    all_fronts = list(filter(lambda x: not pd.isna(x), xDecompAgg["robynPareto"]))
    all_fronts = sorted(all_fronts)
    if not all(
        pareto_fronts_vec[i] in all_fronts for i in range(len(pareto_fronts_vec))
    ):
        pareto_fronts_vec = all_fronts

    if check_parallel_plot():
        if not quiet and resultHypParam.shape[0] > 1:
            print(
                ">> Plotting",
                count_mod_out,
                "selected models on",
                OutputCollect["cores"],
                "cores...",
            )
    else:
        if not quiet and resultHypParam.shape[0] > 1:
            print(
                ">> Plotting",
                count_mod_out,
                "selected models on 1 core (MacOS fallback)...",
            )

    if not quiet and count_mod_out > 1:
        pbplot = tqdm(total=count_mod_out)
    temp = OutputCollect["allPareto"]["plotDataCollect"]
    all_plots = []
    cnt = 0

    for pf in pareto_fronts_vec:
        plotMediaShare = xDecompAgg[
            (xDecompAgg["robynPareto"] == pf)
            & (xDecompAgg["rn"].isin(InputCollect["paid_media_spends"]))
        ]
        uniqueSol = plotMediaShare["solID"].unique()

        def parallel_result(sid):
            if True:
                plotMediaShareLoop = plotMediaShare[plotMediaShare["solID"] == sid]
                rsq_train_plot = round(plotMediaShareLoop["rsq_train"].iloc[0], 4)
                rsq_val_plot = round(plotMediaShareLoop["rsq_val"].iloc[0], 4)
                rsq_test_plot = round(plotMediaShareLoop["rsq_test"].iloc[0], 4)
                nrmse_train_plot = round(plotMediaShareLoop["nrmse_train"].iloc[0], 4)
                nrmse_val_plot = round(plotMediaShareLoop["nrmse_val"].iloc[0], 4)
                nrmse_test_plot = round(plotMediaShareLoop["nrmse_test"].iloc[0], 4)
                decomp_rssd_plot = round(plotMediaShareLoop["decomp.rssd"].iloc[0], 4)

            if InputCollect["calibration_input"] is not None:
                mape_lift_plot = round(plotMediaShareLoop["mape"].iloc[0], 4)
            else:
                mape_lift_plot = None

            train_size = round(plotMediaShareLoop["train_size"].iloc[0], 4)

            if val:
                errors = [
                    "Adj.R2: train = {}, val = {}, test = {} |".format(
                        rsq_train_plot, rsq_val_plot, rsq_test_plot
                    ),
                    "NRMSE: train = {}, val = {}, test = {} |".format(
                        nrmse_train_plot, nrmse_val_plot, nrmse_test_plot
                    ),
                    "DECOMP.RSSD = {} |".format(decomp_rssd_plot),
                    "MAPE = {}".format(mape_lift_plot)
                    if mape_lift_plot is not None
                    else "",
                ]
            else:
                errors = [
                    "Adj.R2: train = {} |".format(rsq_train_plot),
                    "NRMSE: train = {} |".format(nrmse_train_plot),
                    "DECOMP.RSSD = {} |".format(decomp_rssd_plot),
                    "MAPE = {}".format(mape_lift_plot)
                    if mape_lift_plot is not None
                    else "",
                ]
            errors = " ".join(errors)

            ## 1. Spend x effect share comparison
            plotMediaShareLoopBar = temp[sid]["plot1data"]["plotMediaShareLoopBar"]
            plotMediaShareLoopLine = temp[sid]["plot1data"]["plotMediaShareLoopLine"]
            ySecScale = temp[sid]["plot1data"]["ySecScale"]

            plotMediaShareLoopBar["variable"] = (
                plotMediaShareLoopBar["variable"].str.replace("_", " ").str.title()
            )

            type = "CPA" if InputCollect["dep_var_type"] == "conversion" else "ROI"
            plotMediaShareLoopLine["type_colour"] = type_colour = "#03396C"

            p1 = (
                ggplot(plotMediaShareLoopBar, aes(x="rn", y="value", fill="variable"))
                + geom_bar(stat="identity", width=0.5, position="dodge")
                + geom_text(
                    aes(y=0, label="value"),
                    format_string="{:.1%}",
                    hjust=-0.1,
                    position=position_dodge(width=0.5),
                    fontweight="bold",
                )
                + geom_line(
                    plotMediaShareLoopLine,
                    aes(x="rn", y="value / ySecScale", group=1),
                    color=type_colour,
                    size=1,
                )
                + geom_point(
                    plotMediaShareLoopLine,
                    aes(x="rn", y="value / ySecScale", group=1),
                    color=type_colour,
                    size=3.5,
                )
                + geom_text(
                    plotMediaShareLoopLine,
                    aes(label="value", x="rn", y="value / ySecScale", group=1),
                    format_string="{:.2f}",
                    color=type_colour,
                    fontweight="bold",
                    hjust=-0.4,
                    size=10,
                )
                + scale_y_continuous(labels=percent_format())
                + coord_flip()
                + theme(axis_text_x=element_blank(), legend_position="top")
                + scale_fill_brewer(palette=3)
                + labs(
                    title=f"Total Spend% VS Effect% with total {type}",
                    y="Total Share by Channel",
                )
            )

            # 2. Waterfall
            plotWaterfallLoop = temp[sid]["plot2data"]["plotWaterfallLoop"]
            plotWaterfallLoop = plotWaterfallLoop.assign(
                rn=np.where(
                    plotWaterfallLoop["rn"].isin(bvars),
                    f"Baseline_L{baseline_level}",
                    plotWaterfallLoop["rn"],
                )
            )
            plotWaterfallLoop = (
                plotWaterfallLoop.groupby("rn")
                .agg({"xDecompAgg": "sum", "xDecompPerc": "sum"})
                .reset_index()
            )
            plotWaterfallLoop = plotWaterfallLoop.sort_values("xDecompPerc")
            plotWaterfallLoop["end"] = 1 - plotWaterfallLoop["xDecompPerc"].cumsum()
            plotWaterfallLoop["start"] = plotWaterfallLoop["end"].shift()
            plotWaterfallLoop["start"] = plotWaterfallLoop["start"].fillna(1)
            plotWaterfallLoop["id"] = range(1, len(plotWaterfallLoop) + 1)
            plotWaterfallLoop["rn"] = pd.Categorical(plotWaterfallLoop["rn"])
            plotWaterfallLoop["sign"] = np.where(
                plotWaterfallLoop["xDecompPerc"] >= 0, "Positive", "Negative"
            )

            p2 = (
                ggplot(plotWaterfallLoop, aes(x="id", fill="sign"))
                + geom_rect(
                    aes(
                        x="rn",
                        xmin="id - 0.45",
                        xmax="id + 0.45",
                        ymin="end",
                        ymax="start",
                    ),
                    stat="identity",
                )
                + scale_x_discrete(
                    "",
                    breaks=plotWaterfallLoop["rn"].cat.categories,
                    labels=plotWaterfallLoop["rn"].cat.categories,
                )
                + scale_y_percent()
                + scale_fill_manual(
                    values={"Positive": "#59B3D2", "Negative": "#E5586E"}
                )
                + theme_lares(background="white", legend="top")
                + geom_text(
                    mapping=aes(label="xDecompAggFormatted"),
                    y=plotWaterfallLoop.apply(
                        lambda x: x["end"] + x["xDecompPerc"] / 2, axis=1
                    ),
                    fontface="bold",
                    lineheight=0.7,
                )
                + coord_flip()
                + labs(
                    title="Response Decomposition Waterfall by Predictor",
                    x=None,
                    y=None,
                    fill="Sign",
                )
            )

            # Formatting the xDecompAgg column
            plotWaterfallLoop["xDecompAggFormatted"] = plotWaterfallLoop.apply(
                lambda x: f"{x['xDecompAgg']:.2f}\n{round(x['xDecompPerc'] * 100, 1)}%",
                axis=1,
            )

            # 3. Adstock rate
            if InputCollect["adstock"] == "geometric":
                dt_geometric = temp[sid]["plot3data"]["dt_geometric"]
                p3 = (
                    ggplot(
                        dt_geometric,
                        aes(x=".data$channels", y=".data$thetas", fill="coral"),
                    )
                    + geom_bar(stat="identity", width=0.5)
                    + theme_lares(background="white", legend="none", grid="Xx")
                    + coord_flip()
                    + geom_text(
                        aes(label=formatNum(100 * ".data$thetas", 1, pos="%")),
                        hjust=-0.1,
                        position=position_dodge(width=0.5),
                        fontface="bold",
                    )
                    + scale_y_percent(limit=[0, 1])
                    + labs(
                        title="Geometric Adstock: Fixed Rate Over Time",
                        y=f"Thetas [by {InputCollect['intervalType']}]",
                        x=None,
                    )
                )

            if InputCollect["adstock"] in ["weibull_cdf", "weibull_pdf"]:
                weibullCollect = temp[sid]["plot3data"]["weibullCollect"]
                wb_type = temp[sid]["plot3data"]["wb_type"]
                p3 = (
                    ggplot(
                        weibullCollect,
                        aes(
                            x=".data$x",
                            y=".data$decay_accumulated",
                            color=".data$channel",
                        ),
                    )
                    + geom_line()
                    + facet_wrap("~ .data$channel")
                    + geom_hline(yintercept=0.5, linetype="dashed", color="gray")
                    + geom_text(
                        x=max(".data$x"),
                        y=0.5,
                        vjust=-0.5,
                        hjust=1,
                        label="Halflife",
                        color="gray",
                    )
                    + theme_lares(background="white", legend="none", grid="Xx")
                    + labs(
                        title=f"Weibull {wb_type} Adstock: Flexible Rate Over Time",
                        x=f"Time unit [{InputCollect['intervalType']}s]",
                        y=None,
                    )
                )

            # 4. Response curves
            dt_scurvePlot = temp["sid"]["plot4data"]["dt_scurvePlot"]
            dt_scurvePlotMean = temp["sid"]["plot4data"]["dt_scurvePlotMean"]
            trim_rate = 1.3

            if trim_rate > 0:
                dt_scurvePlot = dt_scurvePlot[
                    (
                        dt_scurvePlot["spend"]
                        < dt_scurvePlotMean["mean_spend_adstocked"].max() * trim_rate
                    )
                    & (
                        dt_scurvePlot["response"]
                        < dt_scurvePlotMean["mean_response"].max() * trim_rate
                    )
                ]
                dt_scurvePlot = dt_scurvePlot.merge(
                    dt_scurvePlotMean[["channel", "mean_carryover"]],
                    on="channel",
                    how="left",
                )

            if "channel" not in dt_scurvePlotMean.columns:
                dt_scurvePlotMean["channel"] = dt_scurvePlotMean["rn"]

            p4 = (
                ggplot(dt_scurvePlot, aes(x="spend", y="response", color="channel"))
                + geom_line()
                + geom_area(
                    dt_scurvePlot[
                        dt_scurvePlot["spend"] <= dt_scurvePlot["mean_carryover"]
                    ],
                    aes(x="spend", y="response", fill="channel"),
                    position="stack",
                    alpha=0.4,
                    show_legend=False,
                )
                + geom_point(
                    dt_scurvePlotMean, aes(x="mean_spend_adstocked", y="mean_response")
                )
                + geom_text(
                    dt_scurvePlotMean,
                    aes(
                        x="mean_spend_adstocked",
                        y="mean_response",
                        label="mean_spend_adstocked",
                    ),
                    format_string="{:.2e}",
                    hjust=-0.2,
                    show_legend=False,
                )
                + theme(
                    legend_position=(0.9, 0.2),
                    legend_background=element_rect(fill="grey98", color="grey90"),
                )
                + labs(
                    title="Response Curves and Mean Spends by Channel",
                    x="Spend (carryover + immediate)",
                    y="Response",
                )
                + scale_y_continuous(labels=scientific_format())
                + scale_x_continuous(labels=scientific_format())
            )

            # 5. Fitted vs actual
            xDecompVecPlotMelted = temp[sid]["plot5data"]["xDecompVecPlotMelted"]
            xDecompVecPlotMelted = xDecompVecPlotMelted.assign(
                linetype=np.where(
                    xDecompVecPlotMelted["variable"] == "predicted", "solid", "dotted"
                ),
                variable=xDecompVecPlotMelted["variable"].str.title(),
                ds=pd.to_datetime(xDecompVecPlotMelted["ds"], origin="1970-01-01"),
            )

            p5 = (
                ggplot(
                    xDecompVecPlotMelted,
                    aes(x=".data$ds", y=".data$value", color=".data$variable"),
                )
                + geom_path(aes(linetype=".data$linetype"), size=0.6)
                + theme_lares(background="white", legend="top", pal=2)
                + scale_y_abbr()
                + guides(linetype="none")
                + labs(
                    title="Actual vs. Predicted Response",
                    x="Date",
                    y="Response",
                    color=None,
                )
            )

            if val:
                days = sorted(xDecompVecPlotMelted["ds"].unique())
                ndays = len(days)
                train_cut = round(ndays * train_size)
                val_cut = train_cut + round(ndays * (1 - train_size) / 2)

                p5 = (
                    p5
                    + geom_vline(
                        xintercept=days[train_cut], colour="#39638b", alpha=0.8
                    )
                    + geom_text(
                        x=days[train_cut],
                        y=np.inf,
                        hjust=0,
                        vjust=1.2,
                        angle=270,
                        colour="#39638b",
                        alpha=0.5,
                        size=3.2,
                        label=f"Train: {formatNum(100 * train_size, 1, pos='%')}",
                    )
                    + geom_vline(xintercept=days[val_cut], colour="#39638b", alpha=0.8)
                    + geom_text(
                        x=days[val_cut],
                        y=np.inf,
                        hjust=0,
                        vjust=1.2,
                        angle=270,
                        colour="#39638b",
                        alpha=0.5,
                        size=3.2,
                        label=f"Validation: {formatNum(100 * (1 - train_size) / 2, 1, pos='%')}",
                    )
                    + geom_vline(
                        xintercept=days[ndays - 1], colour="#39638b", alpha=0.8
                    )
                    + geom_text(
                        x=days[ndays - 1],
                        y=np.inf,
                        hjust=0,
                        vjust=1.2,
                        angle=270,
                        colour="#39638b",
                        alpha=0.5,
                        size=3.2,
                        label=f"Test: {formatNum(100 * (1 - train_size) / 2, 1, pos='%')}",
                    )
                )

            # 6. Diagnostic: fitted vs residual
            xDecompVecPlot = temp[sid]["plot6data"]["xDecompVecPlot"]

            p6 = (
                qplot(
                    x=".data$predicted",
                    y=".data$actual - .data$predicted",
                    data=xDecompVecPlot,
                )
                + geom_hline(yintercept=0)
                + geom_smooth(se=True, method="loess", formula="y ~ x")
                + scale_x_abbr()
                + scale_y_abbr()
                + theme_lares(background="white")
                + labs(x="Fitted", y="Residual", title="Fitted vs. Residual")
            )

            # 7. Immediate vs carryover
            df_imme_caov = temp[sid]["plot7data"]

            p7 = (
                df_imme_caov.assign(
                    type=pd.Categorical(
                        df_imme_caov["type"], categories=["Carryover", "Immediate"]
                    )
                ).pipe(
                    ggplot,
                    aes(
                        x=".data$percentage",
                        y=".data$rn",
                        fill="reorder(.data$type, as.integer(.data$type))",
                        label='paste0(round(.data$percentage * 100), "%")',
                    ),
                )
                + geom_bar(stat="identity", width=0.5)
                + geom_text(position=position_stack(vjust=0.5))
                + scale_fill_manual(
                    values={"Immediate": "#59B3D2", "Carryover": "coral"}
                )
                + scale_x_percent()
                + theme_lares(background="white", legend="top", grid="Xx")
                + labs(
                    x="% Response",
                    y=None,
                    fill=None,
                    title="Immediate vs. Carryover Response Percentage",
                )
            )

            # 8. Bootstrapped ROI/CPA with CIs
            if "ci_low" in xDecompAgg.columns:
                metric = (
                    "CPA" if InputCollect["dep_var_type"] == "conversion" else "ROI"
                )
                p8 = (
                    xDecompAgg.loc[
                        ~xDecompAgg["ci_low"].isna() & (xDecompAgg["solID"] == sid)
                    ]
                    .loc[:, ["rn", "solID", "boot_mean", "ci_low", "ci_up"]]
                    .pipe(ggplot, aes(x=".data$rn", y=".data$boot_mean"))
                    + geom_point(size=3)
                    + geom_text(
                        aes(label="signif(.data$boot_mean, 2)"), vjust=-0.7, size=3.3
                    )
                    + geom_text(
                        aes(y=".data$ci_low", label="signif(.data$ci_low, 2)"),
                        hjust=1.1,
                        size=2.8,
                    )
                    + geom_text(
                        aes(y=".data$ci_up", label="signif(.data$ci_up, 2)"),
                        hjust=-0.1,
                        size=2.8,
                    )
                    + geom_errorbar(
                        aes(ymin=".data$ci_low", ymax=".data$ci_up"), width=0.25
                    )
                    + labs(
                        title=f"In-cluster bootstrapped {metric} with 95% CI & mean",
                        x=None,
                        y=None,
                    )
                    + coord_flip()
                    + theme_lares(background="white")
                )

                if metric == "ROI":
                    p8 += geom_hline(
                        yintercept=1, alpha=0.5, colour="grey50", linetype="dashed"
                    )
            else:
                p8 = lares.noPlot("No bootstrap results")

            # Aggregate one-pager plots and export
            ver = str(utils.packageVersion("Robyn"))
            rver = utils.sessionInfo().R_version
            onepagerTitle = f"One-pager for Model ID: {sid}"
            onepagerCaption = f"Robyn v{ver} [R-{rver.major}.{rver.minor}]"
            get_height = len(plotMediaShareLoopLine["rn"].unique()) / 5

            pg = (
                (p2 + p5) / (p1 + p8) / (p3 + p7) / (p4 + p6)
                + plot_layout(heights=[get_height, get_height, get_height, 1])
                + plot_annotation(
                    title=onepagerTitle,
                    subtitle=errors,
                    theme=theme_lares(background="white"),
                    caption=onepagerCaption,
                )
            )

            all_plots[sid] = pg

            if export:
                filename = f"{plot_folder}{sid}.png"
                ggsave(
                    filename=filename,
                    plot=pg,
                    limitsize=False,
                    dpi=400,
                    width=17,
                    height=19,
                )
                if count_mod_out == 1:
                    print(f"Exporting charts as: {filename}")

            if check_parallel_plot() and not quiet and count_mod_out > 1:
                cnt += 1

            return all_plots

        with multiprocessing.Pool() as pool:
            results = pool.map(parallel_result, uniqueSol)

        if not quiet and count_mod_out > 1:
            cnt += len(uniqueSol)

    if not quiet and count_mod_out > 1:
        pbplot.close()

    return results[0]


def allocation_plots(
    InputCollect,
    OutputCollect,
    dt_optimOut,
    select_model,
    scenario,
    eval_list,
    export=True,
    plot_folder=None,
    quiet=False,
    **kwargs,
):
    outputs = {}

    adstocked = "(adstocked**) " if dt_optimOut["adstocked"][0] else ""
    total_spend_increase = round(mean(dt_optimOut["optmSpendUnitTotalDelta"]) * 100, 1)
    total_response_increase = round(
        mean(dt_optimOut["optmResponseUnitTotalLift"]) * 100, 1
    )

    subtitle = f"Total {adstocked}spend increase: {total_spend_increase}%\nTotal response increase: {total_response_increase}% with optimised spend allocation"

    metric = "ROAS" if InputCollect["dep_var_type"] == "revenue" else "CPA"

    if metric == "ROAS":
        formulax1 = "ROAS = total response / raw spend | mROAS = marginal response / marginal spend"
        formulax2 = "When reallocating budget, mROAS converges across media within respective bounds"
    else:
        formulax1 = "CPA = raw spend / total response | mCPA = marginal spend / marginal response"
        formulax2 = "When reallocating budget, mCPA converges across media within respective bounds"

    plotDT_scurveMeanResponse = OutputCollect["xDecompAgg"][
        (OutputCollect["xDecompAgg"]["solID"] == select_model)
        & (OutputCollect["xDecompAgg"]["rn"].isin(InputCollect["paid_media_spends"]))
    ]

    # Calculate the statistics
    rsq_train_plot = round(plotDT_scurveMeanResponse["rsq_train"].iloc[0], 4)
    rsq_val_plot = round(plotDT_scurveMeanResponse["rsq_val"].iloc[0], 4)
    rsq_test_plot = round(plotDT_scurveMeanResponse["rsq_test"].iloc[0], 4)
    nrmse_train_plot = round(plotDT_scurveMeanResponse["nrmse_train"].iloc[0], 4)
    nrmse_val_plot = round(plotDT_scurveMeanResponse["nrmse_val"].iloc[0], 4)
    nrmse_test_plot = round(plotDT_scurveMeanResponse["nrmse_test"].iloc[0], 4)
    decomp_rssd_plot = round(plotDT_scurveMeanResponse["decomp_rssd"].iloc[0], 4)
    mape_lift_plot = (
        round(plotDT_scurveMeanResponse["mape"].iloc[0], 4)
        if "calibration_input" in InputCollect
        else None
    )

    # Create the error message string
    if OutputCollect["OutputModels"]["ts_validation"]:
        errors = f"Adj.R2: train = {rsq_train_plot}, val = {rsq_val_plot}, test = {rsq_test_plot} | NRMSE: train = {nrmse_train_plot}, val = {nrmse_val_plot}, test = {nrmse_test_plot} | DECOMP.RSSD = {decomp_rssd_plot} | MAPE = {mape_lift_plot}"
    else:
        errors = f"Adj.R2: train = {rsq_train_plot} | NRMSE: train = {nrmse_train_plot} | DECOMP.RSSD = {decomp_rssd_plot} | MAPE = {mape_lift_plot}"

    init_total_spend = dt_optimOut["initSpendTotal"][0]
    init_total_response = dt_optimOut["initResponseTotal"][0]
    init_total_roi = (
        init_total_response / init_total_spend
        if init_total_spend != 0
        else float("inf")
    )
    init_total_cpa = (
        init_total_spend / init_total_response
        if init_total_response != 0
        else float("inf")
    )

    optm_total_spend_bounded = dt_optimOut["optmSpendTotal"][0]
    optm_total_response_bounded = dt_optimOut["optmResponseTotal"][0]
    optm_total_roi_bounded = (
        optm_total_response_bounded / optm_total_spend_bounded
        if optm_total_spend_bounded != 0
        else float("inf")
    )
    optm_total_cpa_bounded = (
        optm_total_spend_bounded / optm_total_response_bounded
        if optm_total_response_bounded != 0
        else float("inf")
    )

    optm_total_spend_unbounded = dt_optimOut["optmSpendTotalUnbound"][0]
    optm_total_response_unbounded = dt_optimOut["optmResponseTotalUnbound"][0]
    optm_total_roi_unbounded = (
        optm_total_response_unbounded / optm_total_spend_unbounded
        if optm_total_spend_unbounded != 0
        else float("inf")
    )
    optm_total_cpa_unbounded = (
        optm_total_spend_unbounded / optm_total_response_unbounded
        if optm_total_response_unbounded != 0
        else float("inf")
    )

    bound_mult = dt_optimOut["unconstr_mult"][0]

    optm_topped_unbounded = optm_topped_bounded = any_topped = False

    if "total_budget" in eval_list and eval_list["total_budget"] is not None:
        optm_topped_bounded = round(optm_total_spend_bounded) < round(
            eval_list["total_budget"]
        )
        optm_topped_unbounded = round(optm_total_spend_unbounded) < round(
            eval_list["total_budget"]
        )
        any_topped = optm_topped_bounded or optm_topped_unbounded

        if optm_topped_bounded and not quiet:
            print(
                "NOTE: Given the upper/lower constrains, the total budget can't be fully allocated (^)"
            )

    levs1 = eval_list["levs1"]

    if scenario == "max_response":
        levs2 = [
            "Initial",
            f"Bounded{'^' if optm_topped_bounded else ''}",
            f"Unbounded{'^' if optm_topped_unbounded else ''} x{bound_mult}",
        ]
    elif scenario == "target_efficiency":
        levs2 = levs1

    resp_metric = pd.DataFrame(
        {
            "type": pd.Categorical(levs1, categories=levs1),
            "type_lab": pd.Categorical(levs2, categories=levs2),
            "total_spend": [
                init_total_spend,
                optm_total_spend_bounded,
                optm_total_spend_unbounded,
            ],
            "total_response": [
                init_total_response,
                optm_total_response_bounded,
                optm_total_response_unbounded,
            ],
            "total_response_lift": [
                0,
                dt_optimOut["optmResponseUnitTotalLift"][0],
                dt_optimOut["optmResponseUnitTotalLiftUnbound"][0],
            ],
            "total_roi": [
                init_total_roi,
                optm_total_roi_bounded,
                optm_total_roi_unbounded,
            ],
            "total_cpa": [
                init_total_cpa,
                optm_total_cpa_bounded,
                optm_total_cpa_unbounded,
            ],
        }
    )

    df_roi = (
        resp_metric.assign(
            spend=resp_metric["total_spend"], response=resp_metric["total_response"]
        )[["type", "spend", "response"]]
        .melt(id_vars="type", var_name="name")
        .merge(resp_metric, on="type")
    )

    df_roi["name"] = pd.Categorical(
        "total " + df_roi["name"], categories=["total spend", "total response"]
    )
    df_roi["name_label"] = pd.Categorical(
        df_roi["type"] + "\n" + df_roi["name"],
        categories=[
            t + "\n" + n for t in df_roi["type"].unique() for n in ["spend", "response"]
        ],
    )

    df_roi["value_norm"] = df_roi.groupby("name")["value"].transform(
        lambda x: x / x.iloc[0]
    )

    # Calculate metric values and labels
    metric_vals = (
        resp_metric["total_roi"] if metric == "ROAS" else resp_metric["total_cpa"]
    )
    labs = [
        f"{lev}\nSpend: {100 * (spend - resp_metric['total_spend'].iloc[0]) / resp_metric['total_spend'].iloc[0]:.3f}%\nResp: {100 * lift:.3f}%\n{metric}: {m_val:.2f}"
        for lev, spend, lift, m_val in zip(
            levs2,
            resp_metric["total_spend"],
            df_roi["total_response_lift"],
            metric_vals,
        )
    ]
    df_roi["labs"] = pd.Categorical(np.repeat(labs, 2), categories=labs)

    p1 = (
        ggplot(df_roi, aes(x="name", y="value_norm", fill="type"))
        + facet_grid(". ~ labs", scales="free")
        + scale_fill_manual(values=["grey", "steelblue", "darkgoldenrod4"])
        + geom_bar(stat="identity", width=0.6, alpha=0.7)
        + geom_text(
            aes(label="value"), format_string="{:.3f}", color="black", vjust=-0.5
        )
        # The theme_lares() function is specific to R and doesn't have an exact equivalent in plotnine, so we use a basic theme modification
        + theme(
            legend_position="none",
            plot_background=element_blank(),
            axis_text_y=element_blank(),
        )
        + labs(title="Total Budget Optimization Result", fill=None, y=None, x=None)
        + scale_y_continuous(limits=(0, df_roi["value_norm"].max() * 1.2))
    )

    outputs["p1"] = p1

    # 2. Response and spend comparison per channel plot
    df_plots = dt_optimOut.copy()
    df_plots["channel"] = df_plots["channels"].astype("category")
    df_plots["Initial"] = df_plots["initResponseUnitShare"]
    df_plots["Bounded"] = df_plots["optmResponseUnitShare"]
    df_plots["Unbounded"] = df_plots["optmResponseUnitShareUnbound"]

    response_share = df_plots.melt(
        id_vars="channel",
        value_vars=["Initial", "Bounded", "Unbounded"],
        var_name="type",
        value_name="response_share",
    )

    df_plots["Initial"] = df_plots["initSpendShare"]
    df_plots["Bounded"] = df_plots["optmSpendShareUnit"]
    df_plots["Unbounded"] = df_plots["optmSpendShareUnitUnbound"]

    spend_share = df_plots.melt(
        id_vars="channel",
        value_vars=["Initial", "Bounded", "Unbounded"],
        var_name="type",
        value_name="spend_share",
    )

    df_plots = response_share.merge(spend_share, on=["channel", "type"])

    # Create channel ROI or CPA based on 'metric'
    if metric == "ROAS":
        value_vars = "roiUnit"
    else:
        value_vars = "cpaUnit"

    df_plots["Initial"] = df_plots[f"init{value_vars}"]
    df_plots["Bounded"] = df_plots[f"optm{value_vars}"]
    df_plots["Unbounded"] = df_plots[f"optm{value_vars}Unbound"]

    channel_metric = df_plots.melt(
        id_vars="channel",
        value_vars=["Initial", "Bounded", "Unbounded"],
        var_name="type",
        value_name=f"channel_{metric.lower()}",
    )

    df_plots = df_plots.merge(channel_metric, on=["channel", "type"])

    # Same logic for marginal ROI or CPA
    df_plots["Initial"] = df_plots[f"initResponseMargUnit"].apply(
        lambda x: x if metric == "ROAS" else 1 / x
    )
    df_plots["Bounded"] = df_plots[f"optmResponseMargUnit"].apply(
        lambda x: x if metric == "ROAS" else 1 / x
    )
    df_plots["Unbounded"] = df_plots[f"optmResponseMargUnitUnbound"].apply(
        lambda x: x if metric == "ROAS" else 1 / x
    )

    marginal_metric = df_plots.melt(
        id_vars="channel",
        value_vars=["Initial", "Bounded", "Unbounded"],
        var_name="type",
        value_name=f"marginal_{metric.lower()}",
    )

    df_plots = df_plots.merge(marginal_metric, on=["channel", "type"])

    # Final join with resp_metric
    df_plots = df_plots.merge(resp_metric, on="type")

    # Combining different dataframes with different metrics
    df_plot_share_spend = df_plots[
        ["channel", "type", "type_lab", "spend_share"]
    ].copy()
    df_plot_share_spend["metric"] = "spend"
    df_plot_share_spend.rename(columns={"spend_share": "values"}, inplace=True)

    df_plot_share_response = df_plots[
        ["channel", "type", "type_lab", "response_share"]
    ].copy()
    df_plot_share_response["metric"] = "response"
    df_plot_share_response.rename(columns={"response_share": "values"}, inplace=True)

    channel_cols = [col for col in df_plots.columns if col.startswith("channel_")]
    df_plot_share_channel = df_plots[
        ["channel", "type", "type_lab"] + channel_cols
    ].copy()
    df_plot_share_channel["metric"] = metric
    df_plot_share_channel.rename(
        columns={col: "values" for col in channel_cols}, inplace=True
    )

    marginal_cols = [col for col in df_plots.columns if col.startswith("marginal_")]
    df_plot_share_marginal = df_plots[
        ["channel", "type", "type_lab"] + marginal_cols
    ].copy()
    df_plot_share_marginal["metric"] = "m" + metric
    df_plot_share_marginal.rename(
        columns={col: "values" for col in marginal_cols}, inplace=True
    )

    df_plot_share = pd.concat(
        [
            df_plot_share_spend,
            df_plot_share_response,
            df_plot_share_channel,
            df_plot_share_marginal,
        ]
    )

    # Additional data manipulation
    df_plot_share["type"] = pd.Categorical(df_plot_share["type"], categories=levs1)
    df_plot_share["name_label"] = pd.Categorical(
        df_plot_share["type"] + "\n" + df_plot_share["metric"],
        categories=[
            t + "\n" + m for t in levs1 for m in df_plot_share["metric"].unique()
        ],
    )

    # Handling extreme values
    df_plot_share["values"] = df_plot_share["values"].replace([np.inf, -np.inf], np.nan)
    df_plot_share["values"] = df_plot_share["values"].fillna(0)
    df_plot_share["values"] = df_plot_share["values"].round(4)

    # Formatting values_label
    def format_values(row):
        if row["metric"] in ["CPA", "mCPA", "ROAS", "mROAS"]:
            return f"{row['values']:.2f}"
        else:
            return f"{100 * row['values']:.1f}%"

    df_plot_share["values_label"] = df_plot_share.apply(format_values, axis=1)
    df_plot_share["values_label"] = df_plot_share["values_label"].replace(
        ["NA", "NaN"], "-"
    )

    # More data manipulation
    df_plot_share["channel"] = pd.Categorical(
        df_plot_share["channel"], categories=reversed(df_plot_share["channel"].unique())
    )
    df_plot_share["metric"] = df_plot_share["metric"].apply(
        lambda x: x + "%" if x in ["spend", "response"] else x
    )
    df_plot_share["metric"] = pd.Categorical(
        df_plot_share["metric"],
        categories=[
            m + p for m in df_plot_share["metric"].unique() for p in ["", "", "%", "%"]
        ],
    )

    # Normalizing values
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val)

    df_plot_share["values_norm"] = df_plot_share.groupby("name_label")[
        "values"
    ].transform(normalize)
    df_plot_share["values_norm"] = df_plot_share["values_norm"].fillna(0)

    p2 = (
        ggplot(df_plot_share, aes(x="metric", y="channel", fill="type"))
        + geom_tile(aes(alpha="values_norm"), color="white")
        + scale_fill_manual(values=["grey50", "steelblue", "darkgoldenrod4"])
        + scale_alpha(range=(0.6, 1))
        + geom_text(aes(label="values_label"), color="black")
        + facet_grid(". ~ type_lab", scales="free")
        # The theme_lares() function is specific to R and doesn't have an exact equivalent in plotnine, so we use a basic theme modification
        + theme(legend_position="none", plot_background=element_blank())
        + labs(
            title="Budget Allocation per Channel*", fill=None, x=None, y="Paid Channels"
        )
    )

    outputs["p2"] = p2

    ## 3. Response curves
    constr_labels = dt_optimOut.copy()
    constr_labels['constr_label'] = constr_labels.apply(lambda x: f"{x['channels']}\n[{x['constr_low']} - {x['constr_up']}] & [{round(x['constr_low_unb'], 1)} - {round(x['constr_up_unb'], 1)}]", axis=1)
    constr_labels = constr_labels.rename(columns={'channels': 'channel'})[['channel', 'constr_label', 'constr_low_abs', 'constr_up_abs', 'constr_low_unb_abs', 'constr_up_unb_abs']]

    # Left join with plotDT_scurve
    plotDT_scurve = eval_list['plotDT_scurve'].merge(constr_labels, on='channel', how='left')

    # Data manipulation for mainPoints
    mainPoints = eval_list['mainPoints'].merge(constr_labels, on='channel', how='left')
    mainPoints = mainPoints.merge(resp_metric, on='type', how='left')

    mainPoints['type'] = mainPoints['type'].fillna('Carryover').astype(str)
    mainPoints['type'] = pd.Categorical(mainPoints['type'], categories=['Carryover'] + levs1)

    mainPoints['type_lab'] = mainPoints['type_lab'].fillna('Carryover').astype(str)
    mainPoints['type_lab'] = pd.Categorical(mainPoints['type_lab'], categories=['Carryover'] + levs2)

    caov_points = mainPoints[mainPoints['type'] == 'Carryover'][['channel', 'spend_point']].rename(columns={'spend_point': 'caov_spend'})

    # Left join and mutate mainPoints with caov_points
    mainPoints = mainPoints.merge(caov_points, on='channel', how='left')

    def calculate_abs(row, type_index, constr_type):
        if row['type'] == levs1[type_index]:
            return row[f'{constr_type}_abs'] + row['caov_spend']
        else:
            return None

    mainPoints['constr_low_abs'] = mainPoints.apply(lambda row: calculate_abs(row, 1, 'constr_low'), axis=1)
    mainPoints['constr_up_abs'] = mainPoints.apply(lambda row: calculate_abs(row, 1, 'constr_up'), axis=1)
    mainPoints['constr_low_unb_abs'] = mainPoints.apply(lambda row: calculate_abs(row, 2, 'constr_low_unb'), axis=1)
    mainPoints['constr_up_unb_abs'] = mainPoints.apply(lambda row: calculate_abs(row, 2, 'constr_up_unb'), axis=1)

    mainPoints['plot_lb'] = mainPoints.apply(lambda row: row['constr_low_unb_abs'] if pd.isna(row['constr_low_abs']) else row['constr_low_abs'], axis=1)
    mainPoints['plot_ub'] = mainPoints.apply(lambda row: row['constr_up_unb_abs'] if pd.isna(row['constr_up_abs']) else row['constr_up_abs'], axis=1)

    # Creating caption
    caption_parts = [
        f" Given the upper/lower constrains, the total budget ({eval_list['total_budget']}) can't be fully allocated \n" if any_topped else "",
        f"* {formulax1}\n",
        f"* {formulax2}\n",
        "** Dotted lines show budget optimization lower-upper ranges per media"
    ]
    caption = "".join(part for part in caption_parts if part.strip())

    p3 = (
        ggplot(plotDT_scurve)
        + scale_x_continuous(labels=scientific_format())  # scale_x_abbr() is not directly available in plotnine
        + scale_y_continuous(labels=scientific_format())  # scale_y_abbr() is not directly available in plotnine
        + geom_line(aes(x='spend', y='total_response'), show_legend=False, size=0.5)
        + facet_wrap('constr_label', scales='free', ncol=3)
        + geom_area(
            data=plotDT_scurve[plotDT_scurve['spend'] <= plotDT_scurve['mean_carryover']],
            mapping=aes('spend', 'total_response', color='constr_label'),
            stat='identity', position='stack', size=0.1,
            fill='grey50', alpha=0.4, show_legend=False
        )
        + geom_errorbar(
            data=mainPoints[~mainPoints['constr_label'].isna()],
            mapping=aes(x='spend_point', y='response_point', xmin='plot_lb', xmax='plot_ub'),
            color='black', linetype='dotted'
        )
        + geom_point(
            data=mainPoints[~mainPoints['plot_lb'].isna() & ~mainPoints['mean_spend'].isna()],
            mapping=aes(x='plot_lb', y='response_point'), shape=18
        )
        + geom_point(
            data=mainPoints[~mainPoints['plot_ub'].isna() & ~mainPoints['mean_spend'].isna()],
            mapping=aes(x='plot_ub', y='response_point'), shape=18
        )
        + geom_point(
            data=mainPoints[~mainPoints['constr_label'].isna()],
            mapping=aes(x='spend_point', y='response_point', fill='type_lab'),
            size=2.5, shape=21
        )
        + scale_fill_manual(values=["white", "grey", "steelblue", "darkgoldenrod4"])
        # The theme_lares() function is specific to R and doesn't have an exact equivalent in plotnine, so we use a basic theme modification
        + theme(legend_position='top', plot_background=element_blank())
        + labs(
            title="Simulated Response Curve for Selected Allocation Period",
            x=f"Spend** per {InputCollect['intervalType']} (Mean Adstock Zone in Grey)",
            y=f"Total Response [{InputCollect['dep_var_type']}]",
            caption=caption
        )
    )

    outputs['p3'] = p3

    min_period_loc = dt_optimOut['periods'].astype(str).apply(lambda x: x.split(' ')[0]).astype(int).idxmin()
    subtitle = f"{errors}\nSimulation date range: {dt_optimOut.loc[0, 'date_min']} to {dt_optimOut.loc[0, 'date_max']} ({dt_optimOut.loc[min_period_loc, 'periods']}) | Scenario: {scenario}"

    # Calculate the heights for the subplots based on the data
    heights = [
        0.8,
        0.2 + len(dt_optimOut['channels']) * 0.2,
        np.ceil(len(dt_optimOut['channels']) / 3)
    ]

    # Create a matplotlib figure with subplots
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': heights}, figsize=(10, 15))

    # Draw the plots on the subplots
    draw(p1, ax=axs[0])
    draw(p2, ax=axs[1])
    draw(p3, ax=axs[2])

    # Set the title and subtitle
    axs[0].set_title(f"Budget Allocation Onepager for Model ID {select_model}", fontsize=16)
    axs[0].set_xlabel(subtitle, fontsize=12)

    if export:
        # Determine the file suffix based on conditions
        if scenario == "max_response" and metric == "ROAS":
            suffix = "best_roas"
        elif scenario == "max_response" and metric == "CPA":
            suffix = "best_cpa"
        elif scenario == "target_efficiency" and metric == "ROAS":
            suffix = "target_roas"
        elif scenario == "target_efficiency" and metric == "CPA":
            suffix = "target_cpa"
        else:
            suffix = "none"

        # Constructing the filename
        filename = os.path.join(plot_folder, f"{select_model}_reallocated_{suffix}.png")

        # Saving the figure
        fig.set_size_inches(12, 10 + 2 * np.ceil(len(dt_optimOut['channels']) / 3))
        fig.savefig(filename, dpi=350)

        if not quiet:
            print(f"Exporting to: {filename}")

    return outputs


def refresh_plots(InputCollectRF, OutputCollectRF, ReportCollect, export=True, **kwargs):
    selectID = ReportCollect['selectIDs'][-1] if ReportCollect['selectIDs'] else ReportCollect['resultHypParamReport']['solID'][-1]
    print(f">> Plotting refresh results for model: {selectID}")

    # Assuming ReportCollect contains DataFrames for xDecompVecReport and xDecompAggReport
    xDecompVecReport = ReportCollect['xDecompVecReport'][ReportCollect['xDecompVecReport']['solID'] == selectID]
    xDecompAggReport = ReportCollect['xDecompAggReport'][ReportCollect['xDecompAggReport']['solID'] == selectID]

    plot_folder = OutputCollectRF['plot_folder']
    outputs = {}

    xDecompVecReportPlot = xDecompVecReport.copy()
    xDecompVecReportPlot['refreshStart'] = xDecompVecReportPlot.groupby('refreshStatus')['ds'].transform('min')
    xDecompVecReportPlot['refreshEnd'] = xDecompVecReportPlot.groupby('refreshStatus')['ds'].transform('max')
    xDecompVecReportPlot['duration'] = (
        (xDecompVecReportPlot['refreshEnd'].astype('datetime64') - xDecompVecReportPlot['refreshStart'].astype('datetime64')).dt.days
        + InputCollectRF['dayInterval']
    ) / InputCollectRF['dayInterval']

    dt_refreshDates = xDecompVecReportPlot[['refreshStatus', 'refreshStart', 'refreshEnd', 'duration']].drop_duplicates()
    dt_refreshDates['label'] = dt_refreshDates.apply(
        lambda x: f"Initial: {x['refreshStart']}, {x['duration']} {InputCollectRF['intervalType']}s"
        if x['refreshStatus'] == 0
        else f"Refresh #{x['refreshStatus']}: {x['refreshStart']}, {x['duration']} {InputCollectRF['intervalType']}s",
        axis=1
    )

    xDecompVecReportMelted = xDecompVecReportPlot.melt(
        id_vars=['ds', 'refreshStatus', 'refreshStart', 'refreshEnd'],
        value_vars=['dep_var', 'depVarHat'],
        var_name='variable', value_name='value'
    )
    xDecompVecReportMelted['variable'] = xDecompVecReportMelted['variable'].replace({'dep_var': 'actual', 'depVarHat': 'prediction'})

    # Function to calculate R-squared
    def get_rsq(true, predicted):
        residual_sum_of_squares = ((true - predicted) ** 2).sum()
        total_sum_of_squares = ((true - true.mean()) ** 2).sum()
        return 1 - residual_sum_of_squares / total_sum_of_squares

    # Creating the plot
    pFitRF = (
        ggplot(xDecompVecReportMelted, aes(x='ds', y='value', color='variable'))
        + geom_line()
        + geom_rect(
            data=dt_refreshDates,
            mapping=aes(xmin='refreshStart', xmax='refreshEnd', fill='refreshStatus.astype(str)'),
            ymin=float('-inf'), ymax=float('inf'), alpha=0.2
        )
        # Theme customization
        + theme(
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            panel_background=element_blank(),
            legend_background=element_rect(fill='white', alpha=0.4)
        )
        # Additional elements
        + scale_fill_brewer(palette="BuGn")
        + geom_text(
            data=dt_refreshDates, mapping=aes(x='refreshStart', y=xDecompVecReportMelted['value'].max(),
                                            label='label'),
            angle=270, hjust=-0.1, vjust=-0.2, color="gray40"
        )
        + labs(
            title="Model Refresh: Actual vs. Predicted Response",
            subtitle=f"Assembled R2: {round(get_rsq(xDecompVecReportPlot['dep_var'], xDecompVecReportPlot['depVarHat']), 2)}",
            x="Date", y="Response", fill="Refresh", color="Type"
        )
        + scale_y_continuous(labels=scientific_format())
    )

    outputs['pFitRF'] = pFitRF

    if export:
        # Construct the filename
        filename = os.path.join(plot_folder, "report_actual_fitted.png")

        # Set the size of the plot
        pFitRF.save(filename, dpi=900, width=12, height=8, limitsize=False)

        print(f"Plot saved to {filename}")

    xDecompAggReportPlotBase = xDecompAggReport[
        xDecompAggReport['rn'].isin(InputCollectRF['prophet_vars'] + ["(Intercept)"])
    ].copy()
    xDecompAggReportPlotBase['perc'] = xDecompAggReportPlotBase.apply(
        lambda x: x['xDecompPerc'] if x['refreshStatus'] == 0 else x['xDecompPercRF'], axis=1
    )
    xDecompAggReportPlotBase = xDecompAggReportPlotBase.groupby('refreshStatus').agg(
        variable=('rn', lambda x: "baseline"),
        percentage=('perc', 'sum'),
        roi_total=('perc', lambda _: float('nan'))  # NA equivalent in Python
    )

    xDecompAggReportPlot = xDecompAggReport[
        ~xDecompAggReport['rn'].isin(InputCollectRF['prophet_vars'] + ["(Intercept)"])
    ].copy()
    xDecompAggReportPlot['percentage'] = xDecompAggReportPlot.apply(
        lambda x: x['xDecompPerc'] if x['refreshStatus'] == 0 else x['xDecompPercRF'], axis=1
    )
    xDecompAggReportPlot = xDecompAggReportPlot[['refreshStatus', 'rn', 'percentage', 'roi_total']]
    xDecompAggReportPlot = xDecompAggReportPlot.rename(columns={'rn': 'variable'})
    xDecompAggReportPlot = pd.concat([xDecompAggReportPlot, xDecompAggReportPlotBase])
    xDecompAggReportPlot = xDecompAggReportPlot.sort_values(by=['refreshStatus', 'variable'], ascending=[True, False])
    xDecompAggReportPlot['refreshStatus'] = xDecompAggReportPlot['refreshStatus'].apply(
        lambda x: "Init.mod" if x == 0 else f"Refresh{x}"
    )

    # Calculating scaling factor and max y-axis limit for the plot
    ySecScale = 0.75 * max(xDecompAggReportPlot['roi_total'].dropna() / xDecompAggReportPlot['percentage'].max())
    ymax = 1.1 * max(max(xDecompAggReportPlot['roi_total'].dropna() / ySecScale), xDecompAggReportPlot['percentage'].max())

    pBarRF = (
        ggplot(xDecompAggReportPlot, aes(x='variable', y='percentage', fill='variable'))
        + geom_bar(alpha=0.8, position='dodge', stat='identity', na_rm=True)
        + facet_wrap('~refreshStatus', scales='free')
        # The theme_lares() function is specific to R and doesn't have an exact equivalent in plotnine, so we use a basic theme modification
        + theme(legend_position='none', axis_text_x=element_blank(), axis_ticks_x=element_blank())
        + scale_fill_manual(values=robyn_palette()['fill'])
        + geom_text(aes(label=f"{round('percentage' * 100, 1)}%"), size=3, na_rm=True, position=position_dodge(width=0.9), hjust=-0.25)
        + geom_point(aes(x='variable', y='roi_total' / ySecScale, color='variable'), size=4, shape=17, na_rm=True)
        + geom_text(aes(label=round('roi_total', 2), x='variable', y='roi_total' / ySecScale), size=3, na_rm=True, hjust=-0.4, fontface='bold', position=position_dodge(width=0.9))
        + scale_color_manual(values=robyn_palette()['fill'])
        + scale_y_continuous(sec_axis=sec_axis(lambda x: x * ySecScale), breaks=range(0, int(ymax) + 1, 2), limits=(0, ymax), name='Total ROI')
        + coord_flip()
        + labs(
            title="Model Refresh: Decomposition & Paid Media ROI",
            subtitle="Baseline includes intercept and prophet vars: " + ', '.join(InputCollectRF['prophet_vars'])
        )
    )

    outputs['pBarRF'] = pBarRF

    # Exporting the plot if required
    if export:
        filename = os.path.join(plot_folder, "report_decomposition.png")
        pBarRF.save(filename, dpi=900, width=12, height=8, limitsize=False)
        print(f"Plot saved to {filename}")

    return outputs


def refresh_plots_json(output_collect_rf, json_file, export=True, **kwargs):
    with open(json_file, 'r') as file:
        chain_data = json.load(file)

    sol_id = list(chain_data.keys())[-1]
    day_interval = chain_data[sol_id]['InputCollect']['dayInterval']
    interval_type = chain_data[sol_id]['InputCollect']['intervalType']
    rsq = chain_data[sol_id]['ExportedModel']['errors']['rsq_train']
    plot_folder = output_collect_rf['plot_folder']

    # 1. Fitted vs actual
    temp = output_collect_rf['allPareto']['plotDataCollect'][sol_id]
    x_decomp_vec_plot_melted = pd.DataFrame(temp['plot5data']['xDecompVecPlotMelted'])
    x_decomp_vec_plot_melted['linetype'] = x_decomp_vec_plot_melted['variable'].apply(
        lambda x: 'solid' if x == 'predicted' else 'dotted')
    x_decomp_vec_plot_melted['variable'] = x_decomp_vec_plot_melted['variable'].str.title()
    x_decomp_vec_plot_melted['ds'] = pd.to_datetime(x_decomp_vec_plot_melted['ds'], origin='1970-01-01', unit='D')

    # Creating dt_refreshDates dataframe
    def extract_dates(data):
        return {
            'window_start': pd.to_datetime(data['InputCollect']['window_start'], origin='1970-01-01', unit='D'),
            'window_end': pd.to_datetime(data['InputCollect']['window_end'], origin='1970-01-01', unit='D'),
            'duration': data['InputCollect']['refresh_steps']
        }

    dt_refresh_dates = pd.DataFrame({key: extract_dates(value) for key, value in chain_data.items()}).T
    dt_refresh_dates['solID'] = dt_refresh_dates.index
    dt_refresh_dates = dt_refresh_dates[dt_refresh_dates['duration'] > 0]
    dt_refresh_dates['refreshStatus'] = dt_refresh_dates.reset_index().index + 1
    dt_refresh_dates['refreshStart'] = dt_refresh_dates['window_end'] - pd.to_timedelta(dt_refresh_dates['duration'] * day_interval, unit='D')
    dt_refresh_dates['refreshEnd'] = dt_refresh_dates['window_end']

    def create_label(row):
        if row['refreshStatus'] == 0:
            return f"Initial: {row['refreshStart'].date()}, {row['duration']} {interval_type}s"
        else:
            return f"Refresh #{row['refreshStatus']}: {row['refreshStart'].date()}, {row['duration']} {interval_type}s"

    dt_refresh_dates['label'] = dt_refresh_dates.apply(create_label, axis=1)
    dt_refresh_dates = dt_refresh_dates.reset_index(drop=True)

    pFitRF = (ggplot(x_decomp_vec_plot_melted, aes(x='ds', y='value', color='variable', linetype='linetype'))
            + geom_path(size=0.6)
            + geom_rect(data=dt_refresh_dates, mapping=aes(xmin='refreshStart', xmax='refreshEnd', fill='refreshStatus'),
                        ymin=float('-inf'), ymax=float('inf'), alpha=0.2)
            + scale_fill_brewer(palette='BuGn')
            + geom_text(data=dt_refresh_dates, mapping=aes(x='refreshStart', y=x_decomp_vec_plot_melted['value'].max(),
                                                            label='label'), angle=270, hjust=0, vjust=-0.2, color='gray40')
            # The theme_lares is specific to ggplot2, you might need to customize this using theme() in plotnine
            + theme(figure_size=(12, 8), subplots_adjust={'right': 0.85})
            + scale_y_continuous(labels='comma')
            + guides(linetype='none', fill='none')
            + labs(title='Actual vs. Predicted Response', x='Date', y='Response'))

    # Save the plot
    if export:
        ggsave(plot=pFitRF, filename=f'{plot_folder}/report_actual_fitted.png', dpi=300, width=12, height=8, limitsize=False)

    # 2. Stacked bar plot preparation
    df_list = [chain_data[key]['ExportedModel']['summary'].assign(solID=key) for key in chain_data.keys()]
    df = pd.concat(df_list)

    # Adjusting the DataFrame according to the R code logic
    df['solID'] = pd.Categorical(df['solID'], categories=list(chain_data.keys()))
    df['label'] = [f"{sid} [{int(sid) - 1}]" for sid in df['solID']]
    label_levels = [f"{name} [{i}]" for i, name in enumerate(chain_data.keys())]
    df['label'] = pd.Categorical(df['label'], categories=label_levels)

    # Assuming 'prophet_vars' is a list of variables
    prophet_vars = chain_data[list(chain_data.keys())[0]]['InputCollect']['prophet_vars']
    df['variable'] = df['variable'].apply(lambda x: 'baseline' if x in prophet_vars or x == '(Intercept)' else x)

    # Summarizing the data
    df_grouped = df.groupby(['solID', 'label', 'variable']).sum().reset_index()

    # Function to format numbers, similar to formatNum in R
    def format_num(value, digits=2, suffix=""):
        formatted = f"{value:.{digits}f}"
        if suffix:
            formatted += suffix
        return formatted

    # Plotting
    pBarRF = (ggplot(df, aes(y='variable'))
            + geom_col(aes(x='decompPer'))
            + geom_text(aes(x='decompPer', label=df['decompPer'].apply(format_num, args=(2, '%'))),
                        na_rm=True, hjust=-0.2, size=8)  # Adjusted text size for visibility
            + geom_point(aes(x='performance'), na_rm=True, size=2, colour="#39638b")
            + geom_text(aes(x='performance', label=df['performance'].apply(format_num, args=(2,))),
                        na_rm=True, hjust=-0.4, size=8, colour="#39638b")  # Adjusted text size for visibility
            + facet_wrap('~label', scales='free')
            # Uncomment the next line if scale_x_percent is needed (custom implementation may be required)
            # + scale_x_percent(limits=(0, df['performance'].max() * 1.2))
            + labs(title="Model refresh: Decomposition & Paid Media",
                    subtitle="Baseline includes intercept and all prophet vars: " + ', '.join(prophet_vars),
                    x=None, y=None)
            + theme(figure_size=(12, 8), subplots_adjust={'right': 0.85},
                    axis_text_x=element_blank(), axis_ticks_x=element_blank()))

    if export:
        # Constructing the filename
        last_chain_data = chainData[list(chainData.keys())[-1]]['ExportedModel']
        plot_filename = f"{last_chain_data['plot_folder']}report_decomposition.png"

        # Saving the plot
        ggsave(plot=pBarRF, filename=plot_filename, dpi=300, width=12, height=8, limitsize=False)

    # Prepare the outputs dictionary, if needed
    outputs = {'pBarRF': pBarRF}

    # Return the outputs
    return outputs

####################################################################
#' Generate Plots for Time-Series Validation
#'
#' Create a plot to visualize the convergence for each of the datasets
#' when time-series validation is enabled when running \code{robyn_run()}.
#' As a reference, the closer the test and validation convergence points are,
#' the better, given the time-series wasn't overfitted.
#'
#' @rdname robyn_outputs
#' @return Invisible list with \code{ggplot} plots.
#' @export
def ts_validation_fun(output_models, quiet=False, **kwargs):
    filtered_models = output_models["trials"]

    result_hyp_param = pd.concat([model['resultCollect']['resultHypParam'] for model in filtered_models])
    result_hyp_param['i'] = result_hyp_param.groupby('trial').cumcount() + 1
    result_hyp_param.reset_index(drop=True, inplace=True)

    # selected_columns = result_hyp_param[['solID', 'i', 'trial', 'train_size'] + \
    #                                     [col for col in result_hyp_param if col.startswith('rsq_')]]
    # selected_columns['trial'] = selected_columns['trial'].apply(lambda x: f"Trial {x}")

    # # Pivot longer/melt the DataFrame
    # rsq_long = pd.melt(selected_columns, id_vars=['solID', 'i', 'trial', 'train_size'],
    #                    value_vars=[col for col in selected_columns if col.startswith('rsq_')],
    #                    var_name='dataset', value_name='rsq')

    # nrmse_cols = pd.melt(result_hyp_param[['solID'] + [col for col in result_hyp_param if col.startswith('nrmse_')]],
    #                      id_vars=['solID'], value_vars=[col for col in result_hyp_param if col.startswith('nrmse_')],
    #                      var_name='del', value_name='nrmse').drop(columns=['del'])


    # rsq_long_indexed = rsq_long.set_index(['solID', 'i', 'trial', 'dataset'])
    # nrmse_cols_indexed = nrmse_cols.set_index(['solID'])  # Adjust as necessary
    # result_hyp_param_long = pd.concat([rsq_long_indexed, nrmse_cols_indexed], axis=1, join='inner').reset_index()

    rsq_long = pd.melt(
        result_hyp_param[['solID', 'i', 'trial', 'train_size'] + [col for col in result_hyp_param if col.startswith('rsq_')]],
        id_vars=['solID', 'i', 'trial', 'train_size'],
        var_name='dataset',
        value_name='rsq'
    )
    rsq_long['trial'] = rsq_long['trial'].apply(lambda x: f"Trial {x}")
    rsq_long['dataset'] = rsq_long['dataset'].str.replace('rsq_', '')
    print(rsq_long['rsq'])
    rsq_long['rsq'] = winsorize(rsq_long['rsq'], limits=[0.01, 0.99])

    nrmse_long = pd.melt(
        result_hyp_param[['solID'] + [col for col in result_hyp_param if col.startswith('nrmse_')]],
        id_vars=['solID'],
        var_name='dataset',
        value_name='nrmse'
    )
    nrmse_long['dataset'] = nrmse_long['dataset'].str.replace('nrmse_', '')
    nrmse_long['nrmse'] = winsorize(nrmse_long['nrmse'], limits=[0.00, 0.99])

    result_hyp_param_long = pd.merge(rsq_long, nrmse_long[['solID', 'dataset', 'nrmse']], on=['solID', 'dataset'], how='left')
    result_hyp_param_long['dataset'] = result_hyp_param_long['dataset'].str.replace('rsq_', '')

    sns.set_theme(style="whitegrid")

    # Create a figure with a specific size
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First plot
    sns.scatterplot(x='i', y='train_size', data=result_hyp_param, color='black', alpha=0.5, s=12, ax=axs[0])
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Train Size')

    # Second plot
    sns.scatterplot(x='i', y='nrmse', hue='dataset', data=result_hyp_param_long, alpha=0.2, s=9, ax=axs[1])
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('NRMSE [Upper 1% Winsorized]')
    axs[1].legend(title='Dataset', loc='upper right')

    # Adjust the layout
    plt.tight_layout()
    fig.suptitle("Time-series validation & Convergence", fontsize=16)
    plt.subplots_adjust(top=0.88)

    # Instead of displaying the plot, return the figure and axes objects
    return fig, axs

# def ts_validation_fun(output_models, quiet=False, **kwargs):
#     # if not output_models.get('ts_validation', False):
#     #     return None

#     # Extracting the relevant trials
#     trial_names = [f"trial{i}" for i in range(1, output_models['trials'] + 1)]
#     relevant_trials = [output_models[name]['resultCollect']['resultHypParam'] for name in trial_names if name in output_models]

#     # Binding rows and processing
#     result_hyp_param = pd.concat(relevant_trials, ignore_index=True)
#     result_hyp_param['i'] = result_hyp_param.groupby('trial').cumcount() + 1

#     # Data manipulation
#     result_hyp_param_long = result_hyp_param.copy()
#     result_hyp_param_long = result_hyp_param_long.filter(regex='^rsq_|solID|i|trial|train_size$')
#     result_hyp_param_long['trial'] = 'Trial ' + result_hyp_param_long['trial'].astype(str)
#     rsq_cols = result_hyp_param_long.filter(regex='^rsq_').columns
#     nrmse_cols = result_hyp_param.filter(regex='^nrmse_').columns

#     # Melting the DataFrame
#     rsq_melted = result_hyp_param_long.melt(id_vars=['solID', 'i', 'trial', 'train_size'],
#                                         value_vars=rsq_cols,
#                                         var_name='dataset', value_name='rsq')
#     nrmse_melted = result_hyp_param.melt(id_vars=['solID'],
#                                         value_vars=nrmse_cols,
#                                         var_name='del', value_name='nrmse').drop(columns=['del'])

#     # Combining melted dataframes
#     result_hyp_param_long = pd.concat([rsq_melted, nrmse_melted['nrmse']], axis=1)

#     # Winsorizing and final adjustments
#     result_hyp_param_long['rsq'] = winsorize(result_hyp_param_long['rsq'], limits=[0.01, 0.99])
#     result_hyp_param_long['nrmse'] = winsorize(result_hyp_param_long['nrmse'], limits=[0.00, 0.99])
#     result_hyp_param_long['dataset'] = result_hyp_param_long['dataset'].str.replace('rsq_', '')

#     pIters = (ggplot(result_hyp_param, aes(x='i', y='train_size'))
#             + geom_point(fill='black', alpha=0.5, size=1.2, shape=23)
#             # Uncomment the next line if geom_smooth is required
#             # + geom_smooth()
#             + labs(y='Train Size', x='Iteration')
#             # Uncomment and adjust the next line if scale_y_percent and scale_x_abbr are required
#             # + scale_y_continuous(labels='percent') + scale_x_continuous(labels='abbr')
#             + theme(figure_size=(10, 6), plot_background=element_rect(fill='white')))

#     pNRMSE = (ggplot(result_hyp_param_long, aes(x='i', y='nrmse', color='dataset'))
#             + geom_point(alpha=0.2, size=0.9)
#             + geom_smooth(method='gamm', method_args={'formula': 'y ~ s(x, bs="cs")'})
#             + facet_grid('trial ~ .')
#             + geom_hline(yintercept=0, linetype='dashed')
#             + labs(y='NRMSE [Upper 1% Winsorized]', x='Iteration', color='Dataset')
#             # The theme_lares is specific to ggplot2, you might need to customize this using theme() in plotnine
#             + theme(figure_size=(10, 6), plot_background=element_rect(fill='white'))
#             # Uncomment and adjust the next line if scale_x_abbr is required
#             # + scale_x_continuous(labels='abbr')
#             )

#     if export:
#         ggsave(plot=pNRMSE, filename=f'{plot_folder}/pNRMSE.png', dpi=300, width=12, height=8, limitsize=False)
#         ggsave(plot=pIters, filename=f'{plot_folder}/pIters.png', dpi=300, width=12, height=8, limitsize=False)

#     # Return the plots separately
#     return {'pNRMSE': pNRMSE, 'pIters': pIters}

#' @rdname robyn_outputs
#' @param solID Character vector. Model IDs to plot.
#' @param exclude Character vector. Manually exclude variables from plot.
#' @export
def decomp_plot(input_collect, output_collect, sol_id=None, exclude=None):
    # Check options - Implement this as needed based on your application
    # check_opts(sol_id, output_collect['allSolutions'])

    # String manipulation for interval type and variable type
    int_type = input_collect['intervalType'].title()
    if int_type in ['Month', 'Week']:
        int_type += 'ly'
    elif int_type == 'Day':
        int_type = 'Daily'

    var_type = input_collect['dep_var_type'].title()

    pal = plt.cm.get_cmap('viridis').colors

    # Data manipulation
    df = output_collect['xDecompVecCollect']
    df = df[df['solID'].isin(sol_id)]
    df = pd.melt(df, id_vars=['solID', 'ds', 'dep_var'], var_name='variable', value_name='value')
    if exclude:
        df = df[~df['variable'].isin(exclude)]
    df['variable'] = pd.Categorical(df['variable'], categories=reversed(df['variable'].unique()), ordered=True)

    # Plotting
    p = (ggplot(df, aes(x='ds', y='value', fill='variable'))
         + facet_grid('solID ~ .')
         + labs(title=f'{var_type} Decomposition by Variable', x=None, y=f'{int_type} {var_type}', fill=None)
         + geom_area()
         + theme(figure_size=(10, 6), plot_background=element_rect(fill='white'), legend_position='right')
         + scale_fill_manual(values=pal[:len(df['variable'].unique())])
         # Adjust scale_y_continuous if needed to mimic scale_y_abbr in R
         + scale_y_continuous(labels='comma'))

    return p
