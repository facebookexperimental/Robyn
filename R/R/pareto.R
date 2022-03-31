# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

robyn_pareto <- function(InputCollect, OutputModels, pareto_fronts, calibration_constraint = 0.1, quiet = FALSE) {

  hyper_fixed <- attr(OutputModels, "hyper_fixed")
  OutModels <- OutputModels[sapply(OutputModels, function(x) "resultCollect" %in% names(x))]
  resultHypParam <- rbindlist(lapply(OutModels, function(x) x$resultCollect$resultHypParam[, trial := x$trial]))
  resultHypParam[, iterations := (iterNG - 1) * OutputModels$cores + iterPar]
  xDecompAgg <- rbindlist(lapply(OutModels, function(x) x$resultCollect$xDecompAgg[, trial := x$trial]))
  xDecompAgg[, iterations := (iterNG - 1) * OutputModels$cores + iterPar]

  # Assign unique IDs using: trial + iterNG + iterPar
  resultHypParam[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  xDecompAgg[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  xDecompAggCoef0 <- xDecompAgg[rn %in% InputCollect$paid_media_spends, .(coef0 = min(coef, na.rm = TRUE) == 0), by = "solID"]

  if (!hyper_fixed) {
    mape_lift_quantile10 <- quantile(resultHypParam$mape, probs = calibration_constraint, na.rm = TRUE)
    nrmse_quantile90 <- quantile(resultHypParam$nrmse, probs = 0.90, na.rm = TRUE)
    decomprssd_quantile90 <- quantile(resultHypParam$decomp.rssd, probs = 0.90, na.rm = TRUE)
    resultHypParam <- resultHypParam[xDecompAggCoef0, on = "solID"]
    resultHypParam[, mape.qt10 := mape <= mape_lift_quantile10 & nrmse <= nrmse_quantile90 & decomp.rssd <= decomprssd_quantile90]

    resultHypParamPareto <- resultHypParam[mape.qt10 == TRUE]
    px <- rPref::low(resultHypParamPareto$nrmse) * rPref::low(resultHypParamPareto$decomp.rssd)
    resultHypParamPareto <- rPref::psel(resultHypParamPareto, px, top = nrow(resultHypParamPareto))[order(iterNG, iterPar, nrmse)]
    setnames(resultHypParamPareto, ".level", "robynPareto")

    setkey(resultHypParam, solID)
    setkey(resultHypParamPareto, solID)
    resultHypParam <- merge(resultHypParam, resultHypParamPareto[, .(solID, robynPareto)], all.x = TRUE)
  } else {
    resultHypParam[, ":="(mape.qt10 = TRUE, robynPareto = 1, coef0 = NA)]
  }

  xDecompAgg <- xDecompAgg[resultHypParam, robynPareto := i.robynPareto, on = c("iterNG", "iterPar", "trial")]

  decompSpendDist <- rbindlist(lapply(OutModels, function(x) x$resultCollect$decompSpendDist[, trial := x$trial]))
  decompSpendDist <- decompSpendDist[resultHypParam, robynPareto := i.robynPareto, on = c("iterNG", "iterPar", "trial")]
  if (hyper_fixed == FALSE) {
    decompSpendDist[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  } else {
    xDecompAgg[, solID := unique(decompSpendDist$solID)]
    resultHypParam[, solID := unique(decompSpendDist$solID)]
  }

  if (check_parallel()) registerDoParallel(OutputModels$cores) else registerDoSEQ()
  pareto_fronts_vec <- 1:pareto_fronts
  decompSpendDistPar <- decompSpendDist[robynPareto %in% pareto_fronts_vec]
  resultHypParamPar <- resultHypParam[robynPareto %in% pareto_fronts_vec]
  xDecompAggPar <- xDecompAgg[robynPareto %in% pareto_fronts_vec]
  resp_collect <- foreach(
    respN = seq_along(decompSpendDistPar$rn), .combine = rbind) %dorng% {
      get_resp <- robyn_response(
        media_metric = decompSpendDistPar$rn[respN],
        select_model = decompSpendDistPar$solID[respN],
        metric_value = decompSpendDistPar$mean_spend[respN],
        dt_hyppar = resultHypParamPar,
        dt_coef = xDecompAggPar,
        InputCollect = InputCollect,
        OutputCollect = OutputModels,
        quiet = quiet
        )$response
      dt_resp <- data.table(mean_response = get_resp,
                            rn = decompSpendDistPar$rn[respN],
                            solID = decompSpendDistPar$solID[respN])
      return(dt_resp)
    }
  stopImplicitCluster(); registerDoSEQ(); getDoParWorkers()
  setkey(decompSpendDist, solID, rn)
  setkey(resp_collect, solID, rn)
  decompSpendDist <- merge(decompSpendDist, resp_collect, all.x=TRUE)
  decompSpendDist[, ":="(
    roi_mean = mean_response / mean_spend,
    roi_total = xDecompAgg / total_spend,
    cpa_mean = mean_spend / mean_response,
    cpa_total = total_spend / xDecompAgg
  )]
  setkey(xDecompAgg, solID, rn)
  setkey(decompSpendDist, solID, rn)
  xDecompAgg <- merge(xDecompAgg, decompSpendDist[, .(
    rn, solID, total_spend, mean_spend, spend_share, effect_share, roi_mean, roi_total, cpa_total)],
    all.x = TRUE)

  # Pareto loop (no plots)
  mediaVecCollect <- list()
  xDecompVecCollect <- list()
  meanResponseCollect <- list()
  plotDataCollect <- list()

  for (pf in pareto_fronts_vec) {

    plotMediaShare <- xDecompAgg[robynPareto == pf & rn %in% InputCollect$paid_media_spends]
    uniqueSol <- plotMediaShare[, unique(solID)]
    plotWaterfall <- xDecompAgg[robynPareto == pf]
    dt_mod <- copy(InputCollect$dt_mod)
    dt_modRollWind <- copy(InputCollect$dt_modRollWind)

    for (sid in uniqueSol) {
    # parallelResult <- foreach(sid = uniqueSol) %dorng% {

      # Calculations for pareto AND pareto plots

      ## 1. Spend x effect share comparison
      plotMediaShareLoop <- plotMediaShare[solID == sid]
      suppressWarnings(plotMediaShareLoop <- melt.data.table(plotMediaShareLoop, id.vars = c("rn", "nrmse", "decomp.rssd", "rsq_train"), measure.vars = c("spend_share", "effect_share", "roi_total", "cpa_total")))
      plotMediaShareLoop[, rn := factor(rn, levels = sort(InputCollect$paid_media_spends))]
      plotMediaShareLoopBar <- plotMediaShareLoop[variable %in% c("spend_share", "effect_share")]
      plotMediaShareLoopLine <- plotMediaShareLoop[variable == ifelse(InputCollect$dep_var_type == "conversion", "cpa_total", "roi_total")]
      line_rm_inf <- !is.infinite(plotMediaShareLoopLine$value)
      ySecScale <- max(plotMediaShareLoopLine$value[line_rm_inf]) / max(plotMediaShareLoopBar$value) * 1.1
      plot1data <- list(plotMediaShareLoopBar = plotMediaShareLoopBar,
                        plotMediaShareLoopLine = plotMediaShareLoopLine,
                        ySecScale = ySecScale)

      ## 2. Waterfall
      plotWaterfallLoop <- plotWaterfall[solID == sid][order(xDecompPerc)]
      plotWaterfallLoop[, end := cumsum(xDecompPerc)]
      plotWaterfallLoop[, end := 1 - end]
      plotWaterfallLoop[, ":="(start = shift(end, fill = 1, type = "lag"),
                               id = 1:nrow(plotWaterfallLoop),
                               rn = as.factor(rn),
                               sign = as.factor(ifelse(xDecompPerc >= 0, "pos", "neg")))]
      plot2data <- list(plotWaterfallLoop = plotWaterfallLoop)

      ## 3. Adstock rate
      dt_geometric <- weibullCollect <- wb_type <- NULL
      resultHypParamLoop <- resultHypParam[solID == sid]
      get_hp_names <- !(names(InputCollect$hyperparameters) %like% "penalty_*")
      get_hp_names <- names(InputCollect$hyperparameters)[get_hp_names]
      hypParam <- unlist(resultHypParamLoop[, get_hp_names, with = FALSE])
      if (InputCollect$adstock == "geometric") {
        hypParam_thetas <- hypParam[paste0(InputCollect$all_media, "_thetas")]
        dt_geometric <- data.table(channels = InputCollect$all_media, thetas = hypParam_thetas)
      }
      if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
        shapeVec <- hypParam[paste0(InputCollect$all_media, "_shapes")]
        scaleVec <- hypParam[paste0(InputCollect$all_media, "_scales")]
        wb_type <- substr(InputCollect$adstock, 9, 11)
        weibullCollect <- list()
        n <- 1
        for (v1 in seq_along(InputCollect$all_media)) {
          dt_weibull <- data.table(
            x = 1:InputCollect$rollingWindowLength,
            decay_accumulated = adstock_weibull(1:InputCollect$rollingWindowLength
                                                , shape = shapeVec[v1]
                                                , scale = scaleVec[v1]
                                                , type = wb_type)$thetaVecCum,
            type = wb_type,
            channel = InputCollect$all_media[v1]
          )
          dt_weibull[, halflife := which.min(abs(decay_accumulated - 0.5))]
          max_non0 <- max(which(dt_weibull$decay_accumulated>0.001))
          dt_weibull[, cut_time := floor(max_non0 + max_non0/3)]
          weibullCollect[[n]] <- dt_weibull
          n <- n+1
        }
        weibullCollect <- rbindlist(weibullCollect)
        weibullCollect <- weibullCollect[x <= max(weibullCollect$cut_time)]
      }
      plot3data <- list(dt_geometric = dt_geometric,
                        weibullCollect = weibullCollect,
                        wb_type = toupper(wb_type))

      ## 4. Spend response curve
      dt_transformPlot <- dt_mod[, c("ds", InputCollect$all_media), with = FALSE] # independent variables
      dt_transformSpend <- cbind(dt_transformPlot[, .(ds)], InputCollect$dt_input[, c(InputCollect$paid_media_spends), with = FALSE]) # spends of indep vars
      dt_transformSpendMod <- dt_transformPlot[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, ]
      # update non-spend variables
      # if (length(InputCollect$exposure_vars) > 0) {
      #   for (expo in InputCollect$exposure_vars) {
      #     sel_nls <- ifelse(InputCollect$modNLSCollect[channel == expo, rsq_nls > rsq_lm], "nls", "lm")
      #     dt_transformSpendMod[, (expo) := InputCollect$yhatNLSCollect[channel == expo & models == sel_nls, yhat]]
      #   }
      # }
      dt_transformAdstock <- copy(dt_transformPlot)
      dt_transformSaturation <- dt_transformPlot[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]
      m_decayRate <- list()
      for (med in 1:length(InputCollect$all_media)) {
        med_select <- InputCollect$all_media[med]
        m <- dt_transformPlot[, get(med_select)]
        # Adstocking
        if (InputCollect$adstock == "geometric") {
          theta <- hypParam[paste0(InputCollect$all_media[med], "_thetas")]
          x_list <- adstock_geometric(x = m, theta = theta)
        } else if (InputCollect$adstock == "weibull_cdf") {
          shape <- hypParam[paste0(InputCollect$all_media[med], "_shapes")]
          scale <- hypParam[paste0(InputCollect$all_media[med], "_scales")]
          x_list <- adstock_weibull(x = m, shape = shape, scale = scale, type = "cdf")
        } else if (InputCollect$adstock == "weibull_pdf") {
          shape <- hypParam[paste0(InputCollect$all_media[med], "_shapes")]
          scale <- hypParam[paste0(InputCollect$all_media[med], "_scales")]
          x_list <- adstock_weibull(x = m, shape = shape, scale = scale, type = "pdf")
        }
        m_adstocked <- x_list$x_decayed
        dt_transformAdstock[, (med_select) := m_adstocked]
        m_adstockedRollWind <- m_adstocked[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]
        ## Saturation
        alpha <- hypParam[paste0(InputCollect$all_media[med], "_alphas")]
        gamma <- hypParam[paste0(InputCollect$all_media[med], "_gammas")]
        dt_transformSaturation[, (med_select) := saturation_hill(x = m_adstockedRollWind, alpha = alpha, gamma = gamma)]
      }
      dt_transformSaturationDecomp <- copy(dt_transformSaturation)
      for (i in 1:InputCollect$mediaVarCount) {
        coef <- plotWaterfallLoop[rn == InputCollect$all_media[i], coef]
        dt_transformSaturationDecomp[, (InputCollect$all_media[i]) := .SD * coef, .SDcols = InputCollect$all_media[i]]
      }
      dt_transformSaturationSpendReverse <- dt_transformAdstock[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]

      ## Reverse MM fitting
      # dt_transformSaturationSpendReverse <- copy(dt_transformAdstock[, c("ds", InputCollect$all_media), with = FALSE])
      # for (i in 1:InputCollect$mediaVarCount) {
      #   chn <- InputCollect$paid_media_vars[i]
      #   if (chn %in% InputCollect$paid_media_vars[InputCollect$exposure_selector]) {
      #     # Get Michaelis Menten nls fitting param
      #     get_chn <- dt_transformSaturationSpendReverse[, chn, with = FALSE]
      #     Vmax <- InputCollect$modNLSCollect[channel == chn, Vmax]
      #     Km <- InputCollect$modNLSCollect[channel == chn, Km]
      #     # Reverse exposure to spend
      #     dt_transformSaturationSpendReverse[, (chn) := mic_men(x = .SD, Vmax = Vmax, Km = Km, reverse = TRUE), .SDcols = chn] # .SD * Km / (Vmax - .SD) exposure to spend, reverse Michaelis Menthen: x = y*Km/(Vmax-y)
      #   } else if (chn %in% InputCollect$exposure_vars) {
      #     coef_lm <- InputCollect$modNLSCollect[channel == chn, coef_lm]
      #     dt_transformSaturationSpendReverse[, (chn) := .SD / coef_lm, .SDcols = chn]
      #   }
      # }
      # dt_transformSaturationSpendReverse <- dt_transformSaturationSpendReverse[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]

      dt_scurvePlot <- cbind(
        melt.data.table(dt_transformSaturationDecomp[, c("ds", InputCollect$all_media), with = FALSE], id.vars = "ds", variable.name = "channel", value.name = "response"),
        melt.data.table(dt_transformSaturationSpendReverse, id.vars = "ds", value.name = "spend")[, .(spend)]
      )
      # remove outlier introduced by MM nls fitting
      dt_scurvePlot <- dt_scurvePlot[spend >= 0]
      dt_scurvePlotMean <- dt_transformSpend[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, !"ds"][, lapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))), .SDcols = InputCollect$paid_media_spends]
      dt_scurvePlotMean <- melt.data.table(dt_scurvePlotMean, measure.vars = InputCollect$paid_media_spends, value.name = "mean_spend", variable.name = "channel")
      dt_scurvePlotMean[, ":="(mean_spend_scaled = 0, mean_response = 0, next_unit_response = 0)]
      for (med in 1:InputCollect$mediaVarCount) {
        get_med <- InputCollect$paid_media_spends[med]
        get_spend <- dt_scurvePlotMean[channel == get_med, mean_spend]
        get_spend_mm <- get_spend
        # if (get_med %in% InputCollect$paid_media_vars[InputCollect$exposure_selector]) {
        #   Vmax <- InputCollect$modNLSCollect[channel == get_med, Vmax]
        #   Km <- InputCollect$modNLSCollect[channel == get_med, Km]
        #   # Vmax * get_spend/(Km + get_spend)
        #   get_spend_mm <- mic_men(x = get_spend, Vmax = Vmax, Km = Km)
        # } else if (get_med %in% InputCollect$exposure_vars) {
        #   coef_lm <- InputCollect$modNLSCollect[channel == get_med, coef_lm]
        #   get_spend_mm <- get_spend * coef_lm
        # } else {
        #   get_spend_mm <- get_spend
        # }
        m <- dt_transformAdstock[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, get(get_med)]
        # m <- m[m>0] # remove outlier introduced by MM nls fitting
        alpha <- hypParam[which(paste0(get_med, "_alphas") == names(hypParam))]
        gamma <- hypParam[which(paste0(get_med, "_gammas") == names(hypParam))]
        get_response <- saturation_hill(x = m, alpha = alpha, gamma = gamma, x_marginal = get_spend_mm)
        get_response_marginal <- saturation_hill(x = m, alpha = alpha, gamma = gamma, x_marginal = get_spend_mm + 1)

        coef <- plotWaterfallLoop[rn == get_med, coef]
        dt_scurvePlotMean[channel == get_med, mean_spend_scaled := get_spend_mm]
        dt_scurvePlotMean[channel == get_med, mean_response := get_response * coef]
        dt_scurvePlotMean[channel == get_med, next_unit_response := get_response_marginal * coef - mean_response]
      }
      dt_scurvePlotMean[, solID := sid]

      # Exposure response curve
      if (!identical(InputCollect$paid_media_vars, InputCollect$exposure_vars)) {
        exposure_which <- which(InputCollect$paid_media_vars %in% InputCollect$exposure_vars)
        spends_to_fit <- InputCollect$paid_media_spends[exposure_which]
        nls_lm_selector <- InputCollect$exposure_selector[exposure_which]
        dt_expoCurvePlot <- dt_scurvePlot[channel %in% spends_to_fit]
        dt_expoCurvePlot[, exposure_pred := 0]
        for (s in seq_along(spends_to_fit)) {
          get_med <- InputCollect$exposure_vars[s]
          if (nls_lm_selector[s]) {
            Vmax <- InputCollect$modNLSCollect[channel == get_med, Vmax]
            Km <- InputCollect$modNLSCollect[channel == get_med, Km]
            # Vmax * get_spend/(Km + get_spend)
            dt_expoCurvePlot[channel == spends_to_fit[s]
                             , ':='(exposure_pred = mic_men(x = spend, Vmax = Vmax, Km = Km)
                                    ,channel = get_med)]
          } else {
            coef_lm <- InputCollect$modNLSCollect[channel == get_med, coef_lm]
            dt_expoCurvePlot[channel == spends_to_fit[s]
                             , ':='(exposure_pred = spend * coef_lm
                                    , channel = get_med)]
          }
        }
      } else {
        dt_expoCurvePlot <- NULL
      }
      plot4data <- list(dt_scurvePlot = dt_scurvePlot,
                        dt_scurvePlotMean = dt_scurvePlotMean,
                        dt_expoCurvePlot = dt_expoCurvePlot)

      ## 5. Fitted vs actual
      if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0) {
        dt_transformDecomp <- cbind(dt_modRollWind[, c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$context_vars), with = FALSE],
                                    dt_transformSaturation[, InputCollect$all_media, with = FALSE])
      } else {
        dt_transformDecomp <- cbind(dt_modRollWind[, c("ds", "dep_var", InputCollect$context_vars), with = FALSE],
                                    dt_transformSaturation[, InputCollect$all_media, with = FALSE])
      }
      col_order <- c("ds", "dep_var", InputCollect$all_ind_vars)
      setcolorder(dt_transformDecomp, neworder = col_order)
      xDecompVec <- dcast.data.table(xDecompAgg[solID == sid, .(rn, coef, solID)], solID ~ rn, value.var = "coef")
      if (!("(Intercept)" %in% names(xDecompVec))) xDecompVec[, "(Intercept)" := 0]
      setcolorder(xDecompVec, neworder = c("solID", "(Intercept)", col_order[!(col_order %in% c("ds", "dep_var"))]))
      intercept <- xDecompVec$`(Intercept)`
      xDecompVec <- data.table(mapply(
        function(scurved, coefs) scurved * coefs,
        scurved = dt_transformDecomp[, !c("ds", "dep_var"), with = FALSE],
        coefs = xDecompVec[, !c("solID", "(Intercept)")]
      ))
      xDecompVec[, intercept := intercept]
      xDecompVec[, ":="(depVarHat = rowSums(xDecompVec), solID = sid)]
      xDecompVec <- cbind(dt_transformDecomp[, .(ds, dep_var)], xDecompVec)
      xDecompVecPlot <- xDecompVec[, .(ds, dep_var, depVarHat)]
      setnames(xDecompVecPlot, old = c("ds", "dep_var", "depVarHat"), new = c("ds", "actual", "predicted"))
      suppressWarnings(xDecompVecPlotMelted <- melt.data.table(xDecompVecPlot, id.vars = "ds"))
      plot5data <- list(xDecompVecPlotMelted = xDecompVecPlotMelted)

      ## 6. Diagnostic: fitted vs residual
      plot6data <- list(xDecompVecPlot = xDecompVecPlot)

      # Gather all results
      mediaVecCollect <- bind_rows(mediaVecCollect, rbind(
        dt_transformPlot[, ":="(type = "rawMedia", solID = sid)],
        dt_transformSpend[, ":="(type = "rawSpend", solID = sid)],
        dt_transformSpendMod[, ":="(type = "predictedExposure", solID = sid)],
        dt_transformAdstock[, ":="(type = "adstockedMedia", solID = sid)],
        dt_transformSaturation[, ":="(type = "saturatedMedia", solID = sid)],
        dt_transformSaturationSpendReverse[, ":="(type = "saturatedSpendReversed", solID = sid)],
        dt_transformSaturationDecomp[, ":="(type = "decompMedia", solID = sid)],
        fill = TRUE))
      xDecompVecCollect <- bind_rows(xDecompVecCollect, xDecompVec)
      meanResponseCollect <- bind_rows(meanResponseCollect, dt_scurvePlotMean)
      plotDataCollect[[sid]] <- list(
        plot1data = plot1data,
        plot2data = plot2data,
        plot3data = plot3data,
        plot4data = plot4data,
        plot5data = plot5data,
        plot6data = plot6data)
    }
  } # end pareto front loop

  setnames(meanResponseCollect, old = "channel", new = "rn")
  setkey(meanResponseCollect, solID, rn)
  xDecompAgg <- merge(xDecompAgg, meanResponseCollect[, .(rn, solID, mean_response, next_unit_response)], all.x = TRUE)

  pareto_results <- list(
    resultHypParam = resultHypParam,
    xDecompAgg = xDecompAgg,
    mediaVecCollect = as.data.table(mediaVecCollect),
    xDecompVecCollect = as.data.table(xDecompVecCollect),
    plotDataCollect = plotDataCollect
  )

  # if (check_parallel()) stopImplicitCluster()
  # close(pbplot)

  return(pareto_results)
}
