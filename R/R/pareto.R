# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

robyn_pareto <- function(InputCollect, OutputModels,
                         pareto_fronts = "auto",
                         min_candidates = 100,
                         calibration_constraint = 0.1,
                         quiet = FALSE,
                         calibrated = FALSE,
                         ...) {
  hyper_fixed <- OutputModels$hyper_fixed
  OutModels <- OutputModels[unlist(lapply(OutputModels, function(x) "resultCollect" %in% names(x)))]

  resultHypParam <- bind_rows(lapply(OutModels, function(x) {
    mutate(x$resultCollect$resultHypParam, trial = x$trial)
  }))

  xDecompAgg <- bind_rows(lapply(OutModels, function(x) {
    mutate(x$resultCollect$xDecompAgg, trial = x$trial)
  }))

  if (calibrated) {
    resultCalibration <- bind_rows(lapply(OutModels, function(x) {
      x$resultCollect$liftCalibration %>%
        mutate(trial = x$trial) %>%
        rename(rn = .data$liftMedia)
    }))
  } else {
    resultCalibration <- NULL
  }

  if (!hyper_fixed) {
    df_names <- if (calibrated) {
      c("resultHypParam", "xDecompAgg", "resultCalibration")
    } else {
      c("resultHypParam", "xDecompAgg")
    }
    for (df in df_names) {
      assign(df, get(df) %>% mutate(
        iterations = (.data$iterNG - 1) * OutputModels$cores + .data$iterPar
      ))
    }
  } else if (hyper_fixed & calibrated) {
    df_names <- "resultCalibration"
    for (df in df_names) {
      assign(df, get(df) %>% mutate(
        iterations = (.data$iterNG - 1) * OutputModels$cores + .data$iterPar
      ))
    }
  }

  # If recreated model, inherit bootstrap results
  if (length(unique(xDecompAgg$solID)) == 1 & !"boot_mean" %in% colnames(xDecompAgg)) {
    bootstrap <- attr(OutputModels, "bootstrap")
    if (!is.null(bootstrap)) {
      xDecompAgg <- left_join(xDecompAgg, bootstrap, by = c("rn" = "variable"))
    }
  }
  xDecompAggPaid <- xDecompAgg %>% filter(.data$rn %in% InputCollect$paid_media_selected)
  xDecompAggCoef0 <- xDecompAggPaid %>%
    group_by(.data$solID) %>%
    summarise(coef0 = min(.data$coef, na.rm = TRUE) == 0)

  if (!hyper_fixed) {
    mape_lift_quantile10 <- quantile(resultHypParam$mape, probs = calibration_constraint, na.rm = TRUE)
    nrmse_quantile90 <- quantile(resultHypParam$nrmse, probs = 0.90, na.rm = TRUE)
    decomprssd_quantile90 <- quantile(resultHypParam$decomp.rssd, probs = 0.90, na.rm = TRUE)
    resultHypParam <- left_join(resultHypParam, xDecompAggCoef0, by = "solID") %>%
      mutate(
        mape.qt10 =
          .data$mape <= mape_lift_quantile10 &
            .data$nrmse <= nrmse_quantile90 &
            .data$decomp.rssd <= decomprssd_quantile90
      )
    # Calculate Pareto-fronts (for "all" or pareto_fronts)
    resultHypParamPareto <- filter(resultHypParam, .data$mape.qt10 == TRUE)
    paretoResults <- pareto_front(
      xi = resultHypParamPareto$nrmse,
      yi = resultHypParamPareto$decomp.rssd,
      pareto_fronts = ifelse("auto" %in% pareto_fronts, Inf, pareto_fronts),
      sort = FALSE
    )
    resultHypParamPareto <- resultHypParamPareto %>%
      left_join(paretoResults, by = c("nrmse" = "x", "decomp.rssd" = "y")) %>%
      rename("robynPareto" = "pareto_front") %>%
      arrange(.data$iterNG, .data$iterPar, .data$nrmse) %>%
      select(.data$solID, .data$robynPareto) %>%
      group_by(.data$solID) %>%
      arrange(.data$robynPareto) %>%
      slice(1)
    resultHypParam <- left_join(resultHypParam, resultHypParamPareto, by = "solID")
  } else {
    resultHypParam <- mutate(resultHypParam, mape.qt10 = TRUE, robynPareto = 1, coef0 = NA)
  }

  # Calculate combined weighted error scores
  resultHypParam$error_score <- errors_scores(resultHypParam, ts_validation = OutputModels$ts_validation, ...)

  # Bind robynPareto results
  xDecompAgg <- left_join(xDecompAgg, select(resultHypParam, .data$robynPareto, .data$solID), by = "solID")
  xDecompAggMedia <- xDecompAgg %>%
    filter(.data$rn %in% InputCollect$all_media) %>%
    select(c("rn", "solID", "coef", "mean_spend", "mean_exposure", "xDecompAgg", "total_spend", "robynPareto"))

  # Prepare parallel loop
  if (TRUE) {
    if (OutputModels$cores > 1) {
      registerDoParallel(OutputModels$cores)
      registerDoSEQ()
    }
    if (hyper_fixed) pareto_fronts <- 1
    # Get at least 100 candidates for better clustering
    if (nrow(resultHypParam) == 1) pareto_fronts <- 1
    if ("auto" %in% pareto_fronts) {
      n_pareto <- resultHypParam %>%
        filter(!is.na(.data$robynPareto)) %>%
        nrow()
      if (n_pareto <= min_candidates & nrow(resultHypParam) > 1 & !calibrated) {
        stop(paste(
          "Less than", min_candidates, "candidates in pareto fronts.",
          "Increase iterations to get more model candidates or decrease min_candidates in robyn_output()"
        ))
      }
      auto_pareto <- resultHypParam %>%
        filter(!is.na(.data$robynPareto)) %>%
        group_by(.data$robynPareto) %>%
        summarise(n = n_distinct(.data$solID)) %>%
        mutate(n_cum = cumsum(.data$n)) %>%
        filter(.data$n_cum >= min_candidates) %>%
        slice(1)
      message(sprintf(
        ">> Automatically selected %s Pareto-fronts to contain at least %s pareto-optimal models (%s)",
        auto_pareto$robynPareto, min_candidates, auto_pareto$n_cum
      ))
      pareto_fronts <- as.integer(auto_pareto$robynPareto)
    }
    pareto_fronts_vec <- 1:pareto_fronts

    # decompSpendDistPar <- decompSpendDist[decompSpendDist$robynPareto %in% pareto_fronts_vec, ]
    resultHypParamPar <- resultHypParam[resultHypParam$robynPareto %in% pareto_fronts_vec, ]
    # xDecompAggPar <- xDecompAgg[xDecompAgg$robynPareto %in% pareto_fronts_vec, ]
    xDecompAggMediaPar <- xDecompAggMedia %>% filter(.data$robynPareto %in% pareto_fronts_vec)
    respN <- NULL
  }

  if (!quiet) {
    message(sprintf(
      ">>> Calculating response curves for all models' media variables (%s)...",
      nrow(xDecompAggMediaPar)
    ))
  }

  cnt_resp <- nrow(xDecompAggMediaPar)
  pb_resp <- txtProgressBar(min = 0, max = cnt_resp, style = 3)
  resp_collect <- lapply(
    1:cnt_resp,
    function(respN) {
      setTxtProgressBar(pb_resp, respN)
      get_solID <- xDecompAggMediaPar$solID[respN]
      get_media_name <- xDecompAggMediaPar$rn[respN]
      window_start_loc <- InputCollect$rollingWindowStartWhich
      window_end_loc <- InputCollect$rollingWindowEndWhich

      get_resp <- robyn_response(
        select_model = get_solID,
        metric_name = get_media_name,
        date_range = "all",
        dt_hyppar = resultHypParamPar,
        dt_coef = xDecompAggMediaPar,
        InputCollect = InputCollect,
        OutputCollect = OutputModels,
        quiet = TRUE,
        ...
      )
      list_response <- list(
        dt_resp = data.frame(
          mean_response = get_resp$mean_response,
          mean_spend_adstocked = get_resp$mean_input_immediate + get_resp$mean_input_carryover,
          mean_carryover = get_resp$mean_input_carryover,
          rn = get_media_name,
          solID = get_solID
        ),
        dt_resp_vec = data.frame(
          channel = rep(get_media_name, length(get_resp$response_total)),
          response = get_resp$response_total,
          response_carryover = get_resp$response_carryover,
          spend = get_resp$input_total[window_start_loc:window_end_loc],
          solID = rep(get_solID, length(get_resp$response_total))
        )
      )
      return(list_response)
    }
  )
  close(pb_resp)
  dt_resp <- bind_rows(lapply(resp_collect, function(x) x[["dt_resp"]]))
  dt_resp_vec <- bind_rows(lapply(resp_collect, function(x) x[["dt_resp_vec"]]))

  xDecompAgg <- xDecompAgg %>%
    left_join(
      dt_resp,
      by = c("solID", "rn")
    ) %>%
    mutate(
      roi_mean = .data$mean_response / .data$mean_spend,
      roi_total = .data$xDecompAgg / .data$total_spend,
      cpa_mean = .data$mean_spend / .data$mean_response,
      cpa_total = .data$total_spend / .data$xDecompAgg
    )

  # Pareto loop (no plots)
  mediaVecCollect <- list()
  xDecompVecCollect <- list()
  plotDataCollect <- list()
  df_caov_pct_all <- dplyr::tibble()
  dt_mod <- InputCollect$dt_mod
  dt_modRollWind <- InputCollect$dt_modRollWind
  rw_start_loc <- InputCollect$rollingWindowStartWhich
  rw_end_loc <- InputCollect$rollingWindowEndWhich
  dt_ds <- dt_mod[rw_start_loc:rw_end_loc, "ds"]

  for (pf in pareto_fronts_vec) {
    plotMediaShare <- filter(
      xDecompAgg,
      .data$robynPareto == pf,
      .data$rn %in% InputCollect$paid_media_selected
    )
    uniqueSol <- unique(plotMediaShare$solID)
    plotWaterfall <- xDecompAgg %>% filter(.data$robynPareto == pf)
    if (!quiet & length(unique(xDecompAgg$solID)) > 1) {
      message(sprintf(">> Pareto-Front: %s [%s models]", pf, length(uniqueSol)))
    }

    # Calculations for pareto AND pareto plots
    for (sid in uniqueSol) {
      # parallelResult <- foreach(sid = uniqueSol) %dorng% {
      if (!quiet & length(unique(xDecompAgg$solID)) > 1) {
        lares::statusbar(which(sid == uniqueSol), length(uniqueSol), type = "equal")
      }

      ## 1. Spend x effect share comparison
      temp <- plotMediaShare[plotMediaShare$solID == sid, ] %>%
        tidyr::gather(
          "variable", "value",
          c("spend_share", "effect_share", "roi_total", "cpa_total")
        ) %>%
        select(c("rn", "nrmse", "decomp.rssd", "rsq_train", "variable", "value")) %>%
        mutate(rn = factor(.data$rn, levels = sort(InputCollect$paid_media_selected)))
      plotMediaShareLoopBar <- filter(temp, .data$variable %in% c("spend_share", "effect_share"))
      plotMediaShareLoopLine <- filter(temp, .data$variable == ifelse(
        InputCollect$dep_var_type == "conversion", "cpa_total", "roi_total"
      ))
      line_rm_inf <- !is.infinite(plotMediaShareLoopLine$value)
      ySecScale <- max(plotMediaShareLoopLine$value[line_rm_inf]) /
        max(plotMediaShareLoopBar$value) * 1.1
      plot1data <- list(
        plotMediaShareLoopBar = plotMediaShareLoopBar,
        plotMediaShareLoopLine = plotMediaShareLoopLine,
        ySecScale = ySecScale
      )

      ## 2. Waterfall
      plotWaterfallLoop <- plotWaterfall %>%
        filter(.data$solID == sid) %>%
        arrange(.data$xDecompPerc) %>%
        mutate(
          end = 1 - cumsum(.data$xDecompPerc),
          start = lag(.data$end),
          start = ifelse(is.na(.data$start), 1, .data$start),
          id = row_number(),
          rn = as.factor(.data$rn),
          sign = as.factor(ifelse(.data$xDecompPerc >= 0, "Positive", "Negative"))
        ) %>%
        select(
          .data$id, .data$rn, .data$coef,
          .data$xDecompAgg, .data$xDecompPerc,
          .data$start, .data$end, .data$sign
        )
      plot2data <- list(plotWaterfallLoop = plotWaterfallLoop)

      ## 3. Adstock rate
      dt_geometric <- weibullCollect <- wb_type <- NULL
      resultHypParamLoop <- resultHypParam[resultHypParam$solID == sid, ]
      get_hp_names <- !endsWith(names(InputCollect$hyperparameters), "_penalty")
      get_hp_names <- names(InputCollect$hyperparameters)[get_hp_names]
      hypParam <- resultHypParamLoop[, get_hp_names]
      if (InputCollect$adstock == "geometric") {
        hypParam_thetas <- unlist(hypParam[paste0(InputCollect$all_media, "_thetas")])
        dt_geometric <- data.frame(channels = InputCollect$all_media, thetas = hypParam_thetas)
      }
      if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
        shapeVec <- unlist(hypParam[paste0(InputCollect$all_media, "_shapes")])
        scaleVec <- unlist(hypParam[paste0(InputCollect$all_media, "_scales")])
        wb_type <- substr(InputCollect$adstock, 9, 11)
        weibullCollect <- list()
        n <- 1
        for (v1 in seq_along(InputCollect$all_media)) {
          dt_weibull <- data.frame(
            x = 1:InputCollect$rollingWindowLength,
            decay_accumulated = adstock_weibull(
              1:InputCollect$rollingWindowLength,
              shape = shapeVec[v1],
              scale = scaleVec[v1],
              type = wb_type
            )$thetaVecCum,
            type = wb_type,
            channel = InputCollect$all_media[v1]
          ) %>%
            mutate(halflife = which.min(abs(.data$decay_accumulated - 0.5)))
          max_non0 <- max(which(dt_weibull$decay_accumulated > 0.001), na.rm = TRUE)
          dt_weibull$cut_time <- ifelse(max_non0 <= 5, max_non0 * 2, floor(max_non0 + max_non0 / 3))
          weibullCollect[[n]] <- dt_weibull
          n <- n + 1
        }
        weibullCollect <- bind_rows(weibullCollect)
        weibullCollect <- filter(weibullCollect, .data$x <= max(weibullCollect$cut_time))
      }

      plot3data <- list(
        dt_geometric = dt_geometric,
        weibullCollect = weibullCollect,
        wb_type = toupper(wb_type)
      )

      ## 4. Spend response curve
      dt_resp_vec_loop <- cbind(
        dt_ds,
        dt_resp_vec %>%
          filter(.data$solID == sid) %>%
          select(c("channel", "spend", "response"))
      )
      dt_transformAdstock <- dt_resp_vec_loop %>%
        select(c("ds", "channel", "spend")) %>%
        pivot_wider(values_from = "spend", names_from = "channel")
      dt_transformSaturationDecomp <- dt_resp_vec_loop %>%
        select(c("ds", "channel", "response")) %>%
        pivot_wider(values_from = "response", names_from = "channel")
      dt_scurvePlotMean <- plotWaterfall %>%
        filter(.data$solID == sid & !is.na(.data$mean_spend)) %>%
        select(c(
          channel = "rn", "mean_spend", "mean_spend_adstocked",
          "mean_carryover", "mean_response", "solID"
        ))
      # Exposure response curve
      plot4data <- list(
        dt_scurvePlot = dt_resp_vec_loop,
        dt_scurvePlotMean = dt_scurvePlotMean
      )

      ## 5. Fitted vs actual
      temp_order1 <- c("ds", "dep_var")
      temp_order2 <- c("(Intercept)", InputCollect$prophet_vars, InputCollect$context_vars)
      dt_transformDecomp <- dt_modRollWind %>%
        mutate("(Intercept)" = 1) %>%
        select(all_of(c(temp_order1, temp_order2)))
      xDecompVec <- xDecompAgg %>%
        filter(.data$solID == sid & .data$rn %in% temp_order2) %>%
        select(.data$rn, .data$coef) %>%
        pivot_wider(values_from = "coef", names_from = "rn") %>%
        mutate("(Intercept)" = ifelse(
          "(Intercept)" %in% levels(plotWaterfallLoop$rn),
          .data$`(Intercept)`, 0
        ))
      xDecompVec <- bind_cols(
        dt_transformDecomp %>% select(temp_order1),
        data.frame(mapply(
          function(vec, coefs) {
            vec * coefs
          },
          vec = select(dt_transformDecomp, -temp_order1),
          coefs = xDecompVec
        ), check.names = FALSE),
        dt_transformSaturationDecomp %>% select(-"ds")
      ) %>%
        rename("intercept" = "(Intercept)") %>%
        mutate(
          depVarHat = rowSums(select(., -temp_order1)),
          solID = sid
        ) %>%
        select(c(
          "ds", "dep_var", InputCollect$all_ind_vars,
          "intercept", "depVarHat", "solID"
        ))

      xDecompVecPlot <- select(xDecompVec, .data$ds, .data$dep_var, .data$depVarHat) %>%
        rename("actual" = "dep_var", "predicted" = "depVarHat")
      xDecompVecPlotMelted <- xDecompVecPlot %>%
        pivot_longer(names_to = "variable", values_to = "value", -.data$ds) %>%
        arrange(.data$variable, .data$ds)
      rsq <- filter(resultHypParam, .data$solID == sid) %>%
        pull(.data$rsq_train)
      plot5data <- list(xDecompVecPlotMelted = xDecompVecPlotMelted, rsq = rsq)

      ## 6. Diagnostic: fitted vs residual
      plot6data <- list(xDecompVecPlot = xDecompVecPlot)

      ## 7. Immediate vs carryover response

      temp_p7 <- dt_resp_vec %>%
        filter(.data$solID == sid) %>%
        group_by(.data$channel) %>%
        summarise(Total = sum(.data$response), Carryover = sum(.data$response_carryover)) %>%
        mutate(
          Immediate = .data$Total - .data$Carryover,
          perc_imme = 1 - .data$Carryover / .data$Total,
          perc_caov = .data$Carryover / .data$Total,
          carryover_pct = .data$Carryover / .data$Total
        )
      plot7data <- bind_cols(
        temp_p7 %>%
          select(rn = "channel", "Immediate", "Carryover") %>%
          pivot_longer(names_to = "type", values_to = "response", cols = -"rn"),
        temp_p7 %>%
          select(rn = "channel", Immediate = "perc_imme", Carryover = "perc_caov") %>%
          pivot_longer(names_to = "type", values_to = "percentage", cols = -"rn") %>%
          select("percentage"),
        temp_p7 %>%
          select(rn = "channel", Immediate = "perc_caov", Carryover = "perc_caov") %>%
          pivot_longer(names_to = "type", values_to = "carryover_pct", cols = -"rn") %>%
          select("carryover_pct")
      ) %>% mutate(solID = sid)
      df_caov_pct_all <- rbind(df_caov_pct_all, plot7data)

      ## 8. Bootstrapped ROI/CPA with CIs
      # plot8data <- "Empty" # Filled when running robyn_onepagers() with clustering data

      # Gather all results
      mediaVecCollect <- bind_rows(mediaVecCollect, list(
        mutate(dt_transformAdstock, type = "adstockedMedia", solID = sid),
        mutate(dt_transformSaturationDecomp, type = "decompMedia", solID = sid)
      ))
      xDecompVecCollect <- bind_rows(xDecompVecCollect, xDecompVec)
      plotDataCollect[[sid]] <- list(
        plot1data = plot1data,
        plot2data = plot2data,
        plot3data = plot3data,
        plot4data = plot4data,
        plot5data = plot5data,
        plot6data = plot6data,
        plot7data = plot7data
        # plot8data = plot8data
      )
    }
  } # end pareto front loopdev

  pareto_results <- list(
    pareto_solutions = unique(xDecompVecCollect$solID),
    pareto_fronts = pareto_fronts,
    resultHypParam = resultHypParam,
    xDecompAgg = xDecompAgg,
    resultCalibration = resultCalibration,
    mediaVecCollect = mediaVecCollect,
    xDecompVecCollect = xDecompVecCollect,
    plotDataCollect = plotDataCollect,
    df_caov_pct_all = df_caov_pct_all
  )

  if (OutputModels$cores > 1) stopImplicitCluster()

  return(pareto_results)
}

#' @rdname robyn_outputs
#' @param xi,yi Numeric. Coordinates values per observation.
#' @export
pareto_front <- function(xi, yi, pareto_fronts = 1, ...) {
  stopifnot(length(xi) == length(yi))
  d <- data.frame(xi, yi)
  Dtemp <- D <- d[order(d$xi, d$yi, decreasing = FALSE), ]
  df <- data.frame()
  i <- 1
  while (nrow(Dtemp) >= 1 & i <= max(pareto_fronts)) {
    these <- Dtemp[which(!duplicated(cummin(Dtemp$yi))), ]
    these$pareto_front <- i
    df <- rbind(df, these)
    Dtemp <- Dtemp[!row.names(Dtemp) %in% row.names(these), ]
    i <- i + 1
  }
  ret <- merge(x = d, y = df, by = c("xi", "yi"), all.x = TRUE, ...)
  colnames(ret) <- c("x", "y", "pareto_front")
  return(ret)
}

#' @rdname robyn_outputs
#' @param start_date,end_date Character/Date. Dates to consider when calculating
#' immediate and carryover values per channel.
#' @export
robyn_immcarr <- function(
    InputCollect, OutputCollect, solID = NULL,
    start_date = NULL, end_date = NULL, ...) {
  # Define default values when not provided
  if (is.null(solID)) solID <- OutputCollect$resultHypParam$solID[1]
  if (is.null(start_date)) start_date <- InputCollect$window_start
  if (is.null(end_date)) end_date <- InputCollect$window_end
  # Get closer dates to date passed
  start_date <- InputCollect$dt_modRollWind$ds[
    which.min(abs(as.Date(start_date) - InputCollect$dt_modRollWind$ds))
  ]
  end_date <- InputCollect$dt_modRollWind$ds[
    which.min(abs(as.Date(end_date) - InputCollect$dt_modRollWind$ds))
  ]
  # Filter for custom window
  rollingWindowStartWhich <- which(InputCollect$dt_modRollWind$ds == start_date)
  rollingWindowEndWhich <- which(InputCollect$dt_modRollWind$ds == end_date)
  rollingWindow <- rollingWindowStartWhich:rollingWindowEndWhich
  # Calculate saturated dataframes with carryover and immediate parts
  hypParamSam <- OutputCollect$resultHypParam[OutputCollect$resultHypParam$solID == solID, ]
  dt_saturated_dfs <- run_transformations(
    all_media = InputCollect$all_media,
    window_start_loc = InputCollect$rollingWindowStartWhich,
    window_end_loc = InputCollect$rollingWindowEndWhich,
    dt_mod = InputCollect$dt_mod,
    adstock = InputCollect$adstock,
    dt_hyppar = hypParamSam, ...
  )
  # Calculate decomposition
  coefs <- OutputCollect$xDecompAgg$coef[OutputCollect$xDecompAgg$solID == solID]
  names(coefs) <- OutputCollect$xDecompAgg$rn[OutputCollect$xDecompAgg$solID == solID]
  decompCollect <- model_decomp(
    inputs = list(
      coefs = coefs,
      y_pred = dt_saturated_dfs$dt_modSaturated$dep_var[rollingWindow],
      dt_modSaturated = dt_saturated_dfs$dt_modSaturated[rollingWindow, ],
      dt_saturatedImmediate = dt_saturated_dfs$dt_saturatedImmediate[rollingWindow, ],
      dt_saturatedCarryover = dt_saturated_dfs$dt_saturatedCarryover[rollingWindow, ],
      dt_modRollWind = InputCollect$dt_modRollWind[rollingWindow, ],
      refreshAddedStart = start_date
    )
  )
  mediaDecompImmediate <- select(decompCollect$mediaDecompImmediate, -"ds", -"y")
  colnames(mediaDecompImmediate) <- paste0(colnames(mediaDecompImmediate), "_MDI")
  mediaDecompCarryover <- select(decompCollect$mediaDecompCarryover, -"ds", -"y")
  colnames(mediaDecompCarryover) <- paste0(colnames(mediaDecompCarryover), "_MDC")
  temp <- bind_cols(
    decompCollect$xDecompVec,
    mediaDecompImmediate,
    mediaDecompCarryover
  ) %>% mutate(solID = solID)
  vec_collect <- list(
    xDecompVec = select(temp, -dplyr::ends_with("_MDI"), -dplyr::ends_with("_MDC")),
    xDecompVecImmediate = select(temp, -dplyr::ends_with("_MDC"), -all_of(InputCollect$all_media)),
    xDecompVecCarryover = select(temp, -dplyr::ends_with("_MDI"), -all_of(InputCollect$all_media))
  )
  this <- gsub("_MDI", "", colnames(vec_collect$xDecompVecImmediate))
  colnames(vec_collect$xDecompVecImmediate) <- colnames(vec_collect$xDecompVecCarryover) <- this
  df_caov <- vec_collect$xDecompVecCarryover %>%
    group_by(.data$solID) %>%
    summarise(across(InputCollect$all_media, sum))
  df_total <- vec_collect$xDecompVec %>%
    group_by(.data$solID) %>%
    summarise(across(InputCollect$all_media, sum))
  df_caov_pct <- bind_cols(
    select(df_caov, "solID"),
    select(df_caov, -"solID") / select(df_total, -"solID")
  ) %>%
    pivot_longer(cols = InputCollect$all_media, names_to = "rn", values_to = "carryover_pct")
  df_caov_pct[is.na(as.matrix(df_caov_pct))] <- 0
  # Gather everything in an aggregated format
  xDecompVecImmeCaov <- bind_rows(
    select(vec_collect$xDecompVecImmediate, c("ds", InputCollect$all_media, "solID")) %>%
      mutate(type = "Immediate"),
    select(vec_collect$xDecompVecCarryover, c("ds", InputCollect$all_media, "solID")) %>%
      mutate(type = "Carryover")
  ) %>%
    pivot_longer(cols = InputCollect$all_media, names_to = "rn") %>%
    mutate(start_date = start_date, end_date = end_date) %>%
    select("solID", ends_with("_date"), "type", "rn", "value") %>%
    group_by(.data$solID, .data$start_date, .data$end_date, .data$rn, .data$type) %>%
    summarise(response = sum(.data$value), .groups = "drop_last") %>%
    mutate(percentage = .data$response / sum(.data$response)) %>%
    replace(., is.na(.), 0) %>%
    ungroup() %>%
    left_join(df_caov_pct, c("solID", "rn"))
  return(xDecompVecImmeCaov)
}
