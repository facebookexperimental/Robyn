# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Generate and Export Robyn Plots
#'
#' @rdname robyn_outputs
#' @export
robyn_plots <- function(InputCollect, OutputCollect, export = TRUE) {
  check_class("robyn_outputs", OutputCollect)
  pareto_fronts <- OutputCollect$pareto_fronts
  hyper_fixed <- OutputCollect$hyper_fixed
  temp_all <- OutputCollect$allPareto
  all_plots <- list()

  if (!hyper_fixed) {

    ## Prophet
    if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0 ||
      !is.null(InputCollect$factor_vars) && length(InputCollect$factor_vars) > 0) {
      dt_plotProphet <- InputCollect$dt_mod[, c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$factor_vars), with = FALSE]
      dt_plotProphet <- suppressWarnings(melt.data.table(dt_plotProphet, id.vars = "ds"))
      all_plots[["pProphet"]] <- pProphet <- ggplot(
        dt_plotProphet, aes(x = ds, y = value)
      ) +
        geom_line(color = "steelblue") +
        facet_wrap(~variable, scales = "free", ncol = 1) +
        labs(title = "Prophet decomposition") +
        xlab(NULL) + ylab(NULL) + theme_lares() + scale_y_abbr()
      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "prophet_decomp.png"),
          plot = pProphet, limitsize = FALSE,
          dpi = 600, width = 12, height = 3 * length(levels(dt_plotProphet$variable))
        )
      }
    }

    ## Spend exposure model
    if (any(InputCollect$exposure_selector)) {
      all_plots[["pSpendExposure"]] <- pSpendExposure <- wrap_plots(
        InputCollect$plotNLSCollect,
        ncol = ifelse(length(InputCollect$plotNLSCollect) <= 3, length(InputCollect$plotNLSCollect), 3)
      ) +
        plot_annotation(
          title = "Spend-exposure fitting with Michaelis-Menten model",
          theme = theme(plot.title = element_text(hjust = 0.5))
        )
      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "spend_exposure_fitting.png"),
          plot = pSpendExposure, dpi = 600, width = 12, limitsize = FALSE,
          height = ceiling(length(InputCollect$plotNLSCollect) / 3) * 7
        )
      }
    } else {
      # message("No spend-exposure modelling needed. All media variables used for MMM are spend variables")
    }

    ## Hyperparameter sampling distribution
    if (length(temp_all) > 0) {
      resultHypParam <- copy(temp_all$resultHypParam)
      hpnames_updated <- c(names(OutputCollect$OutputModels$hyper_updated), "robynPareto")
      hpnames_updated <- str_replace(hpnames_updated, "lambda", "lambda_hp")
      resultHypParam.melted <- melt.data.table(resultHypParam[, hpnames_updated, with = FALSE],
        id.vars = c("robynPareto")
      )
      resultHypParam.melted <- resultHypParam.melted[variable == "lambda_hp", variable := "lambda"]
      all_plots[["pSamp"]] <- ggplot(
        resultHypParam.melted, aes(x = value, y = variable, color = variable, fill = variable)
      ) +
        geom_violin(alpha = .5, size = 0) +
        geom_point(size = 0.2) +
        theme_lares(legend = "none") +
        labs(
          title = "Hyperparameter Optimisation Sampling",
          subtitle = paste0("Sample distribution", ", iterations = ", OutputCollect$iterations, " x ", OutputCollect$trials, " trial"),
          x = "Hyperparameter space",
          y = NULL
        )
      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "hypersampling.png"),
          plot = all_plots$pSamp, dpi = 600, width = 12, height = 7, limitsize = FALSE
        )
      }
    }

    ## Pareto front
    if (length(temp_all) > 0) {
      pareto_fronts_vec <- 1:pareto_fronts
      resultHypParam <- copy(temp_all$resultHypParam)
      if (!is.null(InputCollect$calibration_input)) {
        resultHypParam[, iterations := ifelse(is.na(robynPareto), NA, iterations)]
      }

      calibrated <- !is.null(InputCollect$calibration_input)
      pParFront <- ggplot(resultHypParam, aes(
        x = .data$nrmse, y = .data$decomp.rssd, colour = .data$iterations
      )) +
        scale_colour_gradient(low = "skyblue", high = "navyblue") +
        labs(
          title = ifelse(!calibrated, "Multi-objective Evolutionary Performance",
            "Multi-objective Evolutionary Performance with Calibration"
          ),
          subtitle = sprintf(
            "2D Pareto fronts with %s, for %s trial%s with %s iterations each",
            OutputCollect$nevergrad_algo, OutputCollect$trials,
            ifelse(pareto_fronts > 1, "s", ""), OutputCollect$iterations
          ),
          x = "NRMSE",
          y = "DECOMP.RSSD",
          colour = "Iterations",
          size = "MAPE",
          alpha = NULL
        ) +
        theme_lares()
      # Add MAPE dimension when calibrated
      if (calibrated) {
        pParFront <- pParFront +
          geom_point(data = resultHypParam, aes(size = .data$mape, alpha = 1 - .data$mape))
      } else {
        pParFront <- pParFront + geom_point()
      }
      # Add pareto front lines
      for (pfs in 1:max(pareto_fronts_vec)) {
        if (pfs == 2) {
          pf_color <- "coral3"
        } else if (pfs == 3) {
          pf_color <- "coral2"
        } else {
          pf_color <- "coral"
        }
        pParFront <- pParFront + geom_line(
          data = resultHypParam[robynPareto == pfs],
          aes(x = .data$nrmse, y = .data$decomp.rssd), colour = pf_color
        )
      }
      all_plots[["pParFront"]] <- pParFront
      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "pareto_front.png"),
          plot = pParFront, limitsize = FALSE,
          dpi = 600, width = 12, height = 7
        )
      }
    }

    ## Ridgeline model convergence
    if (length(temp_all) > 0) {
      xDecompAgg <- copy(temp_all$xDecompAgg)
      dt_ridges <- xDecompAgg[
        rn %in% InputCollect$paid_media_spends,
        .(
          variables = rn,
          roi_total,
          iteration = (iterNG - 1) * OutputCollect$cores + iterPar,
          trial
        )
      ][order(iteration, variables)]
      bin_limits <- c(1, 20)
      qt_len <- ifelse(OutputCollect$iterations <= 100, 1,
        ifelse(OutputCollect$iterations > 2000, 20, ceiling(OutputCollect$iterations / 100))
      )
      set_qt <- floor(quantile(1:OutputCollect$iterations, seq(0, 1, length.out = qt_len + 1)))
      set_bin <- set_qt[-1]
      dt_ridges[, iter_bin := cut(dt_ridges$iteration, breaks = set_qt, labels = set_bin)]
      dt_ridges <- dt_ridges[!is.na(iter_bin)]
      dt_ridges[, iter_bin := factor(iter_bin, levels = sort(set_bin, decreasing = TRUE))]
      dt_ridges[, trial := as.factor(trial)]
      plot_vars <- dt_ridges[, unique(variables)]
      plot_n <- ceiling(length(plot_vars) / 6)
      for (pl in 1:plot_n) {
        loop_vars <- na.omit(plot_vars[(1:6) + 6 * (pl - 1)])
        dt_ridges_loop <- dt_ridges[variables %in% loop_vars, ]
        all_plots[[paste0("pRidges", pl)]] <- pRidges <- ggplot(
          dt_ridges_loop, aes(x = roi_total, y = iter_bin, fill = as.integer(iter_bin), linetype = trial)
        ) +
          scale_fill_distiller(palette = "GnBu") +
          geom_density_ridges(scale = 4, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7) +
          facet_wrap(~ .data$variables, scales = "free") +
          guides(fill = "none", linetype = "none") +
          theme_lares() +
          labs(
            x = "Total ROAS by Channel", y = NULL,
            title = "ROAS Distribution over Iteration Buckets"
          )
        if (export) {
          suppressMessages(ggsave(
            paste0(OutputCollect$plot_folder, "roas_convergence", pl, ".png"),
            plot = pRidges, dpi = 600, width = 12, limitsize = FALSE,
            height = ceiling(length(loop_vars) / 3) * 6
          ))
        }
      }
    }
  } # End of !hyper_fixed

  return(invisible(all_plots))
}


####################################################################
#' Generate and Export Robyn One-Pager Plots
#'
#' @inheritParams robyn_outputs
#' @inheritParams robyn_csv
#' @export
robyn_onepagers <- function(InputCollect, OutputCollect, select_model = NULL, quiet = FALSE, export = TRUE) {
  check_class("robyn_outputs", OutputCollect)
  pareto_fronts <- OutputCollect$pareto_fronts
  hyper_fixed <- OutputCollect$hyper_fixed
  resultHypParam <- copy(OutputCollect$resultHypParam)
  xDecompAgg <- copy(OutputCollect$xDecompAgg)
  if (!is.null(select_model)) {
    if ("clusters" %in% select_model) select_model <- OutputCollect$clusters$models$solID
    resultHypParam <- resultHypParam[solID %in% select_model]
    xDecompAgg <- xDecompAgg[solID %in% select_model]
    if (!quiet) message(">> Generating only cluster results one-pagers (", nrow(resultHypParam), ")...")
  }

  # Prepare for parallel plotting
  if (check_parallel_plot()) registerDoParallel(OutputCollect$cores) else registerDoSEQ()
  if (!hyper_fixed) {
    pareto_fronts_vec <- 1:pareto_fronts
    count_mod_out <- resultHypParam[robynPareto %in% pareto_fronts_vec, .N]
  } else {
    pareto_fronts_vec <- 1
    count_mod_out <- nrow(resultHypParam)
  }
  all_fronts <- unique(xDecompAgg$robynPareto)
  all_fronts <- sort(all_fronts[!is.na(all_fronts)])
  if (!all(pareto_fronts_vec %in% all_fronts)) pareto_fronts_vec <- all_fronts

  if (check_parallel_plot()) {
    if (!quiet) message(paste(">> Plotting", count_mod_out, "selected models on", OutputCollect$cores, "cores..."))
  } else {
    if (!quiet) message(paste(">> Plotting", count_mod_out, "selected models on 1 core (MacOS fallback)..."))
  }

  if (!quiet & count_mod_out > 0) pbplot <- txtProgressBar(min = 0, max = count_mod_out, style = 3)
  temp <- OutputCollect$allPareto$plotDataCollect
  all_plots <- list()
  cnt <- 0

  for (pf in pareto_fronts_vec) { # pf = pareto_fronts_vec[1]

    plotMediaShare <- xDecompAgg[robynPareto == pf & rn %in% InputCollect$paid_media_spends]
    uniqueSol <- plotMediaShare[, unique(solID)]

    # parallelResult <- for (sid in uniqueSol) { # sid = uniqueSol[1]
    parallelResult <- foreach(sid = uniqueSol) %dorng% { # sid = uniqueSol[1]
      plotMediaShareLoop <- plotMediaShare[solID == sid]
      rsq_train_plot <- plotMediaShareLoop[, round(unique(rsq_train), 4)]
      nrmse_plot <- plotMediaShareLoop[, round(unique(nrmse), 4)]
      decomp_rssd_plot <- plotMediaShareLoop[, round(unique(decomp.rssd), 4)]
      mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input), plotMediaShareLoop[, round(unique(mape), 4)], NA)

      errors <- paste0(
        "R2 train: ", rsq_train_plot,
        ", NRMSE = ", nrmse_plot,
        ", DECOMP.RSSD = ", decomp_rssd_plot,
        ifelse(!is.na(mape_lift_plot), paste0(", MAPE = ", mape_lift_plot), "")
      )

      ## 1. Spend x effect share comparison
      plotMediaShareLoopBar <- temp[[sid]]$plot1data$plotMediaShareLoopBar
      plotMediaShareLoopLine <- temp[[sid]]$plot1data$plotMediaShareLoopLine
      ySecScale <- temp[[sid]]$plot1data$ySecScale
      plotMediaShareLoopBar$variable <- stringr::str_to_title(gsub("_", " ", plotMediaShareLoopBar$variable))
      plotMediaShareLoopLine$variable <- "Total ROI"
      p1 <- ggplot(plotMediaShareLoopBar, aes(x = .data$rn, y = .data$value, fill = .data$variable)) +
        geom_bar(stat = "identity", width = 0.5, position = "dodge") +
        geom_text(aes(y = 0, label = paste0(round(.data$value * 100, 1), "%")),
          hjust = -.1, color = "darkblue", position = position_dodge(width = 0.5), fontface = "bold"
        ) +
        geom_line(
          data = plotMediaShareLoopLine, aes(
            x = .data$rn, y = .data$value / ySecScale, group = 1
          ),
          inherit.aes = FALSE, color = "#03396C"
        ) +
        geom_point(
          data = plotMediaShareLoopLine, aes(
            x = .data$rn, y = .data$value / ySecScale, group = 1, color = "Total ROI"
          ),
          inherit.aes = FALSE, size = 4
        ) +
        geom_text(
          data = plotMediaShareLoopLine, aes(
            label = round(.data$value, 2), x = .data$rn, y = .data$value / ySecScale, group = 1
          ),
          fontface = "bold", inherit.aes = FALSE, hjust = -.5, size = 5
        ) +
        scale_y_percent() +
        coord_flip() +
        theme_lares(axis.text.x = element_blank(), legend = "top", grid = "Xx") +
        scale_fill_brewer(palette = 3) +
        scale_color_manual(values = c("Total ROI" = "#03396C")) +
        labs(
          title = paste0("Share of Spend VS Share of Effect with total ", ifelse(InputCollect$dep_var_type == "conversion", "CPA", "ROI")),
          y = "Total Share by Channel", x = NULL, fill = NULL, color = NULL
        )

      ## 2. Waterfall
      plotWaterfallLoop <- temp[[sid]]$plot2data$plotWaterfallLoop
      plotWaterfallLoop$sign <- ifelse(plotWaterfallLoop$sign == "pos", "Positive", "Negative")
      p2 <- suppressWarnings(
        ggplot(plotWaterfallLoop, aes(x = id, fill = sign)) +
          geom_rect(aes(
            x = rn, xmin = id - 0.45, xmax = id + 0.45,
            ymin = end, ymax = start
          ), stat = "identity") +
          scale_x_discrete("", breaks = levels(plotWaterfallLoop$rn), labels = plotWaterfallLoop$rn) +
          scale_y_percent() +
          theme_lares(legend = "top") +
          geom_text(mapping = aes(
            label = paste0(formatNum(xDecompAgg, abbr = TRUE), "\n", round(xDecompPerc * 100, 1), "%"),
            y = rowSums(cbind(plotWaterfallLoop$end, plotWaterfallLoop$xDecompPerc / 2))
          ), fontface = "bold", lineheight = .7) +
          coord_flip() +
          labs(
            title = "Response Decomposition Waterfall by Predictor",
            x = NULL, y = NULL, fill = "Sign"
          )
      )

      ## 3. Adstock rate
      if (InputCollect$adstock == "geometric") {
        dt_geometric <- temp[[sid]]$plot3data$dt_geometric
        p3 <- ggplot(dt_geometric, aes(x = .data$channels, y = .data$thetas, fill = "coral")) +
          geom_bar(stat = "identity", width = 0.5) +
          theme_lares(legend = "none", grid = "Xx") +
          coord_flip() +
          geom_text(aes(label = formatNum(100 * thetas, 1, pos = "%")),
            hjust = -.1, position = position_dodge(width = 0.5), fontface = "bold"
          ) +
          scale_y_percent(limit = c(0, 1)) +
          labs(
            title = "Geometric Adstock: Fixed Decay Rate Over Time",
            y = NULL, x = NULL
          )
      }
      if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
        weibullCollect <- temp[[sid]]$plot3data$weibullCollect
        wb_type <- temp[[sid]]$plot3data$wb_type
        p3 <- ggplot(weibullCollect, aes(x = .data$x, y = .data$decay_accumulated)) +
          geom_line(aes(color = .data$channel)) +
          facet_wrap(~ .data$channel) +
          geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
          geom_text(aes(x = max(.data$x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife"), colour = "gray") +
          theme_lares(legend = "none", grid = "Xx") +
          labs(
            title = paste0("Weibull Adstock ", wb_type, ": Flexible Decay Rate Over Time"),
            x = "Time Unit", y = NULL
          )
      }

      ## 4. Response curves
      dt_scurvePlot <- temp[[sid]]$plot4data$dt_scurvePlot
      dt_scurvePlotMean <- temp[[sid]]$plot4data$dt_scurvePlotMean
      if (!"channel" %in% colnames(dt_scurvePlotMean)) dt_scurvePlotMean$channel <- dt_scurvePlotMean$rn
      p4 <- ggplot(
        dt_scurvePlot[dt_scurvePlot$channel %in% InputCollect$paid_media_spends, ],
        aes(x = .data$spend, y = .data$response, color = .data$channel)
      ) +
        geom_line() +
        geom_point(data = dt_scurvePlotMean, aes(
          x = .data$mean_spend, y = .data$mean_response, color = .data$channel
        )) +
        geom_text(
          data = dt_scurvePlotMean, aes(
            x = .data$mean_spend, y = .data$mean_response, color = .data$channel,
            label = formatNum(.data$mean_spend, 2, abbr = TRUE)
          ),
          show.legend = FALSE, hjust = -0.2
        ) +
        theme_lares(pal = 2) +
        theme(
          legend.position = c(0.9, 0.2),
          legend.background = element_rect(fill = alpha("grey98", 0.6), color = "grey90")
        ) +
        labs(
          title = "Response Curves and Mean Spends by Channel",
          x = "Spend", y = "Response", color = NULL
        ) +
        scale_x_abbr() +
        scale_y_abbr()

      ## 5. Fitted vs actual
      xDecompVecPlotMelted <- temp[[sid]]$plot5data$xDecompVecPlotMelted
      xDecompVecPlotMelted$variable <- stringr::str_to_title(xDecompVecPlotMelted$variable)
      xDecompVecPlotMelted$linetype <- ifelse(xDecompVecPlotMelted$variable == "Predicted", "solid", "dotted")
      p5 <- ggplot(xDecompVecPlotMelted, aes(x = .data$ds, y = .data$value, color = .data$variable)) +
        geom_path(aes(linetype = .data$linetype), size = 0.6) +
        theme_lares(legend = "top", pal = 2) +
        scale_y_abbr() +
        guides(linetype = "none") +
        labs(
          title = "Actual vs. Predicted Response",
          x = "Date", y = "Response", color = NULL
        )

      ## 6. Diagnostic: fitted vs residual
      xDecompVecPlot <- temp[[sid]]$plot6data$xDecompVecPlot
      p6 <- qplot(x = .data$predicted, y = .data$actual - .data$predicted, data = xDecompVecPlot) +
        geom_hline(yintercept = 0) +
        geom_smooth(se = TRUE, method = "loess", formula = "y ~ x") +
        scale_x_abbr() + scale_y_abbr() +
        theme_lares() +
        labs(x = "Fitted", y = "Residual", title = "Fitted vs. Residual")

      ## Aggregate one-pager plots and export
      onepagerTitle <- paste0("Model One-pager, on Pareto Front ", pf, ", ID: ", sid)
      pg <- wrap_plots(p2, p5, p1, p4, p3, p6, ncol = 2) +
        plot_annotation(title = onepagerTitle, subtitle = errors, theme = theme_lares(background = "white"))
      all_plots[[sid]] <- pg

      if (export) {
        ggsave(
          filename = paste0(OutputCollect$plot_folder, "/", sid, ".png"),
          plot = pg, limitsize = FALSE,
          dpi = 400, width = 18, height = 18
        )
      }
      if (check_parallel_plot() & !quiet & count_mod_out > 0) {
        cnt <- cnt + 1
        setTxtProgressBar(pbplot, cnt)
      }
      return(all_plots)
    }
    if (!quiet & count_mod_out > 0) {
      cnt <- cnt + length(uniqueSol)
      setTxtProgressBar(pbplot, cnt)
    }
  }
  if (!quiet & count_mod_out > 0) close(pbplot)
  # Stop cluster to avoid memory leaks
  if (check_parallel_plot()) stopImplicitCluster()
  return(invisible(parallelResult[[1]]))
}

allocation_plots <- function(InputCollect, OutputCollect, dt_optimOut, select_model, scenario, export = TRUE, quiet = FALSE) {

  outputs <- list()

  subtitle <- paste0(
    "Total spend increase: ", dt_optimOut[
      , round(mean(optmSpendUnitTotalDelta) * 100, 1)
    ], "%",
    "\nTotal response increase: ", dt_optimOut[
      , round(mean(optmResponseUnitTotalLift) * 100, 1)
    ], "% with optimised spend allocation"
  )

  # Calculate errors for subtitles
  plotDT_scurveMeanResponse <- OutputCollect$xDecompAgg[
    solID == select_model & rn %in% InputCollect$paid_media_spends]
  errors <- paste0(
    "R2 train: ", plotDT_scurveMeanResponse[, round(mean(rsq_train), 4)],
    ", NRMSE = ", plotDT_scurveMeanResponse[, round(mean(nrmse), 4)],
    ", DECOMP.RSSD = ", plotDT_scurveMeanResponse[, round(mean(decomp.rssd), 4)],
    ", MAPE = ", plotDT_scurveMeanResponse[, round(mean(mape), 4)]
  )

  # 1. Response comparison plot
  plotDT_resp <- select(dt_optimOut, .data$channels, .data$initResponseUnit, .data$optmResponseUnit) %>%
    mutate(channels = as.factor(.data$channels))
  names(plotDT_resp) <- c("channel", "Initial Avg. Spend Share", "Optimised Avg. Spend Share")
  plotDT_resp <- suppressWarnings(melt.data.table(plotDT_resp, id.vars = "channel", value.name = "response"))
  outputs[["p12"]] <- p12 <- ggplot(plotDT_resp, aes(
    y = reorder(.data$channel, -as.integer(.data$channel)),
    x = .data$response, fill = reorder(.data$variable, as.numeric(.data$variable)))) +
    geom_bar(stat = "identity", width = 0.5, position = position_dodge2(reverse = TRUE, padding = 0)) +
    scale_fill_brewer(palette = 3) +
    geom_text(aes(x = 0, label = formatNum(.data$response, 0), hjust = -0.1),
      position = position_dodge2(width = 0.5, reverse = TRUE), fontface = "bold", show.legend = FALSE
    ) +
    theme_lares(legend = "top") +
    scale_x_abbr() +
    labs(
      title = "Initial vs. Optimised Mean Response",
      subtitle = subtitle,
      fill = NULL, x = "Mean Response [#]", y = NULL
    )

  # 2. Budget share comparison plot
  plotDT_share <- select(dt_optimOut, .data$channels, .data$initSpendShare, .data$optmSpendShareUnit) %>%
    mutate(channels = as.factor(.data$channels))
  names(plotDT_share) <- c("channel", "Initial Avg. Spend Share", "Optimised Avg. Spend Share")
  plotDT_share <- suppressWarnings(melt.data.table(plotDT_share, id.vars = "channel", value.name = "spend_share"))
  outputs[["p13"]] <- p13 <- ggplot(plotDT_share, aes(
    y = reorder(.data$channel, -as.integer(.data$channel)),
    x = .data$spend_share, fill = .data$variable)) +
    geom_bar(stat = "identity", width = 0.5, position = position_dodge2(reverse = TRUE, padding = 0)) +
    scale_fill_brewer(palette = 3) +
    geom_text(aes(x = 0, label = formatNum(.data$spend_share * 100, 1, pos = "%"), hjust = -0.1),
      position = position_dodge2(width = 0.5, reverse = TRUE), fontface = "bold", show.legend = FALSE
    ) +
    theme_lares(legend = "top") +
    scale_x_percent() +
    labs(
      title = "Initial vs. Optimised Budget Allocation",
      subtitle = subtitle,
      fill = NULL, x = "Budget Allocation [%]", y = NULL
    )

  ## 3. Response curves
  plotDT_saturation <- melt.data.table(OutputCollect$mediaVecCollect[
    solID == select_model & type == "saturatedSpendReversed"
  ], id.vars = "ds", measure.vars = InputCollect$paid_media_spends, value.name = "spend", variable.name = "channel")
  plotDT_decomp <- melt.data.table(OutputCollect$mediaVecCollect[
    solID == select_model & type == "decompMedia"
  ], id.vars = "ds", measure.vars = InputCollect$paid_media_spends, value.name = "response", variable.name = "channel")
  plotDT_scurve <- data.frame(plotDT_saturation, response = plotDT_decomp$response) %>%
    filter(.data$spend >= 0) %>% as_tibble()

  dt_optimOutScurve <- rbind(
    select(dt_optimOut, .data$channels, .data$initSpendUnit, .data$initResponseUnit) %>% mutate(type = "Initial"),
    select(dt_optimOut, .data$channels, .data$optmSpendUnit, .data$optmResponseUnit) %>% mutate(type = "Optimised"),
    use.names = FALSE
  )
  colnames(dt_optimOutScurve) <- c("channels", "spend", "response", "type")
  dt_optimOutScurve <- dt_optimOutScurve %>%
    group_by(.data$channels) %>%
    mutate(spend_dif = dplyr::last(.data$spend) - dplyr::first(.data$spend),
           response_dif = dplyr::last(.data$response) - dplyr::first(.data$response))

  outputs[["p14"]] <- p14 <- ggplot(data = plotDT_scurve, aes(
    x = .data$spend, y = .data$response, color = .data$channel)) +
    geom_line() +
    geom_point(data = dt_optimOutScurve, aes(
      x = .data$spend, y = .data$response,
      color = .data$channels, shape = .data$type
    ), size = 2.5) +
    # geom_text(
    #   data = dt_optimOutScurve, aes(
    #     x = .data$spend, y = .data$response, color = .data$channels,
    #     hjust = .data$hjust,
    #     label = formatNum(.data$spend, 2, abbr = TRUE)
    #   ),
    #   show.legend = FALSE
    # ) +
    theme_lares(legend.position = c(0.9, 0), pal = 2) +
    theme(
      legend.position = c(0.87, 0.5),
      legend.background = element_rect(fill = alpha("grey98", 0.6), color = "grey90"),
      legend.spacing.y = unit(0.2, 'cm')
    ) +
    labs(
      title = "Response Curve and Mean* Spend by Channel",
      x = "Spend", y = "Response", shape = NULL, color = NULL,
      caption = sprintf(
        "*Based on date range: %s to %s (%s)",
        dt_optimOut$date_min[1],
        dt_optimOut$date_max[1],
        dt_optimOut$periods[1]
      )
    ) +
    scale_x_abbr() +
    scale_y_abbr()

  # Gather all plots into a single one
  p13 <- p13 + labs(subtitle = NULL)
  p12 <- p12 + labs(subtitle = NULL)
  outputs[["plots"]] <- plots <- ((p13 + p12) / p14) + plot_annotation(
    title = paste0("Budget Allocator Optimum Result for Model ID ", select_model),
    subtitle = subtitle,
    theme = theme_lares(background = "white")
  )

  # Gather all plots
  if (export) {
    scenario <- ifelse(scenario == "max_historical_response", "hist", "respo")
    filename <- paste0(OutputCollect$plot_folder, select_model, "_reallocated_", scenario, ".png")
    if (!quiet) message("Exporting charts into file: ", filename)
    ggsave(
      filename = filename,
      plot = plots, limitsize = FALSE,
      dpi = 350, width = 15, height = 12
    )
  }

  return(invisible(outputs))
}
