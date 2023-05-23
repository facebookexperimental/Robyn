# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Generate and Export Robyn Plots
#'
#' @rdname robyn_outputs
#' @return Invisible list with \code{ggplot} plots.
#' @export
robyn_plots <- function(InputCollect, OutputCollect, export = TRUE, ...) {
  check_class("robyn_outputs", OutputCollect)
  pareto_fronts <- OutputCollect$pareto_fronts
  hyper_fixed <- OutputCollect$hyper_fixed
  temp_all <- OutputCollect$allPareto
  all_plots <- list()

  if (!hyper_fixed) {
    ## Prophet
    if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0 ||
      !is.null(InputCollect$factor_vars) && length(InputCollect$factor_vars) > 0) {
      dt_plotProphet <- InputCollect$dt_mod %>%
        select(c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$factor_vars)) %>%
        tidyr::gather("variable", "value", -.data$ds) %>%
        mutate(ds = as.Date(.data$ds, origin = "1970-01-01"))
      all_plots[["pProphet"]] <- pProphet <- ggplot(
        dt_plotProphet, aes(x = .data$ds, y = .data$value)
      ) +
        geom_line(color = "steelblue") +
        facet_wrap(~ .data$variable, scales = "free", ncol = 1) +
        labs(title = "Prophet decomposition", x = NULL, y = NULL) +
        theme_lares() +
        scale_y_abbr()

      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "prophet_decomp.png"),
          plot = pProphet, limitsize = FALSE,
          dpi = 600, width = 12, height = 3 * length(unique(dt_plotProphet$variable))
        )
      }
    }

    # ## Spend exposure model
    # if (any(InputCollect$exposure_selector)) {
    #   all_plots[["pSpendExposure"]] <- pSpendExposure <- wrap_plots(
    #     InputCollect$plotNLSCollect,
    #     ncol = ifelse(length(InputCollect$plotNLSCollect) <= 3, length(InputCollect$plotNLSCollect), 3)
    #   ) +
    #     plot_annotation(
    #       title = "Spend-exposure fitting with Michaelis-Menten model",
    #       theme = theme(plot.title = element_text(hjust = 0.5))
    #     )
    #   if (export) ggsave(
    #     paste0(OutputCollect$plot_folder, "spend_exposure_fitting.png"),
    #     plot = pSpendExposure, dpi = 600, width = 12, limitsize = FALSE,
    #     height = ceiling(length(InputCollect$plotNLSCollect) / 3) * 7
    #   )
    # } else {
    #  # message("No spend-exposure modelling needed. All media variables used for MMM are spend variables")
    # }

    ## Hyperparameter sampling distribution
    if (length(temp_all) > 0) {
      resultHypParam <- temp_all$resultHypParam
      hpnames_updated <- c(names(OutputCollect$OutputModels$hyper_updated))
      hpnames_updated <- str_replace(hpnames_updated, "lambda", "lambda_hp")
      resultHypParam.melted <- resultHypParam %>%
        dplyr::select(any_of(hpnames_updated)) %>%
        tidyr::gather("variable", "value") %>%
        mutate(variable = ifelse(.data$variable == "lambda_hp", "lambda", .data$variable)) %>%
        dplyr::rowwise() %>%
        mutate(type = str_split(.data$variable, "_")[[1]][str_count(.data$variable, "_")[[1]] + 1]) %>%
        mutate(channel = gsub(pattern = paste0("_", .data$type), "", .data$variable)) %>%
        ungroup() %>%
        mutate(
          type = factor(.data$type, levels = unique(.data$type)),
          channel = factor(.data$channel, levels = rev(unique(.data$channel)))
        )
      all_plots[["pSamp"]] <- ggplot(resultHypParam.melted) +
        facet_grid(. ~ .data$type, scales = "free") +
        geom_violin(
          aes(x = .data$value, y = .data$channel, color = .data$channel, fill = .data$channel),
          alpha = .8, size = 0
        ) +
        theme_lares(legend = "none", pal = 1) +
        labs(
          title = "Hyperparameters Optimization Distributions",
          subtitle = paste0(
            "Sample distribution", ", iterations = ",
            OutputCollect$iterations, " x ", OutputCollect$trials, " trial"
          ),
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
      resultHypParam <- temp_all$resultHypParam
      if (!is.null(InputCollect$calibration_input)) {
        resultHypParam <- resultHypParam %>%
          mutate(iterations = ifelse(is.na(.data$robynPareto), NA, .data$iterations))
        # Show blue dots on top of grey dots
        resultHypParam <- resultHypParam[order(!is.na(resultHypParam$robynPareto)), ]
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
        temp <- resultHypParam[resultHypParam$robynPareto %in% pfs, ]
        if (nrow(temp) > 1) {
          pParFront <- pParFront + geom_line(
            data = temp,
            aes(x = .data$nrmse, y = .data$decomp.rssd),
            colour = pf_color
          )
        }
      }
      all_plots[["pParFront"]] <- pParFront
      if (export) {
        ggsave(
          paste0(OutputCollect$plot_folder, "pareto_front.png"),
          plot = pParFront, limitsize = FALSE,
          dpi = 600, width = 12, height = 8
        )
      }
    }

    ## Ridgeline model convergence
    if (length(temp_all) > 0) {
      xDecompAgg <- temp_all$xDecompAgg
      dt_ridges <- xDecompAgg %>%
        filter(.data$rn %in% InputCollect$paid_media_spends) %>%
        mutate(iteration = (.data$iterNG - 1) * OutputCollect$cores + .data$iterPar) %>%
        select(variables = .data$rn, .data$roi_total, .data$iteration, .data$trial) %>%
        arrange(.data$iteration, .data$variables)
      bin_limits <- c(1, 20)
      qt_len <- ifelse(OutputCollect$iterations <= 100, 1,
        ifelse(OutputCollect$iterations > 2000, 20, ceiling(OutputCollect$iterations / 100))
      )
      set_qt <- floor(quantile(1:OutputCollect$iterations, seq(0, 1, length.out = qt_len + 1)))
      set_bin <- set_qt[-1]
      dt_ridges <- dt_ridges %>%
        mutate(iter_bin = cut(.data$iteration, breaks = set_qt, labels = set_bin)) %>%
        filter(!is.na(.data$iter_bin)) %>%
        mutate(
          iter_bin = factor(.data$iter_bin, levels = sort(set_bin, decreasing = TRUE)),
          trial = as.factor(.data$trial)
        )
      plot_vars <- unique(dt_ridges$variables)
      plot_n <- ceiling(length(plot_vars) / 6)
      metric <- ifelse(InputCollect$dep_var_type == "revenue", "ROAS", "CPA")
      for (pl in 1:plot_n) {
        loop_vars <- na.omit(plot_vars[(1:6) + 6 * (pl - 1)])
        dt_ridges_loop <- dt_ridges[dt_ridges$variables %in% loop_vars, ]
        all_plots[[paste0("pRidges", pl)]] <- pRidges <- ggplot(
          dt_ridges_loop, aes(
            x = .data$roi_total, y = .data$iter_bin,
            fill = as.integer(.data$iter_bin),
            linetype = .data$trial
          )
        ) +
          scale_fill_distiller(palette = "GnBu") +
          geom_density_ridges(scale = 4, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7) +
          facet_wrap(~ .data$variables, scales = "free") +
          guides(fill = "none", linetype = "none") +
          theme_lares() +
          labs(
            x = paste(metric, "by Channel"), y = NULL,
            title = paste(metric, "Distribution over Iteration Buckets")
          )
        if (export) {
          suppressMessages(ggsave(
            paste0(OutputCollect$plot_folder, metric, "_convergence", pl, ".png"),
            plot = pRidges, dpi = 600, width = 12, limitsize = FALSE,
            height = ceiling(length(loop_vars) / 3) * 6
          ))
        }
      }
    }
  } # End of !hyper_fixed

  if (isTRUE(OutputCollect$OutputModels$ts_validation)) {
    ts_validation_plot <- ts_validation(OutputCollect$OutputModels, quiet = TRUE, ...)
    ggsave(
      paste0(OutputCollect$plot_folder, "ts_validation", ".png"),
      plot = ts_validation_plot, dpi = 300,
      width = 10, height = 12, limitsize = FALSE
    )
  }

  return(invisible(all_plots))
}


####################################################################
#' Generate and Export Robyn One-Pager Plots
#'
#' @rdname robyn_outputs
#' @return Invisible list with \code{patchwork} plot(s).
#' @export
robyn_onepagers <- function(InputCollect, OutputCollect, select_model = NULL, quiet = FALSE, export = TRUE) {
  check_class("robyn_outputs", OutputCollect)
  if (TRUE) {
    pareto_fronts <- OutputCollect$pareto_fronts
    hyper_fixed <- OutputCollect$hyper_fixed
    resultHypParam <- as_tibble(OutputCollect$resultHypParam)
    xDecompAgg <- as_tibble(OutputCollect$xDecompAgg)
    val <- isTRUE(OutputCollect$OutputModels$ts_validation)
    sid <- NULL # for parallel loops
  }
  if (!is.null(select_model)) {
    if ("clusters" %in% select_model) select_model <- OutputCollect$clusters$models$solID
    resultHypParam <- resultHypParam[resultHypParam$solID %in% select_model, ]
    xDecompAgg <- xDecompAgg[xDecompAgg$solID %in% select_model, ]
    if (!quiet & nrow(resultHypParam) > 1) {
      message(">> Generating only cluster results one-pagers (", nrow(resultHypParam), ")...")
    }
  }

  # Prepare for parallel plotting
  if (check_parallel_plot() && OutputCollect$cores > 1) registerDoParallel(OutputCollect$cores) else registerDoSEQ()
  if (!hyper_fixed) {
    pareto_fronts_vec <- 1:pareto_fronts
    count_mod_out <- nrow(resultHypParam[resultHypParam$robynPareto %in% pareto_fronts_vec, ])
  } else {
    pareto_fronts_vec <- 1
    count_mod_out <- nrow(resultHypParam)
  }
  all_fronts <- unique(xDecompAgg$robynPareto)
  all_fronts <- sort(all_fronts[!is.na(all_fronts)])
  if (!all(pareto_fronts_vec %in% all_fronts)) pareto_fronts_vec <- all_fronts

  if (check_parallel_plot()) {
    if (!quiet & nrow(resultHypParam) > 1) {
      message(paste(">> Plotting", count_mod_out, "selected models on", OutputCollect$cores, "cores..."))
    }
  } else {
    if (!quiet & nrow(resultHypParam) > 1) {
      message(paste(">> Plotting", count_mod_out, "selected models on 1 core (MacOS fallback)..."))
    }
  }

  if (!quiet && count_mod_out > 1) {
    pbplot <- txtProgressBar(min = 0, max = count_mod_out, style = 3)
  }
  temp <- OutputCollect$allPareto$plotDataCollect
  all_plots <- list()
  cnt <- 0

  for (pf in pareto_fronts_vec) { # pf = pareto_fronts_vec[1]

    plotMediaShare <- filter(
      xDecompAgg, .data$robynPareto == pf,
      .data$rn %in% InputCollect$paid_media_spends
    )
    uniqueSol <- unique(plotMediaShare$solID)

    # parallelResult <- for (sid in uniqueSol) { # sid = uniqueSol[1]
    parallelResult <- foreach(sid = uniqueSol) %dorng% { # sid = uniqueSol[1]

      if (TRUE) {
        plotMediaShareLoop <- plotMediaShare[plotMediaShare$solID == sid, ]
        rsq_train_plot <- round(plotMediaShareLoop$rsq_train[1], 4)
        rsq_val_plot <- round(plotMediaShareLoop$rsq_val[1], 4)
        rsq_test_plot <- round(plotMediaShareLoop$rsq_test[1], 4)
        nrmse_train_plot <- round(plotMediaShareLoop$nrmse_train[1], 4)
        nrmse_val_plot <- round(plotMediaShareLoop$nrmse_val[1], 4)
        nrmse_test_plot <- round(plotMediaShareLoop$nrmse_test[1], 4)
        decomp_rssd_plot <- round(plotMediaShareLoop$decomp.rssd[1], 4)
        mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input),
          round(plotMediaShareLoop$mape[1], 4), NA
        )
        train_size <- round(plotMediaShareLoop$train_size[1], 4)
        if (val) {
          errors <- sprintf(
            paste(
              "Adj.R2: train = %s, val = %s, test = %s |",
              "NRMSE: train = %s, val = %s, test = %s |",
              "DECOMP.RSSD = %s | MAPE = %s"
            ),
            rsq_train_plot, rsq_val_plot, rsq_test_plot,
            nrmse_train_plot, nrmse_val_plot, nrmse_test_plot,
            decomp_rssd_plot, mape_lift_plot
          )
        } else {
          errors <- sprintf(
            "Adj.R2: train = %s | NRMSE: train = %s | DECOMP.RSSD = %s | MAPE = %s",
            rsq_train_plot, nrmse_train_plot, decomp_rssd_plot, mape_lift_plot
          )
        }
      }

      ## 1. Spend x effect share comparison
      plotMediaShareLoopBar <- temp[[sid]]$plot1data$plotMediaShareLoopBar
      plotMediaShareLoopLine <- temp[[sid]]$plot1data$plotMediaShareLoopLine
      ySecScale <- temp[[sid]]$plot1data$ySecScale
      plotMediaShareLoopBar$variable <- stringr::str_to_title(gsub("_", " ", plotMediaShareLoopBar$variable))
      type <- ifelse(InputCollect$dep_var_type == "conversion", "CPA", "ROI")
      plotMediaShareLoopLine$type_colour <- type_colour <- "#03396C"
      names(type_colour) <- "type_colour"
      p1 <- ggplot(plotMediaShareLoopBar, aes(x = .data$rn, y = .data$value, fill = .data$variable)) +
        geom_bar(stat = "identity", width = 0.5, position = "dodge") +
        geom_text(aes(y = 0, label = paste0(round(.data$value * 100, 1), "%")),
          hjust = -.1, position = position_dodge(width = 0.5), fontface = "bold"
        ) +
        geom_line(
          data = plotMediaShareLoopLine, aes(
            x = .data$rn, y = .data$value / ySecScale, group = 1
          ),
          color = type_colour, inherit.aes = FALSE
        ) +
        geom_point(
          data = plotMediaShareLoopLine, aes(
            x = .data$rn, y = .data$value / ySecScale, group = 1, color = type_colour
          ),
          inherit.aes = FALSE, size = 3.5
        ) +
        geom_text(
          data = plotMediaShareLoopLine, aes(
            label = round(.data$value, 2), x = .data$rn, y = .data$value / ySecScale, group = 1
          ),
          color = type_colour, fontface = "bold", inherit.aes = FALSE, hjust = -.4, size = 4
        ) +
        scale_y_percent() +
        coord_flip() +
        theme_lares(axis.text.x = element_blank(), legend = "top", grid = "Xx") +
        scale_fill_brewer(palette = 3) +
        scale_color_identity(guide = "legend", labels = type) +
        labs(
          title = paste0("Total Spend% VS Effect% with total ", type),
          y = "Total Share by Channel", x = NULL, fill = NULL, color = NULL
        )

      ## 2. Waterfall
      plotWaterfallLoop <- temp[[sid]]$plot2data$plotWaterfallLoop
      p2 <- suppressWarnings(
        ggplot(plotWaterfallLoop, aes(x = .data$id, fill = .data$sign)) +
          geom_rect(aes(
            x = .data$rn, xmin = .data$id - 0.45, xmax = .data$id + 0.45,
            ymin = .data$end, ymax = .data$start
          ), stat = "identity") +
          scale_x_discrete("", breaks = levels(plotWaterfallLoop$rn), labels = plotWaterfallLoop$rn) +
          scale_y_percent() +
          scale_fill_manual(values = c("Positive" = "#59B3D2", "Negative" = "#E5586E")) +
          theme_lares(legend = "top") +
          geom_text(mapping = aes(
            label = paste0(
              formatNum(.data$xDecompAgg, abbr = TRUE),
              "\n", round(.data$xDecompPerc * 100, 1), "%"
            ),
            y = rowSums(cbind(.data$end, .data$xDecompPerc / 2))
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
          geom_text(aes(label = formatNum(100 * .data$thetas, 1, pos = "%")),
            hjust = -.1, position = position_dodge(width = 0.5), fontface = "bold"
          ) +
          scale_y_percent(limit = c(0, 1)) +
          labs(
            title = "Geometric Adstock: Fixed Rate Over Time",
            y = sprintf("Thetas [by %s]", InputCollect$intervalType), x = NULL
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
            title = paste("Weibull", wb_type, "Adstock: Flexible Rate Over Time"),
            x = sprintf("Time unit [%ss]", InputCollect$intervalType), y = NULL
          )
      }

      ## 4. Response curves
      dt_scurvePlot <- temp[[sid]]$plot4data$dt_scurvePlot
      dt_scurvePlotMean <- temp[[sid]]$plot4data$dt_scurvePlotMean
      trim_rate <- 1.3 # maybe enable as a parameter
      if (trim_rate > 0) {
        dt_scurvePlot <- dt_scurvePlot %>%
          filter(
            .data$spend < max(dt_scurvePlotMean$mean_spend_adstocked) * trim_rate,
            .data$response < max(dt_scurvePlotMean$mean_response) * trim_rate,
            .data$channel %in% InputCollect$paid_media_spends
          ) %>%
          left_join(
            dt_scurvePlotMean[, c("channel", "mean_carryover")], "channel"
          )
      }
      if (!"channel" %in% colnames(dt_scurvePlotMean)) {
        dt_scurvePlotMean$channel <- dt_scurvePlotMean$rn
      }
      p4 <- ggplot(
        dt_scurvePlot, aes(x = .data$spend, y = .data$response, color = .data$channel)
      ) +
        geom_line() +
        geom_area(
          data = group_by(dt_scurvePlot, .data$channel) %>% filter(.data$spend <= .data$mean_carryover),
          aes(x = .data$spend, y = .data$response, color = .data$channel),
          stat = "identity", position = "stack", size = 0.1,
          fill = "grey50", alpha = 0.4, show.legend = FALSE
        ) +
        geom_point(data = dt_scurvePlotMean, aes(
          x = .data$mean_spend_adstocked, y = .data$mean_response, color = .data$channel
        )) +
        geom_text(
          data = dt_scurvePlotMean, aes(
            x = .data$mean_spend_adstocked, y = .data$mean_response, color = .data$channel,
            label = formatNum(.data$mean_spend_adstocked, 2, abbr = TRUE)
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
          x = "Spend (carryover + immediate)", y = "Response", color = NULL
        ) +
        scale_y_abbr() +
        scale_x_abbr()

      ## 5. Fitted vs actual
      xDecompVecPlotMelted <- temp[[sid]]$plot5data$xDecompVecPlotMelted %>%
        mutate(
          linetype = ifelse(.data$variable == "predicted", "solid", "dotted"),
          variable = stringr::str_to_title(.data$variable),
          ds = as.Date(.data$ds, origin = "1970-01-01")
        )
      p5 <- ggplot(
        xDecompVecPlotMelted,
        aes(x = .data$ds, y = .data$value, color = .data$variable)
      ) +
        geom_path(aes(linetype = .data$linetype), size = 0.6) +
        theme_lares(legend = "top", pal = 2) +
        scale_y_abbr() +
        guides(linetype = "none") +
        labs(
          title = "Actual vs. Predicted Response",
          x = "Date", y = "Response", color = NULL
        )
      if (val) {
        days <- sort(unique(p5$data$ds))
        ndays <- length(days)
        train_cut <- round(ndays * train_size)
        val_cut <- train_cut + round(ndays * (1 - train_size) / 2)
        p5 <- p5 +
          # Train
          geom_vline(xintercept = p5$data$ds[train_cut], colour = "#39638b", alpha = 0.8) +
          geom_text(
            x = p5$data$ds[train_cut], y = Inf, hjust = 0, vjust = 1.2,
            angle = 270, colour = "#39638b", alpha = 0.5, size = 3.2,
            label = sprintf("Train: %s", formatNum(100 * train_size, 1, pos = "%"))
          ) +
          # Validation
          geom_vline(xintercept = p5$data$ds[val_cut], colour = "#39638b", alpha = 0.8) +
          geom_text(
            x = p5$data$ds[val_cut], y = Inf, hjust = 0, vjust = 1.2,
            angle = 270, colour = "#39638b", alpha = 0.5, size = 3.2,
            label = sprintf("Validation: %s", formatNum(100 * (1 - train_size) / 2, 1, pos = "%"))
          ) +
          # Test
          geom_vline(xintercept = p5$data$ds[ndays], colour = "#39638b", alpha = 0.8) +
          geom_text(
            x = p5$data$ds[ndays], y = Inf, hjust = 0, vjust = 1.2,
            angle = 270, colour = "#39638b", alpha = 0.5, size = 3.2,
            label = sprintf("Test: %s", formatNum(100 * (1 - train_size) / 2, 1, pos = "%"))
          )
      }

      ## 6. Diagnostic: fitted vs residual
      xDecompVecPlot <- temp[[sid]]$plot6data$xDecompVecPlot
      p6 <- qplot(x = .data$predicted, y = .data$actual - .data$predicted, data = xDecompVecPlot) +
        geom_hline(yintercept = 0) +
        geom_smooth(se = TRUE, method = "loess", formula = "y ~ x") +
        scale_x_abbr() + scale_y_abbr() +
        theme_lares() +
        labs(x = "Fitted", y = "Residual", title = "Fitted vs. Residual")

      ## 7. Immediate vs carryover
      df_imme_caov <- temp[[sid]]$plot7data
      p7 <- df_imme_caov %>%
        mutate(type = factor(.data$type, levels = c("Carryover", "Immediate"))) %>%
        ggplot(aes(
          x = .data$percentage, y = .data$rn, fill = reorder(.data$type, as.integer(.data$type)),
          label = paste0(round(.data$percentage * 100), "%")
        )) +
        geom_bar(stat = "identity", width = 0.5) +
        geom_text(position = position_stack(vjust = 0.5)) +
        scale_fill_manual(values = c("Immediate" = "#59B3D2", "Carryover" = "coral")) +
        scale_x_percent() +
        theme_lares(legend = "top", grid = "Xx") +
        labs(
          x = "% Response", y = NULL, fill = NULL,
          title = "Immediate vs. Carryover Response Percentage"
        )

      ## 8. Bootstrapped ROI/CPA with CIs
      if ("ci_low" %in% colnames(xDecompAgg)) {
        metric <- ifelse(InputCollect$dep_var_type == "conversion", "CPA", "ROI")
        p8 <- xDecompAgg %>%
          filter(!is.na(.data$ci_low), .data$solID == sid) %>%
          select(.data$rn, .data$solID, .data$boot_mean, .data$ci_low, .data$ci_up) %>%
          ggplot(aes(x = .data$rn, y = .data$boot_mean)) +
          geom_point(size = 3) +
          geom_text(aes(label = signif(.data$boot_mean, 2)), vjust = -0.7, size = 3.3) +
          geom_text(aes(y = .data$ci_low, label = signif(.data$ci_low, 2)), hjust = 1.1, size = 2.8) +
          geom_text(aes(y = .data$ci_up, label = signif(.data$ci_up, 2)), hjust = -0.1, size = 2.8) +
          geom_errorbar(aes(ymin = .data$ci_low, ymax = .data$ci_up), width = 0.25) +
          labs(title = paste("In-cluster bootstrapped", metric, "with 95% CI & mean"), x = NULL, y = NULL) +
          coord_flip() +
          theme_lares()
        if (metric == "ROI") {
          p8 <- p8 + geom_hline(yintercept = 1, alpha = 0.5, colour = "grey50", linetype = "dashed")
        }
      } else {
        p8 <- lares::noPlot("No bootstrap results")
      }

      ## Aggregate one-pager plots and export
      ver <- as.character(utils::packageVersion("Robyn"))
      rver <- utils::sessionInfo()$R.version
      onepagerTitle <- sprintf("One-pager for Model ID: %s", sid)
      onepagerCaption <- sprintf("Robyn v%s [R-%s.%s]", ver, rver$major, rver$minor)
      pg <- wrap_plots(p2, p5, p1, p8, p3, p7, p4, p6, ncol = 2) +
        plot_annotation(
          title = onepagerTitle, subtitle = errors,
          theme = theme_lares(background = "white"),
          caption = onepagerCaption
        )
      all_plots[[sid]] <- pg

      if (export) {
        ggsave(
          filename = paste0(OutputCollect$plot_folder, "/", sid, ".png"),
          plot = pg, limitsize = FALSE,
          dpi = 400, width = 17, height = 19
        )
      }
      if (check_parallel_plot() && !quiet && count_mod_out > 1) {
        cnt <- cnt + 1
        setTxtProgressBar(pbplot, cnt)
      }
      return(all_plots)
    }
    if (!quiet && count_mod_out > 1) {
      cnt <- cnt + length(uniqueSol)
      setTxtProgressBar(pbplot, cnt)
    }
  }
  if (!quiet && count_mod_out > 1) close(pbplot)
  # Stop cluster to avoid memory leaks
  if (check_parallel_plot()) stopImplicitCluster()
  return(invisible(parallelResult[[1]]))
}

allocation_plots <- function(InputCollect, OutputCollect, dt_optimOut, select_model,
                             scenario, eval_list, export = TRUE, quiet = FALSE) {
  outputs <- list()

  subtitle <- sprintf(
    paste0(
      "Total %sspend increase: %s%%",
      "\nTotal response increase: %s%% with optimised spend allocation"
    ),
    ifelse(isTRUE(dt_optimOut$adstocked[1]), paste0("(adstocked**) ", ""), ""),
    round(mean(dt_optimOut$optmSpendUnitTotalDelta) * 100, 1),
    round(mean(dt_optimOut$optmResponseUnitTotalLift) * 100, 1)
  )
  metric <- ifelse(InputCollect$dep_var_type == "revenue", "ROAS", "CPA")
  formulax1 <- ifelse(
    metric == "ROAS",
    "ROAS = total response / raw spend | mROAS = marginal response / marginal spend",
    "CPA = raw spend / total response | mCPA =  marginal spend / marginal response"
  )
  formulax2 <- ifelse(
    metric == "ROAS",
    "When reallocating budget, mROAS converges across media within respective bounds",
    "When reallocating budget, mCPA converges across media within respective bounds"
  )

  # Calculate errors for subtitles
  plotDT_scurveMeanResponse <- filter(
    OutputCollect$xDecompAgg,
    .data$solID == select_model,
    .data$rn %in% InputCollect$paid_media_spends
  )
  rsq_train_plot <- round(plotDT_scurveMeanResponse$rsq_train[1], 4)
  rsq_val_plot <- round(plotDT_scurveMeanResponse$rsq_val[1], 4)
  rsq_test_plot <- round(plotDT_scurveMeanResponse$rsq_test[1], 4)
  nrmse_train_plot <- round(plotDT_scurveMeanResponse$nrmse_train[1], 4)
  nrmse_val_plot <- round(plotDT_scurveMeanResponse$nrmse_val[1], 4)
  nrmse_test_plot <- round(plotDT_scurveMeanResponse$nrmse_test[1], 4)
  decomp_rssd_plot <- round(plotDT_scurveMeanResponse$decomp.rssd[1], 4)
  mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input),
    round(plotDT_scurveMeanResponse$mape[1], 4), NA
  )
  if (isTRUE(OutputCollect$OutputModels$ts_validation)) {
    errors <- sprintf(
      paste(
        "Adj.R2: train = %s, val = %s, test = %s |",
        "NRMSE: train = %s, val = %s, test = %s |",
        "DECOMP.RSSD = %s | MAPE = %s"
      ),
      rsq_train_plot, rsq_val_plot, rsq_test_plot,
      nrmse_train_plot, nrmse_val_plot, nrmse_test_plot,
      decomp_rssd_plot, mape_lift_plot
    )
  } else {
    errors <- sprintf(
      "Adj.R2: train = %s | NRMSE: train = %s | DECOMP.RSSD = %s | MAPE = %s",
      rsq_train_plot, nrmse_train_plot, decomp_rssd_plot, mape_lift_plot
    )
  }

  # 1. Response and spend comparison plot
  init_total_spend <- dt_optimOut$initSpendTotal[1]
  init_total_response <- dt_optimOut$initResponseTotal[1]
  init_total_roi <- init_total_response / init_total_spend
  init_total_cpa <- init_total_spend / init_total_response

  optm_total_spend_bounded <- dt_optimOut$optmSpendTotal[1]
  optm_total_response_bounded <- dt_optimOut$optmResponseTotal[1]
  optm_total_roi_bounded <- optm_total_response_bounded / optm_total_spend_bounded
  optm_total_cpa_bounded <- optm_total_spend_bounded / optm_total_response_bounded

  optm_total_spend_unbounded <- dt_optimOut$optmSpendTotalUnbound[1]
  optm_total_response_unbounded <- dt_optimOut$optmResponseTotalUnbound[1]
  optm_total_roi_unbounded <- optm_total_response_unbounded / optm_total_spend_unbounded
  optm_total_cpa_unbounded <- optm_total_spend_unbounded / optm_total_response_unbounded
  bound_mult <- dt_optimOut$unconstr_mult[1]

  optm_topped_unbounded <- optm_topped_bounded <- any_topped <- FALSE
  if (!is.null(eval_list$total_budget)) {
    optm_topped_bounded <- round(optm_total_spend_bounded) < round(eval_list$total_budget)
    optm_topped_unbounded <- round(optm_total_spend_unbounded) < round(eval_list$total_budget)
    any_topped <- optm_topped_bounded || optm_topped_unbounded
    if (optm_topped_bounded & !quiet) {
      message("NOTE: Given the upper/lower constrains, the total budget can't be fully allocated (^)")
    }
  }
  levs1 <- eval_list$levs1
  if (scenario == "max_response") {
    levs2 <- c(
      "Initial",
      paste0("Bounded", ifelse(optm_topped_bounded, "^", "")),
      paste0("Bounded", ifelse(optm_topped_unbounded, "^", ""), " x", bound_mult)
    )
  } else if (scenario == "target_efficiency") {
    levs2 <- levs1
  }

  resp_metric <- data.frame(
    type = factor(levs1, levels = levs1),
    type_lab = factor(levs2, levels = levs2),
    total_spend = c(init_total_spend, optm_total_spend_bounded, optm_total_spend_unbounded),
    total_response = c(init_total_response, optm_total_response_bounded, optm_total_response_unbounded),
    total_response_lift = c(
      0,
      dt_optimOut$optmResponseUnitTotalLift[1],
      dt_optimOut$optmResponseUnitTotalLiftUnbound[1]
    ),
    total_roi = c(init_total_roi, optm_total_roi_bounded, optm_total_roi_unbounded),
    total_cpa = c(init_total_cpa, optm_total_cpa_bounded, optm_total_cpa_unbounded)
  )
  df_roi <- resp_metric %>%
    mutate(spend = .data$total_spend, response = .data$total_response) %>%
    select(.data$type, .data$spend, .data$response) %>%
    pivot_longer(cols = !"type") %>%
    left_join(resp_metric, "type") %>%
    mutate(
      name = factor(paste("total", .data$name), levels = c("total spend", "total response")),
      name_label = factor(
        paste(.data$type, .data$name, sep = "\n"),
        levels = paste(.data$type, .data$name, sep = "\n")
      )
    ) %>%
    group_by(.data$name) %>%
    mutate(value_norm = .data$value / dplyr::first(.data$value))
  metric_vals <- if (metric == "ROAS") resp_metric$total_roi else resp_metric$total_cpa
  labs <- paste(
    paste(levs2, "\n"),
    paste("Spend:", formatNum(
      100 * (resp_metric$total_spend - resp_metric$total_spend[1]) / resp_metric$total_spend[1],
      signif = 3, pos = "%", sign = TRUE
    )),
    unique(paste("Resp:", formatNum(100 * df_roi$total_response_lift, signif = 3, pos = "%", sign = TRUE))),
    paste(metric, ":", round(metric_vals, 2)),
    sep = "\n"
  )
  df_roi$labs <- factor(rep(labs, each = 2), levels = labs)

  outputs[["p1"]] <- p1 <- df_roi %>%
    ggplot(aes(x = .data$name, y = .data$value_norm, fill = .data$type)) +
    facet_grid(. ~ .data$labs, scales = "free") +
    scale_fill_manual(values = c("grey", "steelblue", "darkgoldenrod4")) +
    geom_bar(stat = "identity", width = 0.6, alpha = 0.7) +
    geom_text(aes(label = formatNum(.data$value, signif = 3, abbr = TRUE)), color = "black", vjust = -.5) +
    theme_lares(legend = "none") +
    labs(title = "Total Budget Optimization Result", fill = NULL, y = NULL, x = NULL) +
    scale_y_continuous(limits = c(0, max(df_roi$value_norm * 1.2))) +
    theme(axis.text.y = element_blank())

  # 2. Response and spend comparison per channel plot
  df_plots <- dt_optimOut %>%
    mutate(
      channel = as.factor(.data$channels), Initial = .data$initResponseUnitShare,
      Bounded = .data$optmResponseUnitShare, Unbounded = .data$optmResponseUnitShareUnbound
    ) %>%
    select(.data$channel, .data$Initial, .data$Bounded, .data$Unbounded) %>%
    `colnames<-`(c("channel", levs1)) %>%
    tidyr::pivot_longer(names_to = "type", values_to = "response_share", -.data$channel) %>%
    left_join(
      dt_optimOut %>%
        mutate(
          channel = as.factor(.data$channels),
          Initial = .data$initSpendShare,
          Bounded = .data$optmSpendShareUnit,
          Unbounded = .data$optmSpendShareUnitUnbound
        ) %>%
        select(.data$channel, .data$Initial, .data$Bounded, .data$Unbounded) %>%
        `colnames<-`(c("channel", levs1)) %>%
        tidyr::pivot_longer(names_to = "type", values_to = "spend_share", -.data$channel),
      by = c("channel", "type")
    ) %>%
    left_join(
      dt_optimOut %>%
        mutate(
          channel = as.factor(.data$channels),
          Initial = case_when(
            metric == "ROAS" ~ .data$initRoiUnit,
            TRUE ~ .data$initCpaUnit
          ),
          Bounded = case_when(
            metric == "ROAS" ~ .data$optmRoiUnit,
            TRUE ~ .data$optmCpaUnit
          ),
          Unbounded = case_when(
            metric == "ROAS" ~ .data$optmRoiUnitUnbound,
            TRUE ~ .data$optmCpaUnitUnbound
          )
        ) %>%
        select(.data$channel, .data$Initial, .data$Bounded, .data$Unbounded) %>%
        `colnames<-`(c("channel", levs1)) %>%
        tidyr::pivot_longer(
          names_to = "type",
          values_to = ifelse(metric == "ROAS", "channel_roi", "channel_cpa"),
          -.data$channel
        ),
      by = c("channel", "type")
    ) %>%
    left_join(
      dt_optimOut %>%
        mutate(
          channel = as.factor(.data$channels),
          Initial = case_when(
            metric == "ROAS" ~ .data$initResponseMargUnit,
            TRUE ~ 1 / .data$initResponseMargUnit
          ),
          Bounded = case_when(
            metric == "ROAS" ~ .data$optmResponseMargUnit,
            TRUE ~ 1 / .data$optmResponseMargUnit
          ),
          Unbounded = case_when(
            metric == "ROAS" ~ .data$optmResponseMargUnitUnbound,
            TRUE ~ 1 / .data$optmResponseMargUnitUnbound
          )
        ) %>%
        select(.data$channel, .data$Initial, .data$Bounded, .data$Unbounded) %>%
        `colnames<-`(c("channel", levs1)) %>%
        tidyr::pivot_longer(
          names_to = "type",
          values_to = ifelse(metric == "ROAS", "marginal_roi", "marginal_cpa"),
          -.data$channel
        ),
      by = c("channel", "type")
    ) %>%
    left_join(resp_metric, by = "type")

  df_plot_share <- bind_rows(
    df_plots %>%
      select(c("channel", "type", "type_lab", "spend_share")) %>%
      mutate(metric = "spend") %>%
      rename(values = .data$spend_share),
    df_plots %>%
      select(c("channel", "type", "type_lab", "response_share")) %>%
      mutate(metric = "response") %>%
      rename(values = .data$response_share),
    df_plots %>%
      select(c("channel", "type", "type_lab", starts_with("channel_"))) %>%
      mutate(metric = metric) %>%
      rename(values = starts_with("channel_")),
    df_plots %>%
      select(c("channel", "type", "type_lab", starts_with("marginal_"))) %>%
      mutate(metric = paste0("m", metric)) %>%
      rename(values = starts_with("marginal_"))
  ) %>%
    mutate(
      type = factor(.data$type, levels = levs1),
      name_label = factor(
        paste(.data$type, .data$metric, sep = "\n"),
        levels = unique(paste(.data$type, .data$metric, sep = "\n"))
      ),
      # Deal with extreme cases divided by almost 0
      values = ifelse((.data$values > 1e15 | is.nan(.data$values)), 0, .data$values),
      values = round(.data$values, 4),
      values_label = case_when(
        # .data$metric %in% c("ROAS", "mROAS") ~ paste0("x", round(.data$values, 2)),
        .data$metric %in% c("CPA", "mCPA", "ROAS", "mROAS") ~ formatNum(.data$values, 2, abbr = TRUE),
        TRUE ~ paste0(round(100 * .data$values, 1), "%")
      ),
      # Better fill scale colours
      values_label = ifelse(grepl("NA|NaN", .data$values_label), "-", .data$values_label),
      values = ifelse((is.nan(.data$values) | is.na(.data$values)), 0, .data$values),
    ) %>%
    mutate(
      channel = factor(.data$channel, levels = rev(unique(.data$channel))),
      metric = factor(
        case_when(
          .data$metric %in% c("spend", "response") ~ paste0(.data$metric, "%"),
          TRUE ~ .data$metric
        ),
        levels = paste0(unique(.data$metric), c("%", "%", "", ""))
      )
    ) %>%
    group_by(.data$name_label) %>%
    mutate(
      values_norm = lares::normalize(.data$values),
      values_norm = ifelse(is.nan(.data$values_norm), 0, .data$values_norm)
    )

  outputs[["p2"]] <- p2 <- df_plot_share %>%
    ggplot(aes(x = .data$metric, y = .data$channel, fill = .data$type)) +
    geom_tile(aes(alpha = .data$values_norm), color = "white") +
    scale_fill_manual(values = c("grey50", "steelblue", "darkgoldenrod4")) +
    scale_alpha_continuous(range = c(0.6, 1)) +
    geom_text(aes(label = .data$values_label), colour = "black") +
    facet_grid(. ~ .data$type_lab, scales = "free") +
    theme_lares(legend = "none") +
    labs(
      title = "Budget Allocation per Channel*",
      fill = NULL, x = NULL, y = "Paid Channels"
    )

  ## 3. Response curves
  constr_labels <- dt_optimOut %>%
    mutate(constr_label = sprintf(
      "%s\n[%s - %s] & [%s - %s]", .data$channels, .data$constr_low,
      .data$constr_up, round(.data$constr_low_unb, 1), round(.data$constr_up_unb, 1)
    )) %>%
    select(
      "channel" = "channels", "constr_label", "constr_low_abs",
      "constr_up_abs", "constr_low_unb_abs", "constr_up_unb_abs"
    )
  plotDT_scurve <- eval_list[["plotDT_scurve"]] %>% left_join(constr_labels, "channel")
  mainPoints <- eval_list[["mainPoints"]] %>%
    left_join(constr_labels, "channel") %>%
    left_join(resp_metric, "type") %>%
    mutate(
      type = as.character(.data$type),
      type = factor(ifelse(is.na(.data$type), "Carryover", .data$type),
        levels = c("Carryover", levs1)
      )
    ) %>%
    mutate(
      type_lab = as.character(.data$type_lab),
      type_lab = factor(ifelse(is.na(.data$type_lab), "Carryover", .data$type_lab),
        levels = c("Carryover", levs2)
      )
    )
  caov_points <- mainPoints %>%
    filter(.data$type == "Carryover") %>%
    select("channel", "caov_spend" = "spend_point")
  mainPoints <- mainPoints %>%
    left_join(caov_points, "channel") %>%
    mutate(
      constr_low_abs = ifelse(.data$type == levs1[2], .data$constr_low_abs + .data$caov_spend, NA),
      constr_up_abs = ifelse(.data$type == levs1[2], .data$constr_up_abs + .data$caov_spend, NA),
      constr_low_unb_abs = ifelse(.data$type == levs1[3], .data$constr_low_unb_abs + .data$caov_spend, NA),
      constr_up_unb_abs = ifelse(.data$type == levs1[3], .data$constr_up_unb_abs + .data$caov_spend, NA)
    ) %>%
    mutate(
      plot_lb = ifelse(is.na(.data$constr_low_abs), .data$constr_low_unb_abs, .data$constr_low_abs),
      plot_ub = ifelse(is.na(.data$constr_up_abs), .data$constr_up_unb_abs, .data$constr_up_abs)
    )

  caption <- paste0(
    ifelse(any_topped, sprintf(
      "^ Given the upper/lower constrains, the total budget (%s) can't be fully allocated\n",
      formatNum(eval_list$total_budget, abbr = TRUE)
    ), ""),
    paste0("* ", formulax1, "\n"),
    paste0("* ", formulax2, "\n"),
    paste("** Dotted lines show budget optimization lower-upper ranges per media")
  )

  outputs[["p3"]] <- p3 <- plotDT_scurve %>%
    ggplot() +
    scale_x_abbr() +
    scale_y_abbr() +
    geom_line(aes(x = .data$spend, y = .data$total_response), show.legend = FALSE, size = 0.5) +
    facet_wrap(.data$constr_label ~ ., scales = "free", ncol = 3) +
    geom_area(
      data = group_by(plotDT_scurve, .data$constr_label) %>%
        filter(.data$spend <= .data$mean_carryover),
      aes(x = .data$spend, y = .data$total_response, color = .data$constr_label),
      stat = "align", position = "stack", size = 0.1,
      fill = "grey50", alpha = 0.4, show.legend = FALSE
    ) +
    geom_errorbar(
      data = filter(mainPoints, !is.na(.data$constr_label)),
      mapping = aes(
        x = .data$spend_point, y = .data$response_point,
        xmin = .data$plot_lb, xmax = .data$plot_ub
      ),
      color = "black", linetype = "dotted"
    ) +
    geom_point(
      data = filter(mainPoints, !is.na(.data$plot_lb), !is.na(.data$mean_spend)),
      aes(x = .data$plot_lb, y = .data$response_point), shape = 18
    ) +
    geom_point(
      data = filter(mainPoints, !is.na(.data$plot_ub), !is.na(.data$mean_spend)),
      aes(x = .data$plot_ub, y = .data$response_point), shape = 18
    ) +
    geom_point(data = filter(mainPoints, !is.na(.data$constr_label)), aes(
      x = .data$spend_point, y = .data$response_point, fill = .data$type_lab
    ), size = 2.5, shape = 21) +
    scale_fill_manual(values = c("white", "grey", "steelblue", "darkgoldenrod4")) +
    theme_lares(legend = "top", pal = 2) +
    labs(
      title = "Simulated Response Curve for Selected Allocation Period",
      x = sprintf("Spend** per %s (Mean Adstock Zone in Grey)", InputCollect$intervalType),
      y = sprintf("Total Response [%s]", InputCollect$dep_var_type),
      shape = NULL, color = NULL, fill = NULL,
      caption = caption
    )

  # Gather all plots into a single one
  min_period_loc <- which.min(as.integer(lapply(dt_optimOut$periods, function(x) str_split(x, " ")[[1]][1])))
  outputs[["plots"]] <- plots <- (p1 / p2 / p3) +
    plot_layout(heights = c(
      0.8, 0.2 + length(dt_optimOut$channels) * 0.2,
      ceiling(length(dt_optimOut$channels) / 3)
    )) +
    plot_annotation(
      title = paste0("Budget Allocation Onepager for Model ID ", select_model),
      subtitle = sprintf(
        "%s\nSimulation date range: %s to %s (%s) | Scenario: %s",
        errors,
        dt_optimOut$date_min[1],
        dt_optimOut$date_max[1],
        dt_optimOut$periods[min_period_loc],
        scenario
      ),
      theme = theme_lares(background = "white")
    )

  # Gather all plots
  if (export) {
    suffix <- case_when(
      scenario == "max_response" & metric == "ROAS" ~ "best_roas",
      scenario == "max_response" & metric == "CPA" ~ "best_cpa",
      scenario == "target_efficiency" & metric == "ROAS" ~ "target_roas",
      scenario == "target_efficiency" & metric == "CPA" ~ "target_cpa",
      TRUE ~ "none"
    )
    # suffix <- ifelse(scenario == "max_response", "resp", "effi")
    filename <- paste0(OutputCollect$plot_folder, select_model, "_reallocated_", suffix, ".png")
    if (!quiet) message("Exporting charts into file: ", filename)
    ggsave(
      filename = filename,
      plot = plots, limitsize = FALSE,
      dpi = 350, width = 12, height = 10 + 2 * ceiling(length(dt_optimOut$channels) / 3)
    )
  }

  return(invisible(outputs))
}

refresh_plots <- function(InputCollectRF, OutputCollectRF, ReportCollect, export = TRUE) {
  selectID <- tail(ReportCollect$selectIDs, 1)
  if (is.null(selectID)) selectID <- tail(ReportCollect$resultHypParamReport$solID, 1)
  message(">> Plotting refresh results for model: ", v2t(selectID))
  xDecompVecReport <- filter(ReportCollect$xDecompVecReport, .data$solID %in% selectID)
  xDecompAggReport <- filter(ReportCollect$xDecompAggReport, .data$solID %in% selectID)
  outputs <- list()

  ## 1. Actual vs fitted
  xDecompVecReportPlot <- xDecompVecReport %>%
    group_by(.data$refreshStatus) %>%
    mutate(
      refreshStart = min(.data$ds),
      refreshEnd = max(.data$ds),
      duration = as.numeric(
        (as.integer(.data$refreshEnd) - as.integer(.data$refreshStart) +
          InputCollectRF$dayInterval) / InputCollectRF$dayInterval
      )
    )

  dt_refreshDates <- xDecompVecReportPlot %>%
    select(.data$refreshStatus, .data$refreshStart, .data$refreshEnd, .data$duration) %>%
    distinct() %>%
    mutate(label = ifelse(.data$refreshStatus == 0, sprintf(
      "Initial: %s, %s %ss", .data$refreshStart, .data$duration, InputCollectRF$intervalType
    ),
    sprintf(
      "Refresh #%s: %s, %s %ss", .data$refreshStatus, .data$refreshStart,
      .data$duration, InputCollectRF$intervalType
    )
    ))

  xDecompVecReportMelted <- xDecompVecReportPlot %>%
    select(.data$ds, .data$refreshStart, .data$refreshEnd, .data$refreshStatus,
      actual = .data$dep_var, prediction = .data$depVarHat
    ) %>%
    tidyr::gather(
      key = "variable", value = "value",
      -c("ds", "refreshStatus", "refreshStart", "refreshEnd")
    )

  outputs[["pFitRF"]] <- pFitRF <- ggplot(xDecompVecReportMelted) +
    geom_line(aes(x = .data$ds, y = .data$value, color = .data$variable)) +
    geom_rect(
      data = dt_refreshDates,
      aes(
        xmin = .data$refreshStart, xmax = .data$refreshEnd,
        fill = as.character(.data$refreshStatus)
      ),
      ymin = -Inf, ymax = Inf, alpha = 0.2
    ) +
    theme(
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      panel.background = element_blank(), # legend.position = c(0.1, 0.8),
      legend.background = element_rect(fill = alpha("white", 0.4)),
    ) +
    theme_lares() +
    scale_fill_brewer(palette = "BuGn") +
    geom_text(data = dt_refreshDates, mapping = aes(
      x = .data$refreshStart, y = max(xDecompVecReportMelted$value),
      label = .data$label,
      angle = 270, hjust = -0.1, vjust = -0.2
    ), color = "gray40") +
    labs(
      title = "Model Refresh: Actual vs. Predicted Response",
      subtitle = paste0(
        "Assembled R2: ", round(get_rsq(
          true = xDecompVecReportPlot$dep_var,
          predicted = xDecompVecReportPlot$depVarHat
        ), 2)
      ),
      x = "Date", y = "Response", fill = "Refresh", color = "Type"
    ) +
    scale_y_abbr()

  if (export) {
    ggsave(
      filename = paste0(OutputCollectRF$plot_folder, "report_actual_fitted.png"),
      plot = pFitRF,
      dpi = 900, width = 12, height = 8, limitsize = FALSE
    )
  }

  ## 2. Stacked bar plot
  xDecompAggReportPlotBase <- xDecompAggReport %>%
    filter(.data$rn %in% c(InputCollectRF$prophet_vars, "(Intercept)")) %>%
    mutate(perc = ifelse(.data$refreshStatus == 0, .data$xDecompPerc, .data$xDecompPercRF)) %>%
    select(.data$rn, .data$perc, .data$refreshStatus) %>%
    group_by(.data$refreshStatus) %>%
    summarise(variable = "baseline", percentage = sum(.data$perc), roi_total = NA)

  xDecompAggReportPlot <- xDecompAggReport %>%
    filter(!.data$rn %in% c(InputCollectRF$prophet_vars, "(Intercept)")) %>%
    mutate(percentage = ifelse(.data$refreshStatus == 0, .data$xDecompPerc, .data$xDecompPercRF)) %>%
    select(.data$refreshStatus, variable = .data$rn, .data$percentage, .data$roi_total) %>%
    bind_rows(xDecompAggReportPlotBase) %>%
    arrange(.data$refreshStatus, desc(.data$variable)) %>%
    mutate(refreshStatus = ifelse(
      .data$refreshStatus == 0, "Init.mod",
      paste0("Refresh", .data$refreshStatus)
    ))

  ySecScale <- 0.75 * max(xDecompAggReportPlot$roi_total / max(xDecompAggReportPlot$percentage), na.rm = TRUE)
  ymax <- 1.1 * max(c(xDecompAggReportPlot$roi_total / ySecScale, xDecompAggReportPlot$percentage), na.rm = TRUE)

  outputs[["pBarRF"]] <- pBarRF <- ggplot(
    xDecompAggReportPlot,
    aes(x = .data$variable, y = .data$percentage, fill = .data$variable)
  ) +
    geom_bar(alpha = 0.8, position = "dodge", stat = "identity", na.rm = TRUE) +
    facet_wrap(~ .data$refreshStatus, scales = "free") +
    theme_lares(grid = "X") +
    scale_fill_manual(values = robyn_palette()$fill) +
    geom_text(aes(label = paste0(round(.data$percentage * 100, 1), "%")),
      size = 3, na.rm = TRUE,
      position = position_dodge(width = 0.9), hjust = -0.25
    ) +
    geom_point(
      data = xDecompAggReportPlot,
      aes(
        x = .data$variable,
        y = .data$roi_total / ySecScale,
        color = .data$variable
      ),
      size = 4, shape = 17, na.rm = TRUE,
    ) +
    geom_text(
      data = xDecompAggReportPlot,
      aes(
        label = round(.data$roi_total, 2),
        x = .data$variable,
        y = .data$roi_total / ySecScale
      ),
      size = 3, na.rm = TRUE, hjust = -0.4, fontface = "bold",
      position = position_dodge(width = 0.9)
    ) +
    scale_color_manual(values = robyn_palette()$fill) +
    scale_y_continuous(
      sec.axis = sec_axis(~ . * ySecScale), breaks = seq(0, ymax, 0.2),
      limits = c(0, ymax), name = "Total ROI"
    ) +
    coord_flip() +
    theme(legend.position = "none", axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    labs(
      title = "Model Refresh: Decomposition & Paid Media ROI",
      subtitle = paste0(
        "Baseline includes intercept and prophet vars: ",
        paste(InputCollectRF$prophet_vars, collapse = ", ")
      )
    )

  if (export) {
    ggsave(
      filename = paste0(OutputCollectRF$plot_folder, "report_decomposition.png"),
      plot = pBarRF,
      dpi = 900, width = 12, height = 8, limitsize = FALSE
    )
  }
  return(invisible(outputs))
}

refresh_plots_json <- function(OutputCollectRF, json_file, export = TRUE) {
  outputs <- list()
  chainData <- robyn_chain(json_file)
  solID <- tail(names(chainData), 1)
  dayInterval <- chainData[[solID]]$InputCollect$dayInterval
  intervalType <- chainData[[solID]]$InputCollect$intervalType
  rsq <- chainData[[solID]]$ExportedModel$errors$rsq_train

  ## 1. Fitted vs actual
  temp <- OutputCollectRF$allPareto$plotDataCollect[[solID]]
  xDecompVecPlotMelted <- temp$plot5data$xDecompVecPlotMelted %>%
    mutate(
      linetype = ifelse(.data$variable == "predicted", "solid", "dotted"),
      variable = stringr::str_to_title(.data$variable),
      ds = as.Date(.data$ds, origin = "1970-01-01")
    )
  dt_refreshDates <- data.frame(
    solID = names(chainData),
    window_start = as.Date(unlist(lapply(chainData, function(x) x$InputCollect$window_start)), origin = "1970-01-01"),
    window_end = as.Date(unlist(lapply(chainData, function(x) x$InputCollect$window_end)), origin = "1970-01-01"),
    duration = unlist(c(0, unlist(lapply(chainData, function(x) x$InputCollect$refresh_steps))))
  ) %>%
    filter(.data$duration > 0) %>%
    mutate(refreshStatus = row_number()) %>%
    mutate(
      refreshStart = .data$window_end - dayInterval * .data$duration,
      refreshEnd = .data$window_end
    ) %>%
    mutate(label = ifelse(.data$refreshStatus == 0, sprintf(
      "Initial: %s, %s %ss", .data$refreshStart, .data$duration, intervalType
    ),
    sprintf(
      "Refresh #%s: %s, %s %ss", .data$refreshStatus, .data$refreshStart, .data$duration, intervalType
    )
    )) %>%
    as_tibble()
  outputs[["pFitRF"]] <- pFitRF <- ggplot(xDecompVecPlotMelted) +
    geom_path(aes(x = .data$ds, y = .data$value, color = .data$variable, linetype = .data$linetype), size = 0.6) +
    geom_rect(
      data = dt_refreshDates,
      aes(
        xmin = .data$refreshStart, xmax = .data$refreshEnd,
        fill = as.character(.data$refreshStatus)
      ),
      ymin = -Inf, ymax = Inf, alpha = 0.2
    ) +
    scale_fill_brewer(palette = "BuGn") +
    geom_text(data = dt_refreshDates, mapping = aes(
      x = .data$refreshStart, y = max(xDecompVecPlotMelted$value),
      label = .data$label,
      angle = 270, hjust = 0, vjust = -0.2
    ), color = "gray40") +
    theme_lares(legend = "top", pal = 2) +
    scale_y_abbr() +
    guides(linetype = "none", fill = "none") +
    labs(
      title = "Actual vs. Predicted Response",
      # subtitle = paste("Train R2 =", round(rsq, 4)),
      x = "Date", y = "Response", color = NULL, fill = NULL
    )

  if (export) {
    ggsave(
      filename = paste0(OutputCollectRF$plot_folder, "report_actual_fitted.png"),
      plot = pFitRF,
      dpi = 900, width = 12, height = 8, limitsize = FALSE
    )
  }

  ## 2. Stacked bar plot
  df <- lapply(chainData, function(x) x$ExportedModel$summary) %>%
    bind_rows(.id = "solID") %>%
    as_tibble() %>%
    select(-.data$coef) %>%
    mutate(
      solID = factor(.data$solID, levels = names(chainData)),
      label = factor(
        sprintf("%s [%s]", .data$solID, as.integer(.data$solID) - 1),
        levels = sprintf("%s [%s]", names(chainData), 0:(length(chainData) - 1))
      ),
      variable = ifelse(.data$variable %in% c(chainData[[1]]$InputCollect$prophet_vars, "(Intercept)"),
        "baseline", .data$variable
      )
    ) %>%
    group_by(.data$solID, .data$label, .data$variable) %>%
    summarise_all(sum)

  outputs[["pBarRF"]] <- pBarRF <- df %>%
    ggplot(aes(y = .data$variable)) +
    geom_col(aes(x = .data$decompPer)) +
    geom_text(
      aes(
        x = .data$decompPer,
        label = formatNum(100 * .data$decompPer, signif = 2, pos = "%")
      ),
      na.rm = TRUE, hjust = -0.2, size = 2.8
    ) +
    geom_point(aes(x = .data$performance), na.rm = TRUE, size = 2, colour = "#39638b") +
    geom_text(
      aes(
        x = .data$performance,
        label = formatNum(.data$performance, 2)
      ),
      na.rm = TRUE, hjust = -0.4, size = 2.8, colour = "#39638b"
    ) +
    facet_wrap(. ~ .data$label, scales = "free") +
    # scale_x_percent(limits = c(0, max(df$performance, na.rm = TRUE) * 1.2)) +
    labs(
      title = paste(
        "Model refresh: Decomposition & Paid Media",
        ifelse(chainData[[1]]$InputCollect$dep_var_type == "revenue", "ROI", "CPA")
      ),
      subtitle = paste(
        "Baseline includes intercept and all prophet vars:",
        v2t(chainData[[1]]$InputCollect$prophet_vars, quotes = FALSE)
      ),
      x = NULL, y = NULL
    ) +
    theme_lares(grid = "Y") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

  if (export) {
    ggsave(
      filename = paste0(chainData[[length(chainData)]]$ExportedModel$plot_folder, "report_decomposition.png"),
      plot = pBarRF,
      dpi = 900, width = 12, height = 8, limitsize = FALSE
    )
  }

  return(invisible(outputs))
}


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
ts_validation <- function(OutputModels, quiet = FALSE, ...) {
  if (!isTRUE(OutputModels$ts_validation)) {
    return(NULL)
  }
  resultHypParam <- bind_rows(
    lapply(OutputModels[
      which(names(OutputModels) %in% paste0("trial", seq(OutputModels$trials)))
    ], function(x) x$resultCollect$resultHypParam)
  ) %>%
    group_by(.data$trial) %>%
    mutate(i = row_number()) %>%
    ungroup()

  resultHypParamLong <- suppressWarnings(
    resultHypParam %>%
      select(.data$solID, .data$i, .data$trial, .data$train_size, starts_with("rsq_")) %>%
      mutate(trial = paste("Trial", .data$trial)) %>%
      tidyr::gather("dataset", "rsq", starts_with("rsq_")) %>%
      bind_cols(select(resultHypParam, .data$solID, starts_with("nrmse_")) %>%
        tidyr::gather("del", "nrmse", starts_with("nrmse_")) %>%
        select(.data$nrmse)) %>%
      # group_by(.data$trial, .data$dataset) %>%
      mutate(
        rsq = lares::winsorize(.data$rsq, thresh = c(0.01, 0.99)),
        nrmse = lares::winsorize(.data$nrmse, thresh = c(0.00, 0.99)),
        dataset = gsub("rsq_", "", .data$dataset)
      ) %>%
      ungroup()
  )

  pIters <- resultHypParam %>%
    ggplot(aes(x = .data$i, y = .data$train_size)) +
    geom_point(fill = "black", alpha = 0.5, size = 1.2, pch = 23) +
    # geom_smooth() +
    labs(y = "Train Size", x = "Iteration") +
    scale_y_percent() +
    theme_lares() +
    scale_x_abbr()

  # pRSQ <- ggplot(resultHypParamLong, aes(
  #   x = .data$i, y = .data$rsq,
  #   colour = .data$dataset,
  #   group = as.character(.data$trial)
  # )) +
  #   geom_point(alpha = 0.5, size = 0.9) +
  #   facet_grid(.data$trial ~ .) +
  #   geom_hline(yintercept = 0, linetype = "dashed") +
  #   labs(y = "Adjusted R2 [1% Winsorized]", x = "Iteration", colour = "Dataset") +
  #   theme_lares(legend = "top", pal = 2) +
  #   scale_x_abbr()

  pNRMSE <- ggplot(resultHypParamLong, aes(
    x = .data$i, y = .data$nrmse,
    colour = .data$dataset
    # group = as.character(.data$trial)
  )) +
    geom_point(alpha = 0.2, size = 0.9) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
    facet_grid(.data$trial ~ .) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(y = "NRMSE [Upper 1% Winsorized]", x = "Iteration", colour = "Dataset") +
    theme_lares(legend = "top", pal = 2) +
    scale_x_abbr()

  pw <- (pNRMSE / pIters) +
    patchwork::plot_annotation(title = "Time-series validation & Convergence") +
    patchwork::plot_layout(heights = c(2, 1), guides = "collect") &
    theme_lares(legend = "top")
  return(pw)
}


#' @rdname robyn_outputs
#' @param solID Character vector. Model IDs to plot.
#' @param exclude Character vector. Manually exclude variables from plot.
#' @export
decomp_plot <- function(InputCollect, OutputCollect, solID = NULL, exclude = NULL) {
  check_opts(solID, OutputCollect$allSolutions)
  intType <- str_to_title(case_when(
    InputCollect$intervalType %in% c("month", "week") ~ paste0(InputCollect$intervalType, "ly"),
    InputCollect$intervalType == "day" ~ "daily",
    TRUE ~ InputCollect$intervalType
  ))
  varType <- str_to_title(InputCollect$dep_var_type)
  pal <- names(lares::lares_pal()$palette)
  df <- OutputCollect$xDecompVecCollect[OutputCollect$xDecompVecCollect$solID %in% solID, ] %>%
    select(
      "solID", "ds", "dep_var", any_of("intercept"),
      any_of(unique(OutputCollect$xDecompAgg$rn))
    ) %>%
    tidyr::gather("variable", "value", -.data$ds, -.data$solID, -.data$dep_var) %>%
    filter(!.data$variable %in% exclude) %>%
    mutate(variable = factor(.data$variable, levels = rev(unique(.data$variable))))
  p <- ggplot(df, aes(x = .data$ds, y = .data$value, fill = .data$variable)) +
    facet_grid(.data$solID ~ .) +
    labs(
      title = paste(varType, "Decomposition by Variable"),
      x = NULL, y = paste(intType, varType), fill = NULL
    ) +
    geom_area() +
    theme_lares(legend = "right") +
    scale_fill_manual(values = rev(pal[seq(length(unique(df$variable)))])) +
    scale_y_abbr()
  return(p)
}
