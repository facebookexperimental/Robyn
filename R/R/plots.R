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
      plotMediaShareLoop <- plotMediaShare[plotMediaShare$solID == sid, ]
      rsq_train_plot <- round(plotMediaShareLoop$rsq_train[1], 4)
      rsq_val_plot <- round(plotMediaShareLoop$rsq_val[1], 4)
      rsq_test_plot <- round(plotMediaShareLoop$rsq_test[1], 4)
      nrmse_train_plot <- round(plotMediaShareLoop$nrmse_train[1], 4)
      nrmse_val_plot <- round(plotMediaShareLoop$nrmse_val[1], 4)
      nrmse_test_plot <- round(plotMediaShareLoop$nrmse_test[1], 4)
      decomp_rssd_plot <- round(plotMediaShareLoop$decomp.rssd[1], 4)
      mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input), round(plotMediaShareLoop$mape[1], 4), 0)
      train_size <- round(plotMediaShareLoop$train_size[1], 4)

      if (val) {
        errors <- paste0(
          "NRMSE: train = ", nrmse_train_plot, " | val = ", nrmse_val_plot, " | test = ", nrmse_test_plot,
          "; [Adj.R2: train = ", rsq_train_plot, " | val = ", rsq_val_plot, " | test = ", rsq_test_plot, "]",
          ";\nDECOMP.RSSD = ", decomp_rssd_plot,
          "; MAPE = ", mape_lift_plot
        )
      } else {
        errors <- paste0(
          "NRMSE train = ", nrmse_train_plot,
          "; [Adj.R2 train = ", rsq_train_plot, "]",
          "; DECOMP.RSSD = ", decomp_rssd_plot,
          "; MAPE = ", mape_lift_plot
        )
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
          title = paste0("Share of Spend VS Share of Effect with total ", type),
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
            .data$spend < max(dt_scurvePlotMean$mean_spend) * trim_rate,
            .data$response < max(dt_scurvePlotMean$mean_response) * trim_rate
          )
      }
      if (!"channel" %in% colnames(dt_scurvePlotMean)) {
        dt_scurvePlotMean$channel <- dt_scurvePlotMean$rn
      }
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
                             scenario, export = TRUE, quiet = FALSE) {
  outputs <- list()

  subtitle <- sprintf(
    paste0(
      "Total spend increase: %s%%",
      "\nTotal response increase: %s%% with optimised spend allocation"
    ),
    round(mean(dt_optimOut$optmSpendUnitTotalDelta) * 100, 1),
    round(mean(dt_optimOut$optmResponseUnitTotalLift) * 100, 1)
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
  if (OutputCollect$OutputModels$ts_validation) {
    errors <- paste0(
      "Adj.R2: train = ", rsq_train_plot, " | val = ", rsq_val_plot, " | test = ", rsq_test_plot,
      " ### NRMSE: train = ", nrmse_train_plot, " | val = ", nrmse_val_plot, " | test = ", nrmse_test_plot,
      " ### DECOMP.RSSD = ", decomp_rssd_plot, " ### MAPE = ", mape_lift_plot
    )
  } else {
    errors <- paste0(
      "Adj.R2 train = ", rsq_train_plot, " ### NRMSE train = ", nrmse_train_plot,
      " ### DECOMP.RSSD = ", decomp_rssd_plot, " ### MAPE = ", mape_lift_plot
    )
  }

  # 1. Response comparison plot
  plotDT_resp <- select(dt_optimOut, .data$channels, .data$initResponseUnit, .data$optmResponseUnit) %>%
    mutate(channels = as.factor(.data$channels))
  names(plotDT_resp) <- c("channel", "Initial Mean Response", "Optimised Mean Response")
  plotDT_resp <- tidyr::gather(plotDT_resp, "variable", "response", -.data$channel)
  outputs[["p12"]] <- p12 <- ggplot(plotDT_resp, aes(
    y = reorder(.data$channel, -as.integer(.data$channel)),
    x = .data$response,
    fill = reorder(.data$variable, as.numeric(as.factor(.data$variable)))
  )) +
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
  plotDT_share <- tidyr::gather(plotDT_share, "variable", "spend_share", -.data$channel)
  outputs[["p13"]] <- p13 <- ggplot(plotDT_share, aes(
    y = reorder(.data$channel, -as.integer(.data$channel)),
    x = .data$spend_share, fill = .data$variable
  )) +
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
  plotDT_saturation <- OutputCollect$mediaVecCollect %>%
    filter(.data$solID == select_model, .data$type == "saturatedSpendReversed") %>%
    select(.data$ds, all_of(InputCollect$paid_media_spends)) %>%
    tidyr::gather("channel", "spend", -.data$ds)

  plotDT_decomp <- OutputCollect$mediaVecCollect %>%
    filter(.data$solID == select_model, .data$type == "decompMedia") %>%
    select(.data$ds, all_of(InputCollect$paid_media_spends)) %>%
    tidyr::gather("channel", "response", -.data$ds)

  plotDT_scurve <- data.frame(plotDT_saturation, response = plotDT_decomp$response) %>%
    filter(.data$spend >= 0) %>%
    as_tibble()

  dt_optimOutScurve <- rbind(
    select(dt_optimOut, .data$channels, .data$initSpendUnit, .data$initResponseUnit) %>% mutate(x = "Initial") %>% as.matrix(),
    select(dt_optimOut, .data$channels, .data$optmSpendUnit, .data$optmResponseUnit) %>% mutate(x = "Optimised") %>% as.matrix()
  ) %>% as.data.frame()
  colnames(dt_optimOutScurve) <- c("channels", "spend", "response", "type")
  dt_optimOutScurve <- dt_optimOutScurve %>%
    mutate(spend = as.numeric(.data$spend), response = as.numeric(.data$response)) %>%
    group_by(.data$channels) %>%
    mutate(
      spend_dif = dplyr::last(.data$spend) - dplyr::first(.data$spend),
      response_dif = dplyr::last(.data$response) - dplyr::first(.data$response)
    )

  trim_rate <- 1.6 # maybe enable as a parameter
  if (trim_rate > 0) {
    plotDT_scurve <- plotDT_scurve %>%
      filter(
        .data$spend < max(dt_optimOutScurve$spend) * trim_rate,
        .data$response < max(dt_optimOutScurve$response) * trim_rate
      )
  }
  outputs[["p14"]] <- p14 <- ggplot(data = plotDT_scurve, aes(
    x = .data$spend, y = .data$response, color = .data$channel
  )) +
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
      legend.spacing.y = unit(0.2, "cm")
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

  pRSQ <- ggplot(resultHypParamLong, aes(
    x = .data$i, y = .data$rsq,
    colour = .data$dataset,
    group = as.character(.data$trial)
  )) +
    geom_point(alpha = 0.5, size = 0.9) +
    facet_grid(.data$trial ~ .) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(y = "Adjusted R2 [1% Winsorized]", x = "Iteration", colour = "Dataset") +
    theme_lares(legend = "top", pal = 2) +
    scale_x_abbr()

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
