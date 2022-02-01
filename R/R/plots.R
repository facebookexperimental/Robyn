# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


####################################################################
#' Generate and Export Robyn Plots
#'
#' @inheritParams robyn_outputs
#' @inheritParams robyn_csv
#' @export
robyn_plots <- function(InputCollect, OutputCollect, export = TRUE) {

  check_class("robyn_outputs", OutputCollect)
  pareto_fronts <- OutputCollect$pareto_fronts
  hyper_fixed <- OutputCollect$hyper_fixed
  temp_all <- OutputCollect$allPareto
  all_plots <- list()

  if (!hyper_fixed) {

    ## Prophet
    if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0
        || !is.null(InputCollect$factor_vars) && length(InputCollect$factor_vars) > 0)
    {
      dt_plotProphet <- InputCollect$dt_mod[, c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$factor_vars), with = FALSE]
      dt_plotProphet <- suppressWarnings(melt.data.table(dt_plotProphet, id.vars = "ds"))
      all_plots[["pProphet"]] <- pProphet <- ggplot(
        dt_plotProphet, aes(x = ds, y = value)) +
        geom_line(color = "steelblue") +
        facet_wrap(~variable, scales = "free", ncol = 1) +
        labs(title = "Prophet decomposition") +
        xlab(NULL) + ylab(NULL)
      if (export) ggsave(
        paste0(OutputCollect$plot_folder, "prophet_decomp.png"),
        plot = pProphet,
        dpi = 600, width = 12, height = 3 * length(levels(dt_plotProphet$variable))
      )
    }

    ## Spend exposure model
    if (any(InputCollect$costSelector)) {
      all_plots[["pSpendExposure"]] <- pSpendExposure <- wrap_plots(
        InputCollect$plotNLSCollect,
        ncol = ifelse(length(InputCollect$plotNLSCollect) <= 3, length(InputCollect$plotNLSCollect), 3)
      ) +
        plot_annotation(
          title = "Spend-exposure fitting with Michaelis-Menten model",
          theme = theme(plot.title = element_text(hjust = 0.5))
        )
      if (export) ggsave(
        paste0(OutputCollect$plot_folder, "spend_exposure_fitting.png"),
        plot = pSpendExposure, dpi = 600, width = 12,
        height = ceiling(length(InputCollect$plotNLSCollect) / 3) * 7
      )
    } else {
      message("No spend-exposure modelling needed. All media variables used for MMM are spend variables")
    }

    ## Hyperparameter sampling distribution
    if (length(temp_all) > 0) {
      resultHypParam <- copy(temp_all$resultHypParam)
      resultHypParam.melted <- melt.data.table(resultHypParam[, c(names(InputCollect$hyperparameters), "robynPareto"), with = FALSE], id.vars = c("robynPareto"))
      all_plots[["pSamp"]] <- pSamp <- ggplot(
        resultHypParam.melted, aes(x = value, y = variable, color = variable, fill = variable)) +
        geom_violin(alpha = .5, size = 0) +
        geom_point(size = 0.2) +
        theme(legend.position = "none") +
        labs(
          title = "Hyperparameter optimisation sampling",
          subtitle = paste0("Sample distribution", ", iterations = ", InputCollect$iterations, " * ", InputCollect$trials, " trial"),
          x = "Hyperparameter space",
          y = ""
        )
      if (export) ggsave(
        paste0(OutputCollect$plot_folder, "hypersampling.png"),
        plot = pSamp, dpi = 600, width = 12, height = 7
      )
    }

    ## Pareto front
    if (length(temp_all) > 0) {
      pareto_fronts_vec <- 1:pareto_fronts
      resultHypParam <- copy(temp_all$resultHypParam)
      if (!is.null(InputCollect$calibration_input)) {
        resultHypParam[, iterations := ifelse(is.na(robynPareto), NA, iterations)]
      }
      pParFront <- ggplot(resultHypParam, aes(x = nrmse, y = decomp.rssd, color = iterations)) +
        geom_point(size = 0.5) +
        geom_line(data = resultHypParam[robynPareto == 1], aes(x = nrmse, y = decomp.rssd), colour = "coral4") +
        scale_colour_gradient(low = "navyblue", high = "skyblue") +
        labs(
          title = ifelse(is.null(InputCollect$calibration_input), "Multi-objective evolutionary performance", "Multi-objective evolutionary performance with top 10% calibration"),
          subtitle = paste0("2D Pareto front 1-3 with ", InputCollect$nevergrad_algo, ", iterations = ", InputCollect$iterations, " * ", InputCollect$trials, " trial"),
          x = "NRMSE",
          y = "DECOMP.RSSD"
        )
      if (length(pareto_fronts_vec) > 1) {
        for (pfs in 2:max(pareto_fronts_vec)) {
          if (pfs == 2) {
            pf_color <- "coral3"
          } else if (pfs == 3) {
            pf_color <- "coral2"
          } else {
            pf_color <- "coral"
          }
          pParFront <- pParFront + geom_line(
            data = resultHypParam[robynPareto == pfs],
            aes(x = nrmse, y = decomp.rssd), colour = pf_color)
        }
      }
      all_plots[["pParFront"]] <- pParFront
      if (export) ggsave(
        paste0(OutputCollect$plot_folder, "pareto_front.png"),
        plot = pParFront,
        dpi = 600, width = 12, height = 7
      )
    }

    ## Ridgeline model convergence
    if (length(temp_all) > 0) {
      xDecompAgg <- copy(temp_all$xDecompAgg)
      dt_ridges <- xDecompAgg[rn %in% InputCollect$paid_media_vars
                              , .(variables = rn
                                  , roi_total
                                  , iteration = (iterNG-1)*InputCollect$cores+iterPar
                                  , trial)][order(iteration, variables)]
      bin_limits <- c(1,20)
      qt_len <- ifelse(InputCollect$iterations <=100, 1,
                       ifelse(InputCollect$iterations > 2000, 20, ceiling(InputCollect$iterations/100)))
      set_qt <- floor(quantile(1:InputCollect$iterations, seq(0, 1, length.out = qt_len+1)))
      set_bin <- set_qt[-1]
      dt_ridges[, iter_bin := cut(dt_ridges$iteration, breaks = set_qt, labels = set_bin)]
      dt_ridges <- dt_ridges[!is.na(iter_bin)]
      dt_ridges[, iter_bin := factor(iter_bin, levels = sort(set_bin, decreasing = TRUE))]
      dt_ridges[, trial := as.factor(trial)]
      all_plots[["pRidges"]] <- pRidges <- ggplot(
        dt_ridges, aes(x = roi_total, y = iter_bin, fill = as.integer(iter_bin), linetype = trial)) +
        scale_fill_distiller(palette = "GnBu") +
        geom_density_ridges(scale = 4, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7) +
        facet_wrap(~ variables, scales = "free") +
        guides(fill = "none")+
        theme(panel.background = element_blank()) +
        labs(x = "Total ROAS", y = "Iteration Bucket"
             ,title = "ROAS distribution over iteration"
             ,fill = "iter bucket")
      if (export) suppressMessages(ggsave(
        paste0(OutputCollect$plot_folder, "roas_convergence.png"),
        plot = pRidges, dpi = 600, width = 12,
        height = ceiling(InputCollect$mediaVarCount / 3) * 6
      ))
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
robyn_onepagers <- function(InputCollect, OutputCollect, selected = NULL, quiet = FALSE, export = TRUE) {

  check_class("robyn_outputs", OutputCollect)
  pareto_fronts <- OutputCollect$pareto_fronts
  hyper_fixed <- OutputCollect$hyper_fixed
  resultHypParam <- copy(OutputCollect$resultHypParam)
  xDecompAgg <- copy(OutputCollect$xDecompAgg)
  if (!is.null(selected)) {
    if ("clusters" %in% selected) selected <- OutputCollect$clusters$models$solID
    resultHypParam <- resultHypParam[solID %in% selected]
    xDecompAgg <- xDecompAgg[solID %in% selected]
    if (!quiet) message(">> Exporting only cluster results one-pagers (", nrow(resultHypParam), ")...")
  }

  # Prepare for parallel plotting
  if (check_parallel_plot()) registerDoParallel(InputCollect$cores) else registerDoSEQ()
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
    if (!quiet) message(paste(">> Plotting", count_mod_out, "selected models on", InputCollect$cores, "cores..."))
  } else {
    if (!quiet) message(paste(">> Plotting", count_mod_out, "selected models on 1 core (MacOS fallback)..."))
  }

  if (!quiet & count_mod_out > 0) pbplot <- txtProgressBar(min = 0, max = count_mod_out, style = 3)
  temp <- OutputCollect$allPareto$plotDataCollect
  all_plots <- list()
  cnt <- 0

  for (pf in pareto_fronts_vec) {

    plotMediaShare <- xDecompAgg[robynPareto == pf & rn %in% InputCollect$paid_media_vars]
    uniqueSol <- plotMediaShare[, unique(solID)]

    # parallelResult <- for (sid in uniqueSol) {
    parallelResult <- foreach(sid = uniqueSol) %dorng% {

      plotMediaShareLoop <- plotMediaShare[solID == sid]
      rsq_train_plot <- plotMediaShareLoop[, round(unique(rsq_train), 4)]
      nrmse_plot <- plotMediaShareLoop[, round(unique(nrmse), 4)]
      decomp_rssd_plot <- plotMediaShareLoop[, round(unique(decomp.rssd), 4)]
      mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input), plotMediaShareLoop[, round(unique(mape), 4)], NA)

      ## 1. Spend x effect share comparison
      plotMediaShareLoopBar <- temp[[sid]]$plot1data$plotMediaShareLoopBar
      plotMediaShareLoopLine <- temp[[sid]]$plot1data$plotMediaShareLoopLine
      ySecScale <- temp[[sid]]$plot1data$ySecScale
      p1 <- ggplot(plotMediaShareLoopBar, aes(x = .data$rn, y = .data$value, fill = .data$variable)) +
        geom_bar(stat = "identity", width = 0.5, position = "dodge") +
        geom_text(aes(label = paste0(round(.data$value * 100, 2), "%")),
                  color = "darkblue", position = position_dodge(width = 0.5), fontface = "bold") +
        geom_line(data = plotMediaShareLoopLine, aes(
          x = .data$rn, y = .data$value / ySecScale, group = 1, color = .data$variable),
          inherit.aes = FALSE) +
        geom_point(data = plotMediaShareLoopLine, aes(
          x = .data$rn, y = .data$value / ySecScale, group = 1, color = .data$variable),
          inherit.aes = FALSE, size = 4) +
        geom_text(
          data = plotMediaShareLoopLine, aes(
            label = round(.data$value, 2), x = .data$rn, y = .data$value / ySecScale, group = 1, color = .data$variable),
          fontface = "bold", inherit.aes = FALSE, hjust = -1, size = 6
        ) +
        scale_y_continuous(sec.axis = sec_axis(~ . * ySecScale)) +
        coord_flip() +
        theme(legend.title = element_blank(), legend.position = c(0.9, 0.2), axis.text.x = element_blank()) +
        scale_fill_brewer(palette = "Paired") +
        labs(
          title = paste0("Share of Spend VS Share of Effect with total ", ifelse(InputCollect$dep_var_type == "conversion", "CPA", "ROI")),
          subtitle = paste0(
            "rsq_train: ", rsq_train_plot,
            ", nrmse = ", nrmse_plot,
            ", decomp.rssd = ", decomp_rssd_plot,
            ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
          ),
          y = NULL, x = NULL
        )

      ## 2. Waterfall
      plotWaterfallLoop <- temp[[sid]]$plot2data$plotWaterfallLoop
      p2 <- suppressWarnings(
        ggplot(plotWaterfallLoop, aes(x = id, fill = sign)) +
          geom_rect(aes(x = rn, xmin = id - 0.45, xmax = id + 0.45,
                        ymin = end, ymax = start), stat = "identity") +
          scale_x_discrete("", breaks = levels(plotWaterfallLoop$rn), labels = plotWaterfallLoop$rn) +
          theme(axis.text.x = element_text(angle = 65, vjust = 0.6), legend.position = c(0.1, 0.1)) +
          geom_text(mapping = aes(
            label = paste0(format_unit(xDecompAgg), "\n", round(xDecompPerc * 100, 2), "%"),
            y = rowSums(cbind(plotWaterfallLoop$end, plotWaterfallLoop$xDecompPerc / 2))
          ), fontface = "bold") +
          coord_flip() +
          labs(
            title = "Response decomposition waterfall by predictor",
            subtitle = paste0(
              "rsq_train: ", rsq_train_plot,
              ", nrmse = ", nrmse_plot,
              ", decomp.rssd = ", decomp_rssd_plot,
              ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
            ),
            x = NULL, y = NULL
          ))

      ## 3. Adstock rate
      if (InputCollect$adstock == "geometric") {
        dt_geometric <- temp[[sid]]$plot3data$dt_geometric
        p3 <- ggplot(dt_geometric, aes(x = .data$channels, y = .data$thetas, fill = "coral")) +
          geom_bar(stat = "identity", width = 0.5) +
          theme(legend.position = "none") +
          coord_flip() +
          geom_text(aes(label = paste0(round(thetas * 100, 1), "%")),
                    position = position_dodge(width = 0.5), fontface = "bold") +
          ylim(0, 1) +
          labs(
            title = "Geometric adstock - fixed decay rate over time",
            subtitle = paste0(
              "rsq_train: ", rsq_train_plot,
              ", nrmse = ", nrmse_plot,
              ", decomp.rssd = ", decomp_rssd_plot,
              ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
            ),
            y = NULL, x = NULL
          )
      }
      if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
        weibullCollect <- temp[[sid]]$plot3data$weibullCollect
        wb_type <- temp[[sid]]$plot3data$wb_type
        p3 <- ggplot(weibullCollect, aes(x = .data$x, y = .data$decay_accumulated)) +
          geom_line(aes(color = .data$channel)) +
          facet_wrap(~.data$channel) +
          geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
          geom_text(aes(x = max(.data$x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife"), colour = "gray") +
          theme(legend.position = "none") +
          labs(title = paste0("Weibull adstock ", wb_type," - flexible decay rate over time"),
               subtitle = paste0(
                 "rsq_train: ", rsq_train_plot,
                 ", nrmse = ", nrmse_plot,
                 ", decomp.rssd = ", decomp_rssd_plot,
                 ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
               ),
               x = "Time unit", y = NULL)
      }

      ## 4. Response curve
      dt_scurvePlot <- temp[[sid]]$plot4data$dt_scurvePlot
      dt_scurvePlotMean <- temp[[sid]]$plot4data$dt_scurvePlotMean
      p4 <- ggplot(dt_scurvePlot[dt_scurvePlot$channel %in% InputCollect$paid_media_vars,],
                   aes(x = .data$spend, y = .data$response, color = .data$channel)) +
        geom_line() +
        geom_point(data = dt_scurvePlotMean, aes(x = .data$mean_spend, y = .data$mean_response, color = .data$channel)) +
        geom_text(data = dt_scurvePlotMean, aes(x = .data$mean_spend, y = .data$mean_response, label = round(.data$mean_spend, 0)),
                  show.legend = FALSE, hjust = -0.2) +
        theme(legend.position = c(0.9, 0.2)) +
        labs(
          title = "Response curve and mean spend by channel",
          subtitle = paste0(
            "rsq_train: ", rsq_train_plot,
            ", nrmse = ", nrmse_plot,
            ", decomp.rssd = ", decomp_rssd_plot,
            ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
          ),
          x = "Spend", y = "Response"
        )

      ## 5. Fitted vs actual
      xDecompVecPlotMelted <- temp[[sid]]$plot5data$xDecompVecPlotMelted
      p5 <- ggplot(xDecompVecPlotMelted, aes(x = .data$ds, y = .data$value, color = .data$variable)) +
        geom_line() +
        theme(legend.position = c(0.9, 0.9)) +
        labs(
          title = "Actual vs. predicted response",
          subtitle = paste0(
            "rsq_train: ", rsq_train_plot,
            ", nrmse = ", nrmse_plot,
            ", decomp.rssd = ", decomp_rssd_plot,
            ifelse(!is.na(mape_lift_plot), paste0(", mape.lift = ", mape_lift_plot), "")
          ),
          x = "Date", y = "Response"
        )

      ## 6. Diagnostic: fitted vs residual
      xDecompVecPlot <- temp[[sid]]$plot6data$xDecompVecPlot
      p6 <- qplot(x = .data$predicted, y = .data$actual - .data$predicted, data = xDecompVecPlot) +
        geom_hline(yintercept = 0) +
        geom_smooth(se = TRUE, method = "loess", formula = "y ~ x") +
        xlab("Fitted") + ylab("Residual") + ggtitle("Fitted vs. Residual")

      ## Aggregate one-pager plots and export
      onepagerTitle <- paste0("Model one-pager, on pareto front ", pf, ", ID: ", sid)
      pg <- wrap_plots(p2, p5, p1, p4, p3, p6, ncol = 2) +
        plot_annotation(title = onepagerTitle, theme = theme(plot.title = element_text(hjust = 0.5)))
      all_plots[[sid]] <- pg

      if (export) {
        ggsave(
          filename = paste0(OutputCollect$plot_folder, "/", sid, ".png"),
          plot = pg,
          dpi = 600, width = 18, height = 18
        )
      }
      if (check_parallel_plot() & !quiet & count_mod_out > 0) {
        cnt <- cnt + 1
        setTxtProgressBar(pbplot, cnt)
      }
    }
    if (!quiet & count_mod_out > 0) {
      cnt <- cnt + length(uniqueSol)
      setTxtProgressBar(pbplot, cnt)
    }
  }
  if (!quiet & count_mod_out > 0) close(pbplot)
  # Stop cluster to avoid memory leaks
  if (check_parallel_plot()) stopImplicitCluster()
  return(invisible(all_plots))

}
