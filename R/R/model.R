# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_run, robyn_mmm, model_refit, model_decomp, calibrate_mmm, ridge_lambda

####################################################################
#' The major Robyn modelling function
#'
#' The \code{robyn_run()} function consumes output from \code{robyn_input()},
#' runs the \code{robyn_mmm()} functions and plots and collects the result.
#'
#' @inheritParams robyn_allocator
#' @param plot_folder Character. Path for saving plots. Default
#' to \code{robyn_object} and saves plot in the same directory as \code{robyn_object}.
#' @param plot_folder_sub Character. Customize sub path to save plots. The total
#' path is created with \code{dir.create(file.path(plot_folder, plot_folder_sub))}.
#' For example, plot_folder_sub = "sub_dir".
#' @param dt_hyper_fixed data.frame. Only provide when loading old model results.
#' It consumes hyperparameters from saved csv \code{pareto_hyperparameters.csv}.
#' @param pareto_fronts Integer. Number of Pareto fronts for the output.
#' \code{pareto_fronts = 1} returns the best models trading off \code{NRMSE} &
#' \code{DECOMP.RSSD}. Increase \code{pareto_fronts} to get more model choices.
#' @param plot_pareto Boolean. Set to \code{FALSE} to deactivate plotting
#' and saving model one-pagers. Used when testing models.
#' @param calibration_constraint Numeric. Default to 0.1 and allows 0.01-0.1. When
#' calibrating, 0.1 means top 10% calibrated models are used for pareto-optimal
#' selection. Lower \code{calibration_constraint} increases calibration accuracy.
#' @param lambda_control Numeric. From 0-1. Tunes ridge lambda between
#' lambda.min and lambda.1se.
#' @param refresh Boolean. Set to \code{TRUE} when used in \code{robyn_refresh()}.
#' @param seed Integer. For reproducible results when running nevergrad.
#' @param csv_out Character. Accepts "pareto" or "all". Default to "pareto". Set
#' to "all" will output all iterations as csv.
#' @param ui Boolean. Save additional outputs for UI usage. List outcome.
#' @examples
#' \dontrun{
#' OutputCollect <- robyn_run(
#'   InputCollect = InputCollect,
#'   plot_folder = robyn_object,
#'   pareto_fronts = 3,
#'   plot_pareto = TRUE
#' )
#' }
#' @export
robyn_run <- function(InputCollect,
                      plot_folder = getwd(),
                      plot_folder_sub = NULL,
                      pareto_fronts = 1,
                      plot_pareto = TRUE,
                      calibration_constraint = 0.1,
                      lambda_control = 1,
                      refresh = FALSE,
                      dt_hyper_fixed = NULL,
                      seed = 123L,
                      csv_out = "pareto",
                      ui = FALSE) {

  #####################################
  #### Set local environment

  if (!"hyperparameters" %in% names(InputCollect)) {
    stop("Must provide 'hyperparameters' in robyn_inputs()'s output first")
  }

  t0 <- Sys.time()

  # check path
  check_robyn_object(plot_folder)
  plot_folder <- check_filedir(plot_folder)

  dt_mod <- copy(InputCollect$dt_mod)
  dt_modRollWind <- copy(InputCollect$dt_modRollWind)

  message(
    "Input data has ", nrow(dt_mod), " ", InputCollect$intervalType, "s in total: ",
    dt_mod[, min(ds)], " to ", dt_mod[, max(ds)]
  )
  message(
    ifelse(!refresh, "Initial", "Refresh"), " model is built on rolling window of ", InputCollect$rollingWindowLength, " ", InputCollect$intervalType, "s: ",
    InputCollect$window_start, " to ", InputCollect$window_end
  )
  if (refresh) {
    message("Rolling window moving forward: ", InputCollect$refresh_steps, " ", InputCollect$intervalType)
  }

  calibration_constraint <- check_calibconstr(calibration_constraint, InputCollect$iterations, InputCollect$trials, InputCollect$calibration_input)

  #####################################
  #### Run robyn_mmm on set_trials

  hyper_fixed <- all(sapply(InputCollect$hyperparameters, length) == 1)
  if (hyper_fixed & is.null(dt_hyper_fixed)) {
    stop("hyperparameters can't be all fixed for hyperparameter optimisation. If you want to get old model result, please provide only 1 model / 1 row from OutputCollect$resultHypParam or pareto_hyperparameters.csv from previous runs")
  }
  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)

  if (!is.null(dt_hyper_fixed)) {

    ## Run robyn_mmm if using old model result tables
    dt_hyper_fixed <- as.data.table(dt_hyper_fixed)
    if (nrow(dt_hyper_fixed) != 1) {
      stop("Provide only 1 model / 1 row from OutputCollect$resultHypParam or pareto_hyperparameters.csv from previous runs")
    }
    if (!all(c(hypParamSamName, "lambda") %in% names(dt_hyper_fixed))) {
      stop("dt_hyper_fixed is provided with wrong input. please provide the table OutputCollect$resultHypParam from previous runs or pareto_hyperparameters.csv with desired model ID")
    }

    hyper_fixed <- TRUE
    hyperparameters_fixed <- lapply(dt_hyper_fixed[, hypParamSamName, with = FALSE], unlist)

    model_output_collect <- list()
    model_output_collect[[1]] <- robyn_mmm(
      hyper_collect = hyperparameters_fixed,
      InputCollect = InputCollect,
      # ,iterations = iterations
      # ,cores = cores
      # ,optimizer_name = InputCollect$nevergrad_algo
      lambda_fixed = dt_hyper_fixed$lambda,
      seed = seed
    )

    model_output_collect[[1]]$trial <- 1
    model_output_collect[[1]]$resultCollect$resultHypParam <- model_output_collect[[1]]$resultCollect$resultHypParam[order(iterPar)]

    dt_IDs <- data.table(
      solID = dt_hyper_fixed$solID,
      iterPar = model_output_collect[[1]]$resultCollect$resultHypParam$iterPar
    )

    model_output_collect[[1]]$resultCollect$resultHypParam[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$xDecompAgg[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$xDecompVec[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
    model_output_collect[[1]]$resultCollect$decompSpendDist[dt_IDs, on = .(iterPar), "solID" := .(i.solID)]
  } else {

    ## Run robyn_mmm on set_trials if hyperparameters are not all fixed

    t0 <- Sys.time()

    # enable parallelisation of main modelling loop for MacOS and Linux only
    parallel_processing <- .Platform$OS.type == "unix"
    if (parallel_processing) {
      message(paste(
        "Using", InputCollect$adstock, "adstocking with",
        length(InputCollect$hyperparameters),
        "hyperparameters & 10-fold ridge x-validation on",
        InputCollect$cores, "cores"
      ))
    } else {
      message(paste(
        "Using", InputCollect$adstock, "adstocking with",
        length(InputCollect$hyperparameters),
        "hyperparameters & 10-fold ridge x-validation on 1 core (Windows fallback)"
      ))
    }

    # ng_collect <- list()
    model_output_collect <- list()

    message(paste(
      ">>> Start running", InputCollect$trials, "trials with",
      InputCollect$iterations, "iterations per trial each",
      ifelse(is.null(InputCollect$calibration_input), "with", "with calibration and"),
      InputCollect$nevergrad_algo, "nevergrad algorithm..."
    ))

    for (ngt in 1:InputCollect$trials) {
      message(paste(" Running trial nr.", ngt))
      model_output <- robyn_mmm(
        hyper_collect = InputCollect$hyperparameters,
        InputCollect = InputCollect,
        lambda_control = lambda_control,
        refresh = refresh,
        seed = seed
      )

      check_coef0 <- any(model_output$resultCollect$decompSpendDist$decomp.rssd == Inf)
      if (check_coef0) {
        num_coef0_mod <- model_output$resultCollect$decompSpendDist[decomp.rssd == Inf, uniqueN(paste0(iterNG, "_", iterPar))]
        num_coef0_mod <- ifelse(num_coef0_mod > InputCollect$iterations, InputCollect$iterations, num_coef0_mod)
        message("This trial contains ", num_coef0_mod, " iterations with all 0 media coefficient. Please reconsider your media variable choice if the pareto choices are unreasonable.
                  \nRecommendations are: \n1. increase hyperparameter ranges for 0-coef channels to give Robyn more freedom\n2. split media into sub-channels, and/or aggregate similar channels, and/or introduce other media\n3. increase trials to get more samples\n")
      }
      model_output["trial"] <- ngt
      model_output_collect[[ngt]] <- model_output
    }
    # ng_collect <- rbindlist(ng_collect)
    # px <- low(ng_collect$nrmse) * low(ng_collect$decomp.rssd)
    # ng_collect <- psel(ng_collect, px, top = nrow(ng_collect))[order(trial, nrmse)]
    # ng_out[[which(ng_algos==optmz)]] <- ng_collect
    # }
    # ng_out <- rbindlist(ng_out)
    # setnames(ng_out, ".level", "manual_pareto")
  }

  #####################################
  #### Collect results for plotting

  message(">>> Collecting results...")

  ## collect hyperparameter results
  if (hyper_fixed) {
    names(model_output_collect) <- "trial1"
  } else {
    names(model_output_collect) <- paste0("trial", 1:InputCollect$trials)
  }

  resultHypParam <- rbindlist(lapply(model_output_collect, function(x) x$resultCollect$resultHypParam[, trial := x$trial]))
  resultHypParam[, iterations := (iterNG - 1) * InputCollect$cores + iterPar]
  xDecompAgg <- rbindlist(lapply(model_output_collect, function(x) x$resultCollect$xDecompAgg[, trial := x$trial]))
  xDecompAgg[, iterations := (iterNG - 1) * InputCollect$cores + iterPar]

  # if (hyper_fixed == FALSE) {
  resultHypParam[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  xDecompAgg[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  # }
  xDecompAggCoef0 <- xDecompAgg[rn %in% InputCollect$paid_media_vars, .(coef0 = min(coef) == 0), by = "solID"]

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

  decompSpendDist <- rbindlist(lapply(model_output_collect, function(x) x$resultCollect$decompSpendDist[, trial := x$trial]))
  decompSpendDist <- decompSpendDist[resultHypParam, robynPareto := i.robynPareto, on = c("iterNG", "iterPar", "trial")]
  if (hyper_fixed == FALSE) {
    decompSpendDist[, solID := (paste(trial, iterNG, iterPar, sep = "_"))]
  } else {
    xDecompAgg[, solID := unique(decompSpendDist$solID)]
    resultHypParam[, solID := unique(decompSpendDist$solID)]
  }
  #decompSpendDist <- decompSpendDist[xDecompAgg[rn %in% InputCollect$paid_media_vars, .(rn, xDecompAgg, solID)], on = c("rn", "solID")]

  ## get mean_response
  if (parallel_processing) {
    registerDoParallel(InputCollect$cores)
  } else {
    registerDoSEQ()
  }

  # if (hyper_fixed == FALSE) {pb <- txtProgressBar(min=1, max = length(decompSpendDist$rn), style = 3)}
  pareto_fronts_vec <- 1:pareto_fronts
  decompSpendDistPar <- decompSpendDist[robynPareto %in% pareto_fronts_vec]
  resultHypParamPar <- resultHypParam[robynPareto %in% pareto_fronts_vec]
  xDecompAggPar <- xDecompAgg[robynPareto %in% pareto_fronts_vec]
  resp_collect <- foreach(
    respN = seq_along(decompSpendDistPar$rn)
    , .combine = rbind) %dorng% {
      get_resp <- robyn_response(
        paid_media_var = decompSpendDistPar$rn[respN],
        select_model = decompSpendDistPar[respN, solID],
        spend = decompSpendDistPar[respN, mean_spend],
        dt_hyppar = resultHypParamPar,
        dt_coef = xDecompAggPar,
        InputCollect = InputCollect
      )
      #if (hyper_fixed == FALSE) setTxtProgressBar(pb, n)
      dt_resp <- data.table(mean_response = get_resp
                            ,rn = decompSpendDistPar$rn[respN]
                            ,solID = decompSpendDistPar$solID[respN])
      return(dt_resp)
    }
  #if (hyper_fixed == FALSE) close(pb)
  stopImplicitCluster()
  registerDoSEQ()
  getDoParWorkers()

  setkey(decompSpendDist, solID, rn)
  setkey(resp_collect, solID, rn)
  decompSpendDist <- merge(decompSpendDist, resp_collect, all.x=TRUE)
  #decompSpendDist[, mean_response := resp_collect]
  decompSpendDist[, ":="(
    roi_mean = mean_response / mean_spend,
    roi_total = xDecompAgg / total_spend,
    cpa_mean = mean_spend / mean_response,
    cpa_total = total_spend / xDecompAgg
  )]
  # decompSpendDist[, roi := xDecompMeanNon0/mean_spend]

  setkey(xDecompAgg, solID, rn)
  setkey(decompSpendDist, solID, rn)
  xDecompAgg <- merge(xDecompAgg, decompSpendDist[, .(rn, solID, total_spend, mean_spend, spend_share, effect_share, roi_mean, roi_total, cpa_total)], all.x = TRUE)


  #####################################
  #### Plot overview

  ## set folder to save plat
  if (is.null(plot_folder_sub)) {
    folder_var <- ifelse(!refresh, "init", paste0("rf", InputCollect$refreshCounter))
    plot_folder_sub <- paste0(format(Sys.time(), "%Y-%m-%d %H.%M"), " ", folder_var)
  }
  plotPath <- dir.create(file.path(plot_folder, plot_folder_sub))

  # pareto_fronts_vec <- ifelse(!hyper_fixed, c(1,2,3), 1)
  if (!hyper_fixed) {
    pareto_fronts_vec <- 1:pareto_fronts
    num_pareto123 <- resultHypParam[robynPareto %in% pareto_fronts_vec, .N]
  } else {
    pareto_fronts_vec <- 1
    num_pareto123 <- nrow(resultHypParam)
  }

  message(paste0(">>> Exporting all charts into directory: ", plot_folder, "/", plot_folder_sub, "..."))

  message(">>> Plotting summary charts...")
  local_name <- names(InputCollect$hyperparameters)
  if (!hyper_fixed) {

    ## plot prophet

    if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0
      || !is.null(InputCollect$factor_vars) && length(InputCollect$factor_vars) > 0)
    {
      # pProphet <- prophet_plot_components(InputCollect$modelRecurrence, InputCollect$forecastRecurrence, render_plot = TRUE)

      dt_plotProphet <- InputCollect$dt_mod[, c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$factor_vars), with = FALSE]
      dt_plotProphet <- suppressWarnings(melt.data.table(dt_plotProphet, id.vars = "ds"))
      pProphet <- ggplot(dt_plotProphet, aes(x = ds, y = value)) +
        geom_line(color = "steelblue") +
        facet_wrap(~variable, scales = "free", ncol = 1) +
        labs(title = "Prophet decomposition") +
        xlab(NULL) +
        ylab(NULL)
      # print(pProphet)
      ggsave(paste0(plot_folder, "/", plot_folder_sub, "/", "prophet_decomp.png"),
             plot = pProphet,
             dpi = 600, width = 12, height = 3 * length(levels(dt_plotProphet$variable))
      )
    }


    ## plot spend exposure model

    if (any(InputCollect$costSelector)) {
      pSpendExposure <- wrap_plots(
        InputCollect$plotNLSCollect,
        ncol = ifelse(length(InputCollect$plotNLSCollect) <= 3, length(InputCollect$plotNLSCollect), 3)
      ) +
        plot_annotation(
          title = "Spend-exposure fitting with Michaelis-Menten model",
          theme = theme(plot.title = element_text(hjust = 0.5))
        )
      ggsave(paste0(plot_folder, "/", plot_folder_sub, "/", "spend_exposure_fitting.png"),
             plot = pSpendExposure,
             dpi = 600, width = 12, height = ceiling(length(InputCollect$plotNLSCollect) / 3) * 7
      )
    } else {
      message("No spend-exposure modelling needed. all media variables used for mmm are spend variables ")
    }

    ## plot hyperparameter sampling distribution
    resultHypParam.melted <- melt.data.table(resultHypParam[, c(local_name, "robynPareto"), with = FALSE], id.vars = c("robynPareto"))

    pSamp <- ggplot(data = resultHypParam.melted, aes(x = value, y = variable, color = variable, fill = variable)) +
      geom_violin(alpha = .5, size = 0) +
      geom_point(size = 0.2) +
      theme(legend.position = "none") +
      labs(
        title = "Hyperparameter optimisation sampling",
        subtitle = paste0("Sample distribution", ", iterations = ", InputCollect$iterations, " * ", InputCollect$trials, " trial"),
        x = "Hyperparameter space",
        y = ""
      )
    # print(pSamp)
    ggsave(paste0(plot_folder, "/", plot_folder_sub, "/", "hypersampling.png"),
           plot = pSamp,
           dpi = 600, width = 12, height = 7
    )


    ## plot Pareto front
    if (!is.null(InputCollect$calibration_input)) {
      resultHypParam[, iterations := ifelse(is.na(robynPareto), NA, iterations)]
    }
    pParFront <- ggplot(data = resultHypParam, aes(x = nrmse, y = decomp.rssd, color = iterations)) +
      geom_point(size = 0.5) +
      # stat_smooth(data = resultHypParam, method = 'gam', formula = y ~ s(x, bs = "cs"), size = 0.2, fill = "grey100", linetype="dashed")+
      geom_line(data = resultHypParam[robynPareto == 1], aes(x = nrmse, y = decomp.rssd), colour = "coral4") +
      # geom_line(data = resultHypParam[robynPareto ==2], aes(x=nrmse, y=decomp.rssd), colour = "coral3")+
      # geom_line(data = resultHypParam[robynPareto ==3], aes(x=nrmse, y=decomp.rssd), colour = "coral")+
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

    # print(pParFront)
    ggsave(paste0(plot_folder, "/", plot_folder_sub, "/", "pareto_front.png"),
           plot = pParFront,
           dpi = 600, width = 12, height = 7
    )


    ## plot ridgeline model convergence
    dt_ridges <- xDecompAgg[rn %in% InputCollect$paid_media_vars
                            , .(variables = rn
                                , roi_total
                                , iteration = (iterNG-1)*InputCollect$cores+iterPar
                                , trial)][order(iteration, variables)]
    bin_limits <- c(1,20)
    qt_len <- ifelse(InputCollect$iterations <=100, 1
                     ,ifelse(InputCollect$iterations > 2000, 20, ceiling(InputCollect$iterations/100)))
    set_qt <- floor(quantile(1:InputCollect$iterations, seq(0, 1, length.out = qt_len+1)))
    set_bin <- set_qt[-1]

    dt_ridges[, iter_bin := cut(dt_ridges$iteration, breaks = set_qt, labels = set_bin)]
    dt_ridges <- dt_ridges[!is.na(iter_bin)]
    dt_ridges[, iter_bin := factor(iter_bin, levels = sort(set_bin, decreasing = TRUE))]
    dt_ridges[, trial := as.factor(trial)]

    pRidges <- ggplot(data = dt_ridges, aes(x = roi_total, y = iter_bin, fill = as.integer(iter_bin), linetype = trial)) +
      scale_fill_distiller(palette = "GnBu") +
      geom_density_ridges(scale = 4, col = "white", quantile_lines = TRUE, quantiles = 2, alpha = 0.7) +
      facet_wrap(~ variables, scales = "free") +
      guides(fill = "none")+
      theme(panel.background = element_blank()) +
      labs(x = "Total ROAS", y = "Iteration Bucket"
           ,title = "ROAS distribution over iteration"
           ,fill = "iter bucket")

    suppressMessages(ggsave(paste0(plot_folder, "/", plot_folder_sub, "/", "roas_convergence.png"),
                            plot = pRidges,
                            dpi = 600, width = 12, height = ceiling(InputCollect$mediaVarCount / 3) * 6
    ))
  }


  #####################################
  #### Plot each pareto solution

  # ggplot doesn't work with process forking on MacOS
  # however it works fine on Linux and Windows
  parallel_plotting <- Sys.info()["sysname"] != "Darwin"

  if (plot_pareto) {
    if (parallel_plotting) {
      message(paste(">>> Plotting", num_pareto123, "Pareto optimum models on", InputCollect$cores, "cores..."))
    } else {
      message(paste(">>> Plotting", num_pareto123, "Pareto optimum models on 1 core (MacOS fallback)..."))
    }
  }

  if (parallel_plotting) {
    registerDoParallel(InputCollect$cores)
  } else {
    registerDoSEQ()
  }

  all_fronts <- unique(xDecompAgg$robynPareto)
  all_fronts <- sort(all_fronts[!is.na(all_fronts)])
  if (!all(pareto_fronts_vec %in% all_fronts)) {
    pareto_fronts_vec <- all_fronts
  }

  cnt <- 0
  pbplot <- txtProgressBar(max = num_pareto123, style = 3)

  mediaVecCollect <- list()
  xDecompVecCollect <- list()
  meanResponseCollect <- list()
  for (pf in pareto_fronts_vec) {
    plotMediaShare <- xDecompAgg[robynPareto == pf & rn %in% InputCollect$paid_media_vars]
    plotWaterfall <- xDecompAgg[robynPareto == pf]
    uniqueSol <- plotMediaShare[, unique(solID)]

    parallelResult <- foreach(sid = uniqueSol) %dorng% {

      ## plot spend x effect share comparison
      plotMediaShareLoop <- plotMediaShare[solID == sid]
      rsq_train_plot <- plotMediaShareLoop[, round(unique(rsq_train), 4)]
      nrmse_plot <- plotMediaShareLoop[, round(unique(nrmse), 4)]
      decomp_rssd_plot <- plotMediaShareLoop[, round(unique(decomp.rssd), 4)]
      mape_lift_plot <- ifelse(!is.null(InputCollect$calibration_input), plotMediaShareLoop[, round(unique(mape), 4)], NA)

      suppressWarnings(plotMediaShareLoop <- melt.data.table(plotMediaShareLoop, id.vars = c("rn", "nrmse", "decomp.rssd", "rsq_train"), measure.vars = c("spend_share", "effect_share", "roi_total", "cpa_total")))
      plotMediaShareLoop[, rn := factor(rn, levels = sort(InputCollect$paid_media_vars))]
      plotMediaShareLoopBar <- plotMediaShareLoop[variable %in% c("spend_share", "effect_share")]
      # plotMediaShareLoopBar[, variable:= ifelse(variable=="spend_share", "total spend share", "total effect share")]
      plotMediaShareLoopLine <- plotMediaShareLoop[variable == ifelse(InputCollect$dep_var_type == "conversion", "cpa_total", "roi_total")]
      # plotMediaShareLoopLine[, variable:= "roi_total"]
      line_rm_inf <- !is.infinite(plotMediaShareLoopLine$value)
      ySecScale <- max(plotMediaShareLoopLine$value[line_rm_inf]) / max(plotMediaShareLoopBar$value) * 1.1

      p1 <- ggplot(plotMediaShareLoopBar, aes(x = rn, y = value, fill = variable)) +
        geom_bar(stat = "identity", width = 0.5, position = "dodge") +
        geom_text(aes(label = paste0(round(value * 100, 2), "%")), color = "darkblue", position = position_dodge(width = 0.5), fontface = "bold") +
        geom_line(data = plotMediaShareLoopLine, aes(x = rn, y = value / ySecScale, group = 1, color = variable), inherit.aes = FALSE) +
        geom_point(data = plotMediaShareLoopLine, aes(x = rn, y = value / ySecScale, group = 1, color = variable), inherit.aes = FALSE, size = 4) +
        geom_text(
          data = plotMediaShareLoopLine, aes(label = round(value, 2), x = rn, y = value / ySecScale, group = 1, color = variable),
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
            ", mape.lift = ", mape_lift_plot
          ),
          y = "", x = ""
        )

      ## plot waterfall
      plotWaterfallLoop <- plotWaterfall[solID == sid][order(xDecompPerc)]
      plotWaterfallLoop[, end := cumsum(xDecompPerc)]
      plotWaterfallLoop[, end := 1 - end]
      plotWaterfallLoop[, ":="(start = shift(end, fill = 1, type = "lag"),
                               id = 1:nrow(plotWaterfallLoop),
                               rn = as.factor(rn),
                               sign = as.factor(ifelse(xDecompPerc >= 0, "pos", "neg")))]

      p2 <- suppressWarnings(
        ggplot(plotWaterfallLoop, aes(x = id, fill = sign)) +
          geom_rect(aes(x = rn, xmin = id - 0.45, xmax = id + 0.45, ymin = end, ymax = start), stat = "identity") +
          scale_x_discrete("", breaks = levels(plotWaterfallLoop$rn), labels = plotWaterfallLoop$rn) +
          theme(axis.text.x = element_text(angle = 65, vjust = 0.6), legend.position = c(0.1, 0.1)) +
          geom_text(mapping = aes(
            label = paste0(format_unit(xDecompAgg), "\n", round(xDecompPerc * 100, 2), "%"),
            y = rowSums(cbind(end, xDecompPerc / 2))
          ), fontface = "bold") +
          coord_flip() +
          labs(
            title = "Response decomposition waterfall by predictor",
            subtitle = paste0(
              "rsq_train: ", rsq_train_plot,
              ", nrmse = ", nrmse_plot,
              ", decomp.rssd = ", decomp_rssd_plot,
              ", mape.lift = ", mape_lift_plot
            ),
            x = "",
            y = ""
          ))

      ## plot adstock rate

      resultHypParamLoop <- resultHypParam[solID == sid]
      hypParam <- unlist(resultHypParamLoop[, local_name, with = FALSE])

      if (InputCollect$adstock == "geometric") {

        hypParam_thetas <- hypParam[paste0(InputCollect$all_media, "_thetas")]
        dt_geometric <- data.table(channels = InputCollect$all_media, thetas = hypParam_thetas)

        p3 <- ggplot(dt_geometric, aes(x = channels, y = thetas, fill = "coral")) +
          geom_bar(stat = "identity", width = 0.5) +
          theme(legend.position = "none") +
          coord_flip() +
          geom_text(aes(label = paste0(round(thetas * 100, 1), "%")), position = position_dodge(width = 0.5), fontface = "bold") +
          ylim(0, 1) +
          labs(
            title = "Geometric adstock - fixed decay rate over time",
            subtitle = paste0(
              "rsq_train: ", rsq_train_plot,
              ", nrmse = ", nrmse_plot,
              ", decomp.rssd = ", decomp_rssd_plot,
              ", mape.lift = ", mape_lift_plot
            ),
            y = "", x = ""
          )
      } else if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {

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

        p3 <- ggplot(weibullCollect, aes(x = x, y = decay_accumulated)) +
          geom_line(aes(color = channel)) +
          facet_wrap(~channel) +
          geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
          geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife"), colour = "gray") +
          theme(legend.position = "none") +
          labs(title = paste0("Weibull adstock ",toupper(wb_type)," - flexible decay rate over time"),
               subtitle = paste0(
                 "rsq_train: ", rsq_train_plot,
                 ", nrmse = ", nrmse_plot,
                 ", decomp.rssd = ", decomp_rssd_plot,
                 ", mape.lift = ", mape_lift_plot
               ),
               x = "time unit",
               y = "")
      }


      ## plot response curve

      dt_transformPlot <- dt_mod[, c("ds", InputCollect$all_media), with = FALSE] # independent variables
      dt_transformSpend <- cbind(dt_transformPlot[, .(ds)], InputCollect$dt_input[, c(InputCollect$paid_media_spends), with = FALSE]) # spends of indep vars
      setnames(dt_transformSpend, names(dt_transformSpend), c("ds", InputCollect$paid_media_vars))

      # update non-spend variables
      dt_transformSpendMod <- dt_transformPlot[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, c("ds", InputCollect$paid_media_vars), with = FALSE]
      if (length(InputCollect$exposureVarName) > 0) {
        for (expo in InputCollect$exposureVarName) {
          sel_nls <- ifelse(InputCollect$modNLSCollect[channel == expo, rsq_nls > rsq_lm], "nls", "lm")
          dt_transformSpendMod[, (expo) := InputCollect$yhatNLSCollect[channel == expo & models == sel_nls, yhat]]
        }
      }

      dt_transformAdstock <- copy(dt_transformPlot)
      dt_transformSaturation <- dt_transformPlot[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]
      # chnl_non_spend <- InputCollect$paid_media_vars[!(InputCollect$paid_media_vars==InputCollect$paid_media_spends)]

      m_decayRate <- list()

      for (med in 1:length(InputCollect$all_media)) {
        med_select <- InputCollect$all_media[med]
        m <- dt_transformPlot[, get(med_select)]

        ## adstocking
        if (InputCollect$adstock == "geometric") {
          theta <- hypParam[paste0(InputCollect$all_media[med], "_thetas")]
          x_list <- adstock_geometric(x = m, theta = theta)
        } else if (InputCollect$adstock == "weibull_cdf") {
          shape <- hypParam[paste0(InputCollect$all_media[med], "_shapes")]
          scale <- hypParam[paste0(InputCollect$all_media[med], "_scales")]
          x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "cdf")
        } else if (InputCollect$adstock == "weibull_pdf") {
          shape <- hypParam[paste0(InputCollect$all_media[med], "_shapes")]
          scale <- hypParam[paste0(InputCollect$all_media[med], "_scales")]
          x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "pdf")
        }
        m_adstocked <- x_list$x_decayed
        dt_transformAdstock[, (med_select) := m_adstocked]
        m_adstockedRollWind <- m_adstocked[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]

        ## saturation
        alpha <- hypParam[paste0(InputCollect$all_media[med], "_alphas")]
        gamma <- hypParam[paste0(InputCollect$all_media[med], "_gammas")]
        dt_transformSaturation[, (med_select) := saturation_hill(x = m_adstockedRollWind, alpha = alpha, gamma = gamma)]
      }

      dt_transformSaturationDecomp <- copy(dt_transformSaturation)
      for (i in 1:InputCollect$mediaVarCount) {
        coef <- plotWaterfallLoop[rn == InputCollect$all_media[i], coef]
        dt_transformSaturationDecomp[, (InputCollect$all_media[i]) := .SD * coef, .SDcols = InputCollect$all_media[i]]
      }

      # mediaAdstockFactorPlot <- dt_transformPlot[, lapply(.SD, sum), .SDcols = InputCollect$paid_media_vars]  / dt_transformAdstock[, lapply(.SD, sum), .SDcols = InputCollect$paid_media_vars]
      # dt_transformSaturationAdstockReverse <- data.table(mapply(function(x, y) {x*y},x= dt_transformAdstock[, InputCollect$paid_media_vars, with=FALSE], y= mediaAdstockFactorPlot))
      dt_transformSaturationSpendReverse <- copy(dt_transformAdstock[, c("ds", InputCollect$all_media), with = FALSE])

      for (i in 1:InputCollect$mediaVarCount) {
        chn <- InputCollect$paid_media_vars[i]
        if (chn %in% InputCollect$paid_media_vars[InputCollect$costSelector]) {

          # get Michaelis Menten nls fitting param
          get_chn <- dt_transformSaturationSpendReverse[, chn, with = FALSE]
          Vmax <- InputCollect$modNLSCollect[channel == chn, Vmax]
          Km <- InputCollect$modNLSCollect[channel == chn, Km]

          # reverse exposure to spend
          dt_transformSaturationSpendReverse[, (chn) := mic_men(x = .SD, Vmax = Vmax, Km = Km, reverse = TRUE), .SDcols = chn] # .SD * Km / (Vmax - .SD) exposure to spend, reverse Michaelis Menthen: x = y*Km/(Vmax-y)
        } else if (chn %in% InputCollect$exposureVarName) {
          coef_lm <- InputCollect$modNLSCollect[channel == chn, coef_lm]
          dt_transformSaturationSpendReverse[, (chn) := .SD / coef_lm, .SDcols = chn]
        }
        # spendRatioFitted <- xDecompAgg[rn == chn, mean(total_spend)] / dt_transformSaturationSpendReverse[, sum(.SD), .SDcols = chn]
        # dt_transformSaturationSpendReverse[, (chn):= .SD * spendRatioFitted, .SDcols = chn]
      }

      dt_transformSaturationSpendReverse <- dt_transformSaturationSpendReverse[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]

      dt_scurvePlot <- cbind(
        melt.data.table(dt_transformSaturationDecomp[, c("ds", InputCollect$all_media), with = FALSE], id.vars = "ds", variable.name = "channel", value.name = "response"),
        melt.data.table(dt_transformSaturationSpendReverse, id.vars = "ds", value.name = "spend")[, .(spend)]
      )
      dt_scurvePlot <- dt_scurvePlot[spend >= 0] # remove outlier introduced by MM nls fitting


      dt_scurvePlotMean <- dt_transformSpend[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, !"ds"][, lapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0]))), .SDcols = InputCollect$paid_media_vars]
      dt_scurvePlotMean <- melt.data.table(dt_scurvePlotMean, measure.vars = InputCollect$paid_media_vars, value.name = "mean_spend", variable.name = "channel")
      dt_scurvePlotMean[, ":="(mean_spend_scaled = 0, mean_response = 0, next_unit_response = 0)]

      for (med in 1:InputCollect$mediaVarCount) {
        get_med <- InputCollect$paid_media_vars[med]
        get_spend <- dt_scurvePlotMean[channel == get_med, mean_spend]

        if (get_med %in% InputCollect$paid_media_vars[InputCollect$costSelector]) {
          Vmax <- InputCollect$modNLSCollect[channel == get_med, Vmax]
          Km <- InputCollect$modNLSCollect[channel == get_med, Km]
          get_spend_mm <- mic_men(x = get_spend, Vmax = Vmax, Km = Km) # Vmax * get_spend/(Km + get_spend)
        } else if (get_med %in% InputCollect$exposureVarName) {
          coef_lm <- InputCollect$modNLSCollect[channel == get_med, coef_lm]
          get_spend_mm <- get_spend * coef_lm
        } else {
          get_spend_mm <- get_spend
        }

        m <- dt_transformAdstock[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich, get(get_med)]
        # m <- m[m>0] # remove outlier introduced by MM nls fitting
        alpha <- hypParam[which(paste0(get_med, "_alphas") == names(hypParam))]
        gamma <- hypParam[which(paste0(get_med, "_gammas") == names(hypParam))]
        # gammaTrans <- round(quantile(seq(range(m)[1], range(m)[2], length.out = 100), gamma),4)
        # get_response <-  get_spend_mm**alpha / (get_spend_mm**alpha + gammaTrans**alpha)
        # get_response_marginal <- (get_spend_mm+1)**alpha / ((get_spend_mm+1)**alpha + gammaTrans**alpha)
        get_response <- saturation_hill(x = m, alpha = alpha, gamma = gamma, x_marginal = get_spend_mm)
        get_response_marginal <- saturation_hill(x = m, alpha = alpha, gamma = gamma, x_marginal = get_spend_mm + 1)

        coef <- plotWaterfallLoop[rn == get_med, coef]
        dt_scurvePlotMean[channel == get_med, mean_spend_scaled := get_spend_mm]
        dt_scurvePlotMean[channel == get_med, mean_response := get_response * coef]
        dt_scurvePlotMean[channel == get_med, next_unit_response := get_response_marginal * coef - mean_response]
      }
      dt_scurvePlotMean[, solID := sid]

      p4 <- ggplot(data = dt_scurvePlot[channel %in% InputCollect$paid_media_vars], aes(x = spend, y = response, color = channel)) +
        geom_line() +
        geom_point(data = dt_scurvePlotMean, aes(x = mean_spend, y = mean_response, color = channel)) +
        geom_text(data = dt_scurvePlotMean, aes(x = mean_spend, y = mean_response, label = round(mean_spend, 0)), show.legend = FALSE, hjust = -0.2) +
        theme(legend.position = c(0.9, 0.2)) +
        labs(
          title = "Response curve and mean spend by channel",
          subtitle = paste0(
            "rsq_train: ", rsq_train_plot,
            ", nrmse = ", nrmse_plot,
            ", decomp.rssd = ", decomp_rssd_plot,
            ", mape.lift = ", mape_lift_plot
          ),
          x = "Spend", y = "response"
        )

      ## plot fitted vs actual

      if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0) {
        dt_transformDecomp <- cbind(dt_modRollWind[, c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$context_vars), with = FALSE], dt_transformSaturation[, InputCollect$all_media, with = FALSE])
      } else {
        dt_transformDecomp <- cbind(dt_modRollWind[, c("ds", "dep_var", InputCollect$context_vars), with = FALSE], dt_transformSaturation[, InputCollect$all_media, with = FALSE])
      }
      col_order <- c("ds", "dep_var", InputCollect$all_ind_vars)
      setcolorder(dt_transformDecomp, neworder = col_order)

      xDecompVec <- dcast.data.table(xDecompAgg[solID == sid, .(rn, coef, solID)], solID ~ rn, value.var = "coef")
      if (!("(Intercept)" %in% names(xDecompVec))) {
        xDecompVec[, "(Intercept)" := 0]
      }
      setcolorder(xDecompVec, neworder = c("solID", "(Intercept)", col_order[!(col_order %in% c("ds", "dep_var"))]))
      intercept <- xDecompVec$`(Intercept)`

      xDecompVec <- data.table(mapply(function(scurved, coefs) {
        scurved * coefs
      },
      scurved = dt_transformDecomp[, !c("ds", "dep_var"), with = FALSE],
      coefs = xDecompVec[, !c("solID", "(Intercept)")]
      ))
      xDecompVec[, intercept := intercept]
      xDecompVec[, ":="(depVarHat = rowSums(xDecompVec), solID = sid)]
      xDecompVec <- cbind(dt_transformDecomp[, .(ds, dep_var)], xDecompVec)

      xDecompVecPlot <- xDecompVec[, .(ds, dep_var, depVarHat)]
      setnames(xDecompVecPlot, old = c("ds", "dep_var", "depVarHat"), new = c("ds", "actual", "predicted"))
      suppressWarnings(xDecompVecPlotMelted <- melt.data.table(xDecompVecPlot, id.vars = "ds"))

      p5 <- ggplot(xDecompVecPlotMelted, aes(x = ds, y = value, color = variable)) +
        geom_line() +
        theme(legend.position = c(0.9, 0.9)) +
        labs(
          title = "Actual vs. predicted response",
          subtitle = paste0(
            "rsq_train: ", rsq_train_plot,
            ", nrmse = ", nrmse_plot,
            ", decomp.rssd = ", decomp_rssd_plot,
            ", mape.lift = ", mape_lift_plot
          ),
          x = "date", y = "response"
        )

      ## plot diagnostic: fitted vs residual

      p6 <- qplot(x = predicted, y = actual - predicted, data = xDecompVecPlot) +
        geom_hline(yintercept = 0) +
        geom_smooth(se = TRUE, method = "loess", formula = "y ~ x") +
        xlab("fitted") + ylab("resid") + ggtitle("fitted vs. residual")

      ## save and aggregate one-pager plots

      onepagerTitle <- paste0("Model one-pager, on pareto front ", pf, ", ID: ", sid)

      pg <- wrap_plots(p2, p5, p1, p4, p3, p6, ncol = 2) +
        plot_annotation(title = onepagerTitle, theme = theme(plot.title = element_text(hjust = 0.5)))

      # pg <- arrangeGrob(p2,p5,p1, p4, p3, p6, ncol=2, top = text_grob(onepagerTitle, size = 15, face = "bold"))
      # grid.draw(pg)
      if (plot_pareto) {
        ggsave(
          filename = paste0(plot_folder, "/", plot_folder_sub, "/", sid, ".png"),
          plot = pg,
          dpi = 600, width = 18, height = 18
        )
      }

      ## prepare output
      if (!is.null(InputCollect$organic_vars) && length(InputCollect$organic_vars) > 0) {
        dt_transformSpend[, (InputCollect$organic_vars) := NA]
        dt_transformSpendMod[, (InputCollect$organic_vars) := NA]
        dt_transformSaturationSpendReverse[, (InputCollect$organic_vars) := NA]
      }

      if (!parallel_plotting) {
        cnt <- cnt + 1
        setTxtProgressBar(pbplot, cnt)
      }

      return(list(
        mediaVecCollect = rbind(
          dt_transformPlot[, ":="(type = "rawMedia", solID = sid)],
          dt_transformSpend[, ":="(type = "rawSpend", solID = sid)],
          dt_transformSpendMod[, ":="(type = "predictedExposure", solID = sid)],
          dt_transformAdstock[, ":="(type = "adstockedMedia", solID = sid)],
          dt_transformSaturation[, ":="(type = "saturatedMedia", solID = sid)],
          dt_transformSaturationSpendReverse[, ":="(type = "saturatedSpendReversed", solID = sid)],
          dt_transformSaturationDecomp[, ":="(type = "decompMedia", solID = sid)]
        ),
        xDecompVecCollect = xDecompVec,
        meanResponseCollect = dt_scurvePlotMean
      ))
    } # end solution loop

    cnt <- cnt + length(uniqueSol)
    setTxtProgressBar(pbplot, cnt)

    # append parallel run results
    mediaVecCollect <- append(mediaVecCollect, lapply(parallelResult, function (x) x$mediaVecCollect))
    xDecompVecCollect <- append(xDecompVecCollect, lapply(parallelResult, function (x) x$xDecompVecCollect))
    meanResponseCollect <- append(meanResponseCollect, lapply(parallelResult, function (x) x$meanResponseCollect))
  } # end pareto front loop

  close(pbplot)

  if (parallel_plotting) {
    # stop cluster to avoid memory leaks
    stopImplicitCluster()
  }

  mediaVecCollect <- rbindlist(mediaVecCollect)
  xDecompVecCollect <- rbindlist(xDecompVecCollect)
  meanResponseCollect <- rbindlist(meanResponseCollect)

  setnames(meanResponseCollect, old = "channel", new = "rn")
  setkey(meanResponseCollect, solID, rn)
  xDecompAgg <- merge(xDecompAgg, meanResponseCollect[, .(rn, solID, mean_response, next_unit_response)], all.x = TRUE)

  totalTime <- round(difftime(Sys.time(), t0, units = "mins"), 2)
  message(paste("\nTotal time:", totalTime, "mins"))

  #####################################
  #### Collect results for output

  allSolutions <- xDecompVecCollect[, unique(solID)]

  if (!csv_out %in% c("pareto", "all")) csv_out <- "pareto"
  if (csv_out == "pareto") {
    fwrite(resultHypParam[solID %in% allSolutions], paste0(plot_folder, "/", plot_folder_sub, "/", "pareto_hyperparameters.csv"))
    fwrite(xDecompAgg[solID %in% allSolutions], paste0(plot_folder, "/", plot_folder_sub, "/", "pareto_aggregated.csv"))
  } else if (csv_out == "all") {
    fwrite(resultHypParam, paste0(plot_folder, "/", plot_folder_sub, "/", "all_hyperparameters.csv"))
    fwrite(xDecompAgg, paste0(plot_folder, "/", plot_folder_sub, "/", "all_aggregated.csv"))
  }
  fwrite(mediaVecCollect, paste0(plot_folder, "/", plot_folder_sub, "/", "pareto_media_transform_matrix.csv"))
  fwrite(xDecompVecCollect, paste0(plot_folder, "/", plot_folder_sub, "/", "pareto_alldecomp_matrix.csv"))

  # For internal use -> UI Code
  if (ui) {
    UI <- list(pParFront = pParFront)
  } else UI <- NULL

  OutputCollect <- output <- list(
    resultHypParam = resultHypParam[solID %in% allSolutions],
    xDecompAgg = xDecompAgg[solID %in% allSolutions],
    mediaVecCollect = mediaVecCollect,
    xDecompVecCollect = xDecompVecCollect,
    UI = invisible(UI),
    model_output_collect = model_output_collect,
    allSolutions = allSolutions,
    totalTime = totalTime,
    plot_folder = paste0(plot_folder, "/", plot_folder_sub, "/")
  )
  return(output)
}


####################################################################
#' The core MMM function
#'
#' The \code{robyn_mmm()} function activates Nevergrad to generate samples of
#' hyperparameters, conducts media transformation within each loop, fits the
#' Ridge regression, calibrates the model optionally, decomposes responses
#' and collects the result. It's an inner function within \code{robyn_run()}.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @param hyper_collect List. Containing hyperparameter bounds. Defaults to
#' \code{InputCollect$hyperparameters}.
#' @param iterations Integer. Number of iterations to run.
#' @param lambda.n Integer. Number of lambda cross-validation in \code{glmnet}.
#' Defaults to 100.
#' @param lambda_fixed Boolean. \code{lambda_fixed = TRUE} when inputting
#' old model results.
#' @export
robyn_mmm <- function(hyper_collect,
                      InputCollect,
                      iterations = InputCollect$iterations,
                      lambda.n = 100,
                      lambda_control = 1,
                      lambda_fixed = NULL,
                      refresh = FALSE,
                      seed = 123L) {
  if (reticulate::py_module_available("nevergrad")) {
    ng <- reticulate::import("nevergrad", delay_load = TRUE)
    if (is.integer(seed)) {
      np <- reticulate::import("numpy", delay_load = FALSE)
      np$random$seed(seed)
    }
  } else {
    stop("You must have nevergrad python library installed.")
  }

  ################################################
  #### Collect hyperparameters

  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
  hyper_fixed <- FALSE

  # hyper_collect <- unlist(list(...), recursive = FALSE) # hyper_collect <- InputCollect$hyperparameters; hyper_collect <- hyperparameters_fixed

  # sort hyperparameter list by name
  hyper_bound_list <- list()
  for (i in 1:length(hypParamSamName)) {
    hyper_bound_list[i] <- hyper_collect[hypParamSamName[i]]
    names(hyper_bound_list)[i] <- hypParamSamName[i]
  }

  # get hyperparameters for Nevergrad
  hyper_which <- which(sapply(hyper_bound_list, length) == 2)
  hyper_bound_list_updated <- hyper_bound_list[hyper_which]
  hyper_bound_list_updated_name <- names(hyper_bound_list_updated)
  hyper_count <- length(hyper_bound_list_updated)
  if (hyper_count == 0) {
    hyper_fixed <- TRUE
    if (is.null(lambda_fixed)) {
      stop("when hyperparameters are fixed, lambda_fixed must be provided from the selected lambda in old model")
    }
  }

  # get fixed hyperparameters
  hyper_fixed_which <- which(sapply(hyper_bound_list, length) == 1)
  hyper_bound_list_fixed <- hyper_bound_list[hyper_fixed_which]
  hyper_bound_list_fixed_name <- names(hyper_bound_list_fixed)
  hyper_count_fixed <- length(hyper_bound_list_fixed)

  # hyper_bound_list_fixed <- list(print_S_alphas = 1 , print_S_gammas = 0.5)
  if (InputCollect$cores > 1) {
    dt_hyperFixed <- data.table(sapply(hyper_bound_list_fixed, function(x) rep(x, InputCollect$cores)))
  } else {
    dt_hyperFixed <- as.data.table(matrix(hyper_bound_list_fixed, nrow = 1))
    names(dt_hyperFixed) <- hyper_bound_list_fixed_name
  }

  ################################################
  #### Setup environment

  if (is.null(InputCollect$dt_mod)) {
    stop("Run InputCollect$dt_mod <- robyn_engineering() first to get the dt_mod")
  }

  ## get environment for parallel backend
  InputCollect <- InputCollect
  dt_mod <- copy(InputCollect$dt_mod)
  xDecompAggPrev <- InputCollect$xDecompAggPrev
  rollingWindowStartWhich <- InputCollect$rollingWindowStartWhich
  rollingWindowEndWhich <- InputCollect$rollingWindowEndWhich
  refreshAddedStart <- InputCollect$refreshAddedStart
  dt_modRollWind <- copy(InputCollect$dt_modRollWind)
  refresh_steps <- InputCollect$refresh_steps
  rollingWindowLength <- InputCollect$rollingWindowLength

  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  organic_vars <- InputCollect$organic_vars
  context_vars <- InputCollect$context_vars
  prophet_vars <- InputCollect$prophet_vars
  adstock <- InputCollect$adstock
  context_signs <- InputCollect$context_signs
  paid_media_signs <- InputCollect$paid_media_signs
  prophet_signs <- InputCollect$prophet_signs
  organic_signs <- InputCollect$organic_signs
  all_media <- InputCollect$all_media
  # factor_vars <- InputCollect$factor_vars
  calibration_input <- InputCollect$calibration_input
  optimizer_name <- InputCollect$nevergrad_algo
  cores <- InputCollect$cores

  ################################################
  #### Get spend share

  dt_inputTrain <- InputCollect$dt_input[rollingWindowStartWhich:rollingWindowEndWhich]
  dt_spendShare <- dt_inputTrain[, .(
    rn = paid_media_vars,
    total_spend = sapply(.SD, sum),
    mean_spend = sapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0])))
  ), .SDcols = paid_media_spends]
  dt_spendShare[, ":="(spend_share = total_spend / sum(total_spend))]

  refreshAddedStartWhich <- which(dt_modRollWind$ds == refreshAddedStart)
  dt_spendShareRF <- dt_inputTrain[
    refreshAddedStartWhich:rollingWindowLength,
    .(rn = paid_media_vars,
      total_spend = sapply(.SD, sum),
      mean_spend = sapply(.SD, function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x > 0])))
    ),
    .SDcols = paid_media_spends
  ]
  dt_spendShareRF[, ":="(spend_share = total_spend / sum(total_spend))]
  dt_spendShare[, ":="(total_spend_refresh = dt_spendShareRF$total_spend,
                       mean_spend_refresh = dt_spendShareRF$mean_spend,
                       spend_share_refresh = dt_spendShareRF$spend_share)]


  ################################################
  #### Start Nevergrad loop

  t0 <- Sys.time()

  ## set iterations

  if (hyper_fixed == FALSE) {
    iterTotal <- iterations
    iterPar <- cores
  } else {
    iterTotal <- 1
    iterPar <- 1
  }

  iterNG <- ifelse(hyper_fixed == FALSE, ceiling(iterations / cores), 1)

  # cat("\nRunning", iterTotal,"iterations with evolutionary algorithm on",adstock, "adstocking,", length(hyper_bound_list_updated),"hyperparameters,",lambda.n,"-fold ridge x-validation using", cores,"cores...\n")

  ## start Nevergrad optimiser

  if (length(hyper_bound_list_updated) != 0) {
    my_tuple <- tuple(hyper_count)
    instrumentation <- ng$p$Array(shape = my_tuple, lower = 0., upper = 1.)
    # instrumentation$set_bounds(0., 1.)
    optimizer <- ng$optimizers$registry[optimizer_name](instrumentation, budget = iterTotal, num_workers = cores)
    if (is.null(calibration_input)) {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0))
    } else {
      optimizer$tell(ng$p$MultiobjectiveReference(), tuple(1.0, 1.0, 1.0))
    }
    # Creating a hyperparameter vector to be used in the next learning.
  }

  ## start loop

  resultCollectNG <- list()
  cnt <- 0
  if (hyper_fixed == FALSE) {
    pb <- txtProgressBar(max = iterTotal, style = 3)
  }
  # assign("InputCollect", InputCollect, envir = .GlobalEnv) # adding this to enable InputCollect reading during parallel
  # opts <- list(progress = function(n) setTxtProgressBar(pb, n))

  # enable parallelisation of main modelling loop for MacOS and Linux only
  parallel_processing <- .Platform$OS.type == "unix"

  # create cluster before big for-loop to minimize overhead for parallel backend registering
  if (parallel_processing) {
    registerDoParallel(InputCollect$cores)
  } else {
    registerDoSEQ()
  }

  sysTimeDopar <- system.time({
    for (lng in 1:iterNG) { # lng = 1
      nevergrad_hp <- list()
      nevergrad_hp_val <- list()
      hypParamSamList <- list()
      hypParamSamNG <- c()

      if (hyper_fixed == FALSE) {
        for (co in 1:iterPar) { # co = 1

          ## get hyperparameter sample with ask
          nevergrad_hp[[co]] <- optimizer$ask()
          nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value

          ## scale sample to given bounds
          for (hypNameLoop in hyper_bound_list_updated_name) { # hypNameLoop <- local_name.all[1]
            index <- which(hypNameLoop == hyper_bound_list_updated_name)
            channelBound <- unlist(hyper_bound_list_updated[hypNameLoop])
            hyppar_for_qunif <- nevergrad_hp_val[[co]][index]
            hyppar_scaled <- qunif(hyppar_for_qunif, min(channelBound), max(channelBound))
            hypParamSamNG[hypNameLoop] <- hyppar_scaled
          }
          hypParamSamList[[co]] <- transpose(data.table(hypParamSamNG))
        }

        hypParamSamNG <- rbindlist(hypParamSamList)
        hypParamSamNG <- setnames(hypParamSamNG, names(hypParamSamNG), hyper_bound_list_updated_name)

        ## add fixed hyperparameters

        if (hyper_count_fixed != 0) {
          hypParamSamNG <- cbind(hypParamSamNG, dt_hyperFixed)
          hypParamSamNG <- setcolorder(hypParamSamNG, hypParamSamName)
        }
      } else {
        hypParamSamNG <- as.data.table(matrix(unlist(hyper_bound_list), nrow = 1))
        setnames(hypParamSamNG, names(hypParamSamNG), hypParamSamName)
      }

      ## Parallel start

      nrmse.collect <- c()
      decomp.rssd.collect <- c()
      best_mape <- Inf

      doparCollect <- suppressPackageStartupMessages(
        foreach(i = 1:iterPar) %dorng% { # i = 1
          t1 <- Sys.time()

          #####################################
          #### Get hyperparameter sample

          hypParamSam <- unlist(hypParamSamNG[i])

          #### Tranform media with hyperparameters
          dt_modAdstocked <- dt_mod[, .SD, .SDcols = setdiff(names(dt_mod), "ds")]
          mediaAdstocked <- list()
          mediaVecCum <- list()
          mediaSaturated <- list()
          for (v in 1:length(all_media)) {
            m <- dt_modAdstocked[, get(all_media[v])]

            ## adstocking

            if (adstock == "geometric") {
              theta <- hypParamSam[paste0(all_media[v], "_thetas")]
              x_list <- adstock_geometric(x = m, theta = theta)
            } else if (adstock == "weibull_cdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")]
              scale <- hypParamSam[paste0(all_media[v], "_scales")]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = rollingWindowLength, type = "cdf")
            } else if (adstock == "weibull_pdf") {
              shape <- hypParamSam[paste0(all_media[v], "_shapes")]
              scale <- hypParamSam[paste0(all_media[v], "_scales")]
              x_list <- adstock_weibull(x = m, shape = shape, scale = scale, windlen = rollingWindowLength, type = "pdf")
            } else {
              break
              print("adstock parameter must be geometric, weibull_cdf or weibull_pdf")
            }

            m_adstocked <- x_list$x_decayed
            mediaAdstocked[[v]] <- m_adstocked
            mediaVecCum[[v]] <- x_list$thetaVecCum

            ## saturation
            m_adstockedRollWind <- m_adstocked[rollingWindowStartWhich:rollingWindowEndWhich]

            alpha <- hypParamSam[paste0(all_media[v], "_alphas")]
            gamma <- hypParamSam[paste0(all_media[v], "_gammas")]
            mediaSaturated[[v]] <- saturation_hill(m_adstockedRollWind, alpha = alpha, gamma = gamma)
          }

          names(mediaAdstocked) <- all_media
          dt_modAdstocked[, (all_media) := mediaAdstocked]
          dt_mediaVecCum <- data.table()[, (all_media) := mediaVecCum]

          names(mediaSaturated) <- all_media
          dt_modSaturated <- dt_modAdstocked[rollingWindowStartWhich:rollingWindowEndWhich]
          dt_modSaturated[, (all_media) := mediaSaturated]

          #####################################
          #### Split and prepare data for modelling

          dt_train <- copy(dt_modSaturated)

          ## contrast matrix because glmnet does not treat categorical variables
          y_train <- dt_train$dep_var
          x_train <- model.matrix(dep_var ~ ., dt_train)[, -1]

          ## create lambda sequence with x and y
          # lambda_seq <- ridge_lambda(x=x_train, y=y_train, seq_len = lambda.n, lambda_min_ratio = 0.0001)

          ## define sign control
          dt_sign <- dt_modSaturated[, !"dep_var"] # names(dt_sign)
          x_sign <- c(prophet_signs, context_signs, paid_media_signs, organic_signs)
          names(x_sign) <- c(prophet_vars, context_vars, paid_media_vars, organic_vars)
          check_factor <- sapply(dt_sign, is.factor)

          lower.limits <- c()
          upper.limits <- c()

          for (s in 1:length(check_factor)) {
            if (check_factor[s] == TRUE) {
              level.n <- length(levels(unlist(dt_sign[, s, with = FALSE])))
              if (level.n <= 1) {
                stop("factor variables must have more than 1 level")
              }
              lower_vec <- if (x_sign[s] == "positive") {
                rep(0, level.n - 1)
              } else {
                rep(-Inf, level.n - 1)
              }
              upper_vec <- if (x_sign[s] == "negative") {
                rep(0, level.n - 1)
              } else {
                rep(Inf, level.n - 1)
              }
              lower.limits <- c(lower.limits, lower_vec)
              upper.limits <- c(upper.limits, upper_vec)
            } else {
              lower.limits <- c(lower.limits, ifelse(x_sign[s] == "positive", 0, -Inf))
              upper.limits <- c(upper.limits, ifelse(x_sign[s] == "negative", 0, Inf))
            }
          }

          #####################################
          #### fit ridge regression with x-validation
          cvmod <- cv.glmnet(
            x_train,
            y_train,
            family = "gaussian",
            alpha = 0 # 0 for ridge regression
            # ,lambda = lambda_seq
            , lower.limits = lower.limits,
            upper.limits = upper.limits,
            type.measure = "mse"
            # ,penalty.factor = c(1,1,1,1,1,1,1,1,1)
            # ,nlambda = 100
            # ,nfold = 10
            # ,intercept = FALSE
          ) # plot(cvmod) coef(cvmod)
          # head(predict(cvmod, newx=x_train, s="lambda.1se"))

          lambda_range <- c(cvmod$lambda.min, cvmod$lambda.1se)
          lambda <- lambda_range[1] + (lambda_range[2]-lambda_range[1]) * lambda_control

          #####################################
          #### refit ridge regression with selected lambda from x-validation

          ## if no lift calibration, refit using best lambda
          if (hyper_fixed == FALSE) {
            mod_out <- model_refit(x_train, y_train, lambda = lambda, lower.limits, upper.limits)
          } else {
            mod_out <- model_refit(x_train, y_train, lambda = lambda_fixed[i], lower.limits, upper.limits)
            lambda <- lambda_fixed[i]
          }

          # hypParamSam["lambdas"] <- cvmod$lambda.1se
          # hypParamSamName <- names(hypParamSam)

          decompCollect <- model_decomp(coefs = mod_out$coefs, dt_modSaturated = dt_modSaturated, x = x_train, y_pred = mod_out$y_pred, i = i, dt_modRollWind = dt_modRollWind, refreshAddedStart = refreshAddedStart)
          nrmse <- mod_out$nrmse_train
          mape <- 0
          df.int <- mod_out$df.int


          #####################################
          #### get calibration mape

          if (!is.null(calibration_input)) {
            liftCollect <- calibrate_mmm(decompCollect = decompCollect, calibration_input = calibration_input, paid_media_vars = paid_media_vars, dayInterval = InputCollect$dayInterval)
            mape <- liftCollect[, mean(mape_lift)]
          }

          #####################################
          #### calculate multi-objectives for pareto optimality

          ## decomp objective: sum of squared distance between decomp share and spend share to be minimised
          dt_decompSpendDist <- decompCollect$xDecompAgg[rn %in% paid_media_vars, .(rn, xDecompAgg, xDecompPerc, xDecompMeanNon0Perc, xDecompMeanNon0, xDecompPercRF, xDecompMeanNon0PercRF, xDecompMeanNon0RF)]
          dt_decompSpendDist <- dt_decompSpendDist[dt_spendShare[, .(rn, spend_share, spend_share_refresh, mean_spend, total_spend)], on = "rn"]
          dt_decompSpendDist[, ":="(effect_share = xDecompPerc / sum(xDecompPerc),
                                    effect_share_refresh = xDecompPercRF / sum(xDecompPercRF))]
          decompCollect$xDecompAgg[dt_decompSpendDist[, .(rn, spend_share_refresh, effect_share_refresh)],
                                   ":="(spend_share_refresh = i.spend_share_refresh,
                                        effect_share_refresh = i.effect_share_refresh),
                                   on = "rn"
          ]

          if (!refresh) {
            decomp.rssd <- dt_decompSpendDist[, sqrt(sum((effect_share - spend_share)^2))]
          } else {
            dt_decompRF <- decompCollect$xDecompAgg[, .(rn, decomp_perc = xDecompPerc)][xDecompAggPrev[, .(rn, decomp_perc_prev = xDecompPerc)], on = "rn"]
            decomp.rssd.nonmedia <- dt_decompRF[!(rn %in% paid_media_vars), sqrt(mean((decomp_perc - decomp_perc_prev)^2))]
            decomp.rssd.media <- dt_decompSpendDist[, sqrt(mean((effect_share_refresh - spend_share_refresh)^2))]
            decomp.rssd <- decomp.rssd.media + decomp.rssd.nonmedia / (1 - refresh_steps / rollingWindowLength)
          }

          if (is.nan(decomp.rssd)) {
            # message("all media in this iteration have 0 coefficients")
            decomp.rssd <- Inf
            dt_decompSpendDist[, effect_share := 0]
          }

          ## adstock objective: sum of squared infinite sum of decay to be minimised - deprecated
          # dt_decaySum <- dt_mediaVecCum[,  .(rn = all_media, decaySum = sapply(.SD, sum)), .SDcols = all_media]
          # adstock.ssisd <- dt_decaySum[, sum(decaySum^2)]

          ## calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift

          #####################################
          #### Collect output

          resultHypParam <- data.table()[, (hypParamSamName) := lapply(hypParamSam[1:length(hypParamSamName)], function(x) x)]

          resultCollect <- list(
            resultHypParam = resultHypParam[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , pos = prod(decompCollect$xDecompAgg$pos),
              lambda = lambda
              # ,Score = -mape
              , Elapsed = as.numeric(difftime(Sys.time(), t1, units = "secs")),
              ElapsedAccum = as.numeric(difftime(Sys.time(), t0, units = "secs")),
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            xDecompVec = if (hyper_fixed == TRUE) {
              decompCollect$xDecompVec[, ":="(
                intercept = decompCollect$xDecompAgg[rn == "(Intercept)", xDecompAgg],
                mape = mape,
                nrmse = nrmse,
                decomp.rssd = decomp.rssd
                # ,adstock.ssisd = adstock.ssisd
                , rsq_train = mod_out$rsq_train
                # ,rsq_test = mod_out$rsq_test
                , lambda = lambda,
                iterPar = i,
                iterNG = lng,
                df.int = df.int)]
            } else {
              NULL
            },
            xDecompAgg = decompCollect$xDecompAgg[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , lambda = lambda,
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            liftCalibration = if (!is.null(calibration_input)) {
              liftCollect[, ":="(
                mape = mape,
                nrmse = nrmse,
                decomp.rssd = decomp.rssd
                # ,adstock.ssisd = adstock.ssisd
                , rsq_train = mod_out$rsq_train
                # ,rsq_test = mod_out$rsq_test
                , lambda = lambda,
                iterPar = i,
                iterNG = lng)]
            } else {
              NULL
            },
            decompSpendDist = dt_decompSpendDist[, ":="(
              mape = mape,
              nrmse = nrmse,
              decomp.rssd = decomp.rssd
              # ,adstock.ssisd = adstock.ssisd
              , rsq_train = mod_out$rsq_train
              # ,rsq_test = mod_out$rsq_test
              , lambda = lambda,
              iterPar = i,
              iterNG = lng,
              df.int = df.int)],
            mape.lift = mape,
            nrmse = nrmse,
            decomp.rssd = decomp.rssd,
            iterPar = i,
            iterNG = lng,
            df.int = df.int
            # ,cvmod = cvmod
          )

          best_mape <- min(best_mape, mape)
          if (cnt == iterTotal) {
            print(" === ")
            print(paste0("Optimizer_name: ", optimizer_name, ";  Total_iterations: ", cnt, ";   best_mape: ", best_mape))
          }
          return(resultCollect)
        }
      ) # end foreach parallel

      nrmse.collect <- sapply(doparCollect, function(x) x$nrmse)
      decomp.rssd.collect <- sapply(doparCollect, function(x) x$decomp.rssd)
      mape.lift.collect <- sapply(doparCollect, function(x) x$mape.lift)

      #####################################
      #### Nevergrad tells objectives

      if (hyper_fixed == FALSE) {
        if (is.null(calibration_input)) {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.collect[co], decomp.rssd.collect[co]))
          }
        } else {
          for (co in 1:iterPar) {
            optimizer$tell(nevergrad_hp[[co]], tuple(nrmse.collect[co], decomp.rssd.collect[co], mape.lift.collect[co]))
          }
        }
      }

      resultCollectNG[[lng]] <- doparCollect
      cnt <- cnt + iterPar
      if (hyper_fixed == FALSE) setTxtProgressBar(pb, cnt)
    } ## end NG loop
  }) # end system.time

  message("\n Finished in ", round(sysTimeDopar[3] / 60, 2), " mins")

  # stop cluster to avoid memory leaks
  stopImplicitCluster()

  if (hyper_fixed == FALSE) close(pb)
  registerDoSEQ()
  getDoParWorkers()

  #####################################
  #### Final result collect

  resultCollect <- list(
    resultHypParam = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$resultHypParam))
    }))[order(nrmse)],
    xDecompVec = if (hyper_fixed == TRUE) {
      rbindlist(lapply(resultCollectNG, function(x) {
        rbindlist(lapply(x, function(y) y$xDecompVec))
      }))[order(nrmse, ds)]
    } else {
      NULL
    },
    xDecompAgg = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$xDecompAgg))
    }))[order(nrmse)],
    liftCalibration = if (!is.null(calibration_input)) {
      rbindlist(lapply(resultCollectNG, function(x) {
        rbindlist(lapply(x, function(y) y$liftCalibration))
      }))[order(mape, liftMedia, liftStart)]
    } else {
      NULL
    },
    decompSpendDist = rbindlist(lapply(resultCollectNG, function(x) {
      rbindlist(lapply(x, function(y) y$decompSpendDist))
    }))[order(nrmse)]
    # ,mape = unlist(lapply(doparCollect, function(x) x$mape))
    # ,iterRS = unlist(lapply(doparCollect, function(x) x$iterRS))
    # ,paretoFront= as.data.table(pareto_results_ordered)
    # ,cvmod = lapply(doparCollect, function(x) x$cvmod)
  )
  resultCollect$iter <- length(resultCollect$mape)
  # resultCollect$best.iter <- resultCollect$resultHypParam$iterRS[1]
  resultCollect$elapsed.min <- sysTimeDopar[3] / 60
  resultCollect$resultHypParam[, ElapsedAccum := ElapsedAccum - min(ElapsedAccum) + resultCollect$resultHypParam[which.min(ElapsedAccum), Elapsed]] # adjust accummulated time
  resultCollect$resultHypParam

  return(list(
    resultCollect = resultCollect,
    hyperBoundNG = hyper_bound_list_updated,
    hyperBoundFixed = hyper_bound_list_fixed
  ))
}

####################################################################
#' The response function
#'
#' The \code{robyn_response()} function returns the response for a given
#' spend level of a given \code{paid_media_vars} from a selected model
#' result from a selected model build (initial model, refresh model etc.).
#'
#' @inheritParams robyn_allocator
#' @param paid_media_var A character. Selected paid media variable for the response.
#' Must be within \code{InputCollect$paid_media_vars}
#' @param spend Numeric. The desired spend level to return a response for.
#' @param dt_hyppar A data.table. When \code{robyn_object} is not provided, use
#' \code{dt_hyppar = OutputCollect$resultHypParam}. It must be provided along
#' \code{select_model}, \code{dt_coef} and \code{InputCollect}.
#' @param dt_coef A data.table. When \code{robyn_object} is not provided, use
#' \code{dt_coef = OutputCollect$xDecompAgg}. It must be provided along
#' \code{select_model}, \code{dt_hyppar} and \code{InputCollect}.
#' @examples
#' \dontrun{
#' ## Get marginal response (mResponse) and marginal ROI (mROI) for
#' ## the next 1k on 80k for search_clicks_P, when provided the saved
#' ## robyn_object by the robyn_save() function.
#'
#' # Get response for 80k
#' spend1 <- 80000
#' Response1 <- robyn_response(
#'   robyn_object = robyn_object,
#'   paid_media_var = "search_clicks_P",
#'   spend = spend1
#' )
#'
#' # Get ROI for 80k
#' Response1 / spend1 # ROI for search 80k
#'
#' # Get response for 81k
#' spend2 <- spend1 + 1000
#' Response2 <- robyn_response(
#'   robyn_object = robyn_object,
#'   paid_media_var = "search_clicks_P",
#'   spend = spend2
#' )
#'
#' # Get ROI for 81k
#' Response2 / spend2 # ROI for search 81k
#'
#' # Get marginal response (mResponse) for the next 1k on 80k
#' Response2 - Response1
#'
#' # Get marginal ROI (mROI) for the next 1k on 80k
#' (Response2 - Response1) / (spend2 - spend1)
#'
#'
#' ## Get response for 80k for search_clicks_P from the third model refresh
#'
#' robyn_response(
#'   robyn_object = robyn_object,
#'   select_build = 3,
#'   paid_media_var = "search_clicks_P",
#'   spend = 80000
#' )
#'
#' ## Get response for 80k for search_clicks_P from the a certain model SolID
#' ## in the current model output in the global environment
#'
#' robyn_response(,
#'   paid_media_var = "search_clicks_P",
#'   select_model = "3_10_3",
#'   spend = 80000,
#'   dt_hyppar = OutputCollect$resultHypParam,
#'   dt_coef = OutputCollect$xDecompAgg,
#'   InputCollect = InputCollect
#' )
#' }
#' @export
robyn_response <- function(robyn_object = NULL,
                           select_build = NULL,
                           paid_media_var = NULL,
                           select_model = NULL,
                           spend = NULL,
                           dt_hyppar = NULL,
                           dt_coef = NULL,
                           InputCollect = NULL) {

  ## get input
  if (!is.null(robyn_object)) {

    if (!file.exists(robyn_object)) {
      stop("File does not exist or is somewhere else. Check: ", robyn_object)
    } else {
      Robyn <- readRDS(robyn_object)
      objectPath <- dirname(robyn_object)
      objectName <- sub("'\\..*$", "", basename(robyn_object))
    }

    select_build_all <- 0:(length(Robyn) - 1)
    if (is.null(select_build)) {
      select_build <- max(select_build_all)
      message(
        "Using latest model: ", ifelse(select_build == 0, "initial model", paste0("refresh model nr.", select_build)),
        " for the response function. Use parameter 'select_build' to specify which run to use"
      )
    }

    if (!(select_build %in% select_build_all) | length(select_build) != 1) {
      stop("select_build must be one value of ", paste(select_build_all, collapse = ", "))
    }

    listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
    InputCollect <- Robyn[[listName]][["InputCollect"]]
    OutputCollect <- Robyn[[listName]][["OutputCollect"]]
    dt_hyppar <- OutputCollect$resultHypParam
    dt_coef <- OutputCollect$xDecompAgg
    select_model <- OutputCollect$selectID
  } else if (any(is.null(dt_hyppar), is.null(dt_coef), is.null(InputCollect))) {
    stop(paste(
      "When 'robyn_object' is not provided, then 'dt_hyppar = OutputCollect$resultHypParam',",
      "'dt_coef = OutputCollect$xDecompAgg' and 'InputCollect' must be provided"
    ))
  }

  dt_input <- InputCollect$dt_input
  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  startRW <- InputCollect$rollingWindowStartWhich
  endRW <- InputCollect$rollingWindowEndWhich
  adstock <- InputCollect$adstock
  allSolutions <- dt_hyppar[, unique(solID)]
  spendExpoMod <- InputCollect$modNLSCollect

  ## check inputs
  if (is.null(paid_media_var)) {
    stop(paste0("paid_media_var must be one of these values: ", paste(paid_media_vars, collapse = ", ")))
  } else if (!(paid_media_var %in% paid_media_vars) | length(paid_media_var) != 1) {
    stop(paste0("paid_media_var must be one of these values: ", paste(paid_media_vars, collapse = ", ")))
  }

  if (!(select_model %in% allSolutions)) {
    stop(paste0("select_model must be one of these values: ", paste(allSolutions, collapse = ", ")))
  }

  mediaVar <- dt_input[, get(paid_media_var)]

  if (!is.null(spend)) {
    if (length(spend) != 1 | spend <= 0 | !is.numeric(spend)) {
      stop("'spend' must be a positive number")
    }
  }

  ## transform spend to exposure if necessary
  if (paid_media_var %in% InputCollect$exposureVarName) {

    # use non-0 mean spend as marginal level if spend not provided
    if (is.null(spend)) {
      mediaspend <- dt_input[startRW:endRW, get(paid_media_spends[which(paid_media_vars == paid_media_var)])]
      spend <- mean(mediaspend[mediaspend > 0])
      message("'spend' not provided. Using mean of ", paid_media_var, " as marginal level instead")
    }

    # fit spend to exposure
    nls_select <- spendExpoMod[channel == paid_media_var, rsq_nls > rsq_lm]
    if (nls_select) {
      Vmax <- spendExpoMod[channel == paid_media_var, Vmax]
      Km <- spendExpoMod[channel == paid_media_var, Km]
      spend <- mic_men(x = spend, Vmax = Vmax, Km = Km, reverse = FALSE)
    } else {
      coef_lm <- spendExpoMod[channel == paid_media_var, coef_lm]
      spend <- spend * coef_lm
    }
  } else {

    # use non-0 mean spend as marginal level if spend not provided
    if (is.null(spend)) {
      mediaspend <- dt_input[startRW:endRW, get(paid_media_var)]
      spend <- mean(mediaspend[mediaspend > 0])
      message("spend not provided. using mean of ", paid_media_var, " as marginal levl instead")
    }
  }


  ## adstocking
  if (adstock == "geometric") {
    theta <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_thetas"))]
    x_list <- adstock_geometric(x = mediaVar, theta = theta)
  } else if (adstock == "weibull_cdf") {
    shape <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_shapes"))]
    scale <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_scales"))]
    x_list <- adstock_weibull(x = mediaVar, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "cdf")
  } else if (adstock == "weibull_pdf") {
    shape <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_shapes"))]
    scale <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_scales"))]
    x_list <- adstock_weibull(x = mediaVar, shape = shape, scale = scale, windlen = InputCollect$rollingWindowLength, type = "pdf")
  }
  m_adstocked <- x_list$x_decayed

  ## saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_alphas"))]
  gamma <- dt_hyppar[solID == select_model, get(paste0(paid_media_var, "_gammas"))]
  Saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = spend)

  ## decomp
  coeff <- dt_coef[solID == select_model & rn == paid_media_var, coef]
  Response <- Saturated * coeff

  return(as.numeric(Response))
}


model_decomp <- function(coefs, dt_modSaturated, x, y_pred, i, dt_modRollWind, refreshAddedStart) {

  ## input for decomp
  y <- dt_modSaturated$dep_var
  indepVar <- dt_modSaturated[, (setdiff(names(dt_modSaturated), "dep_var")), with = FALSE]
  x <- as.data.table(x)
  intercept <- coefs[1]
  indepVarName <- names(indepVar)
  indepVarCat <- indepVarName[sapply(indepVar, is.factor)]

  ## decomp x
  xDecomp <- data.table(mapply(function(regressor, coeff) {
    regressor * coeff
  }, regressor = x, coeff = coefs[-1]))
  xDecomp <- cbind(data.table(intercept = rep(intercept, nrow(xDecomp))), xDecomp)
  # xDecompOut <- data.table(sapply(indepVarName, function(x) xDecomp[, rowSums(.SD,), .SDcols = str_which(names(xDecomp), x)]))
  xDecompOut <- cbind(data.table(ds = dt_modRollWind$ds, y = y, y_pred = y_pred), xDecomp)

  ## QA decomp
  y_hat <- rowSums(xDecomp)
  errorTerm <- y_hat - y_pred
  if (prod(round(y_pred) == round(y_hat)) == 0) {
    message("\n### attention for loop ", i, " : manual decomp is not matching linear model prediction. Deviation is ", mean(errorTerm / y) * 100, " % ### \n")
  }

  ## output decomp
  y_hat.scaled <- rowSums(abs(xDecomp))
  xDecompOutPerc.scaled <- abs(xDecomp) / y_hat.scaled
  xDecompOut.scaled <- y_hat * xDecompOutPerc.scaled

  xDecompOutAgg <- sapply(xDecompOut[, c("intercept", indepVarName), with = FALSE], function(x) sum(x))
  xDecompOutAggPerc <- xDecompOutAgg / sum(y_hat)
  xDecompOutAggMeanNon0 <- sapply(xDecompOut[, c("intercept", indepVarName), with = FALSE], function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0[is.nan(xDecompOutAggMeanNon0)] <- 0
  xDecompOutAggMeanNon0Perc <- xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)
  # xDecompOutAggPerc.scaled <- abs(xDecompOutAggPerc)/sum(abs(xDecompOutAggPerc))
  # xDecompOutAgg.scaled <- sum(xDecompOutAgg)*xDecompOutAggPerc.scaled

  refreshAddedStartWhich <- which(xDecompOut$ds == refreshAddedStart)
  refreshAddedEnd <- max(xDecompOut$ds)
  refreshAddedEndWhich <- which(xDecompOut$ds == refreshAddedEnd)
  xDecompOutAggRF <- sapply(xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich, c("intercept", indepVarName), with = FALSE], function(x) sum(x))
  y_hatRF <- y_hat[refreshAddedStartWhich:refreshAddedEndWhich]
  xDecompOutAggPercRF <- xDecompOutAggRF / sum(y_hatRF)
  xDecompOutAggMeanNon0RF <- sapply(xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich, c("intercept", indepVarName), with = FALSE], function(x) ifelse(is.na(mean(x[x > 0])), 0, mean(x[x != 0])))
  xDecompOutAggMeanNon0RF[is.nan(xDecompOutAggMeanNon0RF)] <- 0
  xDecompOutAggMeanNon0PercRF <- xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

  coefsOut <- data.table(coefs, keep.rownames = TRUE)
  coefsOutCat <- copy(coefsOut)
  coefsOut[, rn := if (length(indepVarCat) == 0) {
    rn
  } else {
    sapply(indepVarCat, function(x) str_replace(coefsOut$rn, paste0(x, ".*"), x))
  }]
  coefsOut <- coefsOut[, .(coef = mean(s0)), by = rn]

  decompOutAgg <- cbind(coefsOut, data.table(
    xDecompAgg = xDecompOutAgg,
    xDecompPerc = xDecompOutAggPerc,
    xDecompMeanNon0 = xDecompOutAggMeanNon0,
    xDecompMeanNon0Perc = xDecompOutAggMeanNon0Perc,
    xDecompAggRF = xDecompOutAggRF,
    xDecompPercRF = xDecompOutAggPercRF,
    xDecompMeanNon0RF = xDecompOutAggMeanNon0RF,
    xDecompMeanNon0PercRF = xDecompOutAggMeanNon0PercRF
    # ,xDecompAgg.scaled = xDecompOutAgg.scaled
    # ,xDecompPerc.scaled = xDecompOutAggPerc.scaled
  ))
  decompOutAgg[, pos := xDecompAgg >= 0]

  decompCollect <- list(xDecompVec = xDecompOut, xDecompVec.scaled = xDecompOut.scaled, xDecompAgg = decompOutAgg, coefsOutCat = coefsOutCat)

  return(decompCollect)
} ## decomp end


calibrate_mmm <- function(decompCollect, calibration_input, paid_media_vars, dayInterval) {

  # check if any lift channel doesn't have media var
  check_set_lift <- any(sapply(calibration_input$channel, function(x) {
    any(str_detect(x, paid_media_vars))
  }) == FALSE)
  if (check_set_lift) {
    stop("calibration_input channels must have media variable")
  }

  ## prep lift input
  getLiftMedia <- unique(calibration_input$channel)
  getDecompVec <- decompCollect$xDecompVec

  ## loop all lift input
  liftCollect <- list()
  for (m in 1:length(getLiftMedia)) { # loop per lift channel

    liftWhich <- str_which(calibration_input$channel, getLiftMedia[m])

    liftCollect2 <- list()
    for (lw in 1:length(liftWhich)) { # loop per lift test per channel

      ## get lift period subset
      liftStart <- calibration_input[liftWhich[lw], liftStartDate]
      liftEnd <- calibration_input[liftWhich[lw], liftEndDate]
      liftPeriodVec <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", getLiftMedia[m]), with = FALSE]
      liftPeriodVecDependent <- getDecompVec[ds >= liftStart & ds <= liftEnd, c("ds", "y"), with = FALSE]

      ## scale decomp
      mmmDays <- nrow(liftPeriodVec) * dayInterval
      liftDays <- as.integer(liftEnd - liftStart + 1)
      y_hatLift <- sum(unlist(getDecompVec[, -1])) # total pred sales
      x_decompLift <- sum(liftPeriodVec[, 2])
      x_decompLiftScaled <- x_decompLift / mmmDays * liftDays
      y_scaledLift <- liftPeriodVecDependent[, sum(y)] / mmmDays * liftDays

      ## output
      liftCollect2[[lw]] <- data.table(
        liftMedia = getLiftMedia[m],
        liftStart = liftStart,
        liftEnd = liftEnd,
        liftAbs = calibration_input[liftWhich[lw], liftAbs],
        decompAbsScaled = x_decompLiftScaled,
        dependent = y_scaledLift
      )
    }
    liftCollect[[m]] <- rbindlist(liftCollect2)
  }

  ## get mape_lift
  liftCollect <- rbindlist(liftCollect)[, mape_lift := abs((decompAbsScaled - liftAbs) / liftAbs)]
  return(liftCollect)
}


model_refit <- function(x_train, y_train, lambda, lower.limits, upper.limits) {
  mod <- glmnet(
    x_train,
    y_train,
    family = "gaussian",
    alpha = 0, # 0 for ridge regression
    # https://stats.stackexchange.com/questions/138569/why-is-lambda-within-one-standard-error-from-the-minimum-is-a-recommended-valu
    lambda = lambda,
    lower.limits = lower.limits,
    upper.limits = upper.limits
  ) # coef(mod)

  ## drop intercept if negative
  if (coef(mod)[1] < 0) {
    mod <- glmnet(
      x_train,
      y_train,
      family = "gaussian",
      alpha = 0 # 0 for ridge regression
      , lambda = lambda,
      lower.limits = lower.limits,
      upper.limits = upper.limits,
      intercept = FALSE
    ) # coef(mod)
  } # ; plot(mod); print(mod)

  df.int <- ifelse(coef(mod)[1] < 0, 0, 1)

  y_trainPred <- predict(mod, s = lambda, newx = x_train)
  rsq_train <- get_rsq(true = y_train, predicted = y_trainPred, p = ncol(x_train), df.int = df.int)
  rsq_train

  # y_testPred <- predict(mod, s = lambda, newx = x_test)
  # rsq_test <- get_rsq(true = y_test, predicted = y_testPred); rsq_test

  # mape_mod<- mean(abs((y_test - y_testPred)/y_test)* 100); mape_mod
  coefs <- as.matrix(coef(mod))
  # y_pred <- c(y_trainPred, y_testPred)

  # mean(y_train) sd(y_train)
  nrmse_train <- sqrt(mean((y_train - y_trainPred)^2)) / (max(y_train) - min(y_train))
  # nrmse_test <- sqrt(mean(sum((y_test - y_testPred)^2))) /
  # (max(y_test) - min(y_test)) # mean(y_test) sd(y_test)

  mod_out <- list(
    rsq_train = rsq_train
    # ,rsq_test = rsq_test
    , nrmse_train = nrmse_train
    # ,nrmse_test = nrmse_test
    # ,mape_mod = mape_mod
    , coefs = coefs,
    y_pred = y_trainPred,
    mod = mod,
    df.int = df.int
  )

  return(mod_out)
}

ridge_lambda <- function(x,
                         y,
                         seq_len = 100,
                         lambda_min_ratio = 0.0001) {
  mysd <- function(y) sqrt(sum((y - mean(y))^2) / length(y))
  sx <- scale(x, scale = apply(x, 2, mysd))
  sx <- as.matrix(sx, ncol = ncol(x), nrow = nrow(x))
  # sy <- as.vector(scale(y, scale=mysd(y)))
  sy <- y
  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
  lambda_max <- max(abs(colSums(sx * sy))) / (0.001 * nrow(x))

  lambda_max_log <- log(lambda_max)
  log_step <- (log(lambda_max) - log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
  log_seq <- seq(log(lambda_max), log(lambda_max * lambda_min_ratio), length.out = seq_len)
  lambda_seq <- exp(log_seq)
  return(lambda_seq)
}
