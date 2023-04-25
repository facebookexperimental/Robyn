# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Build Refresh Model
#'
#' @description
#' \code{robyn_refresh()} builds updated models based on
#' the previously built models saved in the \code{Robyn.RDS} object specified
#' in \code{robyn_object}. For example, when updating the initial build with 4
#' weeks of new data, \code{robyn_refresh()} consumes the selected model of
#' the initial build, sets lower and upper bounds of hyperparameters for the
#' new build around the selected hyperparameters of the previous build,
#' stabilizes the effect of baseline variables across old and new builds, and
#' regulates the new effect share of media variables towards the latest
#' spend level. It returns the aggregated results with all previous builds for
#' reporting purposes and produces reporting plots.
#'
#' You must run \code{robyn_save()} to select and save an initial model first,
#' before refreshing.
#'
#' \strong{When should \code{robyn_refresh()} NOT be used:}
#' The \code{robyn_refresh()} function is suitable for
#' updating within "reasonable periods". Two situations are considered better
#' to rebuild model instead of refreshing:
#'
#' 1. Most data is new: If initial model was trained with 100 weeks worth of
#' data but we add +50 weeks of new data.
#'
#' 2. New variables are added: If initial model had less variables than the ones
#' we want to start using on new refresh model.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @inheritParams robyn_outputs
#' @inheritParams robyn_inputs
#' @param dt_input data.frame. Should include all previous data and newly added
#' data for the refresh.
#' @param dt_holidays data.frame. Raw input holiday data. Load standard
#' Prophet holidays using \code{data("dt_prophet_holidays")}.
#' @param refresh_steps Integer. It controls how many time units the refresh
#' model build move forward. For example, \code{refresh_steps = 4} on weekly data
#' means the \code{InputCollect$window_start} & \code{InputCollect$window_end}
#' move forward 4 weeks. If \code{refresh_steps} is smaller than the number of
#' newly provided data points, then Robyn would only use the first N steps of the
#' new data.
#' @param refresh_mode Character. Options are "auto" and "manual". In auto mode,
#' the \code{robyn_refresh()} function builds refresh models with given
#' \code{refresh_steps} repeatedly until there's no more data available. I
#' manual mode, the \code{robyn_refresh()} only moves forward \code{refresh_steps}
#' only once. "auto" mode has been deprecated when using \code{json_file} input.
#' @param refresh_iters Integer. Iterations per refresh. Rule of thumb is, the
#' more new data added, the more iterations needed. More reliable recommendation
#' still needs to be investigated.
#' @param refresh_trials Integer. Trials per refresh. Defaults to 5 trials.
#' More reliable recommendation still needs to be investigated.
#' @param version_prompt Logical. If FALSE, the model refresh version will be
#' selected based on the smallest combined error of normalized NRMSE, DECOMP.RSSD, MAPE.
#' If \code{TRUE}, a prompt will be presented to the user to select one of the refreshed
#' models (one-pagers and Pareto CSV files will already be generated).
#' @param ... Additional parameters to overwrite original custom parameters
#' passed into initial model.
#' @return List. The Robyn object, class \code{robyn_refresh}.
#' @examples
#' \dontrun{
#' # Loading dummy data
#' data("dt_simulated_weekly")
#' data("dt_prophet_holidays")
#' # Set the (pre-trained and exported) Robyn model JSON file
#' json_file <- "~/Robyn_202208081444_init/RobynModel-2_55_4.json"
#'
#' # Run \code{robyn_refresh()} with 13 weeks cadence in auto mode
#' Robyn <- robyn_refresh(
#'   json_file = json_file,
#'   dt_input = dt_simulated_weekly,
#'   dt_holidays = Robyn::dt_prophet_holidays,
#'   refresh_steps = 13,
#'   refresh_mode = "auto",
#'   refresh_iters = 200,
#'   refresh_trials = 5
#' )
#'
#' # Run \code{robyn_refresh()} with 4 weeks cadence in manual mode
#' json_file2 <- "~/Robyn_202208081444_init/Robyn_202208090847_rf/RobynModel-1_2_3.json"
#' Robyn <- robyn_refresh(
#'   json_file = json_file2,
#'   dt_input = dt_simulated_weekly,
#'   dt_holidays = Robyn::dt_prophet_holidays,
#'   refresh_steps = 4,
#'   refresh_mode = "manual",
#'   refresh_iters = 200,
#'   refresh_trials = 5
#' )
#' }
#' @return List. Same as \code{robyn_run()} but with refreshed models.
#' @export
robyn_refresh <- function(json_file = NULL,
                          robyn_object = NULL,
                          dt_input = NULL,
                          dt_holidays = Robyn::dt_prophet_holidays,
                          refresh_steps = 4,
                          refresh_mode = "manual",
                          refresh_iters = 1000,
                          refresh_trials = 3,
                          plot_folder = NULL,
                          plot_pareto = TRUE,
                          version_prompt = FALSE,
                          export = TRUE,
                          calibration_input = NULL,
                          ...) {
  refreshControl <- TRUE
  while (refreshControl) {
    ## Check for NA values
    check_nas(dt_input)
    check_nas(dt_holidays)

    ## Load initial model
    if (!is.null(json_file)) {
      Robyn <- list()
      json <- robyn_read(json_file, step = 2, quiet = TRUE)
      listInit <- suppressWarnings(robyn_recreate(
        json_file = json_file,
        dt_input = dt_input,
        dt_holidays = dt_holidays,
        quiet = FALSE, ...
      ))
      listInit$InputCollect$refreshSourceID <- json$ExportedModel$select_model
      chainData <- robyn_chain(json_file)
      listInit$InputCollect$refreshChain <- attr(chainData, "chain")
      listInit$InputCollect$refreshDepth <- refreshDepth <- length(attr(chainData, "chain"))
      listInit$OutputCollect$hyper_updated <- json$ExportedModel$hyper_updated
      Robyn[["listInit"]] <- listInit
      if (is.null(plot_folder)) {
        objectPath <- json$ExportedModel$plot_folder
      } else {
        objectPath <- plot_folder
      }
      refreshCounter <- 1 # Dummy for now (legacy)
    }
    if (!is.null(robyn_object)) {
      RobynImported <- robyn_load(robyn_object)
      Robyn <- RobynImported$Robyn
      objectPath <- RobynImported$objectPath
      robyn_object <- RobynImported$robyn_object
      refreshCounter <- length(Robyn) - sum(names(Robyn) == "refresh")
      refreshDepth <- NULL # Dummy for now (legacy)
    }
    depth <- ifelse(!is.null(refreshDepth), refreshDepth, refreshCounter)

    objectCheck <- if (refreshCounter == 1) {
      "listInit"
    } else {
      c("listInit", paste0("listRefresh", 1:(refreshCounter - 1)))
    }
    if (!all(objectCheck %in% names(Robyn))) {
      stop(
        "Saved Robyn object is corrupted. It should contain these elements:\n ",
        paste(objectCheck, collapse = ", "),
        ".\n Please, re run the model or fix it manually."
      )
    }

    ## Check rule of thumb: 50% of data shouldn't be new
    check_refresh_data(Robyn, dt_input)

    ## Get previous data
    if (refreshCounter == 1) {
      InputCollectRF <- Robyn$listInit$InputCollect
      listOutputPrev <- Robyn$listInit$OutputCollect
      InputCollectRF$xDecompAggPrev <- listOutputPrev$xDecompAgg
      if (length(unique(Robyn$listInit$OutputCollect$resultHypParam$solID)) > 1) {
        stop("Run robyn_write() first to select and export any Robyn model")
      }
    } else {
      listName <- paste0("listRefresh", refreshCounter - 1)
      InputCollectRF <- Robyn[[listName]][["InputCollect"]]
      listOutputPrev <- Robyn[[listName]][["OutputCollect"]]
      listReportPrev <- Robyn[[listName]][["ReportCollect"]]
      ## Model selection from previous build
      if (!"error_score" %in% names(listOutputPrev$resultHypParam)) {
        listOutputPrev$resultHypParam <- as.data.frame(listOutputPrev$resultHypParam) %>%
          mutate(error_score = errors_scores(., ts_validation = listOutputPrev$OutputModels$ts_validation, ...))
      }
      which_bestModRF <- which.min(listOutputPrev$resultHypParam$error_score)[1]
      listOutputPrev$resultHypParam <- listOutputPrev$resultHypParam[which_bestModRF, ]
      listOutputPrev$xDecompAgg <- listOutputPrev$xDecompAgg[which_bestModRF, ]
      listOutputPrev$mediaVecCollect <- listOutputPrev$mediaVecCollect[which_bestModRF, ]
      listOutputPrev$xDecompVecCollect <- listOutputPrev$xDecompVecCollect[which_bestModRF, ]
    }

    InputCollectRF$refreshCounter <- refreshCounter
    InputCollectRF$refresh_steps <- refresh_steps
    if (refresh_steps >= InputCollectRF$rollingWindowLength) {
      stop("Refresh input data is completely new. Please rebuild model using robyn_run().")
    }

    ## Load new data
    if (TRUE) {
      dt_input <- as_tibble(as.data.frame(dt_input))
      date_input <- check_datevar(dt_input, InputCollectRF$date_var)
      dt_input <- date_input$dt_input # sort date by ascending
      InputCollectRF$dt_input <- dt_input
      dt_holidays <- as_tibble(as.data.frame(dt_holidays))
      InputCollectRF$dt_holidays <- dt_holidays
    }

    #### Update refresh model parameters

    ## Refresh rolling window
    if (TRUE) {
      InputCollectRF$refreshAddedStart <- as.Date(InputCollectRF$window_end) + InputCollectRF$dayInterval
      totalDates <- as.Date(dt_input[, InputCollectRF$date_var][[1]])
      refreshStart <- InputCollectRF$window_start <- as.Date(InputCollectRF$window_start) + InputCollectRF$dayInterval * refresh_steps
      refreshStartWhich <- InputCollectRF$rollingWindowStartWhich <- which.min(abs(difftime(totalDates, refreshStart, units = "days")))
      refreshEnd <- InputCollectRF$window_end <- as.Date(InputCollectRF$window_end) + InputCollectRF$dayInterval * refresh_steps
      refreshEndWhich <- InputCollectRF$rollingWindowEndWhich <- which.min(abs(difftime(totalDates, refreshEnd, units = "days")))
      InputCollectRF$rollingWindowLength <- refreshEndWhich - refreshStartWhich + 1
    }

    if (refreshEnd > max(totalDates)) {
      stop("Not enough data for this refresh. Input data from date ", refreshEnd, " or later required")
    }
    if (!is.null(json_file) && refresh_mode == "auto") {
      message("Input 'refresh_mode' = 'auto' has been deprecated. Changed to 'manual'")
      refresh_mode <- "manual"
    }
    if (refresh_mode == "manual") {
      refreshLooper <- 1
      message(sprintf("\n>>> Building refresh model #%s in %s mode", depth, refresh_mode))
      refreshControl <- FALSE
    } else {
      refreshLooper <- floor(as.numeric(difftime(max(totalDates), refreshEnd, units = "days")) /
        InputCollectRF$dayInterval / refresh_steps)
      message(sprintf(
        "\n>>> Building refresh model #%s in %s mode. %s more to go...",
        depth, refresh_mode, refreshLooper
      ))
    }

    #### Update refresh model parameters

    ## Calibration new data
    if (!is.null(calibration_input)) {
      calibration_input <- bind_rows(
        InputCollectRF$calibration_input %>%
          mutate(
            liftStartDate = as.Date(.data$liftStartDate),
            liftEndDate = as.Date(.data$liftEndDate)
          ), calibration_input
      ) %>% distinct()
      ## Check calibration data
      calibration_input <- check_calibration(
        dt_input = InputCollectRF$dt_input,
        date_var = InputCollectRF$date_var,
        calibration_input = calibration_input,
        dayInterval = InputCollectRF$dayInterval,
        dep_var = InputCollectRF$dep_var,
        window_start = InputCollectRF$window_start,
        window_end = InputCollectRF$window_end,
        paid_media_spends = InputCollectRF$paid_media_spends,
        organic_vars = InputCollectRF$organic_vars
      )
      InputCollectRF$calibration_input <- calibration_input
    }

    ## Refresh hyperparameter bounds
    InputCollectRF$hyperparameters <- refresh_hyps(
      initBounds = Robyn$listInit$OutputCollect$hyper_updated,
      listOutputPrev, refresh_steps,
      rollingWindowLength = InputCollectRF$rollingWindowLength
    )

    ## Feature engineering for refreshed data
    # Note that if custom prophet parameters were passed initially,
    # will be used again unless changed in ...
    InputCollectRF <- robyn_engineering(InputCollectRF, ...)

    ## Refresh model with adjusted decomp.rssd
    # OutputCollectRF <- Robyn$listRefresh1$OutputCollect
    if (is.null(InputCollectRF$calibration_input)) {
      rf_cal_constr <- listOutputPrev[["calibration_constraint"]]
    } else {
      rf_cal_constr <- 1
    }
    OutputCollectRF <- robyn_run(
      InputCollect = InputCollectRF,
      plot_folder = objectPath,
      calibration_constraint = rf_cal_constr,
      add_penalty_factor = listOutputPrev[["add_penalty_factor"]],
      iterations = refresh_iters,
      trials = refresh_trials,
      refresh = TRUE,
      outputs = TRUE, # So we end up with OutputCollect instead of OutputModels
      export = export,
      plot_pareto = plot_pareto,
      ...
    )

    ## Select winner model for current refresh (the lower error_score the better)
    OutputCollectRF$resultHypParam <- OutputCollectRF$resultHypParam %>%
      arrange(.data$error_score) %>%
      select(.data$solID, everything()) %>%
      ungroup()
    bestMod <- OutputCollectRF$resultHypParam$solID[1]

    # Pick best model (and don't crash if not valid)
    selectID <- NULL
    while (length(selectID) == 0) {
      if (version_prompt) {
        selectID <- readline("Input model ID to use for the refresh: ")
        message(
          "Selected model ID: ", selectID, " for refresh model #",
          depth, " based on your input"
        )
        if (!selectID %in% OutputCollectRF$allSolutions) {
          message(sprintf(
            "Selected model (%s) NOT valid.\n  Choose any of: %s",
            selectID, v2t(OutputCollectRF$allSolutions)
          ))
        }
      } else {
        selectID <- bestMod
        message(
          "Selected model ID: ", selectID, " for refresh model #",
          depth, " based on the smallest combined normalised errors"
        )
      }
      if (!isTRUE(selectID %in% OutputCollectRF$allSolutions)) {
        version_prompt <- TRUE
        selectID <- NULL
      }
    }
    OutputCollectRF$selectID <- selectID

    #### Result collect & save

    # Add refreshStatus column to multiple OutputCollectRF data.frames
    these <- c("resultHypParam", "xDecompAgg", "mediaVecCollect", "xDecompVecCollect")
    for (tb in these) {
      OutputCollectRF[[tb]] <- OutputCollectRF[[tb]] %>%
        mutate(
          refreshStatus = refreshCounter,
          bestModRF = .data$solID %in% bestMod
        )
    }

    # Create bestModRF and refreshStatus columns to listOutputPrev data.frames
    if (refreshCounter == 1) {
      for (tb in these) {
        listOutputPrev[[tb]] <- mutate(
          listOutputPrev[[tb]],
          bestModRF = TRUE,
          refreshStatus = 0
        )
      }
      listReportPrev <- listOutputPrev
      names(listReportPrev) <- paste0(names(listReportPrev), "Report")
      listReportPrev$mediaVecReport <- listOutputPrev$mediaVecCollect %>%
        filter(
          .data$ds >= (refreshStart - InputCollectRF$dayInterval * refresh_steps),
          .data$ds <= (refreshEnd - InputCollectRF$dayInterval * refresh_steps)
        ) %>%
        bind_rows(
          OutputCollectRF$mediaVecCollect %>%
            filter(
              .data$bestModRF == TRUE,
              .data$ds >= InputCollectRF$refreshAddedStart,
              .data$ds <= refreshEnd
            )
        ) %>%
        arrange(.data$type, .data$ds, .data$refreshStatus)
      listReportPrev$xDecompVecReport <- listOutputPrev$xDecompVecCollect %>%
        bind_rows(
          OutputCollectRF$xDecompVecCollect %>%
            filter(
              .data$bestModRF == TRUE,
              .data$ds >= InputCollectRF$refreshAddedStart,
              .data$ds <= refreshEnd
            )
        )
    }

    resultHypParamReport <- listReportPrev$resultHypParamReport %>%
      bind_rows(
        filter(OutputCollectRF$resultHypParam, .data$bestModRF == TRUE)
      ) %>%
      mutate(refreshStatus = row_number() - 1)

    xDecompAggReport <- listReportPrev$xDecompAggReport %>%
      bind_rows(
        filter(OutputCollectRF$xDecompAgg, .data$bestModRF == TRUE) %>%
          mutate(refreshStatus = refreshCounter)
      )

    mediaVecReport <- as_tibble(listReportPrev$mediaVecReport) %>%
      mutate(ds = as.Date(.data$ds, origin = "1970-01-01")) %>%
      bind_rows(
        filter(
          mutate(OutputCollectRF$mediaVecCollect,
            ds = as.Date(.data$ds, origin = "1970-01-01")
          ),
          .data$bestModRF == TRUE,
          .data$ds >= InputCollectRF$refreshAddedStart,
          .data$ds <= refreshEnd
        ) %>%
          mutate(refreshStatus = refreshCounter)
      ) %>%
      arrange(.data$type, .data$ds, .data$refreshStatus)

    xDecompVecReport <- listReportPrev$xDecompVecReport %>%
      mutate(ds = as.Date(.data$ds, origin = "1970-01-01")) %>%
      bind_rows(
        filter(
          mutate(OutputCollectRF$xDecompVecCollect,
            ds = as.Date(.data$ds, origin = "1970-01-01")
          ),
          .data$bestModRF == TRUE,
          .data$ds >= InputCollectRF$refreshAddedStart,
          .data$ds <= refreshEnd
        ) %>%
          mutate(refreshStatus = refreshCounter)
      )

    #### Result objects to export
    ReportCollect <- list(
      resultHypParamReport = resultHypParamReport,
      xDecompAggReport = xDecompAggReport,
      mediaVecReport = mediaVecReport,
      xDecompVecReport = xDecompVecReport,
      # Selected models (original + refresh IDs)
      selectIDs = resultHypParamReport$solID
    )
    listNameUpdate <- paste0("listRefresh", refreshCounter)
    Robyn[[listNameUpdate]] <- list(
      InputCollect = InputCollectRF,
      OutputCollect = OutputCollectRF,
      ReportCollect = ReportCollect
    )

    #### Reporting plots
    # InputCollectRF <- Robyn$listRefresh1$InputCollect
    # OutputCollectRF <- Robyn$listRefresh1$OutputCollect
    # ReportCollect <- Robyn$listRefresh1$ReportCollect
    if (!is.null(json_file)) {
      json_temp <- robyn_write(
        InputCollectRF, OutputCollectRF,
        select_model = selectID,
        export = TRUE, quiet = TRUE, ...
      )
      plots <- refresh_plots_json(OutputCollectRF, json_file = attr(json_temp, "json_file"), export = export)
    } else {
      plots <- try(refresh_plots(InputCollectRF, OutputCollectRF, ReportCollect, export = export))
    }

    if (export) {
      message(paste(">>> Exporting refresh CSVs into directory..."))
      write.csv(resultHypParamReport, paste0(OutputCollectRF$plot_folder, "report_hyperparameters.csv"))
      write.csv(xDecompAggReport, paste0(OutputCollectRF$plot_folder, "report_aggregated.csv"))
      write.csv(mediaVecReport, paste0(OutputCollectRF$plot_folder, "report_media_transform_matrix.csv"))
      write.csv(xDecompVecReport, paste0(OutputCollectRF$plot_folder, "report_alldecomp_matrix.csv"))
    }

    if (refreshLooper == 0) {
      refreshControl <- FALSE
      message("Reached maximum available date. No further refresh possible")
    }
  }

  # Save some parameters to print
  Robyn[["refresh"]] <- list(
    selectIDs = ReportCollect$selectIDs,
    refresh_steps = refresh_steps,
    refresh_mode = refresh_mode,
    refresh_trials = refresh_trials,
    refresh_iters = refresh_iters,
    plots = plots
  )

  # Save Robyn object locally
  Robyn <- Robyn[order(names(Robyn))]
  class(Robyn) <- c("robyn_refresh", class(Robyn))
  if (is.null(json_file)) {
    message(">> Exporting results: ", robyn_object)
    saveRDS(Robyn, file = robyn_object)
  } else {
    robyn_write(InputCollectRF, OutputCollectRF, select_model = selectID, ...)
  }
  return(invisible(Robyn))
}

#' @rdname robyn_refresh
#' @aliases robyn_refresh
#' @param x \code{robyn_refresh()} output.
#' @export
print.robyn_refresh <- function(x, ...) {
  top_models <- x$refresh$selectIDs
  top_models <- paste(top_models, sprintf("(%s)", 0:(length(top_models) - 1)))
  print(glued(
    "
Refresh Models: {length(top_models)}
Mode: {x$refresh$refresh_mode}
Steps: {x$refresh$refresh_steps}
Trials: {x$refresh$refresh_trials}
Iterations: {x$refresh$refresh_iters}

Models (IDs):
  {paste(top_models, collapse = ', ')}
"
  ))
}

#' @rdname robyn_refresh
#' @aliases robyn_refresh
#' @param x \code{robyn_refresh()} output.
#' @export
plot.robyn_refresh <- function(x, ...) plot((x$refresh$plots[[1]] / x$refresh$plots[[2]]), ...)

refresh_hyps <- function(initBounds, listOutputPrev, refresh_steps, rollingWindowLength) {
  initBoundsDis <- unlist(lapply(initBounds, function(x) ifelse(length(x) == 2, x[2] - x[1], 0)))
  newBoundsFreedom <- refresh_steps / rollingWindowLength
  message(">>> New bounds freedom: ", round(100 * newBoundsFreedom, 2), "%")
  hyper_updated_prev <- listOutputPrev$hyper_updated
  hypNames <- names(hyper_updated_prev)
  resultHypParam <- as_tibble(listOutputPrev$resultHypParam)
  for (h in seq_along(hypNames)) {
    hn <- hypNames[h]
    getHyp <- resultHypParam[, hn][[1]]
    getDis <- initBoundsDis[hn]
    if (hn == "lambda") {
      lambda_max <- unique(resultHypParam$lambda_max)
      lambda_min <- lambda_max * 0.0001
      getHyp <- getHyp / (lambda_max - lambda_min)
    }
    getRange <- initBounds[hn][[1]]
    if (length(getRange) == 2) {
      newLowB <- getHyp - getDis * newBoundsFreedom
      if (newLowB < getRange[1]) {
        newLowB <- getRange[1]
      }
      newUpB <- getHyp + getDis * newBoundsFreedom
      if (newUpB > getRange[2]) {
        newUpB <- getRange[2]
      }
      newBounds <- unname(c(newLowB, newUpB))
      hyper_updated_prev[hn][[1]] <- newBounds
    } else {
      hyper_updated_prev[hn][[1]] <- getRange
    }
  }
  return(hyper_updated_prev)
}
