# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Export Robyn Model to Local File
#'
#' Use \code{robyn_save()} to select and save as .RDS file the initial model.
#'
#' @inheritParams robyn_allocator
#' @return (Invisible) list with filename and summary. Class: \code{robyn_save}.
#' @export
robyn_save <- function(robyn_object,
                       select_model,
                       InputCollect,
                       OutputCollect,
                       quiet = FALSE) {
  check_robyn_object(robyn_object)
  if (is.null(select_model)) select_model <- OutputCollect[["selectID"]]
  if (!(select_model %in% OutputCollect$resultHypParam$solID)) {
    stop(paste0("'select_model' must be one of these values: ", paste(
      OutputCollect$resultHypParam$solID,
      collapse = ", "
    )))
  }

  output <- list(
    robyn_object = robyn_object,
    select_model = select_model,
    summary = filter(
      OutputCollect$xDecompAgg,
      .data$solID == select_model, !is.na(.data$mean_spend)
    ) %>%
      select(
        channel = .data$rn, .data$coef, .data$mean_spend, .data$mean_response, .data$roi_mean,
        .data$total_spend, total_response = .data$xDecompAgg, .data$roi_total
      ),
    plot = robyn_onepagers(InputCollect, OutputCollect, select_model, quiet = TRUE, export = FALSE)
  )
  if (InputCollect$dep_var_type == "conversion") {
    colnames(output$summary) <- gsub("roi_", "cpa_", colnames(output$summary))
  }
  class(output) <- c("robyn_save", class(output))

  if (file.exists(robyn_object)) {
    if (!quiet) {
      answer <- askYesNo(paste0(robyn_object, " already exists. Are you certain to overwrite it?"))
    } else {
      answer <- TRUE
    }
    if (answer == FALSE | is.na(answer)) {
      message("Stopped export to avoid overwriting")
      return(invisible(output))
    }
  }

  OutputCollect$resultHypParam <- OutputCollect$resultHypParam[
    OutputCollect$resultHypParam$solID == select_model,
  ]
  OutputCollect$xDecompAgg <- OutputCollect$xDecompAgg[
    OutputCollect$resultHypParam$solID == select_model,
  ]
  OutputCollect$mediaVecCollect <- OutputCollect$mediaVecCollect[
    OutputCollect$resultHypParam$solID == select_model,
  ]
  OutputCollect$xDecompVecCollect <- OutputCollect$xDecompVecCollect[
    OutputCollect$resultHypParam$solID == select_model,
  ]
  OutputCollect$selectID <- select_model

  InputCollect$refreshCounter <- 0
  listInit <- list(OutputCollect = OutputCollect, InputCollect = InputCollect)
  Robyn <- list(listInit = listInit)

  saveRDS(Robyn, file = robyn_object)
  if (!quiet) message("Exported results: ", robyn_object)
  return(invisible(output))
}

#' @rdname robyn_save
#' @aliases robyn_save
#' @param x \code{robyn_save()} output.
#' @export
print.robyn_save <- function(x, ...) {
  print(glued(
    "
  Exported file: {x$robyn_object}
  Exported model: {x$select_model}

  Media Summary for Selected Model:
  "
  ))
  print(x$summary)
}

#' @rdname robyn_save
#' @aliases robyn_save
#' @param x \code{robyn_save()} output.
#' @export
plot.robyn_save <- function(x, ...) plot(x$plot[[1]], ...)


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
#' @param dt_input data.frame. Should include all previous data and newly added
#' data for the refresh.
#' @param dt_holidays data.frame. Raw input holiday data. Load standard
#' Prophet holidays using \code{data("dt_prophet_holidays")}.
#' @param refresh_steps Integer. It controls how many time units the refresh
#' model build move forward. For example, \code{refresh_steps = 4} on weekly data
#' means the InputCollect$window_start & InputCollect$window_end move forward
#' 4 weeks.
#' @param refresh_mode Character. Options are "auto" and "manual". In auto mode,
#' the \code{robyn_refresh()} function builds refresh models with given
#' \code{refresh_steps} repeatedly until there's no more data available. I
#' manual mode, the \code{robyn_refresh()} only moves forward \code{refresh_steps}
#' only once.
#' @param refresh_iters Integer. Iterations per refresh. Rule of thumb is, the
#' more new data added, the more iterations needed. More reliable recommendation
#' still needs to be investigated.
#' @param refresh_trials Integer. Trials per refresh. Defaults to 5 trials.
#' More reliable recommendation still needs to be investigated.
#' @param version_prompt Logical. If FALSE, the model refresh version will be
#' selected based on the smallest combined error of normalised NRMSE & DECOMP.RSSD.
#' If TRUE, a prompt will be presented to the user to select one of the refreshed
#' models (one-pagers and pareto csv files will already be generated).
#' @param ... Additional parameters to overwrite original custom parameters
#' passed into initial model.
#' @return List. The Robyn object, class \code{robyn_refresh}.
#' @examples
#' \dontrun{
#' # Set the (pre-trained and exported) Robyn object path
#' robyn_object <- "~/Desktop/Robyn.RDS"
#' # Load dummy data
#' data("dt_simulated_weekly")
#' # Load holidays data
#' data("dt_prophet_holidays")
#'
#' # Run \code{robyn_refresh()} with 13 weeks cadance in auto mode
#' Robyn <- robyn_refresh(
#'   robyn_object = robyn_object,
#'   dt_input = dt_simulated_weekly,
#'   dt_holidays = dt_prophet_holidays,
#'   refresh_steps = 13,
#'   refresh_mode = "auto",
#'   refresh_iters = 200,
#'   refresh_trials = 5
#' )
#'
#' # Run \code{robyn_refresh()} with 4 weeks cadance in manual mode
#' Robyn <- robyn_refresh(
#'   robyn_object = robyn_object,
#'   dt_input = dt_simulated_weekly,
#'   dt_holidays = dt_prophet_holidays,
#'   refresh_steps = 4,
#'   refresh_mode = "manual",
#'   refresh_iters = 200,
#'   refresh_trials = 5
#' )
#' }
#' @return List. Same as \code{robyn_run()} but with refreshed models.
#' @export
robyn_refresh <- function(robyn_object,
                          plot_folder_sub = NULL,
                          dt_input = dt_input,
                          dt_holidays = dt_holidays,
                          refresh_steps = 4,
                          refresh_mode = "manual", # "auto", "manual"
                          refresh_iters = 1000,
                          refresh_trials = 3,
                          plot_pareto = TRUE,
                          version_prompt = FALSE,
                          export = TRUE,
                          ...) {
  refreshControl <- TRUE
  while (refreshControl) {

    ## Check for NA values
    check_nas(dt_input)
    check_nas(dt_holidays)

    ## Load initial model
    if (!exists("robyn_object")) stop("Must speficy robyn_object")
    check_robyn_object(robyn_object)
    if (!file.exists(robyn_object)) {
      stop("File does not exist or is somewhere else. Check: ", robyn_object)
    } else {
      Robyn <- readRDS(robyn_object)
      objectPath <- dirname(robyn_object)
      objectName <- sub("'\\..*$", "", basename(robyn_object))
    }

    ## Count refresh
    refreshCounter <- length(Robyn)
    objectCheck <- if (refreshCounter == 1) {
      c("listInit")
    } else {
      c("listInit", paste0("listRefresh", 1:(refreshCounter - 1)))
    }
    if (!all(objectCheck %in% names(Robyn))) {
      stop("Saved Robyn object is corrupted. It should contain ", paste(
        objectCheck,
        collapse = ",", ". Please rerun model."
      ))
    }

    ## Check rule of thumb: 50% of data shouldn't be new
    original_periods <- nrow(Robyn$listInit$InputCollect$dt_modRollWind)
    new_periods <- nrow(filter(
      dt_input, get(Robyn$listInit$InputCollect$date_var) > Robyn$listInit$InputCollect$window_end
    ))
    it <- Robyn$listInit$InputCollect$intervalType
    if (new_periods > 0.5 * (original_periods + new_periods)) {
      warning(sprintf(
        paste(
          "We recommend re-building a model rather than refreshing this one.",
          "More than 50%% of your refresh data (%s %ss) is new data (%s %ss)"
        ),
        original_periods + new_periods, it, new_periods, it
      ))
    }

    ## Get previous data
    if (refreshCounter == 1) {
      InputCollectRF <- Robyn$listInit$InputCollect
      listOutputPrev <- Robyn$listInit$OutputCollect
      InputCollectRF$xDecompAggPrev <- listOutputPrev$xDecompAgg
      message(">>> Initial model loaded")
      if (length(unique(Robyn$listInit$OutputCollect$resultHypParam$solID)) > 1) {
        stop("Run robyn_save first to select one initial model")
      }
    } else {
      listName <- paste0("listRefresh", refreshCounter - 1)
      InputCollectRF <- Robyn[[listName]][["InputCollect"]]
      listOutputPrev <- Robyn[[listName]][["OutputCollect"]]
      listReportPrev <- Robyn[[listName]][["ReportCollect"]]

      message(paste(">>> Loaded refresh model:", refreshCounter - 1))

      ## model selection from previous build
      which_bestModRF <- which(listOutputPrev$resultHypParam$bestModRF == TRUE)
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
      date_input <- as.data.frame(dt_input)
      dt_holidays <- as.data.frame(dt_holidays)
      date_input <- check_datevar(dt_input, InputCollectRF$date_var)
      dt_input <- date_input$dt_input # sort date by ascending
      InputCollectRF$dt_input <- dt_input
      InputCollectRF$dt_holidays <- dt_holidays
    }

    #### Update refresh model parameters

    ## Refresh rolling window
    totalDates <- as.Date(dt_input[, InputCollectRF$date_var][[1]])
    refreshStart <- as.Date(InputCollectRF$window_start) + InputCollectRF$dayInterval * refresh_steps
    refreshEnd <- as.Date(InputCollectRF$window_end) + InputCollectRF$dayInterval * refresh_steps
    InputCollectRF$refreshAddedStart <- as.Date(InputCollectRF$window_end) + InputCollectRF$dayInterval
    InputCollectRF$window_start <- refreshStart
    InputCollectRF$window_end <- refreshEnd

    refreshStartWhich <- which.min(abs(difftime(totalDates, as.Date(refreshStart), units = "days")))
    refreshEndWhich <- which.min(abs(difftime(totalDates, as.Date(refreshEnd), units = "days")))
    InputCollectRF$rollingWindowStartWhich <- refreshStartWhich
    InputCollectRF$rollingWindowEndWhich <- refreshEndWhich
    InputCollectRF$rollingWindowLength <- refreshEndWhich - refreshStartWhich + 1

    if (refreshEnd > max(totalDates)) {
      stop("Not enough data for this refresh. Input data from date ", refreshEnd, " or later required")
    }
    if (refresh_mode == "manual") {
      refreshLooper <- 1
      message(paste(">>> Refreshing model", refreshCounter, "in", refresh_mode, "mode"))
      refreshControl <- FALSE
    } else {
      refreshLooper <- floor(as.numeric(difftime(max(totalDates), refreshEnd, units = "days")) /
        InputCollectRF$dayInterval / refresh_steps)
      message(paste(
        ">>> Refreshing model", refreshCounter, "in",
        refresh_mode, "mode.", refreshLooper, "more to go..."
      ))
    }

    ## Refresh hyperparameter bounds
    initBounds <- Robyn$listInit$OutputCollect$hyper_updated
    initBoundsDis <- sapply(initBounds, function(x) ifelse(length(x) == 2, x[2] - x[1], 0))
    newBoundsFreedom <- refresh_steps / InputCollectRF$rollingWindowLength

    hyper_updated_prev <- listOutputPrev$hyper_updated
    hypNames <- names(hyper_updated_prev)
    for (h in 1:length(hypNames)) {
      hn <- hypNames[h]
      getHyp <- listOutputPrev$resultHypParam[, hn][[1]]
      getDis <- initBoundsDis[hn]
      if (hn == "lambda") {
        lambda_max <- unique(listOutputPrev$resultHypParam$lambda_max)
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
    InputCollectRF$hyperparameters <- hyper_updated_prev

    #### Update refresh model parameters

    ## Feature engineering for refreshed data
    # Note that if custom prophet parameters were passed initially, will be used again unless changed in ...
    InputCollectRF <- robyn_engineering(InputCollectRF, ...)

    ## refresh model with adjusted decomp.rssd

    OutputCollectRF <- robyn_run(
      InputCollect = InputCollectRF,
      plot_folder = objectPath,
      plot_folder_sub = plot_folder_sub,
      calibration_constraint = listOutputPrev[["calibration_constraint"]],
      add_penalty_factor = listOutputPrev[["add_penalty_factor"]],
      iterations = refresh_iters,
      trials = refresh_trials,
      pareto_fronts = 3,
      refresh = TRUE,
      plot_pareto = plot_pareto,
      ...
    )

    ## Select winner model for current refresh
    # selectID <- OutputCollectRF$resultHypParam[which.min(decomp.rssd), solID] # min decomp.rssd selection
    # norm_nrmse <- .min_max_norm(OutputCollectRF$resultHypParam$nrmse)
    # norm_rssd <- .min_max_norm(OutputCollectRF$resultHypParam$decomp.rssd)
    OutputCollectRF$resultHypParam <- OutputCollectRF$resultHypParam %>%
      mutate(error_dis = sqrt(.min_max_norm(.data$nrmse)^2 + .min_max_norm(.data$decomp.rssd)^2))
    if (version_prompt) {
      selectID <- readline("Input model version to use for the refresh: ")
      OutputCollectRF$selectID <- selectID
      message(
        "Selected model ID: ", selectID, " for refresh model nr.",
        refreshCounter, " based on your input\n"
      )
    } else {
      selectID <- OutputCollectRF$resultHypParam$solID[which.min(OutputCollectRF$resultHypParam$error_dis)]
      OutputCollectRF$selectID <- selectID
      message(
        "Selected model ID: ", selectID, " for refresh model #",
        refreshCounter, " based on the smallest combined error of normalised NRMSE & DECOMP.RSSD"
      )
    }

    # Add bestModRF column to multiple data.frames
    these <- c("resultHypParam", "xDecompAgg", "mediaVecCollect", "xDecompVecCollect")
    for (tb in these) {
      OutputCollectRF[[tb]] <- mutate(
        OutputCollectRF[[tb]],
        bestModRF = .data$solID == selectID
      )
    }

    #### Result collect & save
    if (refreshCounter == 1) {
      listOutputPrev$resultHypParam <- listOutputPrev$resultHypParam %>%
        mutate(error_dis = sqrt(.data$nrmse^2 + .data$decomp.rssd^2))

      # Add bestModRF and refreshCounter column to multiple data.frames
      these <- c("resultHypParam", "xDecompAgg", "mediaVecCollect", "xDecompVecCollect")
      for (tb in these) {
        listOutputPrev[[tb]] <- mutate(
          listOutputPrev[[tb]],
          bestModRF = TRUE, refreshCounter = refreshCounter - 1
        )
      }

      resultHypParamReport <- listOutputPrev$resultHypParam %>%
        filter(.data$bestModRF == TRUE) %>%
        bind_rows(
          OutputCollectRF$resultHypParam %>%
            filter(.data$bestModRF == TRUE)
        )

      xDecompAggReport <- listOutputPrev$xDecompAgg %>%
        filter(.data$bestModRF == TRUE) %>%
        bind_rows(
          OutputCollectRF$xDecompAgg %>%
            filter(.data$bestModRF == TRUE)
        )

      mediaVecReport <- listOutputPrev$mediaVecCollect %>%
        filter(
          .data$bestModRF == TRUE,
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
        arrange(.data$type, .data$ds, .data$refreshCounter)

      xDecompVecReport <- listOutputPrev$xDecompVecCollect %>%
        filter(.data$bestModRF == TRUE) %>%
        bind_rows(
          OutputCollectRF$xDecompVecCollect %>%
            filter(
              .data$bestModRF == TRUE,
              .data$ds >= InputCollectRF$refreshAddedStart,
              .data$ds <= refreshEnd
            )
        )
    } else {
      resultHypParamReport <- listReportPrev$resultHypParamReport %>%
        bind_rows(
          filter(OutputCollectRF$resultHypParam, .data$bestModRF == TRUE)
        )

      xDecompAggReport <- listReportPrev$xDecompAggReport %>%
        bind_rows(
          filter(OutputCollectRF$xDecompAgg, .data$bestModRF == TRUE)
        )

      mediaVecReport <- listReportPrev$mediaVecReport %>%
        bind_rows(
          filter(
            OutputCollectRF$mediaVecCollect,
            .data$bestModRF == TRUE,
            .data$ds >= InputCollectRF$refreshAddedStart,
            .data$ds <= refreshEnd
          )
        ) %>%
        arrange(.data$type, .data$ds, .data$refreshCounter)

      xDecompVecReport <- listReportPrev$xDecompVecReport %>%
        bind_rows(
          filter(
            OutputCollectRF$xDecompVecCollect,
            .data$bestModRF == TRUE,
            .data$ds >= InputCollectRF$refreshAddedStart,
            .data$ds <= refreshEnd
          )
        )
    }

    if (export) {
      write.csv(resultHypParamReport, paste0(OutputCollectRF$plot_folder, "report_hyperparameters.csv"))
      write.csv(xDecompAggReport, paste0(OutputCollectRF$plot_folder, "report_aggregated.csv"))
      write.csv(mediaVecReport, paste0(OutputCollectRF$plot_folder, "report_media_transform_matrix.csv"))
      write.csv(xDecompVecReport, paste0(OutputCollectRF$plot_folder, "report_alldecomp_matrix.csv"))
    }

    #### Save result objects
    ReportCollect <- list(
      resultHypParamReport = resultHypParamReport,
      xDecompAggReport = xDecompAggReport,
      mediaVecReport = mediaVecReport,
      xDecompVecReport = xDecompVecReport
    )

    listNameUpdate <- paste0("listRefresh", refreshCounter)
    Robyn[[listNameUpdate]] <- list(
      InputCollect = InputCollectRF,
      OutputCollect = OutputCollectRF,
      ReportCollect = ReportCollect
    )
    saveRDS(Robyn, file = robyn_object)

    #### Reporting plots
    # InputCollectRF <- Robyn$listRefresh1$InputCollect
    # OutputCollectRF <- Robyn$listRefresh1$OutputCollectRF
    # ReportCollect <- Robyn$listRefresh1$ReportCollect
    plots <- try(refresh_plots(InputCollectRF, OutputCollectRF, ReportCollect, export = export))

    if (refreshLooper == 0) {
      refreshControl <- FALSE
      message("Reached maximum available date. No further refresh possible")
    }
  }

  # Save some parameters to print
  Robyn[["refresh_steps"]] <- refresh_steps
  Robyn[["refresh_mode"]] <- refresh_mode
  Robyn[["refresh_trials"]] <- refresh_trials
  Robyn[["refresh_iters"]] <- refresh_iters
  Robyn[["refresh_plots"]] <- plots

  class(Robyn) <- c("robyn_refresh", class(Robyn))
  return(invisible(Robyn))
}

#' @rdname robyn_refresh
#' @aliases robyn_refresh
#' @param x \code{robyn_refresh()} output.
#' @export
print.robyn_refresh <- function(x, ...) {
  rf_list <- x[grep("Refresh", names(x), value = TRUE)]
  top_models <- data.frame(sapply(rf_list, function(y) y$ReportCollect$resultHypParamReport$solID))
  print(glued(
    "
Refresh Models: {length(rf_list)}
Mode: {x$refresh_mode}
Steps: {x$refresh_steps}
Trials: {x$refresh_trials}
Iterations: {x$refresh_iters}

Models (IDs):
  {paste(top_models_plain, collapse = ', ')}
",
    top_models_plain = sapply(seq_along(top_models), function(i) {
      paste(names(top_models), paste(top_models[, i], collapse = ", "), sep = ": ")
    })
  ))
}
