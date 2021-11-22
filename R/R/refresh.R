# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_save, robyn_refresh

####################################################################
#' Save Robyn object
#'
#' Use \code{robyn_save()} to select and save the initial model.
#'
#' @inheritParams robyn_allocator
#' @return A list containing all information for the initial model.
#' @examples
#' \dontrun{
#' ## Get all model IDs in result from OutputCollect$allSolutions
#'
#' ## Select one from above
#' select_model <- "3_10_3"
#'
#' ## Save the robyn object. Overwriting old object needs confirmation.
#' robyn_object <- "~/Desktop/Robyn.RDS"
#' robyn_save(
#'   robyn_object = robyn_object,
#'   select_model = select_model,
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect
#' )
#' }
#' @export
robyn_save <- function(robyn_object,
                       select_model,
                       InputCollect,
                       OutputCollect) {
  check_robyn_object(robyn_object)

  if (!(select_model %in% OutputCollect$resultHypParam$solID)) {
    stop(paste0("'select_model' must be one of these values: ", paste(
      OutputCollect$resultHypParam$solID,
      collapse = ", "
    )))
  }

  if (file.exists(robyn_object)) {
    answer <- askYesNo(paste0(robyn_object, " already exists. Are you certain to overwrite it?"))
    if (answer == FALSE | is.na(answer)) {
      stop("stopped")
    }
  }

  OutputCollect$resultHypParam <- OutputCollect$resultHypParam[solID == select_model]
  OutputCollect$xDecompAgg <- OutputCollect$xDecompAgg[solID == select_model]
  OutputCollect$mediaVecCollect <- OutputCollect$mediaVecCollect[solID == select_model]
  OutputCollect$xDecompVecCollect <- OutputCollect$xDecompVecCollect[solID == select_model]
  OutputCollect$selectID <- select_model

  InputCollect$refreshCounter <- 0
  # listParamInit <- listParam
  listInit <- list(OutputCollect = OutputCollect, InputCollect = InputCollect)
  Robyn <- list(listInit = listInit)

  saveRDS(Robyn, file = robyn_object)
  # listOutputInit <- NULL;  listParamInit <- NULL
  # load("/Users/gufengzhou/Documents/GitHub/plots/listInit.RDS")
}


####################################################################
#' Build refresh model
#'
#' The \code{robyn_refresh()} function builds update models based on
#' the previously built models saved in the \code{Robyn.RDS} object specified
#' in \code{robyn_object}. For example, when updating the initial build with 4
#' weeks of new data, \code{robyn_refresh()} consumes the selected model of
#' the initial build. it sets lower and upper bounds of hyperparameters for the
#' new build around the selected hyperparameters of the previous build,
#' stabilizes the effect of baseline variables across old and new builds and
#' regulates the new effect share of media variables towards the latest
#' spend level. It returns aggregated result with all previous builds for
#' reporting purpose and produces reporting plots.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_allocator
#' @param dt_input A data.frame. Should include all previous data and newly added
#' data for the refresh.
#' @param dt_holidays A data.frame. Raw input holiday data. Load standard
#' Prophet holidays using \code{data("dt_prophet_holidays")}.
#' @param refresh_steps An integer. It controls how many time units the refresh
#' model build move forward. For example, \code{refresh_steps = 4} on weekly data
#' means the InputCollect$window_start & InputCollect$window_end move forward
#' 4 weeks.
#' @param refresh_mode A character. Options are "auto" and "manual". In auto mode,
#' the \code{robyn_refresh()} function builds refresh models with given
#' \code{refresh_steps} repeatedly until there's no more data available. I
#' manual mode, the \code{robyn_refresh()} only moves forward \code{refresh_steps}
#' only once.
#' @param refresh_iters An integer. Iterations per refresh. Rule of thumb is, the
#' more new data added, the more iterations needed. More reliable recommendation
#' still needs to be investigated.
#' @param refresh_trials An integer. Trials per refresh. Defaults to 5 trials.
#' More reliable recommendation still needs to be investigated.
#' @param plot_pareto A logical value. Set to \code{FALSE} to deactivate plotting
#' and saving model onepagers. Used when testing models.
#' @return A list. The Robyn object.
#' @examples
#' \dontrun{
#'
#' ## NOTE: must run \code{robyn_save()} to select and save an initial model first,
#' ## before refreshing below. The \code{robyn_refresh()} function is suitable for
#' ## updating within "reasonable periods".
#' ## Two situations are considered better to rebuild model:
#' ## 1, most data is new. If initial model has 100 weeks and 80 weeks new data is
#' ## added in refresh, it might be better to rebuild the model
#' ## 2, new variables are added
#'
#' # Set the Robyn object path
#' robyn_object <- "~/Desktop/Robyn.RDS"
#'
#' # Load new data
#' data("dt_simulated_weekly")
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
#' @export
robyn_refresh <- function(robyn_object,
                          plot_folder_sub = NULL,
                          dt_input = dt_input,
                          dt_holidays = dt_holidays,
                          refresh_steps = 4,
                          refresh_mode = "manual", # "auto", "manual"
                          refresh_iters = 1000,
                          refresh_trials = 3,
                          plot_pareto = TRUE) {
  refreshControl <- TRUE
  while (refreshControl) {

    ## load inital model
    if (!exists("robyn_object")) stop("Must speficy robyn_object")
    check_robyn_object(robyn_object)
    if (!file.exists(robyn_object)) {
      stop("File does not exist or is somewhere else. Check: ", robyn_object)
    } else {
      Robyn <- readRDS(robyn_object)
      objectPath <- dirname(robyn_object)
      objectName <- sub("'\\..*$", "", basename(robyn_object))
    }

    ## count refresh
    refreshCounter <- length(Robyn)
    refreshCounter
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

    ## get previous data
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

      message(paste0(">>> Refresh model nr.", refreshCounter - 1, " loaded"))

      ## model selection from previous build
      listOutputPrev$resultHypParam <- listOutputPrev$resultHypParam[bestModRF == TRUE]
      listOutputPrev$xDecompAgg <- listOutputPrev$xDecompAgg[bestModRF == TRUE]
      listOutputPrev$mediaVecCollect <- listOutputPrev$mediaVecCollect[bestModRF == TRUE]
      listOutputPrev$xDecompVecCollect <- listOutputPrev$xDecompVecCollect[bestModRF == TRUE]
    }

    InputCollectRF$refreshCounter <- refreshCounter
    InputCollectRF$refresh_steps <- refresh_steps
    if (refresh_steps >= InputCollectRF$rollingWindowLength) {
      stop("Refresh input data is completely new. Please rebuild model using robyn_run")
    }


    ## load new data
    dt_input <- as.data.table(dt_input)
    date_input <- check_datevar(dt_input, InputCollectRF$date_var)
    dt_input <- date_input$dt_input # sort date by ascending
    dt_holidays <- as.data.table(dt_holidays)
    InputCollectRF$dt_input <- dt_input
    InputCollectRF$dt_holidays <- dt_holidays

    #### update refresh model parameters

    ## refresh rolling window
    totalDates <- as.Date(dt_input[, get(InputCollectRF$date_var)])
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
      message(paste(">>> Refreshing model nr.", refreshCounter, "in", refresh_mode, "mode"))
      refreshControl <- FALSE
    } else {
      refreshLooper <- floor(as.numeric(difftime(max(totalDates), refreshEnd, units = "days")) /
        InputCollectRF$dayInterval / refresh_steps)
      message(paste(
        ">>> Refreshing model nr.", refreshCounter, "in",
        refresh_mode, "mode.", refreshLooper, "more to go..."
      ))
    }

    #### update refresh model parameters


    ## refresh hyperparameter bounds
    initBounds <- Robyn$listInit$InputCollect$hyperparameters
    initBoundsDis <- sapply(initBounds, function(x) {
      return(x[2] - x[1])
    })
    newBoundsFreedom <- refresh_steps / InputCollectRF$rollingWindowLength

    hyperparameters <- InputCollectRF$hyperparameters
    hypNames <- names(hyperparameters)
    for (h in 1:length(hypNames)) {
      getHyp <- listOutputPrev$resultHypParam[, get(hypNames[h])]
      getDis <- initBoundsDis[hypNames[h]]
      newLowB <- getHyp - getDis * newBoundsFreedom
      if (newLowB < initBounds[hypNames[h]][[1]][1]) {
        newLowB <- initBounds[hypNames[h]][[1]][1]
      }
      newUpB <- getHyp + getDis * newBoundsFreedom
      if (newUpB > initBounds[hypNames[h]][[1]][2]) {
        newUpB <- initBounds[hypNames[h]][[1]][2]
      }
      newBounds <- unname(c(newLowB, newUpB))
      hyperparameters[hypNames[h]][[1]] <- newBounds
    }
    InputCollectRF$hyperparameters <- hyperparameters

    ## refresh iterations and trial
    InputCollectRF$iterations <- refresh_iters
    InputCollectRF$trials <- refresh_trials

    #### update refresh model parameters

    ## feature engineering for refreshed data
    InputCollectRF <- robyn_engineering(InputCollect = InputCollectRF)

    ## refresh model with adjusted decomp.rssd

    OutputCollectRF <- robyn_run(
      InputCollect = InputCollectRF,
      plot_folder = objectPath,
      plot_folder_sub = plot_folder_sub,
      pareto_fronts = 1,
      refresh = TRUE,
      plot_pareto = plot_pareto
    )

    ## select winner model for current refresh
    # selectID <- OutputCollectRF$resultHypParam[which.min(decomp.rssd), solID] # min decomp.rssd selection
    OutputCollectRF$resultHypParam[, error_dis := sqrt(nrmse^2 + decomp.rssd^2)] # min error distance selection
    selectID <- OutputCollectRF$resultHypParam[which.min(error_dis), solID]
    OutputCollectRF$selectID <- selectID
    message(
      "Selected model ID: ", selectID, " for refresh model nr.",
      refreshCounter, " based on the smallest combined error of NRMSE & DECOMP.RSSD\n"
    )

    OutputCollectRF$resultHypParam[, bestModRF := solID == selectID]
    OutputCollectRF$xDecompAgg[, bestModRF := solID == selectID]
    OutputCollectRF$mediaVecCollect[, bestModRF := solID == selectID]
    OutputCollectRF$xDecompVecCollect[, bestModRF := solID == selectID]


    #### result collect & save
    if (refreshCounter == 1) {
      listOutputPrev$resultHypParam[, ":="(error_dis = sqrt(nrmse^2 + decomp.rssd^2),
        bestModRF = TRUE, refreshStatus = refreshCounter - 1)]
      listOutputPrev$xDecompAgg[, ":="(bestModRF = TRUE, refreshStatus = refreshCounter - 1)]
      listOutputPrev$mediaVecCollect[, ":="(bestModRF = TRUE, refreshStatus = refreshCounter - 1)]
      listOutputPrev$xDecompVecCollect[, ":="(bestModRF = TRUE, refreshStatus = refreshCounter - 1)]


      resultHypParamReport <- rbind(
        listOutputPrev$resultHypParam[bestModRF == TRUE],
        OutputCollectRF$resultHypParam[bestModRF == TRUE][, refreshStatus := refreshCounter]
      )
      xDecompAggReport <- rbind(
        listOutputPrev$xDecompAgg[bestModRF == TRUE],
        OutputCollectRF$xDecompAgg[bestModRF == TRUE][, refreshStatus := refreshCounter]
      )
      mediaVecReport <- rbind(
        listOutputPrev$mediaVecCollect[
          bestModRF == TRUE & ds >= (refreshStart - InputCollectRF$dayInterval * refresh_steps) &
            ds <= (refreshEnd - InputCollectRF$dayInterval * refresh_steps)
        ][, ds := as.IDate(ds)],
        OutputCollectRF$mediaVecCollect[
          bestModRF == TRUE & ds >= InputCollectRF$refreshAddedStart &
            ds <= refreshEnd
        ][, ':='(refreshStatus = refreshCounter, ds = as.IDate(ds))]
      )
      mediaVecReport <- mediaVecReport[order(type, ds, refreshStatus)]
      xDecompVecReport <- rbind(
        listOutputPrev$xDecompVecCollect[bestModRF == TRUE][, ds := as.IDate(ds)],
        OutputCollectRF$xDecompVecCollect[
          bestModRF == TRUE & ds >= InputCollectRF$refreshAddedStart &
            ds <= refreshEnd
        ][, ':='(refreshStatus = refreshCounter, ds = as.IDate(ds))]
      )
    } else {
      resultHypParamReport <- rbind(
        listReportPrev$resultHypParamReport,
        OutputCollectRF$resultHypParam[bestModRF == TRUE][
          , refreshStatus := refreshCounter])
      xDecompAggReport <- rbind(
        listReportPrev$xDecompAggReport,
        OutputCollectRF$xDecompAgg[bestModRF == TRUE][
          , refreshStatus := refreshCounter])
      mediaVecReport <- rbind(
        listReportPrev$mediaVecReport,
        OutputCollectRF$mediaVecCollect[
          bestModRF == TRUE & ds >= InputCollectRF$refreshAddedStart &
            ds <= refreshEnd
        ][, ':='(refreshStatus = refreshCounter, ds = as.IDate(ds))]
      )
      mediaVecReport <- mediaVecReport[order(type, ds, refreshStatus)]
      xDecompVecReport <- rbind(
        listReportPrev$xDecompVecReport,
        OutputCollectRF$xDecompVecCollect[
          bestModRF == TRUE & ds >= InputCollectRF$refreshAddedStart &
            ds <= refreshEnd
        ][, ':='(refreshStatus = refreshCounter, ds = as.IDate(ds))]
      )
    }

    fwrite(resultHypParamReport, paste0(OutputCollectRF$plot_folder, "report_hyperparameters.csv"))
    fwrite(xDecompAggReport, paste0(OutputCollectRF$plot_folder, "report_aggregated.csv"))
    fwrite(mediaVecReport, paste0(OutputCollectRF$plot_folder, "report_media_transform_matrix.csv"))
    fwrite(xDecompVecReport, paste0(OutputCollectRF$plot_folder, "report_alldecomp_matrix.csv"))


    #### reporting plots
    ## actual vs fitted

    xDecompVecReportPlot <- copy(xDecompVecReport)
    xDecompVecReportPlot[, ":="(refreshStart = min(ds),
      refreshEnd = max(ds)), by = "refreshStatus"]
    xDecompVecReportPlot[, duration := as.numeric(
      (refreshEnd - refreshStart + InputCollectRF$dayInterval) / InputCollectRF$dayInterval
    )]
    getRefreshStarts <- sort(unique(xDecompVecReportPlot$refreshStart))[-1]
    dt_refreshDates <- unique(xDecompVecReportPlot[
      , .(refreshStatus = as.factor(refreshStatus), refreshStart, refreshEnd, duration)
    ])
    dt_refreshDates[, label := ifelse(dt_refreshDates$refreshStatus == 0,
      paste0(
        "initial: ", dt_refreshDates$refreshStart, ", ",
        dt_refreshDates$duration, InputCollectRF$intervalType, "s"
      ),
      paste0(
        "refresh nr.", dt_refreshDates$refreshStatus, ": ", dt_refreshDates$refreshStart,
        ", ", dt_refreshDates$duration, InputCollectRF$intervalType, "s"
      )
    )]

    xDecompVecReportMelted <- melt.data.table(xDecompVecReportPlot[
      , .(ds, refreshStart, refreshEnd, refreshStatus, actual = dep_var, predicted = depVarHat)
    ],
    id.vars = c("ds", "refreshStatus", "refreshStart", "refreshEnd")
    )
    pFitRF <- ggplot(data = xDecompVecReportMelted) +
      geom_line(aes(x = ds, y = value, color = variable)) +
      geom_rect(
        data = dt_refreshDates, aes(xmin = refreshStart, xmax = refreshEnd, fill = refreshStatus),
        ymin = -Inf, ymax = Inf, alpha = 0.2
      ) +
      theme(
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), # legend.position = c(0.1, 0.8),
        legend.background = element_rect(fill = alpha("white", 0.4)),
      ) +
      scale_fill_brewer(palette = "BuGn") +
      geom_text(data = dt_refreshDates, mapping = aes(
        x = refreshStart, y = max(xDecompVecReportMelted$value), label = label,
        angle = 270, hjust = -0.1, vjust = -0.2
      ), color = "gray40") +
      # geom_vline(xintercept = getRefreshStarts, linetype="dotted") +
      labs(
        title = "Model refresh: actual vs. predicted response",
        subtitle = paste0(
          "Assembled rsq: ", round(get_rsq(
            true = xDecompVecReportPlot$dep_var, predicted = xDecompVecReportPlot$depVarHat
          ), 2)
          # ,"\nRefresh dates: ", paste(getRefreshStarts, collapse = ", ")
        ),
        x = "date", y = "response"
      )
    # print(pFitRF)
    ggsave(
      filename = paste0(OutputCollectRF$plot_folder, "report_actual_fitted.png"),
      plot = pFitRF,
      dpi = 900, width = 12, height = 8
    )

    ## stacked bar plot

    xDecompAggReportPlotBase <- xDecompAggReport[
      rn %in% c(InputCollectRF$prophet_vars, "(Intercept)"),
      .(rn, perc = ifelse(refreshStatus == 0, xDecompPerc, xDecompPercRF), refreshStatus)
    ]
    xDecompAggReportPlotBase <- xDecompAggReportPlotBase[
      , .(variable = "baseline", percentage = sum(perc)),
      by = refreshStatus
    ][, roi_total := NA]
    xDecompAggReportPlot <- xDecompAggReport[
      !(rn %in% c(InputCollectRF$prophet_vars, "(Intercept)")),
      .(refreshStatus, variable = rn, percentage = ifelse(refreshStatus == 0, xDecompPerc, xDecompPercRF), roi_total)
    ]
    xDecompAggReportPlot <- rbind(xDecompAggReportPlot, xDecompAggReportPlotBase)[order(refreshStatus, -variable)]
    xDecompAggReportPlot[, refreshStatus := ifelse(refreshStatus == 0, "init.mod", paste0("refresh", refreshStatus))]
    ySecScale <- max(na.omit(xDecompAggReportPlot$roi_total)) / max(xDecompAggReportPlot$percentage) * 0.75
    ymax <- max(c(na.omit(xDecompAggReportPlot$roi_total) / ySecScale, xDecompAggReportPlot$percentage)) * 1.1

    pBarRF <- ggplot(data = xDecompAggReportPlot, mapping = aes(x = variable, y = percentage, fill = variable)) +
      geom_bar(alpha = 0.8, position = "dodge", stat = "identity") +
      facet_wrap(~refreshStatus, scales = "free") +
      scale_fill_manual(values = robyn_palette()$fill) +
      geom_text(aes(label = paste0(round(percentage * 100, 1), "%")),
        size = 3,
        position = position_dodge(width = 0.9), hjust = -0.25
      ) +
      geom_point(aes(x = variable, y = roi_total / ySecScale, color = variable),
        size = 4, shape = 17, na.rm = TRUE,
        data = xDecompAggReportPlot
      ) +
      geom_text(aes(label = round(roi_total, 2), x = variable, y = roi_total / ySecScale),
        size = 3, na.rm = TRUE, hjust = -0.4, fontface = "bold",
        position = position_dodge(width = 0.9),
        data = xDecompAggReportPlot
      ) +
      scale_color_manual(values = robyn_palette()$fill) +
      scale_y_continuous(
        sec.axis = sec_axis(~ . * ySecScale), breaks = seq(0, ymax, 0.2),
        limits = c(0, ymax), name = "roi_total"
      ) +
      coord_flip() +
      theme(legend.position = "none", axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
      labs(
        title = "Model refresh: Decomposition & paid media ROI",
        subtitle = paste0(
          "baseline includes intercept and all prophet vars: ",
          paste(InputCollectRF$prophet_vars, collapse = ", ")
        )
      )
    # print(pBarRF)
    ggsave(
      filename = paste0(OutputCollectRF$plot_folder, "report_decomposition.png"),
      plot = pBarRF,
      dpi = 900, width = 12, height = 8
    )

    #### save result objects

    ReportCollect <- list(
      resultHypParamReport = resultHypParamReport,
      xDecompAggReport = xDecompAggReport,
      mediaVecReport = mediaVecReport,
      xDecompVecReport = xDecompVecReport
    )
    # assign("ReportCollect", ReportCollect)

    listHolder <- list(
      InputCollect = InputCollectRF,
      OutputCollect = OutputCollectRF,
      ReportCollect = ReportCollect
    )


    listNameUpdate <- paste0("listRefresh", refreshCounter)
    # assign(listNameUpdate, listHolder)
    Robyn[[listNameUpdate]] <- listHolder

    saveRDS(Robyn, file = robyn_object)

    if (refreshLooper == 0) {
      refreshControl <- FALSE
      message("Reached maximum available date. No further refresh possible")
    }
  }
  invisible(Robyn)
}
