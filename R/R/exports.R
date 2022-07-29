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
                       select_model = NULL,
                       InputCollect,
                       OutputCollect,
                       quiet = FALSE) {
  check_robyn_name(robyn_object)
  if (is.null(select_model)) select_model <- OutputCollect[["selectID"]]
  if (!select_model %in% OutputCollect$allSolutions) {
    stop(paste0("Input 'select_model' must be one of these values: ", paste(
      OutputCollect$allSolutions,
      collapse = ", "
    )))
  }

  summary <- filter(OutputCollect$xDecompAgg, .data$solID == select_model) %>%
    select(variable = .data$rn, .data$coef, decomp = .data$xDecompPerc,
           .data$total_spend, mean_non0_spend = .data$mean_spend)

  hyps_name <- c("thetas", "shapes", "scales", "alphas", "gammas")
  regex <- paste(paste0("_", hyps_name), collapse = "|")
  hyps <- filter(OutputCollect$resultHypParam,.data$solID == select_model) %>%
    select(contains(hyps_name)) %>%
    tidyr::gather() %>%
    tidyr::separate(.data$key, into = c("channel", "none"),
                    sep = regex, remove = FALSE) %>%
    mutate(hyperparameter = gsub("^.*_", "", .data$key)) %>%
    select(.data$channel, .data$hyperparameter, .data$value) %>%
    tidyr::spread(key = "hyperparameter", value = "value")

  output <- list(
    robyn_object = robyn_object,
    select_model = select_model,
    summary = summary,
    hyperparameters = hyps,
    window = c(InputCollect$window_start, InputCollect$window_end),
    periods = InputCollect$rollingWindowLength,
    interval = InputCollect$intervalType,
    adstock = InputCollect$adstock,
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
  listInit <- list(InputCollect = InputCollect, OutputCollect = OutputCollect)
  Robyn <- list(listInit = listInit)

  class(Robyn) <- c("robyn_exported", class(Robyn))
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
  Window: {x$window[1]} to {x$window[2]} ({x$periods} {x$interval}s)

  Summary Values on Selected Model:"))

  print(x$summary %>%
          mutate(decomp = formatNum(100 * .data$decomp, pos = "%")) %>%
          dplyr::mutate_if(is.numeric, function(x) formatNum(x, 4, abbr = TRUE)) %>%
          replace(., . == "NA", '-'))

  print(glued(
    "\n\n  Hyper-parameters for channel transformations:\n    Adstock: {x$adstock}"))

  print(x$hyperparameters)
}

#' @rdname robyn_save
#' @aliases robyn_save
#' @param x \code{robyn_save()} output.
#' @export
plot.robyn_save <- function(x, ...) plot(x$refresh$plot[[1]], ...)

#' @rdname robyn_save
#' @aliases robyn_save
#' @return (Invisible) list with imported results
#' @export
robyn_load <- function(robyn_object, select_build = NULL, quiet = FALSE) {
  if ("robyn_exported" %in% class(robyn_object) | is.list(robyn_object)) {
    Robyn <- robyn_object
    objectPath <- Robyn$listInit$OutputCollect$plot_folder
    robyn_object <- paste0(objectPath, "/Robyn_", Robyn$listInit$OutputCollect$selectID, ".RDS")
    if (!dir.exists(objectPath)) {
      stop("Directory does not exist or is somewhere else. Check: ", objectPath)
    }
  } else {
    if (!"character" %in% class(robyn_object)) {
      stop("Input 'robyn_object' must be a character input or 'robyn_exported' object")
    }
    check_robyn_name(robyn_object)
    Robyn <- readRDS(robyn_object)
    objectPath <- dirname(robyn_object)
  }
  select_build_all <- 0:(length(Robyn) - 1)
  if (is.null(select_build)) {
    select_build <- max(select_build_all)
    if (!quiet) {
      message(
        ">>> Loaded Model: ",
        ifelse(select_build == 0, "Initial model", paste0("Refresh model #", select_build))
      )
    }
  }
  if (!(select_build %in% select_build_all) | length(select_build) != 1) {
    stop("Input 'select_build' must be one value of ", paste(select_build_all, collapse = ", "))
  }
  listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
  InputCollect <- Robyn[[listName]][["InputCollect"]]
  OutputCollect <- Robyn[[listName]][["OutputCollect"]]
  select_model <- OutputCollect$selectID
  output <- list(
    Robyn = Robyn,
    InputCollect = InputCollect,
    OutputCollect = OutputCollect,
    select_model = select_model,
    objectPath = objectPath,
    robyn_object = robyn_object
  )
  return(invisible(output))
}
