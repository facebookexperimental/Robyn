# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Input Data Check & Transformation
#'
#' \code{robyn_json()} generates a JSON file with all the information
#' required to replicate a single Robyn model.
#'
#' @inheritParams robyn_outputs
#' @param select_model Character. Which model (by \code{solID}) do you
#' want to export into a JSON file?
#' @param dir Character. Existing directory to export JSON file to.
#' @examples
#' \dontrun{
#' InputCollectJSON <- robyn_inputs(
#'   dt_input = Robyn::dt_simulated_weekly,
#'   dt_holidays = Robyn::dt_prophet_holidays,
#'   json_file = "~/Desktop/RobynModel-1_29_12.json"
#' )
#' print(InputCollectJSON)
#' }
#' @return (invisible) List. Contains all inputs and outputs of exported model.
#' Class: \code{robyn_json}.
#' @export
robyn_json <- function(InputCollect, OutputCollect = NULL, select_model = NULL, dir = getwd()) {

  # Checks
  stopifnot(inherits(InputCollect, "robyn_inputs"))
  if (!is.null(OutputCollect)) {
    stopifnot(inherits(OutputCollect, "robyn_outputs"))
    stopifnot(select_model %in% OutputCollect$allSolutions)
  }

  # InputCollect JSON
  ret <- list()
  skip <- which(sapply(InputCollect, function(x) is.list(x) | is.null(x)))
  skip <- skip[!names(skip) %in% c("calibration_input", "hyperparameters", "custom_params")]
  ret[["InputCollect"]] <- inputs <- InputCollect[-skip]
  # toJSON(inputs, pretty = TRUE)

  # ExportedModel JSON (improve: exclude InputCollect stuff, add the rest)
  if (!is.null(OutputCollect)) {
    ExportedModel <- robyn_save(
      InputCollect = InputCollect,
      OutputCollect = OutputCollect,
      select_model = select_model,
      quiet = TRUE
    )
    # which(sapply(OutputCollect, function(x) is.list(x) | is.null(x)))
    these <- which(!names(ExportedModel) %in% c("robyn_object", "plot"))
    ret[["ExportedModel"]] <- outputs <- ExportedModel[these]
    # toJSON(outputs, pretty = TRUE)
  } else {
    select_model <- "inputs"
  }

  filename <- sprintf("%s/RobynModel-%s.json", dir, select_model)
  write_json(ret, filename, pretty = TRUE)
  message(sprintf("Exported model %s as %s", select_model, filename))
  class(ret) <- c("robyn_json", class(ret))
  return(invisible(ret))
}

# robyn_json(InputCollect, OutputCollect, select_model, dir = "~/Desktop")
# this <- read_json("~/Desktop/RobynModel-1_26_16.json", simplifyVector = TRUE)

# temp <- robyn_run(dt_input = dt_input, dt_holidays = dt_holidays,
#                   json_file = "~/Desktop/RobynModel-1_26_16.json")
