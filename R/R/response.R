# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Response Function
#'
#' \code{robyn_response()} returns the response for a given
#' spend level of a given \code{paid_media_vars} from a selected model
#' result and selected model build (initial model, refresh model, etc.).
#'
#' @inheritParams robyn_allocator
#' @param media_metric A character. Selected media variable for the response.
#' Must be one value from paid_media_spends, paid_media_vars or organic_vars
#' @param metric_value Numeric. Desired metric value to return a response for.
#' @param dt_hyppar A data.frame. When \code{robyn_object} is not provided, use
#' \code{dt_hyppar = OutputCollect$resultHypParam}. It must be provided along
#' \code{select_model}, \code{dt_coef} and \code{InputCollect}.
#' @param dt_coef A data.frame. When \code{robyn_object} is not provided, use
#' \code{dt_coef = OutputCollect$xDecompAgg}. It must be provided along
#' \code{select_model}, \code{dt_hyppar} and \code{InputCollect}.
#' @examples
#' \dontrun{
#' # Having InputCollect and OutputCollect objects
#'
#' # Get marginal response (mResponse) and marginal ROI (mROI) for
#' # the next 1k on 80k for search_S
#' spend1 <- 80000
#' Response1 <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   media_metric = "search_S",
#'   metric_value = spend1
#' )$response
#' # Get ROI for 80k
#' Response1 / spend1 # ROI for search 80k
#'
#' # Get response for 81k
#' spend2 <- spend1 + 1000
#' Response2 <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   media_metric = "search_S",
#'   metric_value = spend2
#' )$response
#'
#' # Get ROI for 81k
#' Response2 / spend2 # ROI for search 81k
#' # Get marginal response (mResponse) for the next 1k on 80k
#' Response2 - Response1
#' # Get marginal ROI (mROI) for the next 1k on 80k
#' (Response2 - Response1) / (spend2 - spend1)
#'
#' # Example of getting paid media exposure response curves
#' imps <- 1000000
#' response_imps <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   media_metric = "facebook_I",
#'   metric_value = imps
#' )$response
#' response_per_1k_imps <- response_imps / imps * 1000
#' response_per_1k_imps
#'
#' # Get response for 80k for search_S from the a certain model SolID
#' # in the current model output in the global environment
#' robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   media_metric = "search_S",
#'   metric_value = 80000,
#'   dt_hyppar = OutputCollect$resultHypParam,
#'   dt_coef = OutputCollect$xDecompAgg
#' )
#' }
#' @return List. Response value and plot. Class: \code{robyn_response}.
#' @export
robyn_response <- function(InputCollect = NULL,
                           OutputCollect = NULL,
                           json_file = NULL,
                           robyn_object = NULL,
                           select_build = NULL,
                           media_metric = NULL,
                           select_model = NULL,
                           metric_value = NULL,
                           dt_hyppar = NULL,
                           dt_coef = NULL,
                           quiet = FALSE,
                           ...) {
  ## Get input

  ### Use previously exported model using json_file
  if (!is.null(json_file)) {
    if (is.null(InputCollect)) InputCollect <- robyn_inputs(json_file = json_file, ...)
    if (is.null(OutputCollect)) {
      OutputCollect <- robyn_run(
        InputCollect = InputCollect,
        json_file = json_file,
        export = FALSE,
        quiet = quiet,
        ...
      )
    }
    if (is.null(dt_hyppar)) dt_hyppar <- OutputCollect$resultHypParam
    if (is.null(dt_coef)) dt_coef <- OutputCollect$xDecompAgg
  } else {
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
        if (!quiet && length(select_build_all) > 1) {
          message(
            "Using latest model: ", ifelse(select_build == 0, "initial model", paste0("refresh model #", select_build)),
            " for the response function. Use parameter 'select_build' to specify which run to use"
          )
        }
      }
      if (!(select_build %in% select_build_all) || length(select_build) != 1) {
        stop("'select_build' must be one value of ", paste(select_build_all, collapse = ", "))
      }
      listName <- ifelse(select_build == 0, "listInit", paste0("listRefresh", select_build))
      InputCollect <- Robyn[[listName]][["InputCollect"]]
      OutputCollect <- Robyn[[listName]][["OutputCollect"]]
      dt_hyppar <- OutputCollect$resultHypParam
      dt_coef <- OutputCollect$xDecompAgg
    } else {
      # Try to get some pre-filled values
      if (is.null(dt_hyppar)) dt_hyppar <- OutputCollect$resultHypParam
      if (is.null(dt_coef)) dt_coef <- OutputCollect$xDecompAgg
      if (any(is.null(dt_hyppar), is.null(dt_coef), is.null(InputCollect), is.null(OutputCollect))) {
        stop("When 'robyn_object' is not provided, 'InputCollect' & 'OutputCollect' must be provided")
      }
    }
  }

  if ("selectID" %in% names(OutputCollect)) {
    select_model <- OutputCollect$selectID
  }

  ## Prep environment
  if (TRUE) {
    dt_input <- InputCollect$dt_input
    startRW <- InputCollect$rollingWindowStartWhich
    endRW <- InputCollect$rollingWindowEndWhich
    adstock <- InputCollect$adstock
    spendExpoMod <- InputCollect$modNLS$results
    paid_media_vars <- InputCollect$paid_media_vars
    paid_media_spends <- InputCollect$paid_media_spends
    exposure_vars <- InputCollect$exposure_vars
    organic_vars <- InputCollect$organic_vars
    allSolutions <- unique(dt_hyppar$solID)
  }

  if (!(select_model %in% allSolutions)) {
    stop(paste0(
      "Input 'select_model' must be one of these values: ",
      paste(allSolutions, collapse = ", ")
    ))
  }

  ## Get media values
  if (media_metric %in% paid_media_spends && length(media_metric) == 1) {
    metric_type <- "spend"
  } else if (media_metric %in% exposure_vars && length(media_metric) == 1) {
    metric_type <- "exposure"
  } else if (media_metric %in% organic_vars && length(media_metric) == 1) {
    metric_type <- "organic"
  } else {
    stop(paste(
      "Invalid 'media_metric' input. It must be any media variable from",
      "paid_media_spends (spend), paid_media_vars (exposure),",
      "or organic_vars (organic); NOT:", media_metric,
      paste("\n- paid_media_spends:", v2t(paid_media_spends, quotes = FALSE)),
      paste("\n- paid_media_vars:", v2t(paid_media_vars, quotes = FALSE)),
      paste("\n- organic_vars:", v2t(organic_vars, quotes = FALSE))
    ))
  }

  if (any(is.nan(metric_value))) metric_value <- NULL
  check_metric_value(metric_value, media_metric)

  ## Transform exposure to spend when necessary
  if (metric_type == "exposure") {
    get_spend_name <- paid_media_spends[which(paid_media_vars == media_metric)]
    expo_vec <- dt_input[, media_metric][[1]]
    # Use non-0 mean as marginal level if metric_value not provided
    if (is.null(metric_value)) {
      metric_value <- mean(expo_vec[startRW:endRW][expo_vec[startRW:endRW] > 0])
      if (!quiet) message("Input 'metric_value' not provided. Using mean of ", media_metric, " instead")
    }
    # Fit spend to exposure
    spend_vec <- dt_input[, get_spend_name][[1]]
    if (is.null(spendExpoMod)) {
      stop("Can't calculate exposure to spend response. Please, recreate your InputCollect object")
    }
    temp <- filter(spendExpoMod, .data$channel == media_metric)
    nls_select <- temp$rsq_nls > temp$rsq_lm
    if (nls_select) {
      Vmax <- spendExpoMod$Vmax[spendExpoMod$channel == media_metric]
      Km <- spendExpoMod$Km[spendExpoMod$channel == media_metric]
      media_vec <- mic_men(x = spend_vec, Vmax = Vmax, Km = Km, reverse = FALSE)
    } else {
      coef_lm <- spendExpoMod$coef_lm[spendExpoMod$channel == media_metric]
      media_vec <- spend_vec * coef_lm
    }
    hpm_name <- get_spend_name
  } else {
    media_vec <- dt_input[, media_metric][[1]]
    # use non-0 means marginal level if spend not provided
    if (is.null(metric_value)) {
      metric_value <- mean(media_vec[startRW:endRW][media_vec[startRW:endRW] > 0])
      if (!quiet) message("Input 'metric_value' not provided. Using mean of ", media_metric, " instead")
    }
    hpm_name <- media_metric
  }

  ## Adstocking
  if (adstock == "geometric") {
    theta <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_thetas")]]
    x_list <- adstock_geometric(x = media_vec, theta = theta)
  } else if (adstock == "weibull_cdf") {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]]
    x_list <- adstock_weibull(x = media_vec, shape = shape, scale = scale, type = "cdf")
  } else if (adstock == "weibull_pdf") {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]]
    x_list <- adstock_weibull(x = media_vec, shape = shape, scale = scale, type = "pdf")
  }
  m_adstocked <- x_list$x_decayed

  ## Saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_alphas")]]
  gamma <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_gammas")]]
  Saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = metric_value)
  m_saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma)

  ## Decomp
  coeff <- dt_coef[dt_coef$solID == select_model & dt_coef$rn == hpm_name, ][["coef"]]
  response_vec <- m_saturated * coeff
  Response <- as.numeric(Saturated * coeff)

  ## Plot optimal response
  media_type <- ifelse(metric_type == "organic", "organic", "paid")
  dt_line <- data.frame(metric = m_adstockedRW, response = response_vec, channel = media_metric)
  dt_point <- data.frame(input = metric_value, output = Response)
  p_res <- ggplot(dt_line, aes(x = .data$metric, y = .data$response)) +
    geom_line(color = "steelblue") +
    geom_point(data = dt_point, aes(x = .data$input, y = .data$output), size = 3) +
    labs(
      title = paste(
        "Saturation curve of", media_type, "media:", media_metric,
        ifelse(metric_type == "spend", "spend metric", "exposure metric")
      ),
      subtitle = sprintf(
        "Response of %s @ %s",
        formatNum(dt_point$output, signif = 4),
        formatNum(dt_point$input, signif = 4)
      ),
      x = "Metric", y = "Response"
    ) +
    theme_lares() +
    scale_x_abbr() +
    scale_y_abbr()

  class(Response) <- unique(c("robyn_response", class(Response)))
  return(list(
    response = Response,
    plot = p_res
  ))
}
