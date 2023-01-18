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
#' @param metric_ds Character. Date(s) to apply adstocked transformations.
#' One of: NULL, "all", "last", or "last_n" (where
#' n is the last N dates available), date (i.e. "2022-03-27"), or date range
#' (i.e. \code{c("2022-01-01", "2022-12-31")}).
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
                           metric_ds = NULL,
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

  ## Check metric values
  if (any(is.nan(metric_value))) metric_value <- NULL
  check_metric_value(metric_value, media_metric)
  metric_value_updated <- metric_value

  ## Get media type based on provided values
  metric_type <- check_metric_type(media_metric, paid_media_spends, paid_media_vars, exposure_vars, organic_vars)

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

  ## Adstocking original
  theta <- scale <- shape <- NULL
  if (adstock == "geometric") {
    theta <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_thetas")]]
  }
  if (grepl("weibull", adstock)) {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]]
  }
  x_list <- transform_adstock(media_vec, adstock, theta = theta, shape = shape, scale = scale)
  m_adstocked <- x_list$x_decayed
  inflation <- 1

  ## Adstocking simulation
  if (!is.null(metric_ds)) {
    all_dates <- pull(dt_input, InputCollect$date_var)
    ds_list <- check_metric_dates(metric_value, metric_ds, all_dates, quiet)
    metric_ds <- ds_list$metric_ds
    metric_value <- ds_list$metric_value
    new_media_vec <- media_vec[ds_list$metric_which]
    x_list_sim <- transform_adstock(new_media_vec, adstock, theta = theta, shape = shape, scale = scale)
    # When no sim, metric_value_updated = metric_value
    metric_value_updated <- x_list_sim$x_decayed
    inflation <- metric_value_updated / metric_value # divide or difference?
  }

  ## Saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_alphas")]]
  gamma <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_gammas")]]
  Saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = metric_value_updated)
  m_saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma)
  ## Decomp
  coeff <- dt_coef[dt_coef$solID == select_model & dt_coef$rn == hpm_name, ][["coef"]]
  response_vec <- m_saturated * coeff
  Response <- as.numeric(Saturated * coeff)
  dt_line <- data.frame(metric = m_adstockedRW, response = response_vec, channel = media_metric)
  dt_point <- data.frame(input = metric_value_updated, output = Response)

  # Reference non-adstocked data when using updated metric values
  if (!is.null(metric_ds)) {
    SaturatedR <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = metric_value)
    ResponseR <- as.numeric(SaturatedR * coeff)
    dt_pointR <- data.frame(input = metric_value, output = ResponseR)
  } else {
    ResponseR <- Response
    SaturatedR <- Saturated
  }

  ## Plot optimal response
  p_res <- ggplot(dt_line, aes(x = .data$metric, y = .data$response)) +
    geom_line(color = "steelblue") +
    geom_point(data = dt_point, aes(x = .data$input, y = .data$output), size = 3) +
    labs(
      title = paste(
        "Saturation curve of",
        ifelse(metric_type == "organic", "organic", "paid"),
        "media:", media_metric,
        ifelse(!is.null(metric_ds), "adstocked", ""),
        ifelse(metric_type == "spend", "spend metric", "exposure metric")
      ),
      subtitle = ifelse(length(unique(metric_value)) == 1, sprintf(
        "Response of %s @ %s%s",
        formatNum(dt_point$output, signif = 4),
        formatNum(dt_point$input, signif = 4),
        ifelse(!is.null(metric_ds) && length(unique(metric_value)) == 1, sprintf(
          " [adstocked from %s]", formatNum(dt_pointR$input, signif = 4)
        ), "")
      ), ""),
      x = "Metric", y = "Response",
      caption = ifelse(
        !is.null(metric_ds),
        sprintf(
          "Using adstocked metric results from %s%s%s",
          head(metric_ds, 1),
          ifelse(length(metric_ds) > 1, paste(" to", tail(metric_ds, 1)), ""),
          ifelse(length(metric_ds) > 1, paste0(" [", length(metric_ds), " periods]"), "")
        ),
        ""
      )
    ) +
    theme_lares() +
    scale_x_abbr() +
    scale_y_abbr()
  if (!is.null(metric_ds) && length(unique(metric_value)) == 1) {
    p_res <- p_res +
      geom_point(data = dt_pointR, aes(x = .data$input, y = .data$output), size = 3, shape = 8)
  }
  ret <- list(
    response = Response,
    metric = metric_value_updated,
    response_ref = ResponseR,
    metric_ref = metric_value,
    inflation = inflation,
    date = metric_ds,
    plot = p_res
  )
  class(ret) <- unique(c("robyn_response", class(ret)))
  return(ret)
}

# ####### SCENARIOS CHECK FOR metric_ds
# metric_value <- 71427
# all_dates <- dt_input$DATE
# check_metric_dates(metric_value, metric_ds = NULL, all_dates, quiet = FALSE)
# check_metric_dates(metric_value, metric_ds = "last", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, metric_ds = "last_5", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, metric_ds = "all", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, metric_ds = c("2018-01-01"), all_dates, quiet = FALSE)
# check_metric_dates(metric_value, metric_ds = c("2018-01-01", "2018-07-11"), all_dates, quiet = FALSE) # WARNING
# check_metric_dates(metric_value, metric_ds = c("2018-01-01", "2018-07-09"), all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), metric_ds = "last_4", all_dates, quiet = FALSE) # ERROR
# check_metric_dates(c(50000, 60000), metric_ds = "last_2", all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), metric_ds = c("2018-12-31", "2019-01-07"), all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), metric_ds = c("2018-12-31"), all_dates, quiet = FALSE) # ERROR

check_metric_dates <- function(metric_value, metric_ds = NULL, all_dates = metric_ds, quiet = FALSE) {
  metric_value_updated <- metric_value
  if (is.null(metric_ds)) {
    metric_ds_val <- NULL
    metric_ds_loc <- 1
  } else {
    if (grepl("last|all", metric_ds[1])) {
      ## Using last_n as metric_ds range
      if ("all" %in% metric_ds) metric_ds <- paste0("last_", length(all_dates))
      last_n <- ifelse(grepl("_", metric_ds[1]), as.integer(gsub("last_", "", metric_ds)), 1)
      metric_ds <- tail(all_dates, last_n)
      metric_ds_loc <- which(all_dates %in% metric_ds)
      metric_ds_val <- all_dates[metric_ds_loc]
      rg <- v2t(range(metric_ds_val), sep = ":", quotes = FALSE)
      if (length(metric_value_updated) == 1) metric_value_updated <- rep(metric_value_updated, last_n)
      if (length(metric_value_updated) != last_n) {
        stop("Input 'metric_value' must have length of 1 or same length as 'last_n'")
      }
    } else {
      ## Using dates as metric_ds range
      if (all(is.Date(as.Date(metric_ds, origin = "1970-01-01")))) {
        metric_ds <- as.Date(metric_ds, origin = "1970-01-01")
        if (length(metric_ds) == 1) {
          ## Using only 1 date
          if (all(metric_ds %in% all_dates)) {
            metric_ds_val <- metric_ds
            metric_ds_loc <- which(metric_ds %in% all_dates)
            if (!quiet) message("Using ds '", metric_ds_val, "' as the response period")
          } else {
            metric_ds_loc <- which.min(abs(metric_ds - all_dates))
            metric_ds_val <- all_dates[metric_ds_loc]
            if (!quiet) warning("Input 'metric_ds' (", metric_ds, ") has no match. Picking closest date: ", metric_ds_val)
          }
        } else if (length(metric_ds) == 2) {
          ## Using two dates as "from-to" date range
          metric_ds_loc <- unlist(lapply(metric_ds, function(x) which.min(abs(x - all_dates))))
          metric_ds_loc <- metric_ds_loc[1]:metric_ds_loc[2]
          metric_ds_val <- all_dates[metric_ds_loc]
          metric_ds_n <- length(metric_ds_val)
          if (!quiet & !all(metric_ds %in% metric_ds_val)) {
            warning(paste(
              "At least one date in 'metric_ds' input do not match any date.",
              "Picking closest dates for range:", paste(range(metric_ds_val), collapse = ":")
            ))
          }
          rg <- v2t(range(metric_ds_val), sep = ":", quotes = FALSE)
          if (!quiet) message("Using ds ", rg, " (", metric_ds_n, " period(s) included) as the response period")
          if (length(metric_value_updated) == 1) {
            metric_value_updated <- rep(metric_value_updated, metric_ds_n)
          }
        } else {
          ## Manually inputting each date
          if (all(metric_ds %in% all_dates)) {
            metric_ds_val <- metric_ds
            metric_ds_loc <- which(all_dates %in% metric_ds)
            avlb_dates <- all_dates[(all_dates >= min(metric_ds_val) & all_dates <= max(metric_ds_val))]
            if (length(metric_ds_loc) != length(avlb_dates) & !quiet)
              warning(sprintf("There are %s skipping dates within your 'metric_ds' input",
                              length(avlb_dates) - length(metric_ds_loc)))
          } else {
            metric_ds_loc <- unlist(lapply(metric_ds, function(x) which.min(abs(x - all_dates))))
            if (all(na.omit(metric_ds_loc - lag(metric_ds_loc)) == 1)) {
              metric_ds_val <- all_dates[metric_ds_loc]
              if (!quiet) warning("At least one date in 'metric_ds' do not match ds. Picking closest date: ", metric_ds_val)
            } else {
              stop("Input 'metric_ds' needs to have sequential dates")
            }
            rg <- v2t(range(metric_ds_val), sep = ":", quotes = FALSE)
            if (length(metric_value_updated) == 1) {
              metric_value_updated <- metric_value_updated / length(metric_ds_val)
            }
          }
        }
      } else {
        stop("Input 'metric_ds' must have date format '2023-01-01' or use 'last_n'")
      }
    }
    if (length(metric_value_updated) != length(metric_ds_val)) {
      stop("Input 'metric_value' must be length 1 or same as input 'metric_ds'")
    }
  }
  return(list(
    metric_ds = metric_ds_val,
    metric_which = metric_ds_loc,
    metric_value = metric_value_updated
  ))
}
