# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Response and Saturation Curves
#'
#' \code{robyn_response()} returns the response for a given
#' spend level of a given \code{paid_media_vars} from a selected model
#' result and selected model build (initial model, refresh model, etc.).
#'
#' @inheritParams robyn_allocator
#' @param metric_name A character. Selected media variable for the response.
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
#'   metric_name = "search_S",
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
#'   metric_name = "search_S",
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
#'   metric_name = "facebook_I",
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
#'   metric_name = "search_S",
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
                           select_model = NULL,
                           metric_name = NULL,
                           metric_value = NULL,
                           date_range = NULL,
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
    dayInterval <- InputCollect$dayInterval
  }

  if (!isTRUE(select_model %in% allSolutions) || is.null(select_model)) {
    stop(paste0(
      "Input 'select_model' must be one of these values: ",
      paste(allSolutions, collapse = ", ")
    ))
  }

  ## Get use case based on inputs
  usecase <- which_usecase(metric_value, date_range)

  ## Check inputs with usecases
  metric_type <- check_metric_type(metric_name, paid_media_spends, paid_media_vars, exposure_vars, organic_vars)
  all_dates <- pull(dt_input, InputCollect$date_var)
  all_values <- pull(dt_input, metric_name)

  if (usecase == "all_historical_vec") {
    ds_list <- check_metric_dates(date_range = "all", all_dates, dayInterval, quiet, ...)
    #val_list <- check_metric_value(metric_value, metric_name, all_values, ds_list$metric_loc)
  } else if (usecase == "unit_metric_default_last_n") {
    ds_list <- check_metric_dates(date_range = paste0("last_", length(metric_value)), all_dates, dayInterval, quiet, ...)
    #val_list <- check_metric_value(metric_value, metric_name, all_values, ds_list$metric_loc)
  } else {
    ds_list <- check_metric_dates(date_range, all_dates, dayInterval, quiet, ...)
  }
  val_list <- check_metric_value(metric_value, metric_name, all_values, ds_list$metric_loc)
  date_range_updated <- ds_list$date_range_updated
  metric_value_updated <- val_list$metric_value_updated
  all_values_updated <- val_list$all_values_updated

  ## Transform exposure to spend when necessary
  if (metric_type == "exposure") {
    get_spend_name <- paid_media_spends[which(paid_media_vars == metric_name)]
    # expo_vec <- dt_input[, metric_name][[1]]
    # Use non-0 mean as marginal level if metric_value not provided
    # if (is.null(metric_value)) {
    #   metric_value <- mean(expo_vec[startRW:endRW][expo_vec[startRW:endRW] > 0])
    #   if (!quiet) message("Input 'metric_value' not provided. Using mean of ", metric_name, " instead")
    # }
    # Fit spend to exposure
    # spend_vec <- dt_input[, get_spend_name][[1]]
    if (is.null(spendExpoMod)) {
      stop("Can't calculate exposure to spend response. Please, recreate your InputCollect object")
    }
    temp <- filter(spendExpoMod, .data$channel == metric_name)
    nls_select <- temp$rsq_nls > temp$rsq_lm
    if (nls_select) {
      Vmax <- spendExpoMod$Vmax[spendExpoMod$channel == metric_name]
      Km <- spendExpoMod$Km[spendExpoMod$channel == metric_name]
      input_immediate <- mic_men(x = metric_value_updated, Vmax = Vmax, Km = Km, reverse = TRUE)
    } else {
      coef_lm <- spendExpoMod$coef_lm[spendExpoMod$channel == metric_name]
      input_immediate <- metric_value_updated / coef_lm
    }
    all_values_updated[ds_list$metric_loc] <- input_immediate
    hpm_name <- get_spend_name
  } else {
    # use non-0 means marginal level if spend not provided
    # if (is.null(metric_value)) {
    #   metric_value <- mean(media_vec[startRW:endRW][media_vec[startRW:endRW] > 0])
    #   if (!quiet) message("Input 'metric_value' not provided. Using mean of ", metric_name, " instead")
    # }
    input_immediate <- metric_value_updated
    hpm_name <- metric_name
  }

  ## Adstocking original
  media_vec_origin <- dt_input[, metric_name][[1]]
  theta <- scale <- shape <- NULL
  if (adstock == "geometric") {
    theta <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_thetas")]][[1]]
  }
  if (grepl("weibull", adstock)) {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_shapes")]][[1]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_scales")]][[1]]
  }
  x_list <- transform_adstock(media_vec_origin, adstock, theta = theta, shape = shape, scale = scale)
  m_adstocked <- x_list$x_decayed
  # net_carryover_ref <- m_adstocked - media_vec_origin

  ## Adstocking simulation
  x_list_sim <- transform_adstock(all_values_updated, adstock, theta = theta, shape = shape, scale = scale)
  media_vec_sim <- x_list_sim$x_decayed
  media_vec_sim_imme <- if (adstock == "weibull_pdf") x_list_sim$x_imme else x_list_sim$x
  input_total <- media_vec_sim[ds_list$metric_loc]
  input_immediate <- media_vec_sim_imme[ds_list$metric_loc]
  input_carryover <- input_total - input_immediate

  ## Saturation
  m_adstockedRW <- m_adstocked[startRW:endRW]
  alpha <- head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_alphas")]], 1)
  gamma <- head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(hpm_name, "_gammas")]], 1)
  metric_saturated_total <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = input_total)
  metric_saturated_carryover <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma, x_marginal = input_carryover)
  metric_saturated_immediate <- metric_saturated_total - metric_saturated_carryover

  ## Decomp
  coeff <- dt_coef[dt_coef$solID == select_model & dt_coef$rn == hpm_name, ][["coef"]]
  m_saturated <- saturation_hill(x = m_adstockedRW, alpha = alpha, gamma = gamma)
  m_resposne <- m_saturated * coeff
  response_total <- as.numeric(metric_saturated_total * coeff)
  response_carryover <- as.numeric(metric_saturated_carryover * coeff)
  response_immediate <- response_total - response_carryover

  dt_line <- data.frame(metric = m_adstockedRW, response = m_resposne, channel = metric_name)
  dt_point <- data.frame(input = input_total, output = response_total, ds = date_range_updated)

  # Reference non-adstocked data when using updated metric values
  dt_point_caov <- data.frame(input = input_carryover, output = response_carryover)
  dt_point_imme <- data.frame(input = input_immediate, output = response_immediate)

  ## Plot optimal response
  p_res <- ggplot(dt_line, aes(x = .data$metric, y = .data$response)) +
    geom_line(color = "steelblue") +
    geom_point(data = dt_point, aes(x = .data$input, y = .data$output), size = 3) +
    labs(
      title = paste(
        "Saturation curve of",
        ifelse(metric_type == "organic", "organic", "paid"),
        "media:", metric_name,
        ifelse(!is.null(date_range_updated), "adstocked", ""),
        ifelse(metric_type == "spend", "spend metric", "exposure metric")
      ),
      subtitle = ifelse(length(unique(input_total)) == 1, sprintf(
        paste(
          "Carryover* Response: %s @ Input %s",
          "Immediate Response: %s @ Input %s",
          "Total (C+I) Response: %s @ Input %s",
          sep = "\n"
        ),
        num_abbr(dt_point_caov$output), num_abbr(dt_point_caov$input),
        num_abbr(dt_point_imme$output), num_abbr(dt_point_imme$input),
        num_abbr(dt_point$output), num_abbr(dt_point$input)
      ), ""),
      x = "Input", y = "Response",
      caption = sprintf(
        "Response period: %s%s%s",
        head(date_range_updated, 1),
        ifelse(length(date_range_updated) > 1, paste(" to", tail(date_range_updated, 1)), ""),
        ifelse(length(date_range_updated) > 1, paste0(" [", length(date_range_updated), " periods]"), "")
      )
    ) +
    theme_lares() +
    scale_x_abbr() +
    scale_y_abbr()
  if (length(unique(metric_value)) == 1) {
    p_res <- p_res +
      geom_point(data = dt_point_caov, aes(x = .data$input, y = .data$output), size = 3, shape = 8)
  }

  ret <- list(
    metric_name = metric_name,
    date = date_range_updated,
    input_total = input_total,
    input_carryover = input_carryover,
    input_immediate = input_immediate,
    response_total = response_total,
    response_carryover = response_carryover,
    response_immediate = response_immediate,
    usecase = usecase,
    plot = p_res
  )
  class(ret) <- unique(c("robyn_response", class(ret)))
  return(ret)
}

which_usecase <- function(metric_value, date_range) {
  case_when(
    # Case 1: raw historical spend and all dates -> model decomp as out of the model (no mean spends)
    is.null(metric_value) & is.null(date_range) ~ "all_historical_vec",
    # Case 2: same as case 1 for date_range
    is.null(metric_value) & !is.null(date_range) ~ "selected_historical_vec",
    ######### Simulations: use metric_value, not the historical real spend anymore
    # Cases 3-4: metric_value for "total budget" for date_range period
    length(metric_value) == 1 & is.null(date_range) ~ "total_metric_default_range",
    length(metric_value) == 1 & !is.null(date_range) ~ "total_metric_selected_range",
    # Cases 5-6: individual period values, not total; requires date_range to be the same length as metric_value
    length(metric_value) > 1 & is.null(date_range) ~ "unit_metric_default_last_n",
    TRUE ~ "unit_metric_selected_dates"
  )
}

# ####### SCENARIOS CHECK FOR date_range
# metric_value <- 71427
# all_dates <- dt_input$DATE
# check_metric_dates(metric_value, date_range = NULL, all_dates, quiet = FALSE)
# check_metric_dates(metric_value, date_range = "last", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, date_range = "last_5", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, date_range = "all", all_dates, quiet = FALSE)
# check_metric_dates(metric_value, date_range = c("2018-01-01"), all_dates, quiet = FALSE)
# check_metric_dates(metric_value, date_range = c("2018-01-01", "2018-07-11"), all_dates, quiet = FALSE) # WARNING
# check_metric_dates(metric_value, date_range = c("2018-01-01", "2018-07-09"), all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), date_range = "last_4", all_dates, quiet = FALSE) # ERROR
# check_metric_dates(c(50000, 60000), date_range = "last_2", all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), date_range = c("2018-12-31", "2019-01-07"), all_dates, quiet = FALSE)
# check_metric_dates(c(50000, 60000), date_range = c("2018-12-31"), all_dates, quiet = FALSE) # ERROR
# check_metric_dates(0, date_range = c("2018-12-31"), all_dates, quiet = FALSE)
