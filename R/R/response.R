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
#' @param dt_hyppar A data.frame. When \code{json_file} is not provided, use
#' \code{dt_hyppar = OutputCollect$resultHypParam}. It must be provided along
#' \code{select_model}, \code{dt_coef} and \code{InputCollect}.
#' @param dt_coef A data.frame. When \code{json_file} is not provided, use
#' \code{dt_coef = OutputCollect$xDecompAgg}. It must be provided along
#' \code{select_model}, \code{dt_hyppar} and \code{InputCollect}.
#' @examples
#' \dontrun{
#' # Having InputCollect and OutputCollect objects
#' ## Recreate original saturation curve
#' Response <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "facebook_S"
#' )
#' Response$plot
#'
#' ## Or you can call a JSON file directly (a bit slower)
#' # Response <- robyn_response(
#' #   json_file = "your_json_path.json",
#' #   dt_input = dt_simulated_weekly,
#' #   dt_holidays = dt_prophet_holidays,
#' #   metric_name = "facebook_S"
#' # )
#'
#' ## Get the "next 100 dollar" marginal response on Spend1
#' Spend1 <- 20000
#' Response1 <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "facebook_S",
#'   metric_value = Spend1, # total budget for date_range
#'   date_range = "last_1" # last two periods
#' )
#' Response1$plot
#'
#' Spend2 <- Spend1 + 100
#' Response2 <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "facebook_S",
#'   metric_value = Spend2,
#'   date_range = "last_1"
#' )
#' # ROAS for the 100$ from Spend1 level
#' (Response2$response_total - Response1$response_total) / (Spend2 - Spend1)
#'
#' ## Get response from for a given budget and date_range
#' Spend3 <- 100000
#' Response3 <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "facebook_S",
#'   metric_value = Spend3, # total budget for date_range
#'   date_range = "last_5" # last 5 periods
#' )
#' Response3$plot
#'
#' ## Example of getting paid media exposure response curves
#' imps <- 10000000
#' response_imps <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "facebook_I",
#'   metric_value = imps
#' )
#' response_imps$response_total / imps * 1000
#' response_imps$plot
#'
#' ## Example of getting organic media exposure response curves
#' sendings <- 30000
#' response_sending <- robyn_response(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = select_model,
#'   metric_name = "newsletter",
#'   metric_value = sendings
#' )
#' # response per 1000 sendings
#' response_sending$response_total / sendings * 1000
#' response_sending$plot
#' }
#' @return List. Response value and plot. Class: \code{robyn_response}.
#' @export
robyn_response <- function(InputCollect = NULL,
                           OutputCollect = NULL,
                           json_file = NULL,
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
    # Get pre-filled values
    if (is.null(dt_hyppar)) dt_hyppar <- OutputCollect$resultHypParam
    if (is.null(dt_coef)) dt_coef <- OutputCollect$xDecompAgg
    if (any(is.null(dt_hyppar), is.null(dt_coef), is.null(InputCollect), is.null(OutputCollect))) {
      stop("When 'json_file' is not provided, 'InputCollect' & 'OutputCollect' must be provided")
    }
  }

  if ("selectID" %in% names(OutputCollect)) {
    select_model <- OutputCollect$selectID
  }

  ## Prep environment
  if (TRUE) {
    dt_input <- InputCollect$dt_input
    dt_mod <- InputCollect$dt_mod
    window_start_loc <- InputCollect$rollingWindowStartWhich
    window_end_loc <- InputCollect$rollingWindowEndWhich
    window_loc <- window_start_loc:window_end_loc
    adstock <- InputCollect$adstock
    # spendExpoMod <- InputCollect$ExposureCollect$df_cpe
    paid_media_vars <- InputCollect$paid_media_vars
    paid_media_spends <- InputCollect$paid_media_spends
    paid_media_selected <- InputCollect$paid_media_selected
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

  ## Check inputs
  metric_type <- check_metric_type(metric_name, paid_media_spends, paid_media_vars, paid_media_selected, exposure_vars, organic_vars)
  metric_name_updated <- metric_type$metric_name_updated
  all_dates <- dt_input[[InputCollect$date_var]]
  all_values <- dt_mod[[metric_name_updated]]
  ds_list <- check_metric_dates(date_range = date_range, all_dates[1:window_end_loc], dayInterval, quiet, ...)
  val_list <- check_metric_value(metric_value, metric_name_updated, all_values, ds_list$metric_loc)
  if (!is.null(metric_value) & is.null(date_range)) {
    stop("Must specify date_range when using metric_value")
  }
  date_range_updated <- ds_list$date_range_updated
  all_values_updated <- val_list$all_values_updated

  ## Get hyperparameters & beta coef
  theta <- scale <- shape <- NULL
  if (adstock == "geometric") {
    theta <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(metric_name_updated, "_thetas")]][[1]]
  }
  if (grepl("weibull", adstock)) {
    shape <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(metric_name_updated, "_shapes")]][[1]]
    scale <- dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(metric_name_updated, "_scales")]][[1]]
  }
  alpha <- head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(metric_name_updated, "_alphas")]], 1)
  gamma <- head(dt_hyppar[dt_hyppar$solID == select_model, ][[paste0(metric_name_updated, "_gammas")]], 1)
  coeff <- dt_coef[dt_coef$solID == select_model & dt_coef$rn == metric_name_updated, ][["coef"]]

  ## Historical transformation
  hist_transform <- transform_decomp(
    all_values = all_values,
    adstock, theta, shape, scale, alpha, gamma,
    window_loc, coeff, metric_loc = ds_list$metric_loc
  )
  dt_line <- data.frame(
    metric = hist_transform$input_total[window_loc],
    response = hist_transform$response_total,
    channel = metric_name_updated
  )
  dt_point <- data.frame(
    mean_input_immediate = hist_transform$mean_input_immediate,
    mean_input_carryover = hist_transform$mean_input_carryover,
    mean_input_total = hist_transform$mean_input_immediate + hist_transform$mean_input_carryover,
    mean_response_immediate = hist_transform$mean_response_total - hist_transform$mean_response_carryover,
    mean_response_carryover = hist_transform$mean_response_carryover,
    mean_response_total = hist_transform$mean_response_total
  )
  if (!is.null(date_range)) {
    dt_point_sim <- data.frame(
      input = hist_transform$sim_mean_spend + hist_transform$sim_mean_carryover,
      output = hist_transform$sim_mean_response
    )
  }

  ## Simulated transformation
  if (!is.null(metric_value)) {
    hist_transform_sim <- transform_decomp(
      all_values = all_values_updated,
      adstock, theta, shape, scale, alpha, gamma,
      window_loc, coeff, metric_loc = ds_list$metric_loc,
      calibrate_inflexion = hist_transform$inflexion
    )
    dt_point_sim <- data.frame(
      input = hist_transform_sim$sim_mean_spend + hist_transform_sim$sim_mean_carryover,
      output = hist_transform_sim$sim_mean_response
    )
  }

  ## Plot optimal response
  p_res <- ggplot(dt_line, aes(x = .data$metric, y = .data$response)) +
    geom_line(color = "steelblue") +
    geom_point(
      data = dt_point,
      aes(x = .data$mean_input_total, y = .data$mean_response_total),
      size = 3, color = "grey"
    ) +
    labs(
      title = paste(
        "Saturation curve of", metric_type$metric_type,
        "media:", metric_type$metric_name_updated
      ),
      subtitle = sprintf(
        paste(
          "Response: %s @ mean input %s",
          "Response: %s @ mean input carryover %s",
          "Response: %s @ mean input immediate %s",
          sep = "\n"
        ),
        num_abbr(dt_point$mean_response_total),
        num_abbr(dt_point$mean_input_total),
        num_abbr(dt_point$mean_response_carryover),
        num_abbr(dt_point$mean_input_carryover),
        num_abbr(dt_point$mean_response_immediate),
        num_abbr(dt_point$mean_input_immediate)
      ),
      x = "Input", y = "Response",
      caption = sprintf(
        "Response period: %s%s%s",
        head(date_range_updated, 1),
        ifelse(length(date_range_updated) > 1, paste(" to", tail(date_range_updated, 1)), ""),
        ifelse(length(date_range_updated) > 1, paste0(" [", length(date_range_updated), " periods]"), "")
      )
    ) +
    theme_lares(background = "white") +
    scale_x_abbr() +
    scale_y_abbr()
  if (!is.null(metric_value) | !is.null(date_range)) {
    p_res <- p_res +
      geom_point(data = dt_point_sim, aes(x = .data$input, y = .data$output), size = 3, color = "blue")
  }
  if (!is.null(metric_value)) {
    sim_mean_spend <- hist_transform_sim$sim_mean_spend
    sim_mean_carryover <- hist_transform_sim$sim_mean_carryover
    sim_mean_response <- hist_transform_sim$sim_mean_response
  } else {
    sim_mean_spend <- sim_mean_carryover <- sim_mean_response <- NULL
  }

  ret <- list(
    metric_name = metric_name_updated,
    date = date_range_updated,
    input_total = hist_transform$input_total,
    input_carryover = hist_transform$input_carryover,
    input_immediate = hist_transform$input_immediate,
    response_total = hist_transform$response_total,
    response_carryover = hist_transform$response_carryover,
    response_immediate = hist_transform$response_immediate,
    inflexion = hist_transform$inflexion,
    mean_input_immediate = hist_transform$mean_input_immediate,
    mean_input_carryover = hist_transform$mean_input_carryover,
    mean_response_total = hist_transform$mean_response_total,
    mean_response_carryover = hist_transform$mean_response_carryover,
    mean_response = hist_transform$mean_response,
    sim_mean_spend = sim_mean_spend,
    sim_mean_carryover = sim_mean_carryover,
    sim_mean_response = sim_mean_response,
    usecase = usecase,
    plot = p_res
  )
  class(ret) <- unique(c("robyn_response", class(ret)))
  return(ret)
}

which_usecase <- function(metric_value, date_range) {
  usecase <- case_when(
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
  if (!is.null(date_range)) {
    if (length(date_range) == 1 & as.character(date_range[1]) == "all") {
      usecase <- "all_historical_vec"
    }
  }
  return(usecase)
}

transform_decomp <- function(all_values, adstock, theta, shape, scale, alpha, gamma,
                             window_loc, coeff, metric_loc, calibrate_inflexion = NULL) {
  ## adstock
  x_list <- transform_adstock(x = all_values, adstock, theta, shape, scale)
  input_total <- x_list$x_decayed
  input_immediate <- if (adstock == "weibull_pdf") x_list$x_imme else x_list$x
  input_carryover <- input_total - input_immediate
  input_total_rw <- input_total[window_loc]
  input_carryover_rw <- input_carryover[window_loc]
  ## saturation
  saturated_total <- saturation_hill(
    x = input_total_rw,
    alpha = alpha, gamma = gamma
  )
  saturated_carryover <- saturation_hill(
    x = input_total_rw,
    alpha = alpha, gamma = gamma, x_marginal = input_carryover_rw
  )
  saturated_immediate <- saturated_total$x_saturated - saturated_carryover$x_saturated
  ## simulate mean response of all_values periods
  mean_input_immediate <- mean(input_immediate[window_loc])
  mean_input_carryover <- mean(input_carryover_rw)
  if (length(window_loc) != length(saturated_total$x_saturated)) {
    mean_response <- mean(saturated_total$x_saturated[window_loc] * coeff)
  } else {
    mean_response <- mean(saturated_total$x_saturated * coeff)
  }
  mean_response_total <- fx_objective(
    x = mean_input_immediate,
    coeff = coeff,
    alpha = alpha,
    inflexion = saturated_total$inflexion,
    x_hist_carryover = mean_input_carryover,
    get_sum = FALSE
  )
  mean_response_carryover <- fx_objective(
    x = 0,
    coeff = coeff,
    alpha = alpha,
    inflexion = saturated_total$inflexion,
    x_hist_carryover = mean_input_carryover,
    get_sum = FALSE
  )
  ## simulate mean response of date_range periods
  sim_mean_spend <- mean(input_immediate[metric_loc])
  sim_mean_carryover <- mean(input_carryover[metric_loc])
  if (is.null(calibrate_inflexion)) calibrate_inflexion <- saturated_total$inflexion
  sim_mean_response <- fx_objective(
    x = sim_mean_spend,
    coeff = coeff,
    alpha = alpha,
    # use historical true inflexion when metric_value is provided
    inflexion = calibrate_inflexion,
    x_hist_carryover = sim_mean_carryover,
    get_sum = FALSE
  )

  ret <- list(
    input_total = input_total,
    input_immediate = input_immediate,
    input_carryover = input_carryover,
    saturated_total = saturated_total$x_saturated,
    saturated_carryover = saturated_carryover$x_saturated,
    saturated_immediate = saturated_immediate,
    response_total = saturated_total$x_saturated * coeff,
    response_carryover = saturated_carryover$x_saturated * coeff,
    response_immediate = saturated_immediate * coeff,
    inflexion = saturated_total$inflexion,
    mean_input_immediate = mean_input_immediate,
    mean_input_carryover = mean_input_carryover,
    mean_response_total = mean_response_total,
    mean_response = mean_response,
    mean_response_carryover = mean_response_carryover,
    sim_mean_spend = sim_mean_spend,
    sim_mean_carryover = sim_mean_carryover,
    sim_mean_response = sim_mean_response
  )
  return(ret)
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
