# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Budget Allocator
#'
#' \code{robyn_allocator()} function returns a new split of media
#' variable spends that maximizes the total media response.
#'
#' @inheritParams robyn_run
#' @inheritParams robyn_outputs
#' @param robyn_object Character or List. Path of the \code{Robyn.RDS} object
#' that contains all previous modeling information or the imported list.
#' @param select_build Integer. Default to the latest model build. \code{select_build = 0}
#' selects the initial model. \code{select_build = 1} selects the first refresh model.
#' @param InputCollect List. Contains all input parameters for the model.
#' Required when \code{robyn_object} is not provided.
#' @param OutputCollect List. Containing all model result.
#' Required when \code{robyn_object} is not provided.
#' @param select_model Character. A model \code{SolID}. When \code{robyn_object}
#' is provided, \code{select_model} defaults to the already selected \code{SolID}. When
#' \code{robyn_object} is not provided, \code{select_model} must be provided with
#' \code{InputCollect} and \code{OutputCollect}, and must be one of
#' \code{OutputCollect$allSolutions}.
#' @param optim_algo Character. Default to \code{"SLSQP_AUGLAG"}, short for "Sequential Least-Squares
#' Quadratic Programming" and "Augmented Lagrangian". Alternatively, "\code{"MMA_AUGLAG"},
#' short for "Methods of Moving Asymptotes". More details see the documentation of
#' NLopt \href{https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/}{here}.
#' @param scenario Character. Accepted options are: \code{"max_historical_response"}.
#' Scenario \code{"max_historical_response"} simulates the scenario
#' "What's the revenue/conversions lift potential with the same spend level in \code{date_range}
#' and what is the spend and expected response mix?".
#' Deprecated scenario: \code{"max_response_expected_spend"}.
#' @param channel_constr_low,channel_constr_up Numeric vectors. The lower and upper bounds
#' for each paid media variable when maximizing total media response. For example,
#' \code{channel_constr_low = 0.7} means minimum spend of the variable is 70% of historical
#' average, using non-zero spend values, within \code{date_min} and \code{date_max} date range.
#' Both constrains must be length 1 (same for all values) OR same length and order as
#' \code{paid_media_spends}. It's not recommended to 'exaggerate' upper bounds, especially
#' if the new level is way higher than historical level. Lower bound must be >=0.01,
#' and upper bound should be < 5.
#' @param channel_constr_multiplier Numeric. Default to 3. For example, if
#' \code{channel_constr_low} and \code{channel_constr_up} are 0.8 to 1.2, the range is 0.4.
#' The allocator will also show the optimum solution for a larger constraint range of
#' 0.4 x 3 = 1.2, or 0.4 to 1.6, to show the optimization potential to support allocation
#' interpretation and decision.
#' @param date_range Character. Date(s) to apply adstocked transformations and pick mean spends
#' per channel. Set one of: NULL, "all", "last", or "last_n" (where
#' n is the last N dates available), date (i.e. "2022-03-27"), or date range
#' (i.e. \code{c("2022-01-01", "2022-12-31")}).
#' @param total_budget Numeric. Total marketing budget for all paid channels for the
#' period in \code{date_range}.
#' @param maxeval Integer. The maximum iteration of the global optimization algorithm.
#' Defaults to 100000.
#' @param constr_mode Character. Options are \code{"eq"} or \code{"ineq"},
#' indicating constraints with equality or inequality.
#' @return A list object containing allocator result.
#' @examples
#' \dontrun{
#' # Having InputCollect and OutputCollect results
#' AllocatorCollect <- robyn_allocator(
#'   InputCollect = InputCollect,
#'   OutputCollect = OutputCollect,
#'   select_model = "1_2_3",
#'   scenario = "max_historical_response",
#'   channel_constr_low = 0.7,
#'   channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5),
#'   channel_constr_multiplier = 4,
#'   date_range = "last_26",
#'   export = FALSE
#' )
#' # Print a summary
#' print(AllocatorCollect)
#' # Plot the allocator one-pager
#' plot(AllocatorCollect)
#' }
#' @return List. Contains optimized allocation results and plots.
#' @export
robyn_allocator <- function(robyn_object = NULL,
                            select_build = 0,
                            InputCollect = NULL,
                            OutputCollect = NULL,
                            select_model = NULL,
                            json_file = NULL,
                            scenario = "max_historical_response",
                            channel_constr_low = 0.5,
                            channel_constr_up = 2,
                            channel_constr_multiplier = 3,
                            date_range = NULL,
                            total_budget = NULL,
                            optim_algo = "SLSQP_AUGLAG",
                            maxeval = 100000,
                            constr_mode = "eq",
                            export = TRUE,
                            quiet = FALSE,
                            ui = FALSE,
                            ...) {
  #####################################
  #### Set local environment

  ### Use previously exported model using json_file
  if (!is.null(json_file)) {
    if (is.null(InputCollect)) InputCollect <- robyn_inputs(json_file = json_file, ...)
    if (is.null(OutputCollect)) {
      OutputCollect <- robyn_run(
        json_file = json_file, plot_folder = robyn_object, ...
      )
    }
    if (is.null(select_model)) select_model <- OutputCollect$selectID
  }

  ## Collect inputs
  if (!is.null(robyn_object) && (is.null(InputCollect) && is.null(OutputCollect))) {
    if ("robyn_exported" %in% class(robyn_object)) {
      imported <- robyn_object
      robyn_object <- imported$robyn_object
    } else {
      imported <- robyn_load(robyn_object, select_build, quiet)
    }
    InputCollect <- imported$InputCollect
    OutputCollect <- imported$OutputCollect
    select_model <- imported$select_model
  } else if (any(is.null(InputCollect), is.null(OutputCollect), is.null(select_model))) {
    stop("When 'robyn_object' is not provided, then InputCollect, OutputCollect, select_model must be provided")
  }

  message(paste(">>> Running budget allocator for model ID", select_model, "..."))

  ## Set local data & params values
  paid_media_spends <- InputCollect$paid_media_spends
  media_order <- order(paid_media_spends)
  mediaSpendSorted <- paid_media_spends[media_order]
  if (length(channel_constr_low) == 1) channel_constr_low <- rep(channel_constr_low, length(paid_media_spends))
  if (length(channel_constr_up) == 1) channel_constr_up <- rep(channel_constr_up, length(paid_media_spends))
  names(channel_constr_low) <- paid_media_spends
  names(channel_constr_up) <- paid_media_spends
  channel_constr_low <- channel_constr_low[media_order]
  channel_constr_up <- channel_constr_up[media_order]
  dt_hyppar <- filter(OutputCollect$resultHypParam, .data$solID == select_model)
  dt_bestCoef <- filter(OutputCollect$xDecompAgg, .data$solID == select_model, .data$rn %in% paid_media_spends)

  ## Check inputs and parameters
  check_allocator(
    OutputCollect, select_model, paid_media_spends, scenario,
    channel_constr_low, channel_constr_up, constr_mode
  )

  ## Sort media
  dt_coef <- select(dt_bestCoef, .data$rn, .data$coef)
  get_rn_order <- order(dt_bestCoef$rn)
  dt_coefSorted <- dt_coef[get_rn_order, ]
  dt_bestCoef <- dt_bestCoef[get_rn_order, ]
  coefSelectorSorted <- dt_coefSorted$coef > 0
  names(coefSelectorSorted) <- dt_coefSorted$rn

  dt_hyppar <- select(dt_hyppar, hyper_names(InputCollect$adstock, mediaSpendSorted)) %>%
    select(sort(colnames(.)))
  dt_bestCoef <- dt_bestCoef[dt_bestCoef$rn %in% mediaSpendSorted, ]
  channelConstrLowSorted <- channel_constr_low[mediaSpendSorted]
  channelConstrUpSorted <- channel_constr_up[mediaSpendSorted]

  ## Get hill parameters for each channel
  hills <- get_hill_params(
    InputCollect, OutputCollect, dt_hyppar, dt_coef, mediaSpendSorted, select_model
  )
  alphas <- hills$alphas
  inflexions <- hills$inflexions
  coefs_sorted <- hills$coefs_sorted

  # Spend values based on date range set
  dt_optimCost <- slice(InputCollect$dt_mod, InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich)
  new_date_range <- check_metric_dates(date_range, dt_optimCost$ds, InputCollect$dayInterval, quiet = FALSE, is_allocator = TRUE)
  date_min <- head(new_date_range$date_range_updated, 1)
  date_max <- tail(new_date_range$date_range_updated, 1)
  check_daterange(date_min, date_max, dt_optimCost$ds)
  if (is.null(date_min)) date_min <- min(dt_optimCost$ds)
  if (is.null(date_max)) date_max <- max(dt_optimCost$ds)
  if (date_min < min(dt_optimCost$ds)) date_min <- min(dt_optimCost$ds)
  if (date_max > max(dt_optimCost$ds)) date_max <- max(dt_optimCost$ds)
  histFiltered <- filter(dt_optimCost, .data$ds >= date_min & .data$ds <= date_max)

  histSpendAll <- unlist(summarise_all(select(dt_optimCost, any_of(mediaSpendSorted)), sum))
  histSpendAllTotal <- sum(histSpendAll)
  histSpendAllUnit <- unlist(summarise_all(select(dt_optimCost, any_of(mediaSpendSorted)), mean))
  histSpendAllUnitTotal <- sum(histSpendAllUnit)
  histSpendAllShare <- histSpendAllUnit / histSpendAllUnitTotal

  histSpendWindow <- unlist(summarise_all(select(histFiltered, any_of(mediaSpendSorted)), sum))
  histSpendWindowTotal <- sum(histSpendWindow)
  initSpendUnit <- histSpendWindowUnit <- unlist(summarise_all(select(histFiltered, any_of(mediaSpendSorted)), mean))
  histSpendWindowUnitTotal <- sum(histSpendWindowUnit)
  histSpendWindowShare <- histSpendWindowUnit / histSpendWindowUnitTotal

  simulation_period <- initial_mean_period <- unlist(summarise_all(select(histFiltered, any_of(mediaSpendSorted)), length))
  nDates <- lapply(mediaSpendSorted, function(x) histFiltered$ds)
  names(nDates) <- mediaSpendSorted
  message(sprintf("Date Window: %s:%s (%s %ss)", date_min, date_max, unique(initial_mean_period), InputCollect$intervalType))
  zero_spend_channel <- names(histSpendWindow[histSpendWindow == 0])

  initSpendUnitTotal <- sum(initSpendUnit)
  initSpendShare <- initSpendUnit / initSpendUnitTotal
  total_budget_unit <- ifelse(is.null(total_budget), initSpendUnitTotal, total_budget / unique(simulation_period))
  # total_budget_window <- total_budget_unit * unique(simulation_period)

  ## Get use case based on inputs
  usecase <- which_usecase(initSpendUnit[1], date_range)
  usecase <- paste(usecase, ifelse(!is.null(total_budget), "+ defined_budget", "+ historical_budget"))

  # Response values based on date range -> mean spend
  initResponseUnit <- NULL
  initResponseMargUnit <- NULL
  hist_carryover <- list()
  for (i in seq_along(mediaSpendSorted)) {
    resp <- robyn_response(
      json_file = json_file,
      robyn_object = robyn_object,
      select_build = select_build,
      select_model = select_model,
      metric_name = mediaSpendSorted[i],
      metric_value = initSpendUnit[i],
      date_range = date_range,
      dt_hyppar = OutputCollect$resultHypParam,
      dt_coef = OutputCollect$xDecompAgg,
      InputCollect = InputCollect,
      OutputCollect = OutputCollect,
      quiet = TRUE,
      is_allocator = TRUE,
      ...
    )
    # val <- sort(resp$response_total)[round(length(resp$response_total) / 2)]
    # histSpendUnit[i] <- resp$input_immediate[which(resp$response_total == val)]
    hist_carryover[[i]] <- resp$input_carryover
    # get simulated response
    resp_simulate <- fx_objective(
      x = initSpendUnit[i],
      coeff = coefs_sorted[[mediaSpendSorted[i]]],
      alpha = alphas[[paste0(mediaSpendSorted[i], "_alphas")]],
      inflexion = inflexions[[paste0(mediaSpendSorted[i], "_gammas")]],
      x_hist_carryover = mean(resp$input_carryover),
      get_sum = FALSE
    )
    resp_simulate_plus1 <- fx_objective(
      x = initSpendUnit[i] + 1,
      coeff = coefs_sorted[[mediaSpendSorted[i]]],
      alpha = alphas[[paste0(mediaSpendSorted[i], "_alphas")]],
      inflexion = inflexions[[paste0(mediaSpendSorted[i], "_gammas")]],
      x_hist_carryover = mean(resp$input_carryover),
      get_sum = FALSE
    )
    names(hist_carryover[[i]]) <- resp$date
    initResponseUnit <- c(initResponseUnit, resp_simulate)
    initResponseMargUnit <- c(initResponseMargUnit, resp_simulate_plus1 - resp_simulate)
  }
  names(initResponseUnit) <- names(hist_carryover) <- mediaSpendSorted
  if (length(zero_spend_channel) == 0 && !quiet) {
    message("Media variables with 0 spending during date range: ", v2t(zero_spend_channel))
    # hist_carryover[zero_spend_channel] <- 0
  }

  ## Set initial values and bounds
  channelConstrLowSortedExt <- ifelse(
    1 - (1 - channelConstrLowSorted) * channel_constr_multiplier < 0,
    0, 1 - (1 - channelConstrLowSorted) * channel_constr_multiplier
  )
  channelConstrUpSortedExt <- 1 + (channelConstrUpSorted - 1) * channel_constr_multiplier
  temp_init_all <- initSpendUnit
  # if no spend within window as initial spend, use historical average
  if (length(zero_spend_channel) > 0) temp_init_all[zero_spend_channel] <- histSpendAllUnit[zero_spend_channel]
  # Exclude channels with 0 coef from optimisation
  temp_ub_all <- channelConstrUpSorted
  temp_lb_all <- channelConstrLowSorted
  temp_ub_ext_all <- channelConstrUpSortedExt
  temp_lb_ext_all <- channelConstrLowSortedExt

  x0 <- x0_all <- lb <- lb_all <- temp_init_all * temp_lb_all
  ub <- ub_all <- temp_init_all * temp_ub_all
  x0_ext <- x0_ext_all <- lb_ext <- lb_ext_all <- temp_init_all * temp_lb_ext_all
  ub_ext <- ub_ext_all <- temp_init_all * temp_ub_ext_all

  ## Exclude 0 coef and 0 constraint channels for the optimisation
  check_allocator_constrains(channel_constr_low, channel_constr_up)
  skip_these <- (channel_constr_low == 0 & channel_constr_up == 0)
  zero_constraint_channel <- mediaSpendSorted[skip_these]
  if (any(skip_these) && !quiet) {
    message("Excluded variables (constrained to 0): ", zero_constraint_channel)
  }
  if (!all(coefSelectorSorted)) {
    zero_coef_channel <- setdiff(names(coefSelectorSorted), mediaSpendSorted[coefSelectorSorted])
    message("Excluded variables (coefficients are 0): ", paste(zero_coef_channel, collapse = ", "))
  } else {
    zero_coef_channel <- as.character()
  }
  channel_for_allocation_loc <- mediaSpendSorted %in% c(zero_coef_channel, zero_constraint_channel)
  channel_for_allocation <- mediaSpendSorted[!channel_for_allocation_loc]
  if (length(zero_coef_channel) > 0) {
    temp_init <- temp_init_all[channel_for_allocation]
    temp_ub <- temp_ub_all[channel_for_allocation]
    temp_lb <- temp_lb_all[channel_for_allocation]
    temp_ub_ext <- temp_ub_ext_all[channel_for_allocation]
    temp_lb_ext <- temp_lb_ext_all[channel_for_allocation]
    x0 <- x0_all[channel_for_allocation]
    lb <- lb_all[channel_for_allocation]
    ub <- ub_all[channel_for_allocation]
    x0_ext <- x0_ext_all[channel_for_allocation]
    lb_ext <- lb_ext_all[channel_for_allocation]
    ub_ext <- ub_ext_all[channel_for_allocation]
  }

  x0 <- lb <- temp_init * temp_lb
  ub <- temp_init * temp_ub
  x0_ext <- lb_ext <- temp_init * temp_lb_ext
  ub_ext <- temp_init * temp_ub_ext

  # Gather all values that will be used internally on optim (nloptr)
  coefs_eval <- coefs_sorted[channel_for_allocation]
  alphas_eval <- alphas[paste0(channel_for_allocation, "_alphas")]
  inflexions_eval <- inflexions[paste0(channel_for_allocation, "_gammas")]
  hist_carryover_eval <- hist_carryover[channel_for_allocation]

  eval_list <- list(
    coefs_eval = coefs_eval,
    alphas_eval = alphas_eval,
    inflexions_eval = inflexions_eval,
    # mediaSpendSortedFiltered = mediaSpendSorted,
    total_budget = total_budget,
    total_budget_unit = total_budget_unit,
    hist_carryover_eval = hist_carryover_eval
  )
  # So we can implicitly use these values within eval_f()
  options("ROBYN_TEMP" = eval_list)

  ## Set optim options
  if (optim_algo == "MMA_AUGLAG") {
    local_opts <- list(
      "algorithm" = "NLOPT_LD_MMA",
      "xtol_rel" = 1.0e-10
    )
  } else if (optim_algo == "SLSQP_AUGLAG") {
    local_opts <- list(
      "algorithm" = "NLOPT_LD_SLSQP",
      "xtol_rel" = 1.0e-10
    )
  }

  ## Run optim
  nlsMod <- nloptr::nloptr(
    x0 = x0,
    eval_f = eval_f,
    eval_g_eq = if (constr_mode == "eq") eval_g_eq else NULL,
    eval_g_ineq = if (constr_mode == "ineq") eval_g_ineq else NULL,
    lb = lb, ub = ub,
    opts = list(
      "algorithm" = "NLOPT_LD_AUGLAG",
      "xtol_rel" = 1.0e-10,
      "maxeval" = maxeval,
      "local_opts" = local_opts
    )
  )

  optmSpendUnit <- nlsMod$solution
  optmResponseUnit <- -eval_f(optmSpendUnit)[["objective.channel"]]
  x_hist_carryover <- unlist(lapply(hist_carryover_eval, mean))

  optmResponseMargUnit <- mapply(
    fx_objective,
    x = optmSpendUnit + 1,
    coeff = coefs_eval,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    x_hist_carryover = x_hist_carryover,
    get_sum = FALSE,
    SIMPLIFY = TRUE
  ) - optmResponseUnit

  nlsModUnbound <- nloptr::nloptr(
    x0 = x0_ext,
    eval_f = eval_f,
    eval_g_eq = if (constr_mode == "eq") eval_g_eq else NULL,
    eval_g_ineq = if (constr_mode == "ineq") eval_g_ineq else NULL,
    lb = lb_ext, ub = ub_ext,
    opts = list(
      "algorithm" = "NLOPT_LD_AUGLAG",
      "xtol_rel" = 1.0e-10,
      "maxeval" = maxeval,
      "local_opts" = local_opts
    )
  )

  optmSpendUnitUnbound <- nlsModUnbound$solution
  optmResponseUnitUnbound <- -eval_f(optmSpendUnitUnbound)[["objective.channel"]]
  optmResponseMargUnitUnbound <- mapply(
    fx_objective,
    x = optmSpendUnitUnbound + 1,
    coeff = coefs_eval,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    x_hist_carryover = x_hist_carryover,
    get_sum = FALSE,
    SIMPLIFY = TRUE
  ) - optmResponseUnitUnbound

  ## Collect output
  names(optmSpendUnit) <- names(optmResponseUnit) <- names(optmResponseMargUnit) <-
    names(optmSpendUnitUnbound) <- names(optmResponseUnitUnbound) <-
    names(optmResponseMargUnitUnbound) <- channel_for_allocation
  mediaSpendSorted %in% names(optmSpendUnit)
  optmSpendUnitOut <- optmResponseUnitOut <- optmResponseMargUnitOut <-
    optmSpendUnitUnboundOut <- optmResponseUnitUnboundOut <-
    optmResponseMargUnitUnboundOut <- initSpendUnit
  optmSpendUnitOut[channel_for_allocation_loc] <-
    optmResponseUnitOut[channel_for_allocation_loc] <-
    optmResponseMargUnitOut[channel_for_allocation_loc] <-
    optmSpendUnitUnboundOut[channel_for_allocation_loc] <-
    optmResponseUnitUnboundOut[channel_for_allocation_loc] <-
    optmResponseMargUnitUnboundOut[channel_for_allocation_loc] <- 0
  optmSpendUnitOut[!channel_for_allocation_loc] <- optmSpendUnit
  optmResponseUnitOut[!channel_for_allocation_loc] <- optmResponseUnit
  optmResponseMargUnitOut[!channel_for_allocation_loc] <- optmResponseMargUnit
  optmSpendUnitUnboundOut[!channel_for_allocation_loc] <- optmSpendUnitUnbound
  optmResponseUnitUnboundOut[!channel_for_allocation_loc] <- optmResponseUnitUnbound
  optmResponseMargUnitUnboundOut[!channel_for_allocation_loc] <- optmResponseMargUnitUnbound

  dt_optimOut <- data.frame(
    solID = select_model,
    dep_var_type = InputCollect$dep_var_type,
    channels = mediaSpendSorted,
    date_min = date_min,
    date_max = date_max,
    periods = sprintf("%s %ss", initial_mean_period, InputCollect$intervalType),
    constr_low = temp_lb_all,
    constr_low_abs = lb_all,
    constr_up = temp_ub_all,
    constr_up_abs = ub_all,
    unconstr_mult = channel_constr_multiplier,
    constr_low_unb = temp_lb_ext_all,
    constr_low_unb_abs = lb_ext_all,
    constr_up_unb = temp_ub_ext_all,
    constr_up_unb_abs = ub_ext_all,
    # Historical spends
    histSpendAll = histSpendAll,
    histSpendAllTotal = histSpendAllTotal,
    histSpendAllUnit = histSpendAllUnit,
    histSpendAllUnitTotal = histSpendAllUnitTotal,
    histSpendAllShare = histSpendAllShare,
    histSpendWindow = histSpendWindow,
    histSpendWindowTotal = histSpendWindowTotal,
    histSpendWindowUnit = histSpendWindowUnit,
    histSpendWindowUnitTotal = histSpendWindowUnitTotal,
    histSpendWindowShare = histSpendWindowShare,
    # Initial spends for allocation
    initSpendUnit = initSpendUnit,
    initSpendUnitTotal = initSpendUnitTotal,
    initSpendShare = initSpendShare,
    initSpendTotal = initSpendUnitTotal * unique(simulation_period),
    # initSpendUnitRaw = histSpendUnitRaw,
    # adstocked = adstocked,
    # adstocked_start_date = as.Date(ifelse(adstocked, head(resp$date, 1), NA), origin = "1970-01-01"),
    # adstocked_end_date = as.Date(ifelse(adstocked, tail(resp$date, 1), NA), origin = "1970-01-01"),
    # adstocked_periods = length(resp$date),
    initResponseUnit = initResponseUnit,
    initResponseUnitTotal = sum(initResponseUnit),
    initResponseMargUnit = initResponseMargUnit,
    initResponseTotal = sum(initResponseUnit) * unique(simulation_period),
    initResponseUnitShare = initResponseUnit / sum(initResponseUnit),
    initRoiUnit = initResponseUnit / initSpendUnit,
    # Budget change
    total_budget_unit = total_budget_unit,
    total_budget_unit_delta = total_budget_unit / initSpendUnitTotal - 1,
    # Optimized
    optmSpendUnit = optmSpendUnitOut,
    optmSpendUnitDelta = (optmSpendUnitOut / initSpendUnit - 1),
    optmSpendUnitTotal = sum(optmSpendUnitOut),
    optmSpendUnitTotalDelta = sum(optmSpendUnitOut) / initSpendUnitTotal - 1,
    optmSpendShareUnit = optmSpendUnitOut / sum(optmSpendUnitOut),
    optmSpendTotal = sum(optmSpendUnitOut) * unique(simulation_period),
    optmSpendUnitUnbound = optmSpendUnitUnboundOut,
    optmSpendUnitDeltaUnbound = (optmSpendUnitUnboundOut / initSpendUnit - 1),
    optmSpendUnitTotalUnbound = sum(optmSpendUnitUnboundOut),
    optmSpendUnitTotalDeltaUnbound = sum(optmSpendUnitUnboundOut) / initSpendUnitTotal - 1,
    optmSpendShareUnitUnbound = optmSpendUnitUnboundOut / sum(optmSpendUnitUnboundOut),
    optmSpendTotalUnbound = sum(optmSpendUnitUnboundOut) * unique(simulation_period),
    optmResponseUnit = optmResponseUnitOut,
    optmResponseMargUnit = optmResponseMargUnitOut,
    optmResponseUnitTotal = sum(optmResponseUnitOut),
    optmResponseTotal = sum(optmResponseUnitOut) * unique(simulation_period),
    optmResponseUnitShare = optmResponseUnitOut / sum(optmResponseUnitOut),
    optmRoiUnit = optmResponseUnitOut / optmSpendUnitOut,
    optmResponseUnitLift = (optmResponseUnitOut / initResponseUnit) - 1,
    optmResponseUnitUnbound = optmResponseUnitUnboundOut,
    optmResponseMargUnitUnbound = optmResponseMargUnitUnboundOut,
    optmResponseUnitTotalUnbound = sum(optmResponseUnitUnboundOut),
    optmResponseTotalUnbound = sum(optmResponseUnitUnboundOut) * unique(simulation_period),
    optmResponseUnitShareUnbound = optmResponseUnitUnboundOut / sum(optmResponseUnitUnboundOut),
    optmRoiUnitUnbound = optmResponseUnitUnboundOut / optmSpendUnitUnboundOut,
    optmResponseUnitLiftUnbound = (optmResponseUnitUnboundOut / initResponseUnit) - 1
  ) %>%
    mutate(
      optmResponseUnitTotalLift = (.data$optmResponseUnitTotal / .data$initResponseUnitTotal) - 1,
      optmResponseUnitTotalLiftUnbound = (.data$optmResponseUnitTotalUnbound / .data$initResponseUnitTotal) - 1
    )
  .Options$ROBYN_TEMP <- NULL # Clean auxiliary method

  ## Calculate curves and main points for each channel
  levs1 <- c("Initial", "Bounded", paste0("Bounded x", channel_constr_multiplier))
  dt_optimOutScurve <- rbind(
    select(dt_optimOut, .data$channels, .data$initSpendUnit, .data$initResponseUnit) %>%
      mutate(x = levs1[1]) %>% as.matrix(),
    select(dt_optimOut, .data$channels, .data$optmSpendUnit, .data$optmResponseUnit) %>%
      mutate(x = levs1[2]) %>% as.matrix(),
    select(dt_optimOut, .data$channels, .data$optmSpendUnitUnbound, .data$optmResponseUnitUnbound) %>%
      mutate(x = levs1[3]) %>% as.matrix()
  ) %>%
    `colnames<-`(c("channels", "spend", "response", "type")) %>%
    rbind(data.frame(channels = dt_optimOut$channels, spend = 0, response = 0, type = "Carryover")) %>%
    mutate(spend = as.numeric(.data$spend), response = as.numeric(.data$response)) %>%
    group_by(.data$channels)
  plotDT_adstocked <- OutputCollect$mediaVecCollect %>%
    filter(.data$solID == select_model, .data$type == "adstockedMedia") %>%
    select(.data$ds, all_of(InputCollect$paid_media_spends)) %>%
    tidyr::gather("channel", "spend", -.data$ds)

  plotDT_scurve <- list()
  for (i in channel_for_allocation) { # i <- channels[i]
    carryover_vec <- eval_list$hist_carryover_eval[[i]]
    dt_optimOutScurve <- dt_optimOutScurve %>%
      mutate(spend = ifelse(
        .data$channels == i & .data$type %in% levs1,
        .data$spend + mean(carryover_vec), ifelse(
          .data$channels == i & .data$type == "Carryover",
          mean(carryover_vec), .data$spend
        )
      ))
    get_max_x <- max(filter(dt_optimOutScurve, .data$channels == i)$spend) * 1.5
    simulate_spend <- seq(0, get_max_x, length.out = 100)
    simulate_response <- fx_objective(
      x = simulate_spend,
      coeff = eval_list$coefs_eval[[i]],
      alpha = eval_list$alphas_eval[[paste0(i, "_alphas")]],
      inflexion = eval_list$inflexions_eval[[paste0(i, "_gammas")]],
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    simulate_response_carryover <- fx_objective(
      x = mean(carryover_vec),
      coeff = eval_list$coefs_eval[[i]],
      alpha = eval_list$alphas_eval[[paste0(i, "_alphas")]],
      inflexion = eval_list$inflexions_eval[[paste0(i, "_gammas")]],
      x_hist_carryover = 0,
      get_sum = FALSE
    )
    plotDT_scurve[[i]] <- data.frame(
      channel = i, spend = simulate_spend,
      mean_carryover = mean(carryover_vec),
      carryover_response = simulate_response_carryover,
      total_response = simulate_response
    )
    dt_optimOutScurve <- dt_optimOutScurve %>%
      mutate(response = ifelse(
        .data$channels == i & .data$type == "Carryover",
        simulate_response_carryover, .data$response
      ))
  }
  eval_list[["plotDT_scurve"]] <- plotDT_scurve <- as_tibble(bind_rows(plotDT_scurve))
  mainPoints <- dt_optimOutScurve %>%
    rename("response_point" = "response", "spend_point" = "spend", "channel" = "channels")
  temp_caov <- mainPoints %>% filter(.data$type == "Carryover")
  mainPoints$mean_spend <- mainPoints$spend_point - temp_caov$spend_point
  mainPoints$mean_spend <- ifelse(mainPoints$type == "Carryover", mainPoints$spend_point, mainPoints$mean_spend)
  mainPoints$type <- factor(mainPoints$type, levels = c("Carryover", levs1))
  mainPoints$roi_mean <- mainPoints$response_point / mainPoints$mean_spend
  mresp_caov <- filter(mainPoints, .data$type == "Carryover")$response_point
  mresp_init <- filter(mainPoints, .data$type == levels(mainPoints$type)[2])$response_point - mresp_caov
  mresp_b <- filter(mainPoints, .data$type == levels(mainPoints$type)[3])$response_point - mresp_caov
  mresp_unb <- filter(mainPoints, .data$type == levels(mainPoints$type)[4])$response_point - mresp_caov
  mainPoints$marginal_response <- c(mresp_init, mresp_b, mresp_unb, rep(0, length(mresp_init)))
  mainPoints$roi_marginal <- mainPoints$marginal_response / mainPoints$mean_spend
  mainPoints$cpa_marginal <- mainPoints$mean_spend / mainPoints$marginal_response
  eval_list[["mainPoints"]] <- mainPoints

  ## Plot allocator results
  plots <- allocation_plots(
    InputCollect, OutputCollect,
    filter(dt_optimOut, .data$channels %in% channel_for_allocation),
    select_model, scenario, eval_list, export, quiet
  )

  ## Export results into CSV
  if (export) {
    export_dt_optimOut <- dt_optimOut
    if (InputCollect$dep_var_type == "conversion") {
      colnames(export_dt_optimOut) <- gsub("Roi", "CPA", colnames(export_dt_optimOut))
    }
    write.csv(export_dt_optimOut, paste0(OutputCollect$plot_folder, select_model, "_reallocated.csv"))
  }

  output <- list(
    dt_optimOut = dt_optimOut,
    mainPoints = mainPoints,
    nlsMod = nlsMod,
    plots = plots,
    scenario = scenario,
    usecase = usecase,
    total_budget = total_budget,
    skipped = c(zero_coef_channel, zero_constraint_channel),
    # skipped_budget = sum(skipped_budget),
    no_spend = zero_spend_channel,
    ui = if (ui) plots else NULL
  )

  class(output) <- c("robyn_allocator", class(output))
  return(output)
}

#' @rdname robyn_allocator
#' @aliases robyn_allocator
#' @param x \code{robyn_allocator()} output.
#' @export
print.robyn_allocator <- function(x, ...) {
  temp <- x$dt_optimOut[!is.nan(x$dt_optimOut$optmRoiUnit), ]
  print(glued(
    "
Model ID: {x$dt_optimOut$solID[1]}
Scenario: {x$scenario}
Use case: {x$usecase}
Window: {x$dt_optimOut$date_min[1]}:{x$dt_optimOut$date_max[1]} ({x$dt_optimOut$periods[1]})

Dep. Variable Type: {temp$dep_var_type[1]}
Media Skipped (coef = 0 | constrained @ 0): {v2t(x$skipped, quotes = FALSE)} {no_spend}
Relative Spend Increase: {spend_increase_p}% ({spend_increase})
Total Response Increase (Optimized): {signif(100 * x$dt_optimOut$optmResponseUnitTotalLift[1], 3)}%

Allocation Summary:
  {summary}
",
    no_spend = ifelse(length(x$no_spend) > 0, paste("| (spend = 0):", v2t(x$no_spend, quotes = FALSE)), ""),
    spend_increase_p = num_abbr(100 * x$dt_optimOut$optmSpendUnitTotalDelta[1], 3),
    spend_increase = formatNum(
      sum(x$dt_optimOut$optmSpendUnitTotal) - sum(x$dt_optimOut$initSpendUnitTotal),
      abbr = TRUE, sign = TRUE
    ),
    summary = paste(sprintf(
      "
- %s:
  Optimizable bound: [%s%%, %s%%],
  Initial spend share: %s%% -> Optimized bounded: %s%%
  Initial response share: %s%% -> Optimized bounded: %s%%
  Initial abs. mean spend: %s -> Optimized: %s [Delta = %s%%]",
      temp$channels,
      100 * temp$constr_low - 100,
      100 * temp$constr_up - 100,
      signif(100 * temp$initSpendShare, 3),
      signif(100 * temp$optmSpendShareUnit, 3),
      signif(100 * temp$initResponseUnitShare, 3),
      signif(100 * temp$optmResponseUnitShare, 3),
      formatNum(temp$initSpendUnit, 3, abbr = TRUE),
      formatNum(temp$optmSpendUnit, 3, abbr = TRUE),
      formatNum(100 * temp$optmSpendUnitDelta, signif = 2)
    ), collapse = "\n  ")
  ))
}

#' @rdname robyn_allocator
#' @aliases robyn_allocator
#' @param x \code{robyn_allocator()} output.
#' @export
plot.robyn_allocator <- function(x, ...) plot(x$plots$plots, ...)

eval_f <- function(X) {
  # eval_list <- get("eval_list", pos = as.environment(-1))
  eval_list <- getOption("ROBYN_TEMP")
  coefs_eval <- eval_list[["coefs_eval"]]
  alphas_eval <- eval_list[["alphas_eval"]]
  inflexions_eval <- eval_list[["inflexions_eval"]]
  # mediaSpendSortedFiltered <- eval_list[["mediaSpendSortedFiltered"]]
  hist_carryover_eval <- eval_list[["hist_carryover_eval"]]

  objective <- -sum(mapply(
    fx_objective,
    x = X,
    coeff = coefs_eval,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    x_hist_carryover = hist_carryover_eval,
    SIMPLIFY = TRUE
  ))

  gradient <- c(mapply(
    fx_gradient,
    x = X,
    coeff = coefs_eval,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    x_hist_carryover = hist_carryover_eval,
    SIMPLIFY = TRUE
  ))

  objective.channel <- mapply(
    fx_objective.chanel,
    x = X,
    coeff = coefs_eval,
    alpha = alphas_eval,
    inflexion = inflexions_eval,
    x_hist_carryover = hist_carryover_eval,
    SIMPLIFY = TRUE
  )

  optm <- list(objective = objective, gradient = gradient, objective.channel = objective.channel)
  return(optm)
}

fx_objective <- function(x, coeff, alpha, inflexion, x_hist_carryover, get_sum = TRUE) {
  # Apply Michaelis Menten model to scale spend to exposure
  # if (criteria) {
  #   xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
  # } else if (chnName %in% names(mm_lm_coefs)) {
  #   xScaled <- x * mm_lm_coefs[chnName]
  # } else {
  #   xScaled <- x
  # }

  # Adstock scales
  xAdstocked <- x + mean(x_hist_carryover)
  # Hill transformation
  if (get_sum == TRUE) {
    xOut <- coeff * sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
  } else {
    xOut <- coeff * ((1 + inflexion**alpha / xAdstocked**alpha)**-1)
  }
  return(xOut)
}

# https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)
fx_gradient <- function(x, coeff, alpha, inflexion, x_hist_carryover
                        # , chnName, vmax, km, criteria
) {
  # Apply Michaelis Menten model to scale spend to exposure
  # if (criteria) {
  #   xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
  # } else if (chnName %in% names(mm_lm_coefs)) {
  #   xScaled <- x * mm_lm_coefs[chnName]
  # } else {
  #   xScaled <- x
  # }

  # Adstock scales
  xAdstocked <- x + mean(x_hist_carryover)
  xOut <- -coeff * sum((alpha * (inflexion**alpha) * (xAdstocked**(alpha - 1))) / (xAdstocked**alpha + inflexion**alpha)**2)
  return(xOut)
}

fx_objective.chanel <- function(x, coeff, alpha, inflexion, x_hist_carryover
                                # , chnName, vmax, km, criteria
) {
  # Apply Michaelis Menten model to scale spend to exposure
  # if (criteria) {
  #   xScaled <- mic_men(x = x, Vmax = vmax, Km = km) # vmax * x / (km + x)
  # } else if (chnName %in% names(mm_lm_coefs)) {
  #   xScaled <- x * mm_lm_coefs[chnName]
  # } else {
  #   xScaled <- x
  # }

  # Adstock scales
  xAdstocked <- x + mean(x_hist_carryover)
  xOut <- -coeff * sum((1 + inflexion**alpha / xAdstocked**alpha)**-1)
  return(xOut)
}

eval_g_eq <- function(X) {
  eval_list <- getOption("ROBYN_TEMP")
  constr <- sum(X) - eval_list$total_budget_unit
  grad <- rep(1, length(X))
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}

eval_g_ineq <- function(X) {
  eval_list <- getOption("ROBYN_TEMP")
  constr <- sum(X) - eval_list$total_budget_unit
  grad <- rep(1, length(X))
  return(list(
    "constraints" = constr,
    "jacobian" = grad
  ))
}

get_adstock_params <- function(InputCollect, dt_hyppar) {
  if (InputCollect$adstock == "geometric") {
    getAdstockHypPar <- unlist(select(dt_hyppar, na.omit(str_extract(names(dt_hyppar), ".*_thetas"))))
  } else if (InputCollect$adstock %in% c("weibull_cdf", "weibull_pdf")) {
    getAdstockHypPar <- unlist(select(dt_hyppar, na.omit(str_extract(names(dt_hyppar), ".*_shapes|.*_scales"))))
  }
  return(getAdstockHypPar)
}

get_hill_params <- function(InputCollect, OutputCollect = NULL, dt_hyppar, dt_coef, mediaSpendSorted, select_model, chnAdstocked = NULL) {
  hillHypParVec <- unlist(select(dt_hyppar, na.omit(str_extract(names(dt_hyppar), ".*_alphas|.*_gammas"))))
  alphas <- hillHypParVec[str_which(names(hillHypParVec), "_alphas")]
  gammas <- hillHypParVec[str_which(names(hillHypParVec), "_gammas")]
  if (is.null(chnAdstocked)) {
    chnAdstocked <- filter(
      OutputCollect$mediaVecCollect,
      .data$type == "adstockedMedia",
      .data$solID == select_model
    ) %>%
      select(all_of(mediaSpendSorted)) %>%
      slice(InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich)
  }
  inflexions <- unlist(lapply(seq(ncol(chnAdstocked)), function(i) {
    c(range(chnAdstocked[, i]) %*% c(1 - gammas[i], gammas[i]))
  }))
  names(inflexions) <- names(gammas)
  coefs <- dt_coef$coef
  names(coefs) <- dt_coef$rn
  coefs_sorted <- coefs[mediaSpendSorted]
  return(list(
    alphas = alphas,
    inflexions = inflexions,
    coefs_sorted = coefs_sorted
  ))
}
