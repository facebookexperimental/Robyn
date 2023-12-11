# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Input Data Check & Transformation
#'
#' \code{robyn_inputs()} is the function to input all model parameters and
#' check input correctness for the initial model build. It includes the
#' engineering process results that conducts trend, season,
#' holiday & weekday decomposition using Facebook's time-series forecasting
#' library \code{prophet} and fit a nonlinear model to spend and exposure
#' metrics in case exposure metrics are used in \code{paid_media_vars}.
#'
#' @section Guide for calibration source:
#'  \enumerate{
#'    \item We strongly recommend to use experimental and causal results
#'    that are considered ground truth to calibrate MMM. Usual experiment
#'    types are people-based (e.g. Facebook conversion lift) and
#'    geo-based (e.g. Facebook GeoLift).
#'    \item Currently, Robyn only accepts point-estimate as calibration
#'    input. For example, if 10k$ spend is tested against a hold-out
#'    for channel A, then input the incremental return as point-estimate
#'    as the example below.
#'    \item The point-estimate has to always match the spend in the variable.
#'    For example, if channel A usually has 100k$ weekly spend and the
#'    experimental HO is 70%, input the point-estimate for the 30k$, not the 70k$.
#' }
#'
#' @param dt_input data.frame. Raw input data. Load simulated
#' dataset using \code{data("dt_simulated_weekly")}
#' @param dt_holidays data.frame. Raw input holiday data. Load standard
#' Prophet holidays using \code{data("dt_prophet_holidays")}
#' @param date_var Character. Name of date variable. Daily, weekly
#' and monthly data supported. Weekly requires week-start of Monday or Sunday.
#' \code{date_var} must have format "2020-01-01" (YYY-MM-DD).
#' Default to automatic date detection.
#' @param dep_var Character. Name of dependent variable. Only one allowed
#' @param dep_var_type Character. Type of dependent variable
#' as "revenue" or "conversion". Will be used to calculate ROI or CPI,
#' respectively. Only one allowed and case sensitive.
#' @param paid_media_spends Character vector. Names of the paid media variables.
#' The values on each of these variables must be numeric. Also,
#' \code{paid_media_spends} must have same order and length as
#' \code{paid_media_vars} respectively.
#' @param paid_media_vars Character vector. Names of the paid media variables'
#' exposure level metrics (impressions, clicks, GRP etc) other than spend.
#' The values on each of these variables must be numeric. These variables are not
#' being used to train the model but to check relationship and recommend to
#' split media channels into sub-channels (e.g. fb_retargeting, fb_prospecting,
#' etc.) to gain more variance. \code{paid_media_vars} must have same
#' order and length as \code{paid_media_spends} respectively and is not required.
#' @param paid_media_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for \code{paid_media_vars}. Must have same
#' order and same length as \code{paid_media_vars}. By default, all values are
#' set to 'positive'.
#' @param context_vars Character vector. Typically competitors,
#' price & promotion, temperature, unemployment rate, etc.
#' @param context_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for context_vars. Must have same
#' order and same length as \code{context_vars}. By default it's
#' set to 'defualt'.
#' @param organic_vars Character vector. Typically newsletter sendings,
#' push-notifications, social media posts etc. Compared to \code{paid_media_vars}
#' \code{organic_vars} are often marketing activities without clear spends.
#' @param organic_signs Character vector. Choose any of
#' "default", "positive", "negative". Control
#' the signs of coefficients for \code{organic_vars} Must have same
#' order and same length as \code{organic_vars}. By default, all values are
#' set to "positive".
#' @param factor_vars Character vector. Specify which of the provided
#' variables in organic_vars or context_vars should be forced as a factor.
#' @param prophet_vars Character vector. Include any of "trend",
#' "season", "weekday", "holiday" or NULL. Highly recommended
#' to use all for daily data and "trend", "season", "holiday" for
#' weekly and above cadence. Set to NULL to skip prophet's functionality.
#' @param prophet_signs Character vector. Choose any of
#' "default", "positive", "negative". Control
#' the signs of coefficients for \code{prophet_vars}. Must have same
#' order and same length as \code{prophet_vars}. By default, all values are
#' set to "default".
#' @param prophet_country Character. Only one country allowed.
#' Includes national holidays for all countries, whose list can
#' be found loading \code{data("dt_prophet_holidays")}.
#' @param adstock Character. Choose any of "geometric", "weibull_cdf",
#' "weibull_pdf". Weibull adstock is a two-parametric function and thus more
#' flexible, but takes longer time than the traditional geometric one-parametric
#' function. CDF, or cumulative density function of the Weibull function allows
#' changing decay rate over time in both C and S shape, while the peak value will
#' always stay at the first period, meaning no lagged effect. PDF, or the
#' probability density function, enables peak value occurring after the first
#' period when shape >=1, allowing lagged effect. Run \code{plot_adstock()} to
#' see the difference visually. Time estimation: with geometric adstock, 2000
#' iterations * 5 trials on 8 cores, it takes less than 30 minutes. Both Weibull
#' options take up to twice as much time.
#' @param hyperparameters List. Contains hyperparameter lower and upper bounds.
#' Names of elements in list must be identical to output of \code{hyper_names()}.
#' To fix hyperparameter values, provide only one value.
#' @param window_start,window_end Character. Set start and end dates of modelling
#' period. Recommended to not start in the first date in dataset to gain adstock
#' effect from previous periods. Also, columns to rows ratio in the input data
#' to be >=10:1, or in other words at least 10 observations to 1 independent variable.
#' This window will determine the date range of the data period within your dataset
#' you will be using to specifically regress the effects of media, organic and
#' context variables on your dependent variable. We recommend using a full
#' \code{dt_input} dataset with a minimum of 1 year of history, as it will be used
#' in full for the model calculation of trend, seasonality and holidays effects.
#' Whereas the window period will determine how much of the full data set will be
#' used for media, organic and context variables.
#' @param calibration_input data.frame. Optional. Provide experimental results to
#' calibrate. Your input should include the following values for each experiment:
#' channel, liftStartDate, liftEndDate, liftAbs, spend, confidence, metric.
#' You can calibrate any spend or organic variable with a well designed experiment.
#' You can also use experimental results from multiple channels; to do so,
#' provide concatenated channel value, i.e. "channel_A+channel_B".
#' Check "Guide for calibration source" section.
#' @param InputCollect Default to NULL. \code{robyn_inputs}'s output when
#' \code{hyperparameters} are not yet set.
#' @param json_file Character. JSON file to import previously exported inputs
#' (needs \code{dt_input} and \code{dt_holidays} parameters too).
#' @param ... Additional parameters passed to \code{prophet} functions.
#' @examples
#' # Using dummy simulated data
#' InputCollect <- robyn_inputs(
#'   dt_input = Robyn::dt_simulated_weekly,
#'   dt_holidays = Robyn::dt_prophet_holidays,
#'   date_var = "DATE",
#'   dep_var = "revenue",
#'   dep_var_type = "revenue",
#'   prophet_vars = c("trend", "season", "holiday"),
#'   prophet_country = "DE",
#'   context_vars = c("competitor_sales_B", "events"),
#'   paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
#'   paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
#'   organic_vars = "newsletter",
#'   factor_vars = "events",
#'   window_start = "2016-11-23",
#'   window_end = "2018-08-22",
#'   adstock = "geometric",
#'   # To be defined separately
#'   hyperparameters = NULL,
#'   calibration_input = NULL
#' )
#' print(InputCollect)
#' @return List. Contains all input parameters and modified results
#' using \code{Robyn:::robyn_engineering()}. This list is ready to be
#' used on other functions like \code{robyn_run()} and \code{print()}.
#' Class: \code{robyn_inputs}.
#' @export
robyn_inputs <- function(dt_input = NULL,
                         dep_var = NULL,
                         dep_var_type = NULL,
                         date_var = "auto",
                         paid_media_spends = NULL,
                         paid_media_vars = NULL,
                         paid_media_signs = NULL,
                         organic_vars = NULL,
                         organic_signs = NULL,
                         context_vars = NULL,
                         context_signs = NULL,
                         factor_vars = NULL,
                         dt_holidays = Robyn::dt_prophet_holidays,
                         prophet_vars = NULL,
                         prophet_signs = NULL,
                         prophet_country = NULL,
                         adstock = NULL,
                         hyperparameters = NULL,
                         window_start = NULL,
                         window_end = NULL,
                         calibration_input = NULL,
                         json_file = NULL,
                         InputCollect = NULL,
                         ...) {
  ### Use case 3: running robyn_inputs() with json_file
  if (!is.null(json_file)) {
    json <- robyn_read(json_file, step = 1, ...)
    if (is.null(dt_input)) stop("Must provide 'dt_input' input; 'dt_holidays' input optional")
    if (!is.null(hyperparameters)) {
      warning("Replaced hyperparameters input with json_file's fixed hyperparameters values")
    }
    for (i in seq_along(json$InputCollect)) {
      assign(names(json$InputCollect)[i], json$InputCollect[[i]])
    }
  }

  ### Use case 1: running robyn_inputs() for the first time
  if (is.null(InputCollect)) {
    dt_input <- as_tibble(dt_input)
    if (!is.null(dt_holidays)) dt_holidays <- as_tibble(dt_holidays)

    ## Check for NA and all negative values
    dt_input <- check_allneg(dt_input)
    check_nas(dt_input)
    check_nas(dt_holidays)

    ## Check vars names (duplicates and valid)
    check_varnames(
      dt_input, dt_holidays,
      dep_var, date_var,
      context_vars, paid_media_spends,
      organic_vars)

    ## Check date input (and set dayInterval and intervalType)
    date_input <- check_datevar(dt_input, date_var)
    dt_input <- date_input$dt_input # sorted date by ascending
    date_var <- date_input$date_var # when date_var = "auto"
    dayInterval <- date_input$dayInterval
    intervalType <- date_input$intervalType

    ## Check dependent var
    check_depvar(dt_input, dep_var, dep_var_type)

    ## Check prophet
    if (is.null(dt_holidays) || is.null(prophet_vars)) {
      dt_holidays <- prophet_vars <- prophet_country <- prophet_signs <- NULL
    }
    prophet_signs <- check_prophet(dt_holidays, prophet_country, prophet_vars, prophet_signs, dayInterval)

    ## Check baseline variables (and maybe transform context_signs)
    context <- check_context(dt_input, context_vars, context_signs)
    context_signs <- context$context_signs

    ## Check paid media variables (set mediaVarCount and maybe transform paid_media_signs)
    if (is.null(paid_media_vars)) paid_media_vars <- paid_media_spends
    paidmedia <- check_paidmedia(dt_input, paid_media_vars, paid_media_signs, paid_media_spends)
    paid_media_signs <- paidmedia$paid_media_signs
    mediaVarCount <- paidmedia$mediaVarCount
    exposure_vars <- paid_media_vars[!(paid_media_vars == paid_media_spends)]

    ## Check organic media variables (and maybe transform organic_signs)
    organic <- check_organicvars(dt_input, organic_vars, organic_signs)
    organic_signs <- organic$organic_signs

    ## Check factor_vars
    factor_vars <- check_factorvars(dt_input, factor_vars, context_vars, organic_vars)

    ## Check all vars
    all_media <- c(paid_media_spends, organic_vars)
    all_ind_vars <- c(tolower(prophet_vars), context_vars, all_media)
    check_allvars(all_ind_vars)

    ## Check data dimension
    check_datadim(dt_input, all_ind_vars, rel = 10)

    ## Check window_start & window_end (and transform parameters/data)
    windows <- check_windows(dt_input, date_var, all_media, window_start, window_end)

    if (TRUE) {
      dt_input <- windows$dt_input
      window_start <- windows$window_start
      rollingWindowStartWhich <- windows$rollingWindowStartWhich
      refreshAddedStart <- windows$refreshAddedStart
      window_end <- windows$window_end
      rollingWindowEndWhich <- windows$rollingWindowEndWhich
      rollingWindowLength <- windows$rollingWindowLength
    }

    ## Check adstock
    adstock <- check_adstock(adstock)

    ## Check calibration and iters/trials
    calibration_input <- check_calibration(
      dt_input, date_var, calibration_input, dayInterval, dep_var,
      window_start, window_end, paid_media_spends, organic_vars
    )

    ## Not used variables
    unused_vars <- colnames(dt_input)[!colnames(dt_input) %in% c(
      dep_var, date_var, context_vars, paid_media_vars, paid_media_spends, organic_vars
    )]

    # Check for no-variance columns on raw data (after removing not-used)
    check_novar(select(dt_input, -all_of(unused_vars)))

    # Calculate total media spend used to model
    paid_media_total <- dt_input[
      rollingWindowEndWhich:rollingWindowLength, ] %>%
      select(paid_media_vars) %>% sum()

    ## Collect input
    InputCollect <- list(
      dt_input = dt_input,
      dt_holidays = dt_holidays,
      dt_mod = NULL,
      dt_modRollWind = NULL,
      xDecompAggPrev = NULL,
      date_var = date_var,
      dayInterval = dayInterval,
      intervalType = intervalType,
      dep_var = dep_var,
      dep_var_type = dep_var_type,
      prophet_vars = tolower(prophet_vars),
      prophet_signs = prophet_signs,
      prophet_country = prophet_country,
      context_vars = context_vars,
      context_signs = context_signs,
      paid_media_vars = paid_media_vars,
      paid_media_signs = paid_media_signs,
      paid_media_spends = paid_media_spends,
      paid_media_total = paid_media_total,
      mediaVarCount = mediaVarCount,
      exposure_vars = exposure_vars,
      organic_vars = organic_vars,
      organic_signs = organic_signs,
      all_media = all_media,
      all_ind_vars = all_ind_vars,
      factor_vars = factor_vars,
      unused_vars = unused_vars,
      window_start = window_start,
      rollingWindowStartWhich = rollingWindowStartWhich,
      window_end = window_end,
      rollingWindowEndWhich = rollingWindowEndWhich,
      rollingWindowLength = rollingWindowLength,
      totalObservations = nrow(dt_input),
      refreshAddedStart = refreshAddedStart,
      adstock = adstock,
      hyperparameters = hyperparameters,
      calibration_input = calibration_input,
      custom_params = list(...)
    )

    if (!is.null(hyperparameters)) {
      ### Conditional output 1.2
      ## Running robyn_inputs() for the 1st time & 'hyperparameters' provided --> run robyn_engineering()

      ## Check hyperparameters
      hyperparameters <- check_hyperparameters(
        hyperparameters, adstock, paid_media_spends, organic_vars, exposure_vars
      )
      InputCollect <- robyn_engineering(InputCollect, ...)
    }
  } else {
    ### Use case 2: adding 'hyperparameters' and/or 'calibration_input' using robyn_inputs()
    # Check for legacy (deprecated) inputs
    check_legacy_input(InputCollect)

    ## Check calibration data
    calibration_input <- check_calibration(
      dt_input = InputCollect$dt_input,
      date_var = InputCollect$date_var,
      calibration_input = calibration_input,
      dayInterval = InputCollect$dayInterval,
      dep_var = InputCollect$dep_var,
      window_start = InputCollect$window_start,
      window_end = InputCollect$window_end,
      paid_media_spends = InputCollect$paid_media_spends,
      organic_vars = InputCollect$organic_vars
    )

    ## Update calibration_input
    if (!is.null(calibration_input)) InputCollect$calibration_input <- calibration_input
    if (!is.null(hyperparameters)) InputCollect$hyperparameters <- hyperparameters
    if (is.null(InputCollect$hyperparameters) && is.null(hyperparameters)) {
      stop("Must provide hyperparameters in robyn_inputs()")
    } else {
      ### Conditional output 2.1
      ## 'hyperparameters' provided --> run robyn_engineering()
      ## Update & check hyperparameters
      if (is.null(InputCollect$hyperparameters)) InputCollect$hyperparameters <- hyperparameters
      InputCollect$hyperparameters <- check_hyperparameters(
        InputCollect$hyperparameters, InputCollect$adstock, InputCollect$all_media
      )
      InputCollect <- robyn_engineering(InputCollect, ...)
    }

    # Check for no-variance columns (after filtering modeling window)
    dt_mod_model_window <- InputCollect$dt_mod %>%
      select(-any_of(InputCollect$unused_vars)) %>%
      filter(
        .data$ds >= InputCollect$window_start,
        .data$ds <= InputCollect$window_end
      )
    check_novar(dt_mod_model_window, InputCollect)
  }

  if (!is.null(json_file)) {
    pending <- which(!names(json$InputCollect) %in% names(InputCollect))
    InputCollect <- append(InputCollect, json$InputCollect[pending])
  }

  # Save R and Robyn's versions
  if (TRUE) {
    ver <- as.character(utils::packageVersion("Robyn"))
    rver <- utils::sessionInfo()$R.version
    origin <- ifelse(is.null(utils::packageDescription("Robyn")$Repository), "dev", "stable")
    InputCollect$version <- sprintf(
      "Robyn (%s) v%s [R-%s.%s]",
      origin, ver, rver$major, rver$minor
    )
  }

  class(InputCollect) <- c("robyn_inputs", class(InputCollect))
  return(InputCollect)
}

#' @param x \code{robyn_inputs()} output.
#' @rdname robyn_inputs
#' @aliases robyn_inputs
#' @export
print.robyn_inputs <- function(x, ...) {
  mod_vars <- paste(setdiff(names(x$dt_mod), c("ds", "dep_var")), collapse = ", ")
  print(glued(
    "
Total Observations: {nrow(x$dt_input)} ({x$intervalType}s)
Input Table Columns ({ncol(x$dt_input)}):
  Date: {x$date_var}
  Dependent: {x$dep_var} [{x$dep_var_type}]
  Paid Media: {paste(x$paid_media_vars, collapse = ', ')}
  Paid Media Spend: {paste(x$paid_media_spends, collapse = ', ')}
  Context: {paste(x$context_vars, collapse = ', ')}
  Organic: {paste(x$organic_vars, collapse = ', ')}
  Prophet (Auto-generated): {prophet}
  Unused variables: {unused}

Date Range: {range}
Model Window: {windows} ({x$rollingWindowEndWhich - x$rollingWindowStartWhich + 1} {x$intervalType}s)
With Calibration: {!is.null(x$calibration_input)}
Custom parameters: {custom_params}

Adstock: {x$adstock}
{hyps}
",
    range = paste(range(x$dt_input[, x$date_var][[1]]), collapse = ":"),
    windows = paste(x$window_start, x$window_end, sep = ":"),
    custom_params = if (length(x$custom_params) > 0) paste("\n", flatten_hyps(x$custom_params)) else "None",
    prophet = if (length(x$prophet_vars) > 0) {
      sprintf("%s on %s", paste(x$prophet_vars, collapse = ", "), x$prophet_country)
    } else {
      "\033[0;31mDeactivated\033[0m"
    },
    unused = if (length(x$unused_vars) > 0) {
      paste(x$unused_vars, collapse = ", ")
    } else {
      "None"
    },
    hyps = if (!is.null(x$hyperparameters)) {
      glued(
        "Hyper-parameters ranges:\n{flatten_hyps(x$hyperparameters)}"
      )
    } else {
      paste("Hyper-parameters:", "\033[0;31mNot set yet\033[0m")
    }
    # lares::formatColoured("Not set yet", "red", cat = FALSE)
  ))
}


####################################################################
#' Get correct hyperparameter names
#'
#' Output all hyperparameter names and help specifying the list of
#' hyperparameters that is inserted into \code{robyn_inputs(hyperparameters = ...)}
#'
#' @section Guide to setup hyperparameters:
#'  \enumerate{
#'    \item Get correct hyperparameter names:
#'    All variables in \code{paid_media_vars} or \code{organic_vars} require hyperprameters
#'    and will be transformed by adstock & saturation. Difference between \code{paid_media_vars}
#'    and \code{organic_vars} is that \code{paid_media_vars} has spend that
#'    needs to be specified in \code{paid_media_spends} specifically. Run \code{hyper_names()}
#'    to get correct hyperparameter names. All names in hyperparameters must
#'    equal names from \code{hyper_names()}, case sensitive.
#'    \item{Get guidance for setting hyperparameter bounds:
#'    For geometric adstock, use theta, alpha & gamma. For both weibull adstock options,
#'    use shape, scale, alpha, gamma.}
#'    \itemize{
#'    \item{Theta: }{In geometric adstock, theta is decay rate. guideline for usual media genre:
#'    TV c(0.3, 0.8), OOH/Print/Radio c(0.1, 0.4), digital c(0, 0.3)}
#'    \item{Shape: }{In weibull adstock, shape controls the decay shape. Recommended c(0.0001, 2).
#'    The larger, the more S-shape. The smaller, the more L-shape. Channel-type specific
#'    values still to be investigated}
#'    \item{Scale: }{In weibull adstock, scale controls the decay inflexion point. Very conservative
#'    recommended bounce c(0, 0.1), because scale can increase adstocking half-life greatly.
#'    Channel-type specific values still to be investigated}
#'    \item{Gamma: }{In s-curve transformation with hill function, gamma controls the inflexion point.
#'    Recommended bounce c(0.3, 1). The larger the gamma, the later the inflection point
#'    in the response curve}
#'    }
#'    \item{Set each hyperparameter bounds. They either contains two values e.g. c(0, 0.5),
#'    or only one value (in which case you've "fixed" that hyperparameter)}
#' }
#'
#' @section Helper plots:
#' \describe{
#'   \item{plot_adstock}{Get adstock transformation example plot,
#' helping you understand geometric/theta and weibull/shape/scale transformation}
#'   \item{plot_saturation}{Get saturation curve transformation example plot,
#' helping you understand hill/alpha/gamma transformation}
#' }
#'
#' @param adstock Character. Default to \code{InputCollect$adstock}.
#' Accepts "geometric", "weibull_cdf" or "weibull_pdf"
#' @param all_media Character vector. Default to \code{InputCollect$all_media}.
#' Includes \code{InputCollect$paid_media_spends} and \code{InputCollect$organic_vars}.
#' @examples
#' \donttest{
#' media <- c("facebook_S", "print_S", "tv_S")
#' hyper_names(adstock = "geometric", all_media = media)
#'
#' hyperparameters <- list(
#'   facebook_S_alphas = c(0.5, 3), # example bounds for alpha
#'   facebook_S_gammas = c(0.3, 1), # example bounds for gamma
#'   facebook_S_thetas = c(0, 0.3), # example bounds for theta
#'   print_S_alphas = c(0.5, 3),
#'   print_S_gammas = c(0.3, 1),
#'   print_S_thetas = c(0.1, 0.4),
#'   tv_S_alphas = c(0.5, 3),
#'   tv_S_gammas = c(0.3, 1),
#'   tv_S_thetas = c(0.3, 0.8)
#' )
#'
#' # Define hyper_names for weibull adstock
#' hyper_names(adstock = "weibull", all_media = media)
#'
#' hyperparameters <- list(
#'   facebook_S_alphas = c(0.5, 3), # example bounds for alpha
#'   facebook_S_gammas = c(0.3, 1), # example bounds for gamma
#'   facebook_S_shapes = c(0.0001, 2), # example bounds for shape
#'   facebook_S_scales = c(0, 0.1), # example bounds for scale
#'   print_S_alphas = c(0.5, 3),
#'   print_S_gammas = c(0.3, 1),
#'   print_S_shapes = c(0.0001, 2),
#'   print_S_scales = c(0, 0.1),
#'   tv_S_alphas = c(0.5, 3),
#'   tv_S_gammas = c(0.3, 1),
#'   tv_S_shapes = c(0.0001, 2),
#'   tv_S_scales = c(0, 0.1)
#' )
#' }
#' @return Character vector. Names of hyper-parameters that should be defined.
#' @export
hyper_names <- function(adstock, all_media) {
  adstock <- check_adstock(adstock)
  if (adstock == "geometric") {
    local_name <- sort(apply(expand.grid(all_media, HYPS_NAMES[
      grepl("thetas|alphas|gammas", HYPS_NAMES)
    ]), 1, paste, collapse = "_"))
  } else if (adstock %in% c("weibull_cdf", "weibull_pdf")) {
    local_name <- sort(apply(expand.grid(all_media, HYPS_NAMES[
      grepl("shapes|scales|alphas|gammas", HYPS_NAMES)
    ]), 1, paste, collapse = "_"))
  }
  return(local_name)
}


####################################################################
#' Check hyperparameter limits
#'
#' Reference data.frame that shows the upper and lower bounds valid
#' for each hyperparameter.
#'
#' @examples
#' hyper_limits()
#' @return Dataframe. Contains upper and lower bounds for each hyperparameter.
#' @export
hyper_limits <- function() {
  data.frame(
    thetas = c(">=0", "<1"),
    alphas = c(">0", "<10"),
    gammas = c(">0", "<=1"),
    shapes = c(">=0", "<20"),
    scales = c(">=0", "<=1")
  )
}

####################################################################
# Apply prophet decomposition and spend exposure transformation
#
# \code{robyn_engineering()} is included in the \code{robyn_inputs()}
# function and will only run after all the condition checks are passed.
# It applies the decomposition of trend, season, holiday and weekday
# from \code{prophet} and builds the nonlinear fitting model when
# using non-spend variables in \code{paid_media_vars}, for example
# impressions for Facebook variables.
#
# @rdname robyn_inputs
robyn_engineering <- function(x, quiet = FALSE, ...) {
  if (!quiet) message(">> Running feature engineering...")
  InputCollect <- x
  check_InputCollect(InputCollect)
  dt_input <- select(InputCollect$dt_input, -any_of(InputCollect$unused_vars))
  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  factor_vars <- InputCollect$factor_vars
  rollingWindowStartWhich <- InputCollect$rollingWindowStartWhich
  rollingWindowEndWhich <- InputCollect$rollingWindowEndWhich

  # dt_inputRollWind
  dt_inputRollWind <- dt_input[rollingWindowStartWhich:rollingWindowEndWhich, ]

  # dt_transform
  dt_transform <- dt_input
  colnames(dt_transform)[colnames(dt_transform) == InputCollect$date_var] <- "ds"
  colnames(dt_transform)[colnames(dt_transform) == InputCollect$dep_var] <- "dep_var"
  dt_transform <- arrange(dt_transform, .data$ds)

  # dt_transformRollWind
  dt_transformRollWind <- dt_transform[rollingWindowStartWhich:rollingWindowEndWhich, ]

  ################################################################
  #### Model exposure metric from spend

  exposure_selector <- paid_media_spends != paid_media_vars
  names(exposure_selector) <- paid_media_vars

  if (any(exposure_selector)) {
    modNLSCollect <- list()
    yhatCollect <- list()
    plotNLSCollect <- list()
    mediaCostFactor <- colSums(subset(dt_inputRollWind, select = paid_media_spends), na.rm = TRUE) /
      colSums(subset(dt_inputRollWind, select = paid_media_vars), na.rm = TRUE)

    for (i in 1:InputCollect$mediaVarCount) {
      if (exposure_selector[i]) {
        # Run models (NLS and/or LM)
        dt_spendModInput <- subset(dt_inputRollWind, select = c(paid_media_spends[i], paid_media_vars[i]))
        results <- fit_spend_exposure(dt_spendModInput, mediaCostFactor[i], paid_media_vars[i])
        # Compare NLS & LM, takes LM if NLS fits worse
        mod <- results$res
        exposure_selector[i] <- if (is.null(mod$rsq_nls)) FALSE else mod$rsq_nls > mod$rsq_lm
        # Data to create plot
        dt_plotNLS <- data.frame(
          channel = paid_media_vars[i],
          yhatNLS = if (exposure_selector[i]) results$yhatNLS else results$yhatLM,
          yhatLM = results$yhatLM,
          y = results$data$exposure,
          x = results$data$spend
        )
        caption <- glued("
          nls: AIC = {aic_nls} | R2 = {r2_nls}
          lm: AIC = {aic_lm} | R2 = {r2_lm}",
          aic_nls = signif(AIC(if (exposure_selector[i]) results$modNLS else results$modLM), 3),
          r2_nls = signif(if (exposure_selector[i]) mod$rsq_nls else mod$rsq_lm, 3),
          aic_lm = signif(AIC(results$modLM), 3),
          r2_lm = signif(mod$rsq_lm, 3)
        )
        dt_plotNLS <- dt_plotNLS %>%
          pivot_longer(
            cols = c("yhatNLS", "yhatLM"),
            names_to = "models", values_to = "yhat"
          ) %>%
          mutate(models = str_remove(tolower(.data$models), "yhat"))
        models_plot <- ggplot(
          dt_plotNLS, aes(x = .data$x, y = .data$y, color = .data$models)
        ) +
          geom_point() +
          geom_line(aes(y = .data$yhat, x = .data$x, color = .data$models)) +
          labs(
            title = "Exposure-Spend Models Fit Comparison",
            x = sprintf("Spend [%s]", paid_media_spends[i]),
            y = sprintf("Exposure [%s]", paid_media_vars[i]),
            caption = caption,
            color = "Model"
          ) +
          theme_lares(background = "white", legend = "top") +
          scale_x_abbr() +
          scale_y_abbr()

        # Save results into modNLSCollect. plotNLSCollect, yhatCollect
        modNLSCollect[[paid_media_vars[i]]] <- mod
        plotNLSCollect[[paid_media_vars[i]]] <- models_plot
        yhatCollect[[paid_media_vars[i]]] <- dt_plotNLS
      }
    }
    modNLSCollect <- bind_rows(modNLSCollect)
    yhatNLSCollect <- bind_rows(yhatCollect)
    yhatNLSCollect$ds <- rep(dt_transformRollWind$ds, nrow(yhatNLSCollect) / nrow(dt_transformRollWind))
  } else {
    modNLSCollect <- plotNLSCollect <- yhatNLSCollect <- NULL
  }

  # Give recommendations and show warnings
  if (!is.null(modNLSCollect) && !quiet) {
    threshold <- 0.80
    final_print <- these <- NULL # TRUE if we accumulate a common message
    metrics <- c("R2 (nls)", "R2 (lm)")
    names(metrics) <- c("rsq_nls", "rsq_lm")
    for (m in seq_along(metrics)) {
      temp <- which(modNLSCollect[[names(metrics)[m]]] < threshold)
      if (length(temp) > 0) {
        # warning(sprintf(
        #   "%s: weak relationship for %s and %s spend",
        #   metrics[m],
        #   v2t(modNLSCollect$channel[temp], and = "and"),
        #   ifelse(length(temp) > 1, "their", "its")
        # ))
        final_print <- TRUE
        these <- modNLSCollect$channel[temp]
      }
    }
    if (isTRUE(final_print)) {
      message(
        paste(
          "NOTE: potential improvement on splitting channels for better exposure fitting.",
          "Threshold (Minimum R2) =", threshold,
          "\n  Check: InputCollect$modNLS$plots outputs"
        ),
        "\n  Weak relationship for: ", v2t(these), " and their spend"
      )
    }
  }

  ################################################################
  #### Clean & aggregate data

  ## Transform all factor variables
  if (length(factor_vars) > 0) {
    dt_transform <- mutate_at(dt_transform, factor_vars, as.factor)
  }

  ################################################################
  #### Obtain prophet trend, seasonality and change-points

  if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0) {
    if (length(InputCollect[["custom_params"]]) > 0) {
      custom_params <- InputCollect[["custom_params"]]
    } else {
      custom_params <- list(...)
    } # custom_params <- list()
    robyn_args <- setdiff(
      unique(c(
        names(as.list(args(robyn_run))),
        names(as.list(args(robyn_outputs))),
        names(as.list(args(robyn_inputs))),
        names(as.list(args(robyn_refresh)))
      )),
      c("", "...")
    )
    prophet_custom_args <- setdiff(names(custom_params), robyn_args)
    # if (length(prophet_custom_args) > 0) {
    #   message(paste("Using custom prophet parameters:", paste(prophet_custom_args, collapse = ", ")))
    # }

    dt_transform <- prophet_decomp(
      dt_transform,
      dt_holidays = InputCollect$dt_holidays,
      prophet_country = InputCollect$prophet_country,
      prophet_vars = InputCollect$prophet_vars,
      prophet_signs = InputCollect$prophet_signs,
      factor_vars = factor_vars,
      context_vars = InputCollect$context_vars,
      organic_vars = InputCollect$organic_vars,
      paid_media_spends = paid_media_spends,
      intervalType = InputCollect$intervalType,
      dayInterval = InputCollect$dayInterval,
      custom_params = custom_params
    )
  }

  ################################################################
  #### Finalize enriched input

  dt_transform <- subset(dt_transform, select = c("ds", "dep_var", InputCollect$all_ind_vars))
  InputCollect[["dt_mod"]] <- dt_transform
  InputCollect[["dt_modRollWind"]] <- dt_transform[rollingWindowStartWhich:rollingWindowEndWhich, ]
  InputCollect[["dt_inputRollWind"]] <- dt_inputRollWind
  InputCollect[["modNLS"]] <- list(
    results = modNLSCollect,
    yhat = yhatNLSCollect,
    plots = plotNLSCollect
  )
  return(InputCollect)
}


####################################################################
#' Conduct prophet decomposition
#'
#' When \code{prophet_vars} in \code{robyn_inputs()} is specified, this
#' function decomposes trend, season, holiday and weekday from the
#' dependent variable.
#'
#' @inheritParams robyn_inputs
#' @param dt_transform A data.frame with all model features.
#' Must contain \code{ds} column for time variable values and
#' \code{dep_var} column for dependent variable values.
#' @param context_vars,paid_media_spends,intervalType,dayInterval,prophet_country,prophet_vars,prophet_signs,factor_vars
#' As included in \code{InputCollect}
#' @param custom_params List. Custom parameters passed to \code{prophet()}
#' @return A list containing all prophet decomposition output.
prophet_decomp <- function(dt_transform, dt_holidays,
                           prophet_country, prophet_vars, prophet_signs,
                           factor_vars, context_vars, organic_vars, paid_media_spends,
                           intervalType, dayInterval, custom_params) {
  check_prophet(dt_holidays, prophet_country, prophet_vars, prophet_signs, dayInterval)
  recurrence <- select(dt_transform, .data$ds, .data$dep_var) %>% rename("y" = "dep_var")
  holidays <- set_holidays(dt_transform, dt_holidays, intervalType)
  use_trend <- "trend" %in% prophet_vars
  use_holiday <- "holiday" %in% prophet_vars
  use_season <- "season" %in% prophet_vars | "yearly.seasonality" %in% prophet_vars
  use_monthly <- "monthly" %in% prophet_vars
  use_weekday <- "weekday" %in% prophet_vars | "weekly.seasonality" %in% prophet_vars

  dt_regressors <- bind_cols(recurrence, select(
    dt_transform, all_of(c(paid_media_spends, context_vars, organic_vars))
  )) %>%
    mutate(ds = as.Date(.data$ds))

  prophet_params <- list(
    holidays = if (use_holiday) holidays[holidays$country %in% prophet_country, ] else NULL,
    yearly.seasonality = ifelse("yearly.seasonality" %in% names(custom_params),
      custom_params[["yearly.seasonality"]],
      use_season
    ),
    weekly.seasonality = ifelse("weekly.seasonality" %in% names(custom_params) & dayInterval <= 7,
      custom_params[["weekly.seasonality"]],
      use_weekday
    ),
    daily.seasonality = FALSE # No hourly models allowed
  )
  custom_params$yearly.seasonality <- custom_params$weekly.seasonality <- NULL
  prophet_params <- append(prophet_params, custom_params)
  modelRecurrence <- do.call(prophet, as.list(prophet_params))
  if (use_monthly) {
    modelRecurrence <- add_seasonality(
      modelRecurrence,
      name = "monthly", period = 30.5, fourier.order = 5
    )
  }

  # dt_regressors <<- dt_regressors
  # modelRecurrence <<- modelRecurrence

  if (!is.null(factor_vars) && length(factor_vars) > 0) {
    dt_ohe <- dt_regressors %>%
      select(all_of(factor_vars)) %>%
      ohse(drop = FALSE) %>%
      select(-any_of(factor_vars))
    ohe_names <- names(dt_ohe)
    for (addreg in ohe_names) modelRecurrence <- add_regressor(modelRecurrence, addreg)
    dt_ohe <- select(dt_regressors, -all_of(factor_vars)) %>% bind_cols(dt_ohe)
    mod_ohe <- fit.prophet(modelRecurrence, dt_ohe)
    dt_forecastRegressor <- predict(mod_ohe, dt_ohe)
    forecastRecurrence <- select(dt_forecastRegressor, -contains("_lower"), -contains("_upper"))
    for (aggreg in factor_vars) {
      oheRegNames <- grep(paste0("^", aggreg, ".*"), names(forecastRecurrence), value = TRUE)
      get_reg <- rowSums(select(forecastRecurrence, all_of(oheRegNames)))
      dt_transform[, aggreg] <- scale(get_reg, center = min(get_reg), scale = FALSE)
    }
  } else {
    if (dayInterval == 1) {
      warning(
        "Currently, there's a known issue with prophet that may crash this use case.",
        "\n Read more here: https://github.com/facebookexperimental/Robyn/issues/472"
      )
    }
    mod <- fit.prophet(modelRecurrence, dt_regressors)
    forecastRecurrence <- predict(mod, dt_regressors) # prophet::prophet_plot_components(modelRecurrence, forecastRecurrence)
  }

  these <- seq_along(unlist(recurrence[, 1]))
  if (use_trend) dt_transform$trend <- forecastRecurrence$trend[these]
  if (use_season) dt_transform$season <- forecastRecurrence$yearly[these]
  if (use_monthly) dt_transform$monthly <- forecastRecurrence$monthly[these]
  if (use_weekday) dt_transform$weekday <- forecastRecurrence$weekly[these]
  if (use_holiday) dt_transform$holiday <- forecastRecurrence$holidays[these]
  return(dt_transform)
}

####################################################################
#' Fit a nonlinear model for media spend and exposure
#'
#' This function is called in \code{robyn_engineering()}. It uses
#' the Michaelis-Menten function to fit the nonlinear model. Fallback
#' model is the simple linear model \code{lm()} in case the nonlinear
#' model is fitting worse. A bad fit here might result in unreasonable
#' model results. Two options are recommended: Either splitting the
#' channel into sub-channels to achieve better fit, or just use
#' spend as \code{paid_media_vars}
#'
#' @param dt_spendModInput data.frame. Containing channel spends and
#' exposure data.
#' @param mediaCostFactor Numeric vector. The ratio between raw media
#' exposure and spend metrics.
#' @param paid_media_var Character. Paid media variable.
#' @return List. Containing the all spend-exposure model results.
fit_spend_exposure <- function(dt_spendModInput, mediaCostFactor, paid_media_var) {
  if (ncol(dt_spendModInput) != 2) stop("Pass only 2 columns")
  colnames(dt_spendModInput) <- c("spend", "exposure")

  # Model 1: Michaelis-Menten model Vmax * spend/(Km + spend)
  tryCatch(
    {
      nlsStartVal <- list(
        Vmax = max(dt_spendModInput$exposure),
        Km = max(dt_spendModInput$exposure) / 2
      )

      modNLS <- nlsLM(exposure ~ Vmax * spend / (Km + spend),
        data = dt_spendModInput,
        start = nlsStartVal,
        control = nls.control(warnOnly = TRUE)
      )
      yhatNLS <- predict(modNLS)
      modNLSSum <- summary(modNLS)
      rsq_nls <- get_rsq(true = dt_spendModInput$exposure, predicted = yhatNLS)

      # # QA nls model prediction: check
      # yhatNLSQA <- modNLSSum$coefficients[1,1] * dt_spendModInput$spend / (modNLSSum$coefficients[2,1] + dt_spendModInput$spend) #exposure = v  * spend / (k + spend)
      # identical(yhatNLS, yhatNLSQA)
    },
    error = function(cond) {
      modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
    },
    warning = function(cond) {
      modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
    },
    finally = if (!exists("modNLS")) modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
  )

  # Model 2: Build lm comparison model
  modLM <- lm(exposure ~ spend - 1, data = dt_spendModInput)
  yhatLM <- predict(modLM)
  modLMSum <- summary(modLM)
  rsq_lm <- modLMSum$adj.r.squared
  if (is.na(rsq_lm)) stop("Please check if ", paid_media_var, " contains only 0s")
  # if (max(rsq_lm, rsq_nls) < 0.7) {
  #   warning(paste(
  #     "Spend-exposure fitting for", paid_media_var,
  #     "has rsq = ", round(max(rsq_lm, rsq_nls), 4),
  #     "To increase the fit, try splitting the variable.",
  #     "Otherwise consider using spend instead."
  #   ))
  # }

  output <- list(
    res = data.frame(
      channel = paid_media_var,
      Vmax = if (!is.null(modNLS)) modNLSSum$coefficients[1, 1] else NA,
      Km = if (!is.null(modNLS)) modNLSSum$coefficients[2, 1] else NA,
      aic_nls = if (!is.null(modNLS)) AIC(modNLS) else NA,
      aic_lm = AIC(modLM),
      bic_nls = if (!is.null(modNLS)) BIC(modNLS) else NA,
      bic_lm = BIC(modLM),
      rsq_nls = if (!is.null(modNLS)) rsq_nls else 0,
      rsq_lm = rsq_lm,
      coef_lm = coef(modLMSum)[1]
    ),
    yhatNLS = yhatNLS,
    modNLS = modNLS,
    yhatLM = yhatLM,
    modLM = modLM,
    data = dt_spendModInput,
    type = ifelse(is.null(modNLS), "lm", "mm")
  )
  return(output)
}

####################################################################
#' Detect and set date variable interval
#'
#' Robyn only accepts daily, weekly and monthly data. This function
#' is only called in \code{robyn_engineering()}.
#'
#' @param dt_transform A data.frame. Transformed input data.
#' @param dt_holidays A data.frame. Raw input holiday data.
#' @param intervalType A character. Accepts one of the values:
#' \code{c("day","week","month")}
#' @return List. Containing the all spend-exposure model results.
set_holidays <- function(dt_transform, dt_holidays, intervalType) {
  opts <- c("day", "week", "month")
  if (!intervalType %in% opts) {
    stop("Pass a valid 'intervalType'. Any of: ", paste(opts, collapse = ", "))
  }

  if (intervalType == "day") {
    holidays <- dt_holidays
  }

  if (intervalType == "week") {
    weekStartInput <- lubridate::wday(dt_transform$ds[1], week_start = 1)
    if (!weekStartInput %in% c(1, 7)) stop("Week start has to be Monday or Sunday")
    holidays <- dt_holidays %>%
      mutate(ds = floor_date(as.Date(.data$ds, origin = "1970-01-01"), unit = "week", week_start = weekStartInput)) %>%
      select(.data$ds, .data$holiday, .data$country, .data$year) %>%
      group_by(.data$ds, .data$country, .data$year) %>%
      summarise(holiday = paste(.data$holiday, collapse = ", "), n = n())
  }

  if (intervalType == "month") {
    if (!all(day(dt_transform$ds) == 1)) {
      stop("Monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
    }
    holidays <- dt_holidays %>%
      # mutate(ds = cut(.data$ds, intervalType)) %>%
      mutate(ds = cut(as.Date(.data$ds, origin = "1970-01-01"), intervalType)) %>%
      select(.data$ds, .data$holiday, .data$country, .data$year) %>%
      group_by(.data$ds, .data$country, .data$year) %>%
      summarise(holiday = paste(.data$holiday, collapse = ", "), n = n())
  }

  return(holidays)
}
