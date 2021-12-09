# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_inputs(), hyper_names(), robyn_engineering()

####################################################################
#' Input data sanity check & transformation
#'
#' \code{robyn_inputs()} is the function to input all model parameters and
#' check input correctness for the initial model build. It includes the
#' \code{robyn_engineering()} function that conducts trend, season,
#' holiday & weekday decomposition using Facebook's time-serie forecasting
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
#' @param dt_input A data.frame. Raw input data. Load simulated
#' dataset using \code{data("dt_simulated_weekly")}
#' @param dt_holidays A data.frame. Raw input holiday data. Load standard
#' Prophet holidays using \code{data("dt_prophet_holidays")}
#' @param date_var A character. Name of date variable. Daily, weekly
#' and monthly data supported. Weekly requires weekstart of Monday or Sunday.
#' date_var must have format "2020-01-01". Default to automatic date detection.
#' @param dep_var Character. Name of dependent variable. Only one allowed
#' @param dep_var_type Character. Type of dependent variable
#' as "revenue" or "conversion". Only one allowed and case sensitive.
#' @param prophet_vars Character vector. Include any of "trend",
#' "season", "weekday", "holiday". Are case-sensitive. Highly recommended
#' to use all for daily data and "trend", "season", "holiday" for
#' weekly and above cadence.
#' @param prophet_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for prophet variables. Must have same
#' order and same length as \code{prophet_vars}.
#' @param prophet_country Character. Only one country allowed once.
#' Including national holidays for 59 countries, whose list can
#' be found loading \code{data("dt_prophet_holidays")}.
#' @param context_vars Character vector. Typically competitors,
#' price & promotion, temperature, unemployment rate, etc.
#' @param context_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for context_vars. Must have same
#' order and same length as \code{context_vars}.
#' @param paid_media_vars Character vector. Recommended to use exposure
#' level metrics (impressions, clicks, GRP etc) other than spend. Also
#' recommended to split media channel into sub-channels
#' (e.g. fb_retargeting, fb_prospecting etc.) to gain more variance.
#' paid_media_vars only accept numerical variable
#' @param paid_media_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for paid_media_vars. Must have same
#' order and same length as \code{paid_media_vars}.
#' @param paid_media_spends Character vector. When using exposure level
#' metrics (impressions, clicks, GRP etc) in paid_media_vars, provide
#' corresponding spends for ROAS calculation. For spend metrics in
#' paid_media_vars, use the same name. media_spend_vars must have same
#' order and same length as \code{paid_media_vars}.
#' @param organic_vars Character vector. Typically newsletter sendings,
#' push-notifications, social media posts etc. Compared to paid_media_vars
#' organic_vars are often  marketing activities without clear spends
#' @param organic_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for organic_signs. Must have same
#' order and same length as \code{organic_vars}.
#' @param factor_vars Character vector. Specify which of the provided
#' variables in organic_vars or context_vars should be forced as a factor
#' @param adstock Character. Choose any of \code{c("geometric", "weibull_cdf",
#' "weibull_pdf")}. Weibull adtock is a two-parametric function and thus more
#' flexible, but takes longer time than the traditional geometric one-parametric
#' function. CDF, or cumulative density function of the Weibull function allows
#' changing decay rate over time in both C and S shape, while the peak value will
#' always stay at the first period, meaning no lagged effect. PDF, or the
#' probability density function, enables peak value occuring after the first
#' period when shape >=1, allowing lagged effect. Run \code{plot_adstock()} to
#' see the difference visually. Time estimation: with geometric adstock, 2000
#' iterations * 5 trials on 8 cores, it takes less than 30 minutes. Both Weibull
#' options take up to twice as much time.
#' @param hyperparameters List containing hyperparameter lower and upper bounds.
#' Names of elements in list must be identical to output of \code{hyper_names()}
#' @param window_start Character. Set start date of modelling period.
#' Recommended to not start in the first date in dataset to gain adstock
#' effect from previous periods.
#' @param window_end Character. Set end date of modelling period. Recommended
#' to have columns to rows ratio in the input data to be >=10:1, or in other
#' words at least 10 observations to 1 independent variable.
#' @param cores Integer. Default to \code{parallel::detectCores()}
#' @param iterations Integer. Recommended 2000 for default
#' \code{nevergrad_algo = "TwoPointsDE"}
#' @param trials Integer. Recommended 5 for default
#' \code{nevergrad_algo = "TwoPointsDE"}
#' @param nevergrad_algo Character. Default to "TwoPointsDE". Options are
#' \code{c("DE","TwoPointsDE", "OnePlusOne", "DoubleFastGADiscreteOnePlusOne",
#' "DiscreteOnePlusOne", "PortfolioDiscreteOnePlusOne", "NaiveTBPSA",
#' "cGA", "RandomSearch")}
#' @param calibration_input A data.table. Optional provide experimental results.
#' Check "Guide for calibration source" section.
#' @param InputCollect Default to NULL. \code{robyn_inputs}'s output when
#' \code{hyperparameters} are not yet set.
#' @param ... Additional parameters passed to \code{prophet} functions.
#' @examples
#' # load similated input data
#' data("dt_simulated_weekly")
#'
#' # load standard prophet holidays
#' data("dt_prophet_holidays")
#' \dontrun{
#' InputCollect <- robyn_inputs(
#'   dt_input = dt_simulated_weekly,
#'   dt_holidays = dt_prophet_holidays,
#'   date_var = "DATE",
#'   dep_var = "revenue",
#'   dep_var_type = "revenue",
#'   prophet_vars = c("trend", "season", "holiday"),
#'   prophet_signs = c("default", "default", "default"),
#'   prophet_country = "DE",
#'   context_vars = c("competitor_sales_B", "events"),
#'   context_signs = c("default", "default"),
#'   paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
#'   paid_media_signs = c("positive", "positive", "positive", "positive", "positive"),
#'   paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
#'   organic_vars = c("newsletter"),
#'   organic_signs = c("positive"),
#'   factor_vars = c("events"),
#'   window_start = "2016-11-23",
#'   window_end = "2018-08-22",
#'   adstock = "geometric",
#'   iterations = 2000,
#'   trials = 5,
#'   hyperparameters = hyperparameters # to be defined separately
#'   , calibration_input = dt_calibration # to be defined separately
#' )
#' }
#' @return A list containing the all input parameters and modified input data from
#' \code{robyn_engineering()}. The list is passed to further functions like
#' \code{robyn_run()}, \code{robyn_save()} and \code{robyn_allocator()}
#' @export
robyn_inputs <- function(dt_input = NULL,
                         dt_holidays = NULL,
                         date_var = "auto",
                         dep_var = NULL,
                         dep_var_type = NULL,
                         prophet_vars = NULL,
                         prophet_signs = NULL,
                         prophet_country = NULL,
                         context_vars = NULL,
                         context_signs = NULL,
                         paid_media_vars = NULL,
                         paid_media_signs = NULL,
                         paid_media_spends = NULL,
                         organic_vars = NULL,
                         organic_signs = NULL,
                         factor_vars = NULL,
                         adstock = NULL,
                         hyperparameters = NULL,
                         window_start = NULL,
                         window_end = NULL,
                         cores = parallel::detectCores(),
                         iterations = 2000,
                         trials = 5,
                         nevergrad_algo = "TwoPointsDE",
                         calibration_input = NULL,
                         InputCollect = NULL,
                         ...) {

  ### Use case 1: running robyn_inputs() for the first time
  if (is.null(InputCollect)) {
    dt_input <- as.data.table(dt_input)
    dt_holidays <- as.data.table(dt_holidays)

    # check for NA values
    check_nas(dt_input)
    check_nas(dt_holidays)

    # check vars names (duplicates and valid)
    check_varnames(dt_input, dt_holidays,
                   dep_var, date_var,
                   context_vars, paid_media_vars,
                   organic_vars)

    ## check date input (and set dayInterval and intervalType)
    date_input <- check_datevar(dt_input, date_var)
    dt_input <- date_input$dt_input # sort date by ascending
    date_var <- date_input$date_var # when date_var = "auto"
    dayInterval <- date_input$dayInterval
    intervalType <- date_input$intervalType

    ## check dependent var
    check_depvar(dt_input, dep_var, dep_var_type)

    ## check prophet
    check_prophet(dt_holidays, prophet_country, prophet_vars, prophet_signs)

    ## check baseline variables (and maybe transform context_signs)
    context <- check_context(dt_input, context_vars, context_signs)
    context_signs <- context$context_signs

    ## check paid media variables (set mediaVarCount and maybe transform paid_media_signs)
    paidmedia <- check_paidmedia(dt_input, paid_media_vars, paid_media_signs, paid_media_spends)
    paid_media_signs <- paidmedia$paid_media_signs
    mediaVarCount <- paidmedia$mediaVarCount
    exposureVarName <- paid_media_vars[!(paid_media_vars == paid_media_spends)]

    ## check organic media variables (and maybe transform organic_signs)
    organic <- check_organicvars(dt_input, organic_vars, organic_signs)
    organic_signs <- organic$organic_signs

    ## check factor_vars
    check_factorvars(factor_vars, context_vars, organic_vars)

    ## check all vars
    all_media <- c(paid_media_vars, organic_vars)
    all_ind_vars <- c(prophet_vars, context_vars, all_media)
    check_allvars(all_ind_vars)

    ## check data dimension
    check_datadim(dt_input, all_ind_vars, rel = 10)

    ## check window_start & window_end (and transform parameters/data)
    windows <- check_windows(dt_input, date_var, all_media, window_start, window_end)
    dt_input <- windows$dt_input
    window_start <- windows$window_start
    rollingWindowStartWhich <- windows$rollingWindowStartWhich
    refreshAddedStart <- windows$refreshAddedStart
    window_end <- windows$window_end
    rollingWindowEndWhich <- windows$rollingWindowEndWhich
    rollingWindowLength <- windows$rollingWindowLength

    ## check adstock
    adstock <- check_adstock(adstock)

    ## check hyperparameters (if passed)
    check_hyperparameters(hyperparameters, adstock, all_media)

    ## check calibration and iters/trials
    calibration_input <- check_calibration(dt_input, date_var, calibration_input, dayInterval)

    ## collect input
    InputCollect <- output <- list(
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
      prophet_vars = prophet_vars,
      prophet_signs = prophet_signs,
      prophet_country = prophet_country,
      context_vars = context_vars,
      context_signs = context_signs,
      paid_media_vars = paid_media_vars,
      paid_media_signs = paid_media_signs,
      paid_media_spends = paid_media_spends,
      mediaVarCount = mediaVarCount,
      exposureVarName = exposureVarName,
      organic_vars = organic_vars,
      organic_signs = organic_signs,
      all_media = all_media,
      all_ind_vars = all_ind_vars,
      factor_vars = factor_vars,
      cores = cores,
      window_start = window_start,
      rollingWindowStartWhich = rollingWindowStartWhich,
      window_end = window_end,
      rollingWindowEndWhich = rollingWindowEndWhich,
      rollingWindowLength = rollingWindowLength,
      refreshAddedStart = refreshAddedStart,
      adstock = adstock,
      iterations = iterations,
      nevergrad_algo = nevergrad_algo,
      trials = trials,
      hyperparameters = hyperparameters,
      calibration_input = calibration_input
    )

    ### Use case 1: running robyn_inputs() for the first time
    if (!is.null(hyperparameters)) {
      ### conditional output 1.2
      ## running robyn_inputs() for the 1st time & 'hyperparameters' provided --> run robyn_engineering()
      check_iteration(calibration_input, iterations, trials)
      output <- robyn_engineering(InputCollect = InputCollect, ...)
    }
  } else {
    ### Use case 2: adding 'hyperparameters' and/or 'calibration_input' using robyn_inputs()
    ## check calibration and iters/trials
    calibration_input <- check_calibration(
      InputCollect$dt_input,
      InputCollect$date_var,
      calibration_input,
      InputCollect$dayInterval
    )
    ## update calibration_input
    if (!is.null(calibration_input)) InputCollect$calibration_input <- calibration_input
    if (is.null(InputCollect$hyperparameters) & is.null(hyperparameters)) {
      stop("must provide hyperparameters in robyn_inputs()")
    } else {
      ### conditional output 2.1
      ## 'hyperparameters' provided --> run robyn_engineering()
      ## update & check hyperparameters
      if (is.null(InputCollect$hyperparameters)) InputCollect$hyperparameters <- hyperparameters
      check_hyperparameters(InputCollect$hyperparameters, InputCollect$adstock, InputCollect$all_media)
      check_iteration(InputCollect$calibration_input, InputCollect$iterations, InputCollect$trials)
      output <- robyn_engineering(InputCollect = InputCollect, ...)
    }
  }
  return(output)
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
#'    and will be transformed by adstock & saturation. Difference between \code{organic_vars}
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
#' @param adstock A character. Default to \code{InputCollect$adstock}.
#' Accepts "geometric", "weibull_cdf" or "weibull_pdf"
#' @param all_media A character vector. Default to \code{InputCollect$all_media}.
#' Includes \code{InputCollect$paid_media_vars} and \code{InputCollect$organic_vars}.
#' @examples
#' \dontrun{
#' # Having InputCollect as robyn_inputs() output
#' # Define hyper_names for geometric adstock
#' hyper_names(adstock = "geometric", all_media = InputCollect$all_media)
#'
#' hyperparameters <- list(
#'   facebook_I_alphas = c(0.5, 3) # example bounds for alpha
#'   , facebook_I_gammas = c(0.3, 1) # example bounds for gamma
#'   , facebook_I_thetas = c(0, 0.3) # example bounds for theta
#'
#'   , print_S_alphas = c(0.5, 3),
#'   print_S_gammas = c(0.3, 1),
#'   print_S_thetas = c(0.1, 0.4),
#'   tv_S_alphas = c(0.5, 3),
#'   tv_S_gammas = c(0.3, 1),
#'   tv_S_thetas = c(0.3, 0.8),
#'   search_clicks_P_alphas = c(0.5, 3),
#'   search_clicks_P_gammas = c(0.3, 1),
#'   search_clicks_P_thetas = c(0, 0.3),
#'   ooh_S_alphas = c(0.5, 3),
#'   ooh_S_gammas = c(0.3, 1),
#'   ooh_S_thetas = c(0.1, 0.4),
#'   newsletter_alphas = c(0.5, 3),
#'   newsletter_gammas = c(0.3, 1),
#'   newsletter_thetas = c(0.1, 0.4)
#' )
#'
#' # Define hyper_names for weibull adstock
#' hyper_names(adstock = "weibull", all_media = InputCollect$all_media)
#'
#' hyperparameters <- list(
#'   facebook_I_alphas = c(0.5, 3) # example bounds for alpha
#'   , facebook_I_gammas = c(0.3, 1) # example bounds for gamma
#'   , facebook_I_shapes = c(0.0001, 2) # example bounds for shape
#'   , facebook_I_scales = c(0, 0.1) # example bounds for scale
#'
#'   , print_S_alphas = c(0.5, 3),
#'   print_S_gammas = c(0.3, 1),
#'   print_S_shapes = c(0.0001, 2),
#'   print_S_scales = c(0, 0.1),
#'   tv_S_alphas = c(0.5, 3),
#'   tv_S_gammas = c(0.3, 1),
#'   tv_S_shapes = c(0.0001, 2),
#'   tv_S_scales = c(0, 0.1),
#'   search_clicks_P_alphas = c(0.5, 3),
#'   search_clicks_P_gammas = c(0.3, 1),
#'   search_clicks_P_shapes = c(0.0001, 2),
#'   search_clicks_P_scales = c(0, 0.1),
#'   ooh_S_alphas = c(0.5, 3),
#'   ooh_S_gammas = c(0.3, 1),
#'   ooh_S_shapes = c(0.0001, 2),
#'   ooh_S_scales = c(0, 0.1),
#'   newsletter_alphas = c(0.5, 3),
#'   newsletter_gammas = c(0.3, 1),
#'   newsletter_shapes = c(0.0001, 2),
#'   newsletter_scales = c(0, 0.1)
#' )
#' }
#' @export
hyper_names <- function(adstock, all_media) {
  adstock <- check_adstock(adstock)
  global_name <- c("thetas", "shapes", "scales", "alphas", "gammas", "lambdas")
  if (adstock == "geometric") {
    local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% "thetas|alphas|gammas"]), 1, paste, collapse = "_"))
  } else if (adstock %in% c("weibull_cdf","weibull_pdf")) {
    local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% "shapes|scales|alphas|gammas"]), 1, paste, collapse = "_"))
  }
  return(local_name)
}

####################################################################
#' Apply prophet decomposition and spend exposure transformation
#'
#' This function is included in the \code{robyn_inputs()} function and
#' will only run after all the condition checks are passed. It
#' applies the decomposition of trend, season, holiday and weekday
#' from \code{prophet} and builds the nonlinear fitting model when
#' using non-spend variables in \code{paid_media_vars}, for example
#' impressions for Facebook variables.
#'
#' @inheritParams robyn_inputs
#' @param InputCollect Default to \code{InputCollect}
#' @return A list containing the all input parameters and modified input
#' data. The list is passed to further functions like
#' \code{robyn_run()}, \code{robyn_save()} and \code{robyn_allocator()}.
#' @export
robyn_engineering <- function(InputCollect, ...) {
  check_InputCollect(InputCollect)

  dt_input <- InputCollect$dt_input
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
  dt_transform <- dt_transform[order(dt_transform$ds), ]

  # dt_transformRollWind
  dt_transformRollWind <- dt_transform[rollingWindowStartWhich:rollingWindowEndWhich, ]

  ################################################################
  #### model exposure metric from spend

  mediaCostFactor <- colSums(subset(dt_inputRollWind, select = paid_media_spends), na.rm = TRUE) /
    colSums(subset(dt_inputRollWind, select = paid_media_vars), na.rm = TRUE)

  costSelector <- paid_media_spends != paid_media_vars
  names(costSelector) <- paid_media_vars

  if (any(costSelector)) {
    modNLSCollect <- list()
    yhatCollect <- list()
    plotNLSCollect <- list()

    for (i in 1:InputCollect$mediaVarCount) {
      if (costSelector[i]) {

        # run models (NLS and/or LM)
        dt_spendModInput <- subset(dt_inputRollWind, select = c(paid_media_spends[i], paid_media_vars[i]))
        results <- fit_spend_exposure(dt_spendModInput, mediaCostFactor[i], paid_media_vars[i])
        # compare NLS & LM, takes LM if NLS fits worse
        mod <- results$res
        costSelector[i] <- if (is.null(mod$rsq_nls)) FALSE else mod$rsq_nls > mod$rsq_lm
        # data to create plot
        dt_plotNLS <- data.table(
          channel = paid_media_vars[i],
          yhatNLS = if (costSelector[i]) results$yhatNLS else results$yhatLM,
          yhatLM = results$yhatLM,
          y = results$data$exposure,
          x = results$data$spend
        )
        dt_plotNLS <- melt.data.table(dt_plotNLS,
          id.vars = c("channel", "y", "x"),
          variable.name = "models", value.name = "yhat"
        )
        dt_plotNLS[, models := str_remove(tolower(models), "yhat")]
        # create plot
        models_plot <- ggplot(
          dt_plotNLS, aes(x = .data$x, y = .data$y, color = .data$models)
        ) +
          geom_point() +
          geom_line(aes(y = .data$yhat, x = .data$x, color = .data$models)) +
          labs(
            caption = paste0(
              "y=", paid_media_vars[i], ", x=", paid_media_spends[i],
              "\nnls: aic=", round(AIC(if (costSelector[i]) results$modNLS else results$modLM), 0),
              ", rsq=", round(if (costSelector[i]) mod$rsq_nls else mod$rsq_lm, 4),
              "\nlm: aic= ", round(AIC(results$modLM), 0), ", rsq=", round(mod$rsq_lm, 4)
            ),
            title = "Models fit comparison",
            x = "Spend", y = "Exposure", color = "Model"
          ) +
          theme_minimal() +
          theme(legend.position = "top", legend.justification = "left")

        # save results into modNLSCollect. plotNLSCollect, yhatCollect
        modNLSCollect[[paid_media_vars[i]]] <- mod
        plotNLSCollect[[paid_media_vars[i]]] <- models_plot
        yhatCollect[[paid_media_vars[i]]] <- dt_plotNLS
      }
    }

    modNLSCollect <- rbindlist(modNLSCollect)
    yhatNLSCollect <- rbindlist(yhatCollect)
    yhatNLSCollect$ds <- rep(dt_transformRollWind$ds, nrow(yhatNLSCollect) / nrow(dt_transformRollWind))
  } else {
    modNLSCollect <- plotNLSCollect <- yhatNLSCollect <- NULL
  }

  # getSpendSum <- colSums(subset(dt_input, select = paid_media_spends), na.rm = TRUE)
  # getSpendSum <- data.frame(rn = paid_media_vars, spend = getSpendSum, row.names = NULL)

  ################################################################
  #### clean & aggregate data

  ## transform all factor variables
  if (length(factor_vars) > 0) {
    dt_transform[, (factor_vars) := lapply(.SD, as.factor), .SDcols = factor_vars]
  }

  ################################################################
  #### Obtain prophet trend, seasonality and change-points

  if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0) {
    dt_transform <- prophet_decomp(
      dt_transform,
      dt_holidays = InputCollect$dt_holidays,
      prophet_country = InputCollect$prophet_country,
      prophet_vars = InputCollect$prophet_vars,
      prophet_signs = InputCollect$prophet_signs,
      factor_vars = factor_vars,
      context_vars = InputCollect$context_vars,
      paid_media_vars = paid_media_vars,
      intervalType = InputCollect$intervalType,
      ...
    )
  }

  ################################################################
  #### Finalize enriched input

  dt_transform <- subset(dt_transform, select = c("ds", "dep_var", InputCollect$all_ind_vars))
  InputCollect[["dt_mod"]] <- dt_transform
  InputCollect[["dt_modRollWind"]] <- dt_transform[rollingWindowStartWhich:rollingWindowEndWhich, ]
  InputCollect[["dt_inputRollWind"]] <- dt_inputRollWind
  InputCollect[["modNLSCollect"]] <- modNLSCollect
  InputCollect[["plotNLSCollect"]] <- plotNLSCollect
  InputCollect[["yhatNLSCollect"]] <- yhatNLSCollect
  InputCollect[["costSelector"]] <- costSelector
  InputCollect[["mediaCostFactor"]] <- mediaCostFactor
  return(InputCollect)
}


####################################################################
#' Conduct prophet decomposition
#'
#' When \code{prophet_vars} in \code{robyn_inputs()} is specified, this
#' function decomposes trend, season, holiday and weekday from the
#' dependent variable.
#' @param dt_transform A data.frame with all model features.
#' @param dt_holidays As in \code{robyn_inputs()}
#' @param prophet_country As in \code{robyn_inputs()}
#' @param prophet_vars As in \code{robyn_inputs()}
#' @param prophet_signs As in \code{robyn_inputs()}
#' @param factor_vars As in \code{robyn_inputs()}
#' @param context_vars As in \code{robyn_inputs()}
#' @param paid_media_vars As in \code{robyn_inputs()}
#' @param intervalType As included in \code{InputCollect}
#' @param ... Additional prophet parameters
#' @return A list containing all prophet decomposition output.
prophet_decomp <- function(dt_transform, dt_holidays,
                           prophet_country, prophet_vars, prophet_signs,
                           factor_vars, context_vars, paid_media_vars, intervalType,
                           ...) {
  check_prophet(dt_holidays, prophet_country, prophet_vars, prophet_signs)
  recurrence <- subset(dt_transform, select = c("ds", "dep_var"))
  colnames(recurrence)[2] <- "y"

  holidays <- set_holidays(dt_transform, dt_holidays, intervalType)
  use_trend <- any(str_detect("trend", prophet_vars))
  use_season <- any(str_detect("season", prophet_vars))
  use_weekday <- any(str_detect("weekday", prophet_vars))
  use_holiday <- any(str_detect("holiday", prophet_vars))

  dt_regressors <- cbind(recurrence, subset(dt_transform, select = c(context_vars, paid_media_vars)))
  modelRecurrence <- prophet(
    holidays = if (use_holiday) holidays[country == prophet_country] else NULL,
    yearly.seasonality = use_season,
    weekly.seasonality = use_weekday,
    daily.seasonality = FALSE,
    ...
  )

  if (!is.null(factor_vars) && length(factor_vars) > 0) {
    dt_ohe <- as.data.table(model.matrix(y ~ ., dt_regressors[, c("y", factor_vars), with = FALSE]))[, -1]
    ohe_names <- names(dt_ohe)
    for (addreg in ohe_names) modelRecurrence <- add_regressor(modelRecurrence, addreg)
    dt_ohe <- cbind(dt_regressors[, !factor_vars, with = FALSE], dt_ohe)
    mod_ohe <- fit.prophet(modelRecurrence, dt_ohe)
    dt_forecastRegressor <- predict(mod_ohe, dt_ohe)
    forecastRecurrence <- dt_forecastRegressor[, str_detect(
      names(dt_forecastRegressor), "_lower$|_upper$",
      negate = TRUE
    ), with = FALSE]
    for (aggreg in factor_vars) {
      oheRegNames <- na.omit(str_extract(names(forecastRecurrence), paste0("^", aggreg, ".*")))
      forecastRecurrence[, (aggreg) := rowSums(.SD), .SDcols = oheRegNames]
      get_reg <- forecastRecurrence[, get(aggreg)]
      dt_transform[, (aggreg) := scale(get_reg, center = min(get_reg), scale = FALSE)]
    }
  } else {
    mod <- fit.prophet(modelRecurrence, dt_regressors)
    forecastRecurrence <- predict(mod, dt_regressors)
  }

  if (use_trend) {
    dt_transform$trend <- forecastRecurrence$trend[1:nrow(recurrence)]
  }
  if (use_season) {
    dt_transform$season <- forecastRecurrence$yearly[1:nrow(recurrence)]
  }
  if (use_weekday) {
    dt_transform$weekday <- forecastRecurrence$weekly[1:nrow(recurrence)]
  }
  if (use_holiday) {
    dt_transform$holiday <- forecastRecurrence$holidays[1:nrow(recurrence)]
  }

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
#' @param dt_spendModInput A data.frame with channel spends and exposure
#' data
#' @param mediaCostFactor A numeric vector. The ratio between raw media
#' exposure and spend metrics.
#' @param paid_media_vars A character vector. All paid media variables.
#' @return A list containing the all spend-exposure model results.
fit_spend_exposure <- function(dt_spendModInput, mediaCostFactor, paid_media_vars) {
  if (ncol(dt_spendModInput) != 2) stop("Pass only 2 columns")
  colnames(dt_spendModInput) <- c("spend", "exposure")

  # remove spend == 0 to avoid DIV/0 error
  # dt_spendModInput$spend[dt_spendModInput$spend == 0] <- 0.01
  # # adapt exposure with avg when spend == 0
  # dt_spendModInput$exposure <- ifelse(
  #   dt_spendModInput$exposure == 0, dt_spendModInput$spend / mediaCostFactor,
  #   dt_spendModInput$exposure
  # )

  # Model 1: Michaelis-Menten model Vmax * spend/(Km + spend)
  tryCatch(
    {
      nlsStartVal <- list(
        Vmax = dt_spendModInput[, max(exposure)],
        Km = dt_spendModInput[, max(exposure) / 2]
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
      message("Michaelis-Menten fitting for ", paid_media_vars, " out of range. Using lm instead")
      modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
    },
    warning = function(cond) {
      message("Michaelis-Menten fitting for ", paid_media_vars, " out of range. Using lm instead")
      modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
    },
    finally = {
      if (!exists("modNLS")) modNLS <- yhatNLS <- modNLSSum <- rsq_nls <- NULL
    }
  )

  # build lm comparison model
  modLM <- lm(exposure ~ spend - 1, data = dt_spendModInput)
  yhatLM <- predict(modLM)
  modLMSum <- summary(modLM)
  rsq_lm <- get_rsq(true = dt_spendModInput$exposure, predicted = yhatLM)
  if (is.na(rsq_lm)) {
    stop("Please check if ", paid_media_vars, " contains only 0s")
  }
  if (max(rsq_lm, rsq_nls) < 0.7) {
    warning(paste(
      "Spend-exposure fitting for", paid_media_vars,
      "has rsq = ", max(rsq_lm, rsq_nls),
      "To increase the fit, try splitting the variable.",
      "Otherwise consider using spend instead."
    ))
  }

  output <- list(
    res = data.table(
      channel = paid_media_vars,
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
    data = dt_spendModInput
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
#' @return A list containing the all spend-exposure model results.
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
    dt_holidays$dsWeekStart <- floor_date(dt_holidays$ds, unit = "week", week_start = 1)
    holidays <- dt_holidays[, .(ds = dsWeekStart, holiday, country, year)]
    holidays <- holidays[, lapply(.SD, paste0, collapse = "#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
  }

  if (intervalType == "month") {
    monthStartInput <- all(day(dt_transform[, ds]) == 1)
    if (!monthStartInput) {
      stop("Monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
    }
    dt_holidays[, dsMonthStart := cut(as.Date(ds), intervalType)]
    holidays <- dt_holidays[, .(ds = dsMonthStart, holiday, country, year)]
    holidays <- holidays[, lapply(.SD, paste0, collapse = "#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
  }

  return(holidays)
}
