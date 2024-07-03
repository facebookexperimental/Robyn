# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

############# Auxiliary non-exported functions #############

OPTS_PDN <- c("positive", "negative", "default")
HYPS_NAMES <- c("thetas", "shapes", "scales", "alphas", "gammas", "penalty")
HYPS_OTHERS <- c("lambda", "train_size")
LEGACY_PARAMS <- c("cores", "iterations", "trials", "intercept_sign", "nevergrad_algo")

check_nas <- function(df, channels = NULL) {
  if (!is.null(channels)) df <- select(df, all_of(channels))
  name <- deparse(substitute(df))
  if (sum(is.na(df)) > 0) {
    naVals <- lares::missingness(df)
    strs <- sprintf("%s (%s | %s%%)", naVals$variable, naVals$missing, naVals$missingness)
    stop(paste0(
      "Dataset ", name, " contains missing (NA) values. ",
      "These values must be removed or fixed for Robyn to properly work.\n  Missing values: ",
      paste(strs, collapse = ", ")
    ))
  }
  have_inf <- unlist(lapply(df, function(x) sum(is.infinite(x))))
  if (any(have_inf > 0)) {
    stop(paste0(
      "Dataset ", name, " contains Inf values. ",
      "These values must be removed or fixed for Robyn to properly work.\n  Check: ",
      paste(names(which(have_inf > 0)), collapse = ", ")
    ))
  }
}

check_novar <- function(dt_input, InputCollect = NULL) {
  novar <- lares::zerovar(dt_input)
  if (length(novar) > 0) {
    msg <- sprintf(
      "There are %s column(s) with no-variance: %s. \nPlease, remove the variable(s) to proceed...",
      length(novar), v2t(novar)
    )
    if (!is.null(InputCollect)) {
      msg <- sprintf(
        "%s\n>>> Note: there's no variance on these variables because of the modeling window filter (%s:%s)",
        msg,
        InputCollect$window_start,
        InputCollect$window_end
      )
    }
    stop(msg)
  }
}

check_allneg <- function(df) {
  all_negative <- unlist(lapply(df, function(x) all(x <= 0)))
  df <- mutate_at(df, names(which(all_negative)), function(x) abs(x))
  return(df)
}

check_varnames <- function(dt_input, dt_holidays,
                           dep_var, date_var,
                           context_vars, paid_media_spends,
                           organic_vars) {
  dfs <- list(dt_input = dt_input, dt_holidays = dt_holidays)
  for (i in seq_along(dfs)) {
    # Which names to check by data.frame
    table_name <- names(dfs[i])
    if (table_name == "dt_input") {
      vars <- c(
        dep_var, date_var, context_vars,
        paid_media_spends, organic_vars, "auto"
      )
    }
    if (table_name == "dt_holidays") {
      vars <- c("ds", "country") # holiday?
    }
    df <- dfs[[i]]
    vars <- vars[vars != "auto"]
    # Duplicate names
    if (length(vars) != length(unique(vars))) {
      these <- names(table(vars)[table(vars) > 1])
      stop(paste(
        "You have duplicated variable names for", table_name, "in different parameters.",
        "Check:", paste(these, collapse = ", ")
      ))
    }
    # Names with spaces
    with_space <- grepl(" ", vars)
    if (sum(with_space) > 0) {
      stop(paste(
        "You have invalid variable names on", table_name, "with spaces.\n  ",
        "Please fix columns:", v2t(vars[with_space])
      ))
    }
  }
}

check_datevar <- function(dt_input, date_var = "auto") {
  if (date_var[1] == "auto") {
    is_date <- which(unlist(lapply(dt_input, is.Date)))
    if (length(is_date) == 1) {
      date_var <- names(is_date)
      message(paste("Automatically detected 'date_var':", date_var))
    } else {
      stop("Can't automatically find a single date variable to set 'date_var'")
    }
  }
  if (is.null(date_var) || length(date_var) > 1 || !(date_var %in% names(dt_input))) {
    stop("You must provide only 1 correct date variable name for 'date_var'")
  }
  dt_input <- data.frame(arrange(dt_input, as.factor(!!as.symbol(date_var))))
  dt_input[, date_var] <- as.Date(dt_input[[date_var]], origin = "1970-01-01")
  date_var_dates <- c(
    as.Date(dt_input[, date_var][[1]], origin = "1970-01-01"),
    as.Date(dt_input[, date_var][[2]], origin = "1970-01-01")
  )
  if (any(table(date_var_dates) > 1)) {
    stop("Date variable shouldn't have duplicated dates (panel data)")
  }
  if (any(is.na(date_var_dates)) || any(is.infinite(date_var_dates))) {
    stop("Dates in 'date_var' must have format '2020-12-31' and can't contain NA nor Inf values")
  }
  dayInterval <- as.integer(difftime(
    date_var_dates[2],
    date_var_dates[1],
    units = "days"
  ))
  intervalType <- if (dayInterval == 1) {
    "day"
  } else if (dayInterval == 7) {
    "week"
  } else if (dayInterval %in% 28:31) {
    "month"
  } else {
    stop(paste(date_var, "data has to be daily, weekly or monthly"))
  }
  output <- list(
    date_var = date_var,
    dayInterval = dayInterval,
    intervalType = intervalType,
    dt_input = as_tibble(dt_input)
  )
  invisible(return(output))
}

check_depvar <- function(dt_input, dep_var, dep_var_type) {
  if (is.null(dep_var)) {
    stop("Must provide a valid dependent variable name for 'dep_var'")
  }
  if (!dep_var %in% names(dt_input)) {
    stop("Must provide a valid dependent name for 'dep_var'")
  }
  if (length(dep_var) > 1) {
    stop("Must provide only 1 dependent variable name for 'dep_var'")
  }
  if (!(is.numeric(dt_input[, dep_var][[1]]) || is.integer(dt_input[, dep_var][[1]]))) {
    stop("'dep_var' must be a numeric or integer variable")
  }
  if (is.null(dep_var_type)) {
    stop("Must provide a dependent variable type for 'dep_var_type'")
  }
  if (!dep_var_type %in% c("conversion", "revenue") || length(dep_var_type) != 1) {
    stop("'dep_var_type' must be 'conversion' or 'revenue'")
  }
}

check_prophet <- function(dt_holidays, prophet_country, prophet_vars, prophet_signs, dayInterval) {
  check_vector(prophet_vars)
  check_vector(prophet_signs)
  if (is.null(dt_holidays) || is.null(prophet_vars)) {
    return(invisible(NULL))
  } else {
    prophet_vars <- tolower(prophet_vars)
    opts <- c("trend", "season", "monthly", "weekday", "holiday")
    if (!"holiday" %in% prophet_vars) {
      if (!is.null(prophet_country)) {
        warning(paste(
          "Input 'prophet_country' is defined as", prophet_country,
          "but 'holiday' is not setup within 'prophet_vars' parameter"
        ))
      }
      prophet_country <- NULL
    }
    if (!all(prophet_vars %in% opts)) {
      stop("Allowed values for 'prophet_vars' are: ", paste(opts, collapse = ", "))
    }
    if ("weekday" %in% prophet_vars && dayInterval > 7) {
      warning("Ignoring prophet_vars = 'weekday' input given your data granularity")
    }
    if ("holiday" %in% prophet_vars && (
      is.null(prophet_country) || length(prophet_country) > 1 |
        isTRUE(!prophet_country %in% unique(dt_holidays$country)))) {
      stop(paste(
        "You must provide 1 country code in 'prophet_country' input.",
        length(unique(dt_holidays$country)), "countries are included:",
        paste(unique(dt_holidays$country), collapse = ", "),
        "\nIf your country is not available, manually include data to 'dt_holidays'",
        "or remove 'holidays' from 'prophet_vars' input."
      ))
    }
    if (is.null(prophet_signs)) {
      prophet_signs <- rep("default", length(prophet_vars))
    }
    if (length(prophet_signs) == 1) {
      prophet_signs <- rep(prophet_signs, length(prophet_vars))
    }
    if (!all(prophet_signs %in% OPTS_PDN)) {
      stop("Allowed values for 'prophet_signs' are: ", paste(OPTS_PDN, collapse = ", "))
    }
    if (length(prophet_signs) != length(prophet_vars)) {
      stop("'prophet_signs' must have same length as 'prophet_vars'")
    }
    return(invisible(prophet_signs))
  }
}

check_context <- function(dt_input, context_vars, context_signs) {
  if (!is.null(context_vars)) {
    if (is.null(context_signs)) context_signs <- rep("default", length(context_vars))
    if (!all(context_signs %in% OPTS_PDN)) {
      stop("Allowed values for 'context_signs' are: ", paste(OPTS_PDN, collapse = ", "))
    }
    if (length(context_signs) != length(context_vars)) {
      stop("Input 'context_signs' must have same length as 'context_vars'")
    }
    temp <- context_vars %in% names(dt_input)
    if (!all(temp)) {
      stop(paste(
        "Input 'context_vars' not included in data. Check:",
        v2t(context_vars[!temp])
      ))
    }
    return(invisible(list(context_signs = context_signs)))
  }
}

check_vector <- function(x) {
  if (!is.null(names(x)) || is.list(x)) {
    stop(sprintf("Input '%s' must be a valid vector", deparse(substitute(x))))
  }
}

check_paidmedia <- function(dt_input, paid_media_vars, paid_media_signs, paid_media_spends) {
  if (is.null(paid_media_spends)) {
    stop("Must provide 'paid_media_spends'")
  }
  check_vector(paid_media_vars)
  check_vector(paid_media_signs)
  check_vector(paid_media_spends)
  expVarCount <- length(paid_media_vars)
  spendVarCount <- length(paid_media_spends)

  temp <- paid_media_vars %in% names(dt_input)
  if (!all(temp)) {
    stop(paste(
      "Input 'paid_media_vars' not included in data. Check:",
      v2t(paid_media_vars[!temp])
    ))
  }
  temp <- paid_media_spends %in% names(dt_input)
  if (!all(temp)) {
    stop(paste(
      "Input 'paid_media_spends' not included in data. Check:",
      v2t(paid_media_spends[!temp])
    ))
  }
  if (is.null(paid_media_signs)) {
    paid_media_signs <- rep("positive", expVarCount)
  }
  if (!all(paid_media_signs %in% OPTS_PDN)) {
    stop("Allowed values for 'paid_media_signs' are: ", paste(OPTS_PDN, collapse = ", "))
  }
  if (length(paid_media_signs) == 1) {
    paid_media_signs <- rep(paid_media_signs, length(paid_media_vars))
  }
  if (length(paid_media_signs) != length(paid_media_vars)) {
    stop("Input 'paid_media_signs' must have same length as 'paid_media_vars'")
  }
  if (spendVarCount != expVarCount) {
    stop("Input 'paid_media_spends' must have same length as 'paid_media_vars'")
  }
  is_num <- unlist(lapply(dt_input[, paid_media_vars], is.numeric))
  if (!all(is_num)) {
    stop("All your 'paid_media_vars' must be numeric. Check: ", v2t(paid_media_vars[!is_num]))
  }
  is_num <- unlist(lapply(dt_input[, paid_media_spends], is.numeric))
  if (!all(is_num)) {
    stop("All your 'paid_media_spends' must be numeric. Check: ", v2t(paid_media_spends[!is_num]))
  }
  get_cols <- any(dt_input[, unique(c(paid_media_vars, paid_media_spends))] < 0)
  if (get_cols) {
    check_media_names <- unique(c(paid_media_vars, paid_media_spends))
    df_check <- dt_input[, check_media_names]
    check_media_val <- unlist(lapply(df_check, function(x) any(x < 0)))
    stop(
      paste(names(check_media_val)[check_media_val], collapse = ", "),
      " contains negative values. Media must be >=0"
    )
  }
  return(invisible(list(
    paid_media_signs = paid_media_signs,
    expVarCount = expVarCount,
    paid_media_vars = paid_media_vars
  )))
}

check_organicvars <- function(dt_input, organic_vars, organic_signs) {
  if (is.null(organic_vars)) {
    return(invisible(NULL))
  }
  check_vector(organic_vars)
  check_vector(organic_signs)
  temp <- organic_vars %in% names(dt_input)
  if (!all(temp)) {
    stop(paste(
      "Input 'organic_vars' not included in data. Check:",
      v2t(organic_vars[!temp])
    ))
  }
  if (!is.null(organic_vars) && is.null(organic_signs)) {
    organic_signs <- rep("positive", length(organic_vars))
    # message("'organic_signs' were not provided. Using 'positive'")
  }
  if (!all(organic_signs %in% OPTS_PDN)) {
    stop("Allowed values for 'organic_signs' are: ", paste(OPTS_PDN, collapse = ", "))
  }
  if (length(organic_signs) != length(organic_vars)) {
    stop("Input 'organic_signs' must have same length as 'organic_vars'")
  }
  is_num <- unlist(lapply(dt_input[, organic_vars], is.numeric))
  if (!all(is_num)) {
    stop("All your 'organic_vars' must be numeric. Check: ", v2t(organic_vars[!is_num]))
  }
  return(invisible(list(organic_signs = organic_signs)))
}

check_factorvars <- function(dt_input, factor_vars = NULL, context_vars = NULL) {
  check_vector(factor_vars)
  check_vector(context_vars)
  temp <- select(dt_input, all_of(context_vars))
  are_not_numeric <- !sapply(temp, is.numeric)
  if (any(are_not_numeric)) {
    these <- are_not_numeric[!names(are_not_numeric) %in% factor_vars]
    these <- these[these]
    if (length(these) > 0) {
      message("Automatically set these variables as 'factor_vars': ", v2t(names(these)))
      factor_vars <- c(factor_vars, names(these))
    }
  }
  if (!is.null(factor_vars)) {
    if (!all(factor_vars %in% context_vars)) {
      stop("Input 'factor_vars' must be any from 'context_vars' inputs")
    }
  }
  return(factor_vars)
}

check_allvars <- function(all_ind_vars) {
  if (length(all_ind_vars) != length(unique(all_ind_vars))) {
    stop("All input variables must have unique names")
  }
}

check_datadim <- function(dt_input, all_ind_vars, rel = 10) {
  num_obs <- nrow(dt_input)
  if (num_obs < length(all_ind_vars) * rel) {
    warning(paste(
      "There are", length(all_ind_vars), "independent variables &",
      num_obs, "data points.", "We recommend row:column ratio of", rel, "to 1"
    ))
  }
  if (ncol(dt_input) <= 2) {
    stop("Provide a valid 'dt_input' input with at least 3 columns or more")
  }
}

check_windows <- function(dt_input, date_var, all_media, window_start, window_end) {
  dates_vec <- as.Date(dt_input[, date_var][[1]], origin = "1970-01-01")

  if (is.null(window_start)) {
    window_start <- min(dates_vec)
  } else {
    window_start <- as.Date(as.character(window_start), "%Y-%m-%d", origin = "1970-01-01")
    if (is.na(window_start)) {
      stop(sprintf("Input 'window_start' must have date format, i.e. '%s'", Sys.Date()))
    } else if (window_start < min(dates_vec)) {
      window_start <- min(dates_vec)
      message(paste(
        "Input 'window_start' is smaller than the earliest date in input data.",
        "It's automatically set to the earliest date:", window_start
      ))
    } else if (window_start > max(dates_vec)) {
      stop("Input 'window_start' can't be larger than the the latest date in input data: ", max(dates_vec))
    }
  }

  rollingWindowStartWhich <- which.min(abs(difftime(
    dates_vec,
    window_start,
    units = "days"
  )))
  if (!window_start %in% dates_vec) {
    window_start <- dt_input[rollingWindowStartWhich, date_var][[1]]
    message("Input 'window_start' is adapted to the closest date contained in input data: ", window_start)
  }
  refreshAddedStart <- window_start

  if (is.null(window_end)) {
    window_end <- max(dates_vec)
  } else {
    window_end <- as.Date(as.character(window_end), "%Y-%m-%d", origin = "1970-01-01")
    if (is.na(window_end)) {
      stop(sprintf("Input 'window_end' must have date format, i.e. '%s'", Sys.Date()))
    } else if (window_end > max(dates_vec)) {
      window_end <- max(dates_vec)
      message(paste(
        "Input 'window_end' is larger than the latest date in input data.",
        "It's automatically set to the latest date:", window_end
      ))
    } else if (window_end < window_start) {
      window_end <- max(dates_vec)
      message(paste(
        "Input 'window_end' must be >= 'window_start.",
        "It's automatically set to the latest date:", window_end
      ))
    }
  }

  rollingWindowEndWhich <- which.min(abs(difftime(dates_vec, window_end, units = "days")))
  if (!(window_end %in% dates_vec)) {
    window_end <- dt_input[rollingWindowEndWhich, date_var][[1]]
    message("Input 'window_end' is adapted to the closest date contained in input data: ", window_end)
  }
  rollingWindowLength <- rollingWindowEndWhich - rollingWindowStartWhich + 1

  dt_init <- dt_input[rollingWindowStartWhich:rollingWindowEndWhich, all_media]

  init_all0 <- dplyr::select_if(dt_init, is.numeric) %>% colSums(.) == 0
  if (any(init_all0)) {
    stop(
      "These media channels contains only 0 within training period ",
      dt_input[rollingWindowStartWhich, date_var][[1]], " to ",
      dt_input[rollingWindowEndWhich, date_var][[1]], ": ",
      paste(names(dt_init)[init_all0], collapse = ", "),
      "\nRecommendation: adapt InputCollect$window_start, remove or combine these channels"
    )
  }
  output <- list(
    dt_input = dt_input,
    window_start = window_start,
    rollingWindowStartWhich = rollingWindowStartWhich,
    refreshAddedStart = refreshAddedStart,
    window_end = window_end,
    rollingWindowEndWhich = rollingWindowEndWhich,
    rollingWindowLength = rollingWindowLength
  )
  return(invisible(output))
}

check_adstock <- function(adstock) {
  if (is.null(adstock)) {
    stop("Input 'adstock' can't be NULL. Set any of: 'geometric', 'weibull_cdf' or 'weibull_pdf'")
  }
  if (adstock == "weibull") adstock <- "weibull_cdf"
  if (!adstock %in% c("geometric", "weibull_cdf", "weibull_pdf")) {
    stop("Input 'adstock' must be 'geometric', 'weibull_cdf' or 'weibull_pdf'")
  }
  return(adstock)
}

check_hyperparameters <- function(hyperparameters = NULL, adstock = NULL,
                                  paid_media_spends = NULL, organic_vars = NULL,
                                  exposure_vars = NULL, prophet_vars = NULL,
                                  contextual_vars = NULL) {
  if (is.null(hyperparameters)) {
    message(paste(
      "Input 'hyperparameters' not provided yet. To include them, run",
      "robyn_inputs(InputCollect = InputCollect, hyperparameters = ...)"
    ))
  } else {
    if (!"train_size" %in% names(hyperparameters)) {
      hyperparameters[["train_size"]] <- c(0.5, 0.8)
      warning("Automatically added missing hyperparameter range: 'train_size' = c(0.5, 0.8)")
    }
    # Non-adstock hyperparameters check
    check_train_size(hyperparameters)
    # Adstock hyperparameters check
    hyperparameters_ordered <- hyperparameters[order(names(hyperparameters))]
    get_hyp_names <- names(hyperparameters_ordered)
    original_order <- sapply(names(hyperparameters), function(x) which(x == get_hyp_names))
    ref_hyp_name_spend <- hyper_names(adstock, all_media = paid_media_spends)
    ref_hyp_name_expo <- hyper_names(adstock, all_media = exposure_vars)
    ref_hyp_name_org <- hyper_names(adstock, all_media = organic_vars)
    ref_hyp_name_other <- get_hyp_names[get_hyp_names %in% HYPS_OTHERS]
    # Excluding lambda (first HYPS_OTHERS) given its range is not customizable
    ref_all_media <- sort(c(ref_hyp_name_spend, ref_hyp_name_org, HYPS_OTHERS))
    all_ref_names <- c(ref_hyp_name_spend, ref_hyp_name_expo, ref_hyp_name_org, HYPS_OTHERS)
    all_ref_names <- all_ref_names[order(all_ref_names)]
    # Adding penalty variations to the dictionary
    if (any(grepl("_penalty", paste0(get_hyp_names)))) {
      ref_hyp_name_penalties <- paste0(
        c(paid_media_spends, organic_vars, prophet_vars, contextual_vars), "_penalty"
      )
      all_ref_names <- c(all_ref_names, ref_hyp_name_penalties)
    } else {
      ref_hyp_name_penalties <- NULL
    }
    if (!all(get_hyp_names %in% all_ref_names)) {
      wrong_hyp_names <- get_hyp_names[which(!(get_hyp_names %in% all_ref_names))]
      stop(
        "Input 'hyperparameters' contains following wrong names: ",
        paste(wrong_hyp_names, collapse = ", ")
      )
    }
    total <- length(get_hyp_names)
    total_in <- length(c(ref_hyp_name_spend, ref_hyp_name_org, ref_hyp_name_penalties, ref_hyp_name_other))
    if (total != total_in) {
      stop(sprintf(
        paste(
          "%s hyperparameter values are required, and %s were provided.",
          "\n Use hyper_names() function to help you with the correct hyperparameters names."
        ),
        total_in, total
      ))
    }
    # Old workflow: replace exposure with spend hyperparameters
    if (any(get_hyp_names %in% ref_hyp_name_expo)) {
      get_expo_pos <- which(get_hyp_names %in% ref_hyp_name_expo)
      get_hyp_names[get_expo_pos] <- ref_all_media[!ref_all_media %in% HYPS_OTHERS][get_expo_pos]
      names(hyperparameters_ordered) <- get_hyp_names
    }
    check_hyper_limits(hyperparameters_ordered, "thetas")
    check_hyper_limits(hyperparameters_ordered, "alphas")
    check_hyper_limits(hyperparameters_ordered, "gammas")
    check_hyper_limits(hyperparameters_ordered, "shapes")
    check_hyper_limits(hyperparameters_ordered, "scales")
    hyperparameters_unordered <- hyperparameters_ordered[original_order]
    return(hyperparameters_unordered)
  }
}

check_train_size <- function(hyps) {
  if ("train_size" %in% names(hyps)) {
    if (!length(hyps$train_size) %in% 1:2) {
      stop("Hyperparameter 'train_size' must be length 1 (fixed) or 2 (range)")
    }
    if (any(hyps$train_size <= 0.1) || any(hyps$train_size > 1)) {
      stop("Hyperparameter 'train_size' values must be defined between 0.1 and 1")
    }
  }
}

check_hyper_limits <- function(hyperparameters, hyper) {
  hyper_which <- which(endsWith(names(hyperparameters), hyper))
  if (length(hyper_which) == 0) {
    return(invisible(NULL))
  }
  limits <- hyper_limits()[[hyper]]
  for (i in hyper_which) {
    values <- hyperparameters[[i]]
    # Lower limit
    ineq <- paste(values[1], limits[1], sep = "", collapse = "")
    lower_pass <- eval(parse(text = ineq))
    if (!lower_pass) {
      stop(sprintf("%s's hyperparameter must have lower bound %s", names(hyperparameters)[i], limits[1]))
    }
    # Upper limit
    ineq <- paste(values[2], limits[2], sep = "", collapse = "")
    upper_pass <- eval(parse(text = ineq)) | length(values) == 1
    if (!upper_pass) {
      stop(sprintf("%s's hyperparameter must have upper bound %s", names(hyperparameters)[i], limits[2]))
    }
    # Order of limits
    order_pass <- !isFALSE(values[1] <= values[2])
    if (!order_pass) {
      stop(sprintf("%s's hyperparameter must have lower bound first and upper bound second", names(hyperparameters)[i]))
    }
  }
}

check_calibration <- function(dt_input, date_var, calibration_input, dayInterval, dep_var,
                              window_start, window_end, paid_media_spends, organic_vars) {
  if (!is.null(calibration_input)) {
    calibration_input <- as_tibble(as.data.frame(calibration_input))
    these <- c("channel", "liftStartDate", "liftEndDate", "liftAbs", "spend", "confidence", "metric", "calibration_scope")
    if (!all(these %in% names(calibration_input))) {
      stop("Input 'calibration_input' must contain columns: ", v2t(these), ". Check the demo script for instruction.")
    }
    if (!is.numeric(calibration_input$liftAbs) || any(is.na(calibration_input$liftAbs))) {
      stop("Check 'calibration_input$liftAbs': all lift values must be valid numerical numbers")
    }
    all_media <- c(paid_media_spends, organic_vars)
    cal_media <- str_split(calibration_input$channel, "\\+|,|;|\\s")
    if (!all(unlist(cal_media) %in% all_media)) {
      these <- unique(unlist(cal_media)[which(!unlist(cal_media) %in% all_media)])
      stop(sprintf(
        "All channels from 'calibration_input' must be any of: %s.\n  Check: %s",
        v2t(all_media), v2t(these)
      ))
    }
    for (i in seq_along(calibration_input$channel)) {
      temp <- calibration_input[i, ]
      if (temp$liftStartDate < (window_start) || temp$liftEndDate > (window_end)) {
        stop(sprintf(
          paste(
            "Your calibration's date range for %s between %s and %s is not within modeling window (%s to %s).",
            "Please, remove this experiment from 'calibration_input'."
          ),
          temp$channel, temp$liftStartDate, temp$liftEndDate, window_start, window_end
        ))
      }
      if (temp$liftStartDate > temp$liftEndDate) {
        stop(sprintf(
          paste(
            "Your calibration's date range for %s between %s and %s should respect liftStartDate <= liftEndDate.",
            "Please, correct this experiment from 'calibration_input'."
          ),
          temp$channel, temp$liftStartDate, temp$liftEndDate
        ))
      }
    }
    if ("spend" %in% colnames(calibration_input)) {
      for (i in seq_along(calibration_input$channel)) {
        temp <- calibration_input[i, ]
        temp2 <- cal_media[[i]]
        if (all(temp2 %in% organic_vars)) next
        dt_input_spend <- filter(
          dt_input, get(date_var) >= temp$liftStartDate,
          get(date_var) <= temp$liftEndDate
        ) %>%
          select(all_of(temp2)) %>%
          sum(.) %>%
          round(., 0)
        if (dt_input_spend > temp$spend * 1.1 || dt_input_spend < temp$spend * 0.9) {
          warning(sprintf(
            paste(
              "Your calibration's spend (%s) for %s between %s and %s does not match your dt_input spend (~%s).",
              "Please, check again your dates or split your media inputs into separate media channels."
            ),
            formatNum(temp$spend, 0), temp$channel, temp$liftStartDate, temp$liftEndDate,
            formatNum(dt_input_spend, 3, abbr = TRUE)
          ))
        }
      }
    }
    if ("confidence" %in% colnames(calibration_input)) {
      for (i in seq_along(calibration_input$channel)) {
        temp <- calibration_input[i, ]
        if (temp$confidence < 0.8) {
          warning(sprintf(
            paste(
              "Your calibration's confidence for %s between %s and %s is lower than 80%%, thus low-confidence.",
              "Consider getting rid of this experiment and running it again."
            ),
            temp$channel, temp$liftStartDate, temp$liftEndDate
          ))
        }
      }
    }
    if ("metric" %in% colnames(calibration_input)) {
      for (i in seq_along(calibration_input$channel)) {
        temp <- calibration_input[i, ]
        if (temp$metric != dep_var) {
          stop(sprintf(
            paste(
              "Your calibration's metric for %s between %s and %s is not '%s'.",
              "Please, remove this experiment from 'calibration_input'."
            ),
            temp$channel, temp$liftStartDate, temp$liftEndDate, dep_var
          ))
        }
      }
    }
    if ("scope" %in% colnames(calibration_input)) {
      these <- c("immediate", "total")
      if (!all(calibration_input$scope %in% these)) {
        stop("Inputs in 'calibration_input$scope' must be any of: ", v2t(these))
      }
    }
  }
  return(calibration_input)
}

check_obj_weight <- function(calibration_input, objective_weights, refresh) {
  obj_len <- ifelse(is.null(calibration_input), 2, 3)
  if (!is.null(objective_weights)) {
    if ((length(objective_weights) != obj_len)) {
      stop(paste0("objective_weights must have length of ", obj_len))
    }
    if (any(objective_weights < 0) | any(objective_weights > 10)) {
      stop("objective_weights must be >= 0 & <= 10")
    }
  }
  if (is.null(objective_weights) & refresh) {
    if (obj_len == 2) {
      objective_weights <- c(0, 1)
    } else {
      objective_weights <- c(0, 1, 1)
    }
  }
  return(objective_weights)
}

check_iteration <- function(calibration_input, iterations, trials, hyps_fixed, refresh) {
  if (!refresh) {
    if (!hyps_fixed) {
      if (is.null(calibration_input) && (iterations < 2000 || trials < 5)) {
        warning("We recommend to run at least 2000 iterations per trial and 5 trials to build initial model")
      } else if (!is.null(calibration_input) && (iterations < 2000 || trials < 10)) {
        warning(paste(
          "You are calibrating MMM. We recommend to run at least 2000 iterations per trial and",
          "10 trials to build initial model"
        ))
      }
    }
  }
}

check_InputCollect <- function(list) {
  names_list <- c(
    "dt_input", "paid_media_vars", "paid_media_spends", "context_vars",
    "organic_vars", "all_ind_vars", "date_var", "dep_var",
    "rollingWindowStartWhich", "rollingWindowEndWhich",
    "factor_vars", "prophet_vars", "prophet_signs", "prophet_country",
    "intervalType", "dt_holidays"
  )
  if (!all(names_list %in% names(list))) {
    not_present <- names_list[!names_list %in% names(list)]
    stop(paste(
      "Some elements where not provided in your inputs list:",
      paste(not_present, collapse = ", ")
    ))
  }

  if (length(list$dt_input) <= 1) {
    stop("Check your 'dt_input' object")
  }
}

check_robyn_name <- function(robyn_object, quiet = FALSE) {
  if (!is.null(robyn_object)) {
    if (!dir.exists(robyn_object)) {
      file_end <- lares::right(robyn_object, 4)
      if (file_end != ".RDS") {
        stop("Input 'robyn_object' must has format .RDS")
      }
    }
  } else {
    if (!quiet) message("Skipping export into RDS file")
  }
}

check_dir <- function(plot_folder) {
  file_end <- substr(plot_folder, nchar(plot_folder) - 3, nchar(plot_folder))
  if (file_end == ".RDS") {
    plot_folder <- dirname(plot_folder)
    message("Using robyn object location: ", plot_folder)
  } else {
    plot_folder <- file.path(dirname(plot_folder), basename(plot_folder))
  }
  if (!dir.exists(plot_folder)) {
    plot_folder <- getwd()
    message("WARNING: Provided 'plot_folder' doesn't exist. Using current working directory: ", plot_folder)
  }
  return(plot_folder)
}

check_calibconstr <- function(calibration_constraint, iterations, trials, calibration_input, refresh) {
  if (!is.null(calibration_input) & !refresh) {
    total_iters <- iterations * trials
    if (calibration_constraint < 0.01 || calibration_constraint > 0.1) {
      message("Input 'calibration_constraint' must be >= 0.01 and <= 0.1. Changed to default: 0.1")
      calibration_constraint <- 0.1
    }
    models_lower <- 500
    if (total_iters * calibration_constraint < models_lower) {
      warning(sprintf(
        paste(
          "Input 'calibration_constraint' set for top %s%% calibrated models.",
          "%s models left for pareto-optimal selection. Minimum suggested: %s"
        ),
        calibration_constraint * 100,
        round(total_iters * calibration_constraint, 0),
        models_lower
      ))
    }
  }
  return(calibration_constraint)
}

check_hyper_fixed <- function(InputCollect, dt_hyper_fixed, add_penalty_factor) {
  hyper_fixed <- !is.null(dt_hyper_fixed)
  # Adstock hyper-parameters
  hypParamSamName <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
  # Add lambda and other hyper-parameters manually
  hypParamSamName <- c(hypParamSamName, HYPS_OTHERS)
  # Add penalty factor hyper-parameters names
  if (add_penalty_factor) {
    for_penalty <- names(select(InputCollect$dt_mod, -.data$ds, -.data$dep_var))
    hypParamSamName <- c(hypParamSamName, paste0(for_penalty, "_penalty"))
  }
  if (hyper_fixed) {
    ## Run robyn_mmm if using old model result tables
    dt_hyper_fixed <- as_tibble(dt_hyper_fixed)
    if (nrow(dt_hyper_fixed) != 1) {
      stop(paste(
        "Provide only 1 model / 1 row from OutputCollect$resultHypParam or",
        "pareto_hyperparameters.csv from previous runs"
      ))
    }
    if (!all(hypParamSamName %in% names(dt_hyper_fixed))) {
      these <- hypParamSamName[!hypParamSamName %in% names(dt_hyper_fixed)]
      stop(paste(
        "Input 'dt_hyper_fixed' is invalid.",
        "Please provide 'OutputCollect$resultHypParam' result from previous runs or",
        "'pareto_hyperparameters.csv' data with desired model ID. Missing values for:", v2t(these)
      ))
    }
  }
  attr(hyper_fixed, "hypParamSamName") <- hypParamSamName
  return(hyper_fixed)
}

# Enable parallelisation of main modelling loop for MacOS and Linux only
check_parallel <- function() "unix" %in% .Platform$OS.type
# ggplot doesn't work with process forking on MacOS; however it works fine on Linux and Windows
check_parallel_plot <- function() !"Darwin" %in% Sys.info()["sysname"]

check_init_msg <- function(InputCollect, cores) {
  opt <- sum(lapply(InputCollect$hyper_updated, length) == 2)
  fix <- sum(lapply(InputCollect$hyper_updated, length) == 1)
  det <- sprintf("(%s to iterate + %s fixed)", opt, fix)
  base <- paste(
    "Using", InputCollect$adstock, "adstocking with",
    length(InputCollect$hyper_updated), "hyperparameters", det
  )
  if (cores == 1) {
    message(paste(base, "with no parallel computation"))
  } else {
    message(paste(base, "on", cores, "cores"))
  }
}

check_class <- function(x, object) {
  if (any(!x %in% class(object))) stop(sprintf("Input object must be class %s", x))
}

check_allocator_constrains <- function(low, upr) {
  if (all(is.na(low)) || all(is.na(upr))) {
    stop("You must define lower (channel_constr_low) and upper (channel_constr_up) constraints")
  }
  max_length <- max(c(length(low), length(upr)))
  if (any(low < 0)) {
    stop("Inputs 'channel_constr_low' must be >= 0")
  }
  if (length(upr) != length(low)) {
    stop("Inputs 'channel_constr_up' and 'channel_constr_low' must have the same length or length 1")
  }
  if (any(upr < low)) {
    stop("Inputs 'channel_constr_up' must be >= 'channel_constr_low'")
  }
}

check_allocator <- function(OutputCollect, select_model, paid_media_spends, scenario,
                            channel_constr_low, channel_constr_up, constr_mode) {
  if (!(select_model %in% OutputCollect$allSolutions)) {
    stop(
      "Provided 'select_model' is not within the best results. Try any of: ",
      paste(OutputCollect$allSolutions, collapse = ", ")
    )
  }
  if ("max_historical_response" %in% scenario) scenario <- "max_response"
  opts <- c("max_response", "target_efficiency") # Deprecated: max_response_expected_spend
  if (!(scenario %in% opts)) {
    stop("Input 'scenario' must be one of: ", paste(opts, collapse = ", "))
  }
  check_allocator_constrains(channel_constr_low, channel_constr_up)
  if (!(scenario == "target_efficiency" & is.null(channel_constr_low) & is.null(channel_constr_up))) {
    if (length(channel_constr_low) != 1 && length(channel_constr_low) != length(paid_media_spends)) {
      stop(paste(
        "Input 'channel_constr_low' have to contain either only 1",
        "value or have same length as 'InputCollect$paid_media_spends':", length(paid_media_spends)
      ))
    }
    if (length(channel_constr_up) != 1 && length(channel_constr_up) != length(paid_media_spends)) {
      stop(paste(
        "Input 'channel_constr_up' have to contain either only 1",
        "value or have same length as 'InputCollect$paid_media_spends':", length(paid_media_spends)
      ))
    }
  }
  opts <- c("eq", "ineq")
  if (!(constr_mode %in% opts)) {
    stop("Input 'constr_mode' must be one of: ", paste(opts, collapse = ", "))
  }
  return(scenario)
}

check_metric_type <- function(metric_name, paid_media_spends, paid_media_vars, exposure_vars, organic_vars) {
  if (metric_name %in% paid_media_spends && length(metric_name) == 1) {
    metric_type <- "spend"
  } else if (metric_name %in% exposure_vars && length(metric_name) == 1) {
    metric_type <- "exposure"
  } else if (metric_name %in% organic_vars && length(metric_name) == 1) {
    metric_type <- "organic"
  } else {
    stop(paste(
      "Invalid 'metric_name' input:", metric_name,
      "\nInput should be any media variable from paid_media_spends (spend),",
      "paid_media_vars (exposure), or organic_vars (organic):",
      paste("\n- paid_media_spends:", v2t(paid_media_spends, quotes = FALSE)),
      paste("\n- paid_media_vars:", v2t(paid_media_vars, quotes = FALSE)),
      paste("\n- organic_vars:", v2t(organic_vars, quotes = FALSE))
    ))
  }
  return(metric_type)
}

check_metric_dates <- function(date_range = NULL, all_dates, dayInterval = NULL, quiet = FALSE, is_allocator = FALSE, ...) {
  ## default using latest 30 days / 4 weeks / 1 month for spend level
  if (is.null(date_range)) {
    if (is.null(dayInterval)) stop("Input 'date_range' or 'dayInterval' must be defined")
    # if (!is_allocator) {
    #   date_range <- "last_1"
    # } else {
    #   date_range <- paste0("last_", case_when(
    #     dayInterval == 1 ~ 30,
    #     dayInterval == 7 ~ 4,
    #     dayInterval >= 30 & dayInterval <= 31 ~ 1,
    #   ))
    # }
    date_range <- "all"
    if (!quiet) message(sprintf("Automatically picked date_range = '%s'", date_range))
  }
  if (grepl("last|all", date_range[1])) {
    ## Using last_n as date_range range
    if ("all" %in% date_range) date_range <- paste0("last_", length(all_dates))
    get_n <- ifelse(grepl("_", date_range[1]), as.integer(gsub("last_", "", date_range)), 1)
    date_range <- tail(all_dates, get_n)
    date_range_loc <- which(all_dates %in% date_range)
    date_range_updated <- all_dates[date_range_loc]
    rg <- v2t(range(date_range_updated), sep = ":", quotes = FALSE)
  } else {
    ## Using dates as date_range range
    if (all(is.Date(as.Date(date_range, origin = "1970-01-01")))) {
      date_range <- as.Date(date_range, origin = "1970-01-01")
      if (length(date_range) == 1) {
        ## Using only 1 date
        if (all(date_range %in% all_dates)) {
          date_range_updated <- date_range
          date_range_loc <- which(all_dates == date_range)
          if (!quiet) message("Using ds '", date_range_updated, "' as the response period")
        } else {
          date_range_loc <- which.min(abs(date_range - all_dates))
          date_range_updated <- all_dates[date_range_loc]
          if (!quiet) warning("Input 'date_range' (", date_range, ") has no match. Picking closest date: ", date_range_updated)
        }
      } else if (length(date_range) == 2) {
        ## Using two dates as "from-to" date range
        date_range_loc <- unlist(lapply(date_range, function(x) which.min(abs(x - all_dates))))
        date_range_loc <- date_range_loc[1]:date_range_loc[2]
        date_range_updated <- all_dates[date_range_loc]
        if (!quiet & !all(date_range %in% date_range_updated)) {
          warning(paste(
            "At least one date in 'date_range' input do not match any date.",
            "Picking closest dates for range:", paste(range(date_range_updated), collapse = ":")
          ))
        }
        rg <- v2t(range(date_range_updated), sep = ":", quotes = FALSE)
        get_n <- length(date_range_loc)
      } else {
        ## Manually inputting each date
        date_range_updated <- date_range
        if (all(date_range %in% all_dates)) {
          date_range_loc <- which(all_dates %in% date_range_updated)
        } else {
          date_range_loc <- unlist(lapply(date_range_updated, function(x) which.min(abs(x - all_dates))))
          rg <- v2t(range(date_range_updated), sep = ":", quotes = FALSE)
        }
        if (all(na.omit(date_range_loc - lag(date_range_loc)) == 1)) {
          date_range_updated <- all_dates[date_range_loc]
          if (!quiet) warning("At least one date in 'date_range' do not match ds. Picking closest date: ", date_range_updated)
        } else {
          stop("Input 'date_range' needs to have sequential dates")
        }
      }
    } else {
      stop("Input 'date_range' must have date format '2023-01-01' or use 'last_n'")
    }
  }
  return(list(
    date_range_updated = date_range_updated,
    metric_loc = date_range_loc
  ))
}

check_metric_value <- function(metric_value, metric_name, all_values, metric_loc) {
  get_n <- length(metric_loc)
  if (any(is.nan(metric_value))) metric_value <- NULL
  if (!is.null(metric_value)) {
    if (!is.numeric(metric_value)) {
      stop(sprintf(
        "Input 'metric_value' for %s (%s) must be a numerical value\n", metric_name, toString(metric_value)
      ))
    }
    if (any(metric_value < 0)) {
      stop(sprintf(
        "Input 'metric_value' for %s must be positive\n", metric_name
      ))
    }
    if (get_n > 1 & length(metric_value) == 1) {
      metric_value_updated <- rep(metric_value / get_n, get_n)
      # message(paste0("'metric_value'", metric_value, " splitting into ", get_n, " periods evenly"))
    } else {
      if (length(metric_value) != get_n) {
        stop("robyn_response metric_value & date_range must have same length\n")
      }
      metric_value_updated <- metric_value
    }
  }
  if (is.null(metric_value)) {
    metric_value_updated <- all_values[metric_loc]
  }
  all_values_updated <- all_values
  all_values_updated[metric_loc] <- metric_value_updated
  return(list(
    metric_value_updated = metric_value_updated,
    all_values_updated = all_values_updated
  ))
}

check_legacy_input <- function(InputCollect,
                               cores = NULL, iterations = NULL, trials = NULL,
                               intercept_sign = NULL, nevergrad_algo = NULL) {
  if (!any(LEGACY_PARAMS %in% names(InputCollect))) {
    return(invisible(InputCollect))
  } # Legacy check
  # Warn the user these InputCollect params will be (are) deprecated
  legacyValues <- InputCollect[LEGACY_PARAMS]
  legacyValues <- legacyValues[!unlist(lapply(legacyValues, is.null))]
  if (length(legacyValues) > 0) {
    warning(sprintf(
      "Using legacy InputCollect values. Please set %s within robyn_run() instead",
      v2t(names(legacyValues))
    ))
  }
  # Overwrite InputCollect with robyn_run() inputs
  if (!is.null(cores)) InputCollect$cores <- cores
  if (!is.null(iterations)) InputCollect$iterations <- iterations
  if (!is.null(trials)) InputCollect$trials <- trials
  if (!is.null(intercept_sign)) InputCollect$intercept_sign <- intercept_sign
  if (!is.null(nevergrad_algo)) InputCollect$nevergrad_algo <- nevergrad_algo
  attr(InputCollect, "deprecated_params") <- TRUE
  return(invisible(InputCollect))
}

check_run_inputs <- function(cores, iterations, trials, intercept_sign, nevergrad_algo) {
  if (is.null(iterations)) stop("Must provide 'iterations' in robyn_run()")
  if (is.null(trials)) stop("Must provide 'trials' in robyn_run()")
  if (is.null(nevergrad_algo)) stop("Must provide 'nevergrad_algo' in robyn_run()")
  opts <- c("non_negative", "unconstrained")
  if (!intercept_sign %in% opts) {
    stop(sprintf("Input 'intercept_sign' must be any of: %s", paste(opts, collapse = ", ")))
  }
}

check_daterange <- function(date_min, date_max, dates) {
  if (!is.null(date_min)) {
    if (length(date_min) > 1) stop("Set a single date for 'date_min' parameter")
    if (date_min < min(dates)) {
      warning(sprintf(
        "Parameter 'date_min' not in your data's date range. Changed to '%s'", min(dates)
      ))
    }
  }
  if (!is.null(date_max)) {
    if (length(date_max) > 1) stop("Set a single date for 'date_max' parameter")
    if (date_max > max(dates)) {
      warning(sprintf(
        "Parameter 'date_max' not in your data's date range. Changed to '%s'", max(dates)
      ))
    }
  }
}

check_refresh_data <- function(Robyn, dt_input) {
  original_periods <- nrow(Robyn$listInit$InputCollect$dt_modRollWind)
  new_periods <- nrow(filter(
    dt_input, get(Robyn$listInit$InputCollect$date_var) > Robyn$listInit$InputCollect$window_end
  ))
  it <- Robyn$listInit$InputCollect$intervalType
  if (new_periods > 0.5 * (original_periods + new_periods)) {
    warning(sprintf(
      paste(
        "We recommend re-building a model rather than refreshing this one.",
        "More than 50%% of your refresh data (%s %ss) is new data (%s %ss)"
      ),
      original_periods + new_periods, it, new_periods, it
    ))
  }
}
