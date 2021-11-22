# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

############# Auxiliary non-exported functions #############

opts_pnd <- c("positive", "negative", "default")

check_nas <- function(df) {
  if (sum(is.na(df)) > 0) {
    name <- deparse(substitute(df))
    stop(paste(
      "Dataset", name, "has missing values.",
      "These values must be removed or fixed for the model to properly run"
    ))
  }
}

check_varnames <- function(dt_input, dt_holidays,
                           dep_var, date_var,
                           context_vars, paid_media_vars,
                           organic_vars) {
  dfs <- list(dt_input = dt_input, dt_holidays = dt_holidays)
  for (i in seq_along(dfs)) {
    # Which names to check by data.frame
    table_name <- names(dfs[i])
    if (table_name == "dt_input") {
      vars <- c(
        dep_var, date_var, context_vars,
        paid_media_vars, organic_vars, "auto"
      )
    }
    if (table_name == "dt_holidays") {
      vars <- c("ds","country") # holiday?
    }
    df <- dfs[[i]]
    # COMMENTED: each check_xvar() will give a better clue
    # # Not present names
    # cols <- c(colnames(df), "auto")
    # if (!all(vars %in% cols)) {
    #   these <- vars[!vars %in% cols]
    #   stop(paste(
    #     "You have set variables that are not present in your", table_name, "dataframe.",
    #     "Check:", paste(these, collapse = ", ")
    #   ))
    # }
    # Duplicate names
    vars <- vars[vars != "auto"]
    if (length(vars) != length(unique(vars))) {
      these <- names(table(vars)[table(vars) > 1])
      stop(paste(
        "You have duplicated variable names for", table_name, "in different parameters.",
        "Check:", paste(these, collapse = ", ")
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
  if (is.null(date_var) | length(date_var) > 1 | !(date_var %in% names(dt_input))) {
    stop("You must provide only 1 correct date variable name for 'date_var'")
  }
  dt_input <- as.data.table(dt_input)
  dt_input <- dt_input[order(get(date_var))]
  date_var_idate <- as.IDate(dt_input[, get(date_var)])
  dt_input[, (date_var):= date_var_idate]
  inputLen <- length(date_var_idate)
  inputLenUnique <- length(unique(date_var_idate))
  if (inputLen != inputLenUnique) {
    stop("Date variable has duplicated dates. Please clean data first")
  }
  if (any(is.na(date_var_idate))) {
    stop("Dates in 'date_var' must have format '2020-12-31'")
  }
  if (any(apply(dt_input, 2, function(x) any(is.na(x) | is.infinite(x))))) {
    stop("'dt_input' has NA or Inf. Please clean data before you proceed")
  }
  dt_input <- dt_input[order(date_var_idate)]
  dayInterval <- as.integer(difftime(
    date_var_idate[2],
    date_var_idate[1],
    units = "days"
  ))
  intervalType <- if (dayInterval == 1) {
    "day"
  } else
  if (dayInterval == 7) {
    "week"
  } else
  if (dayInterval %in% 28:31) {
    "month"
  } else {
    stop(paste(date_var, "data has to be daily, weekly or monthly"))
  }
  invisible(return(list(
    date_var = date_var,
    dayInterval = dayInterval,
    intervalType = intervalType,
    dt_input = dt_input
  )))
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
  if (!(is.numeric(dt_input[, get(dep_var)]) | is.integer(dt_input[, get(dep_var)]))) {
    stop("'dep_var' must be a numeric or integer variable")
  }
  if (is.null(dep_var_type)) {
    stop("Must provide a dependent variable type for 'dep_var_type'")
  }
  if (!dep_var_type %in% c("conversion", "revenue") | length(dep_var_type) != 1) {
    stop("'dep_var_type' must be 'conversion' or 'revenue'")
  }
}

check_prophet <- function(dt_holidays, prophet_country, prophet_vars, prophet_signs) {
  if (is.null(prophet_vars)) {
    prophet_signs <- NULL
    prophet_country <- NULL
    return(invisible(NULL))
  }
  opts <- c("trend", "season", "weekday", "holiday")
  if (!all(prophet_vars %in% opts)) {
    stop("Allowed values for 'prophet_vars' are: ", paste(opts, collapse = ", "))
  }
  if (is.null(prophet_country) | length(prophet_country) > 1 |
    !prophet_country %in% unique(dt_holidays$country)) {
    stop(paste(
      "You must provide 1 country code in 'prophet_country' input.",
      length(unique(dt_holidays$country)), "countries are included:",
      paste(unique(dt_holidays$country), collapse = ", "),
      "\nIf your country is not available, please manually add it to 'dt_holidays'"
    ))
  }
  if (is.null(prophet_signs)) {
    prophet_signs <- rep("default", length(prophet_vars))
    message("'prophet_signs' were not provided. 'default' is used")
  }
  if (!all(prophet_signs %in% opts_pnd)) {
    stop("Allowed values for 'prophet_signs' are: ", paste(opts_pnd, collapse = ", "))
  }
  if (length(prophet_signs) != length(prophet_vars)) {
    stop("'prophet_signs' must have same length as 'prophet_vars'")
  }
}

check_context <- function(dt_input, context_vars, context_signs) {
  if (!is.null(context_vars) & !is.null(context_signs)) {
    if (!all(context_vars %in% names(dt_input))) {
      stop("Provided 'context_vars' is not valid because it's not in your input data")
    } else if (!all(context_signs %in% opts_pnd)) {
      stop("Allowed values for 'context_signs' are: ", paste(opts_pnd, collapse = ", "))
    } else if (length(context_signs) != length(context_vars)) {
      stop("'context_signs' must have same length as 'context_vars'")
    }
    return(invisible(list(context_signs = context_signs)))
  } else if (is.null(context_vars) & is.null(context_signs)) {
    return(invisible(list(context_signs = context_signs)))
  } else {
    stop("Provide 'context_vars' and 'context_signs' at the same time")
  }

  return(invisible(list(context_signs = context_signs)))
}

check_paidmedia <- function(dt_input, paid_media_vars, paid_media_signs, paid_media_spends) {
  if (is.null(paid_media_vars) | is.null(paid_media_spends)) {
    stop("Must provide 'paid_media_vars' and 'paid_media_spends'")
  }

  mediaVarCount <- length(paid_media_vars)
  spendVarCount <- length(paid_media_spends)

  if (!all(paid_media_vars %in% names(dt_input))) {
    stop("Provided 'paid_media_vars' is not included in input data")
  }
  if (!all(paid_media_spends %in% names(dt_input))) {
    stop("Provided 'paid_media_spends' is not included in input data")
  }
  if (is.null(paid_media_signs)) {
    paid_media_signs <- rep("positive", mediaVarCount)
    message("'paid_media_signs' were not provided. Using 'positive'")
  }
  if (!all(paid_media_signs %in% opts_pnd)) {
    stop("Allowed values for 'paid_media_signs' are: ", paste(opts_pnd, collapse = ", "))
  }
  if (length(paid_media_signs) != length(paid_media_vars)) {
    stop("'paid_media_signs' must have same length as 'paid_media_vars'")
  }
  if (spendVarCount != mediaVarCount) {
    stop("'paid_media_spends' must have same length as 'paid_media_vars'")
  }
  if (any(dt_input[, unique(c(paid_media_vars, paid_media_spends)), with = FALSE] < 0)) {
    check_media_names <- unique(c(paid_media_vars, paid_media_spends))
    check_media_val <- sapply(dt_input[, check_media_names, with = FALSE], function(X) {
      any(X < 0)
    })
    stop(
      paste(names(check_media_val)[check_media_val], collapse = ", "),
      "contains negative values. Media must be >=0"
    )
  }
  return(invisible(list(paid_media_signs = paid_media_signs, mediaVarCount = mediaVarCount)))
}

check_organicvars <- function(dt_input, organic_vars, organic_signs) {
  if (is.null(organic_vars)) {
    return(invisible(NULL))
  }
  if (!all(organic_vars %in% names(dt_input))) {
    stop("Provided 'organic_vars' is not included in input data")
  }
  if (!is.null(organic_vars) & is.null(organic_signs)) {
    organic_signs <- rep("positive", length(organic_vars))
    message("'organic_signs' were not provided. Using 'positive'")
  }
  if (!all(organic_signs %in% opts_pnd)) {
    stop("Allowed values for 'organic_signs' are: ", paste(opts_pnd, collapse = ", "))
  }
  if (length(organic_signs) != length(organic_vars)) {
    stop("'organic_signs' must have same length as 'organic_vars'")
  }
  return(invisible(list(organic_signs = organic_signs)))
}

check_factorvars <- function(factor_vars, context_vars, organic_vars) {
  if (!is.null(factor_vars)) {
    if (!all(factor_vars %in% c(context_vars, organic_vars))) {
      stop("'factor_vars' must be any from 'context_vars' or 'organic_vars' inputs")
    }
  }
}

check_allvars <- function(all_ind_vars) {
  if (length(all_ind_vars) != length(unique(all_ind_vars))) {
    stop("Input variables must have unique names")
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
}

check_windows <- function(dt_input, date_var, all_media, window_start, window_end) {
  if (is.null(window_start)) {
    window_start <- min(as.character(dt_input[, get(date_var)]))
  } else if (is.na(as.Date(window_start, "%Y-%m-%d"))) {
    stop("'window_start' must have format '2020-12-31'")
  } else if (window_start < min(as.character(dt_input[, get(date_var)]))) {
    window_start <- min(as.character(dt_input[, get(date_var)]))
    message("'window_start' is smaller than the earliest date in input data. It's set to the earliest date")
  } else if (window_start > max(as.character(dt_input[, get(date_var)]))) {
    stop("'window_start' can't be larger than the the latest date in input data")
  }

  rollingWindowStartWhich <- which.min(abs(difftime(
    as.Date(dt_input[, get(date_var)]),
    as.Date(window_start),
    units = "days"
  )))
  if (!(as.Date(window_start) %in% dt_input[, get(date_var)])) {
    window_start <- dt_input[rollingWindowStartWhich, get(date_var)]
    message("'window_start' is adapted to the closest date contained in input data: ", window_start)
  }
  refreshAddedStart <- window_start

  if (is.null(window_end)) {
    window_end <- max(as.character(dt_input[, get(date_var)]))
  } else if (is.na(as.Date(window_end, "%Y-%m-%d"))) {
    stop("'window_end' must have format '2020-12-31'")
  } else if (window_end > max(as.character(dt_input[, get(date_var)]))) {
    window_end <- max(as.character(dt_input[, get(date_var)]))
    message("'window_end' is larger than the latest date in input data. It's set to the latest date")
  } else if (window_end < window_start) {
    window_end <- max(as.character(dt_input[, get(date_var)]))
    message("'window_end' must be >= 'window_start.' It's set to latest date in input data")
  }

  rollingWindowEndWhich <- which.min(abs(difftime(as.Date(dt_input[, get(date_var)]), as.Date(window_end), units = "days")))
  if (!(as.Date(window_end) %in% dt_input[, get(date_var)])) {
    window_end <- dt_input[rollingWindowEndWhich, get(date_var)]
    message("'window_end' is adapted to the closest date contained in input data: ", window_end)
  }
  rollingWindowLength <- rollingWindowEndWhich - rollingWindowStartWhich + 1

  dt_init <- dt_input[rollingWindowStartWhich:rollingWindowEndWhich, all_media, with = FALSE]
  init_all0 <- colSums(dt_init) == 0
  if (any(init_all0)) {
    stop(
      "These media channels contains only 0 within training period ",
      dt_input[rollingWindowStartWhich, get(date_var)], " to ",
      dt_input[rollingWindowEndWhich, get(date_var)], ": ",
      paste(names(dt_init)[init_all0], collapse = ", "),
      "\nRecommendation: adapt InputCollect$window_start, remove or combine these channels"
    )
  }
  invisible(return(list(
    dt_input = dt_input,
    window_start = window_start,
    rollingWindowStartWhich = rollingWindowStartWhich,
    refreshAddedStart = refreshAddedStart,
    window_end = window_end,
    rollingWindowEndWhich = rollingWindowEndWhich,
    rollingWindowLength = rollingWindowLength
  )))
}

check_adstock <- function(adstock) {
  if (adstock == "weibull") adstock <- "weibull_cdf"
  if (!adstock %in% c("geometric", "weibull_cdf", "weibull_pdf")) {
    stop("'adstock' must be 'geometric', 'weibull_cdf' or 'weibull_pdf'")
  }
  return(adstock)
}

check_hyperparameters <- function(hyperparameters = NULL, adstock = NULL, all_media = NULL) {
  if (is.null(hyperparameters)) {
    message(paste(
      "'hyperparameters' are not provided yet. To include them, run",
      "robyn_inputs(InputCollect = InputCollect, hyperparameters = ...)"
    ))
  } else {
    local_name <- hyper_names(adstock, all_media)
    if (!identical(sort(names(hyperparameters)), local_name)) {
      stop(
        "'hyperparameters' must be a list and contain vectors or values named as followed: ",
        paste(local_name, collapse = ", ")
      )
    }
  }
}

check_calibration <- function(dt_input, date_var, calibration_input, dayInterval) {
  if (!is.null(calibration_input)) {
    calibration_input <- as.data.table(calibration_input)
    if (!all(names(calibration_input) %in% c("channel", "liftStartDate", "liftEndDate", "liftAbs"))) {
      stop("calibration_input must contain columns 'channel', 'liftStartDate', 'liftEndDate', 'liftAbs'")
    }
    if ((min(calibration_input$liftStartDate) < min(dt_input[, get(date_var)])) |
      (max(calibration_input$liftEndDate) > (max(dt_input[, get(date_var)]) + dayInterval - 1))) {
      stop("We recommend you to only use lift results conducted within your MMM input data date range")
    }
  }
  return(calibration_input)
}

check_iteration <- function(calibration_input, iterations, trials) {
  if (is.null(calibration_input) & (iterations < 2000 | trials < 5)) {
    warning("We recommend to run at least 2000 iterations per trial and 5 trials to build initial model")
  } else if (!is.null(calibration_input) & (iterations < 2000 | trials < 10)) {
    warning(paste(
      "You are calibrating MMM. We recommend to run at least 2000 iterations per trial and",
      "10 trials to build initial model"
    ))
  }
}

check_InputCollect <- function(list) {
  names_list <- c(
    "dt_input", "paid_media_vars", "paid_media_spends", "context_vars",
    "organic_vars", "all_ind_vars", "date_var", "dep_var",
    "rollingWindowStartWhich", "rollingWindowEndWhich", "mediaVarCount",
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

check_robyn_object <- function(robyn_object) {
  file_end <- substr(robyn_object, nchar(robyn_object)-5, nchar(robyn_object))
  if (file_end == ".RData") {stop("robyn_object must has format .RDS, not .RData")}
}


check_filedir <- function(plot_folder) {
  file_end <- substr(plot_folder, nchar(plot_folder)-3, nchar(plot_folder))
  if (file_end == ".RDS") {
    plot_folder <- dirname(plot_folder)
    message("Using robyn object location: ", plot_folder)
  } else {
    plot_folder <- file.path(dirname(plot_folder), basename(plot_folder))
  }
  if (!dir.exists(plot_folder)) {
    plot_folder <- getwd()
    message("Provided 'plot_folder' doesn't exist. Using default 'plot_folder = getwd()': ", plot_folder)
  }
  return(plot_folder)
}

check_calibconstr <- function(calibration_constraint, iterations, trials, calibration_input) {
  if (!is.null(calibration_input)) {
    total_iters <- iterations * trials
    if (calibration_constraint <0.01 | calibration_constraint > 0.1) {
      calibration_constraint <- 0.1
      message("calibration_constraint must be >=0.01 and <=0.1. Using default value 0.1")
    } else if (total_iters * calibration_constraint < 500) {
      warning("Calibration constraint set to be top ", calibration_constraint*100, "% calibrated models.",
              " Only ", round(total_iters*calibration_constraint,0), " models left for pareto-optimal selection")
    }
  }
  return(calibration_constraint)
}
