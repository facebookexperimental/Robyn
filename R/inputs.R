# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function robyn_inputs(), hyper_names(), robyn_engineering()

####################################################################
#' Input data transformation
#'
#' Describe function.
#'
#' @param dt_input Input dataset as data.table. Load simulated 
#' dataset using \code{data("dt_input")} 
#' @param date_var Character. Name of date variable. Daily, weekly 
#' and monthly data supported. Weekly requires weekstart of Monday or Sunday.
#' date_var must have format "2020-01-01"
#' @param dep_var Character. Name of dependent variable. Only one allowed
#' @param dep_var_type Character. Type of dependent variable 
#' as "revenue" or "conversion". Only one allowed and case sensitive.
#' @param prophet_vars Character vector. Include any of "trend",
#' "season", "weekday", "holiday". Are case-sensitive. Highly recommended
#' to use all for daily data and "trend", "season", "holiday" for 
#' weekly and above cadence
#' @param prophet_signs Character vector. Choose any of
#' \code{c("default", "positive", "negative")}. Control
#' the signs of coefficients for prophet variables. Must have same
#' order and same length as \code{prophet_vars}.
#' @param prophet_country Character. Only one country allowed once.
#' Including national holidays for 59 countries, whose list can
#' be found loading \code{data(dt_holidays)}.
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
#' @param factor_var Character vector. Specify which of the provided
#' variables in organic_vars or context_vars should be forced as a factor
#' @param adstock Character. Choose any of \code{c("geometric", "weibull")}. 
#' Weibull adtock is a two-parametric function and thus more flexible, but 
#' takes longer time than the traditional geometric one-parametric function.
#' Time estimation: with geometric adstock, 2000 iterations * 5 trials on 8 
#' cores, it takes less than 30 minutes. Weibull takes at least twice as 
#' much time.
#' @param hyperparameters List containing hyperparameter lower and upper bounds. 
#' Names of elements in list must be identical to output of \code{hyper_names()}
#' @param window_start Character. Set start date of modelling period. 
#' Recommended to not start in the first date in dataset to gain adstock
#' effect from previous periods. 
#' @param window_end Character. Set end date of modelling period. Recommended
#' to have ratio of independent variable: data points of 1:10. 
#' @param cores Integer. Default to parallelly::availableCores() 
#' @param iterations Integer. Recommended 2000 for default 
#' \code{nevergrad_algo = "TwoPointsDE"} 
#' @param trials Integer. Recommended 5 for default 
#' \code{nevergrad_algo = "TwoPointsDE"} 
#' @param nevergrad_algo Character. Default to "TwoPointsDE". Options are
#' \code{c("DE","TwoPointsDE", "OnePlusOne", "DoubleFastGADiscreteOnePlusOne", 
#' "DiscreteOnePlusOne", "PortfolioDiscreteOnePlusOne", "NaiveTBPSA", 
#' "cGA", "RandomSearch")}
#' @param calibration_input A data.table. Optional provide experimental results.
#' @param InputCollect Default to NULL. Required when adding hyperparameters
#' Not yet implemented!
#' @examples
#' data("dt_input")
#' data("dt_holidays")
#' 
#' ## Define model input using simulated dataset
#' 
#' # Recommended to keep the object naming InputCollect
#' InputCollect <- robyn_inputs(dt_input = dt_input
#'                              ,dt_holidays = dt_holidays
#'                              
#'                              ,date_var = "DATE" 
#'                              ,dep_var = "revenue" 
#'                              ,dep_var_type = "revenue" 
#'                              
#'                              ,prophet_vars = c("trend", "season", "holiday") 
#'                              ,prophet_signs = c("default","default", "default") 
#'                              ,prophet_country = "DE" 
#'                              
#'                              ,context_vars = c("competitor_sales_B", "events") 
#'                              ,context_signs = c("default", "default")
#'                              
#'                              ,paid_media_vars = c("tv_S","ooh_S"	,	"print_S"	,"facebook_I"	,"search_clicks_P") 
#'                              ,paid_media_signs = c("positive", "positive","positive", "positive", "positive")
#'                              ,paid_media_spends = c("tv_S","ooh_S",	"print_S"	,"facebook_S"	,"search_S") 
#'                              
#'                              ,organic_vars = c("newsletter")
#'                              ,organic_signs = c("positive") 
#'                              
#'                              ,factor_vars = c("events") 
#'                                                            
#'                              ,window_start = "2016-11-23"
#'                              ,window_end = "2018-08-22"
#'                              
#'                              ,adstock = "geometric" 
#'                              ,iterations = 2000 
#'                              ,trials = 5 
#' )
#' 
#' ## Define hyperparameters
#' 
#' hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
#' 
#' hyperparameters <- list(
#'   facebook_I_alphas = c(0.5, 3) # example bounds for alpha
#'   ,facebook_I_gammas = c(0.3, 1) # example bounds for gamma
#'   ,facebook_I_thetas = c(0, 0.3) # example bounds for theta
#'   #,facebook_I_shapes = c(0.0001, 2) # example bounds for shape
#'   #,facebook_I_scales = c(0, 0.1) # example bounds for scale
#'   
#'   ,print_S_alphas = c(0.5, 3)
#'   ,print_S_gammas = c(0.3, 1)
#'   ,print_S_thetas = c(0.1, 0.4)
#'   #,print_S_shapes = c(0.0001, 2)
#'   #,print_S_scales = c(0, 0.1)
#'   
#'   ,tv_S_alphas = c(0.5, 3)
#'   ,tv_S_gammas = c(0.3, 1)
#'   ,tv_S_thetas = c(0.3, 0.8)
#'   #,tv_S_shapes = c(0.0001, 2)
#'   #,tv_S_scales= c(0, 0.1)
#'   
#'   ,search_clicks_P_alphas = c(0.5, 3)
#'   ,search_clicks_P_gammas = c(0.3, 1)
#'   ,search_clicks_P_thetas = c(0, 0.3)
#'   #,search_clicks_P_shapes = c(0.0001, 2)
#'   #,search_clicks_P_scales = c(0, 0.1)
#'   
#'   ,ooh_S_alphas = c(0.5, 3)
#'   ,ooh_S_gammas = c(0.3, 1)
#'   ,ooh_S_thetas = c(0.1, 0.4)
#'   #,ooh_S_shapes = c(0.0001, 2)
#'   #,ooh_S_scales = c(0, 0.1)
#'   
#'   ,newsletter_alphas = c(0.5, 3)
#'   ,newsletter_gammas = c(0.3, 1)
#'   ,newsletter_thetas = c(0.1, 0.4)
#'   #,newsletter_shapes = c(0.0001, 2)
#'   #,newsletter_scales = c(0, 0.1)
#' )
#' 
#' ## Add hyperparameters into robyn_inputs()
#' 
#' InputCollect <- robyn_inputs(InputCollect = InputCollect
#'                              , hyperparameters = hyperparameters)
#' @return List object
#' @export

robyn_inputs <- function(dt_input
                         ,dt_holidays
                         ,date_var = NULL 
                         ,dep_var = NULL 
                         ,dep_var_type = NULL 
                         ,prophet_vars = NULL 
                         ,prophet_signs = NULL 
                         ,prophet_country = NULL 
                         ,context_vars = NULL 
                         ,context_signs = NULL 
                         ,paid_media_vars = NULL 
                         ,paid_media_signs = NULL 
                         ,paid_media_spends = NULL 
                         ,organic_vars = NULL
                         ,organic_signs = NULL
                         ,factor_vars = NULL
                         ,adstock = "geometric"
                         ,hyperparameters = NULL
                         ,window_start = NULL 
                         ,window_end = NULL
                         ,cores = parallelly::availableCores()
                         ,iterations = 500  
                         ,trials = 40 
                         ,nevergrad_algo = "TwoPointsDE" 
                         ,calibration_input = data.table(channel = character(),
                                                         liftStartDate = Date(), 
                                                         liftEndDate = Date(), 
                                                         liftAbs = numeric()) 
                         ,InputCollect = NULL
                         
) {
  
  if (is.null(InputCollect)) {
    
    ## check date input
    inputLen <- length(dt_input[, get(date_var)])
    inputLenUnique <- length(unique(dt_input[, get(date_var)]))
    
    if (is.null(date_var) | !(date_var %in% names(dt_input)) | length(date_var)>1) {
      stop("Must provide correct only 1 date variable name for date_var")
    } else if (any(is.na(as.Date(as.character(dt_input[, get(date_var)]), "%Y-%m-%d")))) {
      stop("Date variable in date_var must have format '2020-12-31'")
    } else if (inputLen != inputLenUnique) {
      stop("Date variable has duplicated dates. Please clean data first")
    } else if (any(apply(dt_input, 2, function(x) any(is.na(x) | is.infinite(x))))) {
      stop("dt_input has NA or Inf. Please clean data first")
    }
    
    dt_input <- dt_input[order(as.Date(dt_input[, get(date_var)]))]
    dayInterval <- as.integer(difftime(as.Date(dt_input[, get(date_var)])[2], as.Date(dt_input[, get(date_var)])[1], units = "days"))
    intervalType <- if(dayInterval==1) {"day"} else if (dayInterval==7) {"week"} else if (dayInterval %in% 28:31) {"month"} else {stop("input data has to be daily, weekly or monthly")}
    
    ## check dependent var
    if (is.null(dep_var) | !(dep_var %in% names(dt_input)) | length(dep_var)>1) {
      stop("Must provide only 1 correct dependent variable name for dep_var")
    } else if ( !(is.numeric(dt_input[, get(dep_var)]) | is.integer(dt_input[, get(dep_var)]))) {
      stop("dep_var must be numeric or integer")
    } else if (!(dep_var_type %in% c("conversion", "revenue")) | length(dep_var_type)!=1) {
      stop("dep_var_type must be conversion or revenue")
    }
    
    ## check prophet
    if (is.null(prophet_vars)) {
      prophet_signs <- NULL; prophet_country <- NULL
    } else if (!all(prophet_vars %in% c("trend","season", "weekday", "holiday"))) {
      stop("allowed values for prophet_vars are 'trend', 'season', 'weekday' and 'holiday'")
    } else if (is.null(prophet_country) | length(prophet_country) >1) {
      stop("1 country code must be provided in prophet_country. ",dt_holidays[, uniqueN(country)], " countries are included: ", paste(dt_holidays[, unique(country)], collapse = ", "), ". If your country is not available, please add it to the holidays.csv first")
    } else if (is.null(prophet_signs)) {
      prophet_signs <- rep("default", length(prophet_vars))
      message("prophet_signs is not provided. 'default' is used")
    } else if (length(prophet_signs) != length(prophet_vars) | !all(prophet_signs %in% c("positive", "negative", "default"))) {
      stop("prophet_signs must have same length as prophet_vars. allowed values are 'positive', 'negative', 'default'")
    }
    
    ## check baseline variables
    if (is.null(context_vars)) {
      context_signs <- NULL
    } else if ( !all(context_vars %in% names(dt_input)) ) {
      stop("Provided context_vars is not included in input data")
    } else if (is.null(context_signs)) {
      context_signs <- rep("default", length(context_vars))
      message("context_signs is not provided. 'default' is used")
    } else if (length(context_signs) != length(context_vars) | !all(context_signs %in% c("positive", "negative", "default"))) {
      stop("context_signs must have same length as context_vars. allowed values are 'positive', 'negative', 'default'")
    }
    
    ## check paid media variables
    mediaVarCount <- length(paid_media_vars)
    spendVarCount <- length(paid_media_spends)
    if (is.null(paid_media_vars) | is.null(paid_media_spends)) {
      stop("Must provide paid_media_vars and paid_media_spends")
    } else if ( !all(paid_media_vars %in% names(dt_input)) ) {
      stop("Provided paid_media_vars is not included in input data")
    } else if (is.null(paid_media_signs)) {
      paid_media_signs <- rep("positive", mediaVarCount)
      message("paid_media_signs is not provided. 'positive' is used")
    } else if (length(paid_media_signs) != mediaVarCount | !all(paid_media_signs %in% c("positive", "negative", "default"))) {
      stop("paid_media_signs must have same length as paid_media_vars. allowed values are 'positive', 'negative', 'default'")
    } else if (!all(paid_media_spends %in% names(dt_input))) {
      stop("Provided paid_media_spends is not included in input data")
    } else if (spendVarCount != mediaVarCount) {
      stop("paid_media_spends must have same length as paid_media_vars.")
    } else if (any(dt_input[, unique(c(paid_media_vars, paid_media_spends)), with=FALSE]<0)) {
      check_media_names <- unique(c(paid_media_vars, paid_media_spends))
      check_media_val <- sapply(dt_input[, check_media_names, with=FALSE], function(X) { any(X <0) })
      stop( paste(names(check_media_val)[check_media_val], collapse = ", "), " contains negative values. Media must be >=0")
    }
    
    exposureVarName <- paid_media_vars[!(paid_media_vars==paid_media_spends)]
    
    ## check organic media variables
    if (!all(organic_vars %in% names(dt_input)) ) {
      stop("Provided organic_vars is not included in input data")
    } else if (!is.null(organic_vars) & is.null(organic_signs)) {
      organic_signs <- rep("positive", length(organic_vars))
      message("organic_signs is not provided. 'positive' is used")
    } else if (length(organic_signs) != length(organic_vars) | !all(organic_signs %in% c("positive", "negative", "default"))) {
      stop("organic_signs must have same length as organic_vars. allowed values are 'positive', 'negative', 'default'")
    }
    
    ## check factor_vars
    if (!is.null(factor_vars)) {
      if (!all(factor_vars %in% c(context_vars, organic_vars))) {stop("factor_vars must be from context_vars or organic_vars")}
    }
    
    ## check all vars
    all_media <- c(paid_media_vars, organic_vars)
    all_ind_vars <- unique(c(prophet_vars, context_vars, all_media))
    all_ind_vars_check <- c(prophet_vars, context_vars, all_media)
    if(!identical(all_ind_vars, all_ind_vars_check)) {stop("Input variables must have unique names")}
    
    ## check data dimension
    num_obs <- nrow(dt_input)
    all_ind_vars
    if (num_obs < length(all_ind_vars)*10 ) {
      message("There are ",length(all_ind_vars), " independent variables & ", num_obs, " data points. We recommend row:column ratio >= 10:1")
    }
    
    
    ## check window_start & window_end
    if (is.null(window_start)) {
      window_start <- min(as.character(dt_input[, get(date_var)]))
    } else if (is.na(as.Date(window_start, "%Y-%m-%d"))) {
      stop("window_start must have format '2020-12-31'")
    } else if (window_start < min(as.character(dt_input[, get(date_var)]))) {
      window_start <- min(as.character(dt_input[, get(date_var)]))
      message("window_start is smaller than the earliest date in input data. It's set to the earliest date")
    } else if (window_start > max(as.character(dt_input[, get(date_var)]))) {
      stop("window_start can't be larger than the the latest date in input data")
    }
    
    rollingWindowStartWhich <- which.min(abs(difftime(as.Date(dt_input[, get(date_var)]), as.Date(window_start), units = "days")))
    if (!(as.Date(window_start) %in% dt_input[, get(date_var)])) {
      window_start <- dt_input[rollingWindowStartWhich, get(date_var)]
      message("window_start is adapted to the closest date contained in input data: ", window_start)
    }
    refreshAddedStart <- window_start
    
    if (is.null(window_end)) {
      window_end <- max(as.character(dt_input[, get(date_var)]))
    } else if (is.na(as.Date(window_end, "%Y-%m-%d"))) {
      stop("window_end must have format '2020-12-31'")
    } else if (window_end > max(as.character(dt_input[, get(date_var)]))) {
      window_end <- max(as.character(dt_input[, get(date_var)]))
      message("window_end is larger than the latest date in input data. It's set to the latest date")
    } else if (window_end < window_start) {
      window_end <- max(as.character(dt_input[, get(date_var)]))
      message("window_end must be >= window_start. It's set to latest date in input data")
    }
    
    rollingWindowEndWhich <- which.min(abs(difftime(as.Date(dt_input[, get(date_var)]), as.Date(window_end), units = "days")))
    if (!(as.Date(window_end) %in% dt_input[, get(date_var)])) {
      window_end <- dt_input[rollingWindowEndWhich, get(date_var)]
      message("window_end is adapted to the closest date contained in input data: ", window_end)
    }
    
    rollingWindowLength <- rollingWindowEndWhich - rollingWindowStartWhich +1
    
    dt_init <- dt_input[rollingWindowStartWhich:rollingWindowEndWhich, all_media, with =FALSE]
    init_all0 <- colSums(dt_init)==0
    if(any(init_all0)) {
      stop("These media channels contains only 0 within training period ",dt_input[rollingWindowStartWhich, get(date_var)], " to ", dt_input[rollingWindowEndWhich, get(date_var)], ": ", paste(names(dt_init)[init_all0], collapse = ", ")
           , " \nRecommendation: adapt InputCollect$window_start, remove or combine these channels")
    }
    
    ## check adstock
    
    if((adstock %in% c("geometric", "weibull")) == FALSE) {stop("adstock must be 'geometric' or 'weibull'")}
    
    
    ## check calibration
    
    if(nrow(calibration_input)>0) {
      if ((min(calibration_input$liftStartDate) < min(dt_input[, get(date_var)])) | (max(calibration_input$liftEndDate) >  (max(dt_input[, get(date_var)]) + dayInterval-1))) {
        stop("we recommend you to only use lift results conducted within your MMM input data date range")
      } else if (iterations < 500 | trials < 80) {
        message("you are calibrating MMM. we recommend to run at least 500 iterations per trial and at least 80 trials at the beginning")
      }
    } else {
      if (iterations < 500 | trials < 40) {message("we recommend to run at least 500 iterations per trial and at least 40 trials at the beginning")}
    }
    
    ## get all hyper names
    global_name <- c("thetas",  "shapes",  "scales",  "alphas",  "gammas",  "lambdas")
    if (adstock == "geometric") {
      local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
    } else if (adstock == "weibull") {
      local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
    }
    
    ## collect input
    InputCollect <- list(dt_input=dt_input
                         , dt_holidays=dt_holidays
                         , dt_mod=NULL
                         , dt_modRollWind=NULL
                         , xDecompAggPrev = NULL
                         ,date_var=date_var
                         ,dayInterval=dayInterval
                         ,intervalType=intervalType
                         
                         ,dep_var=dep_var
                         ,dep_var_type=dep_var_type
                         
                         ,prophet_vars=prophet_vars
                         ,prophet_signs=prophet_signs 
                         ,prophet_country=prophet_country
                         
                         ,context_vars=context_vars 
                         ,context_signs=context_signs
                         
                         ,paid_media_vars=paid_media_vars
                         ,paid_media_signs=paid_media_signs
                         ,paid_media_spends=paid_media_spends
                         ,mediaVarCount=mediaVarCount
                         ,exposureVarName=exposureVarName
                         ,organic_vars=organic_vars
                         ,organic_signs=organic_signs
                         ,all_media=all_media
                         ,all_ind_vars=all_ind_vars
                         
                         ,factor_vars=factor_vars
                         
                         ,cores=cores
                         
                         ,window_start=window_start
                         ,rollingWindowStartWhich=rollingWindowStartWhich
                         ,window_end=window_end
                         ,rollingWindowEndWhich=rollingWindowEndWhich
                         ,rollingWindowLength=rollingWindowLength
                         ,refreshAddedStart=refreshAddedStart
                         
                         ,adstock=adstock
                         ,iterations=iterations
                         
                         ,nevergrad_algo=nevergrad_algo 
                         ,trials=trials 
                         
                         ,hyperparameters = hyperparameters 
                         ,local_name=local_name
                         ,calibration_input=calibration_input
    )
    
    ## check hyperparameter names in hyperparameters
    
    # when hyperparameters is not provided
    if (is.null(hyperparameters)) { 
      
      message("\nhyperparameters is not provided yet. run robyn_inputs(InputCollect = InputCollect, hyperparameter = ...) to add it\n")
      invisible(InputCollect)
      
      # when hyperparameters is provided wrongly
    } else if (!identical(sort(names(hyperparameters)), local_name)) {
      
      stop("\nhyperparameters must be a list and contain vectors or values named as followed: ", paste(local_name, collapse = ", "), "\n")
      
    } else {
      
      # when all provided once correctly
      message("\nAll input in robyn_inputs() correct. running robyn_engineering()")
      outFeatEng <- robyn_engineering(InputCollect = InputCollect, refresh = FALSE)
      invisible(outFeatEng)
      
    }
    
  } else if (!is.null(InputCollect) & is.null(hyperparameters)) {
    
    # when adding hyperparameters and InputCollect is provided, but hyperparameters not
    stop("\nhyperparameters is not provided yet. run robyn_inputs(InputCollect = InputCollect, hyperparameter = ...) to add it\n")
    
  } else {
    
    # when adding hyperparameters correctly
    global_name <- c("thetas",  "shapes",  "scales",  "alphas",  "gammas",  "lambdas")
    if (adstock == "geometric") {
      local_name <- sort(apply(expand.grid(InputCollect$all_media, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
    } else if (adstock == "weibull") {
      local_name <- sort(apply(expand.grid(InputCollect$all_media, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
    }
    
    if (!identical(sort(names(hyperparameters)), local_name)) {
      
      stop("\nhyperparameters must be a list and contain vectors or values named as followed: ", paste(local_name, collapse = ", "), "\n")
      
    } else {
      
      InputCollect$hyperparameters <- hyperparameters
      message("\nAll input in robyn_inputs() correct. running robyn_engineering()")
      outFeatEng <- robyn_engineering(InputCollect = InputCollect, refresh = FALSE)
      invisible(outFeatEng)
      
    }
  }
}

####################################################################
#' Get hyperparameter names
#'
#' Describe function.
#'
#' @param adstock Default to InputCollect$adstock
#' @param all_media Default to InputCollect$all_media
#' @export

hyper_names <- function(adstock = InputCollect$adstock, all_media=InputCollect$all_media) {
  global_name <- c("thetas",  "shapes",  "scales",  "alphas",  "gammas",  "lambdas")
  if (adstock == "geometric") {
    local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% 'thetas|alphas|gammas']), 1, paste, collapse="_"))
  } else if (adstock == "weibull") {
    local_name <- sort(apply(expand.grid(all_media, global_name[global_name %like% 'shapes|scales|alphas|gammas']), 1, paste, collapse="_"))
  }
  return(local_name)
}

####################################################################
#' Transform input data
#'
#' Describe function.
#'
#' @param InputCollect Default to InputCollect
#' @param refresh Default to FALSE. TRUE when using in robyn_refresh()
#' @export

robyn_engineering <- function(InputCollect = InputCollect
                              , refresh = FALSE) {
  
  paid_media_vars <- InputCollect$paid_media_vars
  paid_media_spends <- InputCollect$paid_media_spends
  context_vars <- InputCollect$context_vars
  organic_vars <- InputCollect$organic_vars
  all_media <- InputCollect$all_media
  all_ind_vars <- InputCollect$all_ind_vars
  
  dt_input <- copy(InputCollect$dt_input) # dt_input <- copy(InputCollect$dt_input)
  
  dt_inputRollWind <- dt_input[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich] 
  
  dt_transform <- copy(InputCollect$dt_input) # dt_transform <- copy(InputCollect$dt_input)
  setnames(dt_transform, InputCollect$date_var, "ds", skip_absent = TRUE)
  dt_transform <- dt_transform[, ':='(ds= as.Date(ds))][order(ds)]
  dt_transformRollWind <- dt_transform[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]
  
  setnames(dt_transform, InputCollect$dep_var, "dep_var", skip_absent = TRUE) 
  
  ################################################################
  #### model exposure metric from spend
  
  mediaCostFactor <- unlist(dt_inputRollWind[, lapply(.SD, sum), .SDcols = paid_media_spends] / dt_inputRollWind[, lapply(.SD, sum), .SDcols = paid_media_vars])
  names(mediaCostFactor) <- paid_media_vars
  costSelector <- !(paid_media_spends == paid_media_vars)
  names(costSelector) <- paid_media_vars
  
  if (any(costSelector)) {
    modNLSCollect <- list()
    yhatCollect <- list()
    plotNLSCollect <- list()
    for (i in 1:InputCollect$mediaVarCount) {
      if (costSelector[i]) {
        dt_spendModInput <- dt_inputRollWind[, c(paid_media_spends[i],paid_media_vars[i]), with =FALSE]
        setnames(dt_spendModInput, names(dt_spendModInput), c("spend", "exposure"))
        #dt_spendModInput <- dt_spendModInput[spend !=0 & exposure != 0]
        
        # scale 0 spend and exposure to a tiny number
        dt_spendModInput[, spend:=as.numeric(spend)][spend==0, spend:=0.01] # remove spend == 0 to avoid DIV/0 error
        dt_spendModInput[, exposure:=as.numeric(exposure)][exposure==0, exposure:=spend / mediaCostFactor[i]] # adapt exposure with avg when spend == 0
        
        tryCatch(
          {
            #dt_spendModInput[, exposure:= rep(0,(nrow(dt_spendModInput)))]
            nlsStartVal <- list(Vmax = dt_spendModInput[, max(exposure)], Km = dt_spendModInput[, max(exposure)/2])
            suppressWarnings(modNLS <- nlsLM(exposure ~ Vmax * spend/(Km + spend), #Michaelis-Menten model Vmax * spend/(Km + spend)
                                             data = dt_spendModInput,
                                             start = nlsStartVal
                                             ,control = nls.control(warnOnly = TRUE)))
            
            yhatNLS <- predict(modNLS)
            modNLSSum <- summary(modNLS)
            
            # QA nls model prediction
            yhatNLSQA <- modNLSSum$coefficients[1,1] * dt_spendModInput$spend / (modNLSSum$coefficients[2,1] + dt_spendModInput$spend) #exposure = v  * spend / (k + spend)
            identical(yhatNLS, yhatNLSQA)
            
            rsq_nls <- get_rsq(true = dt_spendModInput$exposure, predicted = yhatNLS)
          },
          
          error=function(cond) {
            message("michaelis menten fitting for ", paid_media_vars[i]," out of range. using lm instead")
          }
        )
        if (!exists("modNLS")) {modNLS <- NULL; yhatNLS <- NULL; modNLSSum <- NULL; rsq_nls <- NULL}
        # build lm comparison model
        modLM <- lm(exposure ~ spend-1, data = dt_spendModInput)
        yhatLM <- predict(modLM)
        modLMSum <- summary(modLM)
        rsq_lm <- get_rsq(true = dt_spendModInput$exposure, predicted = yhatLM) 
        if (is.na(rsq_lm)) {stop("please check if ",paid_media_vars[i]," constains only 0")}
        
        # compare NLS & LM, takes LM if NLS fits worse
        costSelector[i] <- if(is.null(rsq_nls)) {FALSE} else {rsq_nls > rsq_lm}
        
        modNLSCollect[[paid_media_vars[i]]] <- data.table(channel = paid_media_vars[i],
                                                          Vmax = if (!is.null(modNLS)) {modNLSSum$coefficients[1,1]} else {NA},
                                                          Km =  if (!is.null(modNLS)) {modNLSSum$coefficients[2,1]} else {NA},
                                                          aic_nls = if (!is.null(modNLS)) {AIC(modNLS)} else {NA},
                                                          aic_lm = AIC(modLM),
                                                          bic_nls = if (!is.null(modNLS)) {BIC(modNLS)} else {NA},
                                                          bic_lm = BIC(modLM),
                                                          rsq_nls = if (!is.null(modNLS)) {rsq_nls} else {0},
                                                          rsq_lm = rsq_lm,
                                                          coef_lm = coef(modLMSum)[1]
        )
        
        dt_plotNLS <- data.table(channel = paid_media_vars[i],
                                 yhatNLS = if(costSelector[i]) {yhatNLS} else {yhatLM},
                                 yhatLM = yhatLM,
                                 y = dt_spendModInput$exposure,
                                 x = dt_spendModInput$spend)
        dt_plotNLS <- melt.data.table(dt_plotNLS, id.vars = c("channel", "y", "x"), variable.name = "models", value.name = "yhat")
        dt_plotNLS[, models:= str_remove(tolower(models), "yhat")]
        
        yhatCollect[[paid_media_vars[i]]] <- dt_plotNLS
        
        # create plot
        plotNLSCollect[[paid_media_vars[i]]] <- ggplot(dt_plotNLS, aes(x=x, y=y, color = models)) +
          geom_point() +
          geom_line(aes(y=yhat, x=x, color = models)) +
          labs(subtitle = paste0("y=",paid_media_vars[i],", x=", paid_media_spends[i],
                                 "\nnls: aic=", round(AIC(if(costSelector[i]) {modNLS} else {modLM}),0), ", rsq=", round(if(costSelector[i]) {rsq_nls} else {rsq_lm},4),
                                 "\nlm: aic= ", round(AIC(modLM),0), ", rsq=", round(rsq_lm,4)),
               x = "spend",
               y = "exposure"
          ) +
          theme(legend.position = 'bottom')
        
      }
    }
    
    modNLSCollect <- rbindlist(modNLSCollect)
    yhatNLSCollect <- rbindlist(yhatCollect)
    yhatNLSCollect[, ds:= rep(dt_transformRollWind$ds, nrow(yhatNLSCollect)/nrow(dt_transformRollWind))]
    
  } else {
    modNLSCollect <- NULL
    plotNLSCollect <- NULL
    yhatNLSCollect <- NULL
  }
  
  getSpendSum <- dt_input[, lapply(.SD, sum), .SDcols=paid_media_spends]
  names(getSpendSum) <- paid_media_vars
  getSpendSum <- suppressWarnings(melt.data.table(getSpendSum, measure.vars= paid_media_vars, variable.name = "rn", value.name = "spend"))
  
  ################################################################
  #### clean & aggregate data
  
  ## transform all factor variables
  factor_vars <- InputCollect$factor_vars
  if (length(factor_vars)>0) {
    dt_transform[, (factor_vars):= lapply(.SD, as.factor), .SDcols = factor_vars ]
  } 
  
  ################################################################
  #### Obtain prophet trend, seasonality and changepoints
  
  if ( !is.null(InputCollect$prophet_vars) ) {
    
    if(length(InputCollect$prophet_vars) != length(InputCollect$prophet_signs)) {stop("InputCollect$prophet_vars and InputCollect$prophet_signs have to be the same length")}
    if(any(length(InputCollect$prophet_vars)==0, length(InputCollect$prophet_signs)==0)) {stop("InputCollect$prophet_vars and InputCollect$prophet_signs must be both specified")}
    if(!(InputCollect$prophet_country %in% InputCollect$dt_holidays$country)) {stop("InputCollect$prophet_country must be already included in the holidays.csv and as ISO 3166-1 alpha-2 abbreviation")}
    
    recurrance <- dt_transform[, .(ds = ds, y = dep_var)]
    use_trend <- any(str_detect("trend", InputCollect$prophet_vars))
    use_season <- any(str_detect("season", InputCollect$prophet_vars))
    use_weekday <- any(str_detect("weekday", InputCollect$prophet_vars))
    use_holiday <- any(str_detect("holiday", InputCollect$prophet_vars))
    
    if (InputCollect$intervalType == "day") {
      
      holidays <- InputCollect$dt_holidays
      
    } else if (InputCollect$intervalType == "week") {
      
      weekStartInput <- wday(dt_transform[1, ds])
      weekStartMonday <- if(weekStartInput==2) {TRUE} else if (weekStartInput==1) {FALSE} else {stop("week start has to be Monday or Sunday")}
      InputCollect$dt_holidays[, dsWeekStart:= cut(as.Date(ds), breaks = InputCollect$intervalType, start.on.monday = weekStartMonday)]
      holidays <- InputCollect$dt_holidays[, .(ds=dsWeekStart, holiday, country, year)]
      holidays <- holidays[, lapply(.SD, paste0, collapse="#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
      
    } else if (InputCollect$intervalType == "month") {
      
      monthStartInput <- all(day(dt_transform[, ds]) ==1)
      if (monthStartInput==FALSE) {stop("monthly data should have first day of month as datestampe, e.g.'2020-01-01' ")}
      InputCollect$dt_holidays[, dsMonthStart:= cut(as.Date(ds), InputCollect$intervalType)]
      holidays <- InputCollect$dt_holidays[, .(ds=dsMonthStart, holiday, country, year)]
      holidays <- holidays[, lapply(.SD, paste0, collapse="#"), by = c("ds", "country", "year"), .SDcols = "holiday"]
      
    }
    
    
    if (!is.null(factor_vars)) {
      dt_regressors <- cbind(recurrance, dt_transform[, c(context_vars, paid_media_vars), with =FALSE])
      modelRecurrance <- prophet(holidays = if(use_holiday) {holidays[country==InputCollect$prophet_country]} else {NULL}
                                 ,yearly.seasonality = use_season
                                 ,weekly.seasonality = use_weekday
                                 ,daily.seasonality= FALSE)
      # for (addreg in factor_vars) {
      #   modelRecurrance <- add_regressor(modelRecurrance, addreg)
      # }
      
      dt_ohe <- as.data.table(model.matrix(y ~., dt_regressors[, c("y",factor_vars), with =FALSE])[,-1])
      ohe_names <- names(dt_ohe)
      for (addreg in ohe_names) {
        modelRecurrance <- add_regressor(modelRecurrance, addreg)
      }
      dt_ohe <- cbind(dt_regressors[, !factor_vars, with=FALSE], dt_ohe)
      mod_ohe <- fit.prophet(modelRecurrance, dt_ohe)
      # prophet::regressor_coefficients(mxxx)
      dt_forecastRegressor <- predict(mod_ohe, dt_ohe)
      # prophet::prophet_plot_components(mod_ohe, dt_forecastRegressor)
      
      forecastRecurrance <- dt_forecastRegressor[, str_detect(names(dt_forecastRegressor), "_lower$|_upper$", negate = TRUE), with =FALSE]
      for (aggreg in factor_vars) {
        oheRegNames <- na.omit(str_extract(names(forecastRecurrance), paste0("^",aggreg, ".*")))
        forecastRecurrance[, (aggreg):=rowSums(.SD), .SDcols=oheRegNames]
        get_reg <- forecastRecurrance[, get(aggreg)]
        dt_transform[, (aggreg):= scale(get_reg, center = min(get_reg), scale = FALSE)]
        #dt_transform[, (aggreg):= get_reg]
      }
      # modelRecurrance <- fit.prophet(modelRecurrance, dt_regressors)
      # forecastRecurrance <- predict(modelRecurrance, dt_transform[, c("ds",context_vars, paid_media_vars), with =FALSE])
      # prophet_plot_components(modelRecurrance, forecastRecurrance)
      
    } else {
      modelRecurrance<- prophet(recurrance
                                ,holidays = if(use_holiday) {holidays[country==InputCollect$prophet_country]} else {NULL}
                                ,yearly.seasonality = use_season
                                ,weekly.seasonality = use_weekday
                                ,daily.seasonality= FALSE
                                #,changepoint.range = 0.8
                                #,seasonality.mode = 'multiplicative'
                                #,changepoint.prior.scale = 0.1
      )
      
      #futureDS <- make_future_dataframe(modelRecurrance, periods=1, freq = InputCollect$intervalType)
      forecastRecurrance <- predict(modelRecurrance, dt_transform[, "ds", with =FALSE])
      
    }
    
    # plot(modelRecurrance, forecastRecurrance)
    # prophet_plot_components(modelRecurrance, forecastRecurrance, render_plot = TRUE)
    
    if (use_trend) {
      fc_trend <- forecastRecurrance$trend[1:NROW(recurrance)]
      dt_transform[, trend := fc_trend]
      # recurrance[, trend := scale(fc_trend, center = min(fc_trend), scale = FALSE) + 1]
      # dt_transform[, trend := recurrance$trend]
    }
    if (use_season) {
      fc_season <- forecastRecurrance$yearly[1:NROW(recurrance)]
      dt_transform[, season := fc_season]
      # recurrance[, seasonal := scale(fc_season, center = min(fc_season), scale = FALSE) + 1]
      # dt_transform[, season := recurrance$seasonal]
    }
    if (use_weekday) {
      fc_weekday <- forecastRecurrance$weekly[1:NROW(recurrance)]
      dt_transform[, weekday := fc_weekday]
      # recurrance[, weekday := scale(fc_weekday, center = min(fc_weekday), scale = FALSE) + 1]
      # dt_transform[, weekday := recurrance$weekday]
    }
    if (use_holiday) {
      fc_holiday <- forecastRecurrance$holidays[1:NROW(recurrance)]
      dt_transform[, holiday := fc_holiday]
      # recurrance[, holidays := scale(fc_holiday, center = min(fc_holiday), scale = FALSE) + 1]
      # dt_transform[, holiday := recurrance$holidays]
    }
  }
  
  ################################################################
  #### Finalize input
  
  #dt <- dt[, all_name, with = FALSE]
  dt_transform <- dt_transform[, c("ds", "dep_var", all_ind_vars), with = FALSE]
  
  InputCollect$dt_mod <- dt_transform
  InputCollect$dt_modRollWind <- dt_transform[InputCollect$rollingWindowStartWhich:InputCollect$rollingWindowEndWhich]
  InputCollect$dt_inputRollWind <- dt_inputRollWind
  InputCollect$all_media <- all_media
  
  InputCollect[['modNLSCollect']] <- modNLSCollect
  InputCollect[['plotNLSCollect']] <- plotNLSCollect  
  InputCollect[['yhatNLSCollect']] <- yhatNLSCollect  
  InputCollect[['costSelector']] <- costSelector  
  InputCollect[['mediaCostFactor']] <- mediaCostFactor  
  
  return(InputCollect)
}

