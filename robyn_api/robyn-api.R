
# Import necessary libraries
suppressPackageStartupMessages({
  library(arrow)
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library("reticulate")
  use_condaenv("r-reticulate")
  library(Robyn)
  library(plumber)
  library(jsonlite)
  library(tibble)
})



### FUNCTIONS ###

#* Convert hex data to raw bytes
#* This function is called to import the table data such as dt_simulated_weekly, dt_prophet_holidays
hex_to_raw <- function(x) {
  chars <- unlist(regmatches(x, gregexpr("..", x)))
  as.raw(strtoi(chars, base=16L))
}

#* Serialize a ggplot into a hex string by first converting to png
ggplot_serialize <- function(plot,dpi,width,height) {
  temp_file <- tempfile(fileext = ".png")
  ggsave(temp_file, plot, device = "png", dpi = dpi, width = width, height = height, limitsize = FALSE)
  png_data <- readBin(temp_file, "raw", file.info(temp_file)$size)
  hex_string <- paste0(sprintf("%02x", as.integer(png_data)), collapse = "")
  file.remove(temp_file)
  return(hex_string)
}

#* Whether an object is a named list
is_named_list <- function(obj) {
  is_list <- is.list(obj)
  has_names <- !is.null(names(obj))
  return(is_list && has_names)
}

#* Determine whether an object is a ggplot
is_ggplot <- function(obj) {
  inherits(obj, "ggplot")
}

#* Iterates recursively and helps to serialise any ggplot objects
recursive_ggplot_serialize <- function(obj,dpi=900,width=12,height=8) {
  for (key in names(obj)) {
    if (is_ggplot(obj[[key]])) {
      obj[[key]] <- ggplot_serialize(obj[[key]],dpi,width,height)
    }
    else if (is_named_list(obj[[key]])) {
      obj[[key]] <- recursive_ggplot_serialize(obj[[key]],dpi,width,height)
    }
  }
  return(obj)
}

#* Convert YYYY-MM-DD format string into date
convert_dates_to_Date <- function(json_data) {
  # Helper function to recursively traverse and convert date strings to Date objects
  recursive_convert <- function(x) {
    if (is.list(x)) {
      lapply(x, recursive_convert)
    } else if (is.character(x) && length(x) == 1 && grepl("^\\d{4}-\\d{2}-\\d{2}$", x)) {
      as.Date(x)
    } else {
      x
    }
  }
  
  # Recursively convert date strings to Date objects
  converted_data <- recursive_convert(json_data)
  
  return(converted_data)
}

### Robyn functions expect data/objects to be R unique one, but if bypassing data/obj via REST API, we need to convert these into R unique type like tibble or factor.
#* transform InputCollect from API
transform_InputCollect <- function(InputCollect) {

  InputCollect <- jsonlite::fromJSON(InputCollect) %>% convert_dates_to_Date()

  # list > tibble
  vars_to_tibble <- c("dt_input", "dt_holidays", "dt_mod", "dt_modRollWind", "dt_inputRollWind", "calibration_input")
  for (var in vars_to_tibble) {
    InputCollect[[var]] <- as_tibble(InputCollect[[var]])
    InputCollect[[var]][] <- lapply(InputCollect[[var]], function(col) {
      if (all(grepl("^\\d{4}-\\d{2}-\\d{2}$", col))) {
        return(as.Date(col))
        }
      return(col)
      })
    }

  # Null Treatment
  for (var in names(InputCollect)) {
    if(length(InputCollect[[var]])==0) {
      InputCollect[[var]] <- NULL
    }
  }

  return(InputCollect)
}

#* transform OutputCollect from API
transform_OutputCollect <- function(OutputCollect, select_model) {

  OutputCollect <- jsonlite::fromJSON(OutputCollect)

  # Add class name which is used as a checker in Robyn
  class(OutputCollect) <- c("robyn_outputs", "list")

  # convert only target model data
  OutputCollect[['allPareto']][['plotDataCollect']][[select_model]][['plot2data']][['plotWaterfallLoop']] <-
    OutputCollect[['allPareto']][['plotDataCollect']][[select_model]][['plot2data']][['plotWaterfallLoop']] %>%
    as_tibble() %>%
    mutate(across(where(is.character), as.factor))

  return(OutputCollect)
}

#* @apiTitle Robyn API
#* @apiDescription A set of API endpoints to work with the Robyn library.

#* Fetch demo data
#* @post /dt_simulated_weekly
function() {
  return(Robyn::dt_simulated_weekly)
}

#* Fetch holiday data
#* @post /dt_prophet_holidays
function() {
  return(Robyn::dt_prophet_holidays)
}

#* Input parameters, hyperparameters, and datasets for models and post back InputCollects
# * @param modelData Model data feather file in hex format
# * @param holidayData Holiday data feather file in hex format
# * @param jsonInputArgs Additional parameters for robyn_inputs() in json format
# * @param InputCollect
#* @post /robyn_inputs
function(modelData=FALSE, holidayData=FALSE, jsonInputArgs=FALSE, InputCollect=FALSE, calibration_input=FALSE) {
  
  # logic needs to be reviewed as it's MECE.
  if(!modelData==FALSE && !holidayData==FALSE && InputCollect==FALSE && calibration_input==FALSE){
    dt_input <- modelData %>% hex_to_raw() %>% arrow::read_feather()
    dt_holiday <- holidayData %>% hex_to_raw() %>% arrow::read_feather()
    argsInput <- jsonlite::fromJSON(jsonInputArgs)
    InputCollect <- do.call(robyn_inputs, c(list(dt_input = dt_input, dt_holidays = dt_holiday), argsInput))
  }
  else if(modelData==FALSE && holidayData==FALSE && !InputCollect==FALSE && calibration_input==FALSE){
    InputCollect <- transform_InputCollect(InputCollect)
    argsInput <- jsonlite::fromJSON(jsonInputArgs)
    InputCollect <- do.call(robyn_inputs, c(list(InputCollect = InputCollect), argsInput))
  }
  else if(modelData==FALSE && holidayData==FALSE && !InputCollect==FALSE && !calibration_input==FALSE){
      InputCollect <- transform_InputCollect(InputCollect)
      calibration_input <- calibration_input %>% hex_to_raw() %>% arrow::read_feather()
      InputCollect <- do.call(robyn_inputs, c(list(InputCollect = InputCollect, calibration_input = calibration_input)))
  }
  
  return(recursive_ggplot_serialize(InputCollect))

}

# Get error when using calibration
#* Run a model and post back output models
#* @param InputCollect
#* @param jsonRunArgs Additional parameters for robyn_run() in json format
#* @post /robyn_run
function(InputCollect, jsonRunArgs) {

  InputCollect <- transform_InputCollect(InputCollect)
  argsRun <- jsonlite::fromJSON(jsonRunArgs)

  OutputModels <- do.call(robyn_run, c(list(InputCollect = InputCollect), argsRun))

  return(recursive_ggplot_serialize(OutputModels))

}


# Error
# Failed exporting results, but returned model results anyways:
#   Error in robyn_write(InputCollect = InputCollect, OutputModels = OutputModels, : inherits(InputCollect, "robyn_inputs") is not TRUE

#* Run a model selection and post back output collect
#* @param InputCollect
#* @param OutputModels
#* @param jsonOutputsArgs Additional parameters for robyn_outputs() in json format
#* @post /robyn_outputs
function(InputCollect, OutputModels, jsonOutputsArgs, onePagers=FALSE) {

  InputCollect <- transform_InputCollect(InputCollect)
  OutputModels <- jsonlite::fromJSON(OutputModels)
  argsOutputs <- jsonlite::fromJSON(jsonOutputsArgs)

  OutputCollect <- do.call(robyn_outputs, c(list(InputCollect = InputCollect, 
                                                 OutputModels = OutputModels),
                                            argsOutputs))

  return(recursive_ggplot_serialize(OutputCollect))

}

# Memo
# Which should we return, ggplot_serialize or recursive_ggplot_serialize? = list obj which contains hex string or only hex string
#* Call back onepager
#* @param InputCollect
#* @param OutputCollect
#* @param jsonOnepagersArgs Additional parameters for robyn_onepagers() in json format
#* @param dpi
#* @param width
#* @param height
#* @post /robyn_onepagers
function(InputCollect, OutputCollect, jsonOnepagersArgs, dpi=dpi, width=width, height=height) {
  
  argsOonepagers <- jsonlite::fromJSON(jsonOnepagersArgs)
  InputCollect <- transform_InputCollect(InputCollect)
  OutputCollect <- transform_OutputCollect(OutputCollect, argsOonepagers[["select_model"]])
  
  onepager <- do.call(robyn_onepagers, c(list(InputCollect = InputCollect, 
                                              OutputCollect = OutputCollect),
                                         argsOonepagers))
  
  dpi <- dpi %>% as.numeric()
  width <- width %>% as.numeric()
  height <- height %>% as.numeric()
  
  return(ggplot_serialize(onepager[[argsOonepagers[["select_model"]]]], dpi=dpi, width=width, height=height))
  
  }


#* Call back allocator
# * @param InputCollect
# * @param OutputCollect
# * @param dpi
# * @param width
# * @param height
#* @post /robyn_allocator
function(InputCollect, OutputCollect, jsonAllocatorArgs, dpi=dpi, width=width, height=height) {

  argsAllocator <- jsonlite::fromJSON(jsonAllocatorArgs)
  InputCollect <- transform_InputCollect(InputCollect)
  OutputCollect <- transform_OutputCollect(OutputCollect, argsAllocator[["select_model"]])
  
  # AllocatorCollect <- do.call(robyn_allocator, c(list(InputCollect = InputCollect, OutputCollect = OutputCollect), argsAllocator))
  
  AllocatorCollect <- robyn_allocator(
    InputCollect = InputCollect,
    OutputCollect = OutputCollect,
    select_model = argsAllocator[["select_model"]],
    date_range = argsAllocator[["date_range"]],
    total_budget = argsAllocator[["total_budget"]],
    channel_constr_low = argsAllocator[["channel_constr_low"]],
    channel_constr_up = argsAllocator[["channel_constr_up"]],
    channel_constr_multiplier = argsAllocator[["channel_constr_multiplier"]],
    scenario = argsAllocator[["scenario"]],
    export = argsAllocator[["export"]]
  )

  dpi <- dpi %>% as.numeric()
  width <- width %>% as.numeric()
  height <- height %>% as.numeric()
  return(ggplot_serialize(AllocatorCollect$plots$plots, dpi=dpi, width=width, height=height))
}