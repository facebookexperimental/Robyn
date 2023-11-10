# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

### Import necessary libraries ###

# Function to locate and load required virtual environment used to install nevergrad
load_pythonenv <- function(env="r-reticulate"){
  tryCatch(
    {
      library("reticulate")
      if(reticulate::condaenv_exists(env)) {use_condaenv(env)}
      else if (reticulate::virtualenv_exists(env)) {use_virtualenv(env, required = TRUE)}
      else {message('Install nevergrad to proceed')}
    },
    error=function(e) {
      message('Install nevergrad to proceed')
    }
  )
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  load_pythonenv()
  library(jsonlite)
  library(plumber)
  library(Robyn)
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
  
  # Add class name which is used as a checker in Robyn
  class(InputCollect) <- c("robyn_inputs", "list")
  
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
transform_OutputCollect <- function(OutputCollect, select_model=FALSE) {
  
  OutputCollect <- jsonlite::fromJSON(OutputCollect)
  
  # Add class name which is used as a checker in Robyn
  class(OutputCollect) <- c("robyn_outputs", "list")
  
  # convert only target model data
  if (!select_model==FALSE) {
    OutputCollect[['allPareto']][['plotDataCollect']][[select_model]][['plot2data']][['plotWaterfallLoop']] <-
      OutputCollect[['allPareto']][['plotDataCollect']][[select_model]][['plot2data']][['plotWaterfallLoop']] %>%
      as_tibble() %>%
      mutate(across(where(is.character), as.factor))
  }
  
  return(OutputCollect)
}

#* @apiTitle Robyn API
#* @apiDescription A set of API endpoints to work with the Robyn library.

#* Retrieves the version number of the Robyn package
#* This endpoint returns the current version of the Robyn package installed in the R environment.
#* @get /robyn_version
function() {
  as.character(packageVersion("Robyn"))
}

#* Provides demo data for simulated weekly metrics
#* This endpoint returns a dataset of simulated weekly metrics included with the Robyn package for demonstration purposes.
#* @get /dt_simulated_weekly
function() {
  return(Robyn::dt_simulated_weekly)
}

#* Provides holiday data suitable for use with Robyn models
#* This endpoint returns a dataset of holidays that can be used within Robyn models to account for seasonal variations.
#* @get /dt_prophet_holidays
function() {
  return(Robyn::dt_prophet_holidays)
}

#* Receives model-related data and configurations, processes them, and returns an 'InputCollect' object
#* This endpoint handles the ingestion of model data and parameters, converting them into the appropriate format for the Robyn model.
#* @param dt_input A hexadecimal string representing the binary content of a model data feather file.
#* @param dt_holiday A hexadecimal string representing the binary content of a holiday data feather file.
#* @param jsonInputArgs A JSON string of additional parameters to be used with the 'robyn_inputs()' function.
#* @param InputCollect A JSON string representing the 'InputCollect' object created by the 'robyn_inputs()' function.
#* @param calibration_input A hexadecimal string representing the binary content of a calibration data feather file.
#* @post /robyn_inputs
function(dt_input=FALSE, dt_holiday=FALSE, jsonInputArgs=FALSE, InputCollect=FALSE, calibration_input=FALSE) {

  dt_input <- if (!dt_input==FALSE) hex_to_raw(dt_input) %>% arrow::read_feather() else NULL
  dt_holiday <- if (!dt_holiday==FALSE) hex_to_raw(dt_holiday) %>% arrow::read_feather() else NULL
  InputCollect <- if (!InputCollect==FALSE) transform_InputCollect(InputCollect) else NULL
  calibration_input <- if (!calibration_input==FALSE) hex_to_raw(calibration_input) %>% arrow::read_feather() else NULL
  argsInput <- if (!jsonInputArgs==FALSE) jsonlite::fromJSON(jsonInputArgs) else NULL
  
  InputCollect <- do.call(robyn_inputs, c(list(dt_input = dt_input,
                                               dt_holidays = dt_holiday,
                                               InputCollect = InputCollect,
                                               calibration_input = calibration_input
                                               ), argsInput))
  
  return(recursive_ggplot_serialize(InputCollect))
}

#* Executes a Robyn model with the provided inputs and returns serialized model outputs
#* This endpoint is responsible for executing the 'robyn_run()' function with the given model inputs and parameters,
#* then serializes the output for transmission over the API.
#* @param InputCollect A JSON string representing the 'InputCollect' object created by the 'robyn_inputs()' function.
#* @param jsonRunArgs A JSON string of additional parameters for the 'robyn_run()' function.
#* @post /robyn_run
function(InputCollect, jsonRunArgs) {
  
  InputCollect <- transform_InputCollect(InputCollect)
  argsRun <- jsonlite::fromJSON(jsonRunArgs)
  
  OutputModels <- do.call(robyn_run, c(list(InputCollect = InputCollect), argsRun))
  
  return(recursive_ggplot_serialize(OutputModels))
}

#* Executes model selection based on provided inputs and returns the serialized output collection
#* This endpoint takes the results from the 'robyn_run()' function and applies further model selection criteria 
#* as specified in the 'robyn_outputs()' function, then serializes the results for API transmission.
#* @param InputCollect A JSON string representing the model inputs prepared by 'robyn_inputs()'.
#* @param OutputModels A JSON string representing the model outputs generated by 'robyn_run()'.
#* @param jsonOutputsArgs A JSON string containing additional parameters for the 'robyn_outputs()' function.
#* @param onePagers A boolean flag indicating whether to generate one-pager reports; defaults to FALSE.
#* @post /robyn_outputs
function(InputCollect, OutputModels, jsonOutputsArgs) {
  
  InputCollect <- transform_InputCollect(InputCollect)
  OutputModels <- jsonlite::fromJSON(OutputModels)
  argsOutputs <- jsonlite::fromJSON(jsonOutputsArgs)
  
  OutputCollect <- do.call(robyn_outputs, c(list(InputCollect = InputCollect, 
                                                 OutputModels = OutputModels
                                                 ), argsOutputs))
  
  return(recursive_ggplot_serialize(OutputCollect))
}

#* Generates a model one-pager and returns a serialized image
#* This endpoint invokes the 'robyn_onepagers()' function to create a visual summary of the model results,
#* which is then serialized into an image format based on the specified resolution and dimensions.
#* @param InputCollect A JSON string representing the model inputs prepared by 'robyn_inputs()'.
#* @param OutputCollect A JSON string representing the model output collection generated by 'robyn_outputs()'.
#* @param jsonOnepagersArgs A JSON string containing additional parameters for the 'robyn_onepagers()' function.
#* @param dpi The resolution of the image to be returned, specified as dots per inch.
#* @param width The width of the image to be returned, specified in inches.
#* @param height The height of the image to be returned, specified in inches.
#* @post /robyn_onepagers
function(InputCollect, OutputCollect, jsonOnepagersArgs, dpi=100, width=12, height=8) {
  
  argsOonepagers <- jsonlite::fromJSON(jsonOnepagersArgs)
  InputCollect <- transform_InputCollect(InputCollect)
  OutputCollect <- transform_OutputCollect(OutputCollect, argsOonepagers[["select_model"]])
  
  onepager <- do.call(robyn_onepagers, c(list(InputCollect = InputCollect, 
                                              OutputCollect = OutputCollect
                                              ), argsOonepagers))
  
  dpi <- as.numeric(dpi)
  width <- as.numeric(width)
  height <- as.numeric(height)
  
  return(ggplot_serialize(onepager[[argsOonepagers[["select_model"]]]], dpi=dpi, width=width, height=height))
}

#* Generates and returns a serialized image of the allocation one-pager
#* This endpoint facilitates the creation of allocation plots using the 'robyn_allocator()' function. The resulting plots
#* are serialized into an image format as specified by the resolution and dimensions provided in the request.
#* @param InputCollect A JSON string representing the model inputs prepared by 'robyn_inputs()'.
#* @param OutputCollect A JSON string representing the model output collection generated by 'robyn_outputs()'.
#* @param jsonAllocatorArgs A JSON string containing additional parameters for the 'robyn_allocator()' function.
#* @param dpi The resolution of the image to be returned, specified as dots per inch.
#* @param width The width of the image to be returned, specified in inches.
#* @param height The height of the image to be returned, specified in inches.
#* @post /robyn_allocator
function(InputCollect, OutputCollect, jsonAllocatorArgs, dpi=100, width=12, height=8) {
  
  argsAllocator <- jsonlite::fromJSON(jsonAllocatorArgs)
  InputCollect <- transform_InputCollect(InputCollect)
  OutputCollect <- transform_OutputCollect(OutputCollect, argsAllocator[["select_model"]])
  
  AllocatorCollect <- do.call(robyn_allocator, c(list(InputCollect = InputCollect, OutputCollect = OutputCollect), argsAllocator))
  
  dpi <- as.numeric(dpi)
  width <- as.numeric(width)
  height <- as.numeric(height)
  
  return(ggplot_serialize(AllocatorCollect$plots$plots, dpi=dpi, width=width, height=height))
}

#* Exports model data in JSON format
#* This endpoint uses the 'robyn_write()' function to output model data, including inputs and results, as a JSON object.
#* @param InputCollect A JSON string representing the model inputs prepared by 'robyn_inputs()'.
#* @param OutputCollect A JSON string representing the model outputs generated by 'robyn_outputs()'.
#* @param OutputModels A JSON string representing the models created by 'robyn_run()'.
#* @param jsonWriteArgs A JSON string containing additional parameters for the 'robyn_write()' function.
#* @post /robyn_write
function(InputCollect=FALSE, OutputCollect=FALSE, OutputModels=FALSE, jsonWriteArgs) {
  
  writeArgs <- jsonlite::fromJSON(jsonWriteArgs)
  InputCollect <- if (!InputCollect==FALSE) transform_InputCollect(InputCollect) else NULL
  OutputModels <- if (!OutputModels==FALSE) jsonlite::fromJSON(OutputModels) else NULL
  OutputCollect <- if (!OutputCollect==FALSE) transform_OutputCollect(OutputCollect) else NULL
  
  do.call(robyn_write, c(list(InputCollect = InputCollect, OutputCollect = OutputCollect, OutputModels = OutputModels), writeArgs))
}

#* Recreates a model from data files and additional parameters
#* This endpoint reads model data and holiday data from hexadecimal-encoded feather files and additional parameters from a JSON object to recreate a Robyn model.
#* @param dt_input A hexadecimal string of the model data feather file.
#* @param dt_holidays A hexadecimal string of the holiday data feather file.
#* @param jsonRecreateArgs A JSON string containing additional parameters for the 'robyn_recreate()' function.
#* @post /robyn_recreate
function(dt_input, dt_holidays, jsonRecreateArgs) {
  
  recreateArgs <- jsonlite::fromJSON(jsonRecreateArgs)
  dt_input <- dt_input %>% hex_to_raw() %>% arrow::read_feather()
  dt_holidays <- dt_holidays %>% hex_to_raw() %>% arrow::read_feather()
  
  RobynRecreated <- do.call(robyn_recreate, c(list(dt_input = dt_input, dt_holidays = dt_holidays), recreateArgs))
  
  return(recursive_ggplot_serialize(RobynRecreated)) 
}

#* Retrieves the names of hyperparameters based on adstock and media spend data
#* This endpoint calls the 'hyper_names()' function, passing the adstock and a list of paid media spends to get the corresponding hyperparameter names.
#* @param adstock A string representing the name of the adstock parameter.
#* @param all_media A JSON string representing the list of paid media spends.
#* @post /hyper_names
function(adstock, all_media) {
  
  hyper_names_list <- hyper_names(adstock = adstock, all_media = jsonlite::fromJSON(all_media))
  
  return(hyper_names_list)
}