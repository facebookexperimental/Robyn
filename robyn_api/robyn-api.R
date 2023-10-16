suppressPackageStartupMessages(library(arrow))
library(ggplot2)
library(patchwork)
library(Robyn)
library(plumber)
library(jsonlite)
library(tibble)

#* Convert hex data back to raw bytes
hex_to_raw <- function(x) {
  chars <- strsplit(x, "")[[1]]
  as.raw(strtoi(paste0(chars[c(TRUE, FALSE)], chars[c(FALSE, TRUE)]), base=16L))
}

#* Serialises a ggplot into a hex string by first converting to png
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
#* @param modelData Model data feather file in hex format
#* @param holidayData Holiday data feather file in hex format
#* @param jsonInput Additional parameters for robyninputs in json format
#* @post /robyn_inputs
function(modelData, holidayData, jsonInput) {

  dt_input_bytes <- hex_to_raw(modelData)
  dt_input <- arrow::read_feather(dt_input_bytes)

  dt_holiday_bytes <- hex_to_raw(holidayData)
  dt_holiday <- arrow::read_feather(dt_holiday_bytes)

  argsInp <- jsonlite::fromJSON(jsonInput)

  InputCollect <- robyn_inputs(dt_input = dt_input,
                               dt_holidays = dt_prophet_holidays,
                               json_file = argsInp
  )

  return(recursive_ggplot_serialize(InputCollect))

}

#* Run a model and post back output models
#* @param InputCollect
#* @param jsonRunArgs Additional parameters for robynrun in json format
#* @post /robyn_run
function(InputCollect, jsonRunArgs) {

  InputCollect <- jsonlite::fromJSON(InputCollect) %>% convert_dates_to_Date()
  # String > Date
  InputCollect <- convert_dates_to_Date(InputCollect)
  # list > tibble
  vars_to_tibble <- c("dt_input", "dt_holidays", "dt_mod", "dt_modRollWind", "dt_inputRollWind")
  for (var in vars_to_tibble) {
    InputCollect[[var]] <- as_tibble(InputCollect_fromAPI[[var]])
  }
  # Null Treatment
  for (var in names(InputCollect)) {
    if(length(InputCollect[[var]])==0) {
      InputCollect[[var]] <- NULL
    }
  }

  argsRun <- jsonlite::fromJSON(jsonRunArgs)

  OutputModels <- do.call(robyn_run, c(list(InputCollect = InputCollect), argsRun))

  return(recursive_ggplot_serialize(OutputModels))

}


#* Run a model selection and post back output collect
#* @param InputCollect
#* @param OutputModels
#* @param jsonRunArgs Additional parameters for robyn_output in json format
#* @post /robyn_outputs
function(InputCollect, OutputModels, jsonRunArgs, onePagers=FALSE) {

  InputCollect <- jsonlite::fromJSON(InputCollect) %>% convert_dates_to_Date()
  # String > Date
  InputCollect <- convert_dates_to_Date(InputCollect)
  # list > tibble
  vars_to_tibble <- c("dt_input", "dt_holidays", "dt_mod", "dt_modRollWind", "dt_inputRollWind")
  for (var in vars_to_tibble) {
    InputCollect[[var]] <- as_tibble(InputCollect_fromAPI[[var]])
  }
  # Null Treatment
  for (var in names(InputCollect)) {
    if(length(InputCollect[[var]])==0) {
      InputCollect[[var]] <- NULL
    }
  }

  OutputModels <- jsonlite::fromJSON(OutputModels)

  argsRun <- jsonlite::fromJSON(jsonRunArgs)

  OutputCollect <- do.call(robyn_outputs, c(list(InputCollect = InputCollect, OutputModels = OutputModels), argsRun))

  return(recursive_ggplot_serialize(OutputCollect))

}

#* Call back onepager
#* @param InputCollect
#* @param OutputCollect
#* @param select_model
#* @param dpi
#* @param width
#* @param height
#* @post /robyn_onepagers
function(InputCollect, OutputCollect, select_model, dpi=900, width=17, height=19) {

  InputCollect <- jsonlite::fromJSON(InputCollect) %>% convert_dates_to_Date()
  # String > Date
  InputCollect <- convert_dates_to_Date(InputCollect)
  # list > tibble
  vars_to_tibble <- c("dt_input", "dt_holidays", "dt_mod", "dt_modRollWind", "dt_inputRollWind")
  for (var in vars_to_tibble) {
    InputCollect[[var]] <- as_tibble(InputCollect_fromAPI[[var]])
  }
  # Null Treatment
  for (var in names(InputCollect)) {
    if(length(InputCollect[[var]])==0) {
      InputCollect[[var]] <- NULL
    }
  }

  OutputCollect <- jsonlite::fromJSON(OutputCollect)
  class(OutputCollect) <- c("robyn_outputs", "list")

  onepager <- robyn_onepagers(InputCollect, OutputCollect, select_model = select_model, export = FALSE)

  return(recursive_ggplot_serialize(onepager, dpi=900, width=17, height=19))

}

#* Create and post back a .zip of Robyn_YYYYMMDDHHMM_init/
#* @serializer contentType list(type="application/zip")
#* @get /download_outputs_folder
function(res) {
  # Call the function to create the ZIP file
  zip_file_path <- '/Users/yuyatanaka/Downloads/test.zip'

  # Read the ZIP file as binary
  zip_content <- readBin(zip_file_path, "raw", file.info(zip_file_path)$size)

  # Set the response type to application/zip
  res$setHeader("Content-Type", "application/zip")
  # Set the filename for the downloaded ZIP file
  res$setHeader("Content-Disposition", paste("attachment; filename=downloaded_file.zip"))

  # Return the ZIP content
  as.raw(zip_content)
}


# #* Run a model and post back output collect
# #* @param modelData Model data feather file in hex format
# #* @param jsonInput Additional parameters for robyninputs in json format
# #* @param jsonRunArgs Additional parameters for robynrun in json format
# #* @param onePagers Build the one pager files
# #* @post /robynrun
# function(modelData,jsonInput,jsonRunArgs,onePagers=FALSE) {
#
#   dt_input_bytes <- hex_to_raw(modelData)
#   dt_input <- arrow::read_feather(dt_input_bytes)
#   data("dt_prophet_holidays")
#
#   argsInp <- jsonlite::fromJSON(jsonInput)
#   argsRun <- jsonlite::fromJSON(jsonRunArgs)
#
#   InputCollect <- robyn_inputs(
#     dt_input = dt_input,
#     dt_holidays = dt_prophet_holidays,
#     json_file = argsInp
#   )
#
#   OutputModels <- do.call(robyn_run, c(list(InputCollect = InputCollect),argsRun))
#
#   OutputCollect <- robyn_outputs(InputCollect, OutputModels, export=FALSE)
#
#   if(onePagers){
#     one_pagers <- list()
#     for (select_model in OutputCollect$clusters$models$solID) {
#       one_pagers[[select_model]] <- recursive_ggplot_serialize(robyn_onepagers(InputCollect, OutputCollect, select_model = select_model, export = FALSE),dpi=900,width=17,height=19)
#     }
#     OutputCollect$clusters$models$onepagers <- one_pagers
#   }
#
#   return(recursive_ggplot_serialize(OutputCollect))
#
# }
