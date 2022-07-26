# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn App
#'
#' @md
#' @name RobynApp
#' @docType package
#' @importFrom corrr correlate focus
#' @import data.table
#' @importFrom dplyr any_of all_of arrange as_tibble bind_rows contains desc
#' distinct everything filter group_by lag left_join mutate n pull row_number
#' select slice starts_with summarize ungroup `%>%`
#' @importFrom DT dataTableOutput renderDataTable datatable
#' @importFrom gghighlight gghighlight
#' @import ggplot2
#' @importFrom ggcorrplot ggcorrplot
#' @importFrom lubridate floor_date
#' @importFrom patchwork plot_annotation
#' @import Robyn
#' @import scales
#' @importFrom shiny a actionButton br checkboxInput column dateInput div
#' fileInput fluidPage fluidRow h2 h3 h4 headerPanel hr HTML htmlOutput icon
#' isolate mainPanel modalDialog navbarMenu navbarPage nearPoints need numericInput
#' observe observeEvent onSessionEnded plotOutput modalButton
#' radioButtons reactive reactiveValues renderImage renderPlot renderTable renderText renderUI
#' req selectInput showModal shinyApp sidebarLayout sidebarPanel
#' sliderInput splitLayout stopApp tabPanel tableOutput tabsetPanel tags textInput
#' textOutput titlePanel uiOutput updateSelectInput validate verbatimTextOutput showNotification
#' @importFrom shinycssloaders withSpinner
#' @importFrom shinyjs useShinyjs html runjs
#' @importFrom stats coef cor end reorder
#' @importFrom stringr str_replace_all
#' @importFrom utils head packageDescription
#' @importFrom tidyr pivot_longer
#' @importFrom veccompare vector.print.with.and
"_PACKAGE"

# data.table column names used
dt_vars <- c(
  "DATE", "count", "count_max", "date_diff_in_days", "date_previous_row",
  "decomp.rssd", "depVarHat", "dep_var", "flag", "hyp_org", "hyp_paid",
  "mape", "media", "modalButton", "month_abb", "non_NA_count", "nrmse",
  "oinput_reactive", "organic_vars", "pct_diff_vs_count_max",
  "pct_of_non_missing_data", "pct_of_non_missing_data_cat",
  "pct_of_total_media_spend", "rn", "robyn_object_refresh", "rsq_train",
  "solID", "term", "total_media_spend", "total_spend", "updateNumericInput",
  "updateTextInput", "variable", "week_char", "xDecompMeanNon0Perc",
  "xDecompPerc", "..vars_inputted"
)

if (getRversion() >= "2.15.1") {
  globalVariables(c(".", dt_vars))
}
