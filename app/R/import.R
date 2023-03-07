# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn Learn
#'
#' @md
#' @name RobynLearn
#' @docType package
#' @importFrom corrr correlate focus
#' @importFrom dplyr across any_of all_of arrange as_tibble bind_rows contains count desc
#' distinct everything filter group_by lag left_join mutate mutate_at n pull row_number
#' select slice starts_with summarize ungroup `%>%`
#' @importFrom DT dataTableOutput renderDataTable datatable
#' @import ggplot2
#' @importFrom lubridate floor_date year month week yday
#' @importFrom patchwork plot_annotation
#' @import Robyn
#' @importFrom shiny a actionButton br checkboxInput column dateInput div
#' fileInput fluidPage fluidRow h2 h3 h4 headerPanel hr HTML htmlOutput icon
#' isolate mainPanel modalDialog navbarMenu navbarPage nearPoints need numericInput
#' observe observeEvent onSessionEnded plotOutput modalButton
#' radioButtons reactive reactiveValues renderImage renderPlot renderTable renderText renderUI
#' req selectInput showModal shinyApp sidebarLayout sidebarPanel
#' sliderInput splitLayout stopApp tabPanel tableOutput tabsetPanel tags textInput
#' textOutput titlePanel uiOutput updateSelectInput updateTextInput updateNumericInput
#' validate verbatimTextOutput showNotification strong
#' @importFrom shinycssloaders withSpinner
#' @importFrom shinyjs useShinyjs html runjs
#' @importFrom stats coef cor end reorder
#' @importFrom utils head packageDescription packageVersion read.csv
#' @importFrom tidyr pivot_longer
"_PACKAGE"
