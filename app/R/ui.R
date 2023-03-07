# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ui <- function() {
  fluidPage(
    shinyjs::useShinyjs(),
    tags$head(
      # this changes the size of the popovers
      # need to use hyphen not underscore
      tags$style(type = "text/css", HTML(".shiny-notification {
             position:fixed;
             top: calc(50%);
             left: calc(33%);
             font-size: 200%;
             }
             ", ".datepicker {z-index:99999 !important;}", ".navbar-default {
      background-color: DFDFD6 !important;}"))
    ),
    titlePanel(
      splitLayout(HTML('<img src="https://facebookexperimental.github.io/Robyn/img/robyn_logo.png" alt="Logo" width = 50 height = 40> Robyn Learn: Meta Open-Sourced MMM UI'),
        textOutput("version"),
        cellWidths = c("80%", "20%")
      ),
      windowTitle = "Robyn Learn"
    ),
    navbarPage(
      title = "",
      ##################################### Getting Started ##################################
      tabPanel(
        title = "Getting Started",
        fluidRow(
          column(
            align = "center",
            width = 12,
            headerPanel(
              h3("Welcome to Robyn Learn - An Educational Open-Source Marketing Mix Model Development UI where our goal is to promote education of Robyn, a tool aimed to democratize access to MMM and encouraging good marketing practices through data and science",
                style = "font-weight: 500; line-height: 1.1;
                  color:blue;"
              )
            )
          )
        ),
        br(),
        br(),
        fluidRow(
          column(
            width = 12,
            align = "center",
            headerPanel(
              h3(a("To discover more about Project Robyn, please click here visit our website and github repository", href = "https://facebookexperimental.github.io/Robyn/", target = "_blank"),
                style = "font-weight:300;line-height:1.1;color:black"
              )
            ),
            br(),
            br(),
            br(),
            headerPanel(h3("Please zoom out your browser if buttons begin to overlap")),
            br(),
            br(),
            br(),
            br(),
            actionButton("test_data", label = "Get started by Pre-Loading Sample Data", class = "btn-primary btn-lg", icon = icon("laptop-code"))
          )
        ),
      ),
      navbarMenu(
        title = "Create a New Model",
        ##################################### Data Input and Variable Assignment ##################################
        tabPanel(
          (title <- "Data Input and Variable Assignment"),
          sidebarLayout(
            sidebarPanel(
              fileInput("data_file",
                label = h4(
                  "Choose CSV File containing data, with column names in first row ",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("data_file_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                accept = ".csv"
              ),
              fileInput("holiday_file",
                label = h4(
                  "Choose CSV File containing holiday info",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("holiday_file_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                accept = ".csv"
              ),
              selectInput("dep_var", label = h4(
                "Input Dependent Variable Name",
                tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                actionButton("dep_var_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
              ), choices = "", selectize = F),
              selectInput("dep_var_type",
                label = h4(
                  "Dependent Variable Type",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("dep_var_type_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                choices = list("revenue", "conversion")
              ),
              selectInput("date_var", label = h4(
                "Input DATE variable name",
                tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                actionButton("date_var_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
              ), choices = "", selectize = F),
              textInput("date_format_var", label = h4(
                "Input DATE format", tags$style(type = "text/css", "#q2{vertical-align:top;}"),
                actionButton("date_format_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
              ), value = "%Y-%m-%d"),
              numericInput("num_media",
                label = h4(
                  "Number of Paid Media Channels to Include",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("paid_media_vars_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 1, value = 1
              ),
              numericInput("num_organic_media",
                label = h4(
                  "Number of Organic Media Channels to Include",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("organic_media_vars_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 0, value = 0
              ),
              numericInput("num_context",
                label = h4(
                  "Number of Contextual (Non Paid Media, Non Organic Media) Variables",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("context_vars_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 0, value = 0
              ),
              actionButton("init_var_input", label = "Initialize Media & Contextual Variable Inputs")
            ),
            mainPanel(
              dataTableOutput("data_tbl", width = "1000"),
              dataTableOutput("hol_tbl", width = "1000"),
              br(),
              br(),
              uiOutput("var_assignment_descipt"),
              br(),
              uiOutput("media_vars"),
              uiOutput("org_media_vars"),
              uiOutput("context_vars"),
              uiOutput("finalize_vars"),
              br(),
              uiOutput("vars_finalized")
            )
          )
        ),
        ##################################### Exploratory Data Analysis ##################################
        tabPanel(
          (title <- "Exploratory Data Analysis"),
          fluidRow(
            column(
              width = 12,
              actionButton("EDA_initiate", label = "Click to Initiate Exploratory Data Analysis incl. Auto Generated Recommendations"),
              headerPanel(
                h3("Recommendations Based on Exploratory Data Analysis",
                  style = "font-family: 'Lobster';
                        font-weight: 500; line-height: 1.1;
                      color: #990000;"
                )
              ),
              htmlOutput("print_message_1"),
              htmlOutput("print_message_2"),
              htmlOutput("print_message_3a"),
              htmlOutput("print_message_3b"),
              htmlOutput("print_message_4"),
              fluidRow(
                column(
                  12,
                  hr(style = "border-top: 1px solid #000000;"),
                  headerPanel(
                    h3("Refer to the charts below for more detail",
                      style = "font-family: 'Lobster';
                            font-weight: 500; line-height: 1.1;
                            color: #000080;"
                    )
                  ),
                  tabsetPanel(
                    tabPanel("Chart 1", plotOutput("ggplot1")),
                    tabPanel(
                      "Chart 2a-2d",
                      splitLayout(
                        cellWidths = c("50%", "50%"),
                        plotOutput("ggplot2a"),
                        plotOutput("ggplot2b")
                      ),
                      splitLayout(
                        cellWidths = c("50%", "50%"),
                        plotOutput("ggplot2c"),
                        plotOutput("ggplot2d")
                      )
                    ),
                    tabPanel("Chart 3a", plotOutput("ggplot3a")),
                    tabPanel("Chart 3b", plotOutput("ggplot3b")),
                    tabPanel("Chart 4", plotOutput("ggplot4")),
                    tabPanel(
                      "Chart 5",
                      fluidRow(
                        column(
                          width = 3,
                          selectInput("granularity",
                            label = h4(
                              "Data Granularity",
                              tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                              actionButton("granularity_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                            ),
                            choices = list("daily", "weekly")
                          ),
                        ),
                        column(
                          width = 3,
                          uiOutput("var_to_plot_input")
                        )
                      ),
                      fluidRow(
                        plotOutput("ggplot5")
                      )
                    )
                  )
                )
              )
            )
          )
        ),
        ##################################### Model Tuning ##################################
        tabPanel(
          title = "Model Tuning",
          sidebarLayout(
            sidebarPanel(
              width = 5,
              selectInput("adstock_selection",
                label = h4(
                  "Select Adstock Distribution",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("adstock_selection_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                choices = c("geometric", "weibull_cdf", "weibull_pdf"), selected = "geometric", selectize = F
              ),
              uiOutput("model_window_min"),
              uiOutput("model_window_max"),
              numericInput("set_iter",
                label = h4(
                  "Set Iterations per Trial",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("set_iter_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 1, value = 2000
              ),
              numericInput("set_trials",
                label = h4(
                  "Set Total Number of Trials",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("trial_count_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 1, value = 5
              ),
              uiOutput("dest_folder"),
              hr(style = "border-top: 1px solid #000000;"),
              checkboxInput("enable_calibration",
                label = h4(
                  "Enable Model Calibration with Experiments",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("calibration_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                value = FALSE
              ),
              uiOutput("calibration_file"),
              uiOutput("calib_file_date_format"),
              hr(style = "border-top: 1px solid #000000;"),
              checkboxInput("prophet_enable_checkbox",
                label = h4(
                  "Enable Prophet",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("prophet_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                value = TRUE
              ),
              splitLayout(
                uiOutput("prophet_country"),
                uiOutput("prophet_enable"),
                uiOutput("prophet_signs")
              )
            ),
            mainPanel(
              width = 7,
              splitLayout(
                cellWidths = c("40%", "30%", "30%"),
                actionButton("finalize_inputs_popover", label = "Info on Finalizing your Inputs",
                             style = "info", size = "small", width = "100%"),
                actionButton("finalize_hyperparams", label = "1. Finalize Inputs", width = "100%"),
                actionButton("run_model", label = "2. Initiate Modelling", width = "100%")
              ),
              textOutput("model_gen_text"),
              hr(style = "border-top: 1px solid #000000;"),
              actionButton("response_curve_popover", label = "Understanding Cost-Response & Adstock Curves", style = "info", size = "small"),
              splitLayout(
                plotOutput("adstock_curves_samples"),
                plotOutput("response_curves_samples")
              ),
              hr(style = "border-top: 1px solid #000000;"),
              actionButton("hyperparam_slider_popover", label = "Explanation of Default Ranges for Hyperparameters", style = "info", size = "small"),
              br(),
              br(),
              uiOutput("local_hyperparam_sliders_paid"),
              uiOutput("local_hyperparam_sliders_organic"),
              hr(style = "border-top: 1px solid #000000;"),
              dataTableOutput("lift_calib_tbl", width = "1000")
            )
          )
        ),
        ##################################### Model Selection ##################################
        tabPanel(
          (title <- "Model Selection"),
          sidebarLayout(
            sidebarPanel(
              actionButton("initiate_tables", label = "Initiate Tables and Charts", style = "info", size = "large"),
              actionButton("pareto_front_popover", label = "Interpreting your pareto-optimal solutions", style = "info", size = "small"),
              br(),
              br(),
              plotOutput("pParFront", click = "plot_click"),
              tableOutput("model_selection_info"),
              br(),
              br(),
              uiOutput("plots_folder"),
              textInput("plot", label = div("solID"), value = ""),
              actionButton("load_charts", label = "Load Charts"),
              br(),
              br(),
              actionButton("save_model", label = "IMPORTANT! CLICK HERE SAVE CURRENT solID", class = "btn-warning"),
              width = 4
            ),
            mainPanel(
              dataTableOutput("pareto_front_tbl", width = "1000"),
              br(),
              br(),
              uiOutput("model_output_expl_gen"),
              br(),
              br(),
              uiOutput("model_output_expl_1"),
              br(),
              br(),
              uiOutput("model_output_expl_2"),
              br(),
              br(),
              uiOutput("model_output_expl_3"),
              br(),
              br(),
              uiOutput("load_selection_plot")
            )
          )
        ),
        ##################################### Budget Optimization - Scenario Planner ##################################
        tabPanel(
          (title <- "Budget Optimization - Scenario Planner"),
          sidebarLayout(
            sidebarPanel(
              strong(sprintf(
                "NOTE: not working correctly with Robyn >= v3.10.%s Expect update by 2023-H2",
                ifelse(packageVersion("Robyn") >= "3.10",
                       paste(" Please, downgrade version like this:",
                             'remotes::install_github("facebookexperimental/Robyn/R", ref = "v3.9.0").'),
                       ""))),
              hr(),
              actionButton("run_opt", label = "Run Optimizer"),
              hr(),
              textInput("solID", label = h4(
                "Model solID",
                tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                actionButton("opt_solid_pop", label = "", icon = icon("question"), style = "info", size = "extra-small")
              ), value = NULL),
              selectInput("opt_scenario",
                label = h4(
                  "Optimization Scenario",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("opt_scen_pop", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                choices = list("max_historical_response", "max_response_expected_spend"), selectize = F
              ),
              uiOutput("expected_spend"),
              uiOutput("expected_days"),
              br(),
              br(),
              actionButton("opt_sliders", label = "Setting optimization boundaries", style = "info"),
              br(),
              br(),
              uiOutput("sliders")
            ),
            mainPanel(
              withSpinner(plotOutput("optimizer_plot", height = 800)),
              withSpinner(dataTableOutput("optimizer_tbl", width = "1000"))
            )
          )
        )
      ),
      navbarMenu(
        title = "Refresh or work with Existing Model",
        ##################################### Refresh Model ##################################
        tabPanel(
          title = "Refresh Model",
          sidebarLayout(
            sidebarPanel(
              actionButton("existingModel", label = "Refreshing your Existing Model", style = "info"),
              textInput("existing_model_for_refresh", label = h4(
                "Choose your previously Robyn object model path (should be json)",
                tags$style(type = "json", "#q2 {vertical-align: top;}"),
                actionButton("existing_model_for_refresh_popover",
                  label = "", icon = icon("question"),
                  style = "info", size = "extra-small"
                )
              ), value = ""),
              fileInput("data_file_refresh",
                label = h4(
                  "Choose CSV File containing data, with column names in first row ",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("data_file_popover_r", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                accept = ".csv"
              ),
              fileInput("holiday_file_refresh",
                label = h4(
                  "Choose CSV File containing holiday info",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("holiday_file_popover_e", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                accept = ".csv"
              ),
              numericInput("refresh_steps",
                label = h4(
                  "Input refresh time-steps",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("refresh_steps_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 1, value = 1
              ),
              selectInput("refresh_mode",
                choices = c("auto", "manual"), selected = "auto",
                label = h4(
                  "Choose the refresh mode",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("refresh_mode_e", label = "", icon = icon("question"), style = "info", size = "extra-small")
                )
              ),
              numericInput("refresh_iters",
                value = 1000, min = 1, max = NA, step = 1,
                label = h4(
                  "Set Iterations per Trial", tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("refresh_iters_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                )
              ),
              numericInput("refresh_trials",
                label = h4(
                  "Set Total Number of Trials",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("refresh_trials_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                step = 1, min = 1, value = 5
              ),
              textInput("dest_folder_refresh", label = h4(
                "Choose your sub-folder for refresh plots",
                actionButton("dest_folder_refresh_popover",
                  label = "", icon = icon("question"),
                  style = "info", size = "extra-small"
                )
              ), value = paste("refresh", sep = "")),
              actionButton("refresh_run", label = "Run Model Refresh")
            ),
            mainPanel(
              dataTableOutput("data_refresh_dto", width = "1000"),
              dataTableOutput("hol_refresh_dto", width = "1000"),
              br(),
              br(),
              textOutput("refresh_model_gen_text")
            )
          )
        ),

        ########################################### Refresh Model Selection ##################################
        tabPanel(
          (title <- "Refresh Model Selection"),
          sidebarLayout(
            sidebarPanel(
              actionButton("refresh_load_models", label = "Initiate Tables and Charts"),
              actionButton("refresh_pareto_front_popover", label = "Interpreting your pareto-optimal solutions", style = "info", size = "small"),
              br(),
              br(),
              plotOutput("refresh_pParFront", click = "refresh_plot_click"),
              tableOutput("refresh_model_selection_info"),
              br(),
              br(),
              uiOutput("refresh_plots_folder"),
              textInput("refresh_plot", label = div("refresh_solID"), value = ""),
              actionButton("load_refresh_charts", label = "Load Refresh Charts"),
              br(),
              br(),
              actionButton("save_refresh_model", label = "IMPORTANT! CLICK HERE SAVE CURRENT solID", class = "btn-warning"),
              width = 4
            ),
            mainPanel(
              dataTableOutput("ref_pareto_front_tbl", width = "1000"),
              br(),
              br(),
              uiOutput("ref_model_output_expl_gen"),
              br(),
              br(),
              uiOutput("ref_model_output_expl_1"),
              br(),
              br(),
              uiOutput("ref_model_output_expl_2"),
              br(),
              br(),
              uiOutput("ref_model_output_expl_3"),
              br(),
              br(),
              dataTableOutput("ref_model_summary_tbl", width = "1000"),
              br(),
              br(),
              uiOutput("ref_load_selection_plot")
            )
          )
        ),
        ################################## Refresh Model- Budget Optimization - Scenario Planner ##################################
        tabPanel(
          (title <- "Refresh Budget Optimization - Scenario Planner"),
          sidebarLayout(
            sidebarPanel(
              actionButton("run_refresh_opt", label = "Run Refresh Optimizer"),
              hr(),
              textInput("ref_solID", label = h4(
                "Refresh Model solID",
                tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                actionButton("refresh_opt_solid_pop", label = "", icon = icon("question"), style = "info", size = "extra-small")
              ), value = NULL),
              selectInput("refresh_opt_scenario",
                label = h4(
                  "Optimization Scenario",
                  tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
                  actionButton("refresh_opt_scen_pop", label = "", icon = icon("question"), style = "info", size = "extra-small")
                ),
                choices = list("max_historical_response", "max_response_expected_spend"), selectize = F
              ),
              uiOutput("ref_expected_spend"),
              uiOutput("ref_expected_days"),
              br(),
              br(),
              actionButton("re_opt_sliders", label = "Setting optimization boundaries", style = "info"),
              br(),
              br(),
              uiOutput("ref_sliders")
            ),
            mainPanel(
              withSpinner(plotOutput("ref_optimizer_plot", height = 800)),
              withSpinner(dataTableOutput("ref_optimizer_tbl", width = "1000"))
            )
          )
        )
      ),
    )
  )
}
