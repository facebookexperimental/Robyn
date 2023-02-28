# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

server <- function(input, output, session) {
  session$onSessionEnded(function() {
    stopApp()
  })
  Sys.setenv(R_FUTURE_FORK_ENABLE = "true")
  options(future.fork.enable = TRUE)

  ############################# Start/Data Input tab server functionality #######################################

  input_reactive <- reactiveValues()
  mmm_data <- NULL
  input_reactive$version <- version <- paste(ifelse(
    is.null(packageDescription("Robyn")$Repository), "dev", "stable"),
    packageDescription("Robyn")$Version, sep = "-")
  message("ROBYN VERSION: ", version)

  output$version <- renderText(input_reactive$version)

  observeEvent(input$test_data, {
    message(">>> Setting test data...")
    input_reactive$tbl <- Robyn::dt_simulated_weekly
    input_reactive$holiday_data <- Robyn::dt_prophet_holidays
    input_reactive$paid_media_vars <- c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P")
    input_reactive$paid_media_signs <- rep("positive", length(input_reactive$paid_media_vars))
    input_reactive$paid_media_spends <- c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S")
    input_reactive$organic_vars <- "newsletter"
    input_reactive$organic_signs <- "positive"
    input_reactive$context_vars <- "competitor_sales_B"
    input_reactive$context_signs <- "default"
    input_reactive$baseline_var_names_factor_bool_list <- NULL
    input_reactive$factor_vars <- NULL
    input_reactive$date_var <- "DATE"
    input_reactive$date_format_var <- "%Y-%m-%d"
    input_reactive$dt_input <- input_reactive$tbl
    input_reactive$dt_input$DATE <- as.Date(input_reactive$dt_input$DATE, input_reactive$date_format_var)
    input_reactive$dt_holidays <- isolate(input_reactive$holiday_data)
    input_reactive$dep_var <- "revenue"
    input_reactive$dep_var_type <- "revenue"

    updateSelectInput(session, "dep_var", choices = input_reactive$dep_var, selected = input_reactive$dep_var)
    updateSelectInput(session, "dep_var_type", choices = input_reactive$dep_var_type, selected = input_reactive$dep_var_type)
    updateSelectInput(session, "date_var", choices = input_reactive$date_var, selected = input_reactive$date_var)
    updateTextInput(session, "date_format_var", value = input_reactive$date_format_var)
    updateNumericInput(session, "num_media", value = length(input_reactive$paid_media_vars))
    updateNumericInput(session, "num_organic_media", value = length(input_reactive$organic_vars))
    updateNumericInput(session, "num_context", value = length(input_reactive$context_vars))

    message("Automatic fields filled: ", paste(names(input_reactive), collapse = ", "))
    print(head(input_reactive$dt_input))
    msg <- "Sample data loaded, proceed to create a new model tabs"
    showModal(modalDialog(title = msg, easyClose = TRUE, footer = NULL))
    message(msg)
  })

  output$data_tbl <- renderDataTable({
    message(">>> Setting input data...")
    if (input$test_data < 1) {
      file <- input$data_file
      ext <- tools::file_ext(file$datapath)
      validate(need(ext == "csv", "Please upload a csv file"))
      mmm_data <- read.csv(file$datapath)
      input_reactive$tbl <- read.csv(file$datapath)
      input_reactive$data_path <- file$datapath
      datatable(head(mmm_data, n = 5L),
        options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t")
      )
    } else {
      datatable(head(input_reactive$tbl),
        options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t")
      )
    }
  })

  colnames_reactive <- reactive({
    if (!is.null(input$data_file)) {
      file <- input$data_file
      ext <- tools::file_ext(file$datapath)
      validate(need(ext == "csv", "Please upload a csv file"))
      my_data <- read.csv(file$datapath)
    }
  })

  observe({
    updateSelectInput(session, "dep_var", choices = names(colnames_reactive()))
    updateSelectInput(session, "date_var", choices = names(colnames_reactive()))
  })

  output$hol_tbl <- renderDataTable({
    if (input$test_data < 1) {
      hol_file <- input$holiday_file
      hol_ext <- tools::file_ext(hol_file$datapath)
      validate(need(hol_ext == "csv", "Please upload a csv file"))
      holiday_data <- read.csv(hol_file$datapath)
      input_reactive$holiday_data <- read.csv(hol_file$datapath)
      datatable(head(holiday_data, n = 5L), options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t"))
    } else {
      datatable(head(input_reactive$holiday_data, n = 5L), options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t"))
    }
  })

  observeEvent(input$data_file_popover, {
    showModal(modalDialog(
      title = "Choosing a CSV File for your main dataset",
      HTML(paste0(
        "Upload your dataset containing holiday data. If you do not have a separate file",
        ", in the github repository there is a file called holidays.csv that you can access ",
        "here containing holiday data going back and forward many years. Click ",
        a("here.", href = "https://github.com/facebookexperimental/Robyn/blob/main/R/data/dt_simulated_weekly.RData", target = "_blank"),
        " If you do have your own holiday file, ensure the formatting is the same as this file."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$holiday_file_popover, {
    showModal(modalDialog(
      title = "Choosing a CSV File for your holiday dataset",
      HTML(paste0(
        "Upload your dataset here. The file must be of .csv type, and should contain at least a column for a date variable, ",
        "and an independent variable such as revenue or conversions. For an example of what this file could look like, see the de_simulated_data.Rdata",
        "file in the ",
        a("github repository. ", href = "https://github.com/facebookexperimental/Robyn/blob/main/R/data/dt_prophet_holidays.RData", target = "_blank"),
        "Additionally, for more detailed information on this step please refer to the ? pop-ups on this tab. Please pre-load the sample data on the Getting Started tab if you want to explore the tool with that."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$dep_var_popover, {
    showModal(modalDialog(
      title = "Dependent Variable Column Name",
      HTML("Input the column name of your dependent variable here. It must correspond to conversion or revenue"),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$dep_var_type_popover, {
    showModal(modalDialog(
      title = "Dependent Variable Type",
      HTML("Make a selection on the type of your dependent variable between conversion or revenue"),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$date_var_popover, {
    showModal(modalDialog(
      title = "Date Variable Name Input",
      HTML(paste0(
        'Input name of your date variable. <b>Note - Date variable must be in format "YYYY-mm-dd"</b>',
        ". Typically this will be either daily or weekly data. Consider testing both, but having more granular daily data may help the model fit."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$date_format_popover, {
    showModal(modalDialog(
      title = "Date Variable Format Input",
      HTML(paste("To ensure that the date variable formatting is ingested correctly we must specify the formatting. In practice, this will look something like %Y-%m-%d, which corresponds to YYYY-mm-dd or a date like 2021-1-30. If your data is formatted like month/day/year, then your format will be %m/%d/%Y an example of which would be 12/27/2020 etc.",
        "Since we are dealing with daily data at the smallest here, our formatting will use the letters lowercase m (month), lowercase d (day), and uppercase Y (Year) to specify the format. If you are having trouble specifying the correct format, it may be easier to reformat your data to the default format of %Y-%m-%d.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$paid_media_vars_popover, {
    showModal(modalDialog(
      title = "Paid Media Variables",
      HTML(paste("For each Paid Media Channel you plan to measure, you will need to specify which variable names correspond to which channels. ",
        "First, Identify the total number of Paid Media Variables and input that into the Text Box under Number of Paid Media Variables. ",
        "Next, ensure that where possible, you have both a spend variable and a variable that is more closely tied to action. ",
        "For example, let us use Facebook as an example and use Facebook Impressions and Facebook Spend as our variables corresponding to that channel. ",
        "After we hit the Initialize Media & Baseline Variable Inputs button, a number of input fields will appear prompting you to input in the left box the action variable e.g. Impressions or Clicks, and in the right box the spend variable for the same channel",
        "The reason for needing both variables is due to the complex relationship between spend and a variable like impressions. ",
        "In many cases, these variables do not have a 1-1 relationship, and instead the relationship is better fit by a curve than by a straight line. ",
        "Media activity: Data collected for media ideally should reflect how many eyeballs have seen or been exposed to the media (e.g. impressions, GRPs). Spends should also be collected in order to calculate Return On Investment, however it is best practice to use exposure metrics as direct inputs into the model, as this is a better representation than spends of how media activity has been consumed by consumers. For example, 1 dollar spent on TV might yield a different reach than 1 dollar spent on Facebook.",
        "For digital activity, the most commonly used metrics are impressions. Avoid using clicks, as clicks do not account for view through conversions, and it is just as likely that someone can view an ad and convert.",
        "For TV and radio, the most commonly used metrics are Gross Rating Points (GRPs) or Target Audience Rating Points (TARPs).",
        "For print (e.g. newspapers or magazines), the most commonly used metrics are readership.",
        "As mentioned above, aim to collect data that reflects eyeballs or impressions for all other channels.",
        paste0("For more information about this step, see the ", a("step-by-step guide.", href = "https://facebookexperimental.github.io/Robyn/docs/step-by-step-guide/#set_mediavarname-and-set_mediaspendname")),
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$organic_media_vars_popover, {
    showModal(modalDialog(
      title = "Organic Media Variables",
      HTML("In order to more accurately capture the impact of Organic Media variables, we treat them similar to paid media variables by giving them adstocks and saturation curves. This ensures that we capture any latent effect of organic media."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$context_vars_popover, {
    showModal(modalDialog(
      title = "Contextual Variables",
      HTML(paste("Proper representation of Contextual Variables (i.e. your variables that are not Paid/Organic Media Variables) can have a substantial impact on an MMMs ability to fit the data, as well as explain the trends a business sees.",
        "Since there are really an infinite number of different variables you could test to see if they have an impact on your dependent variable, it can help to prioritize some variables that are most commonly used in MMM.",
        "Below we will highlight a few areas where many data scientists may include variables from in their models. This is not an exhaustive list, but should be a decent starting point. Remember that any variable that could have an impact on business performance could be important in helping increase the fit of the model, no matter what it is.",
        "<b>1) Advertiser-specific data</b> - For example - pricing changes, dates of promotions/sales, product launches, changes in shipping availability, etc.",
        "<b>2) Competitor info</b> - For example, Google Search Trends data looking at your competitors search volume index over time, or other competitor behavior that leads to a significant change in the market.",
        "<b>3) Macroeconomic Trends</b> - Unemployment Rates, Consumer Confidence, other data can help explain consumer investment trends in certain categories of goods",
        paste0("<b>4) Unique Time Periods</b> - Many advertisers were impacted in differing ways by unpredictable events like the Covid-19 Pandemic. ", a("Click here for a Facebook IQ article", href = "https://www.facebook.com/business/news/insights/5-ways-to-adjust-marketing-mix-models-for-unexpected-events", target = "_blank"), " highlighting some ways that advertisers can build models that are accounting for unprecented times."),
        "<b>5) Geographic specific data</b> - If your business is affected differently in different regions, for example by something like weather that can be an important factor to include.",
        "Oftentimes, an advertiser may see a model that fits well overall, but very poorly in specific circumstances. Improving the coverage of the models contextual variables may help address some of those issues when they arise.",
        "As with the paid media variables, you will need to enter the total number of contextual variables you would like to include in the model into the  Number of Contextual Variables box below. After you have the number of variables entered and press the Initialize button, you will need to enter the column names that correspond to each of your Contextual variables as well as the sign that you would like the model to adopt for that variables and tell the tool whether that variable is a factor (i.e. categorical, indicator) or not.",
        "For some variables (e.g. competitor trends) you may know for certain that the variable will have a negative affect on your sales as it increases so you can set the sign to negative. For other variables the effect may be clearly positive, or if unsure, simply leave the sign setting as Default.",
        "<b> An important rule of thumb to answer the question - How much data do we need? Can be answered by roughly using 1:10 variables to observations. In other words, if you have 10 independent variables then you should have at least 100 observations. Remember that the variables that are generated via Prophet for seasonality, trend, etc. count towards this. More on Prophet later. This showcases again how more fine grain data has advantages in terms of ability to detect effect sizes for more variables or with less data.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$init_var_input, {
    message(">>> Started variables assignment...")
    output$var_assignment_descipt <- renderUI({
      fluidRow(actionButton("var_assignment_button", label = "Reminder on Variable Assignment", size = "small"))
    })

    output$media_vars <- renderUI({
      if (input$test_data < 1) {
        if (isolate(input$num_media) >= 1) {
          lapply(1:isolate(input$num_media), function(i) {
            fluidRow(column(
              width = 8,
              splitLayout(
                selectInput(paste0("media_var_impr_", toString(i)), label = paste0("Paid Media Variable Action (Imp/Click) Column #", toString(i)), choices = colnames(input_reactive$tbl), selectize = FALSE),
                selectInput(paste0("media_var_spend_", toString(i)), label = , paste0("Paid Media Spend Column #", toString(i)), choices = colnames(input_reactive$tbl), selectize = FALSE)
              ),
              br(),
              br()
            ))
          })
        }
      } else {
        lapply(seq_along(isolate(input_reactive$paid_media_vars)), function(i) {
          fluidRow(column(
            width = 8,
            splitLayout(
              selectInput(paste0("media_var_impr_", toString(i)), label = paste0("Paid Media Variable Action (Imp/Click) Column #", toString(i)), choices = input_reactive$paid_media_vars[i], selected = input_reactive$paid_media_vars[i], selectize = FALSE),
              selectInput(paste0("media_var_spend_", toString(i)), label = , paste0("Paid Media Spend Column #", toString(i)), choices = input_reactive$paid_media_spends[i], selected = input_reactive$paid_media_spends[i], selectize = FALSE)
            ),
            br(),
            br()
          ))
        })
      }
    })

    output$org_media_vars <- renderUI({
      if (input$test_data < 1) {
        if (isolate(input$num_organic_media) >= 1) {
          lapply(1:isolate(input$num_organic_media), function(j) {
            fluidRow(column(
              width = 8,
              selectInput(paste0("org_media_var_impr_", toString(j)), label = paste0("Organic Media Variable Action (Imp/Click) Column #", toString(j)), choices = colnames(input_reactive$tbl), selectize = FALSE),
              br(),
              br()
            ))
          })
        }
      } else {
        lapply(1:isolate(input$num_organic_media), function(j) {
          fluidRow(column(
            width = 8,
            selectInput(paste0("org_media_var_impr_", toString(j)), label = paste0("Organic Media Variable Action (Imp/Click) Column #", toString(j)), choices = input_reactive$organic_vars[j], selected = input_reactive$organic_vars[j], selectize = FALSE),
            br(),
            br()
          ))
        })
      }
    })

    output$context_vars <- renderUI({
      if (input$test_data < 1) {
        if (isolate(input$num_context) >= 1) {
          lapply(1:isolate(input$num_context), function(k) {
            splitLayout(
              selectInput(paste0("baseline_var_name_", toString(k)), label = paste0("Column Name for Contextual Variable #", toString(k)), choices = colnames(input_reactive$tbl), selectize = FALSE),
              radioButtons(paste0("baseline_var_name_sign_", toString(k)), label = div(paste0("Force positive/negative sign for Contextual Variable #", toString(k)), style = "font-size:12px;"), choices = c("default", "positive", "negative"), inline = TRUE),
              checkboxInput(paste0("baseline_var_name_checkbox_", toString(k)), label = div(paste0("Check If Contextual Variable #", toString(k), " is a factor/categorical/indicator."), style = "font-size:12px;"), value = FALSE)
            )
          })
        }
      } else {
        lapply(1:isolate(input$num_context), function(k) {
          splitLayout(
            selectInput(paste0("baseline_var_name_", toString(k)), label = paste0("Column Name for Contextual Variable #", toString(k)), choices = input_reactive$context_vars[k], selected = input_reactive$context_vars[k], selectize = FALSE),
            radioButtons(paste0("baseline_var_name_sign_", toString(k)), label = div(paste0("Force positive/negative sign for Contextual Variable #", toString(k)), style = "font-size:12px;"), choices = "default", selected = "default", inline = TRUE),
            checkboxInput(paste0("baseline_var_name_checkbox_", toString(k)), label = div(paste0("Check If Contextual Variable #", toString(k), " is a factor/categorical/indicator."), style = "font-size:12px;"), value = FALSE)
          )
        })
      }
    })

    output$finalize_vars <- renderUI({
      actionButton("finalize_var_input", label = h4(
        "Finalize Variable Assignment",
        tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
        actionButton("fin_var_assign_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
      ))
    })
  })

  observeEvent(input$fin_var_assign_popover, {
    showModal(modalDialog(
      title = "Finalize Variable Assignment",
      HTML(paste("When you are confident that you have all of your variables and data input and set up correctly, click here to finalize that. ",
        "If there is a mistake, the tool will show an error message and you will need to ensure that all column fields input into boxes are spelled correctly and contained in the data set, and all selections in the left panel are completed.",
        sep = "<br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$var_assignment_button, {
    showModal(modalDialog(
      title = "Variable Assignment Reminder",
      HTML("Remember, if you do not have a impression/click variable for a given media channel input the spend variable associated with that channel into <b>BOTH</b> fields for that channel."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$finalize_var_input, {
    if (input$test_data < 1) {
      input_reactive$paid_media_vars <- NULL
      input_reactive$paid_media_signs <- NULL
      input_reactive$paid_media_spends <- NULL
      input_reactive$organic_vars <- NULL
      input_reactive$organic_signs <- NULL
      input_reactive$context_vars <- NULL
      input_reactive$context_signs <- NULL
      input_reactive$baseline_var_names_factor_bool_list <- NULL
      input_reactive$factor_vars <- NULL
      x <- NULL

      if (isolate(input$num_media) >= 1) {
        message(">>> Processing media")
        lapply(1:isolate(input$num_media), function(m) {
          x <<- input[[paste0("media_var_impr_", toString(m))]]
          if (isolate(input[[paste0("media_var_impr_", toString(m))]]) != "") {
            input_reactive$paid_media_vars <- c(input_reactive$paid_media_vars, isolate(input[[paste0("media_var_impr_", toString(m))]]))
            input_reactive$paid_media_signs <- c(input_reactive$paid_media_signs, "positive")
          }
          if (isolate(input[[paste0("media_var_spend_", toString(m))]]) != "") {
            input_reactive$paid_media_spends <- c(input_reactive$paid_media_spends, isolate(input[[paste0("media_var_spend_", toString(m))]]))
          }
        })
      }
      if (isolate(input$num_context) >= 1) {
        message(">>> Processing context")
        lapply(1:isolate(input$num_context), function(r) {
          input_reactive$context_vars <- c(input_reactive$context_vars, isolate(input[[paste0("baseline_var_name_", toString(r))]]))
          input_reactive$context_signs <- c(input_reactive$context_signs, isolate(input[[paste0("baseline_var_name_sign_", toString(r))]]))
        })
        lapply(1:isolate(input$num_context), function(r) {
          input_reactive$baseline_var_names_factor_bool_list <- c(input_reactive$baseline_var_names_factor_bool_list, isolate(input[[paste0("baseline_var_name_checkbox_", toString(r))]]))
        })
        input_reactive$factor_vars <- input_reactive$context_vars[which(input_reactive$baseline_var_names_factor_bool_list == TRUE)]
      }
      if (isolate(input$num_organic_media) >= 1) {
        message(">>> Processing organic_media")
        lapply(1:isolate(input$num_organic_media), function(x) {
          if (isolate(input[[paste0("org_media_var_impr_", toString(x))]]) != "") {
            input_reactive$organic_vars <- c(input_reactive$organic_vars, isolate(input[[paste0("org_media_var_impr_", toString(x))]]))
            input_reactive$organic_signs <- c(input_reactive$organic_signs, "positive") # need to fix
          }
        })
      }

      input_reactive$med_vars_impr_in_cols <- ifelse(length(intersect(input_reactive$paid_media_vars, colnames(input_reactive$tbl))) ==
        length(unique(input_reactive$paid_media_vars)), TRUE, FALSE)
      input_reactive$med_vars_spend_in_cols <- ifelse(length(intersect(input_reactive$paid_media_spends, colnames(input_reactive$tbl))) ==
        length(unique(input_reactive$paid_media_spends)), TRUE, FALSE)
      input_reactive$org_med_vars_in_cols <- ifelse(length(intersect(input_reactive$organic_vars, colnames(input_reactive$tbl))) ==
        length(unique(input_reactive$organic_vars)), TRUE, FALSE)
      input_reactive$baseline_vars_in_cols <- ifelse(length(intersect(input_reactive$context_vars, as.list(colnames(input_reactive$tbl)))) ==
        length(unique(input_reactive$context_vars)), TRUE, FALSE)

      tryCatch(input_reactive$date_transf_data <- {
        datetrasf <- (isolate(input_reactive$tbl) %>%
          mutate(
            isolate(input$date_var),
            as.Date(isolate(input$date_var),
              format = isolate(input$date_format_var)
            )
          ) %>%
          summarize(sum(is.na(input$date_var))))
      },
      warning = function(w) {
        return("")
      },
      error = function(e) {
        return("")
      },
      finally = function(f) {
        return(a[1, ])
      }
      )

      message(">>> Checking inputs...")
      if ((is.null(isolate(input$data_file)) == FALSE) &
        (is.null(isolate(input$holiday_file)) == FALSE) &
        ((isolate(input$dep_var) == "") == FALSE) &
        ((isolate(input$date_var) == "") == FALSE) &
        ((isolate(input$date_format_var) == "") == FALSE) &
        (input_reactive$date_transf_data == 0) &
        ((isolate(length(intersect(input$dep_var, as.list(colnames(input_reactive$tbl)))) == 1)) == TRUE) &
        ((isolate(length(intersect(input$date_var, as.list(colnames(input_reactive$tbl)))) == 1)) == TRUE) &
        (is.null(isolate(input$num_media)) == FALSE) &
        (is.null(isolate(input$num_organic_media)) == FALSE) &
        (is.null(isolate(input$num_context)) == FALSE) &
        (length(input_reactive$paid_media_vars) >= 2) &
        (length(input_reactive$paid_media_vars) == isolate(input$num_media)) &
        (length(input_reactive$paid_media_spends) == isolate(input$num_media)) &
        (length(input_reactive$organic_vars) == isolate(input$num_organic_media)) &
        (length(input_reactive$context_vars) == isolate(input$num_context)) &
        (input_reactive$tbl %>% select(c(
          input_reactive$paid_media_vars, input_reactive$organic_vars,
          isolate(input$dep_var)
        )) %>% sapply(is.numeric) %>% all() == TRUE) & # force numeric paid media vars and dep var
        (input_reactive$med_vars_impr_in_cols == TRUE) &
        (input_reactive$med_vars_spend_in_cols == TRUE) &
        (input_reactive$org_med_vars_in_cols == TRUE) &
        (input_reactive$baseline_vars_in_cols == TRUE)) {
        input_reactive$dt_input <- input_reactive$tbl
        input_reactive$date_format <- input$date_format_var
        input_reactive$dt_input$DATE <- as.Date(input_reactive$dt_input[, input$date_var], format = isolate(input$date_format_var))
        input_reactive$dt_input$DATE <- as.Date(gsub("00", "20", input_reactive$dt_input$DATE))
        input_reactive$holiday_data <- isolate(input_reactive$holiday_data)
        input_reactive$holiday_data$ds <- as.Date(input_reactive$holiday_data$ds)
        input_reactive$dep_var <- isolate(input$dep_var)
        input_reactive$dep_var_type <- isolate(input$dep_var_type)
        input_reactive$date_var <- "DATE"
        msg <- "Variable Input Succesful - Please click anywhere on the screen to proceed."
        showModal(modalDialog(title = msg, easyClose = TRUE, footer = NULL))
        message(msg)
      } else {
        error_message <- NULL
        if (is.null(isolate(input$data_file)) == TRUE) {
          error_message <- paste(error_message, "Issue with Data File", sep = "<br><br>")
        }
        if (is.null(isolate(input$holiday_file)) == TRUE) {
          error_message <- paste(error_message, "Issue with Holiday File", sep = "<br><br>")
        }
        if (((isolate(input$dep_var) == "") == TRUE) || isolate(length(intersect(input$dep_var, as.list(colnames(input_reactive$tbl)))) != 1)) {
          error_message <- paste(error_message, "Issue with Dependent Variable Name - Either none input, or input case-sensitive column name does not exist in data file", sep = "<br><br>")
        }
        if (((isolate(input$date_var) == "") == TRUE) || isolate(length(intersect(input$date_var, as.list(colnames(input_reactive$tbl)))) != 1)) {
          error_message <- paste(error_message, "Issue with Date Var Name, either none input or input case-sensitive column name does not exist in data file", sep = "<br><br>")
        }
        if (((isolate(input$date_format_var) == "") == TRUE) || (input_reactive$date_transf_data > 0)) {
          error_message <- paste(error_message, "Issue with Date Format Var, either none input or input format does not work for all rows. I.e. 1+ rows are transformed to NA", sep = "<br><br>")
        }
        if (is.null(isolate(input$num_media)) == TRUE) {
          error_message <- paste(error_message, "No number input for number of media variables", sep = "<br>")
        }
        if (is.null(isolate(input$num_context)) == TRUE) {
          error_message <- paste(error_message, "No number input for number of baseline variables", sep = "<br>")
        }
        if (length(input_reactive$paid_media_vars) != isolate(input$num_media)) {
          error_message <- paste(error_message, "Missing Input for >=1 media impression/click variable column names. If you do not have a media impression/click variable for a given channel enter its spend variable into the impression/click field for that variable, so you will have that spend variable input both times.", sep = "<br><br>")
        }
        if (length(input_reactive$paid_media_spends) != isolate(input$num_media)) {
          error_message <- paste(error_message, "Missing Input for >=1 media spend variable column names", sep = "<br><br>")
        }
        if (length(input_reactive$context_vars) != isolate(input$num_context)) {
          error_message <- paste(error_message, "Missing Input for >=1 baseline variable column names", sep = "<br><br>")
        }
        if (length(input_reactive$paid_media_vars) < 2) {
          error_message <- paste(error_message, "Robyn requires at least 2 paid media variables. Please add additional variables")
        }
        if (input_reactive$med_vars_impr_in_cols == FALSE) {
          error_message <- paste(error_message, "At least 1 column name in the input media impression/click variable column names does not match the column names in the data. Remember they must be input case-sensitive.", sep = "<br><br>")
        }
        if (input_reactive$med_vars_spend_in_cols == FALSE) {
          error_message <- paste(error_message, "At least 1 column name in the input media spend variable column names does not match the column names in the data. Remember they must be input case-sensitive.", sep = "<br><br>")
        }
        if (input_reactive$baseline_vars_in_cols == FALSE) {
          error_message <- paste(error_message, "At least 1 column name in the input baseline variable column names does not match the column names in the data. Remember they must be input case-sensitive.", sep = "<br><br>")
        }
        if (input_reactive$tbl %>% select(input_reactive$paid_media_vars) %>% sapply(is.numeric) %>% all() == FALSE) {
          error_message <- paste(error_message, 'At least 1 paid media variable column is not of the type NUMERIC - ensure that any non-numeric characters are removed from all paid media columns (e.g. "$", ",")', sep = "<br><br>")
        }
        if (input$num_organic_media > 0) {
          if (input_reactive$tbl %>% select(input_reactive$organic_media_vars) %>% sapply(is.numeric) %>% all() == FALSE) {
            error_message <- paste(error_message, 'At least 1 Organic media variable column is not of the type NUMERIC - ensure that any non-numeric characters are removed from all paid media columns (e.g. "$", ",")', sep = "<br><br>")
          }
        }
        if (input_reactive$tbl %>% select(isolate(input$dep_var)) %>% sapply(is.numeric) %>% all() == FALSE) {
          error_message <- paste(error_message, 'Dependent variable column is not of the type NUMERIC - ensure that any non-numeric characters are removed from the dependent variable column (e.g. "$", ",")', sep = "<br><br>")
        }
        if (input_reactive$tbl %>% is.na() %>% any() == TRUE) {
          error_message <- paste(error_message, "Dataset has <NA> or missing values. These values must be removed or fixed for the model to properly run. Please investigate row number(s) - ", paste(which(rowSums(is.na(input_reactive$tbl)) > 0), collapse = ", "), sep = "<br><br>")
        }
        message(">>> Failed to process inputs:\n", error_message)
        showModal(modalDialog(
          title = HTML(paste("<b>Variable Inputs not saved due to errors -</b>", error_message, sep = "<br>")),
          easyClose = TRUE,
          footer = NULL
        ))
      }
    } else {
      msg <- "Variable Input Succesful - Please click anywhere on the screen to proceed."
      showModal(modalDialog(title = msg, easyClose = TRUE, footer = NULL))
      message(msg)
    }
  })

  ######################## Exploratory Data Analysis tab functionality ######################
  observeEvent(input$EDA_initiate, {
    vars_inputted <- unique(c(
      input_reactive$date_var, input_reactive$dep_var,
      input_reactive$paid_media_spends, input_reactive$paid_media_vars,
      input_reactive$organic_vars, input_reactive$context_vars
    ))
    eda_input <- as.data.frame(input_reactive$dt_input)[, vars_inputted]
    ##############################################################
    ####   1. Examine data completeness for all variables    #####
    ##   Get percent of non-missing data for all variables   #####
    ##############################################################
    no_rows <- length(eda_input$DATE) # Get number of rows in the input data
    nonNA_counts <- as.data.frame(lapply(eda_input, function(x) {
      sum(!is.na(x))
    }))
    # nonNA_counts <- eda_input[, lapply(.SD, function(x) sum(!is.na(x))), .SDcols = names(eda_input)]
    input_reactive$nonNA_counts_long <- nonNA_counts %>%
      pivot_longer(cols = everything(), names_to = "variable", values_to = "non_NA_count")
    input_reactive$nonNA_counts_long <- as.data.frame(input_reactive$nonNA_counts_long)
    input_reactive$nonNA_counts_long$pct_of_non_missing_data <- round(input_reactive$nonNA_counts_long$non_NA_count * 1.00 / no_rows, 2)
    input_reactive$nonNA_counts_long$pct_of_non_missing_data_cat <- ifelse(input_reactive$nonNA_counts_long$pct_of_non_missing_data == 1, "Data is complete", "Has missing data")

    # Dynamic warning message based on pct of non-missing data:
    message_1_bad <- paste0(
      "<B>1. Percent of Non-Missing Data for Each Variable:</B>", "<br>",
      "Variable ",
      ifelse(length(input_reactive$nonNA_counts_long[input_reactive$nonNA_counts_long$pct_of_non_missing_data_cat == "Has missing data", ]) > 0,
        paste(input_reactive$nonNA_counts_long[input_reactive$nonNA_counts_long$pct_of_non_missing_data_cat == "Has missing data", ], sep = ","),
        "(None)"
      ),
      " has missing data. ",
      "", "The Robyn MMM model will not run properly on a dataset with missing data. Examine the variable(s) with missing data to see if the missing data can be added or imputed.
                                         There are different techniques to impute or interpolate missing data for time series type of data, such as mean, median, linear, and spline interpolation, etc.
                                         We encourage you to do some research to see what makes the most sensse for your data. ",
      " ", "Refer to chart 1 (Percent of non-missing data for each variable) for more detail."
    )

    message_1_good <- paste0(
      "<B>1. Percent of Non-Missing Data for Each Variable:</B>",
      "<br>",
      "None of the variables has missing data. No action is needed here.",
      " ", "You can still refer to chart 1 (Percent of non-missing data for each variable) for more detail."
    )

    # Decide which message to show:
    message_1 <- ifelse(sum(input_reactive$nonNA_counts_long$pct_of_non_missing_data_cat == "Has missing data") >= 1,
      message_1_bad, message_1_good
    )

    output$print_message_1 <- renderText(paste0(message_1, "<br>", "<br>"))


    ####################################################################
    ###        2. Examine completeness of time periods:             ####
    ##   Get number of observations by year, month, week, weekday   ####
    ##     to capture any inconsistency in the input data           ####
    ####################################################################

    # Prepare the data for getting the missing time periods:
    date_column <- as.data.frame(eda_input$DATE[order(as.Date(eda_input$DATE))])
    colnames(date_column) <- "DATE"
    date_column$date_previous_row <- lag(date_column$DATE, 1)
    date_column$date_diff_in_days <- as.numeric(difftime(date_column$DATE, date_column$date_previous_row, units = "days"))

    # Calculate the correct lag
    correct_lag <- min(as.numeric(date_column$date_diff_in_days), na.rm = TRUE)
    date_column$date_diff_incorrect <- date_column$date_diff_in_days != correct_lag
    date_column$date_diff_incorrect[is.na(date_column$date_diff_incorrect)] <- FALSE

    # Dynamic warning message based on missing time periods:
    message_2_bad <- paste0(
      "<B>2. Missing Time Periods:</B>", "<br>",
      "There seems to be some missing time period(s) after: ",
      ifelse(date_column$date_diff_incorrect,
        paste(date_column$date_previous_row[date_column$date_diff_incorrect == TRUE], sep = ","),
        "(None)"
      ),
      ".",
      " ", "Data completeness is crucial to the quality of the model.",
      " ", "Examine the missing time periods to see if data for those time periods can be added.",
      " ", "Refer to charts 2a-2d (Number of observations by year, month, week and weekday) to see if the missing time periods follow certain patterns."
    )

    message_2_good <- paste0(
      "<B>2. Missing Time Periods:</B>",
      "<br>",
      "The time periods in your data seem to be complete. No action is needed here.",
      " ", "You can still refer to charts 2a-2d (Number of observations by year, month, week and weekday) to check the patterns of the time periods along these dimensions."
    )

    # Decide which message to show:
    message_2 <- ifelse(sum(date_column$date_diff_incorrect == TRUE) >= 1,
      message_2_bad, message_2_good
    )

    output$print_message_2 <- renderText(paste0(message_2, "<br>", "<br>"))

    #############################################################################################
    ###        3a. Examine pair-wise correlation between all independent numeric variables:  ####
    ###                     to capture any unwanted highly-correlated variables              ####
    #############################################################################################
    # Only select the media spend variables and numeric baseline vars for pair-wise correlation:

    a <- eda_input[, input_reactive$paid_media_spends]
    if (is.null(input_reactive$context_vars)) {
      NULL
    } else {
      a <- cbind(a, eda_input[, (input_reactive$context_vars)])
    }
    if (is.null(input_reactive$organic_vars)) {
      NULL
    } else {
      a <- cbind(a, eda_input[, (input_reactive$organic_vars)])
    }

    dt_pw_corr <- a

    tryCatch(
      dt_pw_corr_n <- dt_pw_corr[, sapply(dt_pw_corr, is.numeric)],
      error = function(e) {}
    )
    corr <- round(cor(dt_pw_corr_n, use = "complete.obs", method = "pearson"), 2) # calculate correlation matrix

    idx <- as.data.frame(which(abs(corr) >= 0.8, arr.ind = TRUE)) # get the indices for the matrix entries with abs(correlations) >=0.8
    idx <- idx[which(idx$row > idx$col), ]
    pair_name_1 <- rownames(corr)[idx$row]
    pair_name_2 <- colnames(corr)[idx$col]

    # Get list of highly correlated variable pairs
    high_corr_var_pairs <- function() {
      message <- ""
      for (i in seq_along(pair_name_1))
      {
        message <- paste0(message, "(", pair_name_1[i], ", ", pair_name_2[i], ") ")
      }
      return(message)
    }

    # Get dynamic message based on correlation matrix between independent variables
    message_3a_bad <- paste0(
      "<B>3a. Correlation Between Independent Variables:</B>",
      "<br>", "Variable pair(s) ", high_corr_var_pairs(),
      "have a correlation magnitude of >=0.8 with each other.",
      "<br>", "1) Some high correlations between independent variables are expected. For example, when two media channels have a similar spending pattern, they will naturally have a higher intercorrelation. In this case, the higher intercorrelation reflects reality and it makes sense to keep both independent variables in the model.",
      "<br>", "2) Other higher correlations between independent variables may suggest redundancy. Consider the example of including both percent of revenue spent on video creatives and percent of revenue spent on mobile-optimal creatives. In this case, one variable clearly includes the other and it might make sense to select the one with the higher correlation with the dependent variable to include in the model.",
      "<br>", "3) When you have both media exposure data (such as impressions, clicks, GRPs, etc.) and spend data for a specific channel, the recommendation is to choose a media exposure variable over the spend variable to include in the model, especially for offline channels where spend level doesnt always accurately reflect the media exposure level.",
      "<br>", "Refer to chart 3a (Pair-wise correlation between independent variable) for more detail."
    )

    message_3a_good <- paste0(
      "<B>3a. Correlation Between Independent Variables:</B>",
      "<br>", "None of the independent variables has a correlation magnitude of >=0.8 with each other. No immediate action is needed here. You can still refer to some general recommendations regarding correlation between independent variables below:",
      "<br>", "1) Some high correlations between independent variables are expected. For example, when two media channels have a similar spending pattern, they will naturally have a higher intercorrelation. In this case, the higher intercorrelation reflects reality and it makes sense to keep both independent variables in the model.",
      "<br>", "2) Other higher correlations between independent variables may suggest redundancy. Consider the example of including both percent of revenue spent on video creatives and percent of revenue spent on mobile-optimal creatives. In this case, one variable clearly includes the other and it might make sense to select the one with the higher correlation with the dependent variable to include in the model.",
      "<br>", "3) When you have both media exposure data (such as impressions, clicks, GRPs, etc.) and spend data for a specific channel, the recommendation is to choose a media exposure variable over the spend variable to include in the model, especially for offline channels where spend level doesnt always accurately reflect the media exposure level.",
      "<br>", "Refer to chart 3a (Pair-wise correlation between independent variable) for more detail."
    )

    # Decide which message to show:
    message_3a <- ifelse(length(which(abs(corr) >= 0.8 & corr != 1)) >= 1, message_3a_bad, message_3a_good)

    output$print_message_3a <- renderText(paste0("<B>3a. Correlation Between Independent Variables:</B><br>Only one independent variable, no correlation calculation possible"))



    ###############################################################################################
    ###        3b. Examine correlation of all numeric variables against dependent variable:    ####
    ###                 to capture any unwanted highly-correlated variables                    ####
    ###############################################################################################
    # Get a data set for all numeric variables:
    eda_input_N <- eda_input[, sapply(eda_input, is.numeric)]

    # Get correlation of all numeric variables vs. dependent variable:
    corr_w_dep_var <- eda_input_N %>%
      correlate(use = "complete.obs") %>%
      focus(all_of(input_reactive$dep_var))
    corr_w_dep_var <- as.data.frame(corr_w_dep_var)
    corr_w_dep_var$flag <- ifelse(abs(corr_w_dep_var[input$dep_var]) >= 0.8, ">= 0.8", "< 0.8")

    # Dynamic warning message based on correlation with dependent variable:
    message_3b <- paste0(
      "<B>3b. Correlation with Dependent Variable:</B>",
      "<br>", "Some variables such as the baseline variables naturally have a higher correlation with the dependent variable, which is expected and is no cause for concern. However, one should avoid embedding the dependent variable within the independent variable. Examine the independent variable(s) with high correlations with the dependent variable to see if they are expected or not.",
      " ", "Refer to chart 3b (Correlation with dependent variable) for more detail."
    )


    output$print_message_3b <- renderText(paste0(message_3b, "<br>", "<br>"))


    ##################################################################################
    ###             4. Look at Percent of Total Media Spend by Channel            ####
    ###     to see if it makes sense to combine or break out certain channels     ####
    ##################################################################################

    # aggregate data to yearly sales by channel:
    eda_input_media_spend_vars <- eda_input[which(colnames(eda_input) %in% c("DATE", input_reactive$paid_media_spends)), ]
    eda_input_media_spend_vars$year <- year(eda_input_media_spend_vars$DATE)
    yearly_media_spend <- eda_input_media_spend_vars %>%
      select(colnames(subset(eda_input_media_spend_vars, select = -c(get("DATE"))))) %>%
      group_by(year) %>%
      summarize(across(everything(), sum))

    yearly_media_spend <- yearly_media_spend[order(yearly_media_spend$year), ]
    yearly_media_spend$total_media_spend <- yearly_media_spend %>%
      select(colnames(subset(yearly_media_spend, select = -c(get("year"))))) %>%
      rowSums()

    yearly_media_spend_pct <- yearly_media_spend %>%
      mutate_at(colnames(subset(yearly_media_spend, select = -c(get("year")))), ~ (. / yearly_media_spend$total_media_spend))

    colnames(yearly_media_spend_pct)[which(colnames(yearly_media_spend_pct) %in% colnames(subset(yearly_media_spend_pct, select = -c(get("year"), get("total_media_spend")))))] <-
      lapply(colnames(subset(yearly_media_spend_pct, select = -c(get("year"), get("total_media_spend")))), function(x) {
        paste0("pct_", x)
      })

    # Only keep the pct_spend variables
    yearly_media_spend_pct <- yearly_media_spend_pct %>% select(year | starts_with("pct_"))

    # transform data to long format
    yearly_media_spend_pct_long <- yearly_media_spend_pct %>%
      pivot_longer(!year, names_to = "media", values_to = "pct_of_total_media_spend")
    yearly_media_spend_pct_long <- as.data.frame(yearly_media_spend_pct_long)
    yearly_media_spend_pct_long$media <- gsub("pct_", "", yearly_media_spend_pct_long$media)


    # Prepare data for dynamic warning message based % of total media spend by channel:
    yearly_media_spend_pct_matrix <- as.matrix(yearly_media_spend_pct)

    idx_low_pct <- as.data.frame(which(yearly_media_spend_pct_matrix < 0.05, arr.ind = TRUE)) # get the indices for the matrix entries with value < 0.05
    pair_year_low_pct <- yearly_media_spend_pct_matrix[idx_low_pct$row, 1]
    pair_channel_low_pct <- gsub("pct_", "", colnames(yearly_media_spend_pct_matrix)[idx_low_pct$col])
    pair_channel_low_pct <- gsub("_S", "", colnames(yearly_media_spend_pct_matrix)[idx_low_pct$col])

    idx_high_pct <- as.data.frame(which(yearly_media_spend_pct_matrix > 0.6 & yearly_media_spend_pct_matrix <= 1, arr.ind = TRUE)) # get the indices for the matrix entries with value >= 0.6
    pair_year_high_pct <- yearly_media_spend_pct_matrix[idx_high_pct$row, 1]
    pair_channel_high_pct <- gsub("_S", "", colnames(yearly_media_spend_pct_matrix)[idx_high_pct$col])
    pair_channel_high_pct <- gsub("pct_", "", colnames(yearly_media_spend_pct_matrix)[idx_high_pct$col])

    # Get list of channels with low or high share of total media spend:
    low_pct_pairs <- function() {
      message <- ""
      if (length(pair_year_low_pct) == 0) {
        return(message)
      } else {
        for (i in seq_along(pair_year_low_pct))
        {
          message <- paste0(message, " (", pair_channel_low_pct[i], " in ", pair_year_low_pct[i], ")")
        }
        return(message)
      }
    }

    high_pct_pairs <- function() {
      message <- ""
      if (length(pair_year_high_pct) == 0) {
        return(message)
      } else {
        for (i in seq_along(pair_year_high_pct))
        {
          message <- paste0(message, " (", pair_channel_high_pct[i], " in ", pair_year_high_pct[i], ")")
        }
        return(message)
      }
    }

    # Dynamic warning message based % of total media spend by channel:

    message_4 <- paste0(
      "<B>4. Share of Total Media Spend for Each Channel:</B>",
      "<br>", "In general, as long as there is good variation in a channels spend level and its correlation with the dependent variable is not low, we recommend keeping that channel as a separate independent variable in the model. There are a couple of things to consider though when a channels share of total spend is very low or high:",
      "<br>", "1) When a channels share of spend is high, consider meaningful splits for that channel such as prospecting/retargeting split, brand/DR campaign split etc. to potentially improve the model efficiency.",
      "<br>", "2) When a channels share of spend is low and correlation with the dependent variable is also low, consider potentially grouping that channel into other similar channel(s).",
      "<br>", "3) It is always good practice to run experiments to calibrate the MMM results to set the ground-truth and avoid bias. This is especially true when the share of spend is very low or high for a channel.",
      "<br>", "You can refer to chart 4 for share of total media spend for each channel by year."
    )

    output$print_message_4 <- renderText(paste0(message_4, "<br>", "<br>"))


    ####################################################
    ####                   Plots                   #####
    ####################################################
    pal1 <- c("Has missing data" = "tomato2", "Data is complete" = "#989898") # Set legend colors
    output$ggplot1 <- renderPlot(
      {
        ggplot(
          input_reactive$nonNA_counts_long,
          aes(
            x = reorder(get("variable"), -get("pct_of_non_missing_data")),
            y = get("pct_of_non_missing_data"),
            fill = get("pct_of_non_missing_data_cat"),
            label = paste0(get("pct_of_non_missing_data") * 100, "%")
          )
        ) +
          geom_col(width = 0.6) +
          labs(
            title = "1. Percent of non-missing data for each variable",
            subtitle = paste0("Examine the variables with missing data", " ", "(highlighted in red below)"),
            x = "Variable",
            y = "% of non-missing data",
            caption = paste0("Total number of observations from input data: ", length(input_reactive$tbl$DATE)),
            fill = paste0("Data Completeness")
          ) +
          theme_bw(base_size = 14) +
          scale_fill_manual(values = pal1, limits = names(pal1)) +
          scale_y_continuous(position = "right") +
          geom_text(hjust = 1.1, size = 3.6, colour = "white") +
          coord_flip() +
          theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
          theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick")) +
          theme(axis.title.x = element_text(vjust = 1, hjust = 1)) +
          theme(legend.title = element_text(size = 12))
      },
      height = 600
    )


    # 2.a count by year data, flag if pct_diff_vs_count_max is > 5%:
    input_reactive$tbl$year <- year(input_reactive$tbl$DATE)
    input_reactive$year_counts <- input_reactive$tbl %>%
      count(year) %>%
      arrange(get("."), year)
    colnames(input_reactive$year_counts)[2] <- "count"
    input_reactive$year_counts$count_max <- max(input_reactive$year_counts$count)
    input_reactive$year_counts$pct_diff_vs_count_max <- (input_reactive$year_counts$count_max - input_reactive$year_counts$count) / input_reactive$year_counts$count_max
    input_reactive$year_counts$flag <- ifelse(input_reactive$year_counts$pct_diff_vs_count_max >= 0.05, ">= 5%", "< 5%")
    input_reactive$year_counts$year <- as.character(input_reactive$year_counts$year)

    # 2a. Plot count by year:
    pal2 <- c(">= 5%" = "tomato2", "< 5%" = "#989898") # Set legend colors
    output$ggplot2a <- renderPlot({
      ggplot(
        input_reactive$year_counts,
        aes(
          x = .data$year,
          y = .data$count,
          fill = get("flag"),
          label = .data$count
        )
      ) +
        geom_col(width = 0.6) +
        labs(
          title = "2a. Number of observations by year",
          subtitle = paste0("Examine the year(s) with unexpected fewer number of observations"),
          x = "year",
          y = "number of observations",
          caption = paste0("Total number of observations from input data: ", length(input_reactive$tbl$DATE)),
          fill = paste0("% difference vs.", "\n", "max number of", "\n", "observations per year")
        ) +
        theme_bw(base_size = 12) +
        scale_fill_manual(values = pal2, limits = names(pal2)) +
        geom_text(vjust = 1.5, size = 4, colour = "white") +
        theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
        theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick")) +
        theme(legend.title = element_text(size = 12))
    })


    # 2b. count by month data -- need to decide on flag criteria if need any

    input_reactive$tbl$month <- month(input_reactive$tbl$DATE)
    input_reactive$month_counts <- input_reactive$tbl %>%
      count(month) %>%
      arrange(get("."), month)
    colnames(input_reactive$month_counts)[2] <- "count"
    input_reactive$month_counts$count_max <- max(input_reactive$month_counts$count)
    input_reactive$month_counts$pct_diff_vs_count_max <- (input_reactive$month_counts$count_max - input_reactive$month_counts$count) / input_reactive$month_counts$count_max
    input_reactive$month_counts$month_abb <- month.abb[input_reactive$month_counts$month]

    # 2b. Plot count by month:
    xaxis_ticks_2b <- month.abb[seq(1, 12, 1)]

    output$ggplot2b <- renderPlot({
      ggplot(
        input_reactive$month_counts,
        aes(
          x = reorder(get("month_abb"), .data$month),
          y = .data$count,
          label = .data$count
        )
      ) +
        geom_col(width = 0.6, fill = "#989898") +
        labs(
          title = "2b. Number of observations by month",
          subtitle = paste0("Examine the month(s) with unexpected fewer number of observations"),
          x = "month",
          y = "number of observations",
          caption = paste0("Total number of observations from input data: ", length(input_reactive$tbl$DATE))
        ) +
        theme_bw(base_size = 12) +
        scale_x_discrete(limits = xaxis_ticks_2b) +
        geom_text(vjust = 1.5, size = 4, colour = "white") +
        theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
        theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick"))
    })

    # 2c. count by week data, flag if count is smaller than max count per week

    input_reactive$tbl$week <- week(input_reactive$tbl$DATE)
    input_reactive$week_counts <- input_reactive$tbl %>%
      count(week) %>%
      arrange(get("."), week)
    colnames(input_reactive$week_counts)[2] <- "count"
    input_reactive$week_counts$count_max <- max(input_reactive$week_counts$count)
    input_reactive$week_counts$pct_diff_vs_count_max <- (input_reactive$week_counts$count_max - input_reactive$week_counts$count) / input_reactive$week_counts$count_max
    input_reactive$week_counts$week_char <- as.character(input_reactive$week_counts$week)

    # 2c. Plot count by week:
    pal3 <- c("Yes" = "tomato2", "No" = "#989898") # Set legend colors
    xaxis_ticks_2c <- as.character(seq(1, 53, 1))

    output$ggplot2c <- renderPlot({
      ggplot(input_reactive$week_counts, aes(
        x = reorder(get("week_char"), get("week")),
        y = .data$count,
        label = .data$count,
        color = get("week_char")
      )) +
        geom_point(
          size = 3
        ) +
        geom_segment(aes(
          x = reorder(get("week_char"), get("week")),
          xend = reorder(get("week_char"), get("week")),
          y = 0,
          yend = count
        )) +
        labs(
          title = "2c. Number of observations by week",
          subtitle = paste0("Examine the week(s) with unexpected fewer number of observations"),
          x = "week of year",
          y = "number of observations",
          caption = paste0("Total number of observations from input data: ", length(input_reactive$tbl$DATE)),
          color = paste0("Fewer than max", "\n", "number of observations", "\n", "per week?")
        ) +
        scale_x_discrete(limits = xaxis_ticks_2c) +
        scale_color_manual(values = pal3, limits = names(pal3)) +
        theme_bw(base_size = 12) +
        theme(axis.text.x = element_text(angle = 65, vjust = 0.6)) +
        theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
        theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick")) +
        theme(legend.title = element_text(size = 12))
    })

    # 2d. count by weekday data -- need to decide on flag criteria if need any

    input_reactive$tbl$weekday <- weekdays(as.Date(input_reactive$tbl$DATE, format = input$date_format_var))
    input_reactive$weekday_counts <- input_reactive$tbl %>%
      count(input_reactive$tbl$weekday) %>%
      arrange(.by_group = TRUE)
    colnames(input_reactive$weekday_counts)[1:2] <- c("weekday", "count")
    input_reactive$weekday_counts$count_max <- max(input_reactive$weekday_counts$count)
    input_reactive$weekday_counts$pct_diff_vs_count_max <- (input_reactive$weekday_counts$count_max - input_reactive$weekday_counts$count) / input_reactive$weekday_counts$count_max

    # 2d. Plot count by weekday:
    output$ggplot2d <- renderPlot({
      ggplot(
        input_reactive$weekday_counts,
        aes(
          x = .data$weekday,
          y = .data$count,
          label = .data$count
        )
      ) +
        geom_col(width = 0.3, fill = "#989898") +
        labs(
          title = "2d. Number of observations by weekday",
          subtitle = paste0("Examine the weekday(s) with unexpected fewer number of observations"),
          x = "weekday",
          y = "number of observations",
          caption = paste0("Total number of observations from input data: ", length(input_reactive$tbl$DATE))
        ) +
        theme_bw(base_size = 12) +
        geom_text(vjust = 1.5, size = 4, colour = "white") +
        theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
        theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick"))
    })

    # Plot 3a
    output$ggplot3a <- renderPlot(
      {
        lares::corr_cross(eda_input, method = "pearson", ignore = input_reactive$dep_var, top = 20) +
          labs(title = "3a. Pair-wise correlation between independent variables")
      },
      height = 600
    )

    # Plot 3b
    output$ggplot3b <- renderPlot(
      {
        ggplot(corr_w_dep_var, aes(
          x = reorder(get("term"), desc(get(input_reactive$dep_var))),
          y = get(input_reactive$dep_var),
          label = round(get(input_reactive$dep_var), 2)
        )) +
          geom_col(width = 0.6, fill = "#989898") +
          labs(
            title = "3b. Correlation with dependent variable",
            subtitle = paste0("Examine the variable(s) with high correlations with the dependent variable to see if they are expected or not"),
            x = "variable",
            y = paste("correlation with", input_reactive$dep_var)
          ) +
          theme_bw(base_size = 14) +
          geom_text(vjust = -0.5, size = 4, colour = "black", fontface = "bold") +
          theme(plot.title = element_text(size = 16, hjust = 0.5, colour = "blue")) +
          theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick")) +
          theme(legend.title = element_text(size = 12))
      },
      height = 600
    )

    # Plot 4
    # This plot will be different depending on how many years there are in the input data because stacked area chart doesn't work for single year data
    output$ggplot4 <- renderPlot(
      {
        ggplot4_multiple_year <- ggplot(
          yearly_media_spend_pct_long,
          aes(
            x = .data$year, y = get("pct_of_total_media_spend"),
            fill = get("media"),
            label = paste0(round(get("pct_of_total_media_spend") * 100, 1), "%")
          )
        ) +
          geom_area(alpha = 0.6, size = .5, colour = "white") +
          labs(
            title = "4. Share of total media spend for each channel",
            x = "year",
            y = "% of total media spend",
            fill = paste("channel")
          ) +
          theme_bw(base_size = 12) +
          geom_text(size = 4, position = position_stack(vjust = 0.5), fontface = "bold") +
          scale_fill_viridis_d() +
          theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
          # theme(plot.subtitle=element_text(size=14,  hjust=0.5, face="italic", color="firebrick")) +
          theme(legend.title = element_text(size = 12))

        ggplot4_single_year <- ggplot(
          yearly_media_spend_pct_long,
          aes(
            x = as.character(.data$year), y = get("pct_of_total_media_spend"),
            fill = get("media"),
            label = paste0(round(get("pct_of_total_media_spend") * 100, 1), "%")
          )
        ) +
          geom_bar(position = "stack", stat = "identity") +
          labs(
            title = "4. Share of total media spend for each channel",
            x = "year",
            y = "% of total media spend",
            fill = paste("channel")
          ) +
          theme_bw(base_size = 12) +
          geom_text(size = 4, position = position_stack(vjust = 0.5), fontface = "bold", color = "black") +
          scale_colour_viridis_d(option = "D") +
          theme(plot.title = element_text(size = 16, hjust = 0.5, color = "blue", margin = margin(5, 0, 5, 0))) +
          # theme(plot.subtitle=element_text(size=14,  hjust=0.5, face="italic", color="firebrick")) +
          theme(legend.title = element_text(size = 12))

        ifelse(length(yearly_media_spend$year) > 1, print(ggplot4_multiple_year), print(ggplot4_single_year))
      },
      height = 600
    )

    #####################################################################
    ###     5. Look at trends for any continuous variable by year    ####
    ###          and compare each year's trend against the rest      ####
    ###              to detect any obvious data anomalies            ####
    #####################################################################
    # Set input variables:
    # granularity <- "weekly" # Specify the granularity of your input data (choose from "weekly" or "daily")
    # Specify the continuous variable you want to plot trend for

    # Set breaks for X axis
    # brks <- if(input$granularity=="weekly") seq(1,53,4) else seq(1,366,14)

    # Dynamic warning message: None for this section
    output$var_to_plot_input <- renderUI({
      selectInput("var_to_plot",
        label = h4("Paid Media Variable to Plot"),
        choices = c(input_reactive$paid_media_vars, input_reactive$organic_vars, input_reactive$context_vars), selected = NULL
      )
    })


    # Plot 5
    output$ggplot5 <- renderPlot(
      {
        ggplot(eda_input[order(eda_input$DATE), ], aes(x = (if (input$granularity == "weekly") week(get("DATE")) else yday(get("DATE"))))) +
          geom_line(aes(y = get(input$var_to_plot), colour = as.factor(year(get("DATE"))))) +
          labs(
            title = paste("5. Trend of", input$var_to_plot, "by year"),
            subtitle = "Compare each year's trend against the rest to detect obvious data anomalies",
            x = (if (input$granularity == "weekly") "week of year" else "day of year"),
            y = paste(input$var_to_plot),
            colour = "Year"
          ) +
          theme(plot.title = element_text(size = 16, hjust = 0.5, colour = "blue", margin = margin(5, 0, 5, 0))) +
          theme(plot.subtitle = element_text(size = 14, hjust = 0.5, face = "italic", color = "firebrick")) +
          scale_x_continuous(breaks = (if (input$granularity == "weekly") seq(1, 53, 4) else seq(1, 366, 14))) +
          facet_wrap(~ as.factor(year(get("DATE"))))
      },
      height = 600
    )
  })

  observeEvent(input$granularity_popover, {
    showModal(modalDialog(
      title = "Time Grain of dataset",
      "Make a selection between daily or weekly based on the granularity of your data",
      easyClose = TRUE,
      footer = NULL
    ))
  })

  #############################HyperParameter Selection/Model Run ########################################

  observeEvent(input$adstock_selection_popover, {
    showModal(modalDialog(
      title = "Adstock Selection",
      HTML(paste("Adstocks are a critical part of building Marketing Mix Models. Adstocks can be complex to understand but essentially they are a measure of how much media effect carries over from Period X to Period X+1.",
        paste0("In Robyn, there are two different distributions you can choose to develop your adstocks for each paid media channel. These two distributions are ", a("Geometric", href = "https://en.wikipedia.org/wiki/Geometric_distribution", target = "_blank"), " and ", a("Weibull", href = "https://en.wikipedia.org/wiki/Weibull_distribution", target = "_blank"), " distributions."),
        "The distribution you choose is important because there are different hyperparameters for them. The geometric adstock has one hyperparameter - Theta, which represents the decay rate from period X to period X+1 in a generic way. So for example, if we had a channel with an theta value of 0.9, that would mean that only 90% of the effect from period X would carry over into period X+1. And then 90% of the value of X+1 (or 81% of the value of X) would carry over to X+2, and so on.",
        "With Weibull distributions, it is a little more flexible since it has two hyperparameters - shape and scale, whereas geometric only has theta. In the Weibull distribution, Shape is the parameter that controls the decay shape between exponential and s-shape. The larger, the more s-shape, meaning the carryover effect will be stronger and last longer in time. the smaller, the more L-shape, meaning the weaker the carryover effect, and the shorter its effect lasts in time. Scale is the parameter that controls the position of the decay inflection point. Recommended bounds are between 0 and 0.1. This is because scale can inflate adstocking half-life siginificantly.",
        "The additional hyperparameter used in Weibull adstocking can help allow the adstocks to better represent the reality through increased flexibility. At times, this increased flexibility can lead to an increased fit of the model. With that, an extra hyperparameter will also lead to longer run times for the modeling process. Consider trying out both and seeing what works best for your model.",
        "In the graphs to the right, you can explore how changing hyperparameters effect adstocking for your paid media variables. Try checking out both the Weibull adstock and Geometric adstock options to get an understanding of the difference.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$set_iter_popover, {
    showModal(modalDialog(
      title = "Iteration Count",
      HTML("2000 iterations per Trial is the recommended value. More iterations will require more computation time, so use your judgement to decide what is best. Geometric adstock + 2000 iterations + 5 trials with 6 cores takes about an hour to run, Weibull adstocks will take at least double that."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$trial_count_popover, {
    showModal(modalDialog(
      title = "Trial Count",
      HTML("Recommend trial count is 100 with calibration, since it will be minimizing three objective functions (NRMSE, DECOMP.RSSD, MAPE.LIFT) and 40 trials without calibration since it is only minimizing two objective functions (NRMSE, DECOMP.RSSD). <b>Robyn will create Trials * Iterations models</b>, so the more trials and the more iterations you add, the more models Robyn will build and the longer it will take to finish computation."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$calibration_popover, {
    showModal(modalDialog(
      title = "Calibration Set-up",
      HTML(paste("By applying results from randomized controlled-experiments, you may improve the accuracy of your marketing mix models dramatically. It is recommended to run these on a recurrent basis to keep the model calibrated permanently. In general, we want to compare the experiment result with the MMM estimation of a marketing channel. Conceptually, this method is like a Bayesian method, in which we use experiment results as a prior to shrink the coefficients of media variables. A good example of these types of experiments is Facebooks conversion lift tool which can help guide the model towards a specific range of incremental values.",
        '<img src="https://facebookexperimental.github.io/Robyn/img/calibration1.png" alt="Lamp" width = 1000 height = 400>',
        "The figure illustrates the calibration process for one MMM candidate model. Facebooks Nevergrad gradient-free optimization platform allows us to include the <b>MAPE.LIFT</b> as a third optimization score besides <b>Normalized Root Mean Square Error (NRMSE)</b> and <b>decomp.RSSD ratio</b> (Please refer to the automated hyperparameter selection and optimization for further details) providing a set of <b>Pareto optimal model solutions</b> that minimize and converge to a set of Pareto optimal model candidates.",
        'The reason why calibration requires more trials is as calibration with experiments is considered ground-truth, Robyn gives MAPE.LIFT "higher weight" by restricting the population of pareto optimality to the best 10% of MAPE.LIFT. Therefore, calibration requires a larger population and more trials to calculate pareto-optimal results.',
        "This calibration method can be applied to other media channels which run experiments, the more channels that are calibrated, the more accurate the MMM model.",
        "<b>In Robyn, calibration works by uploading a data file with the columns channel, liftStartDate, liftEndDate, liftAbs</b>. channel - Name of the paid media channel the test ran on, liftStartDate - the day the test began, liftEndDate - the day the test ended, liftAbs - the incremental revenue/conversions (whichever is your dependent variable)",
        "<b>Ensure that any calibration data you include matches exactly what you have input as that channel data. For example, if a lift study only covered a portion of your Facebook media, you should not use it to calibrate your entire Facebook media variable.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$prophet_popover, {
    showModal(modalDialog(
      title = "Prophet",
      HTML(paste(paste0("Prophet has been included in the code in order to improve fit and forecast time series by decomposing the data into trend, seasonality, holiday and weekday (if you are using daily data) components. Prophet is a Facebook original procedure for forecasting time series data based on a model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday and weekday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. ", a("More details can be found here", href = "https://facebook.github.io/prophet/", target = "_blank")),
        "If you do not already have trend/seasonality data of your own, we would recommend you consider using Prophet for at least trend & season, but as the complexity of the model and industry being measured increases, it may be worthwhile exploring additional ways to account for time-based trends. Additionally, it could potentially help the fit of the model if you believe holidays or day of the week have an impact on your dependent variable to include those as well.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$finalize_inputs_popover, {
    showModal(modalDialog(
      title = "Info on Finalizing your Inputs - The Last Step before Modeling",
      HTML(paste("Clicking the Finalizing All Model Inputs button will finalize all of your inputs from this tab and the previous tabs. You can always go back and change something and click this button again to update values. If there is an error here you will need to go back and fix an input somewhere.",
        "If everything looks good you will get a message that says so and then you can move on to <b>Initiate ROBYN Modeling</b> and click there! As the model runs, you will be able to see output about the progress updating on the bottom of the page below the hyperparameter sliders and calibration data table.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$response_curve_popover, {
    showModal(modalDialog(
      title = "Response Curves",
      HTML(paste("Cost-Response curves, Response Curves, or Diminishing Returns Curves are one of the key outputs of Marketing Mix Models. The concept is that each additional unit of advertising increases the response, but at a diminishing rate. With these curves, Marketers can understand at what their optimal investment level in a given channel is, or when used in aggregate, how to make overall budget investment decisions.",
        "A key principle that these Response curves follow is the theory of diminishing returns. This means that at a certain point we expect the return on investment of ($X+1 - $X) < ($X - $X-1) or in other words, the profitability of any given channel will eventually reach a point where it is no longer an acceptable return on investment for a business. Typically, when ROI < 1.",
        "These Response Curves also have hyperparameters - Alpha and Gamma. These hyperparameters will effect the shape of the response curve. The higher the <b>Alpha</b> hyperparameter, the more S-shape the curve will be. The lower the Alpha, the more C-shape the hyperparameter will be. For <b>Gamma</b> hyperparameters, the higher the value the higher the inflection point will be, or in other words, the higher the level of investment that the channel will start hitting diminishing returns on investment. The inflection point is also the point at which marginal ROI is at its maximum. A lower Gamma means that the channel will reach a saturation point at a lower level of investment.",
        "Please look at the charts here to get a better understanding of how these response curves may behave. Shortly you will have response curves of your own generated!",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$hyperparam_slider_popover, {
    showModal(modalDialog(
      title = "Default Ranges for Hyperparams",
      HTML(paste("We have set the default hyperparameter ranges for each channel to be fairly wide so that the evolutionary algorithm has ample room to move when testing different combinations of values. For some more guidance from the Robyn team - ",
        "For geometric adstock, use theta, alpha & gamma. For weibull adstock, use shape, scale, alpha, gamma",
        "<b>Theta</b>: In geometric adstock, theta is decay rate. guideline for usual media genre: TV c(0.3, 0.8), OOH/Print/Radio c(0.1, 0.4), digital c(0, 0.3). Please note that the recommended theta values are for weekly data. Increase range to higher values for daily data. For an example of TV, 80% theta means half-life of 4 periods. In other words, for weekly data, the effect of the TV reduces to half after 4 weeks. For daily data, half-life of 80% decay will be 4 days. Adapt theta to match expectation.",
        "<b>Shape</b>: In weibull adstock, shape controls the decay shape. Recommended c(0.0001, 2). The larger, the more S-shape thus the stronger the carry-over effect. The smaller, the more L-shape thus the weaker the carry-over effect.",
        "<b>Scale</b>: In weibull adstock, scale controls the decay inflexion point. Very conservative recommended bounce c(0, 0.1), becausee scale can increase adstocking half-life greatly",
        "<b>Alpha</b>: In s-curve transformation with hill function, alpha controls the shape between exponential and s-shape. Recommended c(0.5, 3). The larger the alpha, the more S-shape. The smaller, the more C-shape",
        "<b>Gamma</b>: In s-curve transformation with hill function, gamma controls the inflexion point. Recommended bounce c(0.3, 1). The larger the gamma, the later the inflection point in the response curve or in other words, the later the maximum point of marginal ROI will be.",
        "This is not meant to be set in stone or a hard recommendation. If a prior analysis points you in a different direction feel free to pursue that",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  output$adstock_curves_samples <- renderPlot({
    adstock <- input$adstock_selection
    plot_adstock(T)
  })

  output$response_curves_samples <- renderPlot({
    plot_saturation(T)
  })

  output$model_window_min <- renderUI({
    dateInput("min_date_model_build",
      label = "Input start date for model data",
      value = min(input_reactive$tbl$DATE), min = min(input_reactive$tbl$DATE), max = max(input_reactive$tbl$DATE)
    )
  })

  output$model_window_max <- renderUI({
    dateInput("max_date_model_build",
      label = "Input end date for model data",
      value = max(input_reactive$tbl$DATE), min = min(input_reactive$tbl$DATE), max = max(input_reactive$tbl$DATE)
    )
  })

  output$local_hyperparam_sliders_paid <- renderUI({
    lapply(seq_along(input_reactive$paid_media_spends), function(i) {
      if (input$adstock_selection == "weibull_cdf") {
        splitLayout(
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_alphas")),
            min = 0.001, max = 3, value = c(0.001, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_gammas")),
            min = 0, max = 3, value = c(0.3, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_shapes"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_shapes")),
            min = 0, max = 1, value = c(0.3, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_scales"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_scale")),
            min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
          )
        )
      } else if (input$adstock_selection == "weibull_pdf") {
        splitLayout(
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_alphas")),
            min = 0.001, max = 3, value = c(0.001, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_gammas")),
            min = 0, max = 3, value = c(0.3, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_shapes"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_shapes")),
            min = 0, max = 1, value = c(0.3, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_scales"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_scale")),
            min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
          )
        )
      } else if (input$adstock_selection == "geometric") {
        splitLayout(
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_alphas")),
            min = 0, max = 3, value = c(0.5, 3), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_gammas")),
            min = 0, max = 1, value = c(0.5, 1), step = 0.01
          ),
          sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i], "_thetas"),
            label = div(style = "font-size:12px", paste0(input_reactive$paid_media_spends[i], "_thetas")),
            min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
          )
        )
      }
    })
  })

  output$local_hyperparam_sliders_organic <- renderUI({
    if (length(input_reactive$organic_vars) > 0) {
      lapply(seq_along(input_reactive$organic_vars), function(i) {
        if (input$adstock_selection == "weibull_cdf") {
          splitLayout(
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_alphas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_alphas")),
              min = 0.001, max = 3, value = c(0.001, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_gammas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_gammas")),
              min = 0, max = 3, value = c(0.3, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_shapes"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_shapes")),
              min = 0, max = 1, value = c(0.3, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_scales"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_scale")),
              min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
            )
          )
        } else if (input$adstock_selection == "weibull_pdf") {
          splitLayout(
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_alphas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_alphas")),
              min = 0.001, max = 3, value = c(0.001, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_gammas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_gammas")),
              min = 0, max = 3, value = c(0.3, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_shapes"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_shapes")),
              min = 0, max = 1, value = c(0.3, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_scales"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_scale")),
              min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
            )
          )
        } else if (input$adstock_selection == "geometric") {
          splitLayout(
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_alphas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_alphas")),
              min = 0, max = 3, value = c(0.5, 3), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_gammas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_gammas")),
              min = 0, max = 1, value = c(0.5, 1), step = 0.01
            ),
            sliderInput(paste0("medVar_", input_reactive$organic_vars[i], "_thetas"),
              label = div(style = "font-size:12px", paste0(input_reactive$organic_vars[i], "_thetas")),
              min = 0, max = 1, value = c(0.1, 0.4), step = 0.01
            )
          )
        }
      })
    }
  })

  output$prophet_country <- renderUI({
    if (isTRUE(input$prophet_enable_checkbox)) {
      fluidRow(column(
        width = 4,
        textInput("country", label = h4(
          "Country\'s Alpha-2 Code ",
          tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
          actionButton("country_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), value = "DE")
      ))
    }
  })

  observeEvent(input$country_popover, {
    showModal(modalDialog(
      title = "Select Country for Prophet",
      HTML(
        paste0(
          "Input the Alpha-2 country code of the region you are measuring. To find the code view the ",
          a("wiki page", href = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2", target = "_blank")
        )
      ),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  input_reactive$prophet_options <- c("trend", "season", "holiday", "weekday")
  output$prophet_enable <- renderUI({
    if (isTRUE(input$prophet_enable_checkbox)) {
      lapply(seq_along(input_reactive$prophet_options), function(i) {
        checkboxInput(paste0("prophet_option_", input_reactive$prophet_options[i]),
          label = div(paste0("Enable Prophet ", input_reactive$prophet_options[i], " decomposition?"), style = "font-size:12px;"), value = T
        )
      })
    }
  })

  output$prophet_signs <- renderUI({
    if (isTRUE(input$prophet_enable_checkbox)) {
      lapply(seq_along(input_reactive$prophet_options), function(i) {
        if (isTRUE(input[[paste0("prophet_option_", input_reactive$prophet_options[i])]])) {
          radioButtons(paste0("prophet_sign_", input_reactive$prophet_options[i]),
            label = div(paste0("Expected Effect Sign for ", input_reactive$prophet_options[i]), style = "font-size:12px;"),
            choices = c("default", "positive", "negative"), selected = "default", inline = T
          )
        }
      })
    }
  })

  output$calibration_file <- renderUI({
    if (isTRUE(input$enable_calibration)) {
      fileInput("calibration_file", label = "Input CSV file with experiment data for calibration", accept = ".csv")
    }
  })

  output$calib_file_date_format <- renderUI({
    if (isTRUE(input$enable_calibration)) {
      fluidPage(
        textInput("calib_date_format_var", label = h4(
          "Input DATE format", tags$style(type = "text/css", "#q2{vertical-align:top;}"),
          actionButton("calib_date_format_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), value = "%Y-%m-%d"),
      )
    }
  })

  observeEvent(input$calib_date_format_popover, {
    showModal(modalDialog(
      title = "Calibration Date Format",
      HTML(paste("To ensure that the date variable formatting is ingested correctly we must specify the formatting. In practice, this will look something like %Y-%m-%d, which corresponds to YYYY-mm-dd or a date like 2021-1-30. If your data is formatted like month/day/year, then your format will be %m/%d/%Y an example of which would be 12/27/2020 etc.",
        "Since we are dealing with daily data at the smallest here, our formatting will use the letters lowercase m (month), lowercase d (day), and uppercase Y (Year) to specify the format. If you are having trouble specifying the correct format, it may be easier to reformat your data to the default format of %Y-%m-%d.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  input_reactive$calib_data <- NULL
  output$lift_calib_tbl <- renderDataTable({
    if (isTRUE(input$enable_calibration)) {
      file <- input$calibration_file
      ext <- tools::file_ext(file$datapath)
      validate(need(ext == "csv", "Please upload a csv file for calibration with experiments"))
      input_reactive$calib_data <- read.csv(file$datapath)
      calib_data <- read.csv(file$datapath)
      datatable(calib_data, options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t"))
    }
  })

  observeEvent(input$finalize_hyperparams, {
    message(paste(">>> Creating", input$adstock_selection, "hyperparameters..."))
    input_reactive$hyp_org <- list()
    input_reactive$hyp_paid <- list()
    input_reactive$hyperparameters <- list()
    if (input$adstock_selection %in% c("weibull_cdf", "weibull_pdf")) {
      vals <- list()
      names_l <- list()
      lapply(seq_along(input_reactive$paid_media_spends), function(i) {
        assign(paste0(input_reactive$paid_media_spends[i], "_alphas"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas")]][2]
        ))
        assign(paste0(input_reactive$paid_media_spends[i], "_gammas"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas")]][2]
        ))
        assign(paste0(input_reactive$paid_media_spends[i], "_shapes"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_shapes")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_shapes")]][2]
        ))
        assign(paste0(input_reactive$paid_media_spends[i], "_scales"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_scales")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_scales")]][2]
        ))
        hyps <- c("alphas", "gammas", "shapes", "scales")
        for (i in seq_along(hyps)) {
          new_val <- eval(parse(text = (ls()[which(grepl(hyps[i], ls()) != 0)])))
          vals[[length(vals) + 1]] <- new_val
          names_val <- ls()[which(grepl(hyps[i], ls()) != 0)]
          names_l[[length(names_l) + 1]] <- names_val
        }
        names(vals) <- names_l
        input_reactive$hyp_paid <- c(input_reactive$hyp_paid, vals)
      })
      if (length(input_reactive$organic_vars > 0)) {
        vals <- list()
        names_l <- list()
        lapply(seq_along(input_reactive$organic_vars), function(i) {
          assign(paste0(input_reactive$organic_vars[i], "_alphas"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_alphas")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_alphas")]][2]
          ))
          assign(paste0(input_reactive$organic_vars[i], "_gammas"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_gammas")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_gammas")]][2]
          ))
          assign(paste0(input_reactive$organic_vars[i], "_shapes"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_shapes")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_shapes")]][2]
          ))
          assign(paste0(input_reactive$organic_vars[i], "_scales"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_scales")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_scales")]][2]
          ))

          hyps <- c("alphas", "gammas", "shapes", "scales")
          for (i in seq_along(hyps)) {
            new_val <- eval(parse(text = (ls()[which(grepl(hyps[i], ls()) != 0)])))
            vals[[length(vals) + 1]] <- new_val
            names_val <- ls()[which(grepl(hyps[i], ls()) != 0)]
            names_l[[length(names_l) + 1]] <- names_val
          }
          names(vals) <- names_l
          input_reactive$hyp_org <- c(input_reactive$hyp_org, vals)
        })
      }
      input_reactive$hyperparameters <- c(input_reactive$hyp_paid, input_reactive$hyp_org)
    } else if (input$adstock_selection == "geometric") {
      vals <- list()
      names_l <- list()
      lapply(seq_along(input_reactive$paid_media_spends), function(i) {
        assign(paste0(input_reactive$paid_media_spends[i], "_alphas"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_alphas")]][2]
        ))
        assign(paste0(input_reactive$paid_media_spends[i], "_gammas"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_gammas")]][2]
        ))
        assign(paste0(input_reactive$paid_media_spends[i], "_thetas"), c(
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_thetas")]][1],
          input[[paste0("medVar_", input_reactive$paid_media_spends[i], "_thetas")]][2]
        ))
        hyps <- c("alphas", "gammas", "thetas")
        for (i in seq_along(hyps)) {
          new_val <- eval(parse(text = (ls()[which(grepl(hyps[i], ls()) != 0)])))
          vals[[length(vals) + 1]] <- new_val
          names_val <- ls()[which(grepl(hyps[i], ls()) != 0)]
          names_l[[length(names_l) + 1]] <- names_val
        }
        names(vals) <- names_l
        input_reactive$hyp_paid <- c(input_reactive$hyp_paid, vals)
      })
      if (length(input_reactive$organic_vars) > 0) {
        vals <- list()
        names_l <- list()
        lapply(seq_along(input_reactive$organic_vars), function(i) {
          assign(paste0(input_reactive$organic_vars[i], "_alphas"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_alphas")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_alphas")]][2]
          ))
          assign(paste0(input_reactive$organic_vars[i], "_gammas"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_gammas")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_gammas")]][2]
          ))
          assign(paste0(input_reactive$organic_vars[i], "_thetas"), c(
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_thetas")]][1],
            input[[paste0("medVar_", input_reactive$organic_vars[i], "_thetas")]][2]
          ))
          hyps <- c("alphas", "gammas", "thetas")
          for (i in seq_along(hyps)) {
            new_val <- eval(parse(text = (ls()[which(grepl(hyps[i], ls()) != 0)])))
            vals[[length(vals) + 1]] <- new_val
            names_val <- ls()[which(grepl(hyps[i], ls()) != 0)]
            names_l[[length(names_l) + 1]] <- names_val
          }
          names(vals) <- names_l
          input_reactive$hyp_org <- c(input_reactive$hyp_org, vals)
          # input_reactive$hyperparameters <- c(input_reactive$hyperparameters, input_reactive$hyp_org)
        })
      }
      input_reactive$hyperparameters <- c(input_reactive$hyp_org, input_reactive$hyp_paid)
    }
    # hyperparameters <<- input_reactive$hyperparameters

    message(">>> Setting up calibration and prophet...")
    input_reactive$activate_prophet <- NULL
    input_reactive$prophet_vars <- NULL
    input_reactive$prophet_signs <- NULL
    input_reactive$prophet_country <- NULL
    input_reactive$dt_calibration <- NULL
    if (exists("input_reactive$calib_data") == FALSE) {
      input_reactive$calib_data <- NULL
    }
    if ((isolate(input$min_date_model_build < isolate(input$max_date_model_build))) &
      (is.null(isolate(input$prophet_enable_checkbox)) == FALSE) &
      ((isTRUE(isolate(input$enable_calibration)) && (is.null(isolate(input$calibration_file)) == FALSE) &
        (length(intersect(c("channel", "liftStartDate", "liftEndDate", "liftAbs"), colnames(input_reactive$calib_data))) == 4)) ||
        (isTRUE(isolate(input$enable_calibration)) == FALSE))
    ) {
      input_reactive$nevergrad_algo <- "TwoPointsDE"
      input_reactive$window_start <- isolate(input$min_date_model_build)
      input_reactive$window_end <- isolate(input$max_date_model_build)
      if (isolate(input$prophet_enable_checkbox) == TRUE) {
        input_reactive$prophet_country <- isolate(input$country)
        if (isolate(input$prophet_option_trend) == TRUE) {
          input_reactive$prophet_vars <- c(input_reactive$prophet_vars, "trend")
        }
        if (isolate(input$prophet_option_holiday) == TRUE) {
          input_reactive$prophet_vars <- c(input_reactive$prophet_vars, "holiday")
        }
        if (isolate(input$prophet_option_season) == TRUE) {
          input_reactive$prophet_vars <- c(input_reactive$prophet_vars, "season")
        }
        if (isolate(input$prophet_option_weekday) == TRUE) {
          input_reactive$prophet_vars <- c(input_reactive$prophet_vars, "weekday")
        }
        if (isolate(input$prophet_option_trend) == TRUE) {
          input_reactive$prophet_signs <- c(input_reactive$prophet_signs, isolate(input$prophet_sign_trend))
        }
        if (isolate(input$prophet_option_holiday) == TRUE) {
          input_reactive$prophet_signs <- c(input_reactive$prophet_signs, isolate(input$prophet_sign_holiday))
        }
        if (isolate(input$prophet_option_season) == TRUE) {
          input_reactive$prophet_signs <- c(input_reactive$prophet_signs, isolate(input$prophet_sign_season))
        }
        if (isolate(input$prophet_option_weekday) == TRUE) {
          input_reactive$prophet_signs <- c(input_reactive$prophet_signs, isolate(input$prophet_sign_weekday))
        }
      }
      if (isTRUE(isolate(input$enable_calibration)) && (is.null(isolate(input$calibration_file)) == FALSE) &
        (length(intersect(c("channel", "liftStartDate", "liftEndDate", "liftAbs"), colnames(input_reactive$calib_data))) == 4)) {
        input_reactive$dt_calibration <- input_reactive$calib_data
        input_reactive$dt_calibration$liftStartDate <- as.Date(get("liftStartDate"), isolate(input$calib_date_format_var))
        input_reactive$dt_calibration$liftEndDate <- as.Date(get("liftEndDate"), isolate(input$calib_date_format_var))
        input_reactive$dt_calibration$liftStartDate <- as.Date(gsub("00", "20", input_reactive$dt_calibration$liftStartDate))
        input_reactive$dt_calibration$liftEndDate <- as.Date(gsub("00", "20", input_reactive$dt_calibration$liftEndDate))
      }

      msg <- "Input_success - Please click anywhere on the screen to proceed"
      showModal(modalDialog(title = msg, easyClose = TRUE, footer = NULL))
      message(msg)
    } else {
      msg <- "Input failed. Please ensure all fields have proper input per tooltip guiance and try again"
      showModal(modalDialog(title = msg, easyClose = TRUE, footer = NULL))
      message(msg)
    }
  })

  observeEvent(input$run_model, {
    message(">>> Preparing to run model...")
    input_reactive$iterations <- input$set_iter
    input_reactive$adstock <- input$adstock_selection
    input_reactive$trials <- input$set_trials

    withCallingHandlers({
      shinyjs::html("model_gen_text", "")
      if (!dir.exists(paste0(input$dest_folder, "plots"))) {
        dir.create(file.path(paste0(input$dest_folder, "plots")))
      }
      input_reactive$robyn_object <- paste0(input$dest_folder, "/plots")
      input_reactive$robyn_json <- paste0(input_reactive$robyn_object, "/robyn.json")
      # input_reactive <- reactiveValuesToList(input_reactive, all.names = TRUE)
      # saveRDS(input_reactive, file = "input_reactive.RDS")
      # input_reactive <- readRDS("input_reactive.RDS")
      message("Creating InputCollect...")
      input_reactive$InputCollect <- tryCatch({
        robyn_inputs(
          dt_input = input_reactive$dt_input,
          dt_holidays = input_reactive$holiday_data,
          adstock = input_reactive$adstock,
          calibration_input = input_reactive$calib_data,
          date_var = input_reactive$date_var,
          dep_var = input_reactive$dep_var,
          dep_var_type = input_reactive$dep_var_type,
          prophet_vars = input_reactive$prophet_vars,
          prophet_signs = input_reactive$prophet_signs,
          prophet_country = input_reactive$prophet_country,
          context_vars = input_reactive$context_vars,
          context_signs = input_reactive$context_signs,
          paid_media_vars = input_reactive$paid_media_vars,
          paid_media_signs = input_reactive$paid_media_signs,
          paid_media_spends = input_reactive$paid_media_spends,
          organic_vars = input_reactive$organic_vars,
          organic_signs = input_reactive$organic_signs,
          factor_vars = input_reactive$factor_vars,
          window_start = input_reactive$window_start,
          window_end = input_reactive$window_end,
          hyperparameters = input_reactive$hyperparameters)
      },
      error = function(e) {
        message("ERROR: ", e$message)
        showNotification(e$message, duration = NULL)
        return(NULL)
      })
      print(input_reactive$InputCollect)
    })

    tryCatch(
      withCallingHandlers(
        {
          shinyjs::html("model_gen_text", "")
          # plots will be saved in the same folder as robyn_object
          input_reactive$OutputCollect <- robyn_run(
            InputCollect = input_reactive$InputCollect,
            iterations = input_reactive$iterations,
            trials = input_reactive$trials,
            outputs = TRUE,
            csv_out = "pareto",
            clusters = TRUE,
            ui = TRUE
            )
          showModal(modalDialog(
            title = "Models Generated Succesfully - Please proceed to the Model Selection Tab",
            easyClose = TRUE,
            footer = NULL
          ))
        },
        message = function(m) {
          shinyjs::html(id = "model_gen_text", html = paste0(m$message, "<br>"), add = TRUE)
        }
      ),
      error = function(e) {
        showNotification(e$message, duration = NULL)
      }
    )
  })

  #############################Model Selection tab server functionality ##################################

  observeEvent(input$pareto_front_popover, {
    showModal(modalDialog(
      title = "Pareto Front",
      HTML(paste(paste0("In Robyn, essentially what we are trying to do is create a large quantity of a large quantity of gradient-free ", a("evolutionary-optimization algorithm", href = "https://facebookresearch.github.io/nevergrad/", target = "_blank"), " model solutions for ", a("pareto-optimal", href = "https://en.wikipedia.org/wiki/Pareto_efficiency", target = "_blank"), " model selection using three objective functions (or two in the case you are not calibrating your results with experimental data). "),
        "The first of these is <b>NRMSE or Normalized Root Mean Square Error</b>. NRMSE is equivalent to the RMSE / mean(observed). In other words, it is a measure of how much error there is between the observed values vs. what the model predicts the value to be. Naturally a high error is worse than a low error, and the closer to 0 the better.",
        "The second of these is <b>Decomp.RSSD or Decomposition Root Sum of Squared Distance</b>. This metric in essence is measuring the distance between the share of the media spend, and the share of effect per the model for the paid media variables. In this sense, Decomp.RSSD is more a measure of quality of the model/business logic since we would expect to disregard models where the results were extremely different than the levels of spend we currently use on certain channels. For example, if the share of spend for a channel was 10% but the share of effect was 90%, that would be concerning and a result marketers would likely not believe.",
        "The third of these is <b>MAPE.lift or the Mean Absolute Percent Error vs. your experimental Calibration data</b>. Calibrating your model with experimental data is an important way to ensure believability of the model and alignment with ground-truth. As such, minimizing the error against this data for the model helps us select solutions that align with that ground-truth best.",
        "In the chart below, you will see a large number of points and a few lines. Each one of these points is an evolutionary algorithm optimised solution, and each solution that occurs on one of the three lines or Pareto-Fronts is a Pareto-Optimal solution. Since we are minimizing 2 or 3 loss functions it can be difficult to make the conclusion that a difference in any one of the loss functions is more important than the others, so best practices would be to dig into a number of the models along the pareto fronts to identify which suits your business best.",
        "We will talk more about how to choose a model that makes sense, and provide some proactive guidance after you select a model solID to investigate. A number of explanatory charts will appear, that will help us identify models that make the most sense.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$initiate_tables, {
    output$pParFront <- renderPlot({
      input_reactive$OutputCollect$UI$pParFront
    })

    output$plots_folder <- renderUI({
      textInput("folder", label = "Directory containing plots", value = input_reactive$OutputCollect$plot_folder)
    })


    output$pareto_front_tbl <- renderDataTable({
      dat <- input_reactive$OutputCollect$clusters$data[input_reactive$OutputCollect$clusters$data$top_sol == TRUE, ]
      datatable(dat, rownames = FALSE, options = list(scrollX = TRUE, scrollY = 200, paging = FALSE, sDom = "t"))
    })

    output$model_selection_info <- renderTable({
      req(input$plot_click)
      nearPoints(input_reactive$OutputCollect$UI$pParFront$data,
        input$plot_click,
        xvar = "nrmse", yvar = "decomp.rssd", maxpoints = 1
      )[, "solID"]
    })
  })

  observeEvent(input$save_model, {
    if ((is.null(isolate(input$plot)) == FALSE) && (input$plot %in% input_reactive$OutputCollect$resultHypParam$solID)) {
      tryCatch(
        robyn_write(select_model = input$plot, InputCollect = input_reactive$InputCollect, OutputCollect = input_reactive$OutputCollect),
        error = function(e) {
          showNotification(e$message, duration = NULL)
        }
      )
      showModal(modalDialog(
        title = paste0("solID - ", input$plot, " saved successfully"),
        easyClose = TRUE,
        footer = NULL
      ))
    } else {
      showModal(modalDialog(
        title = "Either no solID entered, or solID does not exist. Please try again.",
        easyClose = TRUE,
        footer = NULL
      ))
    }
  })

  observeEvent(input$load_charts, {
    #### OUTPUT RECOMMENDATIONS MESSAGING ############

    # overall error messaging
    plotMediaShareLoop <- input_reactive$OutputCollect$xDecompAgg[(input_reactive$OutputCollect$xDecompAgg$solID == input$plot & input_reactive$OutputCollect$xDecompAgg$rn %in% input_reactive$paid_media_vars), ]

    rsq_train_plot <- round(unique(plotMediaShareLoop$rsq_train), 4)
    nrmse_plot <- round(unique(plotMediaShareLoop$nrmse), 4)
    decomp_rssd_plot <- round(unique(plotMediaShareLoop$decomp.rssd), 4)
    mape_lift_plot <- round(unique(plotMediaShareLoop$mape), 4)

    gen_message <- paste(
      paste0("The first metric we use to determine the fit of the model is ", a("R-Squared value", href = "https://en.wikipedia.org/wiki/Coefficient_of_determination", target = "_blank")),
      "At a high level, R-squared measures the proportion of the variance in the dependent variable that is explained by changes in the independent variables. For example, an R-squared value of 0.82 would mean that 82% of the variation in the dependent variable is explained by the independent variables.",
      paste0("In solID - ", input$plot, " the <b>R-squared value is - ", rsq_train_plot, "</b>. While it is often the case that a modeler should strive for a high R-squared, there is no exact goal-value that it needs to be above in order to be an acceptable model, but a common rule of thumb is less than 0.8 is not good, between 0.8 and 0.9 may be the best possible in some cases, but r-squared higher than 0.9 is ideal. That said, models with low R-squared values can likely be improved upon. Common ways to address this are through having a more comprehensive set of independent variables. In other words, there are opportunities to split up larger paid media channels, and include additional baseline (non-media) variables that may explain portions of the outcomes. Consider exploring the guidance on baseline and paid media variables included on the Data Input tab to find some ideas."),
      sep = "<br><br>"
    )

    calib_message <- paste(
      paste0("Since we are calibrating our model with data from randomized control trial or geo based experiments, Robyn also aims to minimize the absolute error of the channels represented in these experiements during the period of the experiment. Minimizing this error can significantly help the believability of the model. The metric we use to calculate this error is mape.lift, or ", a("Mean Absolute Percent Error of Lift/Experimental Results", href = "https://en.wikipedia.org/wiki/Mean_absolute_percentage_error", target = "_blank")),
      paste0("Similar to other measures of error, we look at the differences between observed and predicted values and aim to minimize that. In this case, the <b>Mape.lift is - ", mape_lift_plot, "</b>. The closer this result is to zero, the better as that indicates no difference between the calibration data and the model output."),
      sep = "<br><br>"
    )

    input_reactive$final_gen_message <- ifelse(mape_lift_plot != 0, paste(gen_message, calib_message, sep = "<br><br>"), gen_message)


    # plot 1 messaging#

    plotWaterfall <- input_reactive$OutputCollect$xDecompAgg[input_reactive$OutputCollect$xDecompAgg$solID == input$plot, ]

    plotWaterfallLoop <- plotWaterfall[order(plotWaterfall$xDecompPerc), ]
    plotWaterfallLoop$end <- cumsum(plotWaterfallLoop$xDecompPerc)
    plotWaterfallLoop$end <- 1 - plotWaterfallLoop$end
    plotWaterfallLoop$start <- lag(plotWaterfallLoop$end, n = 1)
    plotWaterfallLoop$id <- seq_along(plotWaterfallLoop[, 1])
    plotWaterfallLoop$sign <- as.factor(ifelse(plotWaterfallLoop$xDecompPerc >= 0, "pos", "neg"))

    high_share <- plotWaterfallLoop[(plotWaterfallLoop$xDecompMeanNon0Perc > 0.4 & plotWaterfallLoop$rn %in% input_reactive$context_vars), ]
    low_share <- plotWaterfallLoop[abs(plotWaterfallLoop$xDecompMeanNon0Perc) < 0.01, ]
    negative <- plotWaterfallLoop[plotWaterfallLoop$sign == "neg" & (plotWaterfallLoop$rn %in% c("season", "weekday", "holiday", "trend", "(Intercept)") == FALSE), ]
    paid_media_vars <- plotWaterfallLoop[is.na(plotWaterfallLoop$total_spend) == FALSE, ]

    generic_message <- "The first plot looks at the overall decomposition of the model. The larger the bar, the larger the proportion of the effect is explained by changes in that particular variable. For instance, if Facebook_I had a share of 25% of the effect, then we would say that on average, Facebook media is causing 25% of the dependent variable on a given time period. This will change of course when looking at different days and when considering baseline variables as well such as seasonality/trend."

    high_share_message <- ifelse(length(high_share$rn) > 0,
      paste("<b>Consideration 1 - High Share of Effect.</b> The variable(s) - ",
        paste(high_share$rn, collapse = ", "),
        " are showing that they have a share of the effect greater than 40%. If this is a non paid-media variable, consider investigating further whether this variable makes sense to include or whether this result makes sense. A case that may occur is a baseline variable that is actually a subset of the dependent variable, and thus should not be used to predict the independent variable as it could be misrepresenting results. If the share of effect vs. share of media spend have a high difference as well that may be concerning.",
        sep = ""
      ), ""
    )

    low_share_message <- ifelse(length(low_share$rn) > 0,
      paste("<b>Consideration 2 - Low/No Share of Effect.</b> The variable(s) - ",
        paste(low_share$rn, collapse = ", "),
        ", are showing that they have a share of the effect between -0.01% and 0.01%. In other words, they have very limited effect. If this seems highly unlikely please investigate further or consider choosing a solution that makes more business sense.",
        sep = ""
      ), ""
    )

    negative_message <- ifelse(length(negative$rn) > 0,
      paste("<b>Consideration 3 - Negative Effect.</b> The variable(s) - ",
        paste(negative$rn, collapse = ", "),
        ", are showing that they have a negative impact on effect. If this seems highly unlikely please investigate further",
        sep = ""
      ), ""
    )

    tot_paid_media_resp_message <- paste("<b>Consideration 4 - Low Paid Media Effect.</b> The Paid Media variable(s) - ",
      paste(paid_media_vars$rn, collapse = ", "),
      " represent", 100 * round(sum(paid_media_vars$xDecompMeanNon0Perc), 2),
      "% of the total effect/dependent variable. If this seems too low, consider whether there may be some inappropriate baseline variables included, or there is not enough paid media data included.",
      "If this seems too high, consider adding additional baseline variables that may further explain business performance. Throughout these explanatory tabs there should be some additional ideas to investigate.",
      "Depending on your marketing spend, it is unlikely that this value should be below 10% or above 90%, but not impossible.",
      sep = " "
    )

    intercept_message <- ifelse(plotWaterfallLoop[plotWaterfallLoop$rn == "(Intercept)", ]$xDecompMeanNon0Perc > 0.3,
      "<b>Consideration 5 - Large Intercept Effect</b>. - The Intercept is contributing a significant amount towards the dependent variable. Consider adding in additional baseline variables that may help better explain the variation in the dependent variable.",
      ""
    )

    no_consid_message <- ifelse(high_share_message == "" & low_share_message == "" & negative_message == "" & tot_paid_media_resp_message == "" & intercept_message == "",
      "No specific consideration callouts in this section, please proceed to further considerations.", ""
    )

    plot1_message <- c(generic_message, high_share_message, low_share_message, negative_message, tot_paid_media_resp_message, intercept_message, no_consid_message)
    input_reactive$plot1_message <- paste(plot1_message[which(plot1_message != "")], collapse = "<br><br>")

    # Plot2_message

    plot2_tab <- input_reactive$OutputCollect$xDecompVecCollect[input_reactive$OutputCollect$xDecompVecCollect$solID == input$plot, ]
    plot2_tab$error <- (plot2_tab$depVarHat / plot2_tab$dep_var) - 1
    plot2_tab$error_abs <- abs(plot2_tab$error)
    plot2_tab_top10 <- plot2_tab[order(plot2_tab$error_abs, decreasing = TRUE), ][1:10, ]

    plot2_tab$month <- floor_date(plot2_tab$ds, unit = "month")
    plot2_monthly <- plot2_tab %>%
      group_by(month) %>%
      summarize(err = sum(abs(get("dep_var") / get("depVarHat") - 1)) / n())
    plot2_monthly_top10 <- plot2_monthly[order(plot2_monthly$err, decreasing = TRUE), ][1:10, ]

    plot2_tab$year <- floor_date(plot2_tab$ds, unit = "year")
    plot2_yearly <- plot2_tab %>%
      group_by(year) %>%
      summarize(err = sum(abs(get("dep_var") / get("depVarHat") - 1)) / n())
    plot2_yearly_top10 <- plot2_yearly[order(plot2_yearly$err, decreasing = TRUE), ]

    plot2_message_1 <- paste("When considering the fit of your model, it can be useful to see how over time the model fit looks. For example,",
      "you may uncover that specific time periods (e.g. promotional periods) have high errors. In that case you could consider adding a baseline variable or splitting media in a way that better represents those periods. ",
      "Another case would be when the model does not fit well for multiple time periods. This may be commonly seen around March 2020, when COVID caused huge changes to supply and demand globally overnight.",
      paste0("In this case, consider reading the article ", a("Adjusting MMM for Unexpected Events", href = "https://www.facebook.com/business/news/insights/5-ways-to-adjust-marketing-mix-models-for-unexpected-events", target = "_blank")),
      paste0("For solID - ", input$plot, " the below readouts will show the time periods where the model had this largest errors vs. its predicted value. As you parse through these consider how you may be able to better account for underlying factors that correspond to these."),
      sep = "<br><br>"
    )

    tbl_html_funct_2col <- function(df, headers = c("date", "absolute_error")) {
      df <- as.data.frame(df)
      html_msg_1 <- '<table style="width:50%">'
      html_msg_2 <- paste0("<tr>", paste("<th>", headers, "</th>", collapse = "", sep = ""), "</tr>")
      html_rows <- ""
      for (i in seq_along(df[, 1])) {
        html_rows <- paste0(
          html_rows,
          paste0("<tr><td>", df[i, headers[1]], "</td><td>", paste0(100 * round(df[i, headers[2]], 3), "%"), "</td></tr>")
        )
      }
      html_msg_4 <- "</table>"
      return(paste0(html_msg_1, html_msg_2, html_rows, html_msg_4))
    }

    input_reactive$plot2_message_2 <- paste(plot2_message_1, "<b>Indv. Time Periods with the largest error</b>",
      tbl_html_funct_2col(plot2_tab_top10, c("ds", "error")),
      "<b>Months with the largest error</b>",
      tbl_html_funct_2col(plot2_monthly_top10, c("month", "err")),
      "<b>Years with the largest error</b>",
      tbl_html_funct_2col(plot2_yearly_top10, c("year", "err")),
      "If there are periods/days that are seeing large errors but may be explainable by something concrete rather than the natural variation, consider adding a variable to describe that relationship to the model.",
      sep = "<br><br>"
    )

    # plot3_message

    input_reactive$plot3_message_1 <- paste("In Robyn, one of the variables that is being minimized is decomp.rssd, which is a measure of how far apart the share of paid media spend, and share of paid media effect are. In other words, we want to optimize away from models that have highly disparate spend & effect shares because it does not make logical sense for the business to dramatically change their historical spend patterns.",
      "In this chart, we examine for each paid media variable the average share of spend and the average share of effect as well as the <b>ROI which is calculated as (mean effect / mean spend).</b>",
      "If your dependent variable is revenue, this is straightforward. On average, if you spend an additional dollar on the media channel in question, you would get ROI dollars back in revenue. If your dependent variable is more along the lines of conversions, the ROI value is not as straightforward. In this case, the ROI can be interpreted as the average number of conversions generated for an additional dollar spent on that channel. In this case, it would make most sense that the value is between 0 and 1.",
      "For channels where the proportion of spend is very low, it may be more likely that ROIs reported are less believable, since they may not hold up as well through extrapolation. Generally, extrapolation beyond a certain % increase is not very reliable, so if your channel had 2% of spend increasing to 10% of spend is a 500% increase which is quite hard to extrapolate in a believable way since there is likely little data at that level of spend.",
      "One thing to remember - If your dependent variable corresponds to conversions and not revenue the ROI will be conversions per dollar spent, not revenue per dollar spent.",
      sep = "<br><br>"
    )

    # plot4_message
    plot4_message_1 <- paste("Cost-Response curves, Response Curves, or Diminishing Returns Curves are one of the key outputs of Marketing Mix Models. With them, Marketers can understand at what their optimal investment level in a given channel is, or when used in aggregate, how to make overall budget investment decisions.",
      "A key principle that these Response curves follow is the theory of diminishing returns. This means that at a certain point we expect the return on investment of ($X+1 - $X) < ($X - $X-1) or in other words, the profitability of any given channel will eventually reach a point where it is no longer an acceptable return on investment for a business. Typically, when ROI < 1.",
      "These Response Curves also have hyperparameters - Alpha and Gamma. These hyperparameters will effect the shape of the response curve. The higher the <b>Alpha</b> hyperparameter, the more S-shape the curve will be. The lower the Alpha, the more C-shape the response-curve will be. Traditionally C-shape was more used. while its simpler, it means the first dollar spent has always the highest marginal ROI, which is often not very intuitive. Using the hill-function, Robyn can switch to S-curves, meaning starting slow, and reaching higher marginal ROI somewhere in the middle, then saturating which is a more intuitive option. For <b>Gamma</b> hyperparameters, the higher the value the higher the inflection point will be, or in other words, the higher the level of investment that the channel will start hitting diminishing returns on investment. A lower Gamma means that the channel will reach a saturation point at a lower level of investment.",
      "Please look at the charts here to get a better understanding of how these response curves may behave. Shortly you will have response curves of your own generated!",
      sep = "<br><br>"
    )

    plot4_tbl_alphas <- input_reactive$OutputCollect$resultHypParam[input_reactive$OutputCollect$resultHypParam$solID == input$plot, ] %>% select(contains("alpha"))
    plot4_tbl_gammas <- input_reactive$OutputCollect$resultHypParam[input_reactive$OutputCollect$resultHypParam$solID == input$plot, ] %>% select(contains("gamma"))
    plot4_alphas_message <- paste(
      paste0("In solID - ", input$plot, " we see that the paid media variables have the following <b>Alpha values</b>"),
      paste0(colnames(plot4_tbl_alphas), " - ", round(plot4_tbl_alphas, 3), collapse = "<br>"),
      sep = "<br><br>"
    )
    plot4_gammas_message <- paste(
      paste0("In solID - ", input$plot, " we see that the paid media variables have the following <b>Gamma values</b>"),
      paste0(colnames(plot4_tbl_gammas), " - ", round(plot4_tbl_gammas, 3), collapse = "<br>"),
      sep = "<br><br>"
    )
    plot4_message_2 <- paste("Much of the analyst interpretation here is evaluating whether these curves look realistic. For example, curves that show no point of diminishing returns may make sense if the investment is still relatively small, or the channel is new, but as the investment continues to rise there should be some declines in efficiency. Additionally, continue to investigate media channels that appear to have very low effect compared to their spend level, or the reverse.",
      "Use your best judgement to determine what seems realistic here, and identify any trends across model solutions about individual media channels that seem to arise.",
      "One final point is to remember that the further you get from your historical mean spend, the higher the chance of performance that is different than what the response curves predict.",
      sep = "<br><br>"
    )
    input_reactive$plot_4_final_message <- paste(plot4_message_1, plot4_alphas_message, plot4_gammas_message, plot4_message_2, sep = "<br><br>")

    # plot5_message

    input_reactive$plot5_message_1 <- paste("In this plot we examine the adstock decay rates for each paid media channel. For a quick refresher, the adstock defines how much effect of media that occurs in time period X is carried over into time period X+1, since we know that not all impact from an advertisement has to occur on the day it is delivered.",
      "Much of the interpretation of this plot as well is determining whether or not these results make intuitive sense. For example, if results are showing that a channel carries over a very large amount of its effect, then that may something worth exploring further. Especially if that channel is one that we usually consider more lower funnel.",
      "What would be considered a high or low adstock will also change depending on the time granularity of your data, so be sure to consider that as well as you interpret these results.",
      sep = "<br><br>"
    )

    # plot6_message
    input_reactive$plot6_message_1 <- paste("In this plot we examine in a way similar to plot 2 Actual vs. Predicted results a plot of the measure of the residuals for each prediction with the size of the prediction on the x-axis. Given that we would hope our predicted values have an error of 0, we would also hope that our errors when comparing predicted values against the actual values should be randomly distributed around 0. If we notice areas where the residuals have a clear trend, that is cause for concern about the validity of the model, and may be representative of heteroskedasticity which may speak to the believability of the fit of the model in certain periods as being better than others in a biased way. An example could be a period where sales and spend drastically increase, and afterwards the residuals are much larger representing unequal variance of the errors. Consider addressing this using additional baseline/non-media variables.",
      "If you do see that the band around the blue average line here does not include the value 0 for portion of your results, it may be worth further exploring those data points to see if there is some bias in the model. If it is only for a period or two, then it is likely not too bad, but if there is significant differences over time then it would be cause for concern.",
      sep = "<br><br>"
    )

    # outputs
    output$model_output_expl_gen <- renderUI({
      fluidRow(
        column(
          width = 6,
          actionButton("model_output_gen_popover", label = "Interpreting Model Suitability Statistics", style = "info", size = "medium")
        )
      )
    })

    output$model_output_expl_1 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("model_output_plot_1_popover", label = "Interpreting plot #1 - Waterfall Decomposition", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("model_output_plot_2_popover", label = "Interpreting plot #2 - Actual vs. Predicted Results", style = "info", size = "medium"),
        )
      )
    })
    output$model_output_expl_2 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("model_output_plot_3_popover", label = "Interpreting plot #3 - Share of Spend vs. Share of Effect", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("model_output_plot_4_popover", label = "Interpreting plot #4 - Response Curves", style = "info", size = "medium"),
        )
      )
    })
    output$model_output_expl_3 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("model_output_plot_5_popover", label = "Interpreting plot #5 - Adstock decay rates", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("model_output_plot_6_popover", label = "Interpreting plot #6 - Fitted vs. Residuals", style = "info", size = "medium"),
        )
      )
    })
  })

  observeEvent(input$model_output_gen_popover, {
    showModal(modalDialog(
      title = "Interpreting Model Suitability Statistics",
      HTML(input_reactive$final_gen_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_1_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #1 - Waterfall Decomposition",
      HTML(input_reactive$plot1_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_2_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #2 - Actual vs. Predicted Results",
      HTML(input_reactive$plot2_message_2),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_3_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #3 - Share of Spend vs. Share of Effect",
      HTML(input_reactive$plot3_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_4_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #4 - Response Curves",
      HTML(input_reactive$plot_4_final_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_5_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #5 - Adstock decay rates",
      HTML(input_reactive$plot5_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$model_output_plot_6_popover, {
    showModal(modalDialog(
      title = "Interpreting plot #6 - Fitted vs. Residuals",
      HTML(input_reactive$plot6_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })



  observeEvent(input$load_charts, {
    output$load_selection_plot <- renderUI({
      withSpinner(plotOutput("model_selection_img"))
    })

    output$model_selection_img <- renderImage(
      {
        filename <- normalizePath(isolate(file.path(input$folder, paste(input$plot, ".png", sep = ""))))
        list(src = filename, width = "100%", height = "auto")
      },
      deleteFile = FALSE
    )
  })

  #################################### optimizer tab server functionality #######################################
  output$optimizer_plot <- renderPlot({
    NULL
  })

  output$expected_spend <- renderUI({
    if (input$opt_scenario == "max_response_expected_spend") {
      fluidRow(column(
        width = 2,
        numericInput("expected_spend_opt", label = h4(
          "Expected Spend in Optimization Period",
          tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
          actionButton("expected_spend_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), step = 1, min = 1, value = 100000)
      ))
    }
  })

  observeEvent(input$expected_spend_popover, {
    showModal(modalButton(
      label = "Expected Spend in Optimization Period",
      "Input your total expected spend for the optimization period in the same currency that the rest of your spend data is in."
    ))
  })

  output$expected_days <- renderUI({
    if (input$opt_scenario == "max_response_expected_spend") {
      fluidRow(column(
        width = 2,
        numericInput("expected_days_opt", label = h4(
          "Expected Days in Optimization Period",
          tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
          actionButton("expected_days_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), step = 1, min = 1, value = 90)
      ))
    }
  })

  observeEvent(input$expected_days_popover, {
    showModal(modalButton(
      label = "Expected Spend in Optimization Period",
      "Input your total expected days for the optimization period. One important nuance - Even if your data is weekly data the input in here should be in days."
    ))
  })

  observeEvent(input$opt_solid_pop, {
    showModal(modalButton(
      label = "Input your selected model's solution ID",
      HTML("In the previous model selection tab, you should have found a model solution solID that you are interested in tracking.")
    ))
  })

  observeEvent(input$opt_scen_pop, {
    showModal(modalButton(
      label = "Choose the Optimization Scenario",
      HTML("There are two options for the optimization scenario. <b>Max historical response</b>, which uses the same spend and amount of time as the historical periods, and <b>Max response expected spend</b>, where you can input a number of days and spend value for scenario planning.")
    ))
  })

  observeEvent(input$opt_sliders, {
    showModal(modalButton(
      label = "Setting and Understanding your Optimization Boundaries",
      HTML("As we think about reallocating budgets, it is important to set some boundaries so that we are not producing results that would be unreasonable to present to the team making budgetary decisions. Oftentimes, this is put in place as a proportion up or down by which you are willing to adjust an individual channels spend by in order to optimize. For each of the sliders below, lets say we set the boundaries to be 0.8 to 1.2. This would mean that we would be comfortable changing the mean spend per period of this channel to be anywhere between 80% or 120% times the mean spend during the historical period. As you think about what makes sense for your business, consider discussing with the marketing team what boundaries they would like to put in place here.")
    ))
  })

  output$sliders <- renderUI({
    lapply(seq_along(input_reactive$paid_media_spends), function(i) {
      sliderInput(paste0("medVar_", input_reactive$paid_media_spends[i]),
        label = div(style = "font-size:12px", input_reactive$paid_media_spends[i]),
        min = 0, max = 3, value = c(0.8, 1.2), step = 0.01
      )
    })
  })

  observeEvent(input$run_opt, {
    output$optimizer_plot <- renderPlot({
      channel_constr_low_list <- unlist(lapply(seq_along(input_reactive$paid_media_spends), function(i) {
        isolate(eval(parse(text = paste0("input$medVar_", input_reactive$paid_media_spends[i]))))[1]
      }))
      channel_constr_up_list <- unlist(lapply(seq_along(input_reactive$paid_media_spends), function(i) {
        isolate(eval(parse(text = paste0("input$medVar_", input_reactive$paid_media_spends[i]))))[2]
      }))
      tryCatch(input_reactive$optim_result <-
        isolate(robyn_allocator(
          InputCollect = input_reactive$InputCollect,
          OutputCollect = input_reactive$OutputCollect # input one of the model IDs in OutputCollect$allSolutions to get optimisation result
          , select_model = isolate(input$solID),
          scenario = isolate(input$opt_scenario) # c(max_historical_response, max_response_expected_spend)
          , expected_spend = isolate(input$expected_spend_opt) # specify future spend volume. only applies when scenario = "max_response_expected_spend"
          , expected_spend_days = isolate(input$expected_days_opt) # specify period for the future spend volumne in days. only applies when scenario = "max_response_expected_spend"
          , channel_constr_low = c(channel_constr_low_list) # must be between 0.01-1 and has same length and order as paid_media_vars
          , channel_constr_up = c(channel_constr_up_list) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
          , ui = TRUE
        )),
      error = function(e) {
        showNotification(e$message, duration = NULL)
      }
      )

      title <- paste0("Budget allocator optimum result for model ID ", isolate(input$solID))
      g <- ((input_reactive$optim_result$ui$p13) + (input_reactive$optim_result$ui$p12) / (input_reactive$optim_result$ui$p14) + plot_annotation(
        title = title, theme = theme(plot.title = element_text(hjust = 0.5))
      ))
      g
    })
    output$optimizer_tbl <- renderDataTable({
      as.data.frame(input_reactive$optim_result$dt_optimOut)
    })
  })
  #################################### Refresh server functionality #######################################

  output$data_refresh_dto <- renderDataTable({
    file_r <- input$data_file_refresh
    ext_r <- tools::file_ext(file_r$datapath)
    validate(need(ext_r == "csv", "Please upload a csv file"))
    input_reactive$refresh_data <- read.csv(file_r$datapath) # save dt_input as global var within server function
    # mmm_data_colnames <- colnames(mmm_data)
    datatable(head(input_reactive$refresh_data, n = 5L),
      options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t")
    )
  })
  colnames_reactive_r <- reactive({
    if (!is.null(input$data_file_refresh)) {
      file_r <- input$data_file_refresh
      ext_r <- tools::file_ext(file_r$datapath)
      validate(need(ext == "csv", "Please upload a csv file"))
      input_reactive$my_data_r <- read.csv(file_r$datapath)
    }
  })


  observeEvent(input$existingModel, {
    showModal(modalDialog(
      title = "Refreshing your previously built MMM",
      HTML(paste(
        'Before refreshing below. The refresh function is suitable for updating within "reasonable periods" Two common situations that are considered better to rebuild model:',
        "<br>", "1. Most data is new. If initial model has 100 weeks and 80 weeks new data is added in refresh, it might be better to rebuild the model",
        "<br>", "2. New variables are added"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$existing_model_for_refresh_popover, {
    showModal(modalDialog(
      title = 'Choose your previously Robyn object model (e.g., Robyn.RDS")',
      HTML(paste0(
        "The refresh function build a model using the new update data-set and based on your previously built model saved.",
        "The refresh function consumes the selected model of the initial build. It sets lower and upper bounds of hyperparameters for",
        "the new build around the selected hyperparameters of the previous build, stabilizes the effect of baseline variables",
        " across old and new builds and regulates the new effect share of media variables towards the latest spend level. ",
        "It returns aggregated result with all previous builds for reporting purpose and produces reporting plots."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$data_file_popover_r, {
    showModal(modalDialog(
      title = "Choosing a CSV File for your main dataset",
      HTML(paste0(
        "Upload your dataset here. The file must be of .csv type, and should contain at least a column for a date variable, ",
        "and an independent variable such as revenue or conversions. For an example of what this file could look like, see the de_simulated_data.csv ",
        "file in the ",
        a("github repository. ", href = "https://github.com/facebookexperimental/Robyn/blob/master/source/de_simulated_data.csv", target = "_blank"),
        "Additionally, for more detailed information on this step please refer to the ",
        a("step by step guide.", href = "https://facebookexperimental.github.io/Robyn/docs/step-by-step-guide#load-data", target = "_blank")
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$holiday_file_popover_e, {
    showModal(modalDialog(
      title = "Choose a file that contains holiday data for your region of choice",
      HTML(paste0(
        "Upload your dataset containing holiday data. If you do not have a separate file",
        ", in the github repository there is a file called holidays.csv that you can access ",
        "here containing holiday data going back and forward many years. Click ",
        a("here.", href = "https://github.com/facebookexperimental/Robyn/blob/master/source/holidays.csv"),
        " If you do have your own holiday file, ensure the formatting is the same as this file."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_steps_popover, {
    showModal(modalDialog(
      title = "Setting up your refresh steps number",
      HTML(paste(
        "Refresh steps number is an integer. It controls how many time-units the refresh model build move forward.",
        "For example, refresh steps = 4 on weekly data, means the start & end dates of modelling period move",
        "forward 4 weeks."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_mode_e, {
    showModal(modalDialog(
      title = "Choose the refresh mode",
      HTML(paste(
        'Refresh mode include options of "auto" and "manual". In auto mode, the refresh function builds refresh',
        "models with given refresh steps (you choose above) repeatedly until there is no more data available. In manual mode,",
        "the refresh function only moves forward the refresh steps only once"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_iters_popover, {
    showModal(modalDialog(
      title = "Setting Iteration Count per Trial for Evolutionary Algorithm",
      HTML(paste(
        "1000 iterations per Trial is the recommended value. ",
        "Rule of thumb is, the more new data added, the more iterations needed",
        "More iterations will require more computation time, so use your judgement",
        "to decide what is best. Geometric adstock + 2000 iterations + 5 trials with ",
        "6 cores takes about an hour to run, Weibull adstocks will take at least double that."
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_trials_popover, {
    showModal(modalDialog(
      title = "Setting Trial Count for Evolutionary Algorithm",
      HTML("Trials per refresh. Defaults to 5 trials."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$dest_folder_refresh_popover, {
    showModal(modalDialog(
      title = "Folder where explanatory model plots will output",
      HTML(paste0(
        'Customize sub path to save plots.For example, Sub-folder = "refresh".',
        "<br>", "The default folder is your initial plot folder (E.g., ~/Documents/GitHub/robynUI_private)"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_pareto_front_popover, {
    showModal(modalDialog(
      title = "Making sense of your model solutions",
      HTML(paste(paste0("In Robyn, essentially what we are trying to do is create a large quantity of a large quantity of gradient-free ", a("evolutionary-optimization algorithm", href = "https://facebookresearch.github.io/nevergrad/", target = "_blank"), " model solutions for ", a("pareto-optimal", href = "https://en.wikipedia.org/wiki/Pareto_efficiency", target = "_blank"), " model selection using three objective functions (or two in the case you are not calibrating your results with experimental data). "),
        "The first of these is <b>NRMSE or Normalized Root Mean Square Error</b>. NRMSE is equivalent to the RMSE / mean(observed). In other words, it is a measure of how much error there is between the observed values vs. what the model predicts the value to be. Naturally a high error is worse than a low error, and the closer to 0 the better.",
        "The second of these is <b>Decomp.RSSD or Decomposition Root Sum of Squared Distance</b>. This metric in essence is measuring the distance between the share of the media spend, and the share of effect per the model for the paid media variables. In this sense, Decomp.RSSD is more a measure of quality of the model/business logic since we would expect to disregard models where the results were extremely different than the levels of spend we currently use on certain channels. For example, if the share of spend for a channel was 10% but the share of effect was 90%, that would be concerning and a result marketers would likely not believe.",
        "The third of these is <b>MAPE.lift or the Mean Absolute Percent Error vs. your experimental Calibration data</b>. Calibrating your model with experimental data is an important way to ensure believability of the model and alignment with ground-truth. As such, minimizing the error against this data for the model helps us select solutions that align with that ground-truth best.",
        "In the chart below, you will see a large number of points and a few lines. Each one of these points is an evolutionary algorithm optimised solution, and each solution that occurs on one of the three lines or Pareto-Fronts is a Pareto-Optimal solution. Since we are minimizing 2 or 3 loss functions it can be difficult to make the conclusion that a difference in any one of the loss functions is more important than the others, so best practices would be to dig into a number of the models along the pareto fronts to identify which suits your business best.",
        "We will talk more about how to choose a model that makes sense, and provide some proactive guidance after you select a model solID to investigate. A number of explanatory charts will appear, that will help us identify models that make the most sense.",
        sep = "<br><br>"
      )),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  output$hol_refresh_dto <- renderDataTable({
    hol_file_r <- input$holiday_file_refresh
    hol_ext_r <- tools::file_ext(hol_file_r$datapath)
    validate(need(hol_ext_r == "csv", "Please upload a csv file"))
    input_reactive$holiday_data_r <- read.csv(hol_file_r$datapath) # reads in the file
    datatable(head(input_reactive$holiday_data_r, n = 5L), options = list(scrollX = TRUE, scrollCollapse = TRUE, lengthChange = FALSE, sDom = "t"))
  })

  observeEvent(input$refresh_run, {
    # RDS path to data file
    model_file_json <- input$existing_model_for_refresh
    if (!exists("model_file_json")) stop("Must specify robyn_object")
    # check_robyn_object(model_file_RDS)
    if (!file.exists(model_file_json)) {
      stop("File does not exist or is somewhere else. Check: ", model_file_json)
    }

    input_reactive$time_steps <- input$refresh_steps
    input_reactive$mode_refresh <- input$refresh_mode
    input_reactive$iterations_refresh <- input$refresh_iters
    input_reactive$trials_refresh <- input$refresh_trials
    input_reactive$plot_folder_refresh <- input$dest_folder_refresh
    input_reactive$robyn_object_refresh <- model_file_json
    input_reactive$holiday_data_r$ds <- as.Date(input_reactive$holiday_data_r$ds)

    tryCatch(
      withCallingHandlers(
        {
          message("Preparing to run refresh model...")
          shinyjs::html("refresh_model_gen_text", "")
          input_reactive$OutputCollect <- robyn_refresh(input_reactive$robyn_object_refresh,
            dt_input = input_reactive$refresh_data,
            dt_holidays = input_reactive$holiday_data_r,
            plot_folder_sub = input_reactive$plot_folder_refresh,
            refresh_steps = input_reactive$time_steps,
            refresh_mode = input_reactive$mode_refresh,
            refresh_iters = input_reactive$iterations_refresh,
            refresh_trials = input_reactive$trials_refresh,
            plot_pareto = TRUE,
            ui = T
          )
          showModal(modalDialog(
            title = "Models Generated Succesfully - Please proceed to the Model Selection Tab",
            easyClose = TRUE,
            footer = NULL
          ))
        },
        message = function(m) {
          shinyjs::html(id = "refresh_model_gen_text", html = paste0(m$message, "<br>"), add = TRUE)
        }
      ),
      error = function(e) {
        showNotification(e$message, duration = NULL)
      }
    )
  })


  #############################Refresh Model Selection tab server functionality ##################################

  observeEvent(input$refresh_load_models, {
    input_reactive$refreshCounter <- length(input_reactive$OutputCollect) - 1
    output$refresh_pParFront <- renderPlot({
      input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$UI$pParFront
    })

    output$refresh_plots_folder <- renderUI({
      textInput("refresh_plots_folder", label = "Directory containing refresh plots", value = input_reactive$OutputCollect$listRefresh1$OutputCollect$plot_folder)
    })


    output$ref_pareto_front_tbl <- renderDataTable({
      dat <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$clusters$data[input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$clusters$data$top_sol == TRUE, ]
      # dat$rsq_train <- round(dat$rsq_train, digits = 4)
      # dat <- dat[order(-rsq_train),]
      # dat <- dat[,c('solID','mape','nrmse','decomp.rssd')]
      datatable(dat, rownames = FALSE, options = list(scrollX = TRUE, scrollY = 200, paging = FALSE, sDom = "t"))
    })

    output$refresh_model_selection_info <- renderTable({
      req(input$refresh_plot_click)
      nearPoints(input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$UI$pParFront$data,
        input$refresh_plot_click,
        xvar = "nrmse", yvar = "decomp.rssd", maxpoints = 1
      )[, c("solID", "rsq_train")]
    })
  })

  observeEvent(input$save_refresh_model, {
    # print(input_reactive$OutputCollect$listRefresh1$OutputCollect$resultHypParam$solID)
    print(input$refresh_plot)
    if ((is.null(isolate(input$refresh_plot)) == FALSE) && (input$refresh_plot %in% input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$resultHypParam$solID)) {
      robyn_object_refresh <- paste0(input$refresh_plots_folder, "/", gsub(":", ".", as.character(Sys.time())), "_solID_", input$refresh_plot, ".RDS")
      robyn_save(robyn_object = robyn_object_refresh, select_model = input$refresh_plot, InputCollect = input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect, OutputCollect = input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect)
      showModal(modalDialog(
        title = paste0("solID - ", input$refresh_plot, " saved successfully"),
        easyClose = TRUE,
        footer = NULL
      ))
    } else {
      showModal(modalDialog(
        title = "Either no refresh_solID entered, or refresh_solID does not exist. Please try again.",
        easyClose = TRUE,
        footer = NULL
      ))
    }
  })

  observeEvent(input$load_refresh_charts, {
    #### OUTPUT RECOMMENDATIONS MESSAGING ############

    # overall error messaging

    plotMediaShareLoop <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompAgg[
      (input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompAgg$solID == input$refresh_plot) &
        (input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompAgg$rn %in%
          input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_vars),
    ]

    rsq_train_plot <- round(unique(plotMediaShareLoop$rsq_train), 4)
    nrmse_plot <- round(unique(plotMediaShareLoop$nrmse), 4)
    decomp_rssd_plot <- round(unique(plotMediaShareLoop$decomp.rssd), 4)
    mape_lift_plot <- round(unique(plotMediaShareLoop$mape), 4)

    gen_message <- paste(
      paste0("The first metric we use to determine the fit of the model is ", a("R-Squared value", href = "https://en.wikipedia.org/wiki/Coefficient_of_determination", target = "_blank")),
      "At a high level, R-squared measures the proportion of the variance in the dependent variable that is explained by changes in the independent variables. For example, an R-squared value of 0.82 would mean that 82% of the variation in the dependent variable is explained by the independent variables.",
      paste0("In refresh_solID - ", input$plot, " the <b>R-squared value is - ", rsq_train_plot, "</b>. While it is often the case that a modeler should strive for a high R-squared, there is no exact goal-value that it needs to be above in order to be an acceptable model, but a common rule of thumb is less than 0.8 is not good, between 0.8 and 0.9 may be the best possible in some cases, but r-squared higher than 0.9 is ideal. That said, models with low R-squared values can likely be improved upon. Common ways to address this are through having a more comprehensive set of independent variables. In other words, there are opportunities to split up larger paid media channels, and include additional baseline (non-media) variables that may explain portions of the outcomes. Consider exploring the guidance on baseline and paid media variables included on the Data Input tab to find some ideas."),
      sep = "<br><br>"
    )

    calib_message <- paste(
      paste0("Since we are calibrating our model with data from randomized control trial or geo based experiments, Robyn also aims to minimize the absolute error of the channels represented in these experiements during the period of the experiment. Minimizing this error can significantly help the believability of the model. The metric we use to calculate this error is mape.lift, or ", a("Mean Absolute Percent Error of Lift/Experimental Results", href = "https://en.wikipedia.org/wiki/Mean_absolute_percentage_error", target = "_blank")),
      paste0("Similar to other measures of error, we look at the differences between observed and predicted values and aim to minimize that. In this case, the <b>Mape.lift is - ", mape_lift_plot, "</b>. The closer this result is to zero, the better as that indicates no difference between the calibration data and the model output."),
      sep = "<br><br>"
    )

    input_reactive$final_gen_message <- ifelse(mape_lift_plot != 0, paste(gen_message, calib_message, sep = "<br><br>"), gen_message)


    # plot 1 messaging#

    plotWaterfall <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompAgg[input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompAgg$solID == input$refresh_plot, ]

    plotWaterfallLoop <- plotWaterfall[order(plotWaterfall$xDecompPerc), ]
    plotWaterfallLoop$end <- cumsum(plotWaterfallLoop$xDecompPerc)
    plotWaterfallLoop$end <- 1 - plotWaterfallLoop$end
    plotWaterfallLoop$start <- lag(plotWaterfallLoop$end, n = 1)
    plotWaterfallLoop$id <- seq_along(plotWaterfallLoop[, 1])
    plotWaterfallLoop$sign <- as.factor(ifelse(plotWaterfallLoop$xDecompPerc >= 0, "pos", "neg"))

    high_share <- plotWaterfallLoop[(plotWaterfallLoop$xDecompMeanNon0Perc > 0.4 & plotWaterfallLoop$rn %in% input_reactive$context_vars), ]
    low_share <- plotWaterfallLoop[abs(plotWaterfallLoop$xDecompMeanNon0Perc) < 0.01, ]
    negative <- plotWaterfallLoop[plotWaterfallLoop$sign == "neg" & (plotWaterfallLoop$rn %in% c("season", "weekday", "holiday", "trend", "(Intercept)") == FALSE), ]
    paid_media_vars <- plotWaterfallLoop[is.na(plotWaterfallLoop$total_spend) == FALSE, ]

    generic_message <- "The first plot looks at the overall decomposition of the model. The larger the bar, the larger the proportion of the effect is explained by changes in that particular variable. For instance, if Facebook_I had a share of 25% of the effect, then we would say that on average, Facebook media is causing 25% of the dependent variable on a given time period. This will change of course when looking at different days and when considering baseline variables as well such as seasonality/trend."

    high_share_message <- ifelse(length(high_share$rn) > 0,
      paste("<b>Consideration 1 - High Share of Effect.</b> The variable(s) - ",
        paste(high_share$rn, collapse = ", "),
        " are showing that they have a share of the effect greater than 40%. If this is a non paid-media variable, consider investigating further whether this variable makes sense to include or whether this result makes sense. A case that may occur is a baseline variable that is actually a subset of the dependent variable, and thus should not be used to predict the independent variable as it could be misrepresenting results. If the share of effect vs. share of media spend have a high difference as well that may be concerning.",
        sep = ""
      ), ""
    )

    low_share_message <- ifelse(length(low_share$rn) > 0,
      paste("<b>Consideration 2 - Low/No Share of Effect.</b> The variable(s) - ",
        paste(low_share$rn, collapse = ", "),
        ", are showing that they have a share of the effect between -0.01% and 0.01%. In other words, they have very limited effect. If this seems highly unlikely please investigate further or consider choosing a solution that makes more business sense.",
        sep = ""
      ), ""
    )

    negative_message <- ifelse(length(negative$rn) > 0,
      paste("<b>Consideration 3 - Negative Effect.</b> The variable(s) - ",
        paste(negative$rn, collapse = ", "),
        ", are showing that they have a negative impact on effect. If this seems highly unlikely please investigate further",
        sep = ""
      ), ""
    )

    tot_paid_media_resp_message <- paste("<b>Consideration 4 - Low Paid Media Effect.</b> The Paid Media variable(s) - ",
      paste(paid_media_vars$rn, collapse = ", "),
      " represent", 100 * round(sum(paid_media_vars$xDecompMeanNon0Perc), 2),
      "% of the total effect/dependent variable. If this seems too low, consider whether there may be some inappropriate baseline variables included, or there is not enough paid media data included.",
      "If this seems too high, consider adding additional baseline variables that may further explain business performance. Throughout these explanatory tabs there should be some additional ideas to investigate.",
      "Depending on your marketing spend, it is unlikely that this value should be below 10% or above 90%, but not impossible.",
      sep = " "
    )

    intercept_message <- ifelse(plotWaterfallLoop[plotWaterfallLoop$rn == "(Intercept)", ]$xDecompMeanNon0Perc > 0.3,
      "<b>Consideration 5 - Large Intercept Effect</b>. - The Intercept is contributing a significant amount towards the dependent variable. Consider adding in additional baseline variables that may help better explain the variation in the dependent variable.",
      ""
    )

    no_consid_message <- ifelse(high_share_message == "" & low_share_message == "" & negative_message == "" & tot_paid_media_resp_message == "" & intercept_message == "",
      "No specific consideration callouts in this section, please proceed to further considerations.", ""
    )

    plot1_message <- c(generic_message, high_share_message, low_share_message, negative_message, tot_paid_media_resp_message, intercept_message, no_consid_message)
    input_reactive$plot1_message <- paste(plot1_message[which(plot1_message != "")], collapse = "<br><br>")

    # Plot2_message

    plot2_tab <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompVecCollect[input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$xDecompVecCollect$solID == input$refresh_plot, ]
    plot2_tab$ds <- as.Date(plot2_tab$ds)
    plot2_tab$error <- (plot2_tab$depVarHat / plot2_tab$dep_var) - 1
    plot2_tab$error_abs <- abs(plot2_tab$error)
    plot2_tab_top10 <- plot2_tab[order(plot2_tab$error_abs, decreasing = TRUE), ][1:10, ]

    plot2_tab$month <- floor_date(plot2_tab$ds, unit = "month")
    plot2_monthly <- plot2_tab %>%
      group_by(month) %>%
      summarize(err = sum(abs(get("dep_var") / get("depVarHat") - 1)) / n())
    plot2_monthly_top10 <- plot2_monthly[order(plot2_monthly$err, decreasing = TRUE), ][1:10, ]

    plot2_tab$year <- floor_date(plot2_tab$ds, unit = "year")
    plot2_yearly <- plot2_tab %>%
      group_by(year) %>%
      summarize(err = sum(abs(get("dep_var") / get("depVarHat") - 1)) / n())
    plot2_yearly_top10 <- plot2_yearly[order(plot2_yearly$err, decreasing = TRUE), ]

    plot2_message_1 <- paste("When considering the fit of your model, it can be useful to see how over time the model fit looks. For example,",
      "you may uncover that specific time periods (e.g. promotional periods) have high errors. In that case you could consider adding a baseline variable or splitting media in a way that better represents those periods. ",
      "Another case would be when the model does not fit well for multiple time periods. This may be commonly seen around March 2020, when COVID caused huge changes to supply and demand globally overnight.",
      paste0("In this case, consider reading the article ", a("Adjusting MMM for Unexpected Events", href = "https://www.facebook.com/business/news/insights/5-ways-to-adjust-marketing-mix-models-for-unexpected-events", target = "_blank")),
      paste0("For refresh_solID - ", input$refresh_plot, " the below readouts will show the time periods where the model had this largest errors vs. its predicted value. As you parse through these consider how you may be able to better account for underlying factors that correspond to these."),
      sep = "<br><br>"
    )

    tbl_html_funct_2col <- function(df, headers = c("date", "absolute_error")) {
      df <- as.data.frame(df)
      html_msg_1 <- '<table style="width:50%">'
      html_msg_2 <- paste0("<tr>", paste("<th>", headers, "</th>", collapse = "", sep = ""), "</tr>")
      html_rows <- ""
      for (i in seq_along(df[, 1])) {
        html_rows <- paste0(
          html_rows,
          paste0("<tr><td>", df[i, headers[1]], "</td><td>", paste0(100 * round(df[i, headers[2]], 3), "%"), "</td></tr>")
        )
      }
      html_msg_4 <- "</table>"
      return(paste0(html_msg_1, html_msg_2, html_rows, html_msg_4))
    }

    input_reactive$plot2_message_2 <- paste(plot2_message_1, "<b>Indv. Time Periods with the largest error</b>",
      tbl_html_funct_2col(plot2_tab_top10, c("ds", "error")),
      "<b>Months with the largest error</b>",
      tbl_html_funct_2col(plot2_monthly_top10, c("month", "err")),
      "<b>Years with the largest error</b>",
      tbl_html_funct_2col(plot2_yearly_top10, c("year", "err")),
      "If there are periods/days that are seeing large errors but may be explainable by something concrete rather than the natural variation, consider adding a variable to describe that relationship to the model.",
      sep = "<br><br>"
    )

    # plot3_message

    input_reactive$plot3_message_1 <- paste("In Robyn, one of the variables that is being minimized is decomp.rssd, which is a measure of how far apart the share of paid media spend, and share of paid media effect are. In other words, we want to optimize away from models that have highly disparate spend & effect shares because it does not make logical sense for the business to dramatically change their historical spend patterns.",
      "In this chart, we examine for each paid media variable the average share of spend and the average share of effect as well as the <b>ROI which is calculated as (mean effect / mean spend).</b>",
      "If your dependent variable is revenue, this is straightforward. On average, if you spend an additional dollar on the media channel in question, you would get ROI dollars back in revenue. If your dependent variable is more along the lines of conversions, the ROI value is not as straightforward. In this case, the ROI can be interpreted as the average number of conversions generated for an additional dollar spent on that channel. In this case, it would make most sense that the value is between 0 and 1.",
      "For channels where the proportion of spend is very low, it may be more likely that ROIs reported are less believable, since they may not hold up as well through extrapolation. Generally, extrapolation beyond a certain % increase is not very reliable, so if your channel had 2% of spend increasing to 10% of spend is a 500% increase which is quite hard to extrapolate in a believable way since there is likely little data at that level of spend.",
      "One thing to remember - If your dependent variable corresponds to conversions and not revenue the ROI will be conversions per dollar spent, not revenue per dollar spent.",
      sep = "<br><br>"
    )

    # plot4_message
    plot4_message_1 <- paste("Cost-Response curves, Response Curves, or Diminishing Returns Curves are one of the key outputs of Marketing Mix Models. With them, Marketers can understand at what their optimal investment level in a given channel is, or when used in aggregate, how to make overall budget investment decisions.",
      "A key principle that these Response curves follow is the theory of diminishing returns. This means that at a certain point we expect the return on investment of ($X+1 - $X) < ($X - $X-1) or in other words, the profitability of any given channel will eventually reach a point where it is no longer an acceptable return on investment for a business. Typically, when ROI < 1.",
      "These Response Curves also have hyperparameters - Alpha and Gamma. These hyperparameters will effect the shape of the response curve. The higher the <b>Alpha</b> hyperparameter, the more S-shape the curve will be. The lower the Alpha, the more C-shape the response-curve will be. Traditionally C-shape was more used. while its simpler, it means the first dollar spent has always the highest marginal ROI, which is often not very intuitive. Using the hill-function, Robyn can switch to S-curves, meaning starting slow, and reaching higher marginal ROI somewhere in the middle, then saturating which is a more intuitive option. For <b>Gamma</b> hyperparameters, the higher the value the higher the inflection point will be, or in other words, the higher the level of investment that the channel will start hitting diminishing returns on investment. A lower Gamma means that the channel will reach a saturation point at a lower level of investment.",
      "Please look at the charts here to get a better understanding of how these response curves may behave. Shortly you will have response curves of your own generated!",
      sep = "<br><br>"
    )

    plot4_tbl_alphas <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$resultHypParam[input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$resultHypParam$solID == input$refresh_plot, ] %>% select(contains("alpha"))
    plot4_tbl_gammas <- input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$resultHypParam[input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect$resultHypParam$solID == input$refresh_plot, ] %>% select(contains("gamma"))
    plot4_alphas_message <- paste(
      paste0("In refresh_solID - ", input$refresh_plot, " we see that the paid media variables have the following <b>Alpha values</b>"),
      paste0(colnames(plot4_tbl_alphas), " - ", round(plot4_tbl_alphas, 3), collapse = "<br>"),
      sep = "<br><br>"
    )
    plot4_gammas_message <- paste(
      paste0("In refresh_solID - ", input$refresh_plot, " we see that the paid media variables have the following <b>Gamma values</b>"),
      paste0(colnames(plot4_tbl_gammas), " - ", round(plot4_tbl_gammas, 3), collapse = "<br>"),
      sep = "<br><br>"
    )
    plot4_message_2 <- paste("Much of the analyst interpretation here is evaluating whether these curves look realistic. For example, curves that show no point of diminishing returns may make sense if the investment is still relatively small, or the channel is new, but as the investment continues to rise there should be some declines in efficiency. Additionally, continue to investigate media channels that appear to have very low effect compared to their spend level, or the reverse.",
      "Use your best judgement to determine what seems realistic here, and identify any trends across model solutions about individual media channels that seem to arise.",
      "One final point is to remember that the further you get from your historical mean spend, the higher the chance of performance that is different than what the response curves predict.",
      sep = "<br><br>"
    )
    input_reactive$plot_4_final_message <- paste(plot4_message_1, plot4_alphas_message, plot4_gammas_message, plot4_message_2, sep = "<br><br>")

    # plot5_message

    input_reactive$plot5_message_1 <- paste("In this plot we examine the adstock decay rates for each paid media channel. For a quick refresher, the adstock defines how much effect of media that occurs in time period X is carried over into time period X+1, since we know that not all impact from an advertisement has to occur on the day it is delivered.",
      "Much of the interpretation of this plot as well is determining whether or not these results make intuitive sense. For example, if results are showing that a channel carries over a very large amount of its effect, then that may something worth exploring further. Especially if that channel is one that we usually consider more lower funnel.",
      "What would be considered a high or low adstock will also change depending on the time granularity of your data, so be sure to consider that as well as you interpret these results.",
      sep = "<br><br>"
    )

    # plot6_message
    input_reactive$plot6_message_1 <- paste("In this plot we examine in a way similar to plot 2 Actual vs. Predicted results a plot of the measure of the residuals for each prediction with the size of the prediction on the x-axis. Given that we would hope our predicted values have an error of 0, we would also hope that our errors when comparing predicted values against the actual values should be randomly distributed around 0. If we notice areas where the residuals have a clear trend, that is cause for concern about the validity of the model, and may be representative of heteroskedasticity which may speak to the believability of the fit of the model in certain periods as being better than others in a biased way. An example could be a period where sales and spend drastically increase, and afterwards the residuals are much larger representing unequal variance of the errors. Consider addressing this using additional baseline/non-media variables.",
      "If you do see that the band around the blue average line here does not include the value 0 for portion of your results, it may be worth further exploring those data points to see if there is some bias in the model. If it is only for a period or two, then it is likely not too bad, but if there is significant differences over time then it would be cause for concern.",
      sep = "<br><br>"
    )

    # outputs
    output$ref_model_output_expl_gen <- renderUI({
      fluidRow(
        column(
          width = 6,
          actionButton("ref_model_output_gen_popover", label = "Interpreting Model Suitability Statistics", style = "info", size = "medium")
        )
      )
    })

    output$ref_model_output_expl_1 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("ref_model_output_plot_1_popover", label = "Interpreting plot #1 - Waterfall Decomposition", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("ref_model_output_plot_2_popover", label = "Interpreting plot #2 - Actual vs. Predicted Results", style = "info", size = "medium"),
        )
      )
    })
    output$ref_model_output_expl_2 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("ref_model_output_plot_3_popover", label = "Interpreting plot #3 - Share of Spend vs. Share of Effect", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("ref_model_output_plot_4_popover", label = "Interpreting plot #4 - Response Curves", style = "info", size = "medium"),
        )
      )
    })
    output$ref_model_output_expl_3 <- renderUI({
      fluidRow(
        column(
          width = 2,
          actionButton("ref_model_output_plot_5_popover", label = "Interpreting plot #5 - Adstock decay rates", style = "info", size = "medium"),
        ),
        column(
          width = 2, offset = 2,
          actionButton("ref_model_output_plot_6_popover", label = "Interpreting plot #6 - Fitted vs. Residuals", style = "info", size = "medium"),
        )
      )
    })
  })

  observeEvent(input$ref_model_output_gen_popover, {
    showModal(modalDialog(
      title = paste0("Model Suitability Statistics Explanations and Recommendations for refresh_solID - ", input$plot),
      HTML(input_reactive$final_gen_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_1_popover, {
    showModal(modalDialog(
      title = paste0("Considerations for the Decomposition of refresh_solID ", input$plot),
      HTML(input_reactive$plot1_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_2_popover, {
    showModal(modalDialog(
      title = "Considerations for interpreting Actual vs. Predicted Results",
      HTML(input_reactive$plot2_message_2),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_3_popover, {
    showModal(modalDialog(
      title = "Interpreting ROI and Share of Spend vs. Effect",
      HTML(input_reactive$plot3_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_4_popover, {
    showModal(modalDialog(
      title = paste0("Interpreting Response Curves of refresh_solID ", input$plot),
      HTML(input_reactive$plot_4_final_message),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_5_popover, {
    showModal(modalDialog(
      title =  paste0("Interpreting Adstock Decay Rates of refresh_solID ", input$plot),
      HTML(input_reactive$plot5_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$ref_model_output_plot_6_popover, {
    showModal(modalDialog(
      title = paste0("Interpreting Fitted Vs. Residual Plots of refresh_solID ", input$plot),
      HTML(input_reactive$plot6_message_1),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$load_refresh_charts, {
    output$ref_load_selection_plot <- renderUI({
      withSpinner(plotOutput("ref_model_selection_img"))
    })

    output$ref_model_selection_img <- renderImage(
      {
        filename <- normalizePath(isolate(file.path(input$refresh_plots_folder, paste(input$refresh_plot, ".png", sep = ""))))
        list(src = filename, width = "100%", height = "auto")
      },
      deleteFile = FALSE
    )
  })

  #################################### Refresh optimizer tab server functionality #######################################
  output$ref_optimizer_plot <- renderPlot({
    NULL
  })

  output$ref_expected_spend <- renderUI({
    if (input$refresh_opt_scenario == "max_response_expected_spend") {
      fluidRow(column(
        width = 2,
        numericInput("ref_expected_spend_opt", label = h4(
          "Expected Spend in Optimization Period",
          tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
          actionButton("expected_spend_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), step = 1, min = 1, value = 100000)
      ))
    }
  })

  observeEvent(input$ref_expected_spend_popover, {
    showModal(modalDialog(
      title = "Expected Spend in Optimization Period",
      HTML("Input your total expected spend for the optimization period in the same currency that the rest of your spend data is in."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  output$ref_expected_days <- renderUI({
    if (input$refresh_opt_scenario == "max_response_expected_spend") {
      fluidRow(column(
        width = 2,
        numericInput("ref_expected_days_opt", label = h4(
          "Expected Days in Optimization Period",
          tags$style(type = "text/css", "#q2 {vertical-align: top;}"),
          actionButton("expected_days_popover", label = "", icon = icon("question"), style = "info", size = "extra-small")
        ), step = 1, min = 1, value = 90)
      ))
    }
  })

  observeEvent(input$ref_expected_days_popover, {
    showModal(modalDialog(
      title = "Expected Days in Optimization Period",
      HTML("Input your total expected days for the optimization period. One important nuance - Even if your data is weekly data the input in here should be in days."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_opt_scen_pop, {
    showModal(modalDialog(
      title = "Choose the Optimization Scenario",
      HTML("There are two options for the optimization scenario. <b>Max historical response</b>, which uses the same spend and amount of time as the historical periods, and <b>Max response expected spend</b>, where you can input a number of days and spend value for scenario planning."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$re_opt_sliders, {
    showModal(modalDialog(
      title = "Setting and Understanding your Optimization Boundaries",
      HTML("As we think about reallocating budgets, it is important to set some boundaries so that we are not producing results that would be unreasonable to present to the team making budgetary decisions. Oftentimes, this is put in place as a proportion up or down by which you are willing to adjust an individual channels spend by in order to optimize. For each of the sliders below, lets say we set the boundaries to be 0.8 to 1.2. This would mean that we would be comfortable changing the mean spend per period of this channel to be anywhere between 80% or 120% times the mean spend during the historical period. As you think about what makes sense for your business, consider discussing with the marketing team what boundaries they would like to put in place here."),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  observeEvent(input$refresh_opt_solid_pop, {
    showModal(modalDialog(
      title = "Input your selected model's solID",
      HTML(paste0("In the previous model selection tab, you should have found a model solution solID that you are interested in tracking.")),
      easyClose = TRUE,
      footer = NULL
    ))
  })

  output$ref_sliders <- renderUI({
    lapply(seq_along(input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends), function(i) {
      sliderInput(paste0("medVar_", input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends[i]),
        label = div(style = "font-size:12px", input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends[i]),
        min = 0, max = 3, value = c(0.8, 1.2), step = 0.01
      )
    })
  })

  observeEvent(input$run_refresh_opt, {
    output$ref_optimizer_plot <- renderPlot({
      channel_constr_low_list <- unlist(lapply(seq_along(input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends), function(i) {
        isolate(eval(parse(text = paste0("input$medVar_", input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends[i]))))[1]
      }))
      channel_constr_up_list <- unlist(lapply(seq_along(input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends), function(i) {
        isolate(eval(parse(text = paste0("input$medVar_", input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect$paid_media_spends[i]))))[2]
      }))
      tryCatch(input_reactive$refresh_optim_result <-
        isolate(robyn_allocator(
          InputCollect = input_reactive$OutputCollect[[input_reactive$refreshCounter]]$InputCollect
          , OutputCollect = input_reactive$OutputCollect[[input_reactive$refreshCounter]]$OutputCollect # input one of the model IDs in OutputCollect$allSolutions to get optimisation result
          , select_model = isolate(input$ref_solID)
          , scenario = isolate(input$refresh_opt_scenario) # c(max_historical_response, max_response_expected_spend)
          , expected_spend = isolate(input$ref_expected_spend_opt) # specify future spend volume. only applies when scenario = "max_response_expected_spend"
          , expected_spend_days = isolate(input$ref_expected_days_opt) # specify period for the future spend volumne in days. only applies when scenario = "max_response_expected_spend"
          , channel_constr_low = c(channel_constr_low_list) # must be between 0.01-1 and has same length and order as paid_media_vars
          , channel_constr_up = c(channel_constr_up_list) # not recommended to 'exaggerate' upper bounds. 1.5 means channel budget can increase to 150% of current level
          , ui = TRUE
        )),
      error = function(e) {
        showNotification(e$message, duration = NULL)
      }
      )

      title <- paste0("Budget allocator optimum result for model ID ", isolate(input$refresh_solID))
      g <- ((input_reactive$refresh_optim_result$ui$p13) + (input_reactive$refresh_optim_result$ui$p12) / (input_reactive$refresh_optim_result$ui$p14) + plot_annotation(
        title = title, theme = theme(plot.title = element_text(hjust = 0.5))
      ))
      g
    })
    output$ref_optimizer_tbl <- renderDataTable({
      as.data.frame(input_reactive$refresh_optim_result$dt_optimOut)
    })
  })
}
