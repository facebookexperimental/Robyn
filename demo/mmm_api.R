library(lares) # Please update to latest dev or stable version
stopifnot(packageVersion("lares") >= "5.2.0")

### COMMENTS
# - Ads Insights limit: Calls within one hour = 60 + 400 * Number of Active ads - 0.001 * User Errors
# - Filtering allowed for MMM breakdown: campaign.id, campaign.name, adset.id, adset.name, country,
# region, dma, device_platform, publisher_platform, platform_position.
# - MMM Report: Ads data is available from April 01, 2020 onwards.

### Generate a taken from API Graph with ads_read permissions
# Api Graph: https://developers.facebook.com/tools/explorer
token <- "YOURTOKEN"

# To create a long-life token, you can setup an App and generate it with
# token <- fb_token(app_id = "AAAAAAAAAAA", app_secret = "XXXXXXXXXXXXX", token)

# Fetch all accounts for a specific Business ID with the API
# accts <- fb_accounts(token, business_id = "BBBBBBBBBB")

# Set which account ID you want to fetch the data from (note it must start with "act_")
account <- "act_1234567890"

# Request async report with a lot of data that can't be queried in a single run
async_report <- fb_insights(
  token = token,
  which = account,
  breakdowns = "mmm",
  start_date = "2023-05-01",
  end_date = "2023-05-15",
  time_increment = 1, # Daily data: 1, Weekly data: 7, Monthly data: "monthly"
  # filtering = dplyr::tibble(field = "country", operator = "IN", value = list("PE")), # Example
  fields = NULL, # Return default MMM fields
  report_level = "adset", # Only adset report_level is accepted for MMM
  async = TRUE # Only available in latest dev version
)

##### SOME API ERRORS YOU MIGHT ENCOUNTER:
# 1. There have been too many calls from this ad-account. Wait a bit and try again.
# For more info, please refer to https://developers.facebook.com/docs/graph-api/overview/rate-limiting.
# 2. Please reduce the amount of data you're asking for, then retry your request.
# 3. Missing permissions.
# 4. Result for MMM request contains total less than the minimum of 100000 impressions.

# Check status for the request report until it's finished (~live)
status <- fb_report_check(token, async_report$report_run_id, live = TRUE)

# Fetch all the results by small chunks (it may crash if too much data in your report; consider sleep param)
report <- fb_insights(
  token = token,
  which = async_report$report_run_id,
  fields = NULL, # No need to select fields when fetching a report
  paginate = TRUE, # First time: check setting to 5 maybe
  limit = 100, # 100 rows per pagination/batch
  quiet = FALSE # Show status messages
)

# Did we get ALL the data?
attr(report, "paging_done")

# If you reach the hourly limit, you can use the URL provided in the warning to continue later on
# or automatically pick it up from the (partial) data.frame returned
error_url <- attr(report, "paging")[["next"]]
report_append <- fb_process(error_url, paginate = TRUE)
attr(report_append, "paging_done")
# Then append: report <- rbind(report, report_append)

# Check the data
str(report)
data.frame(head(report))

#################################################################

# NO REGIONAL & CUSTOM DATA (given Robyn doesn't support panel data (DMA) yet)
# Keep in mind lots of breakdown fields can't be mixed
account <- "act_1234567890"
report_custom <- fb_insights(
  token = token,
  which = account,
  start_date = "2022-05-01",
  end_date = "2023-05-15",
  fields = c("account_id", "account_name",
             "campaign_id", "campaign_name",
             "objective", "optimization_goal",
             "adset_id", "adset_name",
             "impressions", "spend"),
  breakdowns = c("device_platform"),
  filtering = data.frame(
    field = "impressions",
    operator = "GREATER_THAN",
    value = 100
  ),
  report_level = "adset",
  time_increment = 1,
  limit = 100,
  paginate = TRUE
)

# Check the data
str(report_custom)
data.frame(head(report_custom))
