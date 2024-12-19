# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn Dataset: MMM Demo Data
#'
#' Simulated MMM data. Input time series should be daily, weekly or monthly.
#'
#' @family Dataset
#' @docType data
#' @usage data(dt_simulated_weekly)
#' @return data.frame
#' @format An object of class \code{"data.frame"}
#' \describe{
#'   \item{DATE}{Date}
#'   \item{revenue}{Daily total revenue}
#'   \item{tv_S}{Television}
#'   \item{ooh_S}{Out of home}
#'   \item{...}{...}
#' }
#' @examples
#' data(dt_simulated_weekly)
#' head(dt_simulated_weekly)
#' @return Dataframe. Contains simulated dummy dataset to test and run demo.
"dt_simulated_weekly"

# dt_input <- read.csv('data/de_simulated_data.csv')
# save(dt_input, file = "data/dt_input.RData", version = 2)
# dt_simulated_weekly <- as_tibble(dt_simulated_weekly)
# save(dt_simulated_weekly, file = "data/dt_simulated_weekly.RData", version = 2)

####################################################################
#' Robyn Dataset: Holidays by Country
#'
#' Contains \code{prophet}'s "new" default holidays by country.
#' When using own holidays, please keep the header
#' \code{c("ds", "holiday", "country", "year")}.
#'
#'
#' @family Dataset
#' @docType data
#' @usage data(dt_prophet_holidays)
#' @return data.frame
#' @format An object of class \code{"data.frame"}
#' \describe{
#'   \item{ds}{Date}
#'   \item{holiday}{Name of celebrated holiday}
#'   \item{country}{Code for the country (Alpha-2)}
#'   \item{year}{Year of \code{ds}}
#' }
#' @examples
#' data(dt_prophet_holidays)
#' head(dt_prophet_holidays)
#' @return Dataframe. Contains \code{prophet}'s default holidays by country.
"dt_prophet_holidays"

# dt_prophet_holidays <- read.csv("~/Desktop/generated_holidays.csv")
# dt_prophet_holidays <- as_tibble(dt_prophet_holidays)
# lares::missingness(dt_prophet_holidays)
# dt_prophet_holidays <- dplyr::filter(dt_prophet_holidays, !is.na(country))
# save(dt_prophet_holidays, file = "data/dt_prophet_holidays.RData", version = 2)

####################################################################
#' Robyn Dataset: Reach & frequency simulated dataset
#'
#' A simulated cumulated reach and spend dataset by frequency buckets.
#' The headers must be kept as
#' \code{c("spend_cumulated", "response_cumulated", "freq_bucket")}.
#'
#'
#' @family Dataset
#' @docType data
#' @usage data(df_curve_reach_freq)
#' @return data.frame
#' @format An object of class \code{"data.frame"}
#' \describe{
#'   \item{spend_cumulated}{cumulated spend of paid media}
#'   \item{response_cumulated}{cumulated reach of paid media}
#'   \item{freq_bucket}{Frequency bucket for cumulated reach}
#' }
#' @examples
#' data(df_curve_reach_freq)
#' head(df_curve_reach_freq)
#' @return Dataframe.
"df_curve_reach_freq"

# xSample <- round(seq(0, 100000, length.out = 10))
# gammaSamp <- seq(0.3, 1, length.out = 20)
# coeff <- 10000000
# df_curve_reach_freq <- list()
# for (i in seq_along(gammaSamp)) {
#   df_curve_reach_freq[[i]] <- data.frame(
#     spend_cumulated = xSample,
#     response_predicted = (xSample**0.5 / (xSample**0.5 + (gammaSamp[i] * max(xSample))**0.5)) * coeff ,
#     gamma = gammaSamp[i],
#     freq_bucket = as.factor(paste0("reach ", i, "+"))
#   )
# }
# df_curve_reach_freq <- bind_rows(df_curve_reach_freq) %>%
#   mutate(response_cumulated = response_predicted * (1+ runif(length(xSample) * length(gammaSamp), -0.05, 0.05))) %>%
#   select(spend_cumulated, response_cumulated, response_predicted, freq_bucket)
# levels(df_curve_reach_freq$freq_bucket) <- paste0("reach ", seq_along(gammaSamp), "+")
# save(df_curve_reach_freq, file = "data/df_curve_reach_freq.RData", version = 2)
