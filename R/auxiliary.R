# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function format_unit, get_rsq

####################################################################
#' Format unit
#'
#' Describe function.
#'
#' @param x_in xxx

format_unit <- function(x_in) {
  x_out <- sapply(x_in, function(x) {
    if (abs(x) >= 1000000000) {
      x_out <- paste0(round(x/1000000000, 1), " bln")
    } else if (abs(x) >= 1000000 & abs(x)<1000000000) {
      x_out <- paste0(round(x/1000000, 1), " mio")
    } else if (abs(x) >= 1000 & abs(x)<1000000) {
      x_out <- paste0(round(x/1000, 1), " tsd")
    } else {
      x_out <- round(x,0)
    }
  }, simplify = TRUE) 
  return(x_out)
}

####################################################################
#' Calculate R-squared
#'
#' Describe function.
#'
#' @param true xxx
#' @param predicted xxx
#' @param p xxx
#' @param df.int xxx

get_rsq <- function(true, predicted, p = NULL, df.int = NULL) {
  
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- 1 - sse / sst
  
  # adjusted rsq formula from summary.lm: ans$adj.r.squared <- 1 - (1 - ans$r.squared) * ((n - df.int)/rdf) # n = num_obs, p = num_indepvar, rdf = n-p-1
  if (!is.null(p) & !is.null(df.int)) {
    n <- length(true)
    rdf <- n - p - 1
    rsq <- 1 - (1 - rsq) * ((n - df.int)/rdf)
  }
  
  return(rsq)
}

