# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Calculate R-squared
get_rsq <- function(true, predicted, p = NULL, df.int = NULL, n_train = NULL) {
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- 1 - sse / sst # rsq interpreted as variance explained
  rsq_out <- rsq
  if (!is.null(p) && !is.null(df.int)) {
    if (!is.null(n_train)) {
      n <- n_train # for oos dataset, use n from train set for adj. rsq
    } else {
      n <- length(true)
    }
    rdf <- n - p - 1
    rsq_adj <- 1 - (1 - rsq) * ((n - df.int) / rdf)
    rsq_out <- rsq_adj
  }
  return(rsq_out)
}

# Robyn colors
robyn_palette <- function() {
  pal <- c(
    "#21130d", "#351904", "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",
    "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E", "#043F43", "#04272D"
  )
  repeated <- 4
  list(
    fill = rep(pal, repeated),
    colour = rep(c(rep("#FFFFFF", 4), rep("#000000", 7), rep("#FFFFFF", 3)), repeated)
  )
}
# lares::plot_palette(
#   fill = robyn_palette()$fill, colour = robyn_palette()$colour,
#   limit = length(unique(robyn_palette()$fill)))

flatten_hyps <- function(x) {
  if (is.null(x)) {
    return(x)
  }
  temp <- unlist(lapply(x, function(x) {
    sprintf("[%s]", paste(if (is.numeric(x)) signif(x, 6) else x, collapse = ", "))
  }))
  paste(paste0("  ", names(temp), ":"), temp, collapse = "\n")
}

####################################################################
#' Update Robyn Version
#'
#' Update Robyn version from
#' \href{https://github.com/facebookexperimental/Robyn}{Github repository}
#' for latest "dev" version or from
#' \href{https://CRAN.R-project.org/package=Robyn}{CRAN}
#' for latest "stable" version.
#'
#' @param dev Boolean. Dev version? If not, CRAN version.
#' @param ... Parameters to pass to \code{remotes::install_github}
#' or \code{utils::install.packages}, depending on \code{dev} parameter.
#' @return Invisible \code{NULL}.
#' @export
robyn_update <- function(dev = TRUE, ...) {
  if (dev) {
    try_require("remotes")
    # options(timeout = 400)
    install_github(repo = "facebookexperimental/Robyn/R", ...)
  } else {
    utils::install.packages("Robyn", ...)
  }
}

# Merge baseline variables based on baseline_level param input
baseline_vars <- function(InputCollect, baseline_level) {
  stopifnot(length(baseline_level) == 1)
  stopifnot(baseline_level %in% 0:5)
  x <- ""
  if (baseline_level >= 1) {
    x <- c(x, "(Intercept)", "intercept")
  }
  if (baseline_level >= 2) {
    x <- c(x, "trend")
  }
  if (baseline_level >= 3) {
    x <- unique(c(x, InputCollect$prophet_vars))
  }
  if (baseline_level >= 4) {
    x <- c(x, InputCollect$context_vars)
  }
  if (baseline_level >= 5) {
    x <- c(x, InputCollect$organic_vars)
  }
  return(x)
}

# Calculate dot product
.dot_product <- function(range, proportion) {
  mapply(
    function(proportion) {
      c(range %*% c(1 - proportion, proportion))
    },
    proportion = proportion
  )
}

# Calculate quantile interval
.qti <- function(x, interval = 0.95) {
  check_qti(interval)
  int_low <- (1 - interval) / 2
  int_up <- 1 - int_low
  qt_low <- quantile(x, int_low)
  qt_up <- quantile(x, int_up)
  return(c(qt_low, qt_up))
}

# Calculate MSE
.mse_loss <- function(y, y_hat) mean((y - y_hat)^2)

# next_date(c("2021-01-01", "2021-02-01"))
# next_date(c("2021-01-01", "2021-01-08", "2021-01-15"))
# next_date(c(Sys.Date() - 1, Sys.Date()))
.next_date <- function(dates) {
  dates <- as.Date(dates)
  diffs <- diff(dates)
  if (all(diffs == 1)) {
    frequency <- "daily"
  } else if (all(diffs == 7)) {
    frequency <- "weekly"
  } else if (all(format(dates[-length(dates)], "%Y-%m") != format(dates[-1], "%Y-%m"))) {
    frequency <- "monthly"
  } else {
    warning(paste(
      "Unable to determine frequency to calculate next logical date.",
      "Returning last available date."
    ))
    return(as.Date(tail(dates, 1)))
  }
  next_date <- switch(frequency,
    "daily" = dates[length(dates)] + 1,
    "weekly" = dates[length(dates)] + 7,
    "monthly" = seq(dates[length(dates)], by = "1 month", length.out = 2)[2]
  )
  return(as.Date(next_date))
}
