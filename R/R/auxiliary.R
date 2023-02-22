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
