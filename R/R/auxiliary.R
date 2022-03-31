# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Calculate R-squared
get_rsq <- function(true, predicted, p = NULL, df.int = NULL) {
  sse <- sum((predicted - true)^2)
  sst <- sum((true - mean(true))^2)
  rsq <- 1 - sse / sst
  if (!is.null(p) & !is.null(df.int)) {
    n <- length(true)
    rdf <- n - p - 1
    rsq <- 1 - (1 - rsq) * ((n - df.int) / rdf)
  }
  return(rsq)
}

# Robyn colors
robyn_palette <- function() {
  pal <- c("#21130d","#351904","#543005","#8C510A","#BF812D","#DFC27D","#F6E8C3",
           "#F5F5F5","#C7EAE5","#80CDC1","#35978F","#01665E","#043F43", "#04272D")
  list(
    fill = rep(pal, 3),
    colour = rep(c(rep("#FFFFFF", 4), rep("#000000", 7), rep("#FFFFFF", 3)), 3)
  )
}
# lares::plot_palette(
#   fill = robyn_palette()$fill, colour = robyn_palette()$colour,
#   limit = length(unique(robyn_palette()$fill)))

flatten_hyps <- function(x) {
  if (is.null(x)) return(x)
  temp <- sapply(x, function(x) sprintf("[%s]", paste(if(is.numeric(x)) signif(x, 6) else x, collapse = ", ")))
  paste(paste0("  ", names(temp), ":"), temp, collapse = "\n")
}
