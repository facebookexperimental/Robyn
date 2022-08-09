# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
#' Robyn's Educational UI (RobynLearn)
#'
#' Using Shiny on top of Robyn users are enabled to both learn Robyn
#' in a more efficient way, as well as develop end-to-end preliminary
#' models with limited flexibility. All features of Robyn are represented
#' in some capacity in the UI, and it remains the best way for those with
#' limited Marketing Mix Modeling and/or coding experience to familiarize
#' themselves with Robyn & MMM.
#'
#' @param ... Additional parameters
#' @author Kyle Goldberg (kylegoldberg@@fb.com)
#' @export
robyn_learn <- function(...) {
  shinyApp(ui(), server)
}
