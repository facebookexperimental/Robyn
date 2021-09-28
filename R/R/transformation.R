# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Includes function mic_men, adstock_geometric, adstock_weibull,
# saturation_hill, plot_adstock, plot_saturation

####################################################################
#' Michaelis-Menten transformation
#'
#' The Michaelis-Menten function is used to fit the spend exposure relationship
#' for paid media variables, when exposure metrics like impressions, clicks or
#' GRPs are provided in \code{paid_media_vars} instead of spend metric.
#'
#' @param x Numeric value or vector. Input media spend when
#' \code{reverse = FALSE}. Input media exposure metrics (impression, clicks,
#' GRPs, etc.) when \code{reverse = TRUE}.
#' @param Vmax A numeric value. Indicates maximum rate achieved by the system.
#' @param Km A numeric value. The Michaelis constant.
#' @param reverse A logical value. Input media spend when \code{reverse = FALSE}.
#' Input media exposure metrics (impression, clicks, GRPs etc.) when \code{reverse = TRUE}.
#' @export
mic_men <- function(x, Vmax, Km, reverse = FALSE) {
  if (!reverse) {
    mm_out <- exposure <- Vmax * x / (Km + x)
  } else {
    mm_out <- spend <- x * Km / (Vmax - x)
  }
  return(mm_out)
}


####################################################################
#' Geometric adstocking function
#'
#' The geometric adstock function is the classic one-parametric adstock function.
#'
#' @param x A numeric vector.
#' @param theta A numeric value. Indicates the decay rate of the geometric function.
#' @export
adstock_geometric <- function(x, theta) {
  x_decayed <- c(x[1], rep(0, length(x) - 1))
  for (xi in 2:length(x_decayed)) {
    x_decayed[xi] <- x[xi] + theta * x_decayed[xi - 1]
  }

  thetaVecCum <- theta
  for (t in 2:length(x)) {
    thetaVecCum[t] <- thetaVecCum[t - 1] * theta
  } # plot(thetaVecCum)

  return(list(x_decayed = x_decayed, thetaVecCum = thetaVecCum))
}

####################################################################
#' Weibull adstock function
#'
#' The Weibull adstock function is a two-parametric adstock function that allows
#' changing decay rate over time, compared to the fixed decay rate over time as
#' in Geomeric adstock.
#'
#' @param x A numeric vector.
#' @param shape A numeric value. The shape parameter controls the decay shape.
#' The larger the shape value, the more S-shape the curve shows. The smaller the
#' shape value, the more L-shape the curve shows.
#' @param scale A numeric value. The scale parameter controls the decay inflexion
#' point. The larger the scale value, the later the inflexion point occurs.
#' @export
adstock_weibull <- function(x, shape, scale) {
  x.n <- length(x)
  x_bin <- 1:x.n
  scaleTrans <- round(quantile(x_bin, scale), 0) #
  thetaVec <- c(1, 1 - pweibull(head(x_bin, -1), shape = shape, scale = scaleTrans)) # plot(thetaVec)
  thetaVecCum <- cumprod(thetaVec) # plot(thetaVecCum)

  x_decayed <- mapply(function(x, y) {
    x.vec <- c(rep(0, y - 1), rep(x, x.n - y + 1))
    thetaVecCumLag <- shift(thetaVecCum, y - 1, fill = 0)
    x.matrix <- cbind(x.vec, thetaVecCumLag) #  plot(x.vec)
    x.prod <- apply(x.matrix, 1, prod)
    return(x.prod)
  }, x = x, y = x_bin)
  x_decayed <- rowSums(x_decayed)

  return(list(x_decayed = x_decayed, thetaVecCum = thetaVecCum))
}

####################################################################
#' Hill saturation function
#'
#' The Hill function applied here is a two-parametric version of the
#' function that allows the saturation curve to flip between S and C shape.
#'
#' @param x A numeric vector.
#' @param alpha A numeric value. Alpha controls the shape of the saturation curve.
#' The larger the alpha, the more S-shape. The smaller, the more C-shape.
#' @param gamma A numeric value. Gamma controls the inflexion point of the
#' saturation curve. The larger the gamma, the later the inflexion point occurs.
#' @param x_marginal A numeric value. When provided, the function returns the
#' Hill-transformed value of the x_marginal input.
#' @export
saturation_hill <- function(x, alpha, gamma, x_marginal = NULL) {
  gammaTrans <- round(quantile(seq(range(x)[1], range(x)[2], length.out = 100), gamma), 4)

  if (is.null(x_marginal)) {
    x_scurve <- x**alpha / (x**alpha + gammaTrans**alpha) # plot(x_scurve) summary(x_scurve)
  } else {
    x_scurve <- x_marginal**alpha / (x_marginal**alpha + gammaTrans**alpha)
  }
  return(x_scurve)
}


####################################################################
#' Adstocking help plot
#'
#' Produce example plots for both Geometric and Weibull adstock.
#'
#' @param plot Boolean. Do you wish to return the plot?
#' @examples
#' plot_adstock()
#' @export
plot_adstock <- function(plot = TRUE) {
  if (plot) {
    # plot weibull
    weibullCollect <- list()
    shapeVec <- c(2, 2, 2, 2, 2, 2, 0.01, 0.1, 0.5, 1, 1.5, 2)
    scaleVec <- c(0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    paramRotate <- c(rep("scale", 6), rep("shape", 6))

    for (v in 1:length(shapeVec)) {
      dt_weibull <- data.table(
        x = 1:100,
        decay_accumulated = adstock_weibull(rep(1, 100), shape = shapeVec[v], scale = scaleVec[v])$thetaVecCum,
        shape = shapeVec[v],
        scale = scaleVec[v],
        param = paramRotate[v]
      )
      dt_weibull[, halflife := which.min(abs(decay_accumulated - 0.5))]

      weibullCollect[[v]] <- dt_weibull
    }

    weibullCollect <- rbindlist(weibullCollect)
    # weibullCollect[, ':='(shape=as.factor(shape), scale=as.factor(scale))]
    weibullCollect[, scale_shape_halflife := paste(scale, shape, halflife, sep = "_")]
    p1 <- ggplot(weibullCollect[param == "scale"], aes(x = x, y = decay_accumulated)) +
      geom_line(aes(color = scale_shape_halflife)) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
      geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife = time until effect reduces to 50%"), colour = "gray") +
      labs(
        title = "Weibull adstock transformation - scale changes",
        subtitle = "Halflife = time until effect reduces to 50%",
        x = "time unit",
        y = "Media decay accumulated"
      )
    p2 <- ggplot(weibullCollect[param == "shape"], aes(x = x, y = decay_accumulated)) +
      geom_line(aes(color = scale_shape_halflife)) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
      geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife = time until effect reduces to 50%"), colour = "gray") +
      labs(
        title = "Weibull adstock transformation - shape changes",
        subtitle = "Halflife = time until effect reduces to 50%",
        x = "time unit",
        y = "Media decay accumulated"
      )

    ## plot geometric

    geomCollect <- list()
    thetaVec <- c(0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9)

    for (v in 1:length(thetaVec)) {
      thetaVecCum <- 1
      for (t in 2:100) {
        thetaVecCum[t] <- thetaVecCum[t - 1] * thetaVec[v]
      }
      dt_geom <- data.table(
        x = 1:100,
        decay_accumulated = thetaVecCum,
        theta = thetaVec[v]
      )
      dt_geom[, halflife := which.min(abs(decay_accumulated - 0.5))]
      geomCollect[[v]] <- dt_geom
    }
    geomCollect <- rbindlist(geomCollect)
    geomCollect[, theta_halflife := paste(theta, halflife, sep = "_")]

    p3 <- ggplot(geomCollect, aes(x = x, y = decay_accumulated)) +
      geom_line(aes(color = theta_halflife)) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +
      geom_text(aes(x = max(x), y = 0.5, vjust = -0.5, hjust = 1, label = "Halflife = time until effect reduces to 50%"), colour = "gray") +
      labs(
        title = "Geometric adstock transformation - theta changes",
        subtitle = "Halflife = time until effect reduces to 50%",
        x = "time unit",
        y = "Media decay accumulated"
      )
    # print(p3)

    return(wrap_plots(
      A = p1, B = p2, C = p3,
      design = "AC
      BC"
    ))
  }
}


####################################################################
#' Saturation help plot
#'
#' Produce example plots for the Hill saturation curve.
#'
#' @inheritParams plot_adstock
#' @examples
#' plot_saturation()
#' @export
plot_saturation <- function(plot = TRUE) {
  if (plot) {
    xSample <- 1:100
    alphaSamp <- c(0.1, 0.5, 1, 2, 3)
    gammaSamp <- c(0.1, 0.3, 0.5, 0.7, 0.9)

    ## plot alphas
    hillAlphaCollect <- list()
    for (i in 1:length(alphaSamp)) {
      hillAlphaCollect[[i]] <- data.table(
        x = xSample,
        y = xSample**alphaSamp[i] / (xSample**alphaSamp[i] + (0.5 * 100)**alphaSamp[i]),
        alpha = alphaSamp[i]
      )
    }
    hillAlphaCollect <- rbindlist(hillAlphaCollect)
    hillAlphaCollect[, alpha := as.factor(alpha)]
    p1 <- ggplot(hillAlphaCollect, aes(x = x, y = y, color = alpha)) +
      geom_line() +
      labs(
        title = "Cost response with hill function",
        subtitle = "Alpha changes while gamma = 0.5"
      )

    ## plot gammas
    hillGammaCollect <- list()
    for (i in 1:length(gammaSamp)) {
      hillGammaCollect[[i]] <- data.table(
        x = xSample,
        y = xSample**2 / (xSample**2 + (gammaSamp[i] * 100)**2),
        gamma = gammaSamp[i]
      )
    }
    hillGammaCollect <- rbindlist(hillGammaCollect)
    hillGammaCollect[, gamma := as.factor(gamma)]
    p2 <- ggplot(hillGammaCollect, aes(x = x, y = y, color = gamma)) +
      geom_line() +
      labs(
        title = "Cost response with hill function",
        subtitle = "Gamma changes while alpha = 2"
      )

    return(p1 + p2)
  }
}
