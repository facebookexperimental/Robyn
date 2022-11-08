# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

robyn_calibrate <- function(calibration_input,
                            df_raw, # df_raw = InputCollect$dt_mod
                            dayInterval, # dayInterval = InputCollect$dayInterval
                            dt_modAdstocked, # dt_modAdstocked = InputCollect$dt_mod (?)
                            xDecompVec, # xDecompVec = decompCollect$xDecompVec
                            coefs, # coefs = decompCollect$coefsOutCat
                            hypParamSam,
                            wind_start = 1,
                            wind_end = nrow(df_raw),
                            adstock) {
  ds_wind <- df_raw$ds[wind_start:wind_end]
  include_study <- any(
    calibration_input$liftStartDate >= min(ds_wind) &
      calibration_input$liftEndDate <= (max(ds_wind) + dayInterval - 1)
  )
  if (!is.null(calibration_input) & !include_study) {
    warning("All calibration_input in outside modelling window. Running without calibration")
  } else if (!is.null(calibration_input) & include_study) {
    calibration_input <- mutate(
      calibration_input,
      pred = NA, pred_total = NA, decompStart = NA, decompEnd = NA
    )
    split_channels <- strsplit(calibration_input$channel, split = "\\+")

    for (l_study in seq_along(split_channels)) {
      get_channels <- split_channels[[l_study]]
      scope <- calibration_input$calibration_scope[[l_study]]
      study_start <- calibration_input$liftStartDate[[l_study]]
      study_end <- calibration_input$liftEndDate[[l_study]]
      study_pos <- which(df_raw$ds >= study_start & df_raw$ds <= study_end)
      if (study_start %in% df_raw$ds) {
        calib_pos <- study_pos
      } else {
        calib_pos <- c(min(study_pos) - 1, study_pos)
      }
      calibrate_dates <- df_raw[calib_pos, "ds"][[1]]
      calib_pos_rw <- which(xDecompVec$ds %in% calibrate_dates)

      l_chn_collect <- list()
      l_chn_total_collect <- list()
      for (l_chn in seq_along(get_channels)) {
        if (scope == "immediate") {
          m <- df_raw[, get_channels[l_chn]][[1]]
          m_calib <- df_raw[calib_pos, get_channels[l_chn]][[1]]

          ## 1. Adstock
          if (adstock == "geometric") {
            theta <- hypParamSam[paste0(get_channels[l_chn], "_thetas")][[1]][[1]]
            x_list <- adstock_geometric(x = m_calib, theta = theta)
          } else if (adstock == "weibull_cdf") {
            shape <- hypParamSam[paste0(get_channels[l_chn], "_shapes")][[1]][[1]]
            scale <- hypParamSam[paste0(get_channels[l_chn], "_scales")][[1]][[1]]
            x_list <- adstock_weibull(x = m_calib, shape = shape, scale = scale, windlen = length(m), type = "cdf")
          } else if (adstock == "weibull_pdf") {
            shape <- hypParamSam[paste0(get_channels[l_chn], "_shapes")][[1]][[1]]
            scale <- hypParamSam[paste0(get_channels[l_chn], "_scales")][[1]][[1]]
            x_list <- adstock_weibull(x = m_calib, shape = shape, scale = scale, windlen = length(m), type = "pdf")
          }
          m_calib_total_adst <- dt_modAdstocked[calib_pos, get_channels[l_chn]][[1]]
          m_calib_imme_adst <- x_list$x_decayed
          m_calib_hist_adst <- m_calib_total_adst - m_calib_imme_adst
          # Adapt for weibull_pdf with lags
          m_calib_imme_adst[m_calib_hist_adst < 0] <- m_calib_total_adst[m_calib_hist_adst < 0]
          m_calib_hist_adst[m_calib_hist_adst < 0] <- 0

          ## 2. Saturation
          m_adstocked_rw <- dt_modAdstocked[wind_start:wind_end, get_channels[l_chn]][[1]]
          alpha <- hypParamSam[paste0(get_channels[l_chn], "_alphas")][[1]][[1]]
          gamma <- hypParamSam[paste0(get_channels[l_chn], "_gammas")][[1]][[1]]
          m_calib_hist_sat <- saturation_hill(
            m_adstocked_rw,
            alpha = alpha, gamma = gamma, x_marginal = m_calib_hist_adst
          )
          m_calib_hist_decomp <- m_calib_hist_sat * coefs$s0[coefs$rn == get_channels[l_chn]]
          m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
          m_calib_decomp <- m_calib_total_decomp - m_calib_hist_decomp
        }
        if (scope == "total") {
          m_calib_decomp <- m_calib_total_decomp <- xDecompVec[calib_pos_rw, get_channels[l_chn]]
        }
        l_chn_collect[[get_channels[l_chn]]] <- m_calib_decomp
        l_chn_total_collect[[get_channels[l_chn]]] <- m_calib_total_decomp
      }

      if (length(get_channels) > 1) {
        l_chn_collect <- rowSums(bind_cols(l_chn_collect))
        l_chn_total_collect <- rowSums(bind_cols(l_chn_total_collect))
      } else {
        l_chn_collect <- unlist(l_chn_collect, use.names = FALSE)
        l_chn_total_collect <- unlist(l_chn_total_collect, use.names = FALSE)
      }

      calibration_input[l_study, ] <- mutate(
        calibration_input[l_study, ],
        pred = sum(l_chn_collect),
        pred_total = sum(l_chn_total_collect),
        decompStart = range(calibrate_dates)[1],
        decompEnd = range(calibrate_dates)[2]
      )
    }
    liftCollect <- calibration_input %>%
      mutate(
        decompStart = as.Date(.data$decompStart, "1970-01-01"),
        decompEnd = as.Date(.data$decompEnd, "1970-01-01")
      ) %>%
      mutate(
        liftDays = as.numeric(
          difftime(.data$liftEndDate, .data$liftStartDate, units = "days")
        ),
        decompDays = as.numeric(
          difftime(.data$decompEnd, .data$decompStart, units = "days")
        )
      ) %>%
      mutate(
        decompAbsScaled = .data$pred / .data$decompDays * .data$liftDays,
        decompAbsTotalScaled = .data$pred_total / .data$decompDays * .data$liftDays
      ) %>%
      mutate(
        liftMedia = .data$channel,
        liftStart = .data$liftStartDate,
        liftEnd = .data$liftEndDate,
        mape_lift = abs((.data$decompAbsScaled - .data$liftAbs) / .data$liftAbs),
        calibrated_pct = .data$decompAbsScaled / .data$decompAbsTotalScaled
      ) %>%
      dplyr::select(
        .data$liftMedia, .data$liftStart, .data$liftEnd, .data$liftAbs,
        .data$decompStart, .data$decompEnd, .data$decompAbsScaled,
        .data$decompAbsTotalScaled, .data$calibrated_pct, .data$mape_lift
      )

    return(liftCollect)
  }
}
