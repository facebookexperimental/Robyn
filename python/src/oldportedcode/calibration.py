# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
import numpy as np
import pandas as pd
import warnings

from .transformation import saturation_hill, transform_adstock


def robyn_calibrate(
    calibration_input,
    df_raw,
    dayInterval,
    xDecompVec,
    coefs,
    hypParamSam,
    wind_start,
    wind_end,
    adstock,
):
    """
    Calibrates the given input data using the ROBYN algorithm.

    Args:
        calibration_input (DataFrame): The input data for calibration.
        df_raw (DataFrame): The raw data.
        dayInterval (int): The interval in days.
        xDecompVec (DataFrame): The decomposition vector.
        coefs (DataFrame): The coefficients.
        hypParamSam (DataFrame): The hyperparameters.
        wind_start (int): The start index of the window.
        wind_end (int): The end index of the window.
        adstock (str): The adstock type.

    Returns:
        DataFrame: The calibrated data.
    """
    # Convert the R code's dataframe to a Pandas dataframe
    df_raw = pd.DataFrame(df_raw)
    # Extract the necessary columns from the dataframe
    ds_wind = df_raw["ds"][(wind_start-1):(wind_end-1)]
    min_ds = calibration_input["liftStartDate"] >= np.min(ds_wind)
    max_ds = calibration_input["liftEndDate"] <= (np.max(ds_wind) - np.timedelta64(dayInterval - 1,'D'))
    include_study = np.any(min_ds & max_ds)

    if not include_study:
        warnings.warn(
            "All calibration_input in outside modelling window. Running without calibration"
        )  ## Manually added
        return None
    elif include_study and calibration_input is not None:

        calibration_input["pred"] = pd.NA
        calibration_input["pred_total"] = pd.NA
        calibration_input["decompStart"] = pd.NA
        calibration_input["decompEnd"] = pd.NA

        # Split the channels into a list
        temp_channels = calibration_input["channel"].values
        split_channels = list()
        for val in temp_channels:
            str_list = str(val).split("+")
            split_channels.append(str_list)
            # for channel_str in str_list:

        ##split_channels = list(calibration_input["channel"].values)

        # Loop through each channel
        for l_study in range(len(split_channels)):
            # Get the current channel and its corresponding scope
            get_channels = split_channels[l_study]
            scope = calibration_input["calibration_scope"].values[l_study]
            study_start = calibration_input["liftStartDate"].values[l_study]
            study_end = calibration_input["liftEndDate"].values[l_study]
            study_pos = df_raw.index[(df_raw['ds'] >= study_start) & (df_raw['ds'] <= study_end)].values

            if study_start in df_raw['ds'].values:
                calib_pos = study_pos
            else:
                calib_pos = [min(study_pos) - 1]
                calib_pos.extend(study_pos)

            calibrate_dates = df_raw.loc[calib_pos, "ds"].values
            calibrate_dates = pd.to_datetime(calibrate_dates).date

            xDecompVec['ds'] = pd.to_datetime(xDecompVec['ds']).dt.date
            mask = xDecompVec['ds'].isin(calibrate_dates)
            calib_pos_rw = xDecompVec.index[mask]
            xDecompVec_filtered = xDecompVec.loc[calib_pos_rw]

            # Initialize the list of calibrated channels for this scope
            l_chn_collect_scope = []
            l_chn_total_collect_scope = []

            # Initialize the list of calibrated channels
            l_chn_collect = {}
            l_chn_total_collect = {}

            # Loop through each position in the channel
            for l_chn in range(len(get_channels)):
                # Get the current channel and its corresponding position
                if scope == 'immediate':
                    channel = get_channels[l_chn]
                    m = df_raw[get_channels[l_chn]].values ## [0]
                    ##pos = np.where(df_raw["ds"] == channel)[0]
                    ## ps = df_raw[df_raw["ds"] == channel]
                    theta = shape = scale = None
                    if adstock == 'geometric':
                        theta = hypParamSam["{}{}".format(get_channels[l_chn], "_thetas")] ##.values[0][0]

                    if adstock.startswith('weibull'):
                        shape = hypParamSam["{}{}".format(get_channels[l_chn], "_shapes")]##.values[0][0]
                        scale = hypParamSam["{}{}".format(get_channels[l_chn], "_scales")]##.values[0][0]

                    x_list = transform_adstock(m, adstock, theta=theta, shape=shape, scale=scale)

                    if adstock == "weibull_pdf":
                        m_imme = x_list['x_imme']
                    else:
                        m_imme = m

                    m_total = x_list['x_decayed']
                    m_coav = m_total - m_imme

                    m_caov_calib = m_coav[calib_pos]
                    m_total_rw = m_total[wind_start:wind_end]

                    alpha = hypParamSam["{}{}".format(get_channels[l_chn], "_alphas")]##.values[0][0]
                    gamma = hypParamSam["{}{}".format(get_channels[l_chn], "_gammas")]##.values[0][0]

                    m_calib_caov_sat = saturation_hill(m_total_rw, alpha = alpha, gamma = gamma, x_marginal = m_caov_calib)
                    coeff_value = coefs.loc[get_channels[l_chn], 's0']
                    m_calib_caov_decomp = m_calib_caov_sat * coeff_value

                    m_calib_total_decomp = xDecompVec.loc[calib_pos_rw, get_channels[l_chn]]
                    m_calib_decomp= m_calib_total_decomp.reset_index(drop=True) - m_calib_caov_decomp.reset_index(drop=True)

                    if scope == 'total':
                        m_calib_decomp = m_calib_total_decomp = xDecompVec[calib_pos_rw, get_channels[l_chn]]

                    l_chn_collect[get_channels[l_chn]] = m_calib_decomp
                    l_chn_total_collect[get_channels[l_chn]] = m_calib_total_decomp

            l_chn_collect = pd.DataFrame(l_chn_collect)
            l_chn_total_collect = pd.DataFrame(l_chn_total_collect)

            if len(get_channels) > 1:
                # Sum across rows if there are multiple channels
                l_chn_collect = l_chn_collect.sum(axis=1)
                l_chn_total_collect = l_chn_total_collect.sum(axis=1)
            else:
                # If there's only one channel, it's already effectively a single "flattened" Series
                l_chn_collect = l_chn_collect.squeeze()
                l_chn_total_collect = l_chn_total_collect.squeeze()

            calibration_input.at[l_study,"pred"] = l_chn_collect.sum()
            calibration_input.at[l_study,"pred_total"] = l_chn_total_collect.sum()
            calibration_input.at[l_study,"decompStart"] = calibrate_dates[0]
            calibration_input.at[l_study,"decompEnd"] = calibrate_dates[1]

        liftCollect = pd.DataFrame(calibration_input)
        liftCollect[['pred', 'pred_total']] = liftCollect[['pred', 'pred_total']] #.astype(float)
        liftCollect[['decompStart', 'decompEnd']] = liftCollect[['decompStart', 'decompEnd']] #.astype(pd.Timestamp)

        liftCollect['liftDays'] = (liftCollect['liftEndDate'] - liftCollect['liftStartDate']) / np.timedelta64(1, 'D')
        liftCollect['decompDays'] = (liftCollect['decompStart'] - liftCollect['decompEnd']) / np.timedelta64(1, 'D')

        liftCollect['decompAbsScaled'] = liftCollect['pred'] / liftCollect['decompDays'] * liftCollect['liftDays']
        liftCollect['decompAbsTotalScaled'] = liftCollect['pred_total'] / liftCollect['decompDays'] * liftCollect['liftDays']

        liftCollect['liftMedia'] = liftCollect['channel']
        liftCollect['liftStart'] = liftCollect['liftStartDate']
        liftCollect['liftEnd'] = liftCollect['liftEndDate']

        liftCollect['calibrated_pct'] = abs((liftCollect['decompAbsScaled'] - liftCollect['liftAbs']) / liftCollect['liftAbs'])
        liftCollect['mape_lift'] = liftCollect['decompAbsScaled'] / liftCollect['decompAbsTotalScaled']

        return liftCollect[['liftMedia',
            'liftStart',
            'liftEnd',
            'liftAbs',
            'decompStart',
            'decompEnd',
            'decompAbsScaled',
            'decompAbsTotalScaled',
            'calibrated_pct',
            'mape_lift']]
