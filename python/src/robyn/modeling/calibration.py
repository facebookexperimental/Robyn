#pyre-strict

import pandas as pd
from typing import Optional, List, Dict

class Calibration:
    """
    A class used to perform calibration on the provided input data.

    Attributes:
    ----------
    calibration_input : pd.DataFrame
        The input data for calibration.
    df_raw : pd.DataFrame
        The raw data.
    dayInterval : int
        The day interval.
    xDecompVec : pd.DataFrame
        The decomposed vector.
    coefs : pd.DataFrame
        The coefficients.
    hypParamSam : Dict[str, float]
        The hyperparameter samples.
    wind_start : int
        The start of the window (default is 1).
    wind_end : int
        The end of the window (default is the number of rows in df_raw).
    adstock : str
        The adstock type (default is None).

    Methods:
    -------
    calibrate()
        Performs calibration on the provided input data.
    """

    def __init__(self,
                 calibration_input: pd.DataFrame,
                 df_raw: pd.DataFrame,
                 dayInterval: int,
                 xDecompVec: pd.DataFrame,
                 coefs: pd.DataFrame,
                 hypParamSam: Dict[str, float],
                 wind_start: int = 1,
                 wind_end: Optional[int] = None,
                 adstock: Optional[str] = None):
        """
        Initializes the Calibration class.

        Args:
        ----
        calibration_input : pd.DataFrame
            The input data for calibration.
        df_raw : pd.DataFrame
            The raw data.
        dayInterval : int
            The day interval.
        xDecompVec : pd.DataFrame
            The decomposed vector.
        coefs : pd.DataFrame
            The coefficients.
        hypParamSam : Dict[str, float]
            The hyperparameter samples.
        wind_start : int
            The start of the window (default is 1).
        wind_end : int
            The end of the window (default is the number of rows in df_raw).
        adstock : str
            The adstock type (default is None).
        """
        self.calibration_input = calibration_input
        self.df_raw = df_raw
        self.dayInterval = dayInterval
        self.xDecompVec = xDecompVec
        self.coefs = coefs
        self.hypParamSam = hypParamSam
        self.wind_start = wind_start
        self.wind_end = wind_end if wind_end else len(df_raw)
        self.adstock = adstock

    def calibrate(self) -> pd.DataFrame:
        """
        Performs calibration on the provided input data.

        Returns:
        -------
        pd.DataFrame
            The calibrated data.
        """
        # Check if the calibration input is within the modeling window
        ds_wind = self.df_raw['ds'][self.wind_start:self.wind_end]
        include_study = any(
            (self.calibration_input['liftStartDate'] >= min(ds_wind)) &
            (self.calibration_input['liftEndDate'] <= (max(ds_wind) + self.dayInterval - 1))
        )

        if not include_study:
            print("All calibration_input is outside modeling window. Running without calibration")
            return None

        # Initialize the calibrated data
        calibrated_data = self.calibration_input.copy()
        calibrated_data['pred'] = None
        calibrated_data['pred_total'] = None
        calibrated_data['decompStart'] = None
        calibrated_data['decompEnd'] = None

        # Loop through each study
        for index, row in calibrated_data.iterrows():
            # Get the channels
            channels = row['channel'].split('+')

            # Initialize the channel-wise data
            channel_collect = {}
            channel_total_collect = {}

            # Loop through each channel
            for channel in channels:
                # Get the scope and study dates
                scope = row['calibration_scope']
                study_start = row['liftStartDate']
                study_end = row['liftEndDate']

                # Get the study positions
                study_pos = self.df_raw[(self.df_raw['ds'] >= study_start) & (self.df_raw['ds'] <= study_end)].index

                # Get the calibration positions
                if study_start in self.df_raw['ds'].values:
                    calib_pos = study_pos
                else:
                    calib_pos = [min(study_pos) - 1] + list(study_pos)

                # Get the calibration dates
                calibrate_dates = self.df_raw.loc[calib_pos, 'ds'].values

                # Get the calibration positions in the decomposed vector
                calib_pos_rw = self.xDecompVec[self.xDecompVec['ds'].isin(calibrate_dates)].index

                # Perform adstock transformation
                if self.adstock == 'geometric':
                    theta = self.hypParamSam[f'{channel}_thetas']
                elif self.adstock.startswith('weibull'):
                    shape = self.hypParamSam[f'{channel}_shapes']
                    scale = self.hypParamSam[f'{channel}_scales']

                # Get the immediate and total effects
                if scope == 'immediate':
                    m_imme = self.df_raw.loc[:, channel].values
                    m_total = self.df_raw.loc[:, channel].values
                    m_caov = m_total - m_imme

                    # Perform saturation transformation
                    alpha = self.hypParamSam[f'{channel}_alphas']
                    gamma = self.hypParamSam[f'{channel}_gammas']
                    m_calib_caov_sat = self.saturation_hill(m_total, alpha, gamma, m_caov)
                    m_calib_caov_decomp = m_calib_caov_sat * self.coefs.loc[self.coefs['rn'] == channel, 's0'].values[0]
                    m_calib_total_decomp = self.xDecompVec.loc[calib_pos_rw, channel].values
                    m_calib_decomp = m_calib_total_decomp - m_calib_caov_decomp
                elif scope == 'total':
                    m_calib_decomp = m_calib_total_decomp = self.xDecompVec.loc[calib_pos_rw, channel].values

                # Store the channel-wise data
                channel_collect[channel] = m_calib_decomp
                channel_total_collect[channel] = m_calib_total_decomp

            # Combine the channel-wise data
            if len(channels) > 1:
                channel_collect = {k: sum(v) for k, v in channel_collect.items()}
                channel_total_collect = {k: sum(v) for k, v in channel_total_collect.items()}
            else:
                channel_collect = list(channel_collect.values())[0]
                channel_total_collect = list(channel_total_collect.values())[0]

            # Update the calibrated data
            calibrated_data.loc[index, 'pred'] = sum(channel_collect)
            calibrated_data.loc[index, 'pred_total'] = sum(channel_total_collect)
            calibrated_data.loc[index, 'decompStart'] = min(calibrate_dates)
            calibrated_data.loc[index, 'decompEnd'] = max(calibrate_dates)

        # Perform additional calculations
        calibrated_data['decompStart'] = pd.to_datetime(calibrated_data['decompStart'], unit='D', origin='1970-01-01')
        calibrated_data['decompEnd'] = pd.to_datetime(calibrated_data['decompEnd'], unit='D', origin='1970-01-01')
        calibrated_data['liftDays'] = (calibrated_data['liftEndDate'] - calibrated_data['liftStartDate']).dt.days
        calibrated_data['decompDays'] = (calibrated_data['decompEnd'] - calibrated_data['decompStart']).dt.days
        calibrated_data['decompAbsScaled'] = calibrated_data['pred'] / calibrated_data['decompDays'] * calibrated_data['liftDays']
        calibrated_data['decompAbsTotalScaled'] = calibrated_data['pred_total'] / calibrated_data['decompDays'] * calibrated_data['liftDays']
        calibrated_data['liftMedia'] = calibrated_data['channel']
        calibrated_data['liftStart'] = calibrated_data['liftStartDate']
        calibrated_data['liftEnd'] = calibrated_data['liftEndDate']
        calibrated_data['mape_lift'] = abs((calibrated_data['decompAbsScaled'] - calibrated_data['liftAbs']) / calibrated_data['liftAbs'])
        calibrated_data['calibrated_pct'] = calibrated_data['decompAbsScaled'] / calibrated_data['decompAbsTotalScaled']

        # Select the required columns
        calibrated_data = calibrated_data[['liftMedia', 'liftStart', 'liftEnd', 'liftAbs', 'decompStart', 'decompEnd', 'decompAbsScaled', 'decompAbsTotalScaled', 'calibrated_pct', 'mape_lift']]

        return calibrated_data
