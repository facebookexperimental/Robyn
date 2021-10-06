# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from collections import defaultdict
from datetime import timedelta
import matplotlib.pyplot as plt
import math
import os
import time
from prophet import Prophet
# import weibull as weibull
from scipy import stats
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pypref import prefclasses as p





########################################################################################################################
# MAIN


class Robyn(object):

    def __init__(self, country, dateVarName, depVarName, mediaVarName, dt_input):

        # R 2.1
        self.dt_input = dt_input
        self.dt_holidays = pd.read_csv('source/holidays.csv')
        self.mod = None
        self.dt_modRollWind = None
        self.xDecompAggPrev = None
        self.date_var = None
        self.dayInterval = None
        self.intervalType = None
        self.dep_var = None
        self.dep_var_type = None
        self.prophet_vars = None
        self.prophet_signs = None
        self.prophet_country = None
        self.context_vars = None
        self.context_signs = None
        self.paid_media_vars = None
        self.paid_media_signs = None
        self.paid_media_spends = None
        self.organic_vars = None
        self.organic_signs = None
        self.factor_vars = None
        self.cores = 1
        self.window_start = None
        self.window_end = None
        self.rollingWindowStartWhich = None
        self.rollingWindowEndWhich = None
        self.rollingWindowLength = None
        self.refreshAddedStart = None
        self.adstock = None
        self.iterations = 2000
        self.nevergrad_algo = "TwoPointsDE"
        self.trials = 5
        self.hyperparameters = None
        self.calibration_input = None
        self.mediaVarCount = None
        self.exposureVarName = None
        self.local_name = None
        self.all_media = None

        self.check_conditions(dt_input)

    def check_conditions(self, dt_transform):
        """

        :param dt_transform:
        :return:
        """
        ## check date input
        inputLen = dt_transform['date_var'].shape[0]
        inputLenUnique = dt_transform['date_var'].unique()
        try:
            pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d', errors='raise')
        except ValueError:
            print('input date variable should have format "yyyy-mm-dd"')
        if not self.date_var or self.date_var not in dt_transform.columns or len(self.date_var) > 1:
            raise ValueError ('Must provide correct only 1 date variable name for date_var')
        elif inputLen != inputLenUnique:
            raise ValueError('Date variable has duplicated dates. Please clean data first')
        elif dt_transform.isna().any(axis=None) or np.isinf(dt_transform).any():
            raise ValueError('dt_input has NA or Inf. Please clean data first')

        dayInterval = dt_transform['ds'].nlargest(2)
        dayInterval = (dayInterval.iloc[0] - dayInterval.iloc[1]).days
        if dayInterval == 1:
            intervalType = 'day'
        elif dayInterval == 7:
            intervalType = 'week'
        elif 28 <= dayInterval <= 31:
            intervalType = 'month'
        else:
            raise ValueError('input data has to be daily, weekly or monthly')
        self.dayInterval = dayInterval
        self.intervalType = intervalType

        ## check dependent var
        if not self.dep_var or self.dep_var not in dt_transform.columns or len(self.dep_var) > 1:
            raise ValueError('Must provide only 1 correct dependent variable name for dep_var')
        elif not pd.api.types.is_numeric_dtype(dt_transform[self.dep_var]):
            raise ValueError('dep_var must be numeric or integer')
        elif self.dep_var_type not in ['conversion', 'revenue'] or len(self.dep_var_type) != 1:
            raise ValueError('dep_var_type must be conversion or revenue')

        ## check prophet
        if not self.prophet_vars:
            self.prophet_signs = None
            self.prophet_country = None
        elif self.prophet_vars and not set(self.prophet_vars).issubset({'trend', 'season', 'weekday', 'holiday'}):
            raise ValueError('allowed values for prophet_vars are "trend", "season", "weekday" and "holiday"')
        elif not self.prophet_country or len(self.prophet_country) > 1:
            raise ValueError('1 country code must be provided in prophet_country. If your country is not available, '
                             'please add it to the holidays.csv first')
        elif not self.prophet_signs:
            self.prophet_signs = ['default'] * len(self.prophet_vars)
            print('prophet_signs is not provided. "default" is used')
        elif not set(self.prophet_signs).issubset({"positive", "negative", "default"}) or \
                len(self.prophet_signs) != self.prophet_vars:
            raise ValueError('prophet_signs must have same length as prophet_vars. allowed values are "positive", "negative", "default"')

        ## check baseline variables
        if not self.context_vars:
            self.context_signs = None
        elif not set(self.context_vars).issubset(dt_transform.columns):
            raise ValueError('Provided context_vars is not included in input data')
        elif not self.context_signs:
            self.context_signs = ['default'] * len(self.context_vars)
            print('context_signs is not provided. "default" is used')
        elif len(self.context_signs) != len(self.context_vars) or set(self.context_signs).issubset({"positive", "negative", "default"}):
            raise ValueError("context_signs must have same length as context_vars. allowed values are 'positive', "
                             "'negative', 'default'")

        ## check paid media variables
        mediaVarCount = len(self.paid_media_vars)
        spendVarCount = len(self.paid_media_spends)
        if not self.paid_media_vars or not self.paid_media_spends:
            raise ValueError('Must provide paid_media_vars and paid_media_spends')
        elif not set(self.paid_media_vars).issubset(dt_transform.columns):
            raise ValueError('Provided paid_media_vars is not included in input data')
        elif not self.paid_media_signs:
            self.paid_media_signs = ['positive'] * mediaVarCount
            print("paid_media_signs is not provided. 'positive' is used")
        elif len(self.paid_media_signs) != mediaVarCount or set(self.paid_media_signs).issubset({"positive", "negative", "default"}):
            raise ValueError("paid_media_signs must have same length as context_vars. allowed values are 'positive', "
                             "'negative', 'default'")
        elif not set(self.paid_media_spends).issubset(dt_transform.columns):
            raise ValueError('Provided paid_media_spends is not included in input data')
        elif spendVarCount != mediaVarCount:
            raise ValueError('paid_media_spends must have same length as paid_media_vars.')
        elif (dt_transform[self.paid_media_vars + self.paid_media_spends].values < 0).any():
            raise ValueError('contains negative values. Media must be >=0')
        self.exposureVarName = list(set(self.paid_media_vars) - set(self.paid_media_spends))


        ## check organic media variables
        if not set(self.organic_vars).issubset(dt_transform.columns):
            raise ValueError('Provided organic_vars is not included in input data')
        elif self.organic_vars and not self.organic_signs:
            self.organic_signs = ['positive'] * len(self.organic_vars)
            print("organic_signs is not provided. 'positive' is used")
        elif len(self.organic_signs) != len(self.organic_vars) or set(self.organic_signs).issubset({"positive", "negative", "default"}):
            raise ValueError("organic_signs must have same length as context_vars. allowed values are 'positive', "
                             "'negative', 'default'")

        ## check factor_vars
        if not self.factor_vars:
            if not set(self.factor_vars).issubset(self.context_vars + self.organic_vars):
                raise ValueError('factor_vars must be from context_vars or organic_vars')

        ## check all vars
        all_media = self.paid_media_vars + self.organic_vars
        self.all_media = all_media
        all_ind_vars = self.paid_media_vars + self.organic_vars + self.prophet_vars + self.context_vars
        if len(all_ind_vars) < len(set(all_ind_vars)):
            raise ValueError('Input variables must have unique names')

        ## check data dimension
        num_obs = dt_transform.shape[0]
        if num_obs < len(all_ind_vars) * 10:
            raise ValueError('There are' + str(len(all_ind_vars)) + 'independent variables &' + str(num_obs) +
                             'data points. We recommend row:column ratio >= 10:1')

        ## check window_start & window_end
        try:
            self.window_start = min(pd.to_datetime(dt_transform[self.date_var], format='%Y-%m-%d', errors='raise'))
        except ValueError:
            print('input date variable should have format "yyyy-mm-dd"')

        if not self.window_start:
            self.window_start = min(dt_transform[self.date_var])
        elif self.window_start < min(dt_transform[self.date_var]):
            self.window_start = min(dt_transform[self.date_var])
            raise ValueError('window_start is smaller than the earliest date in input data. It\'s set to the earliest date')
        elif self.window_start > max(dt_transform[self.date_var]):
            self.window_start = min(dt_transform[self.date_var])
            raise ValueError('window_start can\'t be larger than the the latest date in input data')

        self.rollingWindowStartWhich = abs(pd.to_datetime(dt_transform[self.date_var] - pd.to_datetime(self.window_start))).idxmin()
        if self.window_start not in dt_transform[self.date_var]:
            self.window_start = dt_transform[self.date_var][self.rollingWindowStartWhich]
            print('window_start is adapted to the closest date contained in input data')

        self.refreshAddedStart = self.window_start

        try:
            self.window_end = max(pd.to_datetime(dt_transform[self.date_var], format='%Y-%m-%d', errors='raise'))
        except ValueError:
            print('input date variable should have format "yyyy-mm-dd"')

        if not self.window_end:
            self.window_end = max(dt_transform[self.date_var])
        elif self.window_end > max(dt_transform[self.date_var]):
            self.window_end = max(dt_transform[self.date_var])
            raise ValueError('window_end is larger than the latest date in input data. It\'s set to the latest date')
        elif self.window_end < self.window_start:
            self.window_end = max(dt_transform[self.date_var])
            raise ValueError('window_end must be >= window_start. It\'s set to latest date in input data')

        self.rollingWindowEndWhich = abs(
            pd.to_datetime(dt_transform[self.date_var] - pd.to_datetime(self.window_end))).idxmin()
        if self.window_end not in dt_transform[self.date_var]:
            self.window_end = dt_transform[self.date_var][self.rollingWindowEndWhich]
            print('window_end is adapted to the closest date contained in input data')

        self.rollingWindowLength = self.rollingWindowEndWhich - self.rollingWindowStartWhich + 1

        dt_init = dt_transform.iloc[self.rollingWindowStartWhich:self.rollingWindowEndWhich+1,].loc[:, all_media]
        if not (dt_init != 0).any(axis=0).all(axis=0):
            raise ValueError('Some media channels contain only 0 within training period')

        ## check adstock
        if self.adstock not in ['geometric', 'weibull']:
            raise ValueError('adstock must be "geometric" or "weibull"')

        ## get all hypernames

        global_name = ["thetas", "shapes", "scales", "alphas", "gammas", "lambdas"]
        if self.adstock == 'geometric':
            local_name = sorted(list([i+"_"+str(j) for i in ['thetas','alphas','gamma'] for j in global_name]))
        elif self.adstock == 'weibull':
            local_name = sorted(list([i+"_"+str(j) for i in ['shapes','scales','alphas','gamma'] for j in global_name]))

        ## check hyperparameter names in hyperparameters

        ## output condition check
        # when hyperparameters is not provided
        if not self.hyperparameters:
            raise ValueError("\nhyperparameters is not provided yet. run Robyn(...hyperparameter = ...) to add it\n")
        # when hyperparameters is provided wrongly
        elif set(self.exposureVarName) != set(self.local_name):
            raise ValueError('hyperparameters must be a list and contain vectors or values' )
        else:
            # check calibration
            if self.calibration_input:
                if (min(self.calibration_input['liftStartDate']) < min(dt_transform[self.date_var])
                    or max(self.calibration_input['liftStartDate']) > max(dt_transform[self.date_var])):
                    raise ValueError('we recommend you to only use experimental results conducted within your MMM input'
                                     'data date range')

                elif self.iterations < 2000 or self.trials < 10:
                    raise ValueError('you are calibrating MMM. we recommend to run at least 2000 iterations per trial and '
                                 'at least 10 trials at the beginning')
                elif self.iterations < 2000 or self.trials < 5:
                    raise ValueError('we recommend to run at least 2000 iterations per trial and at least 5 trials at the beginning')

                #when all provided once correctly
                print('\nAll input in robyn_inputs() correct. Ready to run robyn_run(...)')
                #dt_new = self.robyn_engineering(dt_transform)

            elif not self.hyperparameters:
                raise ValueError("hyperparameters is not provided yet")

        # if self.activate_prophet and not set(self.prophet).issubset({'trend', 'season', 'weekday', 'holiday'}):
        #     raise ValueError('set_prophet must be "trend", "season", "weekday" or "holiday"')
        # if self.activate_baseline:
        #     if len(self.baseVarName) != len(self.baseVarSign):
        #         raise ValueError('set_baseVarName and set_baseVarSign have to be the same length')
        #
        # if len(self.mediaVarName) != len(self.mediaVarSign):
        #     raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')
        # if not (set(self.prophetVarSign).issubset({"positive", "negative", "default"}) and
        #         set(self.baseVarSign).issubset({"positive", "negative", "default"}) and
        #         set(self.mediaVarSign).issubset({"positive", "negative", "default"})):
        #     raise ValueError('set_prophetVarSign, '
        #                      'set_baseVarSign & set_mediaVarSign must be "positive", "negative" or "default"')
        # if self.activate_calibration:
        #     if self.lift.shape[0] == 0:
        #         raise ValueError('please provide lift result or set activate_calibration = FALSE')
        #     if (min(self.lift['liftStartDate']) < min(dt_transform['ds'])
        #             or (max(self.lift['liftEndDate']) > max(dt_transform['ds']) + timedelta(days=self.dayInterval - 1))):
        #         raise ValueError(
        #             'we recommend you to only use lift results conducted within your MMM input data date range')
        #
        #     if self.iter < 500 or self.trial < 80:
        #         raise ValueError('you are calibrating MMM. we recommend to run at least 500 iterations '
        #                          'per trial and at least 80 trials at the beginning')
        #
        # if self.adstock_type not in ['geometric', 'weibull']:
        #     raise ValueError('adstock must be "geometric" or "weibull"')
        # if self.adstock_type == 'geometric':
        #     num_hp_channel = 3
        # else:
        #     num_hp_channel = 4
        # # TODO: check hyperparameter names?
        # if set(self.get_hypernames()) != set(list(self.hyperBounds.keys())):
        #     raise ValueError('set_hyperBoundLocal has incorrect hyperparameters')
        # if dt_transform.isna().any(axis=None):
        #     raise ValueError('input data includes NaN')
        # if np.isinf(dt_transform).any():
        #     raise ValueError('input data includes Inf')

        return None

    def robyn_engineering(self, dt):
        """

            :param dt:
            :param dt_holiday:
            :param d:
            :param set_lift:
            :param set_hyperBoundLocal:
            :return: (DataFrame, dict)
            """


        dt_inputRollWind = dt[self.rollingWindowStartWhich:self.rollingWindowEndWhich+1]
        dt_transform = dt.copy().reset_index()
        dt_transform = dt_transform.rename({self.date_var: 'ds'}, axis=1)
        dt_transform['ds'] = pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d')
        dt_transform = dt_transform.rename({self.dep_var: 'depVar'}, axis=1)
        dt_transformRollWind = dt_transform[self.rollingWindowStartWhich:self.rollingWindowEndWhich+1]

        self.df_holidays['ds'] = pd.to_datetime(self.df_holidays['ds'], format='%Y-%m-%d')
        # # check date format
        # try:
        #     pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d', errors='raise')
        # except ValueError:
        #     print('input date variable should have format "yyyy-mm-dd"')
        #
        # # check variable existence
        # if not self.activate_prophet:
        #     self.prophet = None
        #     self.prophetVarSign = None
        #
        # if not self.activate_baseline:
        #     self.baseVarName = None
        #     self.baseVarSign = None
        #
        # if not self.activate_calibration:
        #     self.lift = None
        #
        # try:
        #     self.mediaSpendName
        # except NameError:
        #     print('set_mediaSpendName must be specified')
        #
        # if len(self.mediaVarName) != len(self.mediaVarSign):
        #     raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')
        #
        # trainSize = round(dt_transform.shape[0] * d['set_modTrainSize'])
        # dt_train = dt_transform[self.mediaVarName].iloc[:trainSize, :]
        # train_all0 = dt_train.loc[:, dt_train.sum(axis=0) == 0]
        # if train_all0.shape[1] != 0:
        #     raise ValueError('These media channels contains only 0 within training period. '
        #                      'Recommendation: increase set_modTrainSize, remove or combine these channels')
        #
        # dayInterval = dt_transform['ds'].nlargest(2)
        # dayInterval = (dayInterval.iloc[0] - dayInterval.iloc[1]).days
        # if dayInterval == 1:
        #     intervalType = 'day'
        # elif dayInterval == 7:
        #     intervalType = 'week'
        # elif 28 <= dayInterval <= 31:
        #     intervalType = 'month'
        # else:
        #     raise ValueError('input data has to be daily, weekly or monthly')
        # self.dayInterval = dayInterval
        # mediaVarCount = len(self.mediaVarName)

        ################################################################
        #### model reach metric from spend
        mediaCostFactor = pd.DataFrame(dt_inputRollWind[self.paid_media_spends].sum(axis=0),
                                       columns=['total_spend']).reset_index()
        var_total = pd.DataFrame(dt_inputRollWind[self.paid_media_vars].sum(axis=0), columns=['total_var']).reset_index()
        mediaCostFactor['mediaCostFactor'] = mediaCostFactor['total_spend'] / var_total['total_var']
        mediaCostFactor = mediaCostFactor.drop(columns=['total_spend'])
        costSelector = pd.Series(self.paid_media_spends) != pd.Series(self.paid_media_vars)

        if len(costSelector) != 0:
            modNLSCollect = defaultdict()
            yhatCollect = []
            plotNLSCollect = []
            for i in range(self.mediaVarCount - 1):
                if costSelector[i]:
                    dt_spendModInput = pd.DataFrame(dt_transform.loc[:, self.paid_media_spends[i]])
                    dt_spendModInput['reach'] = dt_transform.loc[:, self.paid_media_vars[i]]
                    dt_spendModInput.loc[
                        dt_spendModInput[self.paid_media_spends[i]] == 0, self.paid_media_spends[i]] = 0.01
                    dt_spendModInput.loc[dt_spendModInput['reach'] == 0, 'reach'] = \
                        dt_spendModInput[dt_spendModInput['reach'] == 0][self.paid_media_spends[i]] / \
                        mediaCostFactor['mediaCostFactor'][i]

                    # Michaelis-Menten model
                    # vmax = max(dt_spendModInput['reach'])/2
                    # km = max(dt_spendModInput['reach'])
                    # y = michaelis_menten(dt_spendModInput[d['set_mediaSpendName'][i]], vmax, km)
                    try:
                        popt, pcov = curve_fit(self.michaelis_menten, dt_spendModInput[self.paid_media_spends[i]],
                                           dt_spendModInput['reach'])

                        yhatNLS = self.michaelis_menten(dt_spendModInput[self.paid_media_spends[i]], *popt)
                    except ValueError:
                        print('michaelis menten fitting for' + str(self.paid_media_vars[i]) + ' out of range. using lm instead')
                        popt = None
                        pcov = None
                        yhatNLS = None
                    # nls_pred = yhatNLS.predict(np.array(dt_spendModInput[d['set_mediaSpendName'][i]]).reshape(-1, 1))

                    # linear model
                    lm = LinearRegression().fit(np.array(dt_spendModInput[self.paid_media_spends[i]])
                                                .reshape(-1, 1), np.array(dt_spendModInput['reach']).reshape(-1, 1))
                    lm_pred = lm.predict(np.array(dt_spendModInput[self.paid_media_spends[i]]).reshape(-1, 1))

                    # compare NLS & LM, takes LM if NLS fits worse
                    rsq_nls = r2_score(dt_spendModInput['reach'], yhatNLS)
                    rsq_lm = r2_score(dt_spendModInput['reach'], lm_pred)
                    costSelector[i] = rsq_nls > rsq_lm

                    modNLSCollect[self.paid_media_spends[i]] = {'vmax': popt[0], 'km': popt[1], 'rsq_lm': rsq_lm,
                                                             'rsq_nls': rsq_nls, 'coef_lm': lm.coef_}

                    yhat_dt = pd.DataFrame(dt_spendModInput['reach']).rename(columns={'reach': 'y'})
                    yhat_dt['channel'] = self.paid_media_vars[i]
                    yhat_dt['x'] = dt_spendModInput[self.paid_media_spends[i]]
                    yhat_dt['yhat'] = yhatNLS if costSelector[i] else lm_pred
                    yhat_dt['models'] = 'nls' if costSelector[i] else 'lm'
                    yhatCollect.append(yhat_dt)

                    # TODO: generate plots

        self.plotNLSCollect = plotNLSCollect
        self.modNLSCollect = modNLSCollect
        # d['yhatNLSCollect'] = yhatNLSCollect

        getSpendSum = pd.DataFrame(dt_transform[self.paid_media_spends].sum(axis=0), columns=['total_spend']).T

        self.mediaCostFactor = mediaCostFactor
        self.costSelector = costSelector
        self.getSpendSum = getSpendSum

        ################################################################
        #### clean & aggregate data
        # all_name = [['ds'], ['depVar'], self.prophet_vars, self.context_vars, self.paid_media_vars]
        # all_name = set([item for sublist in all_name for item in sublist])
        # # all_mod_name = [['ds'], ['depVar'], d['set_prophet'], d['set_baseVarName'], d['set_mediaVarName']]
        # # all_mod_name = [item for sublist in all_mod_name for item in sublist]
        # if len(all_name) != len(set(all_name)):
        #     raise ValueError('Input variables must have unique names')

        ## transform all factor variables
        if self.factor_vars:
            if len(self.factor_vars) > 0:
                dt_transform[self.factor_vars].apply(lambda x: x.astype('category'))
            else:
                self.factor_vars = None

        ################################################################
        #### Obtain prophet trend, seasonality and changepoints

        if self.prophet_vars:
            if len(self.prophet_vars) != len(self.prophet_signs):
                raise ValueError('prophet_vars and prophet_signs have to be the same length')
            if len(self.prophet_vars) == 0 or len(self.prophet_signs) == 0:
                raise ValueError('if activate_prophet == TRUE, set_prophet and set_prophetVarSign must to specified')
            if self.prophet_country not in self.df_holidays['country'].values:
                raise ValueError(
                    'set_country must be already included in the holidays.csv and as ISO 3166-1 alpha-2 abbreviation')

            recurrence = dt_transform.copy().rename(columns={'depVar': 'y'})
            use_trend = True if 'trend' in self.prophet_vars else False
            use_season = True if 'season' in self.prophet_vars else False
            use_weekday = True if 'weekday' in self.prophet_vars else False
            use_holiday = True if 'holiday' in self.prophet_vars else False

            if self.intervalType == 'day':
                holidays = self.df_holidays
            elif self.intervalType == 'week':
                weekStartInput = dt_transform['ds'][0].weekday()
                if weekStartInput == 0:
                    weekStartMonday = True
                elif weekStartInput == 6:
                    weekStartMonday = False
                else:
                    raise ValueError('week start has to be Monday or Sunday')
                self.df_holidays['weekday'] = self.df_holidays['ds'].apply(lambda x: x.weekday())
                self.df_holidays['dsWeekStart'] = self.df_holidays.apply(lambda x: x['ds'] - timedelta(days=x['weekday']), axis=1)
                self.df_holidays['ds'] = self.df_holidays['dsWeekStart']
                self.df_holidays = self.df_holidays.drop(['dsWeekStart', 'weekday'], axis=1)
                holidays = self.df_holidays.groupby(['ds', 'country', 'year'])['holiday'].apply(
                    lambda x: '#'.join(x)).reset_index()

            elif self.intervalType == 'month':
                monthStartInput = dt_transform['ds'][0].strftime("%d")
                if monthStartInput != '01':
                    raise ValueError("monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
                self.df_holidays['month'] = self.df_holidays['ds'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(normalize=True)
                self.df_holidays['ds'] = self.df_holidays['month']
                self.df_holidays.drop(['month'], axis=1)
                holidays = self.df_holidays.groupby(['ds', 'country', 'year'])['holiday'].apply(
                    lambda x: '#'.join(x)).reset_index()
            h = holidays[holidays['country'] == self.country] if use_holiday else None
            modelRecurrance = Prophet(holidays=h, yearly_seasonality=use_season, weekly_seasonality=use_weekday,
                                      daily_seasonality=False)
            modelRecurrance.fit(recurrence)
            forecastRecurrance = modelRecurrance.predict(dt_transform)

            self.modelRecurrance = modelRecurrance
            self.forecastRecurrance = forecastRecurrance

            # python implementation of scale() is different from R, may need to hard-code the R equivalent
            if use_trend:
                fc_trend = forecastRecurrance['trend'][:recurrence.shape[0]]
                fc_trend = preprocessing.scale(fc_trend)
                dt_transform['trend'] = fc_trend
            if use_season:
                fc_season = forecastRecurrance['yearly'][:recurrence.shape[0]]
                fc_season = preprocessing.scale(fc_season)
                dt_transform['seasonal'] = fc_season
            if use_weekday:
                fc_weekday = forecastRecurrance['weekly'][:recurrence.shape[0]]
                fc_weekday = preprocessing.scale(fc_weekday)
                dt_transform['weekday'] = fc_weekday
            if use_holiday:
                fc_holiday = forecastRecurrance['holidays'][:recurrence.shape[0]]
                fc_holiday = preprocessing.scale(fc_holiday)
                dt_transform['trend'] = fc_holiday

        ################################################################
        #### Finalize input
        # dt_transform < - dt_transform[, c("ds", "dep_var", all_ind_vars),
        #with = FALSE]

        self.dt_mod = dt_transform
        self.dt_modRollWind = dt_transform[self.rollingWindowStartWhich:self.rollingWindowEndWhich+1]
        self.dt_inputRollWind = dt_inputRollWind
        self.modNLSCollect = modNLSCollect
        self.plotNLSCollect = plotNLSCollect
        self.yhatNLSCollect = yhatNLSCollect
        self.costSelector = costSelector
        self.mediaCostFactor = mediaCostFactor

        return None



    def set_param_bounds(self):
        """

        :return:
        """

        pass

    def get_hypernames(self):

        if self.adstock_type == "geometric":
            global_name = ["thetas", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in self.mediaVarName for j in global_name]))
        elif self.adstock_type == "weibull":
            global_name = ["shapes", "scales", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in self.mediaVarName for j in global_name]))

        return local_name

    @staticmethod
    def michaelis_menten(spend, vmax, km):
        """
            :param vmax:
            :param spend:
            :param km:
            :return: float
            """

        return vmax * spend / (km + spend)

    @staticmethod
    def adstockGeometric(x, theta):
        """
            :param x:
            :param theta:
            :return: numpy
            """

        x_decayed = [x[0]] + [0] * (len(x) - 1)
        for i in range(1, len(x_decayed)):
            x_decayed[i] = x[i] + theta * x_decayed[i - 1]

        thetaVecCum = theta
        for t in range(1,len(x)):
            thetaVecCum[t] = thetaVecCum[t-1] * theta

        return x_decayed, thetaVecCum


    @staticmethod
    def helperWeibull(x, y, vec_cum, n):
        """
            :param x:
            :param y:
            :param vec_cum:
            :param n:
            :return: numpy
            """

        x_vec = np.array([0] * (y - 1) + [x] * (n - y + 1))
        vec_lag = np.roll(vec_cum, y - 1)
        vec_lag[: y - 1] = 0
        x_matrix = np.c_[x_vec, vec_lag]
        x_prod = np.multiply.reduce(x_matrix, axis=1)
        print(x_prod)
        return x_prod

    def adstockWeibull(self, x, shape, scale):
        """
            Parameters
            ----------
            x: numpy array
            shape: shape parameter for Weibull
            scale: scale parameter for Weibull
            Returns
            -------
            (list, list)
            """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

        n = len(x)
        bin = list(range(1, n + 1))
        scaleTrans = round(np.quantile(bin, scale))
        thetaVec = 1 - stats.weibull_min.cdf(bin[:-1], shape, scale=scaleTrans)
        thetaVec = np.concatenate(([1], thetaVec))
        thetaVecCum = np.cumprod(thetaVec).tolist()

        x_decayed = list(map(lambda i, j: self.helperWeibull(i, j, vec_cum=thetaVecCum, n=n), x, bin))
        x_decayed = np.concatenate(x_decayed, axis=0).reshape(n, n)
        x_decayed = np.transpose(x_decayed)
        x_decayed = np.sum(x_decayed, axis=1).tolist()

        return x_decayed, thetaVecCum

    def transformation(self, x, adstock, theta=None, shape=None, scale=None, alpha=None, gamma=None, stage=3):
        """
            ----------
            Parameters
            ----------
            x: vector
            adstock: chosen adstock (geometric or weibull)
            theta: decay coefficient
            shape: shape parameter for weibull
            scale: scale parameter for weibull
            alpha: hill function parameter
            gamma: hill function parameter
            Returns
            -------
            s-curve transformed vector
            """

        ## step 1: add decay rate
        if adstock == "geometric":
            x_decayed = self.adstockGeometric(x, theta)

            if stage == "thetaVecCum":
                thetaVecCum = theta
            for t in range(1, len(x) - 1):
                thetaVecCum[t] = thetaVecCum[t - 1] * theta
            # thetaVecCum.plot()

        elif adstock == "weibull":
            x_list = self.adstockWeibull(x, shape, scale)
            x_decayed = x_list['x_decayed']
            # x_decayed.plot()

            if stage == "thetaVecCum":
                thetaVecCum = x_list['thetaVecCum']
            # thetaVecCum.plot()

        else:
            print("alternative must be geometric or weibull")

        ## step 2: normalize decayed independent variable # deprecated
        # x_normalized = x_decayed

        ## step 3: s-curve transformation
        gammaTrans = round(np.quantile(np.linspace(min(x_decayed), max(x_decayed), 100), gamma), 4)
        x_scurve = x_decayed ** alpha / (x_decayed ** alpha + gammaTrans ** alpha)
        # x_scurve.plot()
        if stage in [1, 2]:
            x_out = x_decayed
        # elif stage == 2:
        # x_out = x_normalized
        elif stage == 3:
            x_out = x_scurve
        elif stage == "thetaVecCum":
            x_out = thetaVecCum
        else:
            raise ValueError(
                "hyperparameters out of range. theta range: 0-1 (excl.1), shape range: 0-5 (excl.0), alpha range: 0-5 (excl.0),  gamma range: 0-1 (excl.0)")

        return x_out

    @staticmethod
    def unit_format(x_in):
        """
            Define helper unit format function for axis
            :param x_in: a number in decimal or float format
            :return: the number rounded and in certain cases abbreviated in the thousands, millions, or billions
            """

        # suffixes = ["", "Thousand", "Million", "Billion", "Trillion", "Quadrillion"]
        number = str("{:,}".format(x_in))
        n_commas = number.count(',')
        # print(number.split(',')[0], suffixes[n_commas])

        if n_commas >= 3:
            x_out = f'{round(x_in / 1000000000, 1)} bln'
        elif n_commas == 2:
            x_out = f'{round(x_in / 1000000, 1)} mio'
        elif n_commas == 1:
            x_out = f'{round(x_in / 1000, 1)} tsd'
        else:
            x_out = str(int(round(x_in, 0)))

        return x_out

    @staticmethod
    def get_rsq(val_actual, val_predicted, p = None, df_int = None ):
        # Changed "true" to val_actual because Python could misinterpret True
        """
        :param val_actual: actual value
        :param val_predicted: predicted value
        :param p: number of independent variable
        :param p: number of independent variable
        :return: r-squared
        """
        sse = sum((val_predicted - val_actual) ** 2)
        sst = sum((val_actual - sum(val_actual) / len(val_actual)) ** 2)
        rsq = 1 - sse / sst

        # adjusted rsq formula / # n = num_obs, p = num_indepvar, rdf = n-p-1
        if (p is not None) & (df_int is not None):
            n = len(val_actual)
            rdf = n - p - 1
            rsq = 1- (1 - rsq) * ((n - df_int) / rdf)

        return rsq

    @staticmethod
    def lambdaRidge(x, y, seq_len=100, lambda_min_ratio=0.0001):
        """
            ----------
            Parameters
            ----------
            x: matrix
            y: vector
            Returns
            -------
            lambda sequence
            """

        def mysd(y):
            return math.sqrt(sum((y - sum(y) / len(y)) ** 2) / len(y))

        sx = x.apply(mysd)
        sx = preprocessing.scale(x).T
        sy = y.to_numpy()
        sxy = sx * sy
        sxy = sxy.T
        # return sxy
        lambda_max = max(abs(sxy.sum(axis=0)) / (
                0.001 * x.shape[0]))  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
        lambda_max_log = math.log(lambda_max)

        log_step = (math.log(lambda_max) - math.log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
        log_seq = np.linspace(math.log(lambda_max), math.log(lambda_max * lambda_min_ratio), seq_len)
        lambda_seq = np.exp(log_seq)
        return lambda_seq

    def model_decomp(self, coefs, dt_mod_saturated, x, y_pred, i, dt_mod_rollwind, refresh_added_start):

        """
        :param coef: Pandas Series with index name
        :param dt_modAdstocked: Pandas Dataframe
        :param x: Pandas Dataframe
        :param y_pred: Pandas Series
        :param i: interger
        :return: Collection of decomposition output
        """

        ## input for decomp
        y = dt_mod_saturated["depVar"]
        indepVar = dt_mod_saturated.loc[:, dt_mod_saturated.columns != 'depVar']
        intercept = coefs.iloc[0]
        indepVarName = indepVar.columns.tolist()
        indepVarCat = indepVar.select_dtypes(['category']).columns.tolist()

        ## decomp x
        xDecomp = x * coefs.iloc[1:]
        xDecomp.insert(loc=0, column='intercept', value=[intercept] * len(x))
        xDecompOut = pd.concat([pd.DataFrame({'ds': dt_mod_rollwind["ds"], 'y': y, 'y_pred': y_pred}),
                                xDecomp], axis=1)

        ## QA decomp
        y_hat = xDecomp.sum(axis=1)
        errorTerm = y_hat - y_pred
        if np.prod(round(y_pred) == round(y_hat)) == 0:
            print(
                "### attention for loop " + str(i) + \
                ": manual decomp is not matching linear model prediction. Deviation is " + \
                str(np.mean(errorTerm / y) * 100) + "% ###"
            )

        ## output decomp
        y_hat_scaled = abs(xDecomp).sum(axis=1)
        xDecompOutPerc_scaled = abs(xDecomp).div(y_hat_scaled, axis=0)
        xDecompOut_scaled = xDecompOutPerc_scaled.multiply(y_hat, axis=0)

        xDecompOutAgg = xDecompOut[['intercept'] + indepVarName].sum(axis=0)
        xDecompOutAggPerc = xDecompOutAgg / sum(y_hat)
        xDecompOutAggMeanNon0 = xDecompOut[['intercept'] + indepVarName].mean(axis=0).clip(lower=0)
        xDecompOutAggMeanNon0Perc = xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)

        refreshAddedStartWhich = xDecompOut["ds"][xDecompOut["ds"] == refresh_added_start].index[0]
        refreshAddedEnd = xDecompOut["ds"].iloc[-1]
        refreshAddedEndWhich = xDecompOut["ds"][xDecompOut["ds"] == refreshAddedEnd].index[0]

        xDecompOutAggRF = xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich][['intercept'] + indepVarName].sum(axis=0)
        y_hatRF = y_hat[refreshAddedStartWhich:refreshAddedEndWhich]
        xDecompOutAggPercRF = xDecompOutAggRF / sum(y_hatRF)
        xDecompOutAggMeanNon0RF = xDecompOut[refreshAddedStartWhich:refreshAddedEndWhich][['intercept'] + indepVarName].mean(axis=0).clip(lower=0)
        xDecompOutAggMeanNon0PercRF = xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

        coefsOut = coefs.reset_index(inplace=False)
        coefsOutCat = coefsOut.copy()
        coefsOut = coefsOut.rename(columns={'index': 'rn'})
        if len(indepVarCat) == 0:
            pass
        else:
            for var in indepVarCat:
                coefsOut.rn.replace(r'(^.*' + var + '.*$)', var, regex=True, inplace=True)
        coefsOut = coefsOut.groupby(coefsOut['rn'], sort=False).mean().reset_index()

        frame = {'xDecompAgg': xDecompOutAgg,
                 'xDecompPerc': xDecompOutAggPerc,
                 'xDecompMeanNon0': xDecompOutAggMeanNon0,
                 'xDecompMeanNon0Perc': xDecompOutAggMeanNon0Perc,
                 'xDecompAggRF': xDecompOutAggRF,
                 'xDecompPercRF': xDecompOutAggPercRF,
                 'xDecompMeanNon0RF': xDecompOutAggMeanNon0RF,
                 'xDecompMeanNon0PercRF': xDecompOutAggMeanNon0PercRF
                 }
        frame.index = coefsOut.index
        decompOutAgg = pd.merge(coefsOut, frame, left_index=True, right_index=True)
        decompOutAgg['pos'] = decompOutAgg['xDecompAgg'] >= 0

        decompCollect = {'xDecompVec': xDecompOut,
                         'xDecompVec_scaled': xDecompOut_scaled,
                         'xDecompAgg': decompOutAgg,
                         'coefsOutCat': coefsOutCat}

        return decompCollect

    ########################
    # TODO calibrateLift -> calibrate_mmm

    def calibrate_mmm(self, decompCollect, set_lift, set_mediaVarName):

         lift_channels = list(set_lift.channel)
         check_set_lift = all(item in set_mediaVarName for item in lift_channels)
         if check_set_lift:
             getLiftMedia = list(set(lift_channels))
             getDecompVec = decompCollect['xDecompVec']
         else:
             exit("set_lift channels must have media variable")

         # loop all lift input
         liftCollect = pd.DataFrame(columns = ['liftMedia', 'liftStart', 'liftEnd' ,
                                               'liftAbs', 'decompAbsScaled', 'dependent'])
         for m in getLiftMedia: # loop per lift channel
             liftWhich = list(set_lift.loc[set_lift.channel.isin([m])].index)
             liftCollect2 = pd.DataFrame(columns = ['liftMedia', 'liftStart', 'liftEnd' ,
                                                    'liftAbs', 'decompAbsScaled', 'dependent'])
             for lw in liftWhich: # loop per lift test per channel
                 # get lift period subset
                 liftStart = set_lift['liftStartDate'].iloc[lw]
                 liftEnd = set_lift['liftEndDate'].iloc[lw]
                 liftAbs = set_lift['liftAbs'].iloc[lw]
                 liftPeriodVec = getDecompVec[['ds', m]][(getDecompVec.ds >= liftStart) & (getDecompVec.ds <= liftEnd)]
                 liftPeriodVecDependent = getDecompVec[['ds', 'y']][(getDecompVec.ds >= liftStart) & (getDecompVec.ds <= liftEnd)]

                 # scale decomp
                 mmmDays = len(liftPeriodVec)*7
                 liftDays = abs((liftEnd - liftStart).days) + 1
                 y_hatLift = getDecompVec['y_hat'].sum() # total predicted sales
                 x_decompLift = liftPeriodVec.iloc[:1].sum()
                 x_decompLiftScaled = x_decompLift / mmmDays * liftDays
                 y_scaledLift = liftPeriodVecDependent['y'].sum() / mmmDays * liftDays

                 # output
                 list_to_append = [[getLiftMedia[m], liftStart, liftEnd, liftAbs, x_decompLiftScaled, y_scaledLift]]
                 liftCollect2 = liftCollect2.append(pd.DataFrame(list_to_append,
                                                                 columns = ['liftMedia', 'liftStart', 'liftEnd' ,
                                                                            'liftAbs', 'decompAbsScaled', 'dependent'],
                                                                 ignore_index = True))
             liftCollect = liftCollect.append(liftCollect2, ignore_index = True)
         #get mape_lift
         liftCollect['mape_lift'] = abs((liftCollect['decompAbsScaled'] - liftCollect['liftAbs']) / liftCollect['liftAbs'])

         return liftCollect

    def refit(self, x_train, y_train, lambda_: int, lower_limits: list, upper_limits: list):

        # Call R functions - to match outputs of Robyn in R
        numpy2ri.activate()

        # Define glmnet model in r
        ro.r('''
                r_glmnet <- function (x, y, family, alpha, lambda_, lower_limits, upper_limits, intercept) {

                    library(glmnet)

                    if(intercept == 1){
                    # print("Intercept")
                    mod <- glmnet(
                        x=x,
                        y=y,
                        family=family,
                        alpha=alpha,
                        lambda=lambda_,
                        lower.limits=lower_limits,
                        upper.limits=upper_limits,
                        )
                    } else {
                    # print("No Intercept")
                    mod <- glmnet(
                        x=x,
                        y=y,
                        family=family,
                        alpha=alpha,
                        lambda=lambda_,
                        lower.limits=lower_limits,
                        upper.limits=upper_limits,
                        intercept=FALSE
                        )
                    }
                }
            ''')
        r_glmnet = ro.globalenv['r_glmnet']

        # Create model
        mod = r_glmnet(x=x_train,
                       y=y_train,
                       alpha=1,
                       family="gaussian",
                       lambda_=lambda_,
                       lower_limits=lower_limits,
                       upper_limits=upper_limits,
                       intercept=True
                       )

        # Create model without the intercept if negative
        if mod[0][0] < 0:
            mod = r_glmnet(x=x_train,
                           y=y_train,
                           alpha=1,
                           family="gaussian",
                           lambda_=lambda_,
                           lower_limits=lower_limits,
                           upper_limits=upper_limits,
                           intercept=False
                           )

        df_int = 0 if mod[0][0] < 0 else 1

        # Run model
        ro.r('''
                    r_predict <- function(model, s, newx) {
                        predict(model, s=s, newx=newx)
                    }
                ''')
        r_predict = ro.globalenv['r_predict']
        y_train_pred = r_predict(model=mod, s=1, newx=x_train)
        y_train_pred = y_train_pred.reshape(len(y_train_pred), )  # reshape to be of format (n,)

        # Calc r-squared on training set
        rsq_train = self.get_rsq(val_actual=y_train, val_predicted=y_train_pred, p=len(x_train[0]), df_int=df_int)

        # Get coefficients
        coefs = mod[0]

        # Get normalized root mean square error
        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2) / (max(y_train) - min(y_train)))

        # Update model outputs to include calculated values
        mod_out = {'rsq_train': rsq_train,
                   'nrmse_train': nrmse_train,
                   'coefs': coefs,
                   'y_pred': y_train_pred,
                   'mod': mod,
                   'df_int': df_int}

        self.mod = mod_out

    def mmm(self,
            df,
            adstock_type='geometric',
            optimizer_name='DiscreteOnePlusOne',
            set_iter=100,
            set_cores=6,
            lambda_n=100,
            fixed_out=False,
            fixed_lambda=None):  # This replaces the original mmm + Robyn functions

        ################################################
        # Collect hyperparameters

        # Expand media spend names to to have the hyperparameter names needed based on adstock type
        names_hyper_parameter_sample_names = \
            self.get_hypernames(names_media_variables=self.names_media_spend, adstock_type=adstock_type)
        # names_hyper_parameter_sample_names = \
        #     get_hypernames(names_media_variables=names_media_spend, adstock_type=adstock_type)

        if not fixed_out:
            # input_collect = # todo not sure what this is.  Finish it.
            # todo collects results for parameters?
            input_collect = self.hyperBoundLocal
            # input_collect = None

        ################################################
        # Get spend share

        ################################################
        ### Setup environment

        try:
            self.dt_mod
        except NameError:
            print("robyn_engineering() first to get the dt_mod")

        ## get environment for parallel backend
        dt_input = self.dt_input
        dt_mod = self.dt_mod.copy()
        xDecompAggPrev = self.xDecompAggPrev
        rollingWindowStartWhich = self.rollingWindowStartWhich
        rollingWindowEndWhich = self.rollingWindowEndWhich
        refreshAddedStart = self.refreshAddedStart
        dt_modRollWind = self.dt_modRollWind
        refresh_steps = self.refresh_steps
        rollingWindowLength = self.rollingWindowLength

        paid_media_vars = self.paid_media_vars
        paid_media_spends = self.paid_media_spends
        organic_vars = self.organic_vars
        context_vars = self.context_vars
        prophet_vars = self.prophet_vars
        adstock = self.adstock
        context_signs = self.context_signs
        paid_media_signs = self.paid_media_signs
        prophet_signs = self.prophet_signs
        organic_signs = self.organic_signs
        all_media = self.all_media
        #factor_vars = self.factor_vars
        calibration_input = self.calibration_input
        nevergrad_algo = self.nevergrad_algo
        cores = self.cores


        ################################################
        # Start Nevergrad loop

        # Set iterations
        self.iterations = set_iter
        x = None
        if x == 'fake':
            iter = self.iterations

        # Start Nevergrad optimiser

        # Start loop

        # Get hyperparameter sample with ask

        # Scale sample to given bounds

        # Add fixed hyperparameters

        # Parallel start

        #####################################
        # Get hyperparameter sample

        # Tranform media with hyperparameters

        #####################################
        # Split and prepare data for modelling

        # Contrast matrix because glmnet does not treat categorical variables

        # Define sign control

        #####################################
        ### Fit ridge regression with x-validation

        # TODO discussion on utlizing python glmnet instead of calling r function
        # https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html
        # Seem to not work with windows

        # Call R functions - to match outputs of Robyn in R
        numpy2ri.activate()

        # Define cv.glmnet model in r
        ro.r('''
                r_cv_glmnet <- function (x, y, family, alpha, lower_limits, upper_limits, type_measure) {
                    library(glmnet)
                    mod <- cv.glmnet(
                            data.matrix(x),
                            y=y,
                            family=family,
                            alpha=alpha,
                            lower.limits=lower_limits,
                            upper.limits=upper_limits,
                            type.measure = type_measure
                            )              
                }
            ''')
        r_cv_glmnet = ro.globalenv['r_cv_glmnet']

        # Create model
        cvmod = r_cv_glmnet(x=x_train,
                            y=ro.FloatVector(y_train),
                            family="gaussian",
                            alpha=0,
                            #lambda_=lambda_,
                            lower_limits=lower_limits,
                            upper_limits=upper_limits,
                            type_measure="mse"
                            )

        #TODO remove this section after de-bugging
        '''
        x = np.arange(1, 25).reshape(12, 2)
        y = ro.FloatVector(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]))

        mod = r_cv_glmnet(x=x,
                          y=y,
                          family="gaussian",
                          alpha=0,
                          lower_limits=0,
                          upper_limits=1,
                          type_measure="mse"
                          )
        mod[10]
        '''


        #####################################
        ### Refit ridge regression with selected lambda from x-validation


        # If no lift calibration, refit using best lambda
        if fixed_out:
            mod_out = self.refit(x_train, y_train, lambda_=cvmod[10], lower_limits, upper_limits)
            lambda_ = cvmod[10]
        else:
            mod_out = self.refit(x_train, y_train, lambda_=cvmod[0][i], lower_limits, upper_limits)
            lambda_ = cvmod[0][i]

        decomp_collect = self.model_decomp(coefs=mod_out['coefs'], dt_modSaturated = dt_modSaturated, x = x_train, y_pred = mod_out['y_pred'], i=i, dt_mod_rollwind = dt_modRollWind, refresh_added_start= refreshAddedStart)

        nrmse = mod_out['nrmse_train']
        mape = 0
        df_int = mod_out['df_int']

        #####################################
        # Get calibration mape

        if self.activate_calibration:



        #####################################
        # Calculate multi-objectives for pareto optimality

        # Decomp objective: sum of squared distance between decomp share and spend share to be minimised

        # Adstock objective: sum of squared infinite sum of decay to be minimised? maybe not necessary

        # Calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift

        #####################################
        # Collect output

        # End dopar
        # End parallel

        #####################################
        # Nevergrad tells objectives

        # End NG loop
        # End system.time

        #####################################
        # Get nevergrad pareto results

        #####################################
        # Final result collect

    def fit(self,
            optimizer_name=None,
            set_trial=None,
            set_cores=None,
            fixed_out=False,
            fixed_hyppar_dt=None,
            pareto_fronts=[1,2,3]
            ):

        if optimizer_name is None:
            optimizer_name = self.hyperOptimAlgo
        if set_trial is None:
            set_trial = self.trial
        if set_cores is None:
            set_cores = self.cores
        if fixed_hyppar_dt is None:
            fixed_hyppar_dt = self.fixed_hyppar_dt

        #plot_folder = os.getcwd()
        #pareto_fronts = np.array[1, 2, 3]

        ### start system time

        t0 = time.time()

        ### check if plotting directory exists

        # if (!dir.exists(plot_folder)) {
        # plot_folder < - getwd()
        # message("provided plot_folder doesn't exist. Using default plot_folder = getwd(): ", getwd())
        # }

        ### run mmm function on set_trials
        ## todo set_hyperBoundLocal type?? assume dict
        hyperparameter_fixed = all(value == 0 for value in self.hyperBounds.values())
        hypParamSamName = self.get_hypernames()

        if fixed_out:

            ### run mmm function if using old model result tables

            if fixed_hyppar_dt is None:
                raise ValueError(
                    'when fixed_out=T, please provide the table model_output_resultHypParam from previous runs or '
                    'pareto_hyperparameters.csv with desired model IDs')
            if not all([True for x in hypParamSamName.append('lambda') if x in fixed_hyppar_dt.columns]):
                raise ValueError('fixed.hyppar.dt is provided with wrong input. '
                                 'please provide the table model_output_collect$resultHypParam from previous runs or '
                                 'pareto_hyperparameters.csv with desired model ID')

            model_output_collect = {}
            model_output_collect['resultHypParam'] = self.mmm(fixed_hyppar_dt['hypParamSamName'], set_iter=self.iter,
                                                              set_cores=set_cores, optimizer_name=optimizer_name,
                                                              fixed_out=True, fixed_lambda=list(fixed_hyppar_dt['lambda']))
            model_output_collect['resultHypParam'] = model_output_collect['resultHypParam']['trials'] = 1
            model_output_collect['resultHypParam']['resultCollect']['resultHypParam'] = \
                model_output_collect['resultHypParam']['resultCollect']['resultHypParam'].sort_values(by='iterPar')
            dt_IDmatch = pd.DataFrame({'solID': fixed_hyppar_dt['solID'],
                                       'iterPar': model_output_collect['resultHypParam']['resultCollect']['resultHypParam']['iterPar']})

            model_output_collect['resultHypParam']['resultCollect']['resultHypParam'] = \
                pd.merge(model_output_collect['resultHypParam']['resultCollect']['resultHypParam'], dt_IDmatch,
                         on='iterPar')
            model_output_collect['resultHypParam']['resultCollect']['xDecompAgg'] = \
                pd.merge(model_output_collect['resultHypParam']['resultCollect']['xDecompAgg'], dt_IDmatch,
                         on='iterPar')
            model_output_collect['resultHypParam']['resultCollect']['xDecompVec'] = \
                pd.merge(model_output_collect['resultHypParam']['resultCollect']['xDecompVec'], dt_IDmatch,
                         on='iterPar')
            model_output_collect['resultHypParam']['resultCollect']['decompSpendDist'] = \
                pd.merge(model_output_collect['resultHypParam']['resultCollect']['decompSpendDist'], dt_IDmatch,
                         on='iterPar')

            print("\n######################\nHyperparameters are all fixed\n######################\n")
            print(model_output_collect['resultHypParam']['resultCollect']['xDecompAgg'])

        elif hyperparameter_fixed:
        ## Run f.mmm on set_trials if hyperparameters are all fixed

            model_output_collect = {}
            model_output_collect['resultHypParam'] = self.mmm(self.hyperBounds, set_iter = 1, set_cores = 1,
                                                 optimizer_name = optimizer_name)

            model_output_collect['resultHypParam'] = model_output_collect['resultHypParam']['trials'] = 1
            print("\n######################\nHyperparameters are all fixed\n######################\n")
            print(model_output_collect['resultHypParam']['resultCollect']['xDecompAgg'])

        else:
            ng_out = {}
            ng_algos = optimizer_name
            t0 = time.time()
            for optmz in ng_algos:
                ng_collect = {}
                model_output_collect = {}
                for ngt in range(set_trial-1):
                    if not self.activate_calibration:
                        print("\nRunning trial nr.", ngt,"out of",set_trial,"...\n")
                    else:
                        print("\nRunning trial nr.", ngt,"out of",set_trial,"with calibration...\n")
                    ## todo here we are assume model_output to be nested dict
                    model_output = self.mmm(self.hyperBounds, set_iter=self.iter, set_cores=self.cores,
                                            optimizer_name=optmz)
                    check_coef0 = any(model_output['resultCollect']['decompSpendDist']['decomp.rssd'] == math.inf)
                    if check_coef0:
                        num_coef0_mod = model_output
                        if num_coef0_mod > self.iter:
                            num_coef0_mod = self.iter
                        print("\nThis trial contains ", num_coef0_mod," iterations with all 0 media coefficient. Please "
                                                                      "reconsider your media variable choice if the pareto choices are unreasonable."
                                                                      "\nRecommendations are: \n1. increase hyperparameter ranges for 0-coef channels "
                                                                      "on theta (max.reco. c(0, 0.9) ) and gamma (max.reco. c(0.1, 1) ) to give Robyn more freedom\n2. split "
                                                                      "media into sub-channels, and/or aggregate similar channels, and/or introduce other media\n3. increase trials to get more samples\n")
                    model_output['trials'] = ngt
                    ng_collect[str(ngt)] = model_output['resultCollect']['paretoFront']
                    ng_collect[str(ngt)]['iters'] = self.iter
                    ng_collect[str(ngt)]['ng_optmz'] = optmz
                    model_output_collect[str(ngt)] = model_output

                # todo type of nglist?
                px = p.low(ng_collect['nrmse']) * p.low(ng_collect['decomp.rssd'])
                ng_collect = p.pref.psel(ng_collect, px, top=len(ng_collect)).sort_values(by=['trials','nrmse'])
                ng_out =
            ng_out = ng_out.append(ng_out)
            ng_out.rename(columns={'.level', 'manual_pareto'})

        #### Collect results for plotting

        return model_output_collect



def budget_allocator(self, model_id):  # This is the last step_model allocation
        pass
