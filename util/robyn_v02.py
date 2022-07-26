import sys
import pathlib
#from git import Repo
import numpy as np
from datetime import date
#LOGGING
import logging as logger
import rpy2.rlike.container as rlc


logger.getLogger("robyn_v02 logger")
logger.basicConfig(level=logger.DEBUG,
                    #filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{",
                   stream=sys.stdout)

try:
    ########################################################################################################################
    # IMPORTS

    ##################
    # Base Python imports
    import os.path



    import pandas as pd

    ##################
    # Python R Import
    import rpy2
    import rpy2.situation
    from rpy2 import robjects
    import rpy2.robjects as ro
    from rpy2.robjects import r, pandas2ri
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter, py2rpy
    from rpy2.robjects.packages import importr, data
    import rpy2.robjects.packages as rpackages

    global LOCAL
    LOCAL = True
    root = os.path.dirname(os.path.dirname(__file__))

    print(rpackages.isinstalled('Robyn'))
    pandas2ri.activate()  # Won't work unless activated

    ##################
    # R imports
    # robjects.r['library']('reticulate')
    # robjects.r['import']('sys')  # Gives an error
    utils = importr("utils")
    #utils.chooseCRANmirror(ind=70)
    base = importr('base')

    ##################
    # Import Robyn from R
    #try:
    #robyn = importr('Robyn')
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}

    remotes = importr("remotes")
    remotes.install_local(f"{root}/R",force=True)
    #robyn = importr('Robyn', robject_translations=d,
                    #lib_loc=os.path.join(root, "R"))

    robyn = importr('Robyn')
    #robyn = importr('Robyn', robject_translations=d,
                          #lib_loc="/Users/sinandjevdet/opt/miniconda3/envs/env_r41_new_p37/lib/R/library")



    ########################################################################################################################
    # REVIEW

    print(rpy2.__version__)

    # Setup details
    for row in rpy2.situation.iter_info():
        print(row)


    ########################################################################################################################
    # SETTINGS

    # Set seed
    set_seed = r('set.seed')
    set_seed(123)

    # Force multicore when using RStudio
    # TODO
    # robjects.r['import']('sys')  # Gives an error
    # base.Sys_getenv(R_FUTURE_FORK_ENABLE="true")
    # Sys.setenv(R_FUTURE_FORK_ENABLE="true")
    ########################################################################################################################

    # CHECK ROBYN VERSION/BRANCH

    #TODO set path of repo
    #repo = Repo('Users/sinandjevdet/Robyn')

    ########################################################################################################################


    # READ IN DATA

    # Check simulated dataset or load your own dataset
    # utils.data("dt_simulated_weekly")
    r.data('dt_simulated_weekly')
    r['dt_simulated_weekly'].head()

    # Import data then convert to R data frame ###

    class RFuncs:

        def __init__(self):
            self.find_pkg_func={'find_pkg':'''findPkgAll <- function(pkg) unlist(lapply(.libPaths(), function(lib) find.package(pkg, lib, quiet=TRUE, verbose=FALSE)))'''}

        def set_custom_r_func(self,func_code:str):
            func=robjects.r(func_code)
            return func

        def convert_to_listvector(self,data):
            return robjects.ListVector(data)

        def get_r_type(self):
            rtype=self.set_custom_r_func('typeof')
            return rtype


    rfuncs = RFuncs()
    find_pkg_func= rfuncs.set_custom_r_func(rfuncs.find_pkg_func['find_pkg'])

    logger.info(f"Robyn is in path: {find_pkg_func('Robyn')}")

    def get_r_package_version(pkg_name:str):
        r_package_version = robjects.r['packageVersion']
        return r_package_version(pkg_name)


    logger.info(f"Robyn Version: {get_r_package_version('Robyn')}")

    def get_file(path:str):
        return sorted(pathlib.Path('.').glob(f'**/{path}'))[0]

    def date_split(v:pd.Series,
                   col:str)->dict:
        year=int(v[col].split('-')[0])
        month = int(v[col].split('-')[1])
        day = int(v[col].split('-')[2])
        return {'year': year, 'month': month, 'day': day}

    def set_each_date(v:pd.Series,
                      r:int,
                    df:pd.DataFrame,
                      col:str):
        date_dict = date_split(v,col)
        try:
            df.at[r, col] = date(date_dict['year'], date_dict['month'], date_dict['day'])
        except:
            print(0)
        return df


    def set_date(df:pd.DataFrame):
        for r, v in df.iterrows():
            if 'DATE'  in v:
                n_df=set_each_date(v,r,df,'DATE')
            elif 'ds' in v:
                n_df = set_each_date(v,r, df, 'ds')
            else:
                logger.info("NO DATE")
        return df

    if LOCAL:
        #sim_week_path='/Users/sinandjevdet/PycharmProjects/Robyn/util/data/simulated_weekly.csv'
        sim_week_path= get_file('simulated_weekly.csv')
        proph_hol_path=get_file('prophet_holidays.csv')
    else:
        sim_week_path='util/data/simulated_weekly.csv'

    df_simulated = pd.read_csv(sim_week_path)  # import as pandas data frame
    df_simulated['DATE'] = pd.to_datetime(df_simulated['DATE'],yearfirst=True,format="%Y-%m-%d").dt.date #pd.to_datetime(df_simulated['DATE']).dt.strftime("%Y-%m-%d")
    r_date=base.as_Date(pd.to_datetime(df_simulated['DATE']).dt.strftime("%Y-%m-%d"))
    df_simulated['DATE'] =r_date# df_simulated.DATE.astype(str) #r_date#base.format(r_date, format="%Y-%m-%d")
    #df_simulated=set_date(df_simulated)
    print(df_simulated.head())
    del df_simulated['row_num']
    with localconverter(ro.default_converter + pandas2ri.converter):
      r_df_simulated = ro.conversion.py2rpy(df_simulated)


    # Check holidays from Prophet
    # 59 countries included. If your country is not included, please manually add it.
    # Tip: any events can be added into this table, school break, events et
    r.data('dt_prophet_holidays')
    r['dt_prophet_holidays'].head()

    df_prophet = pd.read_csv(proph_hol_path)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'],yearfirst=True,format="%Y-%m-%d").dt.date #pd.to_datetime(df_simulated['DATE']).dt.strftime("%Y-%m-%d")
    r_date_n = base.as_Date(pd.to_datetime(df_prophet['ds']).dt.strftime("%Y-%m-%d"))
    df_prophet['ds'] = r_date_n#df_prophet.ds.astype(str) #r_date_n#base.format(r_date_n, format="%Y-%m-%d")
    #df_prophet['holiday']=df_prophet.holiday.astype(str)
    del df_prophet[df_prophet.columns[0]]
    #df_prophet = set_date(df_prophet)
    with localconverter(ro.default_converter + pandas2ri.converter):
      r_df_prophet = ro.conversion.py2rpy(df_prophet)

    # Set robyn_object. It must have extension .RDS. The object name can be different than Robyn:
    # TODO
    # robyn_object = "~/Desktop/MyRobyn.RDS"


    ########################################################################################################################
    # Step 2a: For first time user: Model specification in 4 steps

    # 2a-1: First, specify input data & model parameters

    # Run ?robyn_inputs to check parameter definition
    # TODO

    robyn.check_nas(df=df_simulated)

    input_collect= robyn.robyn_inputs(

        dt_input=r_df_simulated
        , dt_holidays=r_df_prophet  ##
        , date_var="DATE"
        , dep_var="revenue"
        , dep_var_type="revenue"
        , prophet_vars=np.array(["trend", "season", "holiday"])
        , prophet_country="DE"
        , context_vars=np.array(["competitor_sales_B", "events"])
        , paid_media_vars=np.array(["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"])
        # "ooh_S","print_S","facebook_I","search_clicks_P" ##
        , paid_media_spends=np.array(["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"])
        # "ooh_S","print_S","facebook_S", "search_S"
        , organic_vars=np.array(["newsletter"])
        , factor_vars=np.array(["events"])  #
        , window_start= "2016-11-21"# date(2016,11,21) #"2016-11-21"
        , window_end= "2018-08-20"#date(2018,8,20)   #"2018-08-20"
        , adstock="geometric"
    )

    print(0)
    #hyper_names = robyn.hyper_names(adstock = input_collect[33], all_media = input_collect[23)
    #TODO run below
    ###### ROBYN CHECK WINDOWS CALL ########
    #robyn.check_windows(dt_input=r_df_simulated,date_var="DATE",all_media=[np.array(["tv_S"]),np.array(["newsletter"])],window_start="2016-11-21", window_end="2018-08-20")
    '''
    hyper_params = pd.DataFrame(data=[[0.5, 3], [0.3, 1],
                           [0, 0.3],[0.5, 3],
                           [0.3, 1],[0.1, 0.4],
                           [0.5, 3],[0.3, 1],
                           [0.3, 0.8],[0.5, 3],
                           [0.3, 1],[0, 0.3],
                           [0.5, 3],[0.3, 1],
                           [0.1, 0.4],[0.5, 3],
                           [0.3, 1],[0.1, 0.4]],
                columns=['facebook_S_alphas', 'facebook_S_gammas',
                  'facebook_S_thetas','print_S_alphas',
                  'print_S_gammas','print_S_thetas',
                  'tv_S_alphas','tv_S_gammas',
                  'tv_S_thetas','search_S_alphas',
                  'search_S_gammas','search_S_thetas',
                  'ooh_S_alphas','ooh_S_gammas',
                  'ooh_S_thetas','newsletter_alphas',
                  'newsletter_gammas','newsletter_thetas']) '''


    param_names=['facebook_S_alphas', 'facebook_S_gammas',
                                                      'facebook_S_thetas', 'print_S_alphas',
                                                      'print_S_gammas', 'print_S_thetas',
                                                      'tv_S_alphas', 'tv_S_gammas',
                                                      'tv_S_thetas', 'search_S_alphas',
                                                      'search_S_gammas', 'search_S_thetas',
                                                      'ooh_S_alphas', 'ooh_S_gammas',
                                                      'ooh_S_thetas', 'newsletter_alphas',
                                                      'newsletter_gammas', 'newsletter_thetas']

    param_vals=[[0.5, 3], [0.3, 1],
                         [0, 0.3], [0.5, 3],
                         [0.3, 1], [0.1, 0.4],
                         [0.5, 3], [0.3, 1],
                         [0.3, 0.8], [0.5, 3],
                         [0.3, 1], [0, 0.3],
                         [0.5, 3], [0.3, 1],
                         [0.1, 0.4], [0.5, 3],
                         [0.3, 1], [0.1, 0.4]]

    param_dict={}
    for e, p in enumerate(param_vals):
        #param_dict[param_names[e]]=np.array(p)
        param_dict[param_names[e]]=robjects.FloatVector(p) # np.array(p)



    # try:
    hp = robjects.ListVector(param_dict)
    hp2 = rlc.TaggedList(param_vals, tags=('facebook_S_alphas', 'facebook_S_gammas',
                                           'facebook_S_thetas', 'print_S_alphas',
                                           'print_S_gammas', 'print_S_thetas',
                                           'tv_S_alphas', 'tv_S_gammas',
                                           'tv_S_thetas', 'search_S_alphas',
                                           'search_S_gammas', 'search_S_thetas',
                                           'ooh_S_alphas', 'ooh_S_gammas',
                                           'ooh_S_thetas', 'newsletter_alphas',
                                           'newsletter_gammas', 'newsletter_thetas'))

    hp_n=robjects.r('list(facebook_S_alphas = c(0.5, 3),facebook_S_gammas = c(0.3, 1) ,'
                  'facebook_S_thetas = c(0, 0.3),print_S_alphas = c(0.5, 3),print_S_gammas = c(0.3, 1),print_S_thetas = c(0.1, 0.4),tv_S_alphas = c(0.5, 3),tv_S_gammas = c(0.3, 1), tv_S_thetas = c(0.3, 0.8),search_S_alphas = c(0.5, 3) ,search_S_gammas = c(0.3, 1),search_S_thetas = c(0, 0.3),ooh_S_alphas = c(0.5, 3),ooh_S_gammas = c(0.3, 1),ooh_S_thetas = c(0.1, 0.4),newsletter_alphas = c(0.5, 3),newsletter_gammas = c(0.3, 1),newsletter_thetas = c(0.1, 0.4))')
    hyper_names = robyn.hyper_names(adstock = input_collect[32], all_media = input_collect[22])
    robyn.check_legacy_input(input_collect)
    calibration_input= robyn.check_calibration(dt_input=input_collect[0],date_var=input_collect[5],calibration_input=input_collect[4],
                                               dayInterval=input_collect[6],  dep_var=input_collect[8], window_start=input_collect[26],
                                               window_end=input_collect[28], paid_media_spends=input_collect[17], organic_vars=input_collect[20])
    input_collect.rx2['hyperparameters'] = hp
    try:
        robyn.robyn_engineering(input_collect)
    except:
        logger.exception("error")
    logger.info("added check_hyperparameters")
    hp_check=robyn.check_hyperparameters(hyperparameters=input_collect.rx2['hyperparameters'], adstock=input_collect[32],
                                paid_media_spends=input_collect.rx2['paid_media_spends'],
                                organic_vars=input_collect.rx2['organic_vars'],
                                exposure_vars=input_collect.rx2['organic_vars'])
    logger.info("DONE check_hyperparameters")

    robyn.robyn_run(InputCollect=input_collect
    , iterations = 2000  # recommended for the dummy dataset
    , trials = 5  # recommended for the dummy dataset
    , outputs = False
    )



    robjects.ListVector(input_collect)[7][0] ="day"
    #robyn.check_windows(dt_input=input_collect.rx2['dt_input'], date_var=input_collect.rx2['dt_input'].rx2['DATE'], window_start=input_collect.rx2['window_start'],
                       # window_end=input_collect[28], all_media=input_collect.rx2['all_media'])
    #input_collect.rx2['hyperparameters']

    try:
        input_collects = robyn.robyn_inputs(InputCollect=input_collect, hyperparameters=hp)
    except:
        logger.exception("ERROR")

    outputs=robyn.robyn_run(
          InputCollect = input_collect # feed in all model specification
          #, cores = NULL # default
          #, add_penalty_factor = FALSE # Untested feature. Use with caution.
          , iterations = 2000 # recommended for the dummy dataset
          , trials = 5 # recommended for the dummy dataset
          , outputs = False # outputs = FALSE disables direct model output
        )


    logger.info('SUCCESS')
    robyn.plot_adstock(plot=True)
    robyn.plot_adstock(plot=True)
    robyn.plot_saturation(plot = True)


except :
    logger.exception("ERROR DUE TO")
