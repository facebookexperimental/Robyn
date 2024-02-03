import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
##from sklearn.stats import bootstrap
##from sklearn.bootstrapping import Bootstrap
## from sklearn.plot_utils import plot_layout
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from matplotlib.ticker import FormatStrFormatter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
## from sklearn.preprocessing import Dropna
from scipy.stats import norm

## Manual imports


def robyn_clusters(input, dep_var_type, cluster_by='hyperparameters', all_media=None, k='auto', limit=1, weights=None, dim_red='PCA', quiet=False, export=False, seed=123):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Check that cluster_by is a valid option
    if cluster_by not in ['performance', 'hyperparameters']:
        raise ValueError("cluster_by must be either 'performance' or 'hyperparameters'")

    # Load data
    if 'robyn_outputs' in dir(input):
        # Load data from robyn_outputs
        if all_media is None:
            aux = colnames(input.mediaVecCollect)
            all_media = aux[:-1]
        path = input.plot_folder
    else:
        # Load data from a dataframe
        path = '.'

    # Pareto and ROI data
    x = input.xDecompAgg
    if cluster_by == 'hyperparameters':
        x = input.resultHypParam
    df = prepare_df(x, all_media, dep_var_type, cluster_by)

    # Auto K selected by less than 5% WSS variance (convergence)
    min_clusters = 3
    limit_clusters = min(len(df) - 1, 30)
    if k == 'auto':
        cls = try_catch(
            lambda: KMeans(n_clusters=None, max_iter=limit_clusters, random_state=seed, tol=0.05).fit(df),
            error=lambda e: print(f"Couldn't automatically create clusters: {e}")
        )
        k = cls.n_clusters_
        if k < min_clusters:
            k = min_clusters
        print(f">> Auto selected k = {k} (clusters) based on minimum WSS variance of {0.05*100}%")

    # Build clusters
    stop_if_not(k in range(min_clusters, 30))
    cls = KMeans(n_clusters=k, max_iter=limit_clusters, random_state=seed).fit(df)

    # Select top models by minimum (weighted) distance to zero
    all_paid = setdiff(names(input.df), [ignore, 'cluster'])
    ts_validation = np.isfinite(input.df['nrmse_test'])
    top_sols = clusters_df(df=input.df, all_paid=all_paid, balance=weights, limit=limit, ts_validation=ts_validation)

    # Build in-cluster CI with bootstrap
    ci_list = confidence_calcs(xDecompAgg, input, all_paid, dep_var_type, k, cluster_by, seed=seed)

    output = {
        'data': pd.DataFrame({'top_sol': input.df['solID'].isin(top_sols['solID']), 'cluster': pd.Series(np.arange(k), dtype=int)}),
        'df_cluster_ci': ci_list['df_ci'],
        'n_clusters': k,
        'boot_n': ci_list['boot_n'],
        'sim_n': ci_list['sim_n'],
        'errors_weights': weights,
        'wss': input.nclusters_plot + theme_lares(background='white'),
        'corrs': input.correlations + labs(title='Top Correlations by Cluster', subtitle=None),
        'clusters_means': input.means,
        'clusters_PCA': input.PCA,
        'clusters_tSNE': input.tSNE,
        'models': top_sols,
        'plot_clusters_ci': plot_clusters_ci(ci_list['sim_collect'], ci_list['df_ci'], dep_var_type, ci_list['boot_n'], ci_list['sim_n']),
        'plot_models_errors': plot_topsols_errors(input.df, top_sols, limit, weights),
        'plot_models_rois': plot_topsols_rois(input.df, top_sols, all_media, limit)
    }

    if export:
        output['data'].to_csv(f'{path}pareto_clusters.csv', index=False)
        output['df_cluster_ci'].to_csv(f'{path}pareto_clusters_ci.csv', index=False)
        plt.figure(figsize=(5, 4))
        sns.heatmap(output['wss'], annot=True, cmap='coolwarm', xticks=range(k), yticks=range(k), square=True)
        plt.savefig(f'{path}pareto_clusters_wss.png', dpi=500, bbox_inches='tight')
        get_height = int(np.ceil(k/2)/2)
        db = (output['plot_clusters_ci'] / (output['plot_models_rois'] + output['plot_models_errors'])) ## TODO: + plot_layout(heights=[get_height, 1], guides='collect')
        suppress_messages(plt.savefig(f'{path}pareto_clusters_detail.png', dpi=500, bbox_inches='tight', width=12, height=4+len(all_paid)*2, limitsize=False))

    return output




def confidence_calcs(xDecompAgg, cls, all_paid, dep_var_type, k, cluster_by, boot_n=1000, sim_n=10000, **kwargs):
    """
    This function takes in a bunch of inputs and does some statistical calculations
    """
    # Filter out missing values and left join with cluster info
    df_clusters_outcome = xDecompAgg.dropna(columns=['total_spend'])
    df_clusters_outcome = pd.merge(df_clusters_outcome, cls.df[['solID', 'cluster']], on='solID')
    df_clusters_outcome = df_clusters_outcome[['solID', 'cluster', 'rn', 'roi_total', 'cpa_total', 'robynPareto']]
    df_clusters_outcome = df_clusters_outcome.groupby(['cluster', 'rn']).size().reset_index(drop=True)
    df_clusters_outcome.columns = ['cluster', 'rn', 'n']

    # Initialize lists to store results
    cluster_collect = []
    chn_collect = []
    sim_collect = []

    for j in range(k):
        # Filter outcome data for current cluster
        df_outcome = df_clusters_outcome[df_clusters_outcome['cluster'] == j]

        if len(unique(df_outcome['solID'])) < 3:
            print(f"Cluster {j} does not contain enough models to calculate CI")
        else:
            # Bootstrap CI
            if cluster_by == 'hyperparameters':
                all_paid = unique(gsub(paste(paste0("_", HYPS_NAMES), collapse='|'), '', all_paid))
            for i in all_paid:
                # Drop CPA == Inf
                ##df_chn = df_outcome[df_outcome['rn'] == i & is.finite(df_outcome['cpa_total'])]
                df_chn = df_outcome[df_outcome['rn'] == i and np.isfinite(df_outcome['cpa_total'])]
                v_samp = df_chn['cpa_total']
                if dep_var_type == 'conversion':
                    df_chn = df_outcome[df_outcome['rn'] == i]
                    v_samp = df_chn['roi_total']

                # Calculate bootstrapped CI
                boot_res = sm.bootci(samp=v_samp, boot_n=boot_n)
                boot_mean = mean(boot_res.boot_means)
                boot_se = boot_res.se
                ci_low = max(0, boot_res.ci[1])
                ci_up = boot_res.ci[2]

                # Collect loop results
                chn_collect.append(df_chn.drop(columns=['cpa_total']))
                chn_collect[-1].rename(columns={'rn': i}, inplace=True)
                chn_collect[-1]['ci_low'] = ci_low
                chn_collect[-1]['ci_up'] = ci_up
                chn_collect[-1]['n'] = len(v_samp)
                chn_collect[-1]['boot_se'] = boot_se
                chn_collect[-1]['boot_mean'] = boot_mean

                # Simulate
                sim_collect.append(pd.DataFrame({'cluster': j, 'rn': i, 'n': len(v_samp), 'boot_mean': boot_mean,
                                                   'x_sim': np.random.normal(boot_mean, boot_se, size=sim_n),
                                                   'y_sim': np.random.normal(boot_mean, boot_se, size=sim_n)}))

            # Collect results for current cluster
            cluster_collect.append(chn_collect)
            cluster_collect[-1].rename(columns={'chn_collect': f'chn_{j}'}, inplace=True)

    # Combine results
    sim_collect = pd.concat(sim_collect)
    sim_collect.columns = ['cluster', 'rn', 'n', 'boot_mean', 'x_sim', 'y_sim']
    sim_collect['cluster_title'] = pd.Series(f'Cl.{j} (n={i})', index=sim_collect.index)
    sim_collect = sim_collect.drop(columns=['x_sim', 'y_sim'])

    df_ci = pd.concat(cluster_collect)
    df_ci.columns = ['rn', 'cluster_title', 'n', 'boot_mean', 'boot_se', 'ci_low', 'ci_up']
    df_ci['dist100'] = (df_ci['ci_up'] - df_ci['ci_low'] + 2 * df_ci['boot_se'] * np.sqrt(df_ci['n'] - 1)) / 99
    df_ci.drop(columns=['ci_low', 'ci_up'], inplace=True)

    return [df_ci, sim_collect, boot_n, sim_n]

import pandas as pd
import numpy as np

def errors_scores(df, balance=None, ts_validation=True, **kwargs):
    # Check length of balance
    if balance is not None:
        assert len(balance) == 3, "Balance must have length 3"

    # Check that error columns are in df
    error_cols = ['nrmse_test' if ts_validation else 'nrmse_train', 'decomp.rssd', 'mape']
    assert all(col in df.columns for col in error_cols), f"Error columns {error_cols} not found in df"

    # Normalize balance values
    if balance is not None:
        balance = balance / sum(balance)

    # Select and rename error columns
    scores = df.select(error_cols).rename(columns={'nrmse': 1})

    # Replace infinite values with maximum finite value
    scores = scores.apply(lambda row: np.nan_to_num(row, posinf=max(row.dropna().values)))

    # Force normalized values
    scores = scores.apply(lambda row: row / row.max())

    # Replace missing values with 0
    scores = scores.fillna(0)

    # Balance errors
    if balance is not None:
        scores = scores.apply(lambda row: balance * row)

    # Calculate error score
    scores = scores.apply(lambda row: np.sqrt(row['nrmse']**2 + row['decomp.rssd']**2 + row['mape']**2))

    return scores



def prepare_df(x, all_media, dep_var_type, cluster_by):
    if cluster_by == "performance":
        # Check options
        check_opts(all_media, unique(x['rn']))

        # Select columns and spread ROI total
        if dep_var_type == "revenue":
            outcome = x[['solID', 'rn', 'roi_total']].copy()
            outcome = pd.get_dummies(outcome, columns=['rn'])
            outcome = outcome.drop(columns=['rn'])
            outcome = outcome.select(columns=['solID'] + all_media)
        elif dep_var_type == "conversion":
            outcome = x[['solID', 'rn', 'cpa_total']].copy()
            outcome = outcome[outcome['cpa_total'].isfinite()]
            outcome = pd.get_dummies(outcome, columns=['rn'])
            outcome = outcome.drop(columns=['rn'])
            outcome = outcome.select(columns=['solID'] + all_media)

        # Remove missing values
        errors = x.dropna()

        # Join errors with outcome
        outcome = pd.merge(outcome, errors, on='solID')
        outcome = outcome.drop(columns=['nrmse', 'nrmse_test', 'nrmse_train', 'decomp.rssd', 'mape'])
    else:
        if cluster_by == "hyperparameters":
            outcome = x[['solID', 'rn'] + HYPS_NAMES].copy()
            outcome = outcome.drop(columns=['rn'])
        else:
            raise ValueError("Invalid cluster_by parameter")

    return outcome


def min_max_norm(x, min=0, max=1):
    x = x[np.isfinite(x)]
    if len(x) == 1:
        return x
    a = np.min(x, axis=0)
    b = np.max(x, axis=0)
    return (max - min) * (x - a) / (b - a) + min


##def clusters_df(df, all_paid, balance=rep(1, 3), limit=1, ts_validation=True, **kwargs):
def clusters_df(df, all_paid, balance=None, limit=1, ts_validation=True, **kwargs):
    if balance is None:
        balance = np.repeat(1, 3)

    df = df.replace(np.nan, 0)
    df['error_score'] = errors_scores(df, balance, ts_validation=ts_validation, **kwargs)
    df = df.groupby('cluster').agg({'error_score': 'mean'})
    df = df.sort_values('error_score', ascending=False)
    df = df.head(limit)
    df['rank'] = df.groupby('cluster').cumcount() + 1
    return df[['cluster', 'rank', 'error_score']]



def plot_clusters_ci(sim_collect, df_ci, dep_var_type, boot_n, sim_n):
    # Convert dep_var_type to CPA or ROAS
    temp = "CPA" if dep_var_type == "conversion" else "ROAS"

    # Filter complete cases in df_ci
    df_ci = df_ci.dropna()

    # Create a ggplot object TODO: use plotnine to make it work
    ## p = plt.ggplot(sim_collect, aes(x='x_sim', y='rn')) \
    ##    + plt.facet_wrap(~df_ci['cluster_title']) \
    ##    + plt.geom_density_ridges_gradient(scale=3, rel_min_height=0.01, size=0.1) \
    ##    + plt.geom_text(data=df_ci, aes(x='boot_mean', y='rn', label='boot_ci'),
    ##                    position=plt.PositionNudge(x=-0.02, y=0.1),
    ##                    colour='grey30', size=3.5) \
    ##    + plt.geom_vline(xintercept=1, linetype='dashed', size=0.5, colour='grey75')

    # Add a title, subtitle, and labels
    ## TODO: use plotnine to make it work
    ##p += plt.labs(title=f"In-Cluster {temp} & bootstrapped 95% CI",
    ##               subtitle="Sampling distribution of cluster mean",
    ##               x=temp, y="Density", fill=temp,
    ##               caption=f"Based on {boot_n} bootstrap results with {sim_n} simulations")

    # Add a horizontal line for ROAS
    ##if temp == "ROAS":
    ##    p += plt.geom_hline(yintercept=1, alpha=0.5, colour='grey50', linetype='dashed')

    # Set theme
    ##p += plt.theme_lares(background='white', legend='none')

    ##return p


def plot_topsols_errors(df, top_sols, limit=1, balance=None):
    # Calculate balance
    if balance is None:
        balance = np.array([1, 1, 1])
    else:
        balance = balance / np.sum(balance)

    # Join dataframes
    joined_df = pd.merge(df, top_sols, on='solID')

    # Calculate alpha and label
    joined_df['alpha'] = np.where(np.isna(joined_df['cluster']), 0.6, 1)
    joined_df['label'] = np.where(np.isna(joined_df['cluster']), np.nan, f"[{joined_df['cluster']}.{joined_df['rank']}]")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style='white')
    sns.heatmap(joined_df.corr(), annot=True, cmap='coolwarm',
                  square=True, xticks=range(3), yticks=range(3),
                  xlabel='Feature 1', ylabel='Feature 2')
    plt.title(f"Selecting Top {limit} Performing Models by Cluster")
    plt.subtitle("Based on minimum (weighted) distance to origin")
    plt.xlabel("NRMSE")
    plt.ylabel("DECOMP.RSSD")
    plt.caption(f"Weights: NRMSE {round(100*balance[0])}%, DECOMP.RSSD {round(100*balance[1])}%, MAPE {round(100*balance[2])}%")
    plt.show()


def plot_topsols_rois(df, top_sols, all_media, limit=1):
    # Create a dataframe with the real ROIs
    real_rois = df.drop(columns=['mape', 'nrmse', 'decomp.rssd'])
    real_rois.columns = ['real_' + col for col in real_rois.columns]

    # Join the real ROIs with the top solutions
    top_sols = pd.merge(top_sols, real_rois, on='solID', how='left')

    # Create a label column
    top_sols['label'] = np.vectorize(lambda x: f"[{x.cluster}.{x.rank}] {x.solID}")(top_sols)

    # Gather the media and ROI data
    top_sols = pd.melt(top_sols, id_vars=['solID', 'label'], value_vars=['media', 'roi'])

    # Filter out non-real media
    top_sols = top_sols[top_sols['media'].str.startswith('real_')]

    # Remove the 'real_' prefix from the media column
    top_sols['media'] = top_sols['media'].str.replace('real_', '')

    # Plot the data TODO: ggplot?
    ##plt.figure(figsize=(10, 6))
    ##sns.barplot(x=reorder(top_sols['media'], top_sols['roi']), y=top_sols['roi'], data=top_sols)
    ##plt.facet_grid(top_sols['label'] ~ ., scale='free', space='free')
    ##plt.geom_col(color='blue', size=10)
    ##plt.coord_flip()
    ##plt.labs(title='Top Performing Models', x=None, y='Mean metric per media')
    ##plt.theme_lares(background='white')
    ##plt.show()




def bootci(samp, boot_n, seed=1, **kwargs):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Handle case where samp has only one element
    if len(samp) == 1:
        return [samp, [np.nan, np.nan], np.nan]

    # Compute sample mean and standard deviation
    samp_mean = np.mean(samp, **kwargs)
    samp_std = np.std(samp, **kwargs)

    # Generate bootstrap samples
    boot_samples = np.random.choice(samp, size=(boot_n, len(samp)), replace=True)

    # Compute means of bootstrap samples
    boot_means = np.mean(boot_samples, axis=0)

    # Compute standard error of the mean
    se = np.std(boot_means, axis=0)

    # Compute 95% confidence interval
    me = norm.ppf(0.975, len(samp) - 1) * samp_std
    ci = [samp_mean - me, samp_mean + me]

    # Plot bootstrap distribution (optional)
    # plt.hist(boot_means, bins=30, alpha=0.5, label='Bootstrap')
    # plt.density(boot_means, color='red', label='Density')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Bootstrap Distribution')
    # plt.show()

    return [boot_means, ci, se]
