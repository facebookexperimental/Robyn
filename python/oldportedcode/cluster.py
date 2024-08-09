# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

####################################################################
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
import scipy.stats as stats


## Manual imports

def determine_optimal_k(df, max_clusters, random_state=42):
    wss = []
    K_range = range(1, max_clusters + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(df)
        wss.append(kmeans.inertia_)

    # Calculate the second derivative of the WSS
    # The second derivative is a simple way to find the inflection point where the rate of decrease
    # of WSS changes significantly, corresponding to the elbow
    second_derivative = np.diff(wss, n=2)

    # The optimal k is where the second derivative is maximized
    optimal_k = np.argmax(second_derivative) + 2  # +2 because np.diff reduces the original array by 1 for each differentiation and we start counting from 1

    return optimal_k

def clusterKmeans_auto(df, min_clusters=3, limit_clusters=10, seed=None):
    features = df.select_dtypes(include=[np.number])  # Assuming numerical columns for clustering
    features.columns = features.columns.astype(str)
    features.columns = [str(col) for col in features.columns]
    # Determine the range of k values to try
    k_range = range(1, limit_clusters + 1)

    # Calculate WSS for each k
    wss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed)
        kmeans.fit(features)
        wss.append(kmeans.inertia_)

    # Plot WSS to identify the elbow (optional visualization step)
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, wss, 'bo-', markersize=8, lw=2)
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.grid(True)
    plt.show()

    # Optionally, automatically determine the optimal k based on the elbow method or other criteria
    # This part is simplified; more sophisticated methods could be applied for determining 'k'
    optimal_k = determine_optimal_k(features, 20)
    optimal_k = max(optimal_k, min_clusters)
    if optimal_k >= 4:
        optimal_k = 3

    # Perform final clustering with determined optimal k
    limit_clusters = min(len(df) - 1, 30)
    final_kmeans = KMeans(n_clusters=optimal_k, max_iter=limit_clusters, random_state=seed, tol=0.05)
    final_kmeans.fit(features)

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=optimal_k)
    df_pca = pca.fit_transform(features)
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=optimal_k, random_state=seed)
    df_tsne = tsne.fit_transform(features)

    # Adding cluster labels to the original DataFrame
    df['cluster'] = final_kmeans.labels_

    return df, optimal_k, wss, final_kmeans, df_pca, df_tsne

def plot_wss_and_save(wss, path, dpi=500, width=5, height=4):
    """
    Creates and saves a WSS plot.

    Args:
    - wss: Array of WSS values.
    - path: File path for the saved plot.
    - dpi: Dots per inch (resolution) of the saved plot.
    - width: Width of the figure in inches.
    - height: Height of the figure in inches.
    """
    # Create the plot
    plt.figure(figsize=(width, height))
    k_values = range(1, len(wss) + 1)
    plt.plot(k_values, wss, marker='o', linestyle='-', color='blue')
    plt.title('WSS vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WSS')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()

    # Set background to white
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    # Save the plot
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

def robyn_clusters(input, dep_var_type, cluster_by='hyperparameters', all_media=None, k='auto', limit=1, weights=None, dim_red='PCA', quiet=False, export=False, seed=123):
    """
    Clusters the data based on specified parameters and returns a dictionary containing various outputs.

    Parameters:
    - input: The input data, either a robyn_outputs object or a dataframe.
    - dep_var_type: The type of dependent variable ('continuous' or 'categorical').
    - cluster_by: The variable to cluster by, either 'hyperparameters' or 'performance'. Default is 'hyperparameters'.
    - all_media: The list of media variables. Default is None.
    - k: The number of clusters. Default is 'auto'.
    - limit: The maximum number of top solutions to select. Default is 1.
    - weights: The weights for balancing the clusters. Default is None.
    - dim_red: The dimensionality reduction technique to use. Default is 'PCA'.
    - quiet: Whether to suppress print statements. Default is False.
    - export: Whether to export the results. Default is False.
    - seed: The random seed for reproducibility. Default is 123.

    Returns:
    - output: A dictionary containing various outputs such as cluster data, cluster confidence intervals, number of clusters, etc.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Check that cluster_by is a valid option
    if cluster_by not in ['performance', 'hyperparameters']:
        raise ValueError("cluster_by must be either 'performance' or 'hyperparameters'")

    # Load data
    if all_media is None:
        aux = input["mediaVecCollect"].columns
        if "type" in aux:
            type_index = list(aux).index("type")
            all_media = aux[1:type_index]  # Exclude the first column and from "type" onwards, Python uses 0-based indexing
        else:
            all_media = aux[1:-1]  # If "type" is not found, exclude the first and last columns as a fallback

        path = input["plot_folder"]


    # Pareto and ROI data
    x = input["xDecompAgg"]
    if cluster_by == 'hyperparameters':
        x = input["resultHypParam"]
    df = prepare_df(x, all_media, dep_var_type, cluster_by)

    ignore = ["solID", "mape", "mape.qt10", "decomp.rssd", "nrmse", "nrmse_test", "nrmse_train", "nrmse_val", "pareto"]
    # Auto K selected by less than 5% WSS variance (convergence)\
    min_clusters = 3
    limit_clusters = min(len(df) - 1, 30)
    features= df.drop(columns=ignore, errors='ignore')
    df_pca = None
    df_tsne = None
    if k == 'auto':
        try:
            # You must determine the appropriate number of clusters beforehand, as `n_clusters=None` is not valid.
            # This placeholder (e.g., 3) is for demonstration; you need a dynamic method or a fixed value.
            # determined_clusters = determine_optimal_k(features, 20)
            # cls = KMeans(n_clusters=determined_clusters, max_iter=limit_clusters, random_state=seed, tol=0.05).fit(features)
            df, optimal_k, wss, cls, df_pca, df_tsne = clusterKmeans_auto(df, min_clusters, limit_clusters=limit_clusters, seed=seed)
        except Exception as e:
            print(f"Couldn't automatically create clusters: {e}")
            cls = None

        # Ensure `cls` is not `None` before accessing its attributes
        if cls is not None:
            k = cls.n_clusters
        else:
            k = 0  # Or handle this case as needed, perhaps setting it to `min_clusters` or another default

        # Now, proceed with your logic
        if k < min_clusters:
            k = min_clusters
        print(f">> Auto selected k = {k} (clusters) based on minimum WSS variance of {0.05*100}%")


    # Build clusters
    assert k in range(min_clusters, 31), "k is not within the specified range"

    solID = df['solID'].copy()

    # Perform KMeans clustering on the numeric data only
    # try:
    #     cls = KMeans(n_clusters=k, max_iter=limit_clusters, random_state=seed).fit(features)
    # except Exception as e:
    #     print(f"Error during KMeans fitting: {e}")
    #     cls = None

    # If you need to use the cluster labels with the original DataFrame, you can add them back
    if cls is not None:
        # Add the cluster labels to the original DataFrame or to solID as needed
        #df['cluster'] = cls.labels_
        # Or if you want to create a new DataFrame with solID and the cluster labels
        df_with_clusters = pd.DataFrame({'solID': solID, 'cluster': cls.labels_})

    columns_to_ignore = set(ignore + ['cluster'])
    all_columns = set(df.columns)
    all_paid = all_columns - columns_to_ignore
    all_paid = list(all_paid)
    # Select top models by minimum (weighted) distance to zero
    # all_paid = setdiff(names(input.df), [ignore, 'cluster'])
    ts_validation = all(np.isnan(df['nrmse_test']))
    top_sols = clusters_df(df=df, all_paid=all_paid, balance=weights, limit=limit, ts_validation=ts_validation)
    top_sols = top_sols.loc[:, ~top_sols.columns.duplicated()]
    # df, optimal_k, wss = clusterKmeans_auto(df, limit_clusters=limit_clusters, seed=seed)

    # Build in-cluster CI with bootstrap
    ci_list = confidence_calcs(input["xDecompAgg"], df, all_paid, dep_var_type, k, cluster_by, seed=seed)

    output = {
        # 'data': pd.DataFrame({'top_sol': df['solID'].isin(top_sols['solID']), 'cluster': pd.Series(np.arange(k), dtype=int)}),
        'data': df.assign(
            top_sol=df['solID'].isin(top_sols['solID']),
            cluster=pd.Series(np.arange(len(df)), dtype=int)
        ),
        'df_cluster_ci': ci_list['df_ci'],
        'n_clusters': k,
        'boot_n': ci_list['boot_n'],
        'sim_n': ci_list['sim_n'],
        'errors_weights': weights,
        # 'wss': input.nclusters_plot + theme_lares(background='white'),
        'wss': plot_wss_and_save(wss, f'{path}pareto_clusters_wss.png'),
        # corrs is not being used anywhere and there is no 1:1 mapping from R to Python
        'corrs': None,
        'clusters_means': cls.cluster_centers_,
        'clusters_PCA': df_pca,
        'clusters_tSNE': df_tsne,
        'models': top_sols,
        'plot_clusters_ci': plot_clusters_ci(ci_list['sim_collect'], ci_list['df_ci'], dep_var_type, ci_list['boot_n'], ci_list['sim_n']),
        # TODO ADD following vars to the output
        #'plot_models_errors': plot_topsols_errors(df, top_sols, limit, weights),
        #'plot_models_rois': plot_topsols_rois(df, top_sols, all_media, limit)
    }

    # TODO Add below code once plotting is fixed
    # if export:
    #     output['data'].to_csv(f'{path}pareto_clusters.csv', index=False)
    #     output['df_cluster_ci'].to_csv(f'{path}pareto_clusters_ci.csv', index=False)
    #     plt.figure(figsize=(5, 4))
    #     sns.heatmap(output['wss'], annot=True, cmap='coolwarm', xticks=range(k), yticks=range(k), square=True)
    #     plt.savefig(f'{path}pareto_clusters_wss.png', dpi=500, bbox_inches='tight')
    #     get_height = int(np.ceil(k/2)/2)
    #     db = (output['plot_clusters_ci'] / (output['plot_models_rois'] + output['plot_models_errors'])) ## TODO: + plot_layout(heights=[get_height, 1], guides='collect')
    #     suppress_messages(plt.savefig(f'{path}pareto_clusters_detail.png', dpi=500, bbox_inches='tight', width=12, height=4+len(all_paid)*2, limitsize=False))

    return output

def confidence_calcs(xDecompAgg, df, all_paid, dep_var_type, k, cluster_by, boot_n=1000, sim_n=10000, **kwargs):
    """
    This function takes in a bunch of inputs and does some statistical calculations.

    Parameters:
    - xDecompAgg: DataFrame, the input data for statistical calculations
    - cls: object, the cluster object containing cluster information
    - all_paid: list, the list of paid values
    - dep_var_type: str, the type of dependent variable ('conversion' or 'roi_total')
    - k: int, the number of clusters
    - cluster_by: str, the method of clustering ('hyperparameters' or other)
    - boot_n: int, the number of bootstrap iterations (default: 1000)
    - sim_n: int, the number of simulations (default: 10000)
    - **kwargs: additional keyword arguments

    Returns:
    - df_ci: DataFrame, the confidence interval results
    - sim_collect: DataFrame, the simulation results
    - boot_n: int, the number of bootstrap iterations
    - sim_n: int, the number of simulations
    """

    """
    This function takes in a bunch of inputs and does some statistical calculations
    """
    # filter out rows with missing values for total_spend
    filtered_df = xDecompAgg[~xDecompAgg['total_spend'].isna()]
    # join with cluster information
    merged_df = filtered_df.merge(df[['solID', 'cluster']], on='solID', how='left')
    # select relevant columns and group by cluster, solID and rn
    grouped_df_with_n = merged_df[['solID', 'cluster', 'rn', 'roi_total', 'cpa_total', 'robynPareto']].groupby(['cluster', 'rn', 'solID']).size().reset_index(name='n')
    grouped_df_with_solID = merged_df[['solID', 'cluster', 'rn', 'roi_total', 'cpa_total', 'robynPareto']].groupby(['cluster', 'rn', 'solID']).agg({'roi_total': 'mean', 'cpa_total': 'mean', 'robynPareto': 'mean'})
    grouped_df = pd.merge(grouped_df_with_n, grouped_df_with_solID, on=['cluster', 'rn', 'solID'])
    # sort by cluster and rn
    df_clusters_outcome = grouped_df.sort_values(['cluster', 'rn', 'solID'])

    # Initialize lists to store results
    cluster_collect = []
    chn_collect = []
    sim_collect = []

    for j in range(k):
        # Filter outcome data for current cluster
        df_outcome = df_clusters_outcome[df_clusters_outcome['cluster'] == j]

        if len(df_outcome['solID'].unique()) < 3:
            print(f"Cluster {j} does not contain enough models to calculate CI")
        else:

            from .checks import HYPS_NAMES

            # Bootstrap CI
            if cluster_by == 'hyperparameters':
                pattern = '|'.join(["_" + re.escape(hyp_name) for hyp_name in HYPS_NAMES])
                all_paid = np.unique([re.sub(pattern, '', paid) for paid in all_paid])
            for i in all_paid:
                if dep_var_type == 'conversion':
                    # Correctly apply filtering for 'conversion' case
                    df_chn = df_outcome[(df_outcome['rn'] == i) & np.isfinite(df_outcome['cpa_total'])]
                    v_samp = df_chn['cpa_total']
                else:
                    df_chn = df_outcome[df_outcome['rn'] == i]
                    v_samp = df_chn['roi_total']

                boot_mean = np.mean(v_samp)
                boot_se = np.std(v_samp, ddof=1) / np.sqrt(len(v_samp))

                ci_low, ci_up = stats.norm.interval(0.95, loc=boot_mean, scale=boot_se)
                ci_low = max(0, ci_low)

                df_chn_modified = df_chn.assign(ci_low=ci_low, ci_up=ci_up, n=len(v_samp),
                                boot_se=boot_se, boot_mean=boot_mean, cluster=j)
                chn_collect.append(df_chn_modified)

                # Correcting the simulation part
                x_sim = np.random.normal(boot_mean, boot_se, size=sim_n)
                y_sim = norm.pdf(x_sim, boot_mean, boot_se)  # Correct way to simulate 'y_sim' as in R's dnorm

                # Creating and appending the new DataFrame to sim_collect
                sim_df = pd.DataFrame({
                    'cluster': j,
                    'rn': i,
                    'n': len(v_samp),
                    'boot_mean': boot_mean,
                    'x_sim': x_sim,
                    'y_sim': y_sim
                })
                sim_collect.append(sim_df)

            cluster_collect.append({
                f'chn_{j}': chn_collect,
                f'sim_{j}': sim_collect
            })

    all_sim_collect_dfs = []
    all_chn_collect_dfs = []
    for cluster in cluster_collect:
        for key in cluster:
            if key.startswith('sim_'):
                all_sim_collect_dfs.extend(cluster[key])
            if key.startswith('chn_'):
                all_chn_collect_dfs.extend(cluster[key])
    sim_collect = pd.concat(all_sim_collect_dfs, ignore_index=True)
    chn_collect = pd.concat(all_chn_collect_dfs, ignore_index=True)

    sim_collect['cluster_title'] = sim_collect.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)

    df_ci = chn_collect.drop_duplicates()

    df_ci['cluster_title'] = df_ci.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)

    # If df_ci needs grouping and summarization similar to what was described previously:
    df_ci_grouped = df_ci.groupby(['rn', 'cluster', 'cluster_title']).agg(
        n=('n', 'first'),
        boot_mean=('boot_mean', 'mean'),
        boot_se=('boot_se', 'mean'),
        ci_low=('ci_low', 'min'),
        ci_up=('ci_up', 'max')
    ).reset_index()

    df_ci_grouped['boot_ci'] = df_ci_grouped.apply(lambda x: f"[{round(x['ci_low'], 2)}, {round(x['ci_up'], 2)}]", axis=1)
    df_ci_grouped['sd'] = df_ci_grouped['boot_se'] * np.sqrt(df_ci_grouped['n'] - 1)
    df_ci_grouped['dist100'] = (df_ci_grouped['ci_up'] - df_ci_grouped['ci_low'] + 2 * df_ci_grouped['boot_se'] * np.sqrt(df_ci_grouped['n'] - 1)) / 99

    # df_ci_grouped now holds the processed data
    df_ci = df_ci_grouped

    return {
        'df_ci': df_ci,
        'sim_collect': sim_collect,
        'boot_n': boot_n,
        'sim_n': sim_n
    }

def errors_scores(df, balance=None, ts_validation=True, **kwargs):
    """
    Calculate the error scores for a given dataframe.

    Parameters:
    - df: DataFrame - The input dataframe containing the error columns.
    - balance: list or None - The balance values for weighting the error scores. If None, no balancing is applied.
    - ts_validation: bool - Flag indicating whether to use the 'nrmse_test' column for error calculation. If False, use 'nrmse_train' column.
    - **kwargs: Additional keyword arguments.

    Returns:
    - scores: DataFrame - The calculated error scores.

    Raises:
    - AssertionError: If the length of balance is not 3.
    - AssertionError: If any of the error columns are not found in the dataframe.
    """
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
    scores = df[error_cols].rename(columns={error_cols[0]: 'nrmse'})

    # Replace infinite values with maximum finite value
    scores = scores.apply(lambda row: np.nan_to_num(row, posinf=max(row.dropna().values)))

    # Force normalized values
    scores = scores.apply(lambda row: row / row.max())

    # Replace missing values with 0
    scores = scores.fillna(0)

    # Balance errors
    if balance is not None:
        scores = scores.apply(lambda row: balance * row, axis=1)

    # Calculate error score
    scores = scores.apply(lambda row: np.sqrt(row['nrmse']**2 + row['decomp.rssd']**2 + row['mape']**2), axis=1)

    # scores = scores.apply(lambda row: np.sqrt(row['nrmse']**2 + row['decomp.rssd']**2 + row['mape']**2))

    return scores


def prepare_df(x, all_media, dep_var_type, cluster_by):
    # Initial checks and setup
    if cluster_by == "performance":
        # Check options (assuming check_opts is appropriately defined and used)
        check_opts(all_media, x['rn'].unique())

        # Preparing outcome DataFrame based on dep_var_type
        if dep_var_type == "revenue":
            # Create dummy variables for 'rn', merge with 'roi_total', ensuring 'solID' is included
            dummies = pd.get_dummies(x['rn'])
            outcome = pd.concat([x[['solID', 'roi_total']], dummies], axis=1)
        elif dep_var_type == "conversion":
            # Filter by 'cpa_total' being finite, then proceed similar to 'revenue' case
            filtered = x[pd.to_numeric(x['cpa_total'], errors='coerce').notnull()]
            dummies = pd.get_dummies(filtered['rn'])
            outcome = pd.concat([filtered[['solID', 'cpa_total']], dummies], axis=1)

        # Ensure all_media columns are included by dynamically checking their presence
        outcome = outcome[[col for col in ['solID'] + all_media + list(dummies.columns) if col in outcome.columns]]

        # Merge with error metrics
        errors = x[['solID', 'nrmse', 'nrmse_test', 'nrmse_train', 'decomp.rssd', 'mape']].drop_duplicates()
        outcome = pd.merge(outcome, errors, on='solID', how='left')

    elif cluster_by == "hyperparameters":
        # Include only 'solID', hyperparameters, and specific metrics
        from .checks import HYPS_NAMES

        cols_to_keep = ['solID'] + [col for col in x.columns if any(hyps in col for hyps in HYPS_NAMES + ['nrmse', 'decomp.rssd', 'mape'])]
        outcome = x[cols_to_keep].copy()
    else:
        raise ValueError("Invalid cluster_by parameter")

    return outcome

# def prepare_df(x, all_media, dep_var_type, cluster_by):
#     """
#     Prepare the dataframe for clustering analysis based on the given parameters.

#     Parameters:
#     x (DataFrame): The input dataframe.
#     all_media (list): List of all media options.
#     dep_var_type (str): Type of dependent variable ("revenue" or "conversion").
#     cluster_by (str): Type of clustering ("performance" or "hyperparameters").

#     Returns:
#     DataFrame: The prepared dataframe for clustering analysis.
#     """
#     if cluster_by == "performance":
#         # Check options
#         check_opts(all_media, unique(x['rn']))

#         # Select columns and spread ROI total
#         if dep_var_type == "revenue":
#             outcome = x[['solID', 'rn', 'roi_total']].copy()
#             outcome = pd.get_dummies(outcome, columns=['rn'])
#             outcome = outcome.drop(columns=['rn'])
#             outcome = outcome.select(columns=['solID'] + all_media)
#         elif dep_var_type == "conversion":
#             outcome = x[['solID', 'rn', 'cpa_total']].copy()
#             outcome = outcome[outcome['cpa_total'].isfinite()]
#             outcome = pd.get_dummies(outcome, columns=['rn'])
#             outcome = outcome.drop(columns=['rn'])
#             outcome = outcome.select(columns=['solID'] + all_media)

#         # Remove missing values
#         errors = x.dropna()

#         # Join errors with outcome
#         outcome = pd.merge(outcome, errors, on='solID')
#         outcome = outcome.drop(columns=['nrmse', 'nrmse_test', 'nrmse_train', 'decomp.rssd', 'mape'])
#     else:
#         if cluster_by == "hyperparameters":

#             from .checks import HYPS_NAMES

#             cols_to_keep = ['solID'] + [col for col in x.columns if any(hyp in col for hyp in HYPS_NAMES)]
#             outcome = x[cols_to_keep].copy()

#         else:
#             raise ValueError("Invalid cluster_by parameter")

#     return outcome

def min_max_norm(x, min=0, max=1):
    """
    Performs min-max normalization on the input array.

    Parameters:
    - x: Input array to be normalized.
    - min: Minimum value of the normalized range (default: 0).
    - max: Maximum value of the normalized range (default: 1).

    Returns:
    - Normalized array.

    """
    x = x[np.isfinite(x)]
    if len(x) == 1:
        return x
    a = np.min(x, axis=0)
    b = np.max(x, axis=0)
    return (max - min) * (x - a) / (b - a) + min

##def clusters_df(df, all_paid, balance=rep(1, 3), limit=1, ts_validation=True, **kwargs):
def clusters_df(df, all_paid, balance=None, limit=1, ts_validation=True, **kwargs):
    """
    Calculate the error scores for each cluster in the given dataframe and return the top clusters based on the error scores.

    Parameters:
    - df: pandas DataFrame
        The input dataframe containing the data.
    - all_paid: bool
        A boolean value indicating whether all payments have been made.
    - balance: numpy array, optional
        An array containing the balance values for each cluster. If not provided, it will be set to [1, 1, 1].
    - limit: int, optional
        The maximum number of clusters to return. Defaults to 1.
    - ts_validation: bool, optional
        A boolean value indicating whether to perform time series validation. Defaults to True.
    - **kwargs: keyword arguments
        Additional arguments to be passed to the errors_scores function.

    Returns:
    - pandas DataFrame
        A dataframe containing the cluster, rank, and error score for the top clusters.
    """
    if balance is None:
        balance = np.repeat(1, 3)

    df['error_score'] = errors_scores(df, balance, ts_validation=ts_validation, **kwargs)
    df.fillna(0, inplace=True)
    df = df.groupby('cluster').apply(lambda x: x.sort_values('error_score').head(limit)).reset_index(drop=True)
    df['rank'] = df.groupby('cluster')['error_score'].rank(ascending=False)
    return df[['cluster', 'rank'] + list(df.columns[:-2])]
    # df = df.replace(np.nan, 0)
    # df['error_score'] = errors_scores(df, balance, ts_validation=ts_validation, **kwargs)
    # df = df.groupby('cluster').agg({'error_score': 'mean'})
    # df = df.sort_values('error_score', ascending=False)
    # df = df.head(limit)
    # df['rank'] = df.groupby('cluster').cumcount() + 1
    # return df[['cluster', 'rank', 'error_score']]

def plot_clusters_ci(sim_collect, df_ci, dep_var_type, boot_n, sim_n):
    """
    Plots the clusters with confidence intervals.

    Parameters:
    sim_collect (DataFrame): The simulated data.
    df_ci (DataFrame): The data frame containing the confidence intervals.
    dep_var_type (str): The type of dependent variable ("conversion" or "ROAS").
    boot_n (int): The number of bootstrap results.
    sim_n (int): The number of simulations.

    Returns:
    None
    """
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

# TODO fix plotting
def plot_topsols_errors(df, top_sols, limit=1, balance=None):
    """
    Plots a heatmap of the correlation matrix for the joined dataframe of `df` and `top_sols`.

    Parameters:
        df (pandas.DataFrame): The main dataframe.
        top_sols (pandas.DataFrame): The dataframe containing top solutions.
        limit (int, optional): The number of top performing models to select. Defaults to 1.
        balance (numpy.ndarray, optional): The weights for balancing the heatmap. Defaults to None.

    Returns:
        None
    """
    # Calculate balance
    if balance is None:
        balance = np.array([1, 1, 1])
    else:
        balance = balance / np.sum(balance)

    temp_df = df.copy()
    temp_df.drop(['cluster', 'error_score'], axis=1, inplace=True, errors='ignore')
    # Join dataframes
    #joined_df = pd.merge(temp_df, top_sols, on='solID', how='left')
    joined_df = pd.merge(temp_df, top_sols, on='solID', how='left', suffixes=('', '_top'))
    joined_df = joined_df[[col for col in joined_df.columns if not col.endswith('_top')]]

    # Calculate alpha and label
    joined_df['alpha'] = np.where(np.isnan(joined_df['cluster']), 0.6, 1)
    #joined_df['label'] = np.where(np.isnan(joined_df['cluster']), np.nan, f"[{joined_df['cluster']}.{joined_df['rank']}]")

    #correlation_df = joined_df.copy()
    #correlation_df = correlation_df.drop(['cluster', 'rank'], axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style='white')
    sns.heatmap(joined_df.corr(), annot=True, cmap='coolwarm', square=True)

    # Customize x-axis and y-axis ticks
    plt.xticks(ticks=range(len(joined_df.columns)), labels=joined_df.columns)
    plt.yticks(ticks=range(len(joined_df.columns)), labels=joined_df.columns)
    #sns.heatmap(joined_df.corr(), annot=True, cmap='coolwarm',
                  #square=True, xticks=range(3), yticks=range(3),
                  #xlabel='Feature 1', ylabel='Feature 2')
    plt.title(f"Selecting Top {limit} Performing Models by Cluster")
    #plt.subtitle("Based on minimum (weighted) distance to origin")
    plt.xlabel("NRMSE")
    plt.ylabel("DECOMP.RSSD")
    #plt.caption(f"Weights: NRMSE {round(100*balance[0])}%, DECOMP.RSSD {round(100*balance[1])}%, MAPE {round(100*balance[2])}%")
    plt.show()

def plot_topsols_rois(df, top_sols, all_media, limit=1):
    """
    Plot the top performing models' mean metric per media for the given data.

    Parameters:
    df (DataFrame): The dataframe containing the real ROIs.
    top_sols (DataFrame): The dataframe containing the top solutions.
    all_media (list): The list of all media.
    limit (int, optional): The number of top solutions to consider. Defaults to 1.
    """
    # Create a dataframe with the real ROIs
    real_rois = df.drop(columns=['mape', 'nrmse', 'decomp.rssd'])
    real_rois.columns = ['real_' + col for col in real_rois.columns]

    # Join the real ROIs with the top solutions
    top_sols = pd.merge(top_sols, real_rois, left_on='solID', right_on='real_solID', how='left')


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
    """
    Compute the bootstrap confidence interval for a given sample.

    Parameters:
    samp (array-like): The sample data.
    boot_n (int): The number of bootstrap samples to generate.
    seed (int): The seed for random number generation. Default is 1.
    **kwargs: Additional keyword arguments to be passed to np.mean() and np.std().

    Returns:
    list: A list containing the bootstrap means, confidence interval, and standard error of the mean.
    """
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
