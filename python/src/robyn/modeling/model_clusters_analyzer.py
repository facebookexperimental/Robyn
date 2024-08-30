# cluster.R https://github.com/facebookexperimental/Robyn/blob/python_rewrite/python/src/oldportedcode/cluster.py

# pyre-strict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from robyn.modeling.entities.modeloutput import ModelOutput
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ModelClustersAnalyzer:
    def __init__(self, seed=123):
        self.seed = seed
        np.random.seed(seed)

    def model_clusters_analyze(
        self,
        input_data,
        dep_var_type,
        cluster_by="hyperparameters",
        all_media=None,
        k="auto",
        max_clusters=10,
        limit=1,
        weights=None,
        dim_red="PCA",
        quiet=False,
        export=False,
    ):
        print("Input data type:", type(input_data))
        if hasattr(input_data, '__dict__'):
            print("Input data attributes:")
            for attr, value in input_data.__dict__.items():
                print(f"  {attr}: {type(value)}")

        if weights is None:
            weights = [1, 1, 1]  # Default weights

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Convert ModelOutput to DataFrame if necessary
        if not isinstance(input_data, pd.DataFrame):
            print("Converting input_data to DataFrame")
            if hasattr(input_data, 'trials'):
                input_df = pd.DataFrame([vars(trial) for trial in input_data.trials])
            else:
                print("Error: input_data does not have 'trials' attribute")
                return None
        else:
            input_df = input_data

        print("Input DataFrame shape:", input_df.shape)
        print("Input DataFrame columns:", input_df.columns)

        # Prepare data
        if cluster_by == "hyperparameters":
            # Instead of filtering by "hyper", use all numeric columns except 'trial' and 'iterations'
            features = input_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['trial', 'iterations']]
        else:
            features = all_media

        print("Selected features:", features)

        if not features:
            print("Error: No features selected for clustering")
            return None

        X = input_df[features]
        print("Feature data shape:", X.shape)
        print("Feature data head:")
        print(X.head())

        if X.empty:
            print("Error: No data available for clustering")
            return None

        # Ensure max_clusters doesn't exceed the number of samples
        max_clusters = min(max_clusters, X.shape[0] - 1)

        # Dimensionality Reduction
        if dim_red == "PCA" and X.shape[1] > 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=min(2, X.shape[1]), random_state=self.seed)
            X_pca = pca.fit_transform(X_scaled)
            X = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

        # Determine number of clusters
        if k == "auto":
            k = self._determine_optimal_k(X, max_clusters)
        else:
            k = min(k, X.shape[0] - 1)  # Ensure k doesn't exceed number of samples

        # Clustering
        if k > 1:
            kmeans = KMeans(n_clusters=k, random_state=self.seed)
            clusters = kmeans.fit_predict(X)
            X['Cluster'] = clusters
        else:
            print("Warning: Not enough distinct data points for meaningful clustering.")
            X['Cluster'] = 0

        # Plotting results
        if not quiet:
            self._plot_clusters(X)

        # Select top models per cluster based on weighted errors
        top_models = self._select_top_models(X, weights, limit)

        return top_models

    def _determine_optimal_k(self, df, max_clusters):
        if df.shape[0] <= 2:
            return 1  # If we have 2 or fewer points, we can't do meaningful clustering

        sse = []
        for k in range(1, min(max_clusters, df.shape[0]) + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.seed)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)

        # Simple elbow method
        diffs = np.diff(sse)
        elbow_point = np.argmax(diffs) + 1  # Adding 1 because diff reduces the array size by 1

        return min(elbow_point + 1, df.shape[0] - 1)  # Ensure we don't exceed n_samples - 1

    def _plot_clusters(self, X):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        if 'PC2' in X.columns:
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=X, palette="viridis")
        else:
            sns.scatterplot(x='PC1', y=X.index, hue='Cluster', data=X, palette="viridis")
        plt.title("Cluster Plot")
        plt.show()


    def _select_top_models(self, df, weights, limit):
            # For now, just return the top 'limit' rows
            return df.head(limit)


# Example usage:
# analyzer = ModelClustersAnalyzer()
# results = analyzer.model_clusters_analyze(input_df, 'revenue', all_media=['TV', 'Radio', 'Online'])
