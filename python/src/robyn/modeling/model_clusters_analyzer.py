# cluster.R https://github.com/facebookexperimental/Robyn/blob/python_rewrite/python/src/oldportedcode/cluster.py

# pyre-strict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ModelClustersAnalyzer:
    def __init__(self, seed=123):
        self.seed = seed
        np.random.seed(seed)

    def model_clusters_analyze(
        self,
        input_df,
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
        if weights is None:
            weights = [1, 1, 1]  # Default weights

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Prepare data
        if cluster_by == "hyperparameters":
            features = input_df.filter(regex="hyper").columns.tolist()
        else:
            features = all_media

        X = input_df[features]

        # Dimensionality Reduction
        if dim_red == "PCA":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=self.seed)
            X_pca = pca.fit_transform(X_scaled)
            X = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

        # Determine number of clusters
        if k == "auto":
            k = self._determine_optimal_k(X, max_clusters)

        # Clustering
        kmeans = KMeans(n_clusters=k, random_state=self.seed)
        clusters = kmeans.fit_predict(X)
        input_df["Cluster"] = clusters

        # Plotting results
        if not quiet:
            self._plot_clusters(X, clusters)

        # Select top models per cluster based on weighted errors
        top_models = self._select_top_models(input_df, weights, limit)

        return top_models

    def _determine_optimal_k(self, df, max_clusters):
        # This uses the Elbow Method to determine the optimal k
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.seed)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("SSE")
        plt.title("Elbow Method For Optimal k")
        plt.show()

        # Placeholder for actual determination logic
        return 3  # Example: return 3 as the optimal number of clusters

    def _plot_clusters(self, X, clusters):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=X.iloc[:, 0],
            y=X.iloc[:, 1],
            hue=clusters,
            palette="viridis",
            s=100,
            alpha=0.6,
        )
        plt.title("Cluster Plot")
        plt.show()

    def _select_top_models(self, df, weights, limit):
        # Placeholder for selecting top models based on weighted errors
        return df.head(limit)  # Example: return top 'limit' rows


# Example usage:
# analyzer = ModelClustersAnalyzer()
# results = analyzer.model_clusters_analyze(input_df, 'revenue', all_media=['TV', 'Radio', 'Online'])
