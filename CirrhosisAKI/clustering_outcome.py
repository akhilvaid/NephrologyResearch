#!/bin/python

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.stats import chi2_contingency

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, OPTICS, DBSCAN
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class Cluster9000:
    def __init__(self, df_X, df_admissions, n_clusters, image_save_dir):
        self.df_X = df_X
        self.df_admissions = df_admissions  # Should have all OUTCOME_*
        self.n_clusters = n_clusters

        self.image_save_dir = image_save_dir

        self.decomposition_algorithms = {
            'Encoder': None,
            'PCA': PCA(n_components=2),
            'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10, n_jobs=-1),
            'FactorAnalysis': FactorAnalysis(n_components=2),
            'TSNE': TSNE(perplexity=30)}

        # It's better to init there here because of differing params
        self.clustering_algorithms = {
            'Spectral': SpectralClustering(self.n_clusters, n_jobs=-1),
            'KMeans': KMeans(self.n_clusters, n_jobs=-1),
            'GaussianMixture': GaussianMixture(self.n_clusters, n_init=10),
            'Agglomerative': AgglomerativeClustering(self.n_clusters),
            'DBSCAN': DBSCAN()}
            # 'AffinityPropagation': AffinityPropagation(),
            # 'OPTICS': OPTICS()}

    def process_X(self, df):
        # This processes the DECOMPOSITION dataframe
        # Rename all the columns of df to FEATURE_*
        clustering_features = ['FEATURE_' + str(i) for i in df.columns]
        df.columns = clustering_features

        # Join df_admissions to this - HADM_ID is the index of df
        outcome_features = [
            i for i in self.df_admissions if i.startswith('OUTCOME_')]
        df_admin = self.df_admissions[outcome_features + ['HADM_ID']].set_index('HADM_ID')

        df = df.join(df_admin, how='inner')

        # Dataframe, clustering features, outcome features
        return df, clustering_features, outcome_features

    def significance_calculator(self, df, clustering_features, outcome_features, labels, plot_name):
        total_patients = len(df)

        for this_outcome in outcome_features:
            outcome_series = df[this_outcome]
            df_significance = pd.crosstab(labels, outcome_series)  # Order of args is important

            stat, p, _, _ = chi2_contingency(df_significance)

            # Check for validity only in case the p value is < .05
            if p < 0.05:
                df_significance['OUTCOME_PERC'] = df_significance.apply(
                    lambda x: x[1] * 100 / x.sum(), axis=1)
                df_significance['POPULATION_PERC'] = df_significance.apply(
                    lambda x: (x[False] + x[True]) * 100 / total_patients, axis=1)
                df_significance.index.name = 'CLUSTER'

                df_X = df[clustering_features]
                sil_score = silhouette_score(df_X, labels)
                ch_score = calinski_harabasz_score(df_X, labels)
                db_score = davies_bouldin_score(df_X, labels)

                print()
                print(df_significance)
                print('Chi2 stat:', stat, 'p-value:', p)
                print('Silhouette score:', sil_score)
                print('Davies Bouldin score:', db_score)
                print('Calinski Harabasz score:', ch_score)
                print()

                # Plot this dataframe - Only the first 2 features will be plotted
                plt.figure()
                df_X = df_X.assign(LABELS=labels)
                palette = sns.color_palette('deep', len(set(labels)))
                fig = sns.scatterplot(
                    x=df_X['FEATURE_0'], y=df_X['FEATURE_1'],
                    hue='LABELS', data=df_X, palette=palette)

                os.makedirs(self.image_save_dir, exist_ok=True)
                save_path = os.path.join(self.image_save_dir, plot_name)
                fig.figure.savefig(save_path, dpi=600)
                plt.clf()

    def cluster(self):
        # Clusters will be created and checked for outcome while also being
        # simultaneously validated
        # All clusters meeting criteria will be shown / plotted
        print('Finding clusters')
        plt.figure()

        for decomp in self.decomposition_algorithms:
            if decomp == 'Encoder':
                df_decomp = self.df_X.copy()

            else:
                df_decomp = pd.DataFrame(
                    self.decomposition_algorithms[decomp].fit_transform(self.df_X),
                    index=self.df_X.index)

            # Plots the first 2 axes of df_decomp for EVERYTHING
            try:
                plt.figure()
                fig = sns.scatterplot(x=df_decomp[0], y=df_decomp[1], data=df_decomp)
                fig.figure.savefig(decomp + '.png', dpi=600)
                plt.clf()
            except:
                print('Plotting error', decomp)

            df_decomp, clustering_features, outcome_features = self.process_X(df_decomp)

            # Start clustering
            for cluster_alg in self.clustering_algorithms:
                print(decomp, cluster_alg)

                try:
                    plot_name = f'{decomp}_{cluster_alg}_{self.n_clusters}.png'

                    cluster_labels = self.clustering_algorithms[cluster_alg].fit_predict(
                        df_decomp[clustering_features])

                    # Calculate significance and validity
                    self.significance_calculator(
                        df_decomp,
                        clustering_features,
                        outcome_features,
                        cluster_labels,
                        plot_name)

                except KeyboardInterrupt:
                    print('Skipping', decomp, cluster_alg)                    
