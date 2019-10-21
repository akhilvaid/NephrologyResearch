#!/bin/python

import sqlite3
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.stats import chi2_contingency, fisher_exact

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, OPTICS, DBSCAN
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding


class Cluster:

    def __init__(self, df, clusters):
        self.df = df  # Autoencoder output
        self.clusters = clusters

        self.df_clusters = pd.DataFrame()
        self.df_final = None

        self.decomposition_algorithms = {
            'Encoder': None,
            'PCA': PCA(n_components=2),
            'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10, n_jobs=-1),
            'FactorAnalysis': FactorAnalysis(n_components=2),
            'TSNE': TSNE}

        # It's better to init there here because of differing params
        self.clustering_algorithms = {
            'Spectral': SpectralClustering(self.clusters, n_jobs=-1),
            'KMeans': KMeans(self.clusters, n_jobs=-1),
            'GaussianMixture': GaussianMixture(self.clusters, n_init=10),
            # 'AffinityPropagation': AffinityPropagation(),
            'Agglomerative': AgglomerativeClustering(self.clusters),
            # 'OPTICS': OPTICS(),
            'DBSCAN': DBSCAN()}

    def cluster(self, plot_clusters=False):
        print('Finding clusters')

        for decomp in self.decomposition_algorithms:
            if decomp == 'Encoder':
                df_decomp = self.df.copy()

            elif decomp == 'TSNE':
                df_decomp = pd.DataFrame(
                    TSNE(perplexity=20).fit_transform(self.df),
                    index=self.df.index)

            else:
                df_decomp = pd.DataFrame(
                    self.decomposition_algorithms[decomp].fit_transform(self.df),
                    index=self.df.index)

            # Plot decomposed dataframes
            if plot_clusters:
                try:
                    plt.figure()
                    fig = sns.scatterplot(x=df_decomp[0], y=df_decomp[1], data=df_decomp)
                    fig.figure.savefig(decomp + '.png', dpi=600)
                except:
                    print('Plotting error')

            for cluster_alg in self.clustering_algorithms:
                print(decomp, cluster_alg)
                try:
                    df_cluster = pd.DataFrame(
                        self.clustering_algorithms[cluster_alg].fit_predict(df_decomp),
                        index=df_decomp.index)
                    column_string = decomp + '_' + cluster_alg
                    df_cluster.columns = [column_string]

                    # Plot each cluster
                    if decomp != 'Encoder' and plot_clusters:
                        df_temp = df_decomp.copy()
                        df_temp['CLUSTER'] = df_cluster[column_string]
                        plt.figure()
                        palette = sns.color_palette('deep', df_temp.CLUSTER.unique().shape[0])
                        fig = sns.scatterplot(
                            x=df_temp[0], y=df_temp[1],
                            hue='CLUSTER', data=df_temp, palette=palette)
                        fig.figure.savefig(column_string + '.png', dpi=600)
                        plt.clf()

                    if self.df_clusters.empty:
                        self.df_clusters = df_cluster
                    else:
                        self.df_clusters = self.df_clusters.join(df_cluster, how='inner')
                except:
                    print('Error', decomp, cluster_alg)


    def outcome_generator(self):
        if self.df_clusters.empty:
            return
        print('Generating outcomes')

        database = sqlite3.connect('MIMIC.db')

        # Attach SUBJECT_IDS to dataframe and reset index
        hadm_ids = self.df_clusters.index.to_list()
        hadm_ids_string = ','.join([str(i) for i in hadm_ids])

        df_id = pd.read_sql_query(
            f'SELECT SUBJECT_ID, HADM_ID FROM ADMISSIONS WHERE HADM_ID IN ({hadm_ids_string})',
            database)
        df_id = df_id.set_index('HADM_ID')
        df_id.index = df_id.index.astype('int')

        self.df_final = self.df_clusters.join(df_id, how='inner')
        self.df_final = self.df_final.reset_index()

        # Attach outcomes to this dataframe
        # Mortality - Read from the hospital admissions csv file
        df_exp = pd.read_csv('data_HospitalAdmissions.csv')
        df_exp = df_exp[['SUBJECT_ID', 'DEATH']]
        self.df_final = self.df_final.merge(df_exp, on='SUBJECT_ID', how='left')

        # Dialysis
        # Dialysis is 5498 - Peritoneal, 3995 - Hemodialysis (PROCEDURES_ICD)
        df_dialysis = pd.read_sql_query(
            f'SELECT HADM_ID, ICD9_CODE FROM PROCEDURES_ICD WHERE ICD9_CODE IN (5498, 3995)',
            database)
        df_dialysis = df_dialysis.query('HADM_ID in @hadm_ids')
        df_dialysis = df_dialysis.drop_duplicates()

        self.df_final = self.df_final.merge(df_dialysis, on='HADM_ID', how='left')
        self.df_final['DIALYSIS'] = self.df_final.apply(
            lambda x: False if pd.isnull(x['ICD9_CODE']) else True, axis=1)

    def significance_calculator(self):
        print('Calculating significance')

        # Create new dataframe
        df_clusters_only = self.df_final.drop([
            'HADM_ID', 'SUBJECT_ID', 'ICD9_CODE'], axis=1)
        df_clusters_only['DEATH'] = df_clusters_only.apply(
            lambda x: 1 if x['DEATH'] is True else 0, axis=1)
        df_clusters_only['DIALYSIS'] = df_clusters_only.apply(
            lambda x: 1 if x['DIALYSIS'] is True else 0, axis=1)

        # Go through each algorithm and generate report
        # Calculate the significance of each outcome using the chi2 metric
        outcomes = set(['DEATH', 'DIALYSIS'])
        iter_columns = set(df_clusters_only.columns) - outcomes

        for this_column in sorted(list(iter_columns)):
            for outcome in outcomes:

                df_22 = pd.crosstab(df_clusters_only[this_column], df_clusters_only[outcome])
                stat, p, dof, expected = chi2_contingency(df_22)

                try:
                    odds_ratio, p_fisher = fisher_exact(df_22)
                except ValueError:
                    p_fisher = 1

                check_here = ''
                if p < .05 or p_fisher < .05:
                    check_here = '!!!!'

                    df_22['MORTALITY_PERC'] = df_22.apply(
                        lambda x: x[1] * 100 / x.sum(), axis=1)
                    print(df_22)

                print(this_column, outcome, stat, p, p_fisher, check_here)

            print()
