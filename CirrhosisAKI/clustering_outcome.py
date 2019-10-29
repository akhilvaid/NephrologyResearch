#!/bin/python

import os
import sqlite3
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()

from scipy.stats import chi2_contingency

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, OPTICS, DBSCAN
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# palette = sns.color_palette('deep').as_hex()
outer_cmap = plt.get_cmap('tab20c')
outer_colors = outer_cmap([1, 5, 13])
palette = outer_colors.tolist() + [outer_cmap(9), '#e66c6a', outer_cmap(17)]


class Cluster9000:
    def __init__(self, df_X, df_admissions, n_clusters, image_save_dir, ask_to_save=False):
        self.df_X = df_X
        self.df_admissions = df_admissions  # Should have all OUTCOME_*
        self.n_clusters = n_clusters

        self.image_save_dir = image_save_dir
        self.ask_to_save = ask_to_save

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
            'Agglomerative': AgglomerativeClustering(self.n_clusters)}
            # 'DBSCAN': DBSCAN()}
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

                if self.ask_to_save:
                    save = input('Save data? ')
                    if save == 'y':
                        filename = self.image_save_dir + '_' + plot_name.replace('.png', '')
                        df_out = df.assign(CLUSTERS=labels)
                        df_out.to_pickle(filename + '.pickle')

                # Plot this dataframe - Only the first 2 features will be plotted
                os.makedirs(self.image_save_dir, exist_ok=True)
                save_path = os.path.join(self.image_save_dir, plot_name)

                plt.figure()
                df_X = df_X.assign(CLUSTERS=labels)
                df_X['CLUSTERS'] = df_X['CLUSTERS'].apply(lambda x: x + 1)
                fig = sns.scatterplot(
                    x=df_X['FEATURE_0'], y=df_X['FEATURE_1'],
                    hue='CLUSTERS', data=df_X, palette=palette[:len(set(labels))])

                fig.set(
                    xticklabels=[], yticklabels=[],
                    xlabel='', ylabel='',
                    title='SPECTRAL CLUSTERING')

                plt.tight_layout()
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


def continuous_outcomes(df_cluster, df_admissions, cluster_col):
    # df_cluster should have a HADM_ID as its index and one column
    # corresponding to each cluster the record is in
    # outcomes should have a column each for
    # database table, column, identifier

    # Database
    database = sqlite3.connect('MIMIC.db')

    # Defining outcomes here for now
    dict_le = {
        50983: 'Sodium',
        50971: 'Potassium',
        50882: 'Bicarbonate',
        50826: 'Tidal Volume',
        50802: 'Base Excess',
        50804: 'Calculated Total CO2',
        51274: 'PT',
        51275: 'PTT',
        50813: 'Lactate',
        50954: 'LDH',
        50878: 'AST',
        50861: 'ALT',
        51251: 'Metamyelocytes',
        51144: 'Bands',
        50868: 'Anion gap',
        50912: 'Creatinine (Blood)',
        51082: 'Creatinine (Urine)',
        51476: 'Epithelial cells (Urine)'
    }
    str_labevents = ','.join([str(i) for i in dict_le])

    print('Creating dataframe')
    df_le = pd.read_sql_query(
        f'SELECT HADM_ID, ITEMID, CHARTTIME, VALUE FROM LABEVENTS WHERE ITEMID IN ({str_labevents})',
        database)

    # Set datatypes
    print('Setting dtypes')
    df_le['HADM_ID'] = pd.to_numeric(
        df_le['HADM_ID'], errors='coerce', downcast='integer')
    df_le['ITEMID'] = pd.to_numeric(
        df_le['ITEMID'], errors='coerce', downcast='integer')
    df_le['CHARTTIME'] = pd.to_datetime(df_le['CHARTTIME'])
    df_le['VALUE'] = pd.to_numeric(
        df_le['VALUE'], errors='coerce')
    df_le = df_le.dropna()

    df_le['HADM_ID'] = df_le['HADM_ID'].astype('int')
    df_le.set_index('HADM_ID', inplace=True)

    # Add admissions data to this
    # It's assumed that df_admissions doesn't have an HADM_ID index
    df_admissions = df_admissions.set_index('HADM_ID')
    df_le = df_le.merge(
        df_admissions[['RESTRICTION_TIME']],
        left_index=True, right_index=True, how='inner')

    # Restrict lab values according to a timedelta of 2 days
    t_delta = datetime.timedelta(days=2)
    keep = df_le.apply(
        lambda x: (x['CHARTTIME'] - x['RESTRICTION_TIME']) >= t_delta,
        axis=1)
    df_le = df_le.assign(KEEP=keep)
    df_le = df_le.query('KEEP == True')

    # Add clustering data to the LABEVENTS dataframe
    df_final = df_le.merge(
        df_cluster,
        left_index=True, right_index=True, how='inner')
    df_final.to_pickle('df_final.pickle')
    df_final = df_final[['ITEMID', 'VALUE', cluster_col]]  # Drop unnecessary columns

    # Group by cluster and ITEMID and remove outliers
    df_final_placeholder = pd.DataFrame()
    df_final_g = df_final.groupby([cluster_col, 'ITEMID'])
    for this_group in df_final_g:
        this_df = this_group[1]

        this_df = this_df[
            np.abs(this_df.VALUE - this_df.VALUE.mean()) <= (3 * this_df.VALUE.std())]
        df_final_placeholder = df_final_placeholder.append(this_df)

    df_final = df_final_placeholder.copy()

    # Plot a new something per specifics
    list_le = list(dict_le)
    n_clusters = len(df_final[cluster_col].unique())

    plot_rows = 3
    plot_columns = 6
    fig, ax = plt.subplots(plot_rows, plot_columns)
    for row in range(plot_rows):
        for col in range(plot_columns):

            # Get data for current dataframe
            current_le = row * 6 + col
            this_le = list_le[current_le]
            le_human_readable = dict_le[this_le]

            this_df = df_final.query('ITEMID == @this_le')
            this_df['ITEMID'] = this_df['ITEMID'].apply(
                lambda x: le_human_readable)  # Series - does not need an axis

            ax[row][col] = sns.violinplot(
                x='ITEMID', y='VALUE', data=this_df,
                hue=cluster_col, ax=ax[row][col],
                palette=palette[:n_clusters])
            ax[row][col].set_xlabel('')  # Leave graph axes unlabelled
            ax[row][col].set_ylabel('')
            ax[row][col].get_legend().remove()

    fig.show()


def discrete_outcomes(df_cluster, cluster_col):
    # Discrete outcomes are best generated at the time of generating HADM_IDs
    # Just keep attaching outcomes to df_cluster
    # Plot them

    # Outcomes from admissions
    outcomes_adm = [
        i for i in df_cluster.columns if i.startswith('OUTCOME_')]

    df_cluster[cluster_col] = df_cluster[cluster_col].apply(
        lambda x: 'CLUSTER ' + str(x + 1))  # Zero indexing? Zero indexing.

    # Parameters for the pie-chart
    n_clusters = len(df_cluster[cluster_col].unique())
    outer_colors = palette[:n_clusters]

    # Plot
    # Each plot will be individual unless specified otherwise
    for this_col in outcomes_adm:
        # Track inner circle iterations
        # I'm not fond of global variables
        global inner_circle_perc_iterations
        inner_circle_perc_iterations = 0

        n_outcomes = len(df_cluster[this_col].unique())
        inner_colors = palette[n_clusters:(n_clusters + n_outcomes)]

        # Set outer and inner values
        df_crosstab = pd.crosstab(df_cluster[cluster_col], df_cluster[this_col])
        outer_values = df_crosstab.apply(lambda x: x.sum(), axis=1)
        inner_values = df_crosstab.values.reshape(1, -1)

        plt.figure(figsize=(16, 16))

        # Plot outer circle
        plt.pie(
            outer_values,
            explode=[0.02] * len(outer_values),
            startangle=90,
            frame=True,
            colors=outer_colors,
            radius=2,
            autopct='%1.1f%%',
            pctdistance=0.85,
            wedgeprops=dict(width=0.7, edgecolor='w'),
            labels=df_crosstab.index)

        # Plot inner circle
        plt.pie(
            inner_values,
            explode=[0.01] * len(inner_values[0]),
            startangle=90,
            radius=1.5,
            colors=inner_colors,
            autopct=make_autopct(inner_values, n_outcomes),
            pctdistance=0.75,
            labeldistance=None,
            wedgeprops=dict(width=0.6, edgecolor='w'),
            labels=df_crosstab.columns.to_list() * n_clusters)

        # Create plot legend
        print(df_crosstab)
        prefix = input('Legend? ')
        title = input('Title? ')

        patches = []
        for i in range(n_outcomes):
            this_color = inner_colors[i]
            label = prefix + str(df_crosstab.columns[i])
            this_patch = mpatches.Patch(color=this_color, label=label)
            patches.append(this_patch)
        plt.legend(handles=patches)
        plt.title(title)

        plt.axis('equal')
        plt.tight_layout()

        plt.show()


def make_autopct(values, at_a_time):
    values = values[0]

    def my_autopct(pct):
        global inner_circle_perc_iterations

        final = [
            values[i * at_a_time:(i + 1) * at_a_time]
            for i in range((len(values) + at_a_time - 1) // at_a_time)]

        current_chunk = inner_circle_perc_iterations // at_a_time
        total = sum(final[current_chunk])
        new_perc = values[inner_circle_perc_iterations] * 100 / total

        inner_circle_perc_iterations += 1
        return '{p:.1f}%'.format(p=new_perc)

    return my_autopct
