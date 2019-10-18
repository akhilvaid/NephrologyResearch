#!/bin/python

import fancyimpute
import pandas as pd

from sklearn.neighbors import KDTree


def feature_extraction(df, data_threshold):
    # The df here is the dataframe for the whole table
    # It's supposed to have an HADM_ID, an ITEMID, and a VALUE
    # This function will return a list of ITEMIDs to use

    # Process dataframe first
    # It requires HADM_ID, ITEMID, CHARTTIME, VALUE
    df['ITEMID'] = pd.to_numeric(df.ITEMID, errors='coerce', downcast='integer')
    df['HADM_ID'] = pd.to_numeric(df.HADM_ID, errors='coerce', downcast='integer')
    df['CHARTTIME'] = pd.to_datetime(df.CHARTTIME)
    df['VALUE'] = pd.to_numeric(df.VALUE, errors='coerce') # TODO - Include categorical data
    df = df.dropna()
    df = df.sort_values(['HADM_ID', 'ITEMID', 'CHARTTIME']).drop('CHARTTIME', axis=1)
    df_g = df.groupby('HADM_ID')

    # Store counts here
    df_all_counts = pd.DataFrame()

    for this_group in df_g:
        this_df = this_group[1]
        this_df = this_df.drop('HADM_ID', axis=1)
        df_counts = this_df.sort_values('ITEMID').groupby('ITEMID').count()

        # Add these values to the counter dataframe
        suffix = '_' + str(this_group[0])
        df_all_counts = df_all_counts.join(df_counts, how='outer', rsuffix=suffix)

    # Take the transpose of this table to calculate features which have data for
    # more than 60% of cases
    # Also filter out cases who have less than 60% data
    feature_count = df_all_counts.T.count() / len(df_all_counts.T)
    features = feature_count[feature_count >= data_threshold].index.to_list()
    df_cleaned_features = df_all_counts.T[features].T

    case_count = df_cleaned_features.count() / len(df_cleaned_features)
    cases = case_count[case_count > data_threshold].index.to_list()
    cases = [int(i.replace('VALUE_', '')) for i in cases[1:]]  # TODO - Account for first patient

    # Create filtered dataframe
    df_filtered = df.query('ITEMID in @features and HADM_ID in @cases')
    return df_filtered


def create_df_central_tendency(df):
    # Create median and standard deviation dataframe
    # Impute from these values
    df_central_tendency = pd.DataFrame()

    # Iterate over this dataframe by HADM_ID
    hadm_ids = df.HADM_ID.unique().tolist()
    for this_hadm in hadm_ids:
        # Store this HADM_ID's values in this df
        df_this_hadm = pd.DataFrame()

        df_hadm_all_data = df.query('HADM_ID == @this_hadm')
        df_hadm_g = df_hadm_all_data.groupby('ITEMID')

        for this_group in df_hadm_g:
            this_itemid = str(this_group[0])
            df_item = this_group[1]

            median = df_item.VALUE.median()
            std = df_item.VALUE.std()
            this_index = [
                this_itemid + '_MEDIAN',
                this_itemid + '_STD']

            df_item = pd.DataFrame(
                [median, std], index=this_index, columns=[this_hadm])
            df_this_hadm = df_this_hadm.append(df_item)

        # Outer join all df_hadms to the df_central_tendency
        df_central_tendency = df_central_tendency.join(
            df_this_hadm, how='outer')

    return df_central_tendency.T


def impute_and_get_nearest_neighbors(df):
    # THE DATAFRAME INDEX NEEDS TO BE THE HADM_ID
    # Returns a dictionary that has the nearest neighbor HADMs
    # for each HADM with a missing value

    # Find indices that have ALL values
    # Subtract that from all dataframe row indices
    all_vals_present = set(df.dropna().index)
    all_indices = set(df.index)
    missing_hadms = sorted(list(all_indices - all_vals_present))

    # Impute
    # Find nearest neighbors (5)
    df_imputed = pd.DataFrame(
        fancyimpute.KNN(k=5).fit_transform(df),
        index=df.index,
        columns=df.columns)

    # Find nearest neighbors from the original dataframe
    tree = KDTree(df)
    _, indices = tree.query(
        df_imputed.loc[missing_hadms], k=5)

    # Create dataframe with this data
    df_nearest_neighbors = pd.DataFrame(indices)
    df_nearest_neighbors['HADM_ID'] = missing_hadms
    df_nearest_neighbors = df_nearest_neighbors.set_index('HADM_ID')
    df_nearest_neighbors_hadms = df_nearest_neighbors.apply(
        lambda x: df.index[x[[0, 1, 2, 3, 4]]].to_list(), axis=1)

    return df_nearest_neighbors_hadms.to_dict()
