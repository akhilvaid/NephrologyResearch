#!/bin/python
# This uses hardcoded paths - check before reuse

import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import generate_bins
import data_references


# Features that will remain constant for each patient during patient stay
def constant_features():
    print('Processing constant features')

    # Demographics
    df_demographics = pd.read_pickle('data_Demographics.pickle')
    df_demographics = df_demographics.set_index('HADM_ID')

    # Prescriptions - All drugs the patient was exposed to during stay
    df_prescriptions = pd.read_pickle('data_Prescriptions.pickle')
    df_prescriptions = df_prescriptions.T

    # NOTE - Best way to get dummies for non unique values
    # Past medical history
    df_pmh = pd.read_csv('data_ChartEvents_C_ALL.csv')
    df_pmh = df_pmh.query('ITEMID == "PMH"').drop(['ITEMID', 'CHARTTIME'], axis=1)
    grouped = df_pmh.groupby('HADM_ID').VALUE.apply(lambda lst: tuple((k, 1) for k in lst))
    category_dicts = [dict(tuples) for tuples in grouped]
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(category_dicts)
    df_pmh_reduced = pd.DataFrame(X, columns=v.get_feature_names(), index=grouped.index)
    df_pmh_reduced = df_pmh_reduced.astype('int8')

    # Join the constant data dataframes. The how depends on the sort of data
    df_final = df_demographics.join(df_prescriptions)
    df_final = df_final.join(df_pmh_reduced, how='outer')
    df_final = df_final.replace(np.nan, 0)

    df_final.to_pickle('RNN_CONSTANT.pickle')


# Features that will vary with time
# Will have to be binned into bin_intervals
def variable_features_labevents(df_restrict, bin_interval):
    print('Processing variable features - labevents')

    df_le = pd.read_csv('data_LabEvents_2.csv')
    df_le_before, df_le_after = generate_bins.split_at_restriction_time(
        df_le, df_restrict, 'STARTTIME')

    for this_function in ('MEAN', 'STD', 'COUNT'):
        print('Processing:', this_function)

        generate_bins.create_time_chunks(
            df_le_before, df_restrict,
            'STARTTIME', bin_interval,
            True, this_function, 'LABEVENTS')
        generate_bins.create_time_chunks(
            df_le_after, df_restrict,
            'STARTTIME', bin_interval,
            False, this_function, 'LABEVENTS')


def variable_features_chartevents(df_restrict, bin_interval):
    print('Processing variable features - chartevents')

    df_ce = pd.read_csv('data_ChartEvents_V_2.csv')

    # Calculate GCS scores by substitution
    df_ce.update(df_ce.query('ITEMID == "MV-GCS-E"').replace(data_references.dict_gcs_eye))
    df_ce.update(df_ce.query('ITEMID == "MV-GCS-V"').replace(data_references.dict_gcs_verbal))
    df_ce.update(df_ce.query('ITEMID == "MV-GCS-M"').replace(data_references.dict_gcs_motor))

    df_ce_before, df_ce_after = generate_bins.split_at_restriction_time(
        df_ce, df_restrict, 'STARTTIME')

    for this_function in ('MEAN', 'STD', 'COUNT'):
        print('Processing:', this_function)

        generate_bins.create_time_chunks(
            df_ce_before, df_restrict,
            'STARTTIME', bin_interval,
            True, this_function, 'CHARTEVENTS')
        generate_bins.create_time_chunks(
            df_ce_after, df_restrict,
            'STARTTIME', bin_interval,
            False, this_function, 'CHARTEVENTS')


def variable_features_inputevents(df_restrict, bin_interval):
    print('Processing variable features - inputevents')

    df_ie = pd.read_pickle('data_InputEvents_2.pickle')
    df_ie = df_ie.drop('PATIENTWEIGHT', axis=1)
    df_ie = df_ie.rename(
        {'START': 'CHARTTIME', 'AMOUNT': 'VALUE'}, axis=1)  # Variable names are hardcoded

    df_ie_before, df_ie_after = generate_bins.split_at_restriction_time(
        df_ie, df_restrict, 'STARTTIME')

    for this_function in ('MEAN', 'STD', 'COUNT'):
        print('Processing:', this_function)

        generate_bins.create_time_chunks(
            df_ie_before, df_restrict,
            'STARTTIME', bin_interval,
            True, this_function, 'INPUTEVENTS')
        generate_bins.create_time_chunks(
            df_ie_after, df_restrict,
            'STARTTIME', bin_interval,
            False, this_function, 'INPUTEVENTS')


def variable_features_outputevents(df_restrict, bin_interval):
    print('Processing variable features - outputevents')

    df_uo = pd.read_pickle('data_UrineOutput_Cumulative.pickle')
    df_uo.ITEMID = 'UrineOutput'

    df_uo_before, df_uo_after = generate_bins.split_at_restriction_time(
        df_uo, df_restrict, 'STARTTIME')

    for this_function in ('SUM',):
        print('Processing:', this_function)

        generate_bins.create_time_chunks(
            df_uo_before, df_restrict,
            'STARTTIME', bin_interval,
            True, this_function, 'OUTPUTEVENTS')
        generate_bins.create_time_chunks(
            df_uo_after, df_restrict,
            'STARTTIME', bin_interval,
            False, this_function, 'OUTPUTEVENTS')

#####################################################################

def consolidate_dataframes(list_df, after_days, interval_length):
    df_consolidated = pd.DataFrame()
    path_prefix = './RNN_Data'

    intervals_to_keep = None
    if after_days:
        intervals_to_keep = int(after_days * (24 / interval_length))
        print(f'Restricting to {after_days} days after restriction time')

    for this_pickle in list_df:
        # Iterate through each pickled file and do a just-in-case final sort
        df_current = pd.read_pickle(
            os.path.join(path_prefix, this_pickle))
        df_current = df_current.sort_values(['HADM_ID', 'TIMESTEPS'])

        # Restrict to these many intervals after the restriction time
        if intervals_to_keep:
            df_current = df_current.groupby('HADM_ID').head(intervals_to_keep)

        # This index simplifies merging
        df_current = df_current.set_index(['HADM_ID', 'TIMESTEPS'])

        # Copy or merge
        if df_consolidated.empty:
            df_consolidated = df_current.copy()
        else:
            df_consolidated = df_consolidated.join(df_current, how='inner')

    return df_consolidated.reset_index()


# Specift how many days after restriction time to keep here
def finalize_dataframe_RNN():
    df_before = consolidate_dataframes(
        data_references.list_variable_before, None, None)
    df_after = consolidate_dataframes(
        data_references.list_variable_after, 7, 6)

    # Sort shouldn't be needed - but JIC
    df_appended = df_before.append(df_after, sort=True, ignore_index=True)
    df_appended = df_appended.set_index('HADM_ID')

    # Already has an HADM_ID index
    df_constant = pd.read_pickle('./RNN_Data/RNN_CONSTANT.pickle')

    # Merge
    df_merged = df_appended.merge(
        df_constant, how='left', left_index=True, right_index=True)

    df_merged.to_pickle('df_RNN.pickle')
    print('Saved: df_RNN.pickle')
