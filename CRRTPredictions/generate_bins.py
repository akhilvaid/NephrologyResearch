#!/bin/python
# Process the measurement dataframe to create chunks of pre-defined
# size and calculate the required list of functions
# for each of those chunks per HADM_ID
# Use df.resample - Acts as a groupby for a dataframe with a timeseries index

import pandas as pd
import numpy as np


def split_at_restriction_time(df_measurements, df_restrict, restriction_var):
    # Return 2 dataframes:
    # One with measurements before restriction time
    # One with after

    df_measurements['HADM_ID'] = df_measurements['HADM_ID'].astype('int')
    df_measurements['CHARTTIME'] = pd.to_datetime(df_measurements.CHARTTIME)

    df_measurements = df_measurements.merge(df_restrict, on='HADM_ID', how='left')
    df_before = df_measurements[df_measurements['CHARTTIME'] < df_measurements[restriction_var]]
    df_after = df_measurements[df_measurements['CHARTTIME'] > df_measurements[restriction_var]]

    df_before = df_before[['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']]
    df_after = df_after[['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']]

    return df_before, df_after


def create_time_chunks(df_measurements, df_restrict, restriction_var, hour_interval, before, function, outfile):
    # hour_interval is expected to be a string
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    # It's recommended to keep this large because
    # the dataframe swells in size otherwise

    # Set correct dtypes and cleanup
    df_measurements['HADM_ID'] = df_measurements['HADM_ID'].astype('int')
    df_measurements['CHARTTIME'] = pd.to_datetime(df_measurements.CHARTTIME)
    df_measurements['VALUE'] = pd.to_numeric(df_measurements['VALUE'], errors='coerce')

    # Housekeeping
    df_measurements = df_measurements.drop_duplicates()
    df_measurements = df_measurements.dropna()
    df_measurements = df_measurements.sort_values(['HADM_ID', 'ITEMID', 'CHARTTIME'])

    # Merge with restriction dataframe
    # Create timedelta in a new column - This keeps values from spilling out of bins
    df_measurements = df_measurements.merge(
        df_restrict[['HADM_ID', restriction_var]], on='HADM_ID', how='left')
    df_measurements['TD'] = df_measurements.apply(
        lambda x: x['CHARTTIME'] - x[restriction_var], axis=1)
    df_measurements = df_measurements.drop(['CHARTTIME', restriction_var], axis=1)

    # Create bins
    df_measurements = df_measurements.groupby(['HADM_ID', 'ITEMID'])
    df_measurements = df_measurements.resample(hour_interval, on='TD')

    # Iterate for each applicable function
    if function == 'MEAN':
        df_bins = df_measurements.mean()['VALUE']
    elif function == 'STD':
        df_bins = df_measurements.std()['VALUE']
    elif function == 'COUNT':
        df_bins = df_measurements.count()['VALUE']
    elif function == 'SUM':
        df_bins = df_measurements.sum()['VALUE']
    else:
        print('Error: No valid function specified')
        exit()

    # Rename each measurement bin to a number
    df_bins = df_bins.reset_index()
    df_bins['COUNTER'] = df_bins.drop('VALUE', axis=1).groupby(['HADM_ID', 'ITEMID']).cumcount()

    before_signifier = ''
    if before:
        df_bins['COUNTER'] += 1
        before_signifier = 'M'

    df_bins['NEWIDX'] = (
        df_bins['ITEMID'].astype('str') + '_' + before_signifier +
        df_bins['COUNTER'].astype('str') + '_' + function)
    df_bins = df_bins.drop(['TD', 'COUNTER'], axis=1)

    df_bins = df_bins.pivot(index='NEWIDX', columns='HADM_ID', values='VALUE')
    df_bins = df_bins.replace(np.nan, 0)
    df_bins = df_bins.T

    # Above dataframe will have to be reshaped to get values for each time jump
    # for each patient to feed into the RNN core
    # We're not using the original df since we don't want any of the bins unrepresented.
    outfile_name = f'RNN_{outfile}_AfterRT_{function}.pickle'
    if before:
        outfile_name = f'RNN_{outfile}_BeforeRT_{function}.pickle'

    df_final = pivot_for_rnn(df_bins, function)
    print('Writing:', outfile_name)
    df_final.to_pickle(outfile_name)


def pivot_row(df):
    hadm_id = df.name
    df = df.reset_index()
    df.columns = ['NEWIDX', 'VALUE']

    # Create a column that has the ITEMID
    df['ITEMID'] = df['NEWIDX'].apply(lambda x: x.split('_')[0])
    df['TIMESTEPS'] = (df.groupby('ITEMID').VALUE.cumcount() + 1).astype(str)
    df['TIMESTEPS'] = df['TIMESTEPS'].apply(lambda x: 'T' + x.zfill(5))  # Add upto 5 leading 0s
    df = df.pivot(index='TIMESTEPS', columns='ITEMID', values='VALUE')
    df = df.replace(np.nan, 0)
    df['HADM_ID'] = hadm_id
    df = df.reset_index()

    return df


def pivot_for_rnn(df_binned, function):
    df_final = pd.DataFrame()

    for i in range(df_binned.shape[0]):
        pivoted_df = pivot_row(df_binned.iloc[i])
        df_final = df_final.append(pivoted_df, sort=True)

    df_final = df_final.set_index(['HADM_ID', 'TIMESTEPS'])
    df_final.columns = [i + function for i in df_final.columns]  # Put an underscore here if needed
    df_final = df_final.reset_index()

    return df_final
