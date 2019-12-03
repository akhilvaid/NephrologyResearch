#!/bin/python

import sqlite3
import datetime

import pandas as pd
import numpy as np

database = sqlite3.connect('../MIMIC.db')


# NOTE
# Currently performing NO time restriction
# All drugs that a patient was exposed to during an admission are included
def prescriptions(df_restrict):
    # Get all HADM_IDs
    hadm_string = ','.join(df_restrict.HADM_ID.astype('str'))

    # Query prescriptions and set proper dtypes
    # df_p: Prescriptions dataframe
    df_p = pd.read_sql_query(
        f'SELECT HADM_ID, NDC FROM PRESCRIPTIONS WHERE HADM_ID IN ({hadm_string})',
        database)

    df_p['NDC'] = pd.to_numeric(df_p['NDC'], errors='coerce', downcast='integer')
    df_p = df_p.dropna()
    df_p = df_p.drop_duplicates()

    df_p['NDC'] = df_p['NDC'].astype('int')

    # Vectorize, vectorize, vectorize
    df_p['PLACEHOLDER'] = np.ones(len(df_p))
    df_p = df_p.pivot(index='NDC', columns='HADM_ID', values='PLACEHOLDER')
    df_p = df_p.replace(np.nan, 0)

    # Reduce memory requirement
    df_p = df_p.astype('int8')
    df_p = df_p.drop(0)

    return df_p


def inputevents_mv(df_restrict, restriction_var, dict_drugs, start_interval):
    all_hadms = df_restrict.HADM_ID.to_list()
    hadm_string = ','.join([str(i) for i in all_hadms])

    df_itemids = pd.DataFrame.from_dict(dict_drugs).T
    df_itemids = df_itemids.reset_index().rename({'index': 'NEWID'}, axis=1)
    df_itemids = df_itemids.melt(id_vars=['NEWID']).drop('variable', axis=1) 
    df_itemids.columns = ['NEWID', 'ITEMID']
    df_itemids = df_itemids.set_index('ITEMID')
    consolidated_item_ids = df_itemids.index.to_list()
    itemid_string = ','.join([str(i) for i in consolidated_item_ids])

    df_ie = pd.read_sql_query(
        f'SELECT * FROM INPUTEVENTS_MV WHERE HADM_ID IN ({hadm_string}) AND ITEMID IN ({itemid_string})',
        database)
    df_ie.rename({'STARTTIME': 'START', 'ENDTIME': 'END'}, axis=1, inplace=True)
    for this_item_id in df_itemids.index:
        df_ie = df_ie.replace(this_item_id, df_itemids.loc[this_item_id].NEWID)

    # dtypes
    df_ie['START'] = pd.to_datetime(df_ie['START'])

    # Merge with initial dataframe
    # Restrict starting time to after the restriction time
    df_ie = df_ie.merge(
        df_restrict[['HADM_ID', restriction_var]], on='HADM_ID', how='left')

    t_delta = datetime.timedelta(days=start_interval)
    df_ie = df_ie[t_delta > (df_ie[restriction_var] - df_ie['START'])]

    df_ie = df_ie[[
        'HADM_ID', 'START', 'ITEMID', 'AMOUNT', 'PATIENTWEIGHT']]

    return df_ie
