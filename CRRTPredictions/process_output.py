#!/bin/python

import sqlite3
import datetime

import pandas as pd

import data_references


database = sqlite3.connect('../MIMIC.db')


def urine_output(df_restrict, restriction_var, start_interval):
    all_hadms = df_restrict.HADM_ID.to_list()
    hadm_string = ','.join([str(i) for i in all_hadms])

    list_uo = data_references.dict_uo['UrineOutput']
    uo_string = ','.join([str(i) for i in list_uo])

    df_uo = pd.read_sql_query(
        f'SELECT HADM_ID, ITEMID, CHARTTIME, VALUE FROM OUTPUTEVENTS WHERE HADM_ID IN ({hadm_string}) AND ITEMID IN ({uo_string})',
        database)

    df_uo['HADM_ID'] = df_uo['HADM_ID'].astype('int')
    df_uo['CHARTTIME'] = pd.to_datetime(df_uo['CHARTTIME'], errors='coerce')
    df_uo = df_uo.sort_values(['HADM_ID', 'CHARTTIME'])

    # Merge and restrict values according to restriction time
    df_uo = df_uo.merge(
        df_restrict[['HADM_ID', restriction_var]], on='HADM_ID', how='left')

    t_delta = datetime.timedelta(days=start_interval)
    df_uo = df_uo[t_delta > (df_uo[restriction_var] - df_uo['CHARTTIME'])]

    # Get cumulative UO
    # df_uo['CUMULATIVEUO'] = df_uo.groupby('HADM_ID').cumsum()['VALUE']

    # Housekeeping
    df_uo = df_uo.drop(['STARTTIME'], axis=1)

    return df_uo
