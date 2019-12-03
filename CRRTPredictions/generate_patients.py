#!/bin/python
# MUST RUN FIRST - GENERATES PATIENT IDENTIFIERS

# Important times:
# Start of CRRT
# End of CRRT
# Death

"""28-day survival after CRRT initiation
Alive at hospital discharge
RRT free at hospital discharge
Composite of RRTFS: RRT and alive at hospital discharge
AKI persistance: whether serum creatinine at discharge was less than or equal peak creatinine"""

import sqlite3
import datetime

import pandas as pd

database = sqlite3.connect('../MIMIC.db')


# This only does the MV records
# ITEMID for CRRT: 225802
def crrt_timings():
    df_crrt = pd.read_sql_query(
        'SELECT SUBJECT_ID, HADM_ID, STARTTIME, ENDTIME FROM PROCEDUREEVENTS_MV WHERE ITEMID = 225802',
        database)

    # Set dtypes
    df_crrt['HADM_ID'] = df_crrt['HADM_ID'].astype('int')
    df_crrt['STARTTIME'] = pd.to_datetime(df_crrt['STARTTIME'], errors='coerce')
    df_crrt['ENDTIME'] = pd.to_datetime(df_crrt['ENDTIME'], errors='coerce')
    df_crrt = df_crrt.dropna()

    # For each HADM_ID, get the first time the CRRT was started
    # and the last time it was ended
    df_crrt_start = df_crrt[['HADM_ID', 'STARTTIME']]\
        .sort_values(['HADM_ID', 'STARTTIME']).groupby('HADM_ID').first()
    df_crrt_end = df_crrt[['HADM_ID', 'ENDTIME']]\
        .sort_values(['HADM_ID', 'ENDTIME']).groupby('HADM_ID').last()

    df_final = df_crrt_start.join(df_crrt_end).reset_index()
    df_final = df_final.merge(
        df_crrt[['HADM_ID', 'SUBJECT_ID']], on='HADM_ID', how='inner').\
            drop_duplicates().reset_index(drop=True)

    return df_final


def generate_outcomes(df_restrict, restriction_var):
    all_sids = df_restrict.SUBJECT_ID.to_list()
    subject_string = ','.join([str(i) for i in all_sids])

    # NOTE
    # Outcome: Mortality at 28 days
    df_death = pd.read_sql_query(
        f'SELECT SUBJECT_ID, DOD FROM PATIENTS WHERE SUBJECT_ID IN ({subject_string})',
        database)
    df_death['DOD'] = pd.to_datetime(df_death['DOD'], errors='coerce')
    df_death = df_death.dropna()

    # Create joint dataframe and corresponding features
    df_restrict = df_restrict.merge(df_death, how='left')
    df_restrict['DEATH_INTERVAL'] = df_restrict.apply(
        lambda x: x['DOD'] - x[restriction_var], axis=1)

    # Comparison time delta = 28 days
    t_delta = datetime.timedelta(days=28)
    df_restrict['OUTCOME_DEATH'] = df_restrict.apply(
        lambda x: x['DEATH_INTERVAL'] <= t_delta, axis=1)

    df_restrict = df_restrict[
        ['SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME', 'OUTCOME_DEATH']]

    return df_restrict
