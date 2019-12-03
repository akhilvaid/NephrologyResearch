#!/bin/python

import sqlite3
import pandas as pd

database = sqlite3.connect('../MIMIC.db')


def demographics(df_restrict):
    # All relevant HADM_IDs
    df_hadms = df_restrict[['HADM_ID']]

    # Admissions dataframe - Contains ADMITTIME
    df_admissions = pd.read_sql_query(
        f'SELECT SUBJECT_ID, HADM_ID, ADMITTIME, ETHNICITY FROM ADMISSIONS', database)

    df_hadms = df_hadms.merge(
        df_admissions, on='HADM_ID', how='inner')

    # Patients dataframe - Contains Subject ID, Gender, DOB
    df_patients = pd.read_sql_query(
        f'SELECT SUBJECT_ID, GENDER, DOB FROM PATIENTS', database)

    df_hadms = df_hadms.merge(df_patients, how='inner', on='SUBJECT_ID')

    # Calculate age
    df_hadms[['ADMITTIME', 'DOB']] = df_hadms[['ADMITTIME', 'DOB']].apply(pd.to_datetime)
    s_age = df_hadms.apply(lambda x: (x['ADMITTIME'] - x['DOB']).total_seconds(), axis=1)
    df_hadms = df_hadms.assign(AGE_SECONDS=s_age)

    # Encode gender
    df_hadms['GENDER'] = df_hadms['GENDER'].apply(lambda x: 1 if x == 'M' else 0)

    # Final dataframe - Just needs dummies for ETHNICITY
    # Get 4 largest ethnic groups - everything else goes into OTHER
    df_final = df_hadms[['HADM_ID', 'GENDER', 'AGE_SECONDS', 'ETHNICITY']]
    largest_groups = [
        'WHITE',
        'BLACK/AFRICAN AMERICAN',
        'HISPANIC OR LATINO',
        'HISPANIC/LATINO - PUERTO RICAN']
    df_final['ETHNICITY'] = df_final['ETHNICITY'].apply(
        lambda x: x if x in largest_groups else 'OTHER')
    df_final = df_final.join(pd.get_dummies(df_final[['ETHNICITY']])).drop('ETHNICITY', axis=1)

    return df_final
