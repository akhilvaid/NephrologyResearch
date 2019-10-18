#!/bin/python

import os
import sqlite3

import pandas as pd

from sklearn.preprocessing import power_transform


class Data:
    # SQLite connection
    database = sqlite3.connect('MIMIC.db')

    # Placeholders
    df_ids = None

    # Dictionary corresponding to CHARTEVENT replacements
    # Replacement values should start at 10000000 to avoid collisions
    # The first 5 are main vitals
    # Anything after that is an accessory measurement included for
    # sake of completeness
    # Any measurement may be included here because features are dropped
    # before rows when below completeness threshold.

    # CORRESPONDING ARRAYS MUST ALL BE THE SAME LENGTH - 2 in this case
    dict_replace_itemids = {
        10000000: [676, 223762],  # Temperature in C
        10000001: [442, 227243],  # Blood pressure - systolic
        10000002: [8440, 227242],  # Blood pressure - diastolic
        10000003: [618, 220210],  # Respiratory rate
        10000004: [211, 220045],  # Heart rate
        10000005: [619, 224688]}  # Respiratory rate - set


def process_demographics(outfile):
    print('Processing demographics...')

    subject_ids = Data.df_ids.SUBJECT_ID.to_list()
    hadm_ids = Data.df_ids.HADM_ID.to_list()

    # Store values here
    df_demographics = pd.DataFrame()

    for this_sid, this_hid in zip(subject_ids, hadm_ids):
        sql = f'SELECT GENDER, DOB FROM PATIENTS WHERE SUBJECT_ID = {this_sid}'
        gender, dob = Data.database.execute(sql).fetchone()

        sql_adm = f'SELECT ADMITTIME, ETHNICITY FROM ADMISSIONS WHERE HADM_ID = {this_hid}'
        admittime, ethnicity = Data.database.execute(sql_adm).fetchone()

        df_this_patient = pd.DataFrame([gender, dob, admittime, ethnicity]).T
        df_this_patient.columns = ['GENDER', 'DOB', 'ADMITTIME', 'ETHNICITY']

        # Process this dataframe
        df_this_patient['DOB'] = pd.to_datetime(df_this_patient['DOB'])
        df_this_patient['ADMITTIME'] = pd.to_datetime(df_this_patient['ADMITTIME'])
        df_this_patient['AGE'] = df_this_patient.apply(
            lambda x: (x['ADMITTIME'] - x['DOB']).days / 365, axis=1)
        df_this_patient['GENDER'] = df_this_patient.apply(
            lambda x: 1 if x['GENDER'] == 'M' else 0, axis=1)

        df_demographics = df_demographics.append(
            df_this_patient.drop(['DOB', 'ADMITTIME'], axis=1))

    # Post process
    df_demographics.index = hadm_ids

    df_demographics['AGE'] = power_transform(
        df_demographics.AGE.values.reshape(-1, 1),
        method='yeo-johnson')
    df_demographics = df_demographics.join(
        pd.get_dummies(df_demographics.ETHNICITY), how='inner')
    df_demographics = df_demographics.drop('ETHNICITY', axis=1)
    df_demographics.to_csv(outfile)

    print('Finished processing demographics.')


def process_chartevents(outfile):
    # The chartevents table is really big so it's not part of the
    # MIMIC SQLite file. It's instead being parsed as raw csv
    # and then being exported to the outfile
    print('Processing chartevents...')

    # Data.dict_replace_itemids will be parsed into a dataframe df_itemids
    # df_itemids is supposed to be a dataframe with its
    # index corresponding to the old itemid and,
    # column next to it corresponding to the new one

    hadm_ids = Data.df_ids.HADM_ID.to_list()

    df_itemids = pd.DataFrame.from_dict(Data.dict_replace_itemids).T
    df_itemids = df_itemids.reset_index().rename({'index': 'NEWID'}, axis=1)
    df_itemids = df_itemids.melt(id_vars=['NEWID']).drop('variable', axis=1) 
    df_itemids.columns = ['NEWID', 'ITEMID']
    df_itemids = df_itemids.set_index('ITEMID')

    consolidated_item_ids = df_itemids.index.to_list()

    cols = ['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']

    # Remove the old exported file in case it exists
    if os.path.exists(outfile):
        os.remove(outfile)
    chartevents_file = 'CSV/CHARTEVENTS.csv'

    chunksize = 10 ** 5
    for df_chunk in pd.read_csv(chartevents_file, chunksize=chunksize):
        df_chunk = df_chunk[cols]

        # Convert dtypes just in case
        # CHARTTIME conversion isn't warranted here
        df_chunk['HADM_ID'] = pd.to_numeric(
            df_chunk.HADM_ID, errors='coerce', downcast='integer')
        df_chunk['ITEMID'] = pd.to_numeric(
            df_chunk.ITEMID, errors='coerce', downcast='integer')

        df_chunk = df_chunk.query(
            'HADM_ID in @hadm_ids and ITEMID in @consolidated_item_ids')

        # TODO - See if there's some way to vectorize this
        # Process replacement ITEMIDs within the loop to keep it moving faster
        for this_item_id in df_itemids.index:
            df_chunk = df_chunk.replace(
                this_item_id,
                df_itemids.loc[this_item_id].NEWID)

        df_chunk.to_csv(outfile, mode='a', header=None, index=False)

    print('Finished processing chartevents.')


def process_labevents(outfile):
    print('Processing labevents...')

    hadm_ids_string_unique = ','.join(
        [str(float(i)) for i in Data.df_ids.HADM_ID.to_list()])
    sql = f'SELECT HADM_ID, ITEMID, CHARTTIME, VALUE FROM LABEVENTS WHERE HADM_ID IN ({hadm_ids_string_unique})'
    df_labevents = pd.read_sql_query(sql, Data.database)
    df_labevents.to_csv(outfile, index=False)

    print('Finished processing labevents.')


def main():
    # We'll be looking at unique subject IDs
    # In case of several admissions for one patient,
    # only the first admission will be counted
    # This list is generated from the KDIGO application module
    hadm_ids = pd.read_csv('HospitalAdmissions.csv', header=None) # Includes repeats
    hadm_ids_string = ','.join(
        [str(i) for i in hadm_ids[0].to_list()])

    # This will be further reduced to only the first hospital admission per patient
    all_ids = Data.database.execute(
        f'SELECT SUBJECT_ID, HADM_ID FROM ADMISSIONS WHERE HADM_ID IN ({hadm_ids_string})').fetchall()
    all_ids.sort()

    df_admissions = pd.DataFrame(all_ids, columns=['SUBJECT_ID', 'HADM_ID'])
    df_admissions = df_admissions.sort_values(['SUBJECT_ID', 'HADM_ID'])
    df_admissions_g = df_admissions.groupby('SUBJECT_ID')

    # Store in Data class
    Data.df_ids = df_admissions_g.first().reset_index()

    # Get demographic info
    process_demographics('data_demographics_out.csv')

    # Process CHARTEVENTS table
    process_chartevents('data_chartevents_out.csv')

    # Process LABEVENTS table
    process_labevents('data_labevents_out.csv')

    # The rest of this is intended workflow:

    # For any non demographic table, filter to a presence threshold
    # through feature_functions.feature_extraction
    # Get nearest missing neighbors through
    # feature_functions.create_df_central_tendency >
    # feature_functions.impute_and_get_nearest_neighbors

    # Inner join array tables after training in the LSTM Imputer
    # Send demographic table and joined tables to the AutoEncoder

    Data.database.close()

if __name__ == '__main__':
    main()
