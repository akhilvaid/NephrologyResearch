#!/bin/python

import os
import sqlite3
import datetime

import pandas as pd


database = sqlite3.connect('MIMIC.db')

# Cirrhosis patients ====================================================================
# Only the Hospital admission IDs are relevant
# ICD_9 codes are: 5712, 5715, 5716
cirrhosis_hadm_ids = database.execute(
    'SELECT HADM_ID FROM DIAGNOSES_ICD WHERE ICD9_CODE IN (5712, 5715, 5716)').fetchall()
cirrhosis_hadm_ids = [str(float(i[0])) for i in cirrhosis_hadm_ids]
cirrhosis_string = ','.join(cirrhosis_hadm_ids)

# Process creatinine first ==============================================================
# Serum creatinine ITEMID is 50912
# Urine Output is 51108
print('Processing creatinine levels')
df_lab = pd.read_sql_query(
    f'SELECT HADM_ID, ITEMID, CHARTTIME, VALUE FROM LABEVENTS WHERE ITEMID = 50912 AND HADM_ID IN ({cirrhosis_string})',
    database)
df_lab['HADM_ID'] = pd.to_numeric(
    df_lab['HADM_ID'], downcast='integer', errors='coerce')
df_lab['CHARTTIME'] = pd.to_datetime(df_lab['CHARTTIME'])
df_lab['VALUE'] = pd.to_numeric(df_lab['VALUE'], errors='coerce')
df_lab = df_lab.sort_values(['HADM_ID', 'CHARTTIME'])

# First measurement is considered baseline
# All groups will be iterated on and ones that fullfill the above criteria
# will have their HADM_IDs added to the tracking set

def check_non_cumulative(df, check_interval_hours, check_type):
    df = df.dropna()
    total_records = len(df)
    for i in range(total_records):
        this_record = df.iloc[i]
        for j in range(i, total_records):
            next_record = df.iloc[j]
            time_delta = next_record.CHARTTIME - this_record.CHARTTIME
            time_delta_hours = time_delta.total_seconds() / 3600

            if time_delta_hours >= check_interval_hours:
                if check_type == 'difference':
                    value_delta = next_record.VALUE - this_record.VALUE
                    if value_delta >= 0.3:
                        return True, next_record.CHARTTIME

                if check_type == 'quotient':
                    value_delta = next_record.VALUE / this_record.VALUE
                    if value_delta >= 1.5:
                        return True, next_record.CHARTTIME
                break

    return False, None


df_creatinine = df_lab.query('ITEMID == 50912')
df_c_group = df_creatinine.groupby('HADM_ID')

HADM_IDS = []
for this_group in df_c_group:
    this_df = this_group[1]

    condition1 = check_non_cumulative(this_df, 48, 'difference')
    condition2 = check_non_cumulative(this_df, 24 * 7, 'quotient')

    # Whichever condition was fulfilled first takes precedence in final processing
    # Just append both for now
    if condition1[0]:
        HADM_IDS.append((this_group[0], condition1[1]))

    if condition2[0]:
        HADM_IDS.append((this_group[0], condition2[1]))

# Process Urine output next =============================================================
# The weight will have to be added as well - this will come from CHARTEVENTS
# The UO will come from the OUTPUTEVENTS table

itemids_uo = [
    40055, 43175, 40069, 40094, 40715, 40473, 40085,
    40057, 40056, 40405, 40428, 40086, 40096, 40651,
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227488, 227489]
itemid_string = ','.join([str(i) for i in itemids_uo])
df_uo = pd.read_sql_query(
    f'SELECT HADM_ID, ITEMID, CHARTTIME, VALUE, VALUEUOM FROM OUTPUTEVENTS WHERE HADM_ID IN ({cirrhosis_string}) AND ITEMID IN ({itemid_string})',
    database)
df_uo['CHARTTIME'] = pd.to_datetime(df_uo['CHARTTIME']) 

# Get weight records from generated csv file
# Create it if it doesn't exist
# Daily weight = 763, 224639
if not os.path.exists('data_WeightRecords.csv'):
    print('Creating weight records table')
    ce_csv = 'CSV/CHARTEVENTS.csv'
    weight_items = (763, 224639)
    chunksize = 10 ** 5

    for this_chunk in pd.read_csv(ce_csv, chunksize=chunksize):
        df_chunk = this_chunk[['HADM_ID', 'ITEMID', 'VALUE']]
        df_chunk['ITEMID'] = pd.to_numeric(
            df_chunk['ITEMID'], downcast='integer', errors='coerce')
        df_chunk = df_chunk.query('ITEMID in @weight_items')
        df_chunk.to_csv('WeightRecords.csv', mode='a', header=None, index=False)

    print('Done creating weight records')

df_weight = pd.read_csv('data_WeightRecords.csv', header=None)
df_weight.columns = ['HADM_ID', 'ITEMID', 'VALUE']
avg_weights = df_weight.groupby('HADM_ID').VALUE.mean().reset_index()

# Merge average weight with uo dataframe
df_uo = df_uo.merge(avg_weights.reset_index(), how='inner', on='HADM_ID')
df_uo = df_uo.rename({'VALUE_x': 'VALUE', 'VALUE_y': 'WEIGHT'}, axis=1)
df_uo['HADM_ID'] = df_uo.HADM_ID.astype('int')

# This is a little complicated and VERY slow
# Iterate over the groupby object
# Get every 6 hour or greater interval by record
# Divide by average weight during that hospital stay

def check_cumulative(df):
    df = df.dropna()
    total_records = len(df)
    df['cmsm'] = df.VALUE.cumsum()
    for i in range(total_records):
        this_record = df.iloc[i]
        for j in range(i, total_records):
            next_record = df.iloc[j]
            time_delta = next_record.CHARTTIME - this_record.CHARTTIME
            time_delta_hours = time_delta.total_seconds() / 3600
            if time_delta_hours >= 6:
                urine_production = next_record.cmsm - this_record.cmsm
                urine_production_ml_per_kg_per_hour = (
                    (urine_production / this_record.WEIGHT) / time_delta_hours)

                if 0 < urine_production_ml_per_kg_per_hour < 0.5:
                    return True, next_record.CHARTTIME
                break

    return False, None


print('Processing UO')
df_uo_g = df_uo.groupby('HADM_ID')
hadm_ids_with_aki_uo = []
for count, this_group in enumerate(df_uo_g):
    this_df = this_group[1]
    this_df = this_df.sort_values(['HADM_ID', 'CHARTTIME'])

    condition3 = check_cumulative(this_df)
    if condition3[0]:
        HADM_IDS.append((this_group[0], condition3[1]))

# Final processing ======================================================================
# Filter all hospital admissions to the first time any of the criteria was met
df_hadms = pd.DataFrame(HADM_IDS, columns=['HADM_ID', 'CHARTTIME'])
df_hadms = df_hadms.sort_values(['HADM_ID', 'CHARTTIME'])
df_hadms = df_hadms.groupby('HADM_ID').first().reset_index()

# Add time of death to each of the hospital admissions, as well as a column detailing
# the interval from the initial time of diagnosis to time of death
# This will require fetchin subject ids first
all_hadms = df_hadms.HADM_ID.to_list()
hadm_string = ','.join([str(i) for i in all_hadms])

df_subject_id = pd.read_sql_query(
    f'SELECT SUBJECT_ID, HADM_ID FROM ADMISSIONS WHERE HADM_ID IN ({hadm_string})', database)
df_hadms = df_hadms.merge(df_subject_id, on='HADM_ID', how='inner')

all_subjects = df_hadms.SUBJECT_ID.to_list()
sid_string = ','.join([str(i) for i in all_subjects])

df_death = pd.read_sql_query(
    f'SELECT SUBJECT_ID, DOD FROM PATIENTS WHERE SUBJECT_ID IN ({sid_string})', database)
df_death['DOD'] = pd.to_datetime(df_death['DOD'], errors='coerce')

df_hadms = df_hadms.merge(df_death, on='SUBJECT_ID', how='outer')
df_hadms['DEATH_INTERVAL'] = df_hadms.apply(
    lambda x: x['DOD'] - x['CHARTTIME'], axis=1)

# Drop all deaths less than 2 days after diagnosis
t_delta_min = datetime.timedelta(days=2)
df_hadms['DEATH_AFTER_2DAYS'] = df_hadms.apply(
    lambda x: x['DEATH_INTERVAL'] > t_delta_min or pd.isnull(x['DEATH_INTERVAL']), axis=1)
df_hadms = df_hadms[df_hadms.DEATH_AFTER_2DAYS == True]

# Comparison time delta = 28 days
t_delta = datetime.timedelta(days=28)
df_hadms['OUTCOME_DEATH'] = df_hadms.apply(
    lambda x: x['DEATH_INTERVAL'] <= t_delta, axis=1)

# Add dialysis to each hospital admission
# This will be evaluated as a separate outcome
df_dialysis = pd.read_sql_query(
    f'SELECT HADM_ID, ICD9_CODE FROM PROCEDURES_ICD WHERE ICD9_CODE IN (5498, 3995)',
    database)
df_dialysis = df_dialysis.query('HADM_ID in @all_hadms')
df_dialysis = df_dialysis.drop_duplicates()

df_hadms = df_hadms.merge(df_dialysis, on='HADM_ID', how='left')
df_hadms['OUTCOME_DIALYSIS'] = df_hadms.apply(
    lambda x: False if pd.isnull(x['ICD9_CODE']) else True, axis=1)

# This line decides which records to get. First or last admission.
df_hadms = df_hadms[
    ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'DEATH_INTERVAL', 'OUTCOME_DEATH', 'OUTCOME_DIALYSIS']]
df_hadms = df_hadms.rename({'CHARTTIME': 'RESTRICTION_TIME'}, axis=1)
df_hadms = df_hadms.sort_values(
    ['SUBJECT_ID', 'RESTRICTION_TIME']).groupby('SUBJECT_ID').last().reset_index()

df_hadms.to_pickle('data_HospitalAdmissions.pickle')
