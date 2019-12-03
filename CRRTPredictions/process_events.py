#!/bin/python

import pprint
import datetime
import subprocess

import pandas as pd


def time_restriction(
        df_restrict, restriction_var,
        csv_file, charting_variable, outfile,
        start_interval, rename_dict):

    # The idea is to merge the df_restrict with df_investigations and
    # decide which values to keep from provided times
    # start_interval is measured in relation to the restriction variable

    if not start_interval:
        print('Performing no time restriction')
        outfile = outfile + f'_ALL.csv'
    else:
        td_start = datetime.timedelta(days=start_interval)
        outfile = outfile + f'_{start_interval}.csv'  # Rename outfile for the sake of clarity

    if rename_dict:
        df_itemids = pd.DataFrame.from_dict(rename_dict).T
        df_itemids = df_itemids.reset_index().rename({'index': 'NEWID'}, axis=1)
        df_itemids = df_itemids.melt(id_vars=['NEWID']).drop('variable', axis=1)
        df_itemids.columns = ['NEWID', 'ITEMID']
        df_itemids = df_itemids.set_index('ITEMID')
        consolidated_item_ids = df_itemids.index.to_list()

        print('Consolidating as:')
        pprint.pprint(df_itemids.sort_values('ITEMID'))

    cols = ['HADM_ID', 'ITEMID', charting_variable, 'VALUE']
    cols_text = ','.join(cols)

    all_hadms = df_restrict.HADM_ID.to_list()

    # Not enough RAM so we're going old school
    # Iterate through the csv file in chunks
    print('Processing', outfile)
    chunksize = 10 ** 5
    for df_chunk in pd.read_csv(csv_file, chunksize=chunksize):

        # Restrict columns
        df_chunk = df_chunk[cols]

        # Set correct dtypes
        s_hadms = pd.to_numeric(
            df_chunk.HADM_ID, errors='coerce', downcast='integer')
        df_chunk.assign(HADM_ID=s_hadms)

        s_items = pd.to_numeric(
            df_chunk.ITEMID, errors='coerce', downcast='integer')
        df_chunk.assign(ITEMID=s_items)

        df_chunk[charting_variable] = pd.to_datetime(
            df_chunk[charting_variable])

        # Remove nulls
        df_chunk = df_chunk.dropna()

        # Filter before proceeding further
        if rename_dict:  # CHARTEVENTS
            df_chunk = df_chunk.query(
                'HADM_ID in @all_hadms and ITEMID in @consolidated_item_ids')
            for this_item_id in df_itemids.index:
                df_chunk = df_chunk.replace(
                    this_item_id,
                    df_itemids.loc[this_item_id].NEWID)

        else:  # All other tables
            df_chunk = df_chunk.query('HADM_ID in @all_hadms')

        df_chunk = df_chunk.sort_values(['HADM_ID', 'ITEMID', charting_variable])

        # Temporal restriction is performed only if start interval is specified
        if start_interval:
            # Merge with restriction dataframe
            df_chunk = df_chunk.merge(
                df_restrict[['HADM_ID', restriction_var]],
                how='left')

            # Restrict dataframe so that it fits temporal conditions
            df_chunk = df_chunk[
                td_start > (df_chunk[restriction_var] - df_chunk[charting_variable])]
            df_chunk.drop(restriction_var, axis=1, inplace=True)

        # Post process
        # Set dtypes again for some godforsaken reason
        df_chunk['HADM_ID'] = df_chunk['HADM_ID'].astype('int')

        # Save to csv
        df_chunk.to_csv(outfile, mode='a', header=None, index=False)

    # Write headers to file so each iteration of loading it doesn't
    # make me want to tear my fingernails out
    subprocess.run(f'sed -i 1i{cols_text} {outfile}', shell=True)
    print('Done processing', outfile)
