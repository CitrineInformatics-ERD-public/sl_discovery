import numpy as np
import pandas as pd
from scipy import interpolate
import os

def _get_prop_id(prop_name):

    df_props = pd.read_csv(os.path.join('data','properties.csv'))
    prop_id = df_props[df_props['propertyname']==prop_name]['propertyid'].values[0]

    return prop_id

def _drop_samples_out_of_temp_range(df, temp, range):
    
    rows = []
    for g in df.groupby('sampleid'):
        
        if len(g[1]['x']) > 1:

            # calc the abs diff between temp  
            temp_delta = abs(temp-min(g[1]['x'], key=lambda x:abs(x-temp)))
        
            if temp_delta < range:
                rows.append(g[1])
        elif temp-range <= g[1]['x'].values[0] <= temp+range:
            rows.append(g[1])

    df_trim = pd.concat(rows)

    return df_trim

def parse_rawdata(data_dir):

    rawdata = pd.read_csv(os.path.join(data_dir,'rawdata.csv.gz'))
    samples = pd.read_csv(os.path.join(data_dir,'samples.csv'))
    papers = pd.read_csv(os.path.join(data_dir,'papers.csv'))

    # merge data into single DF
    df = pd.merge(rawdata, samples, how='left', on=['sampleid', 'paperid'])
    df = pd.merge(df, papers, how='left', on='paperid')
    
    print('Merging into single DF: '.format(df.keys()))

    # filter on desired properties and valid temperature range
    desired_props = ["Temperature", "Seebeck coefficient", "Electrical conductivity", "Thermal conductivity", 
    "Electrical resistivity", "Power factor", "ZT", "Lattice thermal conductivity", "Carrier mobility", 
    "Carrier concentration", "Hall coefficient"]

    desired_prop_ids = [_get_prop_id(prop_name) for prop_name in desired_props]
    print(desired_prop_ids)
    # desired_props = [2, 3, 4, 5, 6, 8, 11, 12, 13, 14]
    df = df[df['propertyid_x']==1]
    df = df[df['propertyid_y'].isin(desired_prop_ids)]

    # define temps for interpolation
    temps = [300]

    # define column headers to add to new DF
    column_headers = ['sampleid', 'composition', 'samplename', 'sampleinfo', 'propertyid_y', 'doi', 'paperid', 'year']
    columns = column_headers + ['int_value', '1']

    # trim dataset to temp range of interest
    df = _drop_samples_out_of_temp_range(df, 300, 25)
    # df = df[(df['x']>200) & (df['x']<1200)]


    # perform interpolation
    int_dfs = []
    for g in df.groupby(['sampleid', 'propertyid_y']):
        
        df_prop = g[1]
        df_prop = df_prop.sort_values(by=['x'])

        # int_vals = np.interp(temps, df_prop['x'], df_prop['y'])
        if len(df_prop['x'])>1:
            int_vals = interpolate.interp1d(df_prop['x'], df_prop['y'], fill_value='extrapolate')(temps)
            rows = []
            for index, i in enumerate(temps):
                row = [df_prop[c].iloc[0] for c in column_headers] + [int_vals[index], i]
                rows.append(row)
            df_temp = pd.DataFrame(rows, columns=columns)
            int_dfs.append(df_temp)

    interpolated_df = pd.concat(int_dfs)

    print('Interpolating data at: {}'.format(temps))

    # reformat interpolated_df
    column_headers = ['sampleid', 'composition', 'samplename', 'sampleinfo', '1', 'doi', 'paperid', 'year']

    pivots = []
    for g in interpolated_df.groupby(['sampleid', '1']):
        pivot = pd.pivot_table(g[1], columns='propertyid_y', values='int_value', index=column_headers)
        pivots.append(pivot)

    formatted_df = pd.concat(pivots)
    formatted_df.to_csv(os.path.join(data_dir,'rawdata_interpolated.csv'))