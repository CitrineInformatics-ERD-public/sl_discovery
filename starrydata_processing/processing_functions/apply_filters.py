import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, periodic_table

def _get_prop_id(prop_name):

    df_props = pd.read_csv('data/properties.csv')
    prop_id = str(df_props[df_props['propertyname']==prop_name]['propertyid'].values[0])
    
    return str(prop_id)


def _filter_prop(df, prop_name, prop_filters):
    
    print('INPUT DF: {}'.format(len(df)))

    if prop_name != 'sigma_E_0':
        prop_id = _get_prop_id(prop_name)
    else:
        prop_id = prop_name
    df = df[(df[prop_id] > prop_filters[prop_name]['min']) & (df[prop_id] < prop_filters[prop_name]['max']) | (np.isnan(df[prop_id]))]
    print('Filtered {} DF: {}'.format(prop_name, len(df)))

    return df

def _validate_composition(composition):
    
    try:   
        comp = Composition(composition).reduced_composition
    except Exception as e:
        return None
    
    dummy_species = [type(e) == periodic_table.DummySpecie for e in comp.elements]

    # dummy species result from mislabeled STARRY data
    if True in dummy_species: 
        return None
    
    # sometimes pymatgen won't id dummy species and will create invalid comp (e.g. A(La0.15T0.05))
    try:   
        comp = Composition(str(comp.reduced_composition))
    except Exception as e:
        return None
    
    return comp


def apply_filters(calc_data_file_path, filtered_data_file_path, prop_filters_file_path):
    
    prop_filters = json.load(open(prop_filters_file_path, 'r'))
    print(prop_filters)

    df = pd.read_csv(calc_data_file_path)

    # filter rows that don't contain a valid formula
    print('Total samples: {}'.format(len(df.groupby('sampleid'))))
    df['reduced_composition'] = df['composition'].apply(_validate_composition)
    df = df.dropna(subset=['reduced_composition'])

    print('Samples with valid composition: {}'.format(len(df.groupby('sampleid'))))

    print('Applying filters...')
    df = _filter_prop(df, 'Temperature', prop_filters)
    df = _filter_prop(df, 'Seebeck coefficient', prop_filters)
    df = _filter_prop(df, 'Electrical conductivity', prop_filters)
    df = _filter_prop(df, 'Thermal conductivity', prop_filters)
    df = _filter_prop(df, 'Power factor', prop_filters)
    df = _filter_prop(df, 'ZT', prop_filters)
    df = _filter_prop(df, 'sigma_E_0', prop_filters)

    print('Filtered DF: {}'.format(len(df)))
    df.to_csv(filtered_data_file_path, index=False)