import json
import numpy as np
import pandas as pd
from ast import literal_eval

    
def _get_sample_property(sample_info, prop_key):
    
    if sample_info != '{}':
        
        sample_info_dict = literal_eval(sample_info)

        if prop_key in sample_info_dict.keys():
            if sample_info_dict[prop_key] != '':
                if sample_info_dict[prop_key]['category'] != '':
                    prop_val = sample_info_dict[prop_key]['category']
                else:
                    prop_val = 'unknown'
            else:
                prop_val = 'unknown'
        else:
            prop_val = 'unknown'

        return prop_val
    else:
        return 'unknown'
    
    
def parse_sampleinfo(input_file_path, output_file_path):

    df = pd.read_csv(input_file_path)
    
#     props = ['Form', 'FabricationProcess', 'MaterialFamily', 'DataType', 'Purity', 'RelativeDensity', 'GrainSize', 'ThermalMeasurement', 'ElectricalMeasurement']

    props = ['Form', 'FabricationProcess', 'MaterialFamily', 'DataType']
    
    for prop in props:
        df[prop] = df['sampleinfo'].apply(lambda x: _get_sample_property(x, prop))

    df.to_csv(output_file_path, index=False)