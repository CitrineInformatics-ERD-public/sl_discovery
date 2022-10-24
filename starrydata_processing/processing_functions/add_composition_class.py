from pymatgen.core import Composition
import pandas as pd
import numpy as np

def _classify_as_111_type(formula):
    
    comp = Composition(formula)
    
    is_metal_or_metalloid = [e.is_metal or e.is_metalloid for e in comp.elements]
    stoich = [float(comp.reduced_composition[e]) for e in comp.elements]
    at_frac = [float(comp.get_atomic_fraction(e)) for e in comp.elements]
    single_ele_majority = [True for af in at_frac if af > 0.4]
    hh_stoich = [True for s in stoich if s<=1.5]

    # ensure there are at least 3 elements and all are metal or metalloid
    if len(comp.elements) >= 3 and False not in is_metal_or_metalloid:       
        if False not in hh_stoich and True not in single_ele_majority:
            if 2.9<=sum(stoich)<=3.1:
                return '111-type'
    
            
def classify_by_composition(input_data_file_path, output_data_file_path, composition_key):
    
    df = pd.read_csv(input_data_file_path)
    print('Adding classifications in {} samples'.format(len(df)))
    df['Composition class'] = df[composition_key].apply(_classify_as_111_type)
    df.to_csv(output_data_file_path, index=False)
