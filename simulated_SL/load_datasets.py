import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import os


def encode_categoricals(df, categorical_inputs):

    df = pd.get_dummies(df, prefix=categorical_inputs)

    return(df)

def group_and_index_formula(df, categorical_inputs, out):
       
    # take mean value of any duplicate chemical formulas, drop rows where output is 0.
    df['formula'] = df['formula'].apply(lambda x: x.strip())
    
    if categorical_inputs != None:
        df = df.groupby(['formula']+categorical_inputs, as_index=False).mean().reset_index()
#         df = df.set_index(['formula']+categorical_inputs)

    else:
        df = df.groupby('formula', as_index=False).mean().reset_index()
#         df = df.set_index('formula')

    df = df[df[out]!=0]


    return df


def get_matbench_dataset(dataset, out):
    
    # get dataset from matminer
    df = load_dataset(dataset)
    
    # remove dft / zero band gap records
    if dataset=='matbench_expt_gap':
        df = df[df['gap expt']!=0]
    
    df['formula'] = df['composition']
#     df['Composition class'] = df['formula'].apply(classify_by_composition)
    del df['composition']

    
    return df


def get_starrydata(filtered=True, out='8', composition_class=None, sample_form=None, material_family=None):

    if filtered==True:
        input_file_path = os.path.join('..','starrydata_processing','data','rawdata_interpolated_filtered_with_classifiers.csv')
    else:
        input_file_path = os.path.join('..','starrydata_processing','data','rawdata_interpolated.csv')

    df = pd.read_csv(input_file_path)
    df = df.dropna(subset=[out])
    
    if out == '8':
        df = df[df[out]<=2]
        
    if out == '2':
        df[out] = df[out]*10**6
        
    if out == '3':
        df[out] = df[out]*10**-2
        
    if composition_class != None:
        df = df[df['Composition class']==composition_class]
        
    if sample_form != None:
        df = df[df['Form']==sample_form]

    if material_family != None:
        df = df[df['MaterialFamily']==material_family]
        
    # filter on room-temp
    df = df[df['1']==300]

    df['formula'] = df['composition']
    del df['composition']

    return df 



def featurize_composition(df, out):

    feat_dict = {'composition': [
            ElementProperty.from_preset(preset_name="magpie"),
#             BandCenter(),
#             Stoichiometry(),
#             AtomicOrbitals(),
#             TMetalFraction(),
#             ElementFraction(),
#             Miedema(),
#             YangSolidSolution(),
        ]}

    # convert string to composition
    df = StrToComposition().featurize_dataframe(df, "formula", ignore_errors=True)

    for feat_type in feat_dict:
        for f in feat_dict[feat_type]:
            df = f.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    
    # replace inf values and drop rows with nan
#     df = df.replace([np.inf, -np.inf], np.nan)
    features = [x for x in df.keys() if 'Magpie' in x]
    df = df.dropna(subset=features+[out])
    # df['random_feat'] = np.random.rand(len(df),1)
    


    
    return df