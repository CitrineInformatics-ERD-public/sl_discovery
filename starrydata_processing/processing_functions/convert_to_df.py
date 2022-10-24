import json
import pandas as pd

# Converting as outlined here: https://starrydata.wordpress.com/2018/09/14/use-datafiles-in-python/
def convert_to_df(sample_ids_file_path, data_dir):
    
    print('Converting samples to DF')

    sample_ids = json.load(open(sample_ids_file_path, 'r'))

    rawdata = []
    papers = []
    figures = []
    samples = []
    properties = []

    for i in sample_ids['sampleid']:
        sampledata=json.load(open('{}samples/{}.json'.format(data_dir, i),'r'))
        if sampledata != None:
            rawdata.append(pd.DataFrame(sampledata["rawdata"]))
            papers.append(pd.DataFrame(sampledata["paper"]))
            figures.append(pd.DataFrame(sampledata["figure"]))
            samples.append(pd.DataFrame(sampledata["sample"]))
            properties.append(pd.DataFrame(sampledata["property"]))
    
    print("Samples aggregrated: {}".format(len(rawdata)))
    df_rawdata = pd.concat(rawdata) 
    df_papers = pd.concat(papers) 
    df_figures = pd.concat(figures) 
    df_samples = pd.concat(samples) 
    df_samples['sampleid'] = df_samples['sampleid'].astype(int)
    df_properties = pd.concat(properties)

    df_rawdata.to_csv('{}rawdata.csv.gz'.format(data_dir), index=False, compression='gzip')
    df_papers.drop_duplicates(subset=['paperid']).sort_values('paperid').to_csv('{}papers.csv'.format(data_dir), index=False)
    df_figures.drop_duplicates(subset=['figureid']).sort_values('figureid').to_csv('{}figures.csv'.format(data_dir), index=False)
    df_samples.drop_duplicates(subset=['sampleid']).sort_values('sampleid').to_csv('{}samples.csv'.format(data_dir), index=False)
    df_properties.drop_duplicates(subset=['propertyid']).sort_values('propertyid').to_csv('{}properties.csv'.format(data_dir), index=False)