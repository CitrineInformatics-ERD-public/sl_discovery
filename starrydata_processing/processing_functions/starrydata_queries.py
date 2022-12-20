import time
import os
import json
import requests

def query_sample_ids(sample_id_file_path, elements_file_path):

    '''
    Queries starry data for list of sample ids in database
    :return: list of sample ids
    '''

    url = "https://www.starrydata2.org/api/sample/search"
    elements = json.load(open(elements_file_path, 'r'))['all']
    ele_string = ",".join(elements) + ',or'
    print("1. Querying for samples with elements: {}".format(ele_string))

    payload = {'atom': ele_string}
    r = requests.get(url, params=payload)
    json.dump(r.json(), open(sample_id_file_path, 'w'))
    print("1. {} downloaded".format(sample_id_file_path))
    return


def _query_sample_data(sample_id):
    sample_url = "https://www.starrydata2.org/api/sample/{}".format(sample_id)
    try:
        sr = requests.get(sample_url)
        return sr.json()
    except:
        return None


def download_samples(sample_id_file_path, data_dir):

    '''
    Queries for samples in sample_id list that have not been previously downloaded.
    :param sample_ids: list of sample ids to download
    :return: N/A
    '''

    sample_ids = json.load(open(sample_id_file_path, 'r'))['sampleid']

    for s_id in sample_ids:
        output_file_path = os.path.join(data_dir,'samples','{}.json'.format(s_id))
        if not os.path.isfile(output_file_path):
            print("DOWNLOADING: ", output_file_path)
            sample_data = _query_sample_data(s_id)
            time.sleep(.500)
            json.dump(sample_data, open(output_file_path, 'w'))