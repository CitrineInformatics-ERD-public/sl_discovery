# Starrydata processing pipeline (as of August 2021)

To create a training dataset from the Starrydata2 database, the following process has been developed.

## Querying and caching sample data
1. query_sample_ids - Pings Starrydata API for new sample ids (for samples that contain any element from ELEMENTS.json) and updates SAMPLE_IDS.json.

2. download_samples - Downloads sample json files to /samples/ dir based on ids in SAMPLE_IDS.json. The user should create an empty /samples/ dir if one does not exist. If a sample json file is already present, download will skip and continue to the next sample in SAMPLE_IDS.json.

3. convert_to_df - Combines sample jsons into dataframes (as speficied here: https://starrydata.wordpress.com/2018/09/14/use-datafiles-in-python/).


## Post-processing of sample data
4. parse_rawdata - Filters rawdata on desired properties and temp range. Interpolates values for props at specified temperatures. The desired properities, temp range, and interpolation temperatures can be edited directly in this file.
- The current temperature range = [200, 1200]
- The current interpolation temperatures = [300]
- The current properties selected = [2, 3, 4, 5, 6, 8, 11, 12, 13, 14] (see property definitions in [properties.csv](data/properties.csv))

5. calculate_props - Calculates properties from extracted data. Calculated values are noted by extracted_NaN column. If prop has a value and that prop is in extracted_NaN, then it was not extracted (i.e. was calculated). For example, if ZT was not reported for a sample but the sample has data for Seebeck, electrical conductivity, and thermal conductivity, the ZT value will be calculated.

6. apply_filters - Applies filters based on props and ranges specified in [PROPERTY_FILTERS.json](processing_functions/PROPERTY_FILTERS.json). This step is intended to reduce potential outliers and errorneous data points. Also computes a reduced_composition for validating extracted compositions.

7. classify_by_composition - Adds "Composition class" column for and labels records with composition-based labels (e.g. "111-type").

8. parse_sampleinfo - Parses sample metadata (e.g Material Family) and adds it to each record.

## Data files:

- samples.csv = All unique samples returned from the current query parameters.
- properties.csv = All unique properties returned from the current query parameters.
- figures.csv = All unique figure metadata returned from the current query parameters.
- papers.csv = All unique references returned from the current query parameters.

- rawdata.csv.gz = The rawdata returned from the current query parameters. This is combined with the above csvs to generate the following:
    - rawdata_interpolated.csv = Dataset after running parse_rawdata().
    - rawdata_interpolated_filtered.csv = Dataset after running calculate_props(). 
    - rawdata_interpolated_filtered.csv = Dataset after running apply_filters().
    - rawdata_interpolated_filtered_with_classifiers.csv = Dataset after running all processing steps.
