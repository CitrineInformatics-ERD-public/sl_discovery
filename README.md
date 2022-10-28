# Quantifying the performance of machine learning models in materials discovery [code repository]

This repository contain data and processing scripts to reproduce work performed in the article: Quantifying machine learning model performance in materials discovery, Borg et al., arXiv.2210.13587 (2022). DOI: [10.48550/arXiv.2210.13587](https://doi.org/10.48550/arXiv.2210.13587). 

## Simulated Sequential Learning (SL) Quickstart

1. install required packages (example below using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

```
conda create -n [ENV_NAME] pip numpy
conda activate [ENV_NAME]
pip install -r requirements.txt
```

2. Setup configuration files

- To perform a simulated SL run, create an SL configuration file (e.g. [test.yaml](https://github.com/CitrineInformatics/sl_discovery/blob/main/simulated_SL/configuration_files/sl_configs/test.yaml)) and a dataset configuration file (e.g. [matbench_expt_gap_test.yaml](https://github.com/CitrineInformatics/sl_discovery/blob/main/simulated_SL/configuration_files/dataset_configs/matbench_expt_gap.yaml).) These files define the parameters for parsing a dataset and configuring the SL run.
- This repo is currently set up for creating datasets from [Matbench](https://matbench.materialsproject.org/) and [Starrydata2](https://www.starrydata2.org/) to address design challenges that connect chemical compositions (i.e. chemcial formula) to a real-valued physical property. 
    - Matbench: The latest matbench dataset will be queried and returned
    - Starrydata2: Uses data queried August 2021. Processing defined in [Starrydata processing](https://github.com/CitrineInformatics/sl_discovery/tree/main/starrydata_processing).

3. Run [1-execute_sl_workflow.ipynb](https://github.com/CitrineInformatics/sl_discovery/blob/main/simulated_SL/1-execute_sl_workflow.ipynb). Path for the configuration file(s) can be set in cell 2.

4. Run [2-quickplot.ipynb](https://github.com/CitrineInformatics/sl_discovery/blob/main/simulated_SL/2-quickplot.ipynb). Quickplot takes a single SL run as input (i.e. for one target range) and generates a figure with 6 subplots:
- (a) Discovery yield as a function of iteration
- (b) Model error as a function of iteration
- (c) Discovery probability as a function of iteration
- (d-f) Discovery accleration factor for n = 1, 3, and 5 target materials.

5. Scripts to generate figures shown in the manuscript are stored in [figures](https://github.com/CitrineInformatics/sl_discovery/tree/main/figures).

&nbsp;
&nbsp;

Configuration file parameters:

- Dataset parameters:
    - dataset (str): The name of the input dataset to be processed (processing steps defined in load_datasets.py)
    - output (str): output property (must be column in dataset)
    - categoricals (str, null): Categorical features


&nbsp;

- Starrydata specific parameters:
    - comp_class (str, null): Selects a subset of records based on composition (e.g. '111-type') using logic we have predefined [here](https://github.com/CitrineInformatics/sl_discovery/blob/main/starrydata_processing/processing_functions/add_composition_class.py).
    - material_family (str, null):  Starrydata generated label for material family.
    - filtered (True/False): Performs filtering of starrydata datasets based on physically-relevant property values (e.g. filters on records where ZT < 2). 
    - sample_form (str, null): Performs filtering of starrydata based on sample form (e.g. 'bulk').


&nbsp;

- SL parameters:
    - n_sample (int): Number of datapoints to sample / downselect from raw data. Set to 0 to use full dataset.
    - n_training (int): Number of training rows to start SL process.
    - iterations (int): Number of SL iterations to perform.
    - trials (int): Number of trials (i.e. independent SL processes) to perform.
    - batch (int): Number of candidates to select at each SL iteration.
    - discovery_break_number (int): Number of candidates to find before halting SL process. If set to 0, SL will continue for n_iterations. 
    - poi (str, null): Point of interest. Index of point to be included in training set. Forces training set to include "point of interest". Typically set to `null`. 
    - holdout_fraction (float): Percent of dataset to holdout (test). 
    - targets (list): Min and max of the target range, e.g. [90, 100] will target 10th decile materials.
