import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"

acq_functions = ['EV', 'EI', 'MU', "Random"]

def get_color(af):
    
    '''
    define trace color by af for Dy and Dp figures
    
    Args:
        af (str): acquisition function (EV, EI, MU, Random)
    Returns:
        color_hex (str): trace color hex
        color_rgba (str): trace color rgba (allows for setting transparancy)
    '''
    
    # if af == 'EV':
    #     color_hex = '#003f5C'
    #     color_rgba = 'rgba(0, 63, 92, 0.1)'
    # if af == 'EI':
    #     color_hex = '#7a5195'
    #     color_rgba = 'rgba(122, 81, 149, 0.1)'
    # if af == 'MU':
    #     color_hex = '#ef5675'
    #     color_rgba = 'rgba(239, 86, 117, 0.1)'
    # if af == 'Random':
    #     color_hex = '#ffa600'
    #     color_rgba = 'rgba(255, 166, 0, 0.1)'
        
    if af == 'EV':
        color_hex = '#5592f3'
        color_rgba = 'rgba(85, 146, 243, 0.1)'
    if af == 'EI':
        color_hex = '#7a5195'
        color_rgba = 'rgba(122, 81, 149, 0.1)'
    if af == 'MU':
        color_hex = '#ef5675'
        color_rgba = 'rgba(239, 86, 117, 0.1)'
    if af == 'Random':
        color_hex = '#ffa600'
        color_rgba = 'rgba(255, 166, 0, 0.1)'
        
    return color_hex, color_rgba


def get_bionomial_std(probability_array):
    '''
    standard deviation of a Bernoulli trial 
    '''
    return [np.sqrt(p*(1-p)) for p in probability_array]


def get_total_targets(df, target_range=[0,10]):
    
    '''
    Gets total targets based on target range
    Args:
        df (pd.DataFrame): sl_workflow dataframe
        target_range (list): Min and max of target range. 10\% range used by default.
    Returns:
        total_targets (int): total targets within range
    '''
    
    # get total datapoints in dataset
    grouped = df.groupby('iteration')
    total_candidates = [g[1]['n_candidates'].values[0] for g in grouped if g[0]==0][0] + [g[1]['n_training'].values[0] for g in grouped if g[0]==0][0]
    
    percent_of_dataset = 0.01*(target_range[1]-target_range[0])
    total_targets = int(total_candidates*percent_of_dataset)
    
    return total_targets


def get_avg_and_stdev(df, af, prop):
    
    '''
    Get avg and stdev of all trials for prop of interest.
    Args:
        df (pd.DataFrame): sl_workflow dataframe
        af (str): acquisition function (EV, EI, MU, Random)
        prop (str): prop / column in dataframe
        
    Returns:
        avg (np.array): average of prop for all trials in df
        std (np.array): stdev of prop for all trials in df
    '''
    
    grouped = df.groupby(['acquisition_function', 'iteration'])

    avg = [np.average(g[1][prop]) for g in grouped if g[0][0] == af]
    std = [np.std(g[1][prop]) for g in grouped if g[0][0] == af]
    
    # slice array if number of targets is less than the number of iterations
    total_targets = get_total_targets(df)
    
    if total_targets < len(avg):
        avg = avg[:int(total_targets)]
        std = std[:total_targets]
        print('arrary sliced: iterations > # of targets', len(avg), total_targets)

    return np.array(avg), np.array(std)


def get_discovery_yield(df, af):
    
    '''
    Calculates discovery yield (fraction of total targets found)
    Args:
        df (pd.DataFrame): sl_workflow dataframe
        af (str): acquisition function (EV, EI, MU, Random)
    Returns:
        total_targets (int): total targets within range
    '''
    
    total_targets = get_total_targets(df)
    
    n_targets_found_avg, n_targets_found_std = get_avg_and_stdev(df, af, 'n_targets_found')

    # normalize by total number of targets
    norm_avg_n_top_found = np.array(n_targets_found_avg) / total_targets
    norm_avg_n_top_found_std = np.array(n_targets_found_std)  / total_targets
    
    return norm_avg_n_top_found, norm_avg_n_top_found_std
    
    
def get_avg_and_std_trace(fp, prop, legend=True):
    
    '''
    gets plotly trace for avg and std curves for property of interest.
    e.g. 
    Args:
        fp = list of file paths that contain sl_workflow dataframes
        prop (str): prop / column in dataframe
        legend (boolean): Enables ability to toggle legend on/off
    
    Returns:
        traces = avg_trace, error_upper_bounds, error_lower_bounds
    '''

    df = pd.read_csv(fp)
    traces = []

    for af in acq_functions:
        
        # to get DY we normalize fraction of targets found by total targets
        if prop == 'fraction_of_targets_found':
            avg, std = get_discovery_yield(df, af)
        else:
            avg, std = get_avg_and_stdev(df, af, prop)
            
        x = list(range(len(avg)))
        marker_color, fill_color = get_color(af)

        trace = go.Scatter(y=avg, x=x, mode='lines', name=af, line=dict(color=marker_color),
#                                 error_y=dict(type='data', array=std, visible=True, thickness=0.5),
                           legendgroup=af,
                           showlegend=legend,
                                )

        e_upper = go.Scatter(
                    name='Upper Bound',
                    x=x,
                    y=avg+std,
                    mode='lines',
                    marker=dict(color=marker_color, opacity=0.5),
                    line=dict(width=0),
                   legendgroup=af,
                    showlegend=False
                )
        
        e_lower = go.Scatter(
                    name='Lower Bound',
                    x=x,
                    y=avg-std,
                    marker=dict(color=marker_color),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=fill_color,
                    fill='tonexty',
                   legendgroup=af,
                    showlegend=False
        )
        traces.extend([trace, e_upper, e_lower])

    return traces

  
def get_discovery_probability(df, af):
    
    # group by iteration and acquisition function
    grouped = df.groupby(['iteration', 'acquisition_function'])
    
    # target_found = binary (yes/no) was the selected candidate in the target window.
    ## for every iteration what were the average number of targets found
    avg = [g[1]['target_found'].mean() for g in grouped if g[0][1] == af]
    
    #first point not an SL run
    avg = avg[1:]
    n_trials = len(df.groupby(['trial']))
    std = [np.sqrt((p*(1-p))/n_trials) for p in avg]
    
#     dp_std = [g[1]['target_found'].std() for g in grouped if g[0][1] == af]

    # slice array if number of targets is less than the number of iterations
    total_targets = get_total_targets(df)
    
    if total_targets < len(avg):
        avg = avg[:int(total_targets)]
        std = std[:total_targets]
        print('arrary sliced: iterations > # of targets', len(avg), total_targets)

    return np.array(avg), np.array(std)
    
    
def get_discovery_probability_traces(fps, x_metric='NDME', legend=True):
    
    '''
            gets plotly trace for dp 
        Args:
            fp: list of file paths that contain sl_workflow dataframes
            x_metric: NDME or RMSE
            y_metric: discovery_prob
            legend (boolean): Enables ability to toggle legend on/off
        Returns:
            traces = avg_trace, error_upper_bounds, error_lower_bounds
    '''
        
    traces = []
    for fp in fps:
        
        print('File: ', fp)
        df = pd.read_csv(fp)
        
        
        for af in acq_functions:
            
            # x can be either NDME or RMSE
            x, x_std = get_avg_and_stdev(df, af, x_metric)
            
            # get disc_prob
            y, y_std = get_discovery_probability(df, af)
            marker_size = np.linspace(4,16,len(x))
                
            marker_color, fill_color = get_color(af)
            
            
            e_upper = go.Scatter(
                    name='Upper Bound',
                    x=x,
                    y=y+y_std,
                    mode='lines',
                    marker=dict(color=marker_color),
                    line=dict(width=0),
                   legendgroup=af,
                    showlegend=False
                )
        
            e_lower = go.Scatter(
                        name='Lower Bound',
                        x=x,
                        y=y-y_std,
                        marker=dict(color=marker_color),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor=fill_color,
                        fill='tonexty',
                       legendgroup=af,
                        showlegend=False
            )

            trace = go.Scatter(x=x, y=y, mode='markers', name=af, 
                               marker=dict(size=marker_size, color='white', line=dict(width=1, color=marker_color)),
                               showlegend=legend,
                              legendgroup=af)
            
            traces.extend([trace, e_upper, e_lower])                    
    
    return traces


## heatmap functions

def calculate_DAF(df, n_targets=1):
    '''
    Calculates and returns Discovery Acceleration Factor (the avg number of SL iterations required to id N compounds in target range)
    Args:
        df (pd.DataFrame): sl_workflow dataframe
        n_targets (int):  adjustable parameter for number of targets researcher wants to find
    Return:
        itt_avg_dict (dict): Dict with avg of n_targets_found
    '''
    
    DAF = {'EV':[], 'EI':[], 'MU':[], 'Random':[]}

    for g in df.groupby(['acquisition_function', 'trial']):

        # af = acquisition function
        af = g[0][0]

        # if desired targets were found
        if n_targets in g[1]['n_targets_found'].values:

            # get the first iteration where n_targets_found == desired number of targets
            iterations_to_n_targets = g[1][g[1]['n_targets_found']==n_targets]['iteration'].values[0]
            # append to dict
            DAF[af].append(iterations_to_n_targets)

    
    # compute average and stdev
    DAF_avg = {key: (round(np.average(val), 1), round(np.std(val), 1)) for key, val in DAF.items()}
    print(DAF_avg)
    # normalize by random search (for a given decile, it should take 10 iterations on average to id a single target)
    n_iter_random = int(n_targets*10)
    
    # to calc norm_std we take the percent error (std/avg) and multiple it by the normalized avg val
    DAF_norm = {}
    for key, val in DAF_avg.items():
        avg = val[0]
        std = val[1]
        norm_avg = n_iter_random/avg
        percent_error = std/avg
        norm_std = norm_avg*percent_error
        DAF_norm[key] = (round(norm_avg,1), round(norm_std,1))

    return DAF_norm


def get_DAF_heatmap_traces(input_file_paths, n_targets=1):
    '''
    Gets plotlty trace for discovery acceleration heatmaps
    Args:
        input_file_paths (str): file path to sl_workflow dataframe
        n_targets (int): adjustable parameter for number of targets researcher wants to find
    Return:
        x (list): acquisition functions
        y (list): target ranges
        z (list): heatmap traces
    '''
    
    z = []  # heatmap color / iterations to target / acceleration factor
    x = ['EV', 'EI', 'MU']  # acquisition functions
    y = []  # target ranges
    
    ev_vals = []
    ei_vals = []
    mu_vals = []
    
    for fp in input_file_paths:
        df = pd.read_csv(fp)

        # calculate normalized DAF
        daf_norm = calculate_DAF(df, n_targets=n_targets)
        
        # reformat for plot
        ev_vals.append(daf_norm['EV'][0])
        ei_vals.append(daf_norm['EI'][0])
        mu_vals.append(daf_norm['MU'][0])

        # target range noted in file name
        target_range = fp.split('-')[-2]+'-'+fp.split('-')[-1].split('.')[0]
        y.append(target_range)
        print(target_range, daf_norm)
        
    z = [ev_vals, ei_vals, mu_vals]
    
    return x,y,z