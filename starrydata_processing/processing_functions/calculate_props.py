import math
from scipy import constants
import numpy as np
import pandas as pd
from processing_functions.transport_models import model_sigma_E_0, model_quality_factor


def _calc_sigma(row, temp_id, sb_id, ec_id):
    # seebeck = V/K, cond = S/m, temp = K
    if np.isnan(row[sb_id]) or np.isnan(row[ec_id]) or np.isnan(row[temp_id]):
        return np.NaN
    else:
        val = model_sigma_E_0(row[sb_id], row[ec_id], row[temp_id])
        return val

    
def _calc_qf(row, temp_id, sb_id, ec_id, tc_id):
    # seebeck = V/K, cond = S/m, temp = K, tc = W/(m*K)
    if np.isnan(row[sb_id]) or np.isnan(row[ec_id]) or np.isnan(row[temp_id]) or np.isnan(row[tc_id]):
        return np.NaN
    else:
        val = model_quality_factor(row[sb_id], row[ec_id], row[temp_id], row[tc_id])
        return val


def _add_calc_flag(row, props):
    
    calc_props = []
    
    # props = ['2', '3', '4', '5', '6', '8', '11']
    
    for p in props:
        if np.isnan(row[p]):
            calc_props.append(p)
            
    return calc_props
    
def calculate_props(int_data_file_path, calc_data_file_path):
    '''
     - calculates props from extracted values
    '''

    df = pd.read_csv(int_data_file_path)
    print('Calculate properties of interest in {} samples'.format(len(df)))
    
    # check for updated col definitions (we use definitions from Aug 2021)
    df_props = pd.read_csv('data/properties.csv')
    print(df_props.keys())

    temp_id = str(df_props[df_props['propertyname']=='Temperature']['propertyid'].values[0])
    sb_id = str(df_props[df_props['propertyname']=='Seebeck coefficient']['propertyid'].values[0])
    ec_id = str(df_props[df_props['propertyname']=='Electrical conductivity']['propertyid'].values[0])
    tc_id = str(df_props[df_props['propertyname']=='Thermal conductivity']['propertyid'].values[0])
    er_id = str(df_props[df_props['propertyname']=='Electrical resistivity']['propertyid'].values[0])
    pf_id = str(df_props[df_props['propertyname']=='Power factor']['propertyid'].values[0])
    zt_id = str(df_props[df_props['propertyname']=='ZT']['propertyid'].values[0])
    ltc_id = str(df_props[df_props['propertyname']=='Lattice thermal conductivity']['propertyid'].values[0])

    props = [sb_id, ec_id, tc_id, er_id, pf_id, zt_id, ltc_id]
    print(props)

    # df['extracted_NaN'] = df.apply(_add_calc_flag, axis=1)
    df['extracted_NaN'] = df.apply(_add_calc_flag, args=(props,), axis=1)

    
    # EC = 3, ER = 5
    # calculate EC from ER 
    print("EC (extracted): ", len(df[ec_id].dropna()))
    ec = df[er_id] ** -1
    df[ec_id] = df[ec_id].fillna(ec)
    print("EC (extracted+calc): ", len(df[ec_id].dropna()))

    # PF = 6, SB = 2, 
    # calc PF from EC and SB
    print("PF (extracted): ", len(df[pf_id].dropna()))
    pf = df[sb_id]**2 * df[ec_id]
    df[pf_id] = df[pf_id].fillna(pf)
    print("PF (calc+extracted): ", len(df[pf_id].dropna()))

    #  ZT = 8, TC = 4
    # calc ZT from EC, SB, and TC
    print("ZT (extracted): ", len(df[zt_id].dropna()))
    zt = df[pf_id]*df[temp_id]*df[tc_id]**-1
    df[zt_id] = df[zt_id].fillna(zt)
    print("ZT (calc+extracted): ", len(df[zt_id].dropna()))
    
    # calc lattice-TC from SB, 
    # L = (1.5+exp(-|S| / 116*(10^-6))) * 10^-8
    # total TC - Lorentz*EC*Temp
    print("lattice TC: ", len(df[ltc_id].dropna()))
    L = (1.5+np.exp(-np.abs(df[sb_id])/(116*10**(-6))))*10**-8
    lattice_tc = df[tc_id]-L*df[ec_id]*df[temp_id]
    df[ltc_id] = df[ltc_id].fillna(lattice_tc)
    print("lattice TC: ", len(df[ltc_id].dropna()))
    
    
    # calc sigma E0
    df['sigma_E_0'] = df.apply(_calc_sigma, args=(temp_id, sb_id, ec_id), axis=1)
    df['log sigma_E_0'] = df['sigma_E_0'].apply(lambda x: math.log(x, 10) if x > 0 else None)
    print('E0 (calc): ', len(df['sigma_E_0'].dropna()))
    print('log E0 (calc): ', len(df['log sigma_E_0'].dropna()))
    
    # calc weighted mobility
    df['weighted_mobility'] = (df['sigma_E_0']*3*(constants.h)**3) / (8*constants.pi*constants.e*(2*constants.m_e*constants.k*df[temp_id])**(1.5))
    print('weighted_mobility (calc): ', len(df['weighted_mobility'].dropna()))

    # calc QF
    df['quality_factor'] = df.apply(_calc_qf, args=(temp_id, sb_id, ec_id, tc_id), axis=1)
    df['log quality_factor'] = df['quality_factor'].apply(lambda x: math.log(x, 10) if x > 0 else None)
    print('QF (calc): ', len(df['quality_factor'].dropna()))
    print('log QF (calc): ', len(df['log quality_factor'].dropna()))


    df.to_csv(calc_data_file_path, index=False)