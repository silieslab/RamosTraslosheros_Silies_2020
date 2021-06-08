# In principle rf_params can be a class and the functions below could be
# methods of that class
import numpy as np


def set_EI_ratio(rf_params, ei_ratio, is_ON):
    '''Update synapse conductances, TODO include synapse polarity in params.'''
    if is_ON:
        exc_amp = rf_params['Mi1']['synapse']['amplitude']
        # inh_amp = rf_params['Mi9']['synapse']['amplitude']
        for key in ['Mi4', 'Mi9']:
            rf_params[key]['synapse']['amplitude'] = exc_amp / ei_ratio
    else:
        exc_amp = rf_params['Tm1']['synapse']['amplitude']
        # inh_amp = rf_params['CT1']['synapse']['amplitude']
        for key in ['CT1']:
            rf_params[key]['synapse']['amplitude'] = exc_amp / ei_ratio
    return rf_params


def normalize_connectivity(rf_params):
    for key in rf_params.keys():
        rf_params[key]['connectivity'] =\
            rf_params[key]['connectivity'].astype(float)
        rf_params[key]['connectivity'] *=\
            1 / np.sum(rf_params[key]['connectivity'])
    return rf_params


def set_nonlinearity(rf_params, nonlinearity_dict):
    for key in nonlinearity_dict.keys():
        rf_params[key]['synapse']['nonlinearity'] = nonlinearity_dict[key]
    return rf_params
