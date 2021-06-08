#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:52:12 2020

@author: lg
"""
# Simulate direction tuning
import numpy as np
import neuronsimulation as neusim
import pickle
import os
import neuronparams as nrnparams

import time
startTime = time.time()


def simulate_sine_direction_tuning(directions, rf_params,
                                   stimulation_time, time_step,
                                   t_freq, g_scale, is_ON, datafile):
    master_output = []
    for i_dir in directions:
        output_dict = neusim.get_sine_voltage_response(rf_params,
                                                       i_dir,
                                                       stimulation_time,
                                                       time_step,
                                                       temporal_freq=t_freq,
                                                       g_scale=g_scale,
                                                       is_ON=is_ON)
        master_output.append(output_dict)

        # Saving the objects:
        with open(datafile, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(master_output, f)


directions = np.arange(0, 2*np.pi, np.pi / 8)
stimulation_time = 6
time_step = 10e-3 # somehow 5e-3 causes some numerical instability
t_freq = 1
g_scale = 1
is_ON = False


project_dir = os.path.dirname(__file__)
data_dir = os.path.join(project_dir, 'output_data_ei')
pathway = 'ON' if is_ON else 'OFF'

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)


for iInh in [1, 2, 4, 8, 16, 32, 64, 96, 128]:
    # Load the paramaters from the file
    params = nrnparams.NeuronsParams()
    params.get_pathway_rf_params(is_ON)
    params.normalize_total_connectivity()
    ei_ratio = 1/iInh
    params.set_EI_ratio(ei_ratio)

    # Simulate all OFF rectified neurons
    if not is_ON:
        nonlinearity_dict = {'Tm2': 'relu', 'Tm9': 'relu', 'CT1': 'relu'}
        params.set_nonlinearity(nonlinearity_dict)

    prefix = 'out_dirTuning'
    suffix = 'relu_tCausal-L2Norm_sDoG-L1Norm_E1I%.2f' % (iInh)
    prefix = pathway + '_' + prefix + '_'
    suffix = suffix + '.pkl'

    filename = prefix + 'gscale_1e%.2f_tf_%.3f_' % (g_scale, t_freq) + suffix
    datafile = os.path.join(project_dir, data_dir, filename)

    simulate_sine_direction_tuning(directions, params.rf_params,
                                   stimulation_time, time_step, t_freq,
                                   g_scale, is_ON, datafile)

    # Simulate all linear neurons
    if not is_ON:
        nonlinearity_dict = {'Tm2': 'linear', 'Tm9': 'linear', 'CT1': 'linear'}
        params.set_nonlinearity(nonlinearity_dict)

    prefix = 'out_dirTuning'
    suffix = 'linear_tCausal-L2Norm_sDoG-L1Norm_E1I%.2f' % (iInh)
    prefix = pathway + '_' + prefix + '_'
    suffix = suffix + '.pkl'

    filename = prefix + 'gscale_1e%.2f_tf_%.3f_' % (g_scale, t_freq) + suffix
    datafile = os.path.join(project_dir, data_dir, filename)

    simulate_sine_direction_tuning(directions, params.rf_params,
                                   stimulation_time, time_step, t_freq,
                                   g_scale, is_ON, datafile)

    # Simulate all rectified neurons but Tm9 linear
    if not is_ON:
        nonlinearity_dict = {'Tm2': 'relu', 'Tm9': 'linear', 'CT1': 'relu'}
        params.set_nonlinearity(nonlinearity_dict)
    prefix = 'out_dirTuning'
    suffix = 'Tm9Linear_tCausal-L2Norm_sDoG-L1Norm_E1I%.2f' % (iInh)
    prefix = pathway + '_' + prefix + '_'
    suffix = suffix + '.pkl'

    filename = prefix + 'gscale_1e%.2f_tf_%.3f_' % (g_scale, t_freq) + suffix
    datafile = os.path.join(project_dir, data_dir, filename)

    simulate_sine_direction_tuning(directions, params.rf_params,
                                   stimulation_time, time_step, t_freq,
                                   g_scale, is_ON, datafile)

    # Simulate all rectified neurons but Tm9 and Tm2 linear
    if not is_ON:
        nonlinearity_dict = {'Tm2': 'linear', 'Tm9': 'linear', 'CT1': 'relu'}
        params.set_nonlinearity(nonlinearity_dict)
    prefix = 'out_dirTuning'
    suffix = 'Tm9Tm2Linear_tCausal-L2Norm_sDoG-L1Norm_E1I%.2f' % (iInh)
    prefix = pathway + '_' + prefix + '_'
    suffix = suffix + '.pkl'

    filename = prefix + 'gscale_1e%.2f_tf_%.3f_' % (g_scale, t_freq) + suffix
    datafile = os.path.join(project_dir, data_dir, filename)

    simulate_sine_direction_tuning(directions, params.rf_params,
                                   stimulation_time, time_step, t_freq,
                                   g_scale, is_ON, datafile)

    filename = prefix + 'gscale_1e%.2f_tf_%.3f_' % (g_scale, t_freq) + suffix
    datafile = os.path.join(project_dir, data_dir, filename)

    simulate_sine_direction_tuning(directions, params.rf_params,
                                   stimulation_time, time_step, t_freq,
                                   g_scale, is_ON, datafile)
