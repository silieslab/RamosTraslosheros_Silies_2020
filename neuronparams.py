#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:32:07 2020
Store receptive field parameters
@author: lg
"""
import numpy as np


# connectivity
# ON pathway originally from
# https://iiif.elifesciences.org/lax:24394%2Felife-24394-fig4-v3.tif/full/,1500/0/default.jpg
# OFF pathway
# https://iiif.elifesciences.org/lax:40025%2Felife-40025-fig2-v2.tif/full/1500,/0/default.jpg
#        ___
#       /   \
#   ,--(  2  )--.
#  /  3 \___/  1 \
#  \    /   \    /
#   )--(  0  )--(
#  / 4  \___/ 6  \
#  \    /   \    /
#   `--(  5  )--'
#       \___/

class NeuronsParams:
    def __init__(self):
        fwhm2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
        # Generate receptive fields
        synapse_params = {'exc': {'type': 'exc',
                                  'tau_rise': 1e-3, 'tau_decay': 88.2e-3,
                                  'amplitude': 2.49e-5, 'e_reversal': 0},
                          'inh': {'type': 'inh',
                                  'tau_rise': 153.9e-3, 'tau_decay': 153.3e-3,
                                  'amplitude': 5e-4, 'e_reversal': -70}}
        rf_params = dict()
        # ON pathway
        rf_params['Mi4'] = {'center': {'sigma': 6.47, 'amplitude': 1},
                            'surround': {'sigma': 16.14, 'amplitude': 0.132},
                            'tau_fast': 0.038, 'tau_slow': 0.6,
                            'amplitude': 0.831,
                            'synapse': {**synapse_params['inh'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([0, 0, 0, 5, 8, 0, 0])}
        rf_params['Mi1'] = {'center': {'sigma': 6.81, 'amplitude': 1},
                            'surround': {'sigma': 28.81, 'amplitude': 0.022},
                            'tau_fast': 0.054, 'tau_slow': 0.318,
                            'amplitude': 1.146,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([36, 0, 2, 4, 2, 25, 5])}
        rf_params['Tm3'] = {'center': {'sigma': 11.91, 'amplitude': 1},
                            'surround': {'sigma': 16.14, 'amplitude': 0},
                            'tau_fast': 0.027, 'tau_slow': 0.260,
                            'amplitude': 1.035,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([17, 1, 2, 3, 3, 9, 6])}
        rf_params['Mi9'] = {'center': {'sigma': 6.37, 'amplitude': 1},
                            'surround': {'sigma': 23.98, 'amplitude': 0.063},
                            'tau_fast': 0.077, 'tau_slow': 0.6,
                            'amplitude': -0.789,
                            'synapse': {**synapse_params['inh'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([0, 0, 0, 0, 0, 0, 13])}
        # OFF pathway upwards T5c
        rf_params['Tm1'] = {'center': {'sigma': 8.12, 'amplitude': 1},
                            'surround': {'sigma': 27.14, 'amplitude': 0.04},
                            'tau_fast': 0.044, 'tau_slow': 0.296,
                            'amplitude': -1.117,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([27, 0, 0, 0, 0, 0, 0])}
        rf_params['Tm2'] = {'center': {'sigma': 7.93, 'amplitude': 1},
                            'surround': {'sigma': 30.52, 'amplitude': 0.035},
                            'tau_fast': 0.014, 'tau_slow': 0.153,
                            'amplitude': -1.038,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([35, 0, 0, 0, 0, 0, 0])}
        rf_params['Tm4'] = {'center': {'sigma': 11.45, 'amplitude': 1},
                            'surround': {'sigma': 34.62, 'amplitude': 0.054},
                            'tau_fast': 0.024, 'tau_slow': 0.249,
                            'amplitude': -1.018,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([20, 0, 0, 0, 0, 0, 0])}
        rf_params['Tm9'] = {'center': {'sigma': 6.92, 'amplitude': 1},
                            'surround': {'sigma': 23.78, 'amplitude': 0.046},
                            'tau_fast': 0.017, 'tau_slow': 0.6,
                            'amplitude': -0.827,
                            'synapse': {**synapse_params['exc'],
                                        'nonlinearity': 'linear'},
                            'connectivity': np.array([0, 0, 64, 0, 0, 0, 0])}
        rf_params['CT1'] = {'center': {'sigma': 6.92, 'amplitude': 1},
                            'surround': {'sigma': 23.78, 'amplitude': 0.046},
                            'tau_fast': 0.017, 'tau_slow': 0.6,
                            'amplitude': -0.827,
                            'synapse': {**synapse_params['inh'],
                                        'nonlinearity': 'relu'},
                            'connectivity': np.array([0, 0, 0, 0, 0, 31, 0])}

        for key in rf_params.keys():
            rf_params[key]['center']['sigma'] *= fwhm2sigma
            rf_params[key]['surround']['sigma'] *= fwhm2sigma

        self.synapse_params = synapse_params
        self.rf_params = rf_params

    # Load the paramaters from the file
    def get_pathway_rf_params(self, is_ON):
        ON_neurons = ['Mi1', 'Mi4', 'Mi9', 'Tm3']
        OFF_neurons = ['Tm1', 'Tm2', 'Tm4', 'Tm9', 'CT1']
        self.is_ON = is_ON
        if is_ON:
            self.rf_params = {k: self.rf_params[k] for k in ON_neurons}
        else:
            self.rf_params = {k: self.rf_params[k] for k in OFF_neurons}

    def set_EI_ratio(self, ei_ratio):
        '''Update synapse conductances, TODO use synapse type value.'''
        if self.is_ON:
            exc_amp = self.rf_params['Mi1']['synapse']['amplitude']
            inh_amp = exc_amp / ei_ratio
            # inh_amp = rf_params['Mi9']['synapse']['amplitude']
            for key in ['Mi4', 'Mi9']:
                self.rf_params[key]['synapse']['amplitude'] = inh_amp
        else:
            exc_amp = self.rf_params['Tm1']['synapse']['amplitude']
            inh_amp = exc_amp / ei_ratio
            # inh_amp = rf_params['CT1']['synapse']['amplitude']
            for key in ['CT1']:
                self.rf_params[key]['synapse']['amplitude'] = inh_amp

    def normalize_connectivity(self):
        for key in self.rf_params.keys():
            self.rf_params[key]['connectivity'] =\
                self.rf_params[key]['connectivity'].astype(float)
            self.rf_params[key]['connectivity'] *=\
                1 / np.sum(self.rf_params[key]['connectivity'])

    def normalize_total_connectivity(self):
        total_connectivity = sum(sum(self.rf_params[key]['connectivity'].astype(float))
                                 for key in self.rf_params.keys())
        for key in self.rf_params.keys():
            self.rf_params[key]['connectivity'] =\
                self.rf_params[key]['connectivity'].astype(float)
            self.rf_params[key]['connectivity'] *=\
                1 / total_connectivity

    def set_nonlinearity(self, nonlinearity_dict):
        for key in nonlinearity_dict.keys():
            self.rf_params[key]['synapse']['nonlinearity'] =\
                nonlinearity_dict[key]
