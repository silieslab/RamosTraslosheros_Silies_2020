import visualstimuli as vs
import receptivefields as rf
import numpy as np
# import matplotlib.pyplot as plt
# Generate stimulus


def get_response_to_sinewave(rf_params, direction, temporal_freq=1, stimulation_time=4,
                             time_step=5e-3, is_ON=True):
    '''Simulate responses to a sinewave of fixed paramaters besides the motion 
    direction, output the series resistance to be used in neuron simulations'''
    # Time array
    t_stim = np.arange(0, stimulation_time + time_step, time_step)
    x_range_rad, y_range_rad = vs.set_stimulus_screen(span_deg=60, resolution_deg=0.5)  
    np.rad2deg(y_range_rad)
    # Set grating parameters.
#    temporal_freq = 1
    spatial_freq = 1 / (np.deg2rad(24))
    phase0 = 0
    min_lum, max_lum = -1, 1
    #direction = np.deg2rad(45 * 0.1)
    orientation = 0
    # wavelength_deg = np.rad2deg(1 / spatial_freq)
    grating = vs.sine_grating(x_range_rad, y_range_rad, t_stim, phase0,
                              orientation, direction, spatial_freq, temporal_freq,
                              min_lum, max_lum)
    
    # Linear filter parameters.
    filter_distance_deg = 5
    filter_duration = 2 # In seconds
    time_points = np.arange(0, filter_duration + time_step, time_step)
    t_responses = t_stim[len(time_points) - 1:]
    rs = dict()
    tf = dict()
    for key in rf_params.keys():
        rs[key], tf[key] = rf.get_Rs_from_RFs_stim(time_points, 
                                                   np.rad2deg(x_range_rad), 
                                                   np.rad2deg(y_range_rad),
                                                   filter_distance_deg, 
                                                   t_stim,
                                                   grating, 
                                                   rf_params[key],
                                                   rf_params[key]['synapse'], 
                                                   key)
    return rs, t_responses
