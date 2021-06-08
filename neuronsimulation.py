#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with DC
current stimulus
'''
# Import modules for plotting and NEURON itself
import numpy as np
import neuron
import simulatesinewave as sw
# import pickle


def get_sine_voltage_response(rf_params, direction,
                              stimulation_time, time_step,
                              temporal_freq=1, g_scale=3, is_ON=True):

    rs, t_responses = sw.get_response_to_sinewave(
        rf_params, direction, temporal_freq=temporal_freq,
        stimulation_time=stimulation_time, time_step=time_step, is_ON=is_ON)

    # %pylab
    print('Input initialization: DONE')
    output_dict = get_voltage_response_from_input_Rs(rf_params,
                                                     rs, t_responses,
                                                     g_scale, is_ON)
    # Save data as a dict
    output_dict.update({'direction': direction, 't_freq': temporal_freq})

    return output_dict


def get_voltage_response_from_input_Rs(rf_params, rs, t_responses,
                                       g_scale=3, is_ON=True):

    ##################################################################
    # Neuron topology is defined using Sections
    ##################################################################
    dendrite = neuron.h.Section(name='dendrite')
    # print out information on the dendrite section to terminal
    neuron.h.psection()

    ##################################################################
    # Set the model geometry
    ##################################################################
    dendrite.L = 40. / 3.         # section length in um
    dendrite.diam = 0.2      # section diameter in um
    dendrite.nseg = 10       # number of segments (compartments)

    ##################################################################
    # Set biophysical parameters
    ##################################################################
    dendrite.Ra = 100       # Axial resistivity in Ohm*cm
    dendrite.cm = 1         # membrane capacitance in uF/cm2

    # insert 'passive' membrane mechanism, adjust parameters.
    # None: without a leak mechanism, the neuron will be a
    # perfect integrator
    dendrite.insert('pas')
    for seg in dendrite:
        seg.pas.g = 1 / 9.72e3  # 0.0002 # membrane conducance in S/cm2
        seg.pas.e = -65.  # leak reversal potential in mV
    # dendrite(0.5).pas.g = 1 / 9.72  # 0.0002 # membrane conducance in S/cm2
    # dendrite(0.5).pas.e = -65.  # leak reversal potential in mV

    ##################################################################
    # Model instrumentation
    ##################################################################

    if is_ON:
        # Attach current clamp to the neuron
        seclamp = {'Mi9': neuron.h.SEClamp(0.85, sec=dendrite),
                   'Tm3': neuron.h.SEClamp(0.65, sec=dendrite),
                   'Mi1': neuron.h.SEClamp(0.45, sec=dendrite),
                   'Mi4': neuron.h.SEClamp(0.15, sec=dendrite)}
    else:
        # Attach current clamp to the neuron
        seclamp = {'Tm9': neuron.h.SEClamp(0.70, sec=dendrite),
                   'Tm4': neuron.h.SEClamp(0.50, sec=dendrite),
                   'Tm2': neuron.h.SEClamp(0.49, sec=dendrite),
                   'Tm1': neuron.h.SEClamp(0.48, sec=dendrite),
                   'CT1': neuron.h.SEClamp(0.15, sec=dendrite),
                   }

    for key in seclamp.keys():
        seclamp[key].dur1 = 1e10  # duration of clamp, set to very large number
        seclamp[key].amp1 = rf_params[key]['synapse']['e_reversal']
        # reversal potential of exc. synapse is 0.
        # reversal potential of inh. synapse is -70.

    neuron.h.psection()
    ##################################################################
    # Set up recording of variables
    ##################################################################
    # NEURON variables can be recorded using Vector objects. Here, we
    # set up recordings of time, voltage and stimulus current with the
    # record attributes.
    t = neuron.h.Vector()
    v = neuron.h.Vector()
    #    v2 = neuron.h.Vector()
    i = {k: neuron.h.Vector() for k in seclamp}
    # i = neuron.h.Vector()
    # recordable variables must be preceded by '_ref_':
    t.record(neuron.h._ref_t)
    v.record(dendrite(0.0)._ref_v)
    # v2.record(dendrite(0.5)._ref_v)
    for key in i.keys():
        i[key].record(seclamp[key]._ref_i)
    # i.record(seclamp._ref_i)

    print('Neuron model configuration: DONE')

    ##################################################################
    # Simulation control
    ##################################################################
    neuron.h.dt = 0.2          # simulation time resolution
    stimulus_time = (t_responses[-1] - t_responses[0]) * 1000
    tstop = 20000.        # simulation duration
    tstop = np.min([tstop, stimulus_time])
    v_init = -65        # membrane voltage(s) at t = 0

    def initialize():
        '''
        initializing function, setting the membrane voltages to v_init
        and resetting all state variables
        '''
        neuron.h.finitialize(v_init)
        neuron.h.fcurrent()

    def integrate():
        '''
        run the simulation up until the simulation duration
        '''
        print('Simulating for %.0f ms' % np.min([tstop, stimulus_time]))
        while neuron.h.t < tstop:
            neuron.h.fadvance()
            stim_res = np.mean(np.diff(t_responses))
            ind = np.int_(np.floor((neuron.h.t/1000) / stim_res))
            # within_step = (neuron.h.t/1000 - t_responses[1] -
            # ind*stim_res)/stim_res
            # seclamp.rs = current[ind] +  within_step *
            # (current[ind+1] - current[ind])
            # in megaohm
            for key in seclamp.keys():
                if 0 <= ind < rs[key].shape[-1]:
                    # Sum inputs in parallel, linear summation of conductances.
                    syn_weights = rf_params[key]['connectivity']
                    # 1e9 Ohm was rs convention for zero conductance.
                    rs_points = rs[key][:, ind]
                    max_g = 1e9
                    g = np.where(rs_points == max_g, 0, 1 / rs_points)
                    total_g = np.dot(syn_weights, g)
                    seclamp[key].rs = np.where(total_g != 0,
                                               10**g_scale / total_g, max_g)
                else:
                    seclamp[key].rs = max_g
    # 1e3 * (1.5 + np.sin(2 * ni * 0.1 * neuron.h.t))

    # run simulation
    initialize()
    integrate()
    print('Neuron model simulation: DONE')

    ##################################################################
    # Plot simulated output
    ##################################################################

    # Convert the neuron vector objects to numpy arrays
    times = np.array(t)
    voltage = np.array(v)
    input_currents = dict()

    for j, key in enumerate(i.keys()):
        input_currents[key] = np.array(i[key])

    # Save data as a dict
    output_dict = {'t': times, 'output_voltage': voltage,
                   'input_currents': input_currents}

    #    # Saving the objects:
    #    with open('Data/output_%.3f_theta_%.3f_tfreq'%
    #              (np.rad2deg(direction), temporal_freq)+'.pkl', 'wb') as f:
    #    Python 3: open(..., 'wb')
    #        pickle.dump(output_dict, f)

    ##################################################################
    # customary cleanup of object references - the psection() function
    # may not write correct information if NEURON still has object
    # references in memory.
    ##################################################################
    i = None
    v = None
    t = None
    seclamp = None
    dendrite = None

    return output_dict
