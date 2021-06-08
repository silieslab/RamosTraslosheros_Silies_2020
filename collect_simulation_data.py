import numpy as np
import analyzesimulation as asim
import os
import re
import pickle


def load_data_from_dir(data_dir):
    filename_list = os.listdir(data_dir)
    conditions = set()
    for file in filename_list:
        condition = re.findall(r'.*tf_.*_(.*)_t.*', file)
        condition = condition[0]
        conditions.add(condition)

    data_dict = dict()
    for key in conditions:
        data_dict[key] = {'i_frac': [], 'dsi': [], 'tc': [], 'dv': [],
                          'dir': [], 'tf': [], 'input_i': [], 'times': []}

    for file in filename_list:
        datafile = os.path.join(data_dir, file)

        inh_fraction = re.findall(r'.*I(.*)[.]pkl', file)
        inh_fraction = float(inh_fraction[0])
        condition = re.findall(r'.*tf_.*_(.*)_t.*', file)
        condition = condition[0]
        # Loading the objects:
        data = asim.SimData(datafile)
        # Discard the first 50ms. Time in ms.
        data.trim_data_beginning(500)
        data.set_baseline_voltage(-65)  # in mV
        delta_v = data.get_voltage_change()
        data.get_amp_spectrum()
        data.get_calcium_response()

        #  Get tuning curve from A1 component
        tc = dict()
        dsi = dict()
        pref_dir = dict()
        tc['max'] = np.max(delta_v, axis=1)
        tc['mean'] = np.mean(data.calcium_responses, axis=1)
        tc['a1'] = [asim.get_freq_amp(data.fft_freq, data.v_amp[i, :],
                    data.t_freqs[i]) for i in range(len(data.t_freqs))]
        for tc_method in tc.keys():
            dsi[tc_method], pref_dir[tc_method] =\
                asim.get_pref_dir(tc[tc_method], data.directions)
            pref_dir[tc_method] = np.rad2deg(pref_dir[tc_method])


        data_dict[condition]['i_frac'].append(inh_fraction)
        data_dict[condition]['dsi'].append(dsi)
        data_dict[condition]['tc'].append(tc)
        data_dict[condition]['dv'].append(delta_v)
        data_dict[condition]['input_i'].append(data.input_i)

# Conditional not useful if we need to analyze across tf and dirs. Although it saves memory
        # if len(data_dict[condition]['dir']) == 0:
        data_dict[condition]['dir'].append(data.directions)
        data_dict[condition]['tf'].append(data.t_freqs)
        data_dict[condition]['times'].append(data.times)
    # Saving the objects:
    # Python 3: open(..., 'wb')
    with open(os.path.join(data_dir, 'summary_dict.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)
