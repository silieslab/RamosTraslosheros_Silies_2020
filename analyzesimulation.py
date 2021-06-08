import pickle
import numpy as np


def get_pref_dir(r, theta):
    C = np.dot(r, np.cos(theta)) / np.sum(np.abs(r))
    S = np.dot(r, np.sin(theta)) / np.sum(np.abs(r))
    R = np.hypot(S, C)
    dsi = R
    pref_theta = np.arctan2(S, C)
    return dsi, pref_theta


def get_freq_amp(freqs, amps, x_freq):
    # Assumes the value lies between two existing points
    ind = np.argmin(np.abs(freqs-x_freq))
    if freqs[ind] >= x_freq:
        x_amp = np.interp(x_freq, freqs[ind-1:ind+1], amps[ind-1:ind+1])
    else:
        x_amp = np.interp(x_freq, freqs[ind:ind+2], amps[ind:ind+2])
    return x_amp

# Variables from NEURON are time(t), current(i), voltage (v), tstop.


class SimData:

    def __init__(self, datafile):
        # Load data from file
        with open(datafile, 'rb') as f:  # Python 3: open(..., 'wb')
            self.data = pickle.load(f)
        # Extract simulation paramaters
        try:
            self.directions = [item['direction'] for item in self.data]
            self.t_freqs = [item['t_freq'] for item in self.data]
        except Exception as e:
            print(e)
            print('This is likely a natural movie response %s' % datafile)
            self.image_inds = [item['image_ind'] for item in self.data]
            self.speeds = [item['speed'] for item in self.data]
            self.y_initial = [item['y_initial'] for item in self.data]
            self.x_initial = [item['x_initial'] for item in self.data]

        self.voltages = np.array([item['output_voltage']
                                  for item in self.data])
        # All simulations have same time points
        self.times = self.data[0]['t']
        self.input_i = []
        self.get_input_currents()
        self.delta_t = np.mean(np.diff(self.times))
        self.sampling_freq = 1e3 / self.delta_t

    def get_input_currents(self):
        for item in self.data:
            dummy = dict()
            for key, value in item['input_currents'].items():
                dummy[key] = value
            self.input_i.append(dummy)

    def get_closest_direction_deg(self, dir):
        curr_ind = np.argmin(np.abs(np.array(self.directions)-dir))
        return np.rad2deg(self.directions[curr_ind])

    def get_amp_spectrum(self):
        v = np.squeeze(self.voltages)
        self.fft_freq = np.fft.fftfreq(self.times.size, 1 / self.sampling_freq)
        v_fft = np.fft.fft(v, axis=1)
        self.v_amp = np.abs(v_fft)

    def trim_data_beginning(self, trim_time_ms):
        ''' Strangely the first time points jumped quickly to high values before
        stabilizing, thus this function is to trim the beginning artifact.'''
        stable_inds = np.squeeze(np.where(self.times >= trim_time_ms))
        self.times = self.times[stable_inds]
        self.voltages = self.voltages[:, stable_inds]
        # v = np.squeeze(self.voltages[:, stable_inds])
        input_i_tmp = self.input_i
        for i in range(len(input_i_tmp)):
            for key, value in input_i_tmp[i].items():
                self.input_i[i][key] = value[stable_inds]

    def set_baseline_voltage(self, baseline_voltage):
        self.baseline_voltage = baseline_voltage

    def get_voltage_change(self):
        return self.voltages - self.baseline_voltage

    def get_calcium_response(self):
        delta_v = self.get_voltage_change()
        self.calcium_responses = np.where(delta_v < 0, 0, delta_v**2)
