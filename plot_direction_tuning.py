#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:55:02 2020

@author: lg
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import plotsimulation as plotsim
import analyzesimulation as asim
import os
import re
import pickle
from collect_simulation_data import load_data_from_dir


def ax_despine(ax):
    # Move left and bottom spines outward by 5 points
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


# Loading the objects:
i_frac = 55.0


data_dir = 'output_data'


try:
    project_dir = os.path.dirname(__file__)
except Exception as e:
    print('Running interactively')
    file_path = '~/model-code/' # Replace by model directory
    project_dir = os.path.dirname(file_path)
finally:
    data_dir = os.path.join(project_dir, data_dir)
    fig_dir = os.path.join(project_dir,
                           re.sub('output_data', 'figures', data_dir))


if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
condition = 'linear'
fig_name_prefix = os.path.join(fig_dir, condition)


try:
    # Load data from file
    with open(os.path.join(data_dir, 'summary_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
except Exception as e:
    print('No file found!! Running data collection script ...')
    print(e)
    load_data_from_dir(data_dir)
finally:
    # Load data from file
    with open(os.path.join(data_dir, 'summary_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)


conds = ['relu', 'linear', 'Tm9Linear', 'Tm9Tm2Linear']


cm = 1/2.54  # centimeters in inches
plt.figure(figsize=(20*cm, 15*cm))
ax1, ax2 = [0]*3, [0]*3
ax1[0] = plt.subplot(231, projection='polar')
ax1[1] = plt.subplot(232, projection='polar')
ax1[2] = plt.subplot(233, projection='polar')
ax2[0] = plt.subplot(234)
ax2[1] = plt.subplot(235)
ax2[2] = plt.subplot(236)

tf = 1
keys = sorted(data_dict.keys())
for i_ax, tc_method in enumerate(sorted(data_dict[keys[0]]['tc'][0].keys())):
    x = 1
    for key in conds:
        dsi_array = []

    for key in conds:
        tf_array = np.array([tf[0] for tf in data_dict[key]['tf']])
        i_frac_array = np.array(data_dict[key]['i_frac'])
        tc_ind = np.squeeze(np.where(np.logical_and(i_frac_array == i_frac,
                                                    tf_array == tf)))
        tc_array = np.array(data_dict[key]['tc'][tc_ind][tc_method])
        norm_tc = tc_array.T / tc_array.max()
        dsi = data_dict[key]['dsi'][tc_ind][tc_method]
        ax2[i_ax].bar(x, dsi)
        plt.sca(ax1[i_ax])
        plt.polar(np.append(data_dict['relu']['dir'][0],
                  data_dict['relu']['dir'][0][0]),
                  np.append(norm_tc, norm_tc[0]), lw=3, label=key)
        x += 1
    ax1[i_ax].label_outer()
    plt.title('DS-tuning (%s)' % tc_method)
    ax2[i_ax].set_title('Direction-selectivity index')
    ax2[i_ax].set_ylabel('DSI')
    ax2[i_ax].get_xaxis().set_visible(False)
ax1[i_ax].legend(loc='upper center')

plt.savefig(os.path.join(fig_dir, 'polarTuning_ei%.2f.pdf' % (i_frac)),
            dpi=300, orientation='portrait', format='pdf', bbox_inches='tight')
plt.show()

# T5 model is for downward motion
pref_dir = np.deg2rad(270)
null_dir = np.deg2rad(90)
# Plot the raw signal as an image
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(nrows=1, ncols=len(conds), sharex=True, sharey=True,
                       figsize=(20*cm, 5*cm))
ax[0].set_xlabel('time (ms)')
ax[0].set_ylabel('ΔV (mV)')
for i_ax, key in enumerate(conds):
    tf_array = np.array([tf[0] for tf in data_dict[key]['tf']])
    i_frac_array = np.array(data_dict[key]['i_frac'])
    tc_ind = np.squeeze(np.where(np.logical_and(i_frac_array == i_frac,
                                                tf_array == tf)))
    directions = data_dict[key]['dir'][0]
    times = data_dict[key]['times'][0]
    pref_dir_ind = np.argmin(np.abs(np.array(directions) - pref_dir))
    pref_dir_deg = np.rad2deg(directions[pref_dir_ind])
    null_dir_ind = np.argmin(np.abs(np.array(directions) - null_dir))
    null_dir_deg = np.rad2deg(directions[null_dir_ind])
    ax[i_ax].set_title(key)
    ax[i_ax].plot(times, data_dict[key]['dv'][tc_ind][pref_dir_ind, :],
                  color='green', label='PD (%.1f deg)' % pref_dir_deg)
    ax[i_ax].plot(times, data_dict[key]['dv'][tc_ind][null_dir_ind, :],
                  color='purple', label='ND (%.1f deg)' % null_dir_deg)
    ax[i_ax].axhline(0)
    ax_despine(ax[i_ax])

ax[i_ax].legend()
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 8}

mpl.rc('font', **font)
plt.savefig(os.path.join(fig_dir, 'pd-nd-traces_ei%.2f.pdf' % (i_frac)),
            dpi=300, orientation='portrait', format='pdf', bbox_inches='tight')
plt.show()


def plot_input_current(t, input_i, title_suffix, fig_name_prefix):
    plt.figure(dpi=150)
    ax = plt.subplot(111)
    for key in input_i:
        plt.plot(t, 1e3*input_i[key], label=key)
    plt.title('Input currents ' + title_suffix)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(fig_name_prefix, 'inputs.png'), dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_input_current_dir(t, input_i, data, dir, fig_name_prefix):
    curr_ind = np.argmin(np.abs(np.array(data.directions)-dir))
    closest_dir = np.rad2deg(data.directions[curr_ind])
    title_suffix = 'direction %.1f' % (closest_dir)
    plot_input_current(t, input_i[curr_ind], title_suffix, fig_name_prefix)
