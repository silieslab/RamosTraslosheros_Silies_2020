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


# Loading the objects:
data_dir = 'output_data_ei'

try:
    project_dir = os.path.dirname(__file__)
except Exception as e:
    print('Running interactively')
    file_path = '~/model-code/' # Replace model directory
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
    print(e)
    print('No file found!! Running data collection script ...')
    load_data_from_dir(data_dir)
finally:
    # Load data from file
    with open(os.path.join(data_dir, 'summary_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)



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
    

conds = ['relu', 'linear', 'Tm9Linear', 'Tm9Tm2Linear']
#%% collect data function does not retrieve the used tuning method (assumed to be A1, now variable)
cm = 1/2.54  # centimeters in inches
plt.figure(figsize=(20*cm, 4*cm))
ax1 = [0]*3
ax1[0] = plt.subplot(131)
ax1[1] = plt.subplot(132)
ax1[2] = plt.subplot(133)

for i_ax, tc_method in enumerate(data_dict[conds[0]]['dsi'][0].keys()):
    plt.sca(ax1[i_ax])
    for key in conds:
        i_order = np.argsort(data_dict[key]['i_frac'])
        dsi_array = np.asarray([dsi[tc_method] for dsi in data_dict[key]['dsi']])
        plt.plot(np.take(data_dict[key]['i_frac'], i_order, axis=0),
                 np.take(dsi_array, i_order, axis=0), label=key, marker='o')
    plt.title('DS-tuning-vs-EI')
    ax1[i_ax].set_title('Direction-selectivity index')
    ax1[i_ax].set_ylabel('DSI (%s)' % tc_method)
    ax1[i_ax].set_xlabel('I/E ratio')
    ax_despine(ax1[i_ax])
ax1[i_ax].legend()


font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 8}

mpl.rc('font', **font)
plt.savefig(os.path.join(fig_dir,
            'dsi-vs-ei-conductance-shift-%s.pdf' % '-'.join(list(data_dict[conds[0]]['dsi'][0]))),
            dpi=300, orientation='portrait', format='pdf', bbox_inches='tight')

plt.show()

