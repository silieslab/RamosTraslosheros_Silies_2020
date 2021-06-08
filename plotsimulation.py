#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 01:55:02 2020

@author: lg
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import numpy as np


def cmap_1dtuning_traces(t, v):
    cmap = plt.get_cmap('inferno_r')
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    if np.all(v < 0):
        norm = mpl.colors.LogNorm(vmin=np.min(-v), vmax=np.max(-v))
        im = ax.pcolormesh(t, np.arange(v.shape[0]+1), (-v),
                           cmap=cmap, norm=norm)
    else:
        norm = mpl.colors.Normalize(vmin=np.min(v), vmax=np.max(v))
        im = ax.pcolormesh(v, cmap=cmap, norm=norm)
    return fig, ax, im


def plot_dir_colormap(t, v, directions, fig_name_prefix):
    fig, ax, im = cmap_1dtuning_traces(t, v)

    ax.set_yticks(np.arange(len(directions))+0.5)
    label_str = ['%i' % i for i in np.rad2deg(directions)]
    ax.set_yticklabels(label_str)
    ax.set_ylabel('Direction (deg)')
    ax.set_xlabel('Time (ms)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Voltage (mV)')
    plt.savefig(fig_name_prefix + '_dir_cmap.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_tf_colormap(t, v, t_freqs, fig_name_prefix):
    fig, ax, im = cmap_1dtuning_traces(t, v)

    ax.set_yticks(np.arange(len(t_freqs))+0.5)
    label_str = ['%.2e' % i for i in t_freqs]
    ax.set_yticklabels(label_str)
    ax.set_ylabel('Direction (deg)')
    ax.set_xlabel('Time (ms)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Voltage (mV)')
    plt.savefig(fig_name_prefix + '_tf_cmap.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def subplot_1dtuning_traces(t, delta_v, label_str, fig_name_prefix):
    '''Multiple subplots for the signal'''
    M = 2
    yticks = ticker.MaxNLocator(M)
    xticks = ticker.MaxNLocator(M)
    n_cols = 4
    n_rows = np.int(np.ceil(delta_v.shape[0] / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 10),
                             sharex=True, sharey=True, dpi=80)
    axes = np.ravel(axes)
    for i_ax in np.arange(len(delta_v)):
        axes[i_ax].plot(t/1e3, delta_v[i_ax, :], label=label_str[i_ax])
        axes[i_ax].xaxis.set_major_locator(xticks)
        axes[i_ax].yaxis.set_major_locator(yticks)
        axes[i_ax].set_title(label_str[i_ax], {'fontsize': 10})
        axes[i_ax].set_ylabel("$\Delta V$")
    for ax in axes.flat:
        ax.label_outer()
    #    axes[i_ax].legend(fontsize='x-small', frameon=False, handlelength=0)
    plt.savefig(fig_name_prefix + '_grid.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_dirTuning(tc, directions_deg, fig_name_prefix):
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    for tc_method in tc.keys():
        tc_norm = tc[tc_method] / np.max(tc[tc_method])
        plt.plot(directions_deg, tc_norm, label=tc_method)
    plt.xlabel('Direction (deg)')
    plt.ylabel('Response (a.u.)')
    plt.legend()
    plt.savefig(fig_name_prefix + '_dir_cart.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_dirTuning_polar(tc, directions, fig_name_prefix):
    plt.figure(dpi=150)
    for tc_method in tc.keys():
        tc[tc_method] = tc[tc_method] / np.max(tc[tc_method])
        plt.polar(np.append(directions, directions[0]),
                  np.append(tc[tc_method], tc[tc_method][0]),
                  label=tc_method)
    plt.legend()
    plt.savefig(fig_name_prefix + '_dir_polar.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_dsi_bar(dsi_dict, fig_name_prefix):
    plt.figure(dpi=150)
    plt.bar(range(len(dsi_dict)), list(dsi_dict.values()), align='center')
    plt.xticks(range(len(dsi_dict)), list(dsi_dict.keys()))
    plt.xlabel('Tuning method')
    plt.ylabel('DSI')
    plt.savefig(fig_name_prefix + '_dsi.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_tf_tuning(t_freqs, tc, tc_method, direction_deg, fig_name_prefix):
    plt.figure(dpi=150)
    plt.plot(np.log2(t_freqs), tc[tc_method], label=tc_method)
    plt.title('Temporal frequency tuning at %.1f deg' % (direction_deg))
    plt.ylabel('Max $\Delta V$ (V)')
    plt.xlabel('Log2(temporal frequency), e.g., 0 -> 2^0 = 1 Hz')
    plt.legend()
    plt.savefig(fig_name_prefix + '_tf_tc.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
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
    plt.savefig(fig_name_prefix + '_inputs.png', dpi=300,
                orientation='portrait', format='png', bbox_inches='tight')
    plt.show()


def plot_input_current_dir(t, input_i, data, dir, fig_name_prefix):
    curr_ind = np.argmin(np.abs(np.array(data.directions)-dir))
    closest_dir = np.rad2deg(data.directions[curr_ind])
    title_suffix = 'direction %.1f' % (closest_dir)
    plot_input_current(t, input_i[curr_ind], title_suffix, fig_name_prefix)
