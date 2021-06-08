#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 01:20:00 2020

@author: lg
"""
import numpy as np


def causal_filter(t, tau):
    ''' For continuous scale parameter tau'''
    return t * np.exp(-t**2 / 2 / tau) / np.sqrt(2 * np.pi) / tau**(3/2)


def cascade_integrator(t, tau, k):
    ''' For discrete number of scales k'''
    from scipy.special import gamma
    # Uniform distribution of scales
    mu = np.sqrt(tau / k)
    # Log distribution of scales
    tau_min = np.mean(np.diff(t))
    c = (tau / tau_min)**(1 / (k-1))
    mu = np.sqrt(tau_min * (c - 1)) * c**((k - 1) / 2)
    return t**(k - 1) * np.exp(-t / mu) / mu**k / gamma(k)


def double_exponential_old(t, tau_rise, tau_decay, amplitude):
    ''' This function computes a double exponential monophasic filter contrained to
    have zero values at start and end points, and a maximum amplitude of 1'''
    tau_ratio = tau_decay / tau_rise
    tau_diff = tau_rise - tau_decay
    tau_norm_factor = amplitude * tau_rise / tau_diff *\
        tau_ratio ** (tau_decay / tau_diff)
    return (np.exp(-t / tau_rise) - np.exp(-t / tau_decay)) * tau_norm_factor


def double_exponential(t, tau_rise, tau_decay, amplitude):
    ''' This function computes a double exponential monophasic filter contrained to
    have zero values at start and end points, and a maximum amplitude of 1'''
    return np.sign(amplitude) * (np.exp(-t / tau_rise) - np.exp(-t / tau_decay))


def low_pass_filter(t, tau_lp):
    dt = np.mean(np.diff(t))
    low_pass = np.sqrt(dt) * 2 * tau_lp**(-3/2) * (t >= 0).astype(float) * \
        t * np.exp(-t / tau_lp)
    return low_pass


def high_pass_filter(t, tau_hp):
    dt = np.mean(np.diff(t))
    high_pass = np.sqrt(dt) * 2 * tau_hp**(-3/2) * (t >= 0).astype(float) * \
        (tau_hp - t) * np.exp(-t / tau_hp)
    return high_pass


def low_plus_high_filter(t, tau_lp, tau_hp, A):
    band_pass = low_pass_filter(t, tau_lp) + np.abs(A) * high_pass_filter(t, tau_hp)
    return band_pass
