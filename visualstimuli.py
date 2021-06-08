import numpy as np


def set_stimulus_screen(span_deg=60, resolution_deg=0.5):
    '''Simulate responses to a sinewave of fixed paramaters besides the motion
    direction, output the series resistance to be used in neuron simulations

    Parameters
    ----------
    span_deg : int, optional
        size of the screen in degrees of visual angle.
    resolution_deg : float
        resolution in degrees of each 'pixel' in the screen.
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    x_range_rad : array_like
        Array containing coordinates of the x coordinates in rads
    y_range_rad : array_like
        Array containing coordinates of the y coordinates in rads

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    '''
    space_range_deg = np.linspace(-span_deg / 2, span_deg / 2, int(span_deg // resolution_deg))
    x_range, y_range = space_range_deg, space_range_deg
    x_range_rad = np.deg2rad(x_range)
    y_range_rad = np.deg2rad(y_range)
    return x_range_rad, y_range_rad


def sine_grating(x_range, y_range, t_range, phase0, orientation, direction,
                 spatial_freq, temporal_freq, min_lum, max_lum):
    kx = 2 * np.pi * spatial_freq * np.cos(direction)
    ky = 2 * np.pi * spatial_freq * np.sin(direction)
    w = 2 * np.pi * temporal_freq
    A = (max_lum - min_lum) / 2
    X, Y = np.meshgrid(x_range, y_range)
    grating = np.empty([x_range.shape[0], y_range.shape[0], t_range.shape[0]])
    i_t = 0
    for t in t_range:
        grating[:, :, i_t] = A * np.sin(kx * X + ky * Y - w * t + phase0)
        i_t += 1
    return grating
