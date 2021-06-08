import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
import timefilters as tfilt



def temporal_filter(t, tau_rise, tau_decay, amplitude):
    ''' This function computes a double exponential monophasic filter contrained to
    have zero values at start and end points, and a maximum amplitude of 1'''
    tau_ratio = tau_decay / tau_rise
    tau_diff = tau_rise - tau_decay
    tau_norm_factor = amplitude * tau_rise / tau_diff *\
        tau_ratio ** (tau_decay / tau_diff)
    return (np.exp(-t / tau_rise) - np.exp(-t / tau_decay)) * tau_norm_factor


def spatial_filter(x_range, y_range, rf_params, x_pos, y_pos):
    sigma_center = rf_params['center']['sigma']
    sigma_surround = rf_params['surround']['sigma']
    amp_center = rf_params['center']['amplitude']
    amp_surround = rf_params['surround']['amplitude']
    filt_center_x = gaussian_profile(x_range, sigma=sigma_center, mu=x_pos)
    filt_center_y = gaussian_profile(y_range, sigma=sigma_center, mu=y_pos)
    filt_surround_x = gaussian_profile(x_range, sigma=sigma_surround, mu=x_pos)
    filt_surround_y = gaussian_profile(y_range, sigma=sigma_surround, mu=y_pos)
    # In numpy the outer product takes the first input as the column vector.
    # Column vector should be the y RF.
    center = amp_center * np.outer(filt_center_y, filt_center_x)
    surround = amp_surround * np.outer(filt_surround_y, filt_surround_x)
    return center, surround


def gaussian_profile(x, mu, sigma):
    return np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))


def get_hexagon_coords(distance):
    i_coord = 1
    x_coords = np.zeros([7, 1])
    y_coords = np.zeros([7, 1])
    for i_theta in np.arange(0, 2 * np.pi, np.pi / 3):
        x_coords[i_coord] = distance * np.cos(i_theta)
        y_coords[i_coord] = distance * np.sin(i_theta)
        i_coord += 1
    return x_coords, y_coords


def get_filter_hex_array(x_range, y_range, distance, rf_params):
    x_coords, y_coords = get_hexagon_coords(distance)
    center_array = np.empty((x_range.size, y_range.size, x_coords.size))
    surround_array = np.empty((x_range.size, y_range.size, x_coords.size))
    # Generate the seven filters in the hexagonal array.
    i_column = 0
    for x, y in zip(x_coords, y_coords):
        center, surround = spatial_filter(x_range, y_range, rf_params, x, y)
        center_array[:, :, i_column] = center
        surround_array[:, :, i_column] = surround
        i_column += 1
    return center_array - surround_array


def plot_hex_spatial_rfs(x_range, y_range, center_surround_filter, distance):
    x_coords, y_coords = get_hexagon_coords(distance)
    fig, ax = plt.subplots(nrows=3, ncols=3, dpi=80)
    max_val = np.max(np.abs(center_surround_filter))
    i_filter = 0
    axes_inds = [(1, 1), (0, 2), (0, 1), (0, 0), (2, 0), (2, 1), (2, 2)]
    M = 3
    yticks = ticker.MaxNLocator(M)
    for ax_inds in axes_inds:
        im = ax[ax_inds].pcolormesh(x_range, y_range,
                                    center_surround_filter[:, :, i_filter],
                                    vmin=-max_val, vmax=max_val, cmap=cm.PRGn)
        ax[ax_inds].plot(x_coords, y_coords,
                         lw=0, marker='.', ms=5, mew=0, mfc='k')
        ax[ax_inds].axis('equal')
        ax[ax_inds].xaxis.set_major_locator(yticks)
        ax[ax_inds].yaxis.set_major_locator(yticks)
        i_filter += 1
    plt.setp([a.get_xticklabels() for a in fig.axes[:-3]], visible=False)
    plt.setp([fig.axes[i].get_yticklabels() for i in [1, 2, 3, 5, 7, 8]],
             visible=False)
    fig.colorbar(im, ax=ax.ravel().tolist())
    fig.delaxes(ax[1, 0])
    fig.delaxes(ax[1, 2])
    return fig, ax,


def get_stim_response(space_time_filter, stimulus):
    '''Compute stimulus response from a single cell STRF.'''
    filter_len = space_time_filter.size
    t_filter_len = space_time_filter.shape[-1]
    # The reshape is happening first in rows then columns and finally times.
    filter_1d = np.reshape(space_time_filter, filter_len, order='F')
    t_stim_start = t_filter_len - 1
    t_stim_end = stimulus.shape[-1]
    i_time = 0
    response = np.zeros(t_stim_end - t_stim_start)
    for t in np.arange(t_stim_start, t_stim_end):
        t_start = t - t_stim_start
        t_end = t + 1
        stim_1d = np.reshape(stimulus[:, :, t_start:t_end], filter_len,
                             order='F')
        response[i_time] = np.dot(filter_1d.T, stim_1d)
        i_time += 1
    return response


def get_stim_response_separable(center_surround_filter, t_filter, stimulus):
    '''Compute stimulus response from a single cell separable STRF.'''
    # from scipy.signal import convolve as convolve
    from scipy.ndimage import convolve1d
    # First reduce the size of the matrix by multiplying by space filter.
    # The filter hex array can be first mutiplied then summed or vice versa?
    # This is important for the function calling this function.

    # s_filter = center_surround_filter / np.sum(center_surround_filter**2)
    # s_filter = center_surround_filter / np.sum(np.abs(center_surround_filter))
    # s_filter = center_surround_filter / np.linalg.norm(center_surround_filter, ord=1)
    s_filter = center_surround_filter / np.sum(center_surround_filter)
    response = np.zeros(stimulus.shape[-1])
    for t in np.arange(0, stimulus.shape[-1]):
        response[t] = np.sum(np.multiply(stimulus[:, :, t], s_filter))
    # Adding origin, corrects for the shift in timing induced by convolution,
    # which makes the neuron respond before the stimulus onset.
    response = convolve1d(response, np.flipud(t_filter), mode='constant', cval=0,
    origin=-int(t_filter.shape[-1] // 2))
    return response


def get_Rs_from_stim(t_responses, response, synapse_params):
    from scipy.integrate import odeint
    # Do LN before conductance to avoid linear responses of rectified neurons.
    # linear_shift should not be used for LN.
    if synapse_params['nonlinearity'] == 'linear':
        response = response
    elif synapse_params['nonlinearity'] == 'relu':
        response = np.where(response <= 0, 0, response)

    def I_stim(t):
        inds = np.int_(np.floor((t - t_responses[0]) /
                       np.mean(np.diff(t_responses))))
        return response[inds] if 0 <= inds < len(t_responses) else 0

    def alpha_psp(g, t):
        tau_rise = synapse_params['tau_rise']
        tau_decay = synapse_params['tau_decay']
        return np.array([(I_stim(t) - g[0]) / tau_rise,
                         (g[0] - g[1]) / tau_decay])
    conductance = odeint(alpha_psp, [0, 0], t_responses)
    # Scale conductance, the first element is the solution for the rise, and
    # second is for the decay synaptic function, the main conductance.
    # Add an offset to avoid negative conductances
    # Spatial and temporal filter gains are normalized to the sum of the filter
    # so maximum and minimum should be maximum and minimum stimulus contrasts
    # which in turn are restricted to [-1, 1].
    main_g = synapse_params['amplitude'] * (conductance[:, 1] + 1) / 2
    # Apply nonlinearity, should tol = 1 / max_g?
    tol = 1e-9
    max_g = 1e9
    # if synapse_params['nonlinearity'] == 'linear':
    #     Rs = np.where(np.abs(main_g) < tol, max_g, 1 / main_g)
    # elif synapse_params['nonlinearity'] == 'linear_shift':
    #     main_g -= np.min(main_g)  # To avoid negatives.
    #     Rs = np.where(np.abs(main_g) < tol, max_g, 1 / main_g)
    # elif synapse_params['nonlinearity'] == 'relu':
    #     Rs = np.where(main_g <= tol, max_g, 1 / main_g)
    # Now that LN was applied before conductance, all conductances get same treatment.
    Rs = np.where(main_g <= tol, max_g, 1 / main_g)
    # Alternatively we could only offset the conductances above threshold for relu
    # and keep the rest at 0.
    return Rs


def get_response_from_RFs_stim(time_points, x_range, y_range, filter_distance_deg,
                               t_stim, stimulus, rf_params, synapse_params,
                               cell_name):
    # Get the pure low pass filter cells with a single (fast) time constant.
    if cell_name in ['Tm9', 'Mi4', 'Mi9', 'CT1']:
        # t_filter = tfilt.causal_filter(time_points, rf_params['tau_fast'])
        t_filter = tfilt.low_pass_filter(time_points, rf_params['tau_fast'])
        t_filter *= np.sign(rf_params['amplitude'])
    # elif cell_name in ['Tm1']:
    else:
        # Slow tau for high pass, fast tau for low pass.
        t_filter = tfilt.high_pass_filter(time_points, rf_params['tau_slow'])
        # t_filter = tfilt.low_plus_high_filter(time_points,
        #                                       rf_params['tau_fast'],
        #                                       rf_params['tau_slow'],
        #                                       rf_params['amplitude'])
        t_filter *= np.sign(rf_params['amplitude'])
    t_filter = np.flipud(t_filter)
    # t_filter = t_filter / np.sum(t_filter**2)
    # t_filter = t_filter / np.max(np.abs(t_filter))
    # artificially keep responses below 1 for contrasts in [-1 1]
    t_filter = t_filter / np.linalg.norm(t_filter, ord=2) / 6
    # t_filter = t_filter / np.abs(np.sum(t_filter)) / 3
    center_surround_filter = get_filter_hex_array(x_range, y_range,
                                                  filter_distance_deg,
                                                  rf_params)
    space_time_filter = center_surround_filter[..., np.newaxis]
    space_time_filter = space_time_filter * t_filter
    t_responses = t_stim[space_time_filter.shape[-1] - 1:]
    response = np.zeros((space_time_filter.shape[-2], stimulus.shape[-1]))
    for i_cell in np.arange(space_time_filter.shape[-2]):
        if rf_params['connectivity'][i_cell] != 0:
            # i_filter = space_time_filter[:, :, i_cell, :]
            # response = get_stim_response(i_filter, stimulus)
            response[i_cell, :] = get_stim_response_separable(
                center_surround_filter[:, :, i_cell],
                t_filter, stimulus)
        else:
            response[i_cell, :] = np.zeros(stimulus.shape[-1])
    return response, t_filter


# @mem.cache
def get_Rs_from_RFs_stim(time_points, x_range, y_range, filter_distance_deg,
                         t_stim, stimulus, rf_params, synapse_params,
                         cell_name):
    response, t_filter = get_response_from_RFs_stim(time_points,
        x_range, y_range, filter_distance_deg, t_stim, stimulus,
        rf_params, synapse_params, cell_name)
    # Repeated part just to get space_time_filter shape, which is
    # screen XY dims x 7 (ommatidia) x nPoints t_filter
    center_surround_filter = get_filter_hex_array(x_range, y_range,
                                                  filter_distance_deg,
                                                  rf_params)
    space_time_filter = center_surround_filter[..., np.newaxis]
    space_time_filter = space_time_filter * t_filter
    t_responses = t_stim[space_time_filter.shape[-1] - 1:]
    # initialize series resistance with empty, check if zero response yields
    # zero series resistance to save on integration time.
    # rs = np.empty((space_time_filter.shape[-2], stimulus.shape[-1] -
    #                space_time_filter.shape[-1] + 1))
    rs = np.empty((space_time_filter.shape[-2], stimulus.shape[-1]))
    for i_cell in np.arange(space_time_filter.shape[-2]):
        rs[i_cell, :] = get_Rs_from_stim(t_stim, response[i_cell, :], synapse_params)
    return rs, t_filter
