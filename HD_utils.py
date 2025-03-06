from scipy.signal import hilbert
import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.ndimage import gaussian_filter1d
import h5py

def get_phase(filtered_lfp):
    """
    get the instantaneous phase of the filtered lfp signal
    """
    
    analytic_signal = hilbert(filtered_lfp)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # wrap the instantaneous_phase to -pi and pi
    instantaneous_phase = np.mod(instantaneous_phase + np.pi, 2 * np.pi)
    
    return instantaneous_phase

def compute_spike_array(spike_times, timestamps):
    """
    Compute the spike array from spike times and timestamps.
    """
    
    #for loop is too slow:
    # spike_array = np.zeros(timestamps.shape)
    # for spike_time in spike_times:
    #     # Find the index of the closest time in t to this spike_time
    #     idx = np.argmin(np.abs(timestamps - spike_time))
    #     spike_array[idx] += 1
    
    # Assume t is sorted. If not, you must sort it first.
    indices = np.searchsorted(timestamps, spike_times, side='right')

    # Ensure no out-of-bounds indices
    indices = np.clip(indices, 0, len(timestamps) - 1)

    # Adjust indices to always refer to the nearest timestamp
    # If the index points to the start of the array, no need to adjust
    # If not, check if the previous index is closer
    prev_close = (indices > 0) & (np.abs(timestamps[indices - 1] - spike_times) < np.abs(timestamps[indices] - spike_times))
    indices[prev_close] -= 1

    spike_array = np.zeros_like(timestamps)
    np.add.at(spike_array, indices, 1) #very fast!
    
    return spike_array

def compute_cross_correlation(spikearray, lfp_theta, sfreq=250, fmin=7, fmax=11):
    '''
    Compute the cross-correlation between the power spectra of a spike train of a cell and the lfp_theta.
    Input:
    - spikearray: Array of spike times for the cell
    - eeg: EEG signal
    '''

    psd_cell, freqs_cell = psd_array_multitaper(spikearray, sfreq=sfreq, fmin=fmin, fmax=fmax, adaptive=True, normalization='full', verbose=False) #, n_jobs=-1,

    psd_theta, freqs_theta = psd_array_multitaper(lfp_theta, sfreq=sfreq, fmin=fmin, fmax=fmax, adaptive=True, normalization='full', verbose=False)

    corr = np.correlate(psd_cell, psd_theta, mode='full')
    lags = np.arange(-len(psd_theta) + 1, len(psd_theta))* (freqs_theta[1] - freqs_theta[0])

    #normalize corr with max value
    corr = corr / np.max(corr)

    #downsample shifted_freqs, corr to 100 dimensions (actually it is 101 dimensions)
    lags = lags[::len(lags)//100]
    corr = corr[::len(corr)//100]
    
    #if the dimension is <101 , pad it with zeros, and if the dimension is >101, cut it
    if len(lags) < 101:
        lags = np.pad(lags, (0, 101-len(lags)), 'constant', constant_values=(0, 0))
        corr = np.pad(corr, (0, 101-len(corr)), 'constant', constant_values=(0, 0))
    else:
        lags = lags[:101]
        corr = corr[:101]

    # # # Cross-correlation between psd_cell and psd_theta by shifting psd_theta
    # half_shift = len(freqs_theta) // 2
    # corr = np.zeros(2 * half_shift + 1)  # Array to hold correlation values

    # # Perform shifts to the left and right
    # for i in range(-half_shift, half_shift + 1):
    #     shifted_psd_theta = np.roll(psd_theta, i)
    #     corr[i + half_shift] = np.corrcoef(psd_cell, shifted_psd_theta)[0, 1]

    # # Create a new frequency axis for plotting the correlation
    # shifted_freqs = np.arange(-half_shift, half_shift + 1) * (freqs_theta[1] - freqs_theta[0])
    
    # print(lags[0],lags[-1])
    
    # #downsample shifted_freqs, corr to 100 dimensions (actually it is 101 dimensions)
    # shifted_freqs = shifted_freqs[::len(shifted_freqs)//100]
    # corr = corr[::len(corr)//100]
    
    # #if the dimension is <101 , pad it with zeros, and if the dimension is >101, cut it
    # if len(shifted_freqs) < 101:
    #     shifted_freqs = np.pad(shifted_freqs, (0, 101-len(shifted_freqs)), 'constant', constant_values=(0, 0))
    #     corr = np.pad(corr, (0, 101-len(corr)), 'constant', constant_values=(0, 0))
    # else:
    #     shifted_freqs = shifted_freqs[:101]
    #     corr = corr[:101]
    
    return lags, corr


def calculate_angular_speed(poh, pot, sigma=5):
    """
    Calculate the head rotation speed (angular speed) from head direction and time data.

    Parameters:
    poh : np.array
        Array of head direction angles in degrees.
    pot : np.array
        Array of time points corresponding to the head direction angles.
    sigma : float
        Standard deviation for Gaussian kernel used in smoothing.

    Returns:
    angular_speed : np.array
        Array of angular speed values.
    """

    # Ensure head direction angles are in radians
    poh_rad = np.deg2rad(poh)

    # Convert angles to complex exponential form
    poh_complex = np.exp(1j * poh_rad)

    # Apply Gaussian smoothing to real and imaginary parts separately
    poh_smooth_real = gaussian_filter1d(poh_complex.real, sigma=sigma)
    poh_smooth_imag = gaussian_filter1d(poh_complex.imag, sigma=sigma)

    # Convert back to angles
    poh_smooth = np.angle(poh_smooth_real + 1j * poh_smooth_imag)

    # Calculate the time differences
    dt = np.diff(pot)

    # Calculate the angle differences
    dtheta = np.diff(poh_smooth)
    
    #deal with the jump between 0 and 2pi
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi
    
    # Calculate the angular speed
    angular_speed = dtheta / dt
    
    angular_speed = angular_speed.flatten()

    #add one element to the beginning of angular_speed to make it the same length as poh
    angular_speed = np.insert(angular_speed, 0, angular_speed[0])

    return angular_speed

def find_continuous_periods_between_low_high(angular_speed, pot, speed_low, speed_high, duration_threshold=1.0):
    if speed_low > 0:
        # Identify where angular speed bwteen low and high
        above_threshold = (angular_speed > speed_low) & (angular_speed < speed_high)
    elif speed_low < 0:
        # Identify where angular speed is below the threshold
        above_threshold = (angular_speed < speed_low) & (angular_speed > speed_high)
    else:
        raise ValueError("Speed threshold must be non-zero")

    # Find the indices where the state changes
    change_indices = np.diff(above_threshold.astype(int), prepend=0, append=0)

    # Identify the start and end indices of the segments
    start_indices = np.where(change_indices == 1)[0]
    end_indices = np.where(change_indices == -1)[0]

    # Ensure there is a matching number of start and end indices
    if len(start_indices) > len(end_indices):
        end_indices = np.append(end_indices, len(angular_speed) - 1)
    elif len(end_indices) > len(start_indices):
        start_indices = np.insert(start_indices, 0, 0)

    continuous_periods = []
    for start, end in zip(start_indices, end_indices):
        if end >= len(pot):
            end = len(pot) - 1
        duration = pot[end] - pot[start]
        if duration >= duration_threshold:
            continuous_periods.append((pot[start], pot[end]))

    return continuous_periods

def find_continuous_periods(angular_speed, pot, speed_threshold=0.5, duration_threshold=1.0):
    if speed_threshold > 0:
        # Identify where angular speed exceeds the threshold
        above_threshold = angular_speed > speed_threshold
    elif speed_threshold < 0:
        # Identify where angular speed is below the threshold
        above_threshold = angular_speed < speed_threshold
    else:
        raise ValueError("Speed threshold must be non-zero")

    # Find the indices where the state changes
    change_indices = np.diff(above_threshold.astype(int), prepend=0, append=0)

    # Identify the start and end indices of the segments
    start_indices = np.where(change_indices == 1)[0]
    end_indices = np.where(change_indices == -1)[0]

    # Ensure there is a matching number of start and end indices
    if len(start_indices) > len(end_indices):
        end_indices = np.append(end_indices, len(angular_speed) - 1)
    elif len(end_indices) > len(start_indices):
        start_indices = np.insert(start_indices, 0, 0)

    continuous_periods = []
    for start, end in zip(start_indices, end_indices):
        if end >= len(pot):
            end = len(pot) - 1
        duration = pot[end] - pot[start]
        if duration >= duration_threshold:
            continuous_periods.append((pot[start], pot[end]))

    return continuous_periods

def get_rotation_spiketimes(poh, pot, pspt, config):

    
    speed_threshold = config['speed_threshold']
    duration_threshold = config['duration_threshold']
    speed_smooth_sigma = config['speed_smooth_sigma']
    
    #calculate angular speed
    angular_speed = calculate_angular_speed(poh, pot, sigma=speed_smooth_sigma)
    
    #CCW
    continuous_periods_CCW = find_continuous_periods(angular_speed, pot, speed_threshold=-speed_threshold, duration_threshold=duration_threshold)
    indx = []
    for i in range(len(continuous_periods_CCW)):
        indx.extend(np.where((pspt > continuous_periods_CCW[i][0]) & (pspt < continuous_periods_CCW[i][1]))[0])
        
    spike_times_CCW = pspt[indx]
    
    #CW 
    continuous_periods_CW = find_continuous_periods(angular_speed, pot, speed_threshold=speed_threshold, duration_threshold=duration_threshold)
    #keep the index when pspt is within continuous_periods
    indx = [] 
    for i in range(len(continuous_periods_CW)):
        indx.extend(np.where((pspt > continuous_periods_CW[i][0]) & (pspt < continuous_periods_CW[i][1]))[0])
    
    spike_times_CW = pspt[indx]
    
    #combine CCW and CW togther
    spike_times_combined = np.concatenate((spike_times_CCW, spike_times_CW))
    
    return spike_times_combined


#%% code for loading data
def load_data(ratname, file_path, trialname='light1'):
    # Load the .mat file
    with h5py.File(file_path, 'r') as f:

        # Access the sdata structure
        sdata = f['sdata']

        # Check if trialname exists in sdata
        if trialname not in sdata:
            print(f"{trialname} not found in {file_path}")
            return None

        # Accessing the light1 group and its datasets
        light1 = sdata[trialname]
        ppox = np.array(light1['pox'])
        ppoy = np.array(light1['poy'])
        pot = np.array(light1['pot'])
        poh = np.array(light1['poh'])
        pov = np.array(light1['pov'])
        f0 = np.array(light1['F0'])
        sintcptFreqy = np.array(light1['sintcptFreqy'])
        global_freq = np.array(light1['maxFreq_type1'])

        # Extract all cells containing the name 'R222'
        cell_names = [key for key in sdata.keys() if ratname in key]
        #print(f"Cell Names: {cell_names}")

        # Initialize dictionary to store data for all cells
        cells_data = {}

        # Iterate over each cell name and extract data
        for cell_name in cell_names:
            part_now = trialname  # Assuming trialname is the part_now equivalent

            pspx = np.array(sdata[cell_name][part_now]['spx'])
            pspy = np.array(sdata[cell_name][part_now]['spy'])
            pspt = np.array(sdata[cell_name][part_now]['spt'])
            pspv = np.array(sdata[cell_name][part_now]['spv'])
            psph = np.array(sdata[cell_name][part_now]['sph'])
            pspm = np.array(sdata[cell_name][part_now]['spm'])
            pval = np.array(sdata[cell_name][part_now]['pval'])
            spike_phase = np.array(sdata[cell_name][part_now]['spike_phase'])
            autocorrelogram = np.array(sdata[cell_name][part_now]['theta_train_long2'])
            hd_mean = np.array(sdata[cell_name][part_now]['hd_mean'])
            hd_std = np.array(sdata[cell_name][part_now]['hd_stdev'])
            tune_width = np.array(sdata[cell_name][part_now]['tuning_width'])
            intrinsic_freq = np.array(sdata[cell_name][part_now]['intrinsic_theta_frequency'])
            hd_rayleigh = np.array(sdata[cell_name][part_now]['hd_rayleigh'])
            hd_rayleigh_shuffle_95 = np.array(sdata[cell_name][part_now]['HDrayleigh_shuff_val'])
            hd_rayleigh_shuffle_99 = np.array(sdata[cell_name][part_now]['HDrayleigh_shuff_val2'])
            peak_firingrate = np.array(sdata[cell_name][part_now]['hd_max_frate'])
            ATI_widthshift = np.array(sdata[cell_name][part_now]['RLdata']['optimal_width_shift'])
            ATI_cwccw = np.array(sdata[cell_name][part_now]['RLdata']['ATI'])

            # Extract and decode cell_type
            cell_type_num = np.array(sdata[cell_name][part_now]['cell_type_num'])
            cell_type_array = np.array(sdata[cell_name][part_now]['thetacell_type'])
            cell_type = ''.join([chr(ascii_val[0]) for ascii_val in cell_type_array])

            # Store the data for this cell
            cells_data[cell_name] = {
                'pspx': pspx,
                'pspy': pspy,
                'pspt': pspt,
                'pspv': pspv,
                'psph': psph,
                'pspm': pspm,
                'pval': pval,
                'spike_phase': spike_phase,
                'autocorrelogram': autocorrelogram,
                'hd_mean': hd_mean,
                'hd_std': hd_std,
                'tune_width': tune_width,
                'intrinsic_freq': intrinsic_freq,
                'cell_type': cell_type,
                'cell_type_num': cell_type_num,
                'hd_rayleigh': hd_rayleigh,
                'hd_rayleigh_shuffle_95': hd_rayleigh_shuffle_95,
                'hd_rayleigh_shuffle_99': hd_rayleigh_shuffle_99,
                'peak_fr': peak_firingrate,
                'ATI_widthshift': ATI_widthshift,
                'ATI_cwccw': ATI_cwccw
            }

        # Create a dictionary to store all the data
        data_dict = {
            'ppox': ppox,
            'ppoy': ppoy,
            'pot': pot,
            'poh': poh,
            'pov': pov,
            'f0': f0,
            'sintcptFreqy': sintcptFreqy,
            'global_freq': global_freq,
            'cell_names': cell_names,
            'cells_data': cells_data
        }

        return data_dict


