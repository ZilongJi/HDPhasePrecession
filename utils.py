import numpy as np
import scipy.io as sio


def compute_spike_array(spike_times, timestamps):
    """
    Compute the spike array from spike times and timestamps.
    """
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


def load_data(mousetrial, brainarea='adn', celltype='hd'):
    
    '''
    Load the data for a given mouse trial
    Inputs:
    mousetrial: string, the name of the mouse trial
    brainarea: string, the brain area to filter the cells
    celltype: string, the cell type to filter the cells
    '''
    
    #load time and head direction
    file = 'Angle.mat'
    
    data = sio.loadmat(mousetrial + '/' + file, struct_as_record=False, squeeze_me=True)
    ang = data.get('ang')
    t = ang.t
    angle_data = ang.data
    
    #load brain area
    file = 'BrainArea.mat'
    
    data = sio.loadmat(mousetrial + '/' + file, struct_as_record=False, squeeze_me=True)
    adn = data.get('adn')
    pos = data.get('pos')
    
    #load cell types
    file = 'CellTypes.mat'
    
    data = sio.loadmat(mousetrial + '/' + file, struct_as_record=False, squeeze_me=True)
    hd = data.get('hd')
    
    #load spike data
    file = 'SpikeData.mat'

    data = sio.loadmat(mousetrial + '/' + file, struct_as_record=False, squeeze_me=True)
    S = data.get('S')
    C = S.C
    spike_times = [cell.tsd.t for cell in C]
    
    #filter out spike times fall out of the range of timestamps
    for i in range(len(spike_times)):
        valid_mask = (spike_times[i] >= t[0]) & (spike_times[i] <= t[-1])
        spike_times[i] = spike_times[i][valid_mask]
    
    return {'t': t, 'angle_data': angle_data, 'adn': adn, 'pos': pos, 'hd': hd, 'spike_times': spike_times}

def load_anticipation(mousetrial):
    
    '''
    Load the data for a given mouse trial
    Inputs:
    mousetrial: string, the name of the mouse trial
    brainarea: string, the brain area to filter the cells
    celltype: string, the cell type to filter the cells
    '''
    
    #load time and head direction
    file = 'HdTuning_AnticipCorr.mat'
    
    data = sio.loadmat(mousetrial + '/' + file, struct_as_record=False, squeeze_me=True)
    
    lagVec = data.get('lagVec')
    hdinfo_lag = data.get('hdInfo_lag')
    
    peak_lag = []
    #get the peak lag for each cell
    for i in range(hdinfo_lag.shape[1]):
        lag = lagVec[hdinfo_lag[:,i].argmax()]
        peak_lag.append(lag)
        
    return peak_lag