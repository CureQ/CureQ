
import numpy as np
import pandas as pd



def isi_lengths(spike_times, t_start, t_end):
    """ 
        Calculate the interspike intervals (ISIs) for a given list of spike times with t_start & t_end as auxilary spikes.

        In:  
            spike_times: list of spike timestamps (must be sorted)
            t_start: start time of the recording
            t_end: end time of the recording
        Out: 
            isi_lengths - ISI distance between spikes, or start and first spike

        Note: the only complexities are with the edges and N==1
    """

    # Total number of spikes
    n_spikes = len(spike_times)

    # If there are no spikes, the entire measurement is one interval
    if n_spikes == 0:
        return [t_end - t_start]

    # Calculate synthetic interval between t_start and the first spike
    if spike_times[0] > t_start:
        # Silence at the beginning
        if n_spikes > 1:
            interval_before_first_spike = max(spike_times[0] - t_start,
                                              spike_times[1] - spike_times[0])
        else:
            interval_before_first_spike = spike_times[0] - t_start
        i_start = 0
    else:
        # First spike is at or before t_start
        # Calculate interval between two spikes
        if n_spikes > 1:
            interval_before_first_spike = spike_times[1] - spike_times[0]
        else:
            interval_before_first_spike = t_start - spike_times[0]
        i_start = 1

    # Calculate synthetic interval between the last spike and t_end
    if spike_times[-1] < t_end:
        # Silence at the end
        if n_spikes > 1:
            interval_after_last_spike = max(t_end - spike_times[-1],
                                            spike_times[-1] - spike_times[-2])
        else:
            interval_after_last_spike = t_end - spike_times[0]
        i_end = n_spikes
    else:
        # Last spike is at or after t_end
        if n_spikes > 1:
            interval_after_last_spike = spike_times[-1] - spike_times[-2]
        else:
            interval_after_last_spike = spike_times[0] - t_end
        i_end = n_spikes - 1

    # Compute ISIs between spikes within the interval
    isi_core = [spike_times[i + 1] - spike_times[i] for i in range(i_start, i_end - 1)]

    # Combine start, core, and end intervals
    isi_lengths = [interval_before_first_spike] + isi_core + [interval_after_last_spike]

    return isi_lengths



def isi_distance(s1, s2, t_start, t_end, MRTS=0.0):
    """
    Compute the ISI-distance between two spike trains s1 and s2.

    Parameters:
    - s1: spike times for train 1
    - s2: spike times for train 2
    - t_start: start time of the interval or measurement
    - t_end: end time of the intervalor measurement
    - MRTS: Minimum Relevant Time Scale (used for adaptive ISI-distance)

    Returns:
    - isi_distance_value: normalized ISI-distance between the two trains
    - df_isi_time: pandas DataFrame with ISI values over the time interval
    """

    isi_value = 0.0  # Cumulative ISI-distance
    start_times = []  # To store start time of each ISI segment
    list_isi = []     # To store ISI differences at each time segment

    # Number of spikes in spiketrains
    Nspikes_1 = len(s1)
    Nspikes_2 = len(s2)

    spikes_1 = np.asarray(s1)
    spikes_2 = np.asarray(s2)

    # Determine ISI before first spike in spiketrain 1 
    if spikes_1[0] > t_start:
        if Nspikes_1 > 1:
            # Use the bigest interval (between t-start to first spike or spike 1 to spike 2.)
            ISI1 = max(spikes_1[0] - t_start, spikes_1[1] - spikes_1[0])
        else:
            ISI1 = spikes_1[0] - t_start
        index1 = -1  # No spike yet before t_start
    else:
        # Calculate isi normal between two spikes
        ISI1 = (spikes_1[1] - spikes_1[0]) if Nspikes_1 > 1 else t_end - spikes_1[0]
        index1 = 0  # First spike is before or at t_start

    # Determine ISI before first spike in spiketrain 2 
    if spikes_2[0] > t_start:
         # Use the bigest interval (between t-start to first spike or spike 1 to spike 2.) 
        ISI2 = max(spikes_2[0] - t_start, spikes_2[1] - spikes_2[0]) if Nspikes_2 > 1 else spikes_2[0] - t_start
        index2 = -1
    else:
        # Calculate isi normal between two spikes
        ISI2 = (spikes_2[1] - spikes_2[0]) if Nspikes_2 > 1 else t_end - spikes_2[0]
        index2 = 0

    # Initialize time tracking and ISI calculation 
    last_t = t_start  # Last processed time
    curr_isi = abs(ISI1 - ISI2) / max(MRTS, max(ISI1, ISI2))  # Initial normalized ISI difference
    index = 1  # Counter for debugging or tracking steps

    # Main loop to process all spikes in order
    while index1 + index2 < Nspikes_1 + Nspikes_2 - 2:
        # Case 1: next spike is from spike train 1
        if (index1 < Nspikes_1 - 1) and ((index2 == Nspikes_2 - 1) or (spikes_1[index1 + 1] < spikes_2[index2 + 1])):
            index1 += 1
            curr_t = spikes_1[index1]

            if index1 < Nspikes_1 - 1:
                ISI1 = spikes_1[index1 + 1] - spikes_1[index1]
            else:
                ISI1 = max(t_end - spikes_1[index1], ISI1) if Nspikes_1 > 1 else t_end - spikes_1[index1]

        # Case 2: next spike is from spike train 2
        elif (index2 < Nspikes_2 - 1) and ((index1 == Nspikes_1 - 1) or (spikes_1[index1 + 1] > spikes_2[index2 + 1])):
            index2 += 1
            curr_t = spikes_2[index2]

            if index2 < Nspikes_2 - 1:
                ISI2 = spikes_2[index2 + 1] - spikes_2[index2]
            else:
                ISI2 = max(t_end - spikes_2[index2], ISI2) if Nspikes_2 > 1 else t_end - spikes_2[index2]

        # Case 3: simultaneous spike in both trains
        else:
            index1 += 1
            index2 += 1
            curr_t = spikes_1[index1]  # Equal timestamps assumed

            if index1 < Nspikes_1 - 1:
                ISI1 = spikes_1[index1 + 1] - spikes_1[index1]
            else:
                ISI1 = max(t_end - spikes_1[index1], ISI1) if Nspikes_1 > 1 else t_end - spikes_1[index1]

            if index2 < Nspikes_2 - 1:
                ISI2 = spikes_2[index2 + 1] - spikes_2[index2]
            else:
                ISI2 = max(t_end - spikes_2[index2], ISI2) if Nspikes_2 > 1 else t_end - spikes_2[index2]

        # Save results for plotting or further analysis
        start_times.append(last_t)
        list_isi.append(curr_isi)

        # Accumulate weighted ISI difference over this interval
        isi_value += curr_isi * (curr_t - last_t)

        # Update current ISI value for next interval
        curr_isi = abs(ISI1 - ISI2) / max(MRTS, max(ISI1, ISI2))
        last_t = curr_t
        index += 1

    # Final interval from last spike to t_end
    isi_value += curr_isi * (t_end - last_t)
    start_times.append(last_t)
    list_isi.append(curr_isi)

    # Return both final distance value and time-resolved ISI differences 
    df_isi_time = pd.DataFrame({
        'time_start': start_times,
        'isi': list_isi
    })

    return isi_value / (t_end - t_start), df_isi_time

def default_thresh(train_list, t_start, t_end):
    """
    Calculates a default threshold based on the interspike intervals (ISIs) from all spike trains, intended for use in the ISI-distance calculation

    Parameters:
    - train_list: List of spiketrains (each is a list of spiketimes).
    - t_start: Start time of the interval to consider.
    - t_end: End time of the interval to consider.

    Returns:
    To Do:
    Math fucntion
    - The root mean square (RMS) of all ISIs across the spike trains. This can be used as a default threshold for similarity or distance measures.
    """
    spike_pool = []

    # Collect all ISIs from each spike train
    for train in train_list:
        spike_pool += isi_lengths(train, t_start, t_end)  # isi_lengths returns list of ISIs for one train

    spike_pool = np.array(spike_pool)

    # Compute and return the root mean square of the ISIs
    return np.sqrt(np.mean(spike_pool ** 2))


def reconcile_spike_trains(spike_trains, t_start, t_end):
    """
    Clean and align spike trains by removing duplicate spikes and clipping spikes outside the time window.

    Parameters:
    - spike_trains: List of spike trains (each is a list of spike times).
    - t_start: Start time of the valid interval.
    - t_end: End time of the valid interval.

    Returns:
    - A list of cleaned spike trains with:
        - Duplicate spike times removed
        - Only spikes within (t_start, t_end) included (with small tolerance for numerical errors)
    """
    Eps = 1e-6  # Small tolerance for floating-point precision issues
    new_spike_trains = []

    for spikes in spike_trains:
        # Remove duplicate spike times
        spikes = np.unique(spikes)

        # Keep only spikes within the valid time window, allowing a small tolerance
        filtered_spikes = [t for t in spikes if t_start - Eps < t < t_end + Eps]

        # Sort the filtered spikes and append to the cleaned list
        new_spike_trains.append(sorted(filtered_spikes))

    return new_spike_trains

