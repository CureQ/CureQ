import numpy as np
import pandas as pd
from CureQ.core.ISI_distance import *

def get_min_dist(spike_time, spike_train, N_spikes, start_index, t_start, t_end):
    """
    Returns the minimal distance of the given spike to the other spiketrain.

    Input
        spike_time: spike_time of spike from spike_train a.
        spik_train: The comparason spike_train b.
        N_spikes: Number of spikes from spike_train_b
        Start_index: First index / spike that will be comparased by the spiketrain. # Note: Hiermee kan ik het sneller laten werken
        t_start, t_end: Beginning and End time of measurement

    Output
        clostest_OSI: Closests outer spiketrain distance value

    """

    # Calculate distance between spike_time and start time.
    distance = abs(spike_time - t_start)

    # Make sure first index is above zero, else set to zero
    if start_index < 0:
        start_index = 0

    # Loop through all index while under the length of the other spike_train
    while start_index < N_spikes:
        # Calculate the difference between (new)spike of spiketrainb and the original spike of spiketrain a.
        new__iso_difference = abs(spike_time - spike_train[start_index])

        # If the di
        if new__iso_difference > distance:
            return distance
        else:
            # When the new distance is smaller it becomes the 'smallest' distance.
            distance = new__iso_difference

        start_index += 1

    # Check if the distance to t_end is not smaller.
    new__iso_difference = abs(t_end - spike_time)
    if new__iso_difference > distance:
        return distance
    else:
        return new__iso_difference



def dist_at_t(isi1, isi2, s1, s2, MRTS, RI):
    """
    Compute instantaneous Spike Distance
        Input
            isi1, isi2 - spike time differences around current times in each trains
            s1, s2 - weighted spike time differences between trains
            MRTS - minimum relevant time scale (0 for legacy logic)
            RI - Rate Independent Adaptive spike distance 
                 (False for legacy SPIKE distance)
        Out: 
            Spike Distance at current time
    """
    meanISI = 0.5 * (isi1 + isi2)   
    limitedISI = max(MRTS, meanISI)

    if RI:
        return 0.5 * (s1 + s2) / limitedISI
    else:
        return 0.5 * (s1 * isi2 + s2 * isi1) / (meanISI * limitedISI)


def spike_distance(spike_times1, spike_times2, t_start, t_end, MRTS=0.0, RI=0):


    # Get number of spikes in each spike train
    num_spikes1 = len(spike_times1)
    num_spikes2 = len(spike_times2)

    # Ensure there is at least one spike in each spike train
    assert num_spikes1 > 0
    assert num_spikes2 > 0

    spike_distance = 0.0
    start_times = []  # To store start time of each ISI segment
    list_SPIKE = []     # To store ISI differences at each time segment
    last_time = t_start  # Initialize tracking of current time interval

    aux1 = np.empty(2)
    aux2 = np.empty(2)

    # Estimate spike train boundaries for interpolation/extrapolation
    aux1[0] = min(t_start, 2 * spike_times1[0] - spike_times1[1]) if num_spikes1 > 1 else t_start
    aux1[1] = max(t_end, 2 * spike_times1[-1] - spike_times1[-2]) if num_spikes1 > 1 else t_end

    aux2[0] = min(t_start, 2 * spike_times2[0] - spike_times2[1]) if num_spikes2 > 1 else t_start
    aux2[1] = max(t_end, 2 * spike_times2[-1] - spike_times2[-2]) if num_spikes2 > 1 else t_end

    # Set up previous time for each train, handling edge conditions
    previous_time1 = t_start if spike_times1[0] == t_start else aux1[0]
    previous_time2 = t_start if spike_times2[0] == t_start else aux2[0]

    # Initialize first spike interval and distances for spike train 1
    if spike_times1[0] > t_start:
        next_time1 = spike_times1[0]
        dt_next1 = get_min_dist(next_time1, spike_times2, num_spikes2, 0, aux2[0], aux2[1])
        isi1 = max(next_time1 - t_start, spike_times1[1] - spike_times1[0]) if num_spikes1 > 1 else next_time1 - t_start
        dt_prev1 = dt_next1
        s1 = dt_prev1
        index1 = -1
    else:
        next_time1 = spike_times1[1] if num_spikes1 > 1 else t_end
        dt_next1 = get_min_dist(next_time1, spike_times2, num_spikes2, 0, aux2[0], aux2[1])
        dt_prev1 = get_min_dist(previous_time1, spike_times2, num_spikes2, 0, aux2[0], aux2[1])
        isi1 = next_time1 - spike_times1[0]
        s1 = dt_prev1
        index1 = 0

    # Initialize first spike interval and distances for spike train 2
    if spike_times2[0] > t_start:
        next_time2 = spike_times2[0]
        dt_next2 = get_min_dist(next_time2, spike_times1, num_spikes1, 0, aux1[0], aux1[1])
        dt_prev2 = dt_next2
        isi2 = max(next_time2 - t_start, spike_times2[1] - spike_times2[0]) if num_spikes2 > 1 else next_time2 - t_start
        s2 = dt_prev2
        index2 = -1
    else:
        next_time2 = spike_times2[1] if num_spikes2 > 1 else t_end
        dt_next2 = get_min_dist(next_time2, spike_times1, num_spikes1, 0, aux1[0], aux1[1])
        dt_prev2 = get_min_dist(previous_time2, spike_times1, num_spikes1, 0, aux1[0], aux1[1])
        isi2 = next_time2 - spike_times2[0]
        s2 = dt_prev2
        index2 = 0

    # Compute initial value of the distance function
    y_start = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
    index = 1

    # Main loop to compute spike distance over the time window
    start_times.append(t_start)
    list_SPIKE.append(y_start)

    while index1 + index2 < num_spikes1 + num_spikes2 - 2:
        if (index1 < num_spikes1 - 1) and (next_time1 < next_time2 or index2 == num_spikes2 - 1):
            # Advance spike train 1
            index1 += 1
            s1 = dt_next1 * (next_time1 - previous_time1) / isi1
            dt_prev1 = dt_next1
            previous_time1 = next_time1
            next_time1 = spike_times1[index1 + 1] if index1 < num_spikes1 - 1 else aux1[1]

            current_time = previous_time1
            # Interpolate s2 to match time of spike train 1
            s2 = (dt_prev2 * (next_time2 - current_time) + dt_next2 * (current_time - previous_time2)) / isi2
            y_end = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            start_times.append(current_time)
            list_SPIKE.append(y_end)
            
            spike_distance += 0.5 * (y_start + y_end) * (current_time - last_time)

            if index1 < num_spikes1 - 1:
                dt_next1 = get_min_dist(next_time1, spike_times2, num_spikes2, index2, aux2[0], aux2[1])
                isi1 = next_time1 - previous_time1
                s1 = dt_prev1
            else:
                dt_next1 = dt_prev1
                isi1 = max(t_end - spike_times1[-1], spike_times1[-1] - spike_times1[-2]) if num_spikes1 > 1 else t_end - spike_times1[-1]
                s1 = dt_prev1
            y_start = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            
            start_times.append(t_start)
            list_SPIKE.append(y_start)
        elif (index2 < num_spikes2 - 1) and (next_time1 > next_time2 or index1 == num_spikes1 - 1):
            # Advance spike train 2
            index2 += 1
            s2 = dt_next2 * (next_time2 - previous_time2) / isi2
            dt_prev2 = dt_next2
            previous_time2 = next_time2
            next_time2 = spike_times2[index2 + 1] if index2 < num_spikes2 - 1 else aux2[1]

            current_time = previous_time2
            s1 = (dt_prev1 * (next_time1 - current_time) + dt_next1 * (current_time - previous_time1)) / isi1
            y_end = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            start_times.append(current_time)
            list_SPIKE.append(y_end)
            spike_distance += 0.5 * (y_start + y_end) * (current_time - last_time)

            if index2 < num_spikes2 - 1:
                dt_next2 = get_min_dist(next_time2, spike_times1, num_spikes1, index1, aux1[0], aux1[1])
                isi2 = next_time2 - previous_time2
                s2 = dt_prev2
            else:
                dt_next2 = dt_prev2
                isi2 = max(t_end - spike_times2[-1], spike_times2[-1] - spike_times2[-2]) if num_spikes2 > 1 else t_end - spike_times2[-1]
                s2 = dt_prev2
            y_start = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            start_times.append(t_start)
            list_SPIKE.append(y_start)

        else:
            # Simultaneous spike or synchronization event
            index1 += 1
            index2 += 1
            previous_time1 = next_time1
            previous_time2 = next_time2
            dt_prev1 = 0.0
            dt_prev2 = 0.0
            current_time = next_time1
            y_end = 0.0
            start_times.append(current_time)
            list_SPIKE.append(y_end)
            spike_distance += 0.5 * (y_start + y_end) * (current_time - last_time)
            y_start = 0.0

            if index1 < num_spikes1 - 1:
                next_time1 = spike_times1[index1 + 1]
                dt_next1 = get_min_dist(next_time1, spike_times2, num_spikes2, index2, aux2[0], aux2[1])
                isi1 = next_time1 - previous_time1
            else:
                next_time1 = aux1[1]
                dt_next1 = dt_prev1
                isi1 = max(t_end - spike_times1[-1], spike_times1[-1] - spike_times1[-2]) if num_spikes1 > 1 else t_end - spike_times1[-1]

            if index2 < num_spikes2 - 1:
                next_time2 = spike_times2[index2 + 1]
                dt_next2 = get_min_dist(next_time2, spike_times1, num_spikes1, index1, aux1[0], aux1[1])
                isi2 = next_time2 - previous_time2
            else:
                next_time2 = aux2[1]
                dt_next2 = dt_prev2
                isi2 = max(t_end - spike_times2[-1], spike_times2[-1] - spike_times2[-2]) if num_spikes2 > 1 else t_end - spike_times2[-1]
                
            
        
        index += 1
        last_time = current_time

    # Final segment from last spike to end of interval
    s1 = dt_next1
    s2 = dt_next2
    y_end = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)

    start_times.append(t_end)
    list_SPIKE.append(y_end)

    # ---- Return both final distance value and time-resolved ISI differences ----
    df_spike = pd.DataFrame({
        'time_start': start_times,
        'isi': list_SPIKE
    })
    
    spike_distance += 0.5 * (y_start + y_end) * (t_end - last_time)

    # Normalize distance by time duration
    return spike_distance / (t_end - t_start), df_spike