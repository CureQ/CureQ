from pathlib import Path
import os
import warnings
import functools

import numpy as np
import pandas as pd
import h5py
import itertools
from statsmodels.tsa.stattools import pacf
from CureQ.core.ISI_distance import *
from CureQ.core._SPIKE_distance import *

def silence_runtime_warnings(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(*args, **kwargs)
    
    return wrapper

@silence_runtime_warnings
def electrode_features(well, parameters):
    """
    Calculate electrode features for all electrodes in a well.

    Parameters
    ----------
    well : int
        The well for which the electrode features are to be calculated.
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed

    Returns
    -------
    features_df : pandas Dataframe
        Dataframe containing electrodes for rows and features for columns.
    
    """
    
    # Define all lists for the features
    wells_df, electrodes_df, spikes, mean_firingrate, mean_ISI, median_ISI, ratio_median_over_mean, IIV, CVI, ISIPACF  = [], [], [], [], [], [], [], [], [], []
    burst_amnt, avg_burst_len, burst_var, burst_CV, mean_IBI, IBI_var, IBI_CV= [], [], [], [], [], [], []
    intraBFR, mean_spikes_per_burst, isolated_spikes, MAD_spikes_per_burst = [], [], [], []
    SCB_rate, IBIPACF, mean_spike_amplitude, median_spike_amplitude, cv_spike_amplitude = [], [], [], [], []
    
    # Calculate which electrodes belong to the well
    electrodes=np.arange((well-1)*parameters['electrode amount'], well*parameters['electrode amount'])

    # Loop through all the measured electrodes
    for electrode in electrodes:
        # Calculate current well number and append to list
        well_nr = round(electrode / parameters['electrode amount'] + 0.505)
        wells_df.append(well_nr)

        # Calculate current MEA electrode number and append to list
        electrode_nr = electrode % parameters['electrode amount'] + 1
        electrodes_df.append(electrode_nr)
        
        # Load in the files
        output_hdf_file=parameters['output hdf file']

        with h5py.File(output_hdf_file, 'r') as f:
            spikedata=f[f"spike_values/well_{well_nr}_electrode_{electrode_nr}_spikes"]
            spikedata=spikedata[:]
            burst_cores=f[f"burst_values/well_{well_nr}_electrode_{electrode_nr}_burst_cores"]
            burst_cores=burst_cores[:]
            burst_spikes=f[f"burst_values/well_{well_nr}_electrode_{electrode_nr}_burst_spikes"]
            burst_spikes=burst_spikes[:]

        '''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Calculate your own features here
        Step by step guide:
        1. Initialize a list prior to this for-loop, you will be adding the calculated feature of every electrode to this list
        2. You have access to the following variables after this line of code:
            spikedata = contains a row for every spike that has been detected, each row has 3 columns:
                spikedata[spike, 0] = the x-value of the spike, aka the time it was registered. Value is given in seconds
                spikedata[spike, 1] = the y value of the spike, aka the amplitude of the spike
                spikedata[spike, 2] = the python index where the spike was found in the list of raw voltage values
            burst_cores = The location of the single channel bursts that were detected in the electrodes, this array contains a row for each burst, each containing 5 columns:
                burst_cores[burst, 0] = the the start of the burst, given in seconds
                burst_cores[burst, 1] = the end of the burst, given in seconds
                burst_cores[burst, 2] = python list index of the start of the burst
                burst_cores[burst, 3] = python list index of the end of the burst
                burst_cores[burst, 4] = burst ID. Each burst is given its own unique (to the electrode) ID,
                    this allows us to couple bursts to spikes contained in bursts, which are located in the variable "burst_spikes"
            burst_spikes = contains information of all the spikes that are found in the single channel bursts.
            contains a row for every single spike in an electrode that is part of a burst, each row contains 4 columns
                burst_spikes[spike, 0:3] = exactly the same as spikedata^
                burst_spikes[spike, 3] = The burst ID of the burst that it participates in
        3. Calculate the feature. If you calculate it in this for-loop, it will automatically be calculated for each of the electrodes
           Warning: sometimes it is not possible to calculate a features due to a lack of detected bursts/spikes.
           always check if there are enough values to calculate the feature using an if-statement, if there are not, append "float("NaN")" to the list
        4. Append your feature to the list
        5. At the end of this function, add your list to the pandas dataframe, and give the column the correct name.
           Now if you run the analysis again, you should see your own feature in the "Features.csv" output file
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''

        """Calculate the total amount of spikes"""
        spike=spikedata.shape[0]
        spikes.append(spike)

        """Calculate the average firing frequency frequency"""
        firing_frequency=spike/(parameters['measurements']/parameters['sampling rate'])
        mean_firingrate.append(firing_frequency)

        """Calculate mean inter spike interval"""
        spikeintervals = []
        # Iterate over every registered spike
        if spikedata.shape[0]<2:
            mean_ISI.append(float("NaN"))
            median_ISI.append(float("NaN"))
            ratio_median_over_mean.append(float("NaN"))
        else:
            spikeintervals=spikedata[1:-1, 0]-spikedata[0:-2, 0]
            mean_ISI_electrode=np.mean(spikeintervals)
            mean_ISI.append(mean_ISI_electrode)
        
            """Calculate the median of the ISIs"""
            median_ISI_electrode=np.median(spikeintervals)
            median_ISI.append(median_ISI_electrode)

            """Calculate the ratio of median ISI over mean ISI"""
            ratio_median_over_mean.append(median_ISI_electrode/mean_ISI_electrode)

        """Calculate Interspike interval variance"""
        if spikedata.shape[0]<3:
            IIV.append(float("NaN"))
        else:
            IIV.append(np.var(spikeintervals))

        """Calculate the coeficient of variation of the interspike intervals"""
        if spikedata.shape[0]<3:
            CVI.append(float("NaN"))
        else:
            CVI.append(np.std(spikeintervals)/np.mean(spikeintervals))

        """Calculate the partial autocorrelation function"""
        if len(spikeintervals)<4:
            ISIPACF.append(float("NaN"))
        else:
            ISIPACF.append(pacf(spikeintervals, nlags=1, method='yw')[1])

        """Calculate the total amount of bursts"""
        burst_amnt.append(len(burst_cores))

        """Calculate the average length of the bursts"""
        burst_lens=[]
        for k in burst_cores:
            burst_lens.append(k[1]-k[0])
        if len(burst_lens)==0:
            avg=float("NaN")
        else:
            avg=np.mean(burst_lens)
        avg_burst_len.append(avg)

        """Calculate the variance of the burst length"""
        if len(burst_lens)<2:
            burst_var.append(float("NaN"))
        else:
            burst_var.append(np.var(burst_lens))

        """Calculate the coefficient of variation of the burst lengths"""
        if len(burst_lens)<2:
            burst_CV.append(float("NaN"))
        else:
            burst_CV.append(np.std(burst_lens)/np.mean(burst_lens))

        """Calculate the average time between bursts"""
        IBIs=[]
        for i in range(len(burst_cores)-1):
            time_to_next_burst=burst_cores[i+1][0]-burst_cores[i][1]
            IBIs.append(time_to_next_burst)
        if len(IBIs)==0:
            mean_IBI.append(float("NaN"))
        else:
            mean_IBI.append(np.mean(IBIs))

        """Calculate the variance of interburst intervals"""
        if len(IBIs)<2:
            IBI_var.append(float("NaN"))
        else:
            IBI_var.append(np.var(IBIs))

        """Calculate the coefficient of variation of the interburst intervals"""
        if len(IBIs)<2:
            IBI_CV.append(float("NaN"))
        else:
            IBI_CV.append(np.std(IBIs)/np.mean(IBIs))

        """Calculate the partial autocorrelation function of the interburst intervals"""
        if len(IBIs)<4:
            IBIPACF.append(float("NaN"))
        else:
            IBIPACF.append(pacf(IBIs, nlags=1, method='yw')[1])

        """Calculate the mean intraburst firing rate and average amount of spikes per burst"""
        IBFRs=[]
        spikes_per_burst=[]
        if len(burst_cores)>0:
            # Iterate over all the bursts
            for i in np.sort(np.unique(burst_cores[:,4])).astype(int):
                # Find which spikes belong to the bursts - here 'i' is the identifier of the burst
                locs=np.where(burst_spikes[:,3]==i)
                # Calculate how many spikes in the burst
                total_burst_spikes = len(locs[0])
                spikes_per_burst.append(total_burst_spikes)
                # Calculate firing rate based on length of the burst
                firingrate=total_burst_spikes/burst_lens[i]
                IBFRs.append(firingrate)

            # Append means to final list
            intraBFR.append(np.mean(IBFRs))
            mean_spikes_per_burst.append(np.mean(spikes_per_burst))
        else:
            intraBFR.append(float("NaN"))
            mean_spikes_per_burst.append(float("NaN"))

        """Calculate the mean absolute deviation of the amount of spikes per burst"""
        if len(burst_cores)>0:
            MAD_spikes_per_burst.append(np.mean(np.absolute(spikes_per_burst - np.mean(spikes_per_burst))))
        else:
            MAD_spikes_per_burst.append(float("NaN"))

        """Calculate the % of spikes that are isolated (not in a burst)"""
        if burst_spikes.shape[0]>0 and spike>0:
            isolated_spikes.append(1-(burst_spikes.shape[0]/spike))
        else:
            isolated_spikes.append(float("NaN"))

        """Calculate the single channel burst rate (bursts/s)"""
        if len(burst_cores)>0:
            SCB_rate.append(len(burst_cores)/(parameters['measurements']/parameters['sampling rate']))
        else:
            SCB_rate.append(float(0))

        """Calculate average spike amplitude"""
        if spikedata.shape[0]==0:
            mean_spike_amplitude.append(float("NaN"))
        else:
            mean_spike_amplitude.append(np.mean(np.abs(spikedata[:, 1])))

        """Calculate median spike amplitude"""
        if spikedata.shape[0]==0:
            median_spike_amplitude.append(float("NaN"))
        else:
            median_spike_amplitude.append(np.median(np.abs(spikedata[:, 1])))

        """Calculate spike amplitude coefficient of variation"""
        if spikedata.shape[0]<2:
            cv_spike_amplitude.append(float("NaN"))
        else:
            cv_spike_amplitude.append(np.std(np.abs(spikedata[:, 1]))/np.mean(np.abs(spikedata[:, 1])))
        
    # Create pandas dataframe with all features as columns 
    features_df = pd.DataFrame({
        "Well": wells_df,
        "Electrode": electrodes_df, 
        "Spikes": spikes,
        "Mean Firing Rate": mean_firingrate,
        "Mean Inter-Spike interval": mean_ISI,
        "Median Inter-Spike interval": median_ISI,
        "Ratio median ISI over mean ISI": ratio_median_over_mean,
        "Inter-spike interval variance": IIV,
        "Coefficient of variation ISI": CVI,
        "Partial Autocorrelaction Function ISI": ISIPACF,
        "Bursts": burst_amnt,
        "Mean Burst Rate": SCB_rate,
        "Mean Burst Length": avg_burst_len,
        "Burst Length Variance": burst_var,
        "Burst Length Coefficient of Variation": burst_CV,
        "Mean Inter-Burst Interval": mean_IBI,
        "Variance Inter-Burst Interval": IBI_var,
        "Coefficient of Variation IBI": IBI_CV,
        "Partial Autocorrelation Function IBI": IBIPACF,
        "Mean Intra-Burst Firing Rate": intraBFR,
        "Mean Spikes per Burst": mean_spikes_per_burst,
        "Mean Absolute Deviation Spikes per Burst": MAD_spikes_per_burst,
        "Isolated Spikes": isolated_spikes,
        "Mean Absolute Spike Amplitude" : mean_spike_amplitude,
        "Median Absolute Spike Amplitude" : median_spike_amplitude,
        "Coefficient of Variation Absolute Spike Amplitude" : cv_spike_amplitude
    })

    if parameters['remove inactive electrodes']:
        # Remove electrodes that do not have enough activity
        features_df=features_df[features_df["Mean Firing Rate"]>parameters['activity threshold']]

    # Calculate how many electrodes were active
    active_electrodes=len(features_df)/parameters['electrode amount']

    # If none of the electrodes have enough activity, make sure we retain the well value
    if len(features_df)==0:
        features_df["Well"]=[well_nr]
    if parameters['remove inactive electrodes']:
        features_df.insert(2, "Active_electrodes", [active_electrodes]*len(features_df))
    return features_df


def electrode_pair_features(parameters, save_data=True):
    """"
    Calculates electrode pair features for unique pairs of each well.

    Paramters
    --------
    Well: int
        The well for wich the features are to be calculated.
    Paramters : dict
        Dictionary containing global paramaters. The function will extract the values needed.

    Returns
    ---------
    features_df : pandas Dataframe
        Dataframe containing electrodes pairs for rows and features for columns.
    """

    # Initialize empty structures for storing results
    pairs_skipped = []
    distance_df = []
    spiketimes_dict = {}
    distance_dict = {}
    matrices = {}

    # Get the path/name of the output HDF5 file from parameters
    output_hdf_file = parameters['output hdf file']

    # Generate time values based on the number of measurements and sampling rate
    time_df = np.arange(0, int(parameters['measurements']) / int(parameters['sampling rate']))
    start_time = np.min(time_df)
    end_time = np.max(time_df)

    # Get the number of wells from parameters
    wells = parameters['well amount']

    # Create an array of electrode IDs (starting from 1)
    electrodes = np.arange(1, parameters['electrode amount'] + 1)

    # Generate all unique electrode pairs (combinations of 2)
    unique_pairs_electrodes = list(itertools.combinations(electrodes, 2))

    # Create labels for matrix axes using electrode numbers
    labels = [f'electrode_{i}' for i in electrodes]


    # Define mapping between synchronicity method names and internal method codes
    method_map = {
        'Adaptive ISI-distance': 'adaptive_ISI_distance',
        'ISI-distance': 'ISI_distance',
        'SPIKE-distance': 'SPIKE_distance',
        'Adaptive SPIKE-distance': 'adaptive_SPIKE_distance'
    }

    # Getting correct method gives auto 'SPIKE-distance' back if nothing is filled.
    method = method_map.get(parameters['synchronicity method'], 'SPIKE_distance')

    with h5py.File(output_hdf_file, 'r') as f:
           
        for well in wells:
            # Reset for each well
            distance_matrix = np.full((len(electrodes), len(electrodes)), np.nan)  # standaard lege waarden
            distances = []

            # Set diagonal to one (synchrony with itself)
            np.fill_diagonal(distance_matrix, 1)

            # Load spiketrain of each electrode
            for electrode in electrodes:
                dataset_path = f"/spike_values/well_{well}_electrode_{electrode}_spikes"
                
                if dataset_path in f:
                    # Load spike data
                    spike_values = f[dataset_path][()]

                    # Get correct columns: Time (column 0), amplitude (column 2), index (column 2)
                    spiketimes = spike_values[:, 0]
                    spiketimes_dict[(well, electrode)] = spiketimes


            for electrode_x, electrode_y in unique_pairs_electrodes:
                
                electrode_pair = (electrode_x,electrode_y)

                # Retrieve spike trains for both electrodes from the dictionary
                spiketrain_x = spiketimes_dict.get((well, electrode_x), None)
                spiketrain_y = spiketimes_dict.get((well, electrode_y), None)
                
                # If one (or both) electrodes is not active -> there is asynchronicity
                if (spiketrain_x is None or spiketrain_x.size == 0) or \
                    (spiketrain_y is None or spiketrain_y.size == 0):
                    electrode_pair_well = (well, electrode_pair)
                    pairs_skipped.append(electrode_pair_well)
                    distance_value = 0


                else:
                    # Calculate the synchronicity with given method
                    if parameters['synchronicity method'] == 'adaptive_ISI_distance' or parameters['synchronicity method'] == 'adaptive_SPIKE_distance':
                        
                        # Calculate automatic threshold
                        spiketrains = [np.asarray(spiketrain_x), np.asarray(spiketrain_y)]
                        threshold = default_thresh(spiketrains, start_time, end_time)
                            
                    else:
                        # No threshold
                        threshold = 0

                    # Calculate synchronicity for ISI-distance            
                    if parameters['synchronicity method'] == 'adaptive_ISI-distance' or parameters['synchronicity method'] == 'ISI_distance':
                        distance_value, _ = isi_distance(spiketrain_x, spiketrain_y, start_time, end_time, threshold)

                    # Calculate synchronicity for SPIKE-distance
                    else:
                        distance_value, _ = spike_distance(spiketrain_x, spiketrain_y, start_time, end_time, threshold, RI=0)
                    
                    # Store synchronicity value in symmetric matrix positions
                    if distance_value is not None:
                        i, j = electrode_x - 1, electrode_y - 1
                        distance_matrix[i, j] = distance_value
                        distance_matrix[j, i] = distance_value

                # Save full matrix and individual value
                matrices[well] = pd.DataFrame(distance_matrix, index=labels, columns=labels) 
                distance_dict[(well, (electrode_x, electrode_y))] = distance_value
                distances.append(distance_value)
                
            # Compute and store mean distance perr well    
            distance_mean = np.mean(distances)
            distance_df.append([well, distance_mean])

        # Calculate the well synchronicitys
        mean_distance_df = pd.DataFrame(distance_df, columns = ["Well", method] )


    # Saving in output hdf5 file
    if save_data:
        with h5py.File(output_hdf_file, 'a') as f:
            f.create_dataset(f'synchronicity_values/{method}_well', data=mean_distance_df)
            for well, matrix in matrices.items():
                f.create_dataset(f'synchronicity_values/{method}_electrodes_pair/matrix_well_{well}', data=matrix)

    electrode_pair_features = mean_distance_df
           
    return  electrode_pair_features



'''Calculate the features per well'''
@silence_runtime_warnings
def well_features(well, parameters):
    """
    Calculate well features for a specific well

    Parameters
    ----------
    well : int
        The well for which the features are to be calculated.
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed

    Returns
    -------
    well_features_df : pandas Dataframe
        Dataframe containing well features
    
    """
    
    output_hdf_file=parameters['output hdf file']

    # Load in the network burst data
    with h5py.File(output_hdf_file, 'r') as f:
        network_cores=f[f"network_values/well_{well}_network_bursts"]
        network_cores=network_cores[:]
        participating_bursts=f[f"network_values/well_{well}_participating_bursts"]
        participating_bursts=participating_bursts[:]

    # Load in data from electrodes from this well
    spikedata_list=[]
    burstcores_list=[]
    burstspikes_list=[]

    electrodes=np.arange(1, parameters['electrode amount']+1)

    for electrode in electrodes:
        with h5py.File(output_hdf_file, 'r') as f:
            spikedata=f[f"spike_values/well_{well}_electrode_{electrode}_spikes"]
            spikedata=spikedata[:]
            burst_cores=f[f"burst_values/well_{well}_electrode_{electrode}_burst_cores"]
            burst_cores=burst_cores[:]
            burst_spikes=f[f"burst_values/well_{well}_electrode_{electrode}_burst_spikes"]
            burst_spikes=burst_spikes[:]
            
        spikedata_list.append(spikedata)
        burstcores_list.append(burst_cores)
        burstspikes_list.append(burst_spikes)
    
    network_bursts, network_burst_duration, network_burst_core_duration, network_IBI, NB_NBc_ratio, NB_firingrate, NB_ISI, mean_spikes_per_network_burst = [], [], [], [], [], [], [], []
    nIBI_var, nIBI_CV, NB_NBC_ratio_left, NB_NBC_ratio_right, lr_NB_ratio, NBC_duration_CV, NIBIPACF, participating_electrodes, portion_spikes_in_nbs, portion_bursts_in_nbs = [], [], [], [], [], [], [], [], [], []

    '''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Calculate your own features here
        Step by step guide:
        1. Initialize a list prior to this for-loop, you will be adding the calculated feature of the well to this list
        2. You have access to the following variables after this line of code:
            network_cores = All the network bursts that have been detected in this well. Contains a row for each network burst, each row contains 5 columns
                network_bursts[burst, 0] = start of the network burst core (seconds)
                network_bursts[burst, 1] = end of the network burst core
                network_bursts[burst, 2] = start of the total network burst
                network_bursts[burst, 3] = end of the total network burst
                network_bursts[burst, 4] = ID of the network burst, used to couple network burst to participating single-channel bursts
            participating_bursts = All the bursts that have participated in a network burst in this well. Contains a row for each single-channel burst (SCB), each row contains 3 colomns
                participating_bursts[burst, 0] = The network burst ID
                participating_bursts[burst, 1] = The electrode on which the SCB happened
                participating_bursts[burst, 2] = The burst ID of the SCB
            spikedata_list
                spikedata from all the electrodes in this particular well. Index 0 is top left electrode, last index is bottom right electrode
                follows the same format as described in the electrode features
            burstcores_list
                same as previous, but for burst cores
            burstspikes_list
                same as previous, but for burst spikes
        3. Calculate the feature.
           Warning: sometimes it is not possible to calculate a features due to a lack of detected network bursts.
           always check if there are enough values to calculate the feature using an if-statement, if there are not, append "float("NaN")" to the list
        4. Append your feature to the list
        5. At the end of this function, add your list to the pandas dataframe, and give it the correct name.
           Now if you run the analysis again, you should see your own feature in the "Features.csv" output file
        If there are any questions, feel free to reach out!
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''

    """Calculate the total amount of network bursts"""
    network_bursts.append(len(network_cores))

    # Fill the features with NaN values if there are no network bursts
    if len(network_cores)==0:
        network_burst_duration.append(float("NaN"))
        network_burst_core_duration.append(float("NaN"))
        NBC_duration_CV.append(float("NaN"))
        NB_NBc_ratio.append(float("NaN"))
        NB_NBC_ratio_left.append(float("NaN"))
        NB_NBC_ratio_right.append(float("NaN"))
        lr_NB_ratio.append(float("NaN"))
        network_IBI.append(float("NaN"))
        nIBI_var.append(float("NaN"))
        nIBI_CV.append(float("NaN"))
        NIBIPACF.append(float("NaN"))
        mean_spikes_per_network_burst.append(float("NaN"))
        NB_firingrate.append(float("NaN"))
        NB_ISI.append(float("NaN"))
        portion_spikes_in_nbs.append(float("NaN"))
        portion_bursts_in_nbs.append(float("NaN"))
        participating_electrodes.append(float("NaN"))
    else:

        """Calculate the average length of a network burst"""
        NB_duration=[]
        for i in range(len(network_cores)):
            NB_duration.append(network_cores[i,3]-network_cores[i,2])
        network_burst_duration.append(np.mean(NB_duration))

        """Calculate the average length of a network burst core"""
        NBC_duration=[]
        for i in range(len(network_cores)):
            NBC_duration.append(network_cores[i,1]-network_cores[i,0])
        network_burst_core_duration.append(np.mean(NBC_duration))
        
        """Calculate the coefficient of variation of the network burst cores duration"""
        if len(NBC_duration)<2:
            NBC_duration_CV.append(float("NaN"))
        else:
            NBC_duration_CV.append(np.std(NBC_duration)/np.mean(NBC_duration))
        
        """Calculate the mean network interburst interval"""
        network_IBIs=[]
        if len(network_cores)<2:
            network_IBI.append(float("NaN"))
        else:
            # Calculate the network interburst interval
            # Iterate over the network bursts
            for i in range(len(network_cores)-1):
                # Calculate the distance to the next NB
                dist=network_cores[i+1, 2] - network_cores[i,3]
                network_IBIs.append(dist)

            # Calculate the mean
            network_IBI.append(np.mean(network_IBIs))

        """Calculate NIBI variance and coefficient of variation"""
        if len(network_IBIs)<2:
            nIBI_var.append(float("NaN"))
            nIBI_CV.append(float("NaN"))
        else:
            # Calculate the variance of the network IBI
            nIBI_var.append(np.var(network_IBIs))
            # Calculate the coefficient of variation of the network IBI
            nIBI_CV.append(np.std(network_IBIs)/np.mean(network_IBIs))

        """Calculate the partial autocorrelation function of the network interburst intervals"""
        if len(network_IBIs)<4:     # At least 4 values are needed for this calculation
            NIBIPACF.append(float("NaN"))
        else:
            NIBIPACF.append(pacf(network_IBIs, nlags=1, method='yw')[1])

        """Calculate the intra-network burst firing rate and network burst inter spike interval"""
        single_nb_firingrate=[]
        single_nb_ISI=[]
        spikes_per_network_burst=[]
        total_spikes_in_nbs=0

        for network_burst in range(len(network_cores)):
            nb_spikes=[]
            # Identify all spikes that happened during the network burst
            # Loop over all electrodes in well
            for channel in range(len(spikedata_list)):
                # Loop over all spikes in the channel
                for spike in range(len(spikedata_list[channel])):
                    # Check whether this spike occured during the network burst
                    if network_cores[network_burst, 2] <= spikedata_list[channel][spike][0] <= network_cores[network_burst, 3]:
                        nb_spikes.append(spikedata_list[channel][spike][0])
            # Calculate the NB firing rate (spikes / durations)
            single_nb_firingrate.append((len(nb_spikes))/(network_cores[network_burst, 3]-network_cores[network_burst, 2]))
            total_spikes_in_nbs+=len(nb_spikes)
            # Calculate the network burst inter spike interval
            # First sort the spike so they are on chronological order again
            nb_spikes=np.sort(nb_spikes)
            # Calculate the intervals
            nb_spikes_intervals=nb_spikes[1:-1]-nb_spikes[0:-2]
            # Take the average
            single_nb_ISI.append(np.mean(nb_spikes_intervals))
            
            spikes_per_network_burst.append(len(nb_spikes))

        # Calculate averages from all network bursts
        NB_firingrate.append(np.mean(single_nb_firingrate))
        NB_ISI.append(np.mean(single_nb_ISI))
        mean_spikes_per_network_burst=np.mean(spikes_per_network_burst)

        """Calculate the portion of spikes that participate in network bursts"""
        total_spikes=0
        for channel in range(len(spikedata_list)):
            total_spikes+=len(spikedata_list[channel])
        if total_spikes_in_nbs==0 or total_spikes==0:
            portion_spikes_in_nbs.append(float("NaN"))
        else:
            portion_spikes_in_nbs.append(total_spikes_in_nbs/total_spikes)

        """Calculate the portion of bursts that participate in network bursts"""
        # We must first remove duplicate bursts, because the spike activity in a single burst can contribute to multiple bursts
        # Burst can be identified by their channel of origin, and their unique identification number within the channel, which are stored in column 1 and 2 of 'participating_bursts'
        original_bursts = []
        # Loop over all bursts
        for burst in participating_bursts:
            is_original = True
            # Check if it is a duplicate
            for original_burst in original_bursts:
                if np.array_equal(original_burst, burst[1:]):
                    is_original = False
            if is_original:
                original_bursts.append(burst[1:])

        burst_participating_in_nbs = len(original_bursts)
        total_bursts=0
        for channel in range(len(burstcores_list)):
            total_bursts+=len(burstcores_list[channel])
        if burst_participating_in_nbs==0 or total_bursts==0:
            portion_bursts_in_nbs.append(float("NaN"))
        else:
            portion_bursts_in_nbs.append(burst_participating_in_nbs/total_bursts)
        
        """Calculate average the network burst to network burst core ratio"""
        ratios=[]
        for i in range(len(network_cores)):
            ratio=(network_cores[i,3]-network_cores[i,2])/(network_cores[i,1]-network_cores[i,0])
            ratios.append(ratio)
        NB_NBc_ratio.append(np.mean(ratios))

        """Calculate the ratio of the length of the left/right outer part of the network burst compared to the core"""
        left_ratios=[]
        right_ratios=[]
        lr_ratios=[]
        for i in range(len(network_cores)):
            burst_core_length=network_cores[i,1]-network_cores[i,0]
            left_outer_burst=network_cores[i,0]-network_cores[i,2]
            right_outer_burst=network_cores[i,3]-network_cores[i,1]
            # Sometimes the burst has no outer edges, so we get a division by 0, check for that here
            if np.min([left_outer_burst, right_outer_burst])==0:
                # In that case append 0
                lr_ratios.append(0)
            else:
                lr_ratios.append(left_outer_burst/right_outer_burst)
            left_ratios.append(left_outer_burst/burst_core_length)
            right_ratios.append(right_outer_burst/burst_core_length)
        NB_NBC_ratio_left.append(np.mean(left_ratios))
        NB_NBC_ratio_right.append(np.mean(right_ratios))
        lr_NB_ratio.append(np.mean(lr_ratios))

        """Calculate the average amount of electrodes that contribute to a network burst"""
        single_participating_channels=[]
        # Loop through the network bursts
        for network_burst in np.unique(participating_bursts[:,0]):
            # Calculate the amount of electrodes that participated
            electrodes_participated=len(np.unique(participating_bursts[participating_bursts[:,0]==network_burst][:,1]))
            single_participating_channels.append(electrodes_participated)
        participating_electrodes.append(np.mean(single_participating_channels))

    # Create pandas dataframe with all features as columns 
    well_features_df = pd.DataFrame({
        "Well":[well],
        "Network Bursts": network_bursts,
        "Mean Network Burst Duration": network_burst_duration,
        "Mean Network Burst Core Duration": network_burst_core_duration,
        "Coefficient of Variation Network Burst Core Duration": NBC_duration_CV,
        "Network Inter-Burst Interval": network_IBI,
        "Partial Autocorrelation Function NIBI": NIBIPACF,
        "Network Burst to Network Burst Core ratio": NB_NBc_ratio,
        "NIBI Variance": nIBI_var,
        "Coefficient of Variation NIBI": nIBI_CV,
        "Mean Spikes per Network Burst": mean_spikes_per_network_burst,
        "Network Burst Firing Rate": NB_firingrate,
        "Network Burst ISI": NB_ISI,
        "Portion of Spikes in Network Bursts": portion_spikes_in_nbs,
        "Portion of Bursts in Network Bursts": portion_bursts_in_nbs,
        "Ratio Left Outer NB over NB Core": NB_NBC_ratio_left,
        "Ratio Right Outer NB over NB Core": NB_NBC_ratio_right,
        "Ratio Left Outer over Right Outer NB": lr_NB_ratio,
        "Participating Electrodes": participating_electrodes
    })
    return well_features_df
 

def feature_output(electrode_features, well_features, electrode_pair_features):
    """
    Function to combine and clean up electrode and well features dataframes

    Parameters
    ----------
    electrode_features : pandas Dataframe
        Electrodes features.
    well_features : pandas Dataframe
        Well features.

    Returns
    -------
    avg_electrode_features : pandas Dataframe
        Combined single electrode and well features

    Notes
    -----
    This function will do the following things:
    
    - Take the average of all single electrode features of a well.
    - Remove unnecessary and duplicate columns.
    - Combine the average electrode and well features.

    """
    # Take the average of all the single-electrode features
    avg_electrode_features=pd.DataFrame(electrode_features.mean()).transpose()

    # Remove unnecessary columns
    avg_electrode_features=avg_electrode_features.drop(columns=["Electrode"])
    well_features=well_features.drop(columns=["Well"])
    electrode_pair_features.drop(columns=["Well"])

    # Combine the dataframes
    avg_electrode_features = pd.concat(
        [avg_electrode_features, well_features, electrode_pair_features],
        axis=1,
        join='inner'
    )
    
    return avg_electrode_features

def recalculate_features(outputfolder, well_amnt, electrode_amnt, electrodes, sampling_rate, measurements):
    """
    Recalculate all well/electrode features, leaving out specific electrodes
    Will save the features in the outputfolder, with _recalculated in the name.

    Parameters
    ----------
    outputfolder : string
        Folder where all the files necessary for feature calculation are located (parameters.json)
    electrodes : np.ndarray
        Boolean array of all electrodes indicating whether or not their features should be calculated or not
    """
    
    # Configure parameters
    parameters={
        "output hdf file" : os.path.join(outputfolder, "output_values.h5"),
        "electrode amount" : electrode_amnt,
        "measurements" : measurements,
        "sampling rate" : sampling_rate,
        "remove inactive electrodes" : False
    }

    outputfolder = Path(outputfolder)

    features_list = []

    # Loop over all wells
    for well in range(well_amnt):
        # Calculate electrode features
        electrode_features_df = electrode_features(well=well+1, parameters=parameters)

        # Retrieve which electrodes to drop
        well_electrodes_mask = electrodes[well*electrode_amnt:(well+1)*electrode_amnt]

        # Drop electrodes
        electrode_features_df = electrode_features_df[well_electrodes_mask]

        # Average features
        avg_electrode_features = pd.DataFrame(electrode_features_df.mean()).transpose()

        # Calculate well features
        well_features_df = well_features(well+1, parameters)

        # Calculate electrode pair features
        electrode_pair_features_df = electrode_pair_features(well+1, parameters)

        # Remove unnecessary columns
        avg_electrode_features=avg_electrode_features.drop(columns=["Electrode"])
        well_features_df=well_features_df.drop(columns=["Well"])
        electrode_pair_features_df.drop(columns=["Well"])
                                        
        # Combine the dataframes
        avg_electrode_features = pd.concat(
        [avg_electrode_features, well_features, electrode_pair_features],
        axis=1,
        join='inner'
        )

        # append to list
        features_list.append(avg_electrode_features)

    combined_df = pd.concat(features_list, ignore_index=True)

    # Replace previous features file
    combined_df.to_csv(f"{os.path.join(outputfolder, os.path.basename(os.path.normpath(outputfolder)))}_Features.csv", index=False)

    # Save npy file that specifies the electrode configuration
    np.save(os.path.join(outputfolder, "excluded_electrodes.npy"), electrodes)