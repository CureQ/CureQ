import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

'''Calculate the features for all the electrodes'''
def electrode_features(well,        # Which electrodes to analyse
                       parameters   # Parameters dictionary
                       ):
    
    # Define all lists for the features
    wells_df, electrodes_df, spikes, mean_firingrate, mean_ISI, median_ISI, ratio_median_over_mean, IIV, CVI, ISIPACF  = [], [], [], [], [], [], [], [], [], []
    burst_amnt, avg_burst_len, burst_var, burst_CV, mean_IBI, IBI_var, IBI_CV= [], [], [], [], [], [], []
    intraBFR, interBFR, mean_spikes_per_burst, isolated_spikes, MAD_spikes_per_burst = [], [], [], [], []
    SCB_rate, IBIPACF = [], []
    

    spikepath=f"{(parameters['output path'])}/spike_values"
    burstpath=f"{(parameters['output path'])}/burst_values"
    networkpath=f"{(parameters['output path'])}/network_data"
    
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
        spikedata=np.load(f'{spikepath}/well_{well_nr}_electrode_{electrode_nr}_spikes.npy')
        burst_cores=np.load(f'{burstpath}/well_{well_nr}_electrode_{electrode_nr}_burst_cores.npy')
        burst_spikes=np.load(f'{burstpath}/well_{well_nr}_electrode_{electrode_nr}_burst_spikes.npy')

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
            electrode_amnt = the amount of electrodes each MEA-well contains
            measurements = the total amount of measurements the MEA has taken in a single channel (recording time * sampling rate)
            hertz = the sampling rate of the MEA
        3. Calculate the feature. If you calculate it in this for-loop, it will automatically be calculated for each of the electrodes
           Warning: sometimes it is not possible to calculate a features due to a lack of detected bursts/spikes.
           always check if there are enough values to calculate the feature using an if-statement, if there are not, append "float("NaN")" to the list
        4. Append your feature to the list
        5. At the end of this function, add your list to the pandas dataframe, and give it the correct name.
           Now if you run the analysis again, you should see your own feature in the "Features.csv" output file
        If there are any questions, feel free to reach out!
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''

        # Calculate the total amount of spikes
        spike=spikedata.shape[0]
        spikes.append(spike)

        # Calculate the average firing frequency frequency
        firing_frequency=spike/(parameters['measurements']/parameters['sampling rate'])
        mean_firingrate.append(firing_frequency)

        # Calculate mean inter spike interval
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
        
            # Calculate the median of the ISIs
            median_ISI_electrode=np.median(spikeintervals)
            median_ISI.append(median_ISI_electrode)

            # Calculate the ratio of median ISI over mean ISI
            ratio_median_over_mean.append(median_ISI_electrode/mean_ISI_electrode)

        # Calculate Interspike interval variance
        if spikedata.shape[0]<3:
            IIV.append(float("NaN"))
        else:
            IIV.append(np.var(spikeintervals))

        # Calculate the coeficient of variation of the interspike intervals
        if spikedata.shape[0]<3:
            CVI.append(float("NaN"))
        else:
            CVI.append(np.std(spikeintervals)/np.mean(spikeintervals))

        # Calculate the partial autocorrelation function
        if len(spikeintervals)<4:
            ISIPACF.append(float("NaN"))
        else:
            ISIPACF.append(pacf(spikeintervals, nlags=1, method='yw')[1])

        # Calculate the total amount of bursts
        burst_amnt.append(len(burst_cores))

        # Calculate the average length of the bursts
        burst_lens=[]
        for k in burst_cores:
            burst_lens.append(k[1]-k[0])
        if len(burst_lens)==0:
            avg=float("NaN")
        else:
            avg=np.mean(burst_lens)
        avg_burst_len.append(avg)

        # Calculate the variance of the burst length
        if len(burst_lens)<2:
            burst_var.append(float("NaN"))
        else:
            burst_var.append(np.var(burst_lens))

        # Calculate the coefficient of variation of the burst lengths
        if len(burst_lens)<2:
            burst_CV.append(float("NaN"))
        else:
            burst_CV.append(np.std(burst_lens)/np.mean(burst_lens))

        # Calculate the average time between bursts
        IBIs=[]
        for i in range(len(burst_cores)-1):
            time_to_next_burst=burst_cores[i+1][0]-burst_cores[i][1]
            IBIs.append(time_to_next_burst)
        if len(IBIs)==0:
            mean_IBI.append(float("NaN"))
        else:
            mean_IBI.append(np.mean(IBIs))

        # Calculate the variance of interburst intervals
        if len(IBIs)<2:
            IBI_var.append(float("NaN"))
        else:
            IBI_var.append(np.var(IBIs))

        # Calculate the coefficient of variation of the interburst intervals
        if len(IBIs)<2:
            IBI_CV.append(float("NaN"))
        else:
            IBI_CV.append(np.std(IBIs)/np.mean(IBIs))

        # Calculate the partial autocorrelation function of the interburst intervals
        if len(IBIs)<4:
            IBIPACF.append(float("NaN"))
        else:
            IBIPACF.append(pacf(IBIs, nlags=1, method='yw')[1])

        # Calculate the mean intraburst firing rate and average amount of spikes per burst
        IBFRs=[]
        spikes_per_burst=[]
        # Iterate over all the bursts
        if len(burst_cores)>0:
            for i in range(len(burst_cores)):
                locs=np.where(burst_spikes[:,3]==i)
                total_burst_spikes = len(locs[0])
                spikes_per_burst.append(total_burst_spikes)
                firingrate=total_burst_spikes/burst_lens[i]
                IBFRs.append(firingrate)
            intraBFR.append(np.mean(IBFRs))
            mean_spikes_per_burst.append(np.mean(spikes_per_burst))
        else:
            intraBFR.append(float("NaN"))
            mean_spikes_per_burst.append(float("NaN"))

        # Calculate the mean absolute deviation of the amount of spikes per burst
        if len(burst_cores)>0:
            MAD_spikes_per_burst.append(np.mean(np.absolute(spikes_per_burst - np.mean(spikes_per_burst))))
        else:
            MAD_spikes_per_burst.append(float("NaN"))

        # Calculte the % of spikes that are isolated (not in a burst)
        if burst_spikes.shape[0]>0 and spike>0:
            isolated_spikes.append(1-(burst_spikes.shape[0]/spike))
        else:
            isolated_spikes.append(float("NaN"))

        # Calculate the single channel burst rate (bursts/s)
        if len(burst_cores)>0:
            SCB_rate.append(len(burst_cores)/(parameters['measurements']/parameters['sampling rate']))
        else:
            SCB_rate.append(float(0))
        
    # Create pandas dataframe with all features as columns 
    features_df = pd.DataFrame({
        "Well": wells_df,
        "Electrode": electrodes_df, 
        "Spikes": spikes,
        "Mean_FiringRate": mean_firingrate,
        "Mean_ISI": mean_ISI,
        "Median_ISI": median_ISI,
        "Ratio_median_ISI_over_mean_ISI": ratio_median_over_mean,
        "Interspike_interval_variance": IIV,
        "Coefficient_of_variation_ISI": CVI,
        "Partial_Autocorrelaction_Function": ISIPACF,
        "Total_number_of_bursts": burst_amnt,
        "Average_length_of_bursts": avg_burst_len,
        "Burst_length_variance": burst_var,
        "Coefficient_of_variation_burst_length": burst_CV,
        "Mean_interburst_interval": mean_IBI,
        "Variance_interburst_interval": IBI_var,
        "Coefficient_of_variation_IBI": IBI_CV,
        "Inter-burst_interval_PACF": IBIPACF,
        "Mean_intra_burst_firing_rate": intraBFR,
        "Mean_spikes_per_burst": mean_spikes_per_burst,
        "MAD_spikes_per_burst": MAD_spikes_per_burst,
        "Isolated_spikes": isolated_spikes,
        "Single_channel_burst_rate": SCB_rate
    })
    if parameters['remove inactive electrodes']:
        # Remove electrodes that do not have enough activity
        features_df=features_df[features_df["Mean_FiringRate"]>parameters['activity threshold']]
    # Calculate how many electrodes were active
    active_electrodes=len(features_df)/parameters['electrode amount']
    # If none of the electrodes have enough activity, make sure we retain the well value
    if len(features_df)==0:
        features_df["Well"]=[well_nr]
    features_df.insert(1, "Active_electrodes", [active_electrodes]*len(features_df))
    return features_df

'''Calculate the features per well'''
def well_features(well, parameters):
    
    spikepath=f"{(parameters['output path'])}/spike_values"
    burstpath=f"{(parameters['output path'])}/burst_values"
    networkpath=f"{(parameters['output path'])}/network_data"

    # Load in the network burst data
    network_cores=np.load(f"{networkpath}/well_{well}_network_bursts.npy")
    participating_bursts=np.load(f"{networkpath}/well_{well}_participating_bursts.npy")

    # Load in data from electrodes from this well
    spikedata_list=[]
    burstcores_list=[]
    burstspikes_list=[]
    electrodes=np.arange(1, parameters['electrode amount']+1)
    for electrode in electrodes:
        spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
        spikedata_list.append(spikedata)
        burst_cores=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_cores.npy')
        burstcores_list.append(burst_cores)
        burst_spikes=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_spikes.npy')
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
                network_cores[burst, 0] = start of the network burst core (seconds)
                network_cores[burst, 1] = end of the network burst core
                network_cores[burst, 2] = start of the total network burst
                network_cores[burst, 3] = end of the total network burst
                network_cores[burst, 4] = ID of the network burst, used to couple network burst to participating single-channel bursts
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
            electrode_amnt = the amount of electrodes each MEA-well contains
            measurements = the total amount of measurements the MEA has taken in a single channel (recording time * sampling rate)
            hertz = the sampling rate of the MEA
        3. Calculate the feature.
           Warning: sometimes it is not possible to calculate a features due to a lack of detected network bursts.
           always check if there are enough values to calculate the feature using an if-statement, if there are not, append "float("NaN")" to the list
        4. Append your feature to the list
        5. At the end of this function, add your list to the pandas dataframe, and give it the correct name.
           Now if you run the analysis again, you should see your own feature in the "Features.csv" output file
        If there are any questions, feel free to reach out!
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''

    # Calculate the total amount of network bursts
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
        # Calculate the average length of a network burst
        NB_duration=[]
        for i in range(len(network_cores)):
            NB_duration.append(network_cores[i,3]-network_cores[i,2])
        network_burst_duration.append(np.mean(NB_duration))

        # Calculate the average length of a network burst core
        NBC_duration=[]
        for i in range(len(network_cores)):
            NBC_duration.append(network_cores[i,1]-network_cores[i,0])
        network_burst_core_duration.append(np.mean(NBC_duration))
        
        # Calculate the coefficient of variation of the network burst cores duration
        if len(NBC_duration)<2:
            NBC_duration_CV.append(float("NaN"))
        else:
            NBC_duration_CV.append(np.std(NBC_duration)/np.mean(NBC_duration))
        
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

        if len(network_IBIs)<2:
            nIBI_var.append(float("NaN"))
            nIBI_CV.append(float("NaN"))
        else:
            # Calculate the variance of the network IBI
            nIBI_var.append(np.var(network_IBIs))
            # Calculate the coefficient of variation of the network IBI
            nIBI_CV.append(np.std(network_IBIs)/np.mean(network_IBIs))

        # Calculate the partial autocorrelation function of the network interburst intervals
        if len(network_IBIs)<4:     # At least 4 values are needed for this calculation
            NIBIPACF.append(float("NaN"))
        else:
            NIBIPACF.append(pacf(network_IBIs, nlags=1, method='yw')[1])

        # Calculate the intra-network burst firing rate and network burst inter spike interval
        single_nb_firingrate=[]
        single_nb_ISI=[]
        spikes_per_network_burst=[]
        total_spikes_in_nbs=0

        for network_burst in range(len(network_cores)):
            nb_spikes=[]
            # Identify all spikes that happened during the network burst
            for channel in range(len(spikedata_list)):
                for spike in range(len(spikedata_list[channel])):
                    # Check whether this spike occured during the network burst
                    if network_cores[network_burst, 2] <= spikedata_list[channel][spike][0] <= network_cores[network_burst, 3]:
                        nb_spikes.append(spikedata_list[channel][spike][0])
            # Calculate the NB firing rate
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
        NB_firingrate.append(np.mean(single_nb_firingrate))
        NB_ISI.append(np.mean(single_nb_ISI))
        mean_spikes_per_network_burst=np.mean(spikes_per_network_burst)

        # Calculate the portion of spikes that participate in network bursts
        total_spikes=0
        for channel in range(len(spikedata_list)):
            total_spikes+=len(spikedata_list[channel])
        if total_spikes_in_nbs==0 or total_spikes==0:
            portion_spikes_in_nbs.append(float("NaN"))
        else:
            portion_spikes_in_nbs.append(total_spikes_in_nbs/total_spikes)

        # Calculate the portion of bursts that participate in network bursts
        burst_participating_in_nbs = len(participating_bursts)
        total_bursts=0
        for channel in range(len(burstcores_list)):
            total_bursts+=len(burstcores_list[channel])
        if burst_participating_in_nbs==0 or total_bursts==0:
            portion_bursts_in_nbs.append(float("NaN"))
        else:
            portion_bursts_in_nbs.append(burst_participating_in_nbs/total_bursts)
        
        # Calculate average the network burst to network burst core ratio
        ratios=[]
        for i in range(len(network_cores)):
            ratio=(network_cores[i,3]-network_cores[i,2])/(network_cores[i,1]-network_cores[i,0])
            ratios.append(ratio)
        NB_NBc_ratio.append(np.mean(ratios))

        # Calculate the ratio of the length of the left/right outer part of the network burst compared to the core
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

        # Calculate the average amount of electrodes that contribute to a network burst
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
        "Network_bursts": network_bursts,
        "Network_burst_duration": network_burst_duration,
        "Network_burst_core_duration": network_burst_core_duration,
        "Network_burst_core_duration_CV": NBC_duration_CV,
        "Network_interburst_interval": network_IBI,
        "Network_IBI_PACF": NIBIPACF,
        "NB_to_NBc_ratio": NB_NBc_ratio,
        "Network_IBI_variance": nIBI_var,
        "Network_IBI_coefficient_of_variation": nIBI_CV,
        "Mean_spikes_per_network_burst": mean_spikes_per_network_burst,
        "Network_burst_firing_rate": NB_firingrate,
        "Network_burst_ISI": NB_ISI,
        "Portion_spikes_in_network_bursts": portion_spikes_in_nbs,
        "Portion_bursts_in_network_bursts": portion_bursts_in_nbs,
        "Ratio_left_outer_burst_over_core": NB_NBC_ratio_left,
        "Ratio_right_outer_burst_over_core": NB_NBC_ratio_right,
        "Ratio_left_outer_right_outer": lr_NB_ratio,
        "Participating_electrodes": participating_electrodes
    })
    return well_features_df
 
'''This function will take the electrode and well features, clean them up an give back the output'''
def feature_output(electrode_features, well_features, electrode_amnt):
    # Take the average of all the single-electrode features
    avg_electrode_features=pd.DataFrame(electrode_features.mean()).transpose()

    # Remove unnecessary columns
    avg_electrode_features=avg_electrode_features.drop(columns=["Electrode"])
    well_features=well_features.drop(columns=["Well"])

    # Combine the dataframes
    avg_electrode_features = pd.concat([avg_electrode_features, well_features], axis=1, join='inner')
    
    return avg_electrode_features


'''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Call the functions and test you custom features with the following code!
This is similair to how the feature extraction gets called during the analysis:


outputpath='Path/to/output/folder'

electrode_amnt=12
well_amnt=12
measurements=(1200*10000)
hertz=10000
activity_threshold=0.1
remove_inactive_electrodes=True

first_iteration=True
for well in range(1, well_amnt+1):
    electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
    # Calculate electrode and well features
    features_df=electrode_features(outputpath, well, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
    well_features_df=well_features(outputpath, well, electrode_amnt, measurements, hertz)

    # If its the first iteration, create the dataframe
    if first_iteration:
        first_iteration=False
        output=feature_output(features_df, well_features_df, electrode_amnt)
    # If its not the first iteration, keep appending to the dataframe
    else:
        output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0, ignore_index=False)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''