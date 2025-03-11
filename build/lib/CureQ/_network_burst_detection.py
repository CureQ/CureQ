import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from KDEpy import FFTKDE
from skimage import filters
import h5py


def _overlap(burst1, burst2):
    """
    Takes two burst values and calculates whether they overlap or not

    Parameters
    ----------
    burst1 : list
        List containing start and end time of burst
    burst2 : list
        List containing start and end time of burst

    """
    if burst1[1]<burst2[0] or burst2[1]<burst1[0]:
        return False
    else:
        return True

def network_burst_detection(wells, parameters, plot_electrodes=False, save_figures=True, savedata=True):
    """
    Perform network burst detection on a specific well.

    Parameters
    ----------
    wells : list
        List of wells to perform network burst detection on.
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed.
    plot_electrodes : bool, optional
        Whether to visualize the burst detection.
    save_figures : bool, optional
        Whether to save the figures as a file.
    savedata : bool, optional
        Whether to save the data using a .npy/.csv file.

    Returns
    -------
    fig : matplotlib.Figure
        Visualization of the network burst detection

    Notes
    -----
    Instead of returning the results of the network burst detection using 'return', the function saves them at a specific file location using .npy and .csv files.
    """

    # Define where to retrieve information from
    output_hdf_file=parameters['output hdf file']

    # Calculate how many channels should be active for a network burst
    min_channels=round(parameters['min channels']*parameters['electrode amount'])
    
    # Initialize arrays
    total_network_burst=[]
    participating_bursts=[]

    time_seconds=np.arange(0, parameters['measurements']/parameters['sampling rate'], 1/parameters['sampling rate'])
    
    # Iterate over the wells (usually only one)
    for well in wells:
        # Create matplotlib figure
        fig = Figure(figsize=(16, 9))

        # Create a GridSpec with specified height ratios
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], figure=fig)

        # Add subplots to the figure
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        
        ax=[ax1, ax2, ax3, ax4]

        # Plot layout
        ax[1].set_ylabel("Active\nbursts")
        ax[0].set_xlim([0,parameters['measurements']/parameters['sampling rate']])
        ax[0].set_ylim([0,parameters['electrode amount']+1])
        ax[0].set_yticks(np.arange(1, parameters['electrode amount']+1, 1))
        ax[3].set_xlabel("Time (s)")
        ax[1].set_ylim([0, parameters['electrode amount']+1])
        ax[0].set_ylabel("Electrode")        
        ax[0].set_title(f"Well {well}")       
        ax[2].set_ylabel("Spike\nactivity\n(KDE)")
        ax[3].set_ylabel("Burst\nspike\nactivity\n(KDE)")

        # Initialize arrays
        well_spikes=[]
        burst_spikes=[]
        spikes_for_kde=np.array([])
        burst_spikes_for_kde=np.array([])

        # Iterate over all the electrodes in this well
        for electrode in range(1,parameters['electrode amount']+1):   
            # Load in spike and burst data
            with h5py.File(output_hdf_file, 'r') as f:
                spikedata=f[f"spike_values/well_{well}_electrode_{electrode}_spikes"]
                spikedata=spikedata[:]
                burstdata=f[f"burst_values/well_{well}_electrode_{electrode}_burst_spikes"]
                burstdata=burstdata[:]
            
            # Extract the spike data from the files
            well_spikes.append(spikedata[:,0])
            spikes_for_kde = np.append(spikes_for_kde, spikedata[:, 0])

            # Extract the burst data from the files
            if len(burstdata)>0:
                burst_spikes.append(burstdata[:,0])
                burst_spikes_for_kde = np.append(burst_spikes_for_kde, burstdata[:,0])
            else:
                burst_spikes.append([])
        
        '''Top plot'''
        # Plot layout
        lineoffsets1=np.arange(1,parameters['electrode amount']+1)
        # Spikes and burst spikes in eventplot
        ax[0].eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        ax[0].eventplot(burst_spikes, alpha=0.25, color='red', lineoffsets=lineoffsets1)

        '''2nd plot'''
        burst_freq=np.zeros(parameters['measurements'])
        burst_cores_list=[]

        # Load in burst core data
        for electrode in range(1,parameters['electrode amount']+1):
            with h5py.File(output_hdf_file, 'r') as f:
                burst_cores=f[f"burst_values/well_{well}_electrode_{electrode}_burst_cores"]
                burst_cores=burst_cores[:]
            burst_cores_list.append(burst_cores)
            for burstcore in burst_cores:
                burst_freq[int(burstcore[2]):int(burstcore[3])]+=1       
        ax[1].plot(time_seconds, burst_freq)

        '''3rd plot'''
        # Plot the spike frequency using kernel density estimate
        # Create the grid for the KDE
        KDE_grid=np.arange(0, parameters['measurements']/parameters['sampling rate'], (1/parameters['sampling rate'])*1)
        if len(spikes_for_kde)>0:
            # Create the KDE using the KDEpy library
            y = FFTKDE(bw=parameters['nbd kde bandwidth'], kernel='gaussian').fit(spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            # Normalise the kde so it ranges from 0 to 1
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # # Plot the KDE
            ax[2].plot(x,y)

        '''Bottom plot'''
        # Check if there are any single channel bursts, if there are none, there cannot be any network bursts and we can skip the network burst detection
        if len(burst_spikes_for_kde)>0:
            # Create the KDE from only spikes that are part of a burst using the KDEpy library
            y = FFTKDE(bw=parameters['nbd kde bandwidth'], kernel='gaussian').fit(burst_spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            # Normalise the kde so it ranges from 0 to 1
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # # Plot the KDE
            ax[3].plot(x,y)

            # Determine the threshold using either Yen or Otsu automatic thresholding
            threshold_method=parameters['thresholding method']
            if threshold_method=='Yen' or threshold_method=='yen':
                threshold=filters.threshold_yen(y, nbins=1000)
            elif threshold_method=='Otsu' or threshold_method=='otsu':
                threshold=filters.threshold_otsu(y, nbins=1000)
            elif threshold_method=='Li' or threshold_method=='li':
                threshold=filters.threshold_li(y)
            elif threshold_method=='Isodata' or threshold_method=='isodata':
                threshold=filters.threshold_isodata(y, nbins=1000)
            elif threshold_method=='Mean' or threshold_method=='mean':
                threshold=filters.threshold_mean(y)
            elif threshold_method=='Minimum' or threshold_method=='minimum':
                threshold=filters.threshold_minimum(y, nbins=1000)
            elif threshold_method=='Triangle' or threshold_method=='triangle':
                threshold=filters.threshold_triangle(y, nbins=1000)
            else:
                raise ValueError(f"\"{threshold_method}\" is not a valid thresholding method")
            
            # Plot the threshold
            ax[3].axhline(y=threshold, color='red', linestyle='-', linewidth=1)

            # Identify regions that cross the threshold
            network_burst_cores=[]
            above_threshold=y>threshold
            i=0
            # Loop through the data
            while i < len(above_threshold):
                if above_threshold[i]:
                    # A threshold crossing has been found
                    start=x[i]
                    # Keep looping while the KDE is still above the threshold
                    while above_threshold[i] and i+1<len(above_threshold):
                        i+=1
                    # When this while loop stops, the line has dropped below the threshold, now save the network burst location
                    end=x[i]
                    network_burst_cores.append([start, end])
                i+=1

            # Validate the network bursts by checking how many channels are participating
            # At the same time, remove burst cores that are too close to the beginning or end of the measurement (buffer)
            buffer=1 #second
            unvalid_burst=[]

            # Loop through the network burst cores
            for i in range(len(network_burst_cores)):
                channels_participating=0
                # Now loop through each of the channels (burst_cores_list contains all single channel bursts)
                for channel in burst_cores_list:
                    channel_participates=False
                    # Now check if there was a burst active at that channel at the time of the network burst core
                    for burst in channel:
                        if _overlap(network_burst_cores[i], burst):
                            channel_participates=True
                    # If there was a burst active, increment the number of channels participating
                    if channel_participates: channels_participating+=1
                # Mark network burst cores that are too close to the start or end of the measurement
                too_close = network_burst_cores[i][0]<buffer or network_burst_cores[i][1]>(parameters['measurements']/parameters['sampling rate'])-buffer
                # Remove network bust cores that are too close to the end/start, or do not have enough participating channels
                if channels_participating<min_channels or too_close:
                    unvalid_burst.append(i)
            # Now remove all the network bursts that are not valid
            for j in sorted(unvalid_burst, reverse=True):
                del network_burst_cores[j]

            # Now calculate the outer edges of the network bursts using the single channel bursts
            # Calculate which SCBs were participating in the network bursts, and add their extremes as the outer edges of the network burst   
            participating_bursts=[]
            # Loop over the network burst cores
            for i in range(len(network_burst_cores)):
                outer_start=network_burst_cores[i][0]
                outer_end=network_burst_cores[i][1]
                # Loop through all the SCBs (burst_cores_list contains all single channel bursts)
                for j in range(len(burst_cores_list)):
                    for burst in burst_cores_list[j]:
                        # Check if the burst overlaps with the network burst core
                        if _overlap(network_burst_cores[i], burst):
                            # If it starts earlier/ends later then any previous SCB, make this the start/end of the network burst
                            if burst[0]<outer_start: outer_start=burst[0]
                            if burst[1]>outer_end: outer_end=burst[1]
                            # Add the identifiers (Network_burst ID, electrode, burst ID) to the list
                            participating_bursts.append([i, j+1, int(burst[4])])
                # Add start+end to network burst cores list
                network_burst_cores[i].append(outer_start)
                network_burst_cores[i].append(outer_end)
                
                # Give each network burst a number (identifier)
                network_burst_cores[i].append(i)
            
            # Fix overlapping bursts
            overlapping_bursts=[]
            # Loop through all the bursts
            for i in range(len(network_burst_cores)-1):
                # Check if there is overlap with the next burst
                if _overlap(network_burst_cores[i][2:4], network_burst_cores[i+1][2:4]):
                    overlapping_bursts.append(i)
                    overlapping_bursts.append(i+1)
            
            #remove outer edges of overlapping bursts
            if len(overlapping_bursts)>0:
                for j in overlapping_bursts:
                    network_burst_cores[j][2] = network_burst_cores[j][0]
                    network_burst_cores[j][3] = network_burst_cores[j][1]

            total_network_burst=(network_burst_cores)
            # Highlight the burst cores in every graph
            for graph in ax[:]:
                for network_burst in network_burst_cores:
                    temp=np.array(network_burst[:4])
                    graph.axvspan(temp[0], temp[1], facecolor='green', alpha=0.75)
                    graph.axvspan(temp[2], temp[3], facecolor='green', alpha=0.5)

        # If there were no bursts in the entire well, save nothing
        else:
            total_network_burst=[]
            participating_bursts=[]

        # Save the network burst data to a file
        if savedata:
            with h5py.File(output_hdf_file, 'a') as f:
                f.create_dataset(f'network_values/well_{well}_network_bursts', data=total_network_burst)
                f.create_dataset(f'network_values/well_{well}_participating_bursts', data=participating_bursts)

    if save_figures:
        # Save the figure
        path=f"{(parameters['output path'])}/figures/well_{well}"
        fig.savefig(path, dpi=240)  # 4k resolution
    if plot_electrodes:
        return fig