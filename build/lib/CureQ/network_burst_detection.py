import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from KDEpy import FFTKDE
from skimage import filters

'''Takes two burst values and calculates whether they overlap or not'''
def overlap(burst1, burst2):
    if burst1[1]<burst2[0] or burst2[1]<burst1[0]:
        return False
    else:
        return True

'''Network burst detection'''
def network_burst_detection(outputpath,         # Path where to retrieve and save data
                            wells,              # Which wells to analyse
                            electrode_amnt,     # Amount of electrodes in a well
                            measurements,       # Measurements done (time*sampling rate)
                            hertz,              # Sampling rate
                            min_channels,       # Percentage of channels that should be active for a network burst
                            threshold_method,   # Method to determine the automatic threshold
                            plot_electrodes,    # Return plot or not
                            save_figures,       # Save plot in outputfolder or not
                            savedata=True       # Save network burst information or not
                            ):
    # Define where to retrieve information from
    spikepath=f'{outputpath}/spike_values'
    burstpath=f'{outputpath}/burst_values'

    # Calculate how many channels should be active for a network burst
    min_channels=round(min_channels*electrode_amnt)
    
    # Initialize arrays
    total_network_burst=[]
    participating_bursts=[]

    time_seconds=np.arange(0, measurements/hertz, 1/hertz)
    
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
        ax[0].set_xlim([0,measurements/hertz])
        ax[0].set_ylim([0,electrode_amnt+1])
        ax[0].set_yticks(np.arange(1, electrode_amnt+1, 1))
        ax[3].set_xlabel("Time in seconds")
        ax[1].set_ylim([0, electrode_amnt+1])
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
        for electrode in range(1,electrode_amnt+1):
            # Load in spike and burst data
            spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
            burstdata=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_spikes.npy')
            
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
        lineoffsets1=np.arange(1,electrode_amnt+1)
        # Spikes and burst spikes in eventplot
        ax[0].eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        ax[0].eventplot(burst_spikes, alpha=0.25, color='red', lineoffsets=lineoffsets1)

        '''2nd plot'''
        burst_freq=np.zeros(measurements)
        burst_cores_list=[]

        # Load in burst core data
        for electrode in range(1,electrode_amnt+1):
            burst_cores=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_cores.npy')
            burst_cores_list.append(burst_cores)
            for burstcore in burst_cores:
                burst_freq[int(burstcore[2]):int(burstcore[3])]+=1       
        ax[1].plot(time_seconds, burst_freq)

        '''3rd plot'''
        # Plot the spike frequency using kernel density estimate
        # Create the grid for the KDE
        KDE_grid=np.arange(0, measurements/hertz, (1/hertz)*10)
        if len(spikes_for_kde)>0:
            # Create the KDE using the KDEpy library
            y = FFTKDE(bw=0.1, kernel='gaussian').fit(spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            # Normalise the kde so it ranges from 0 to 1
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # Plot the KDE
            ax[2].clear()
            ax[2].plot(x,y)

        '''Bottom plot'''
        # Check if there are any single channel bursts, if there are none, there cannot be any network bursts and we can skip the network burst detection
        if len(burst_spikes_for_kde)>0:
            # Create the KDE from only spikes that are part of a burst using the KDEpy library
            y = FFTKDE(bw=0.1, kernel='gaussian').fit(burst_spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            # Normalise the kde so it ranges from 0 to 1
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # Plot the KDE
            ax[3].clear()
            ax[3].plot(x,y)

            # Determine the threshold using either Yen or Otsu automatic thresholding
            if threshold_method=='yen' or threshold_method=='Yen':
                threshold=filters.threshold_yen(y, nbins=1000)
            elif threshold_method=='otsu' or threshold_method=='Otsu':
                threshold=filters.threshold_otsu(y, nbins=1000)
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
                        if overlap(network_burst_cores[i], burst):
                            channel_participates=True
                    # If there was a burst active, increment the number of channels participating
                    if channel_participates: channels_participating+=1
                # Mark network burst cores that are too close to the start or end of the measurement
                too_close = network_burst_cores[i][0]<buffer or network_burst_cores[i][1]>(measurements/hertz)-buffer
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
                        if overlap(network_burst_cores[i], burst):
                            # If it starts earlier/ends later then any previous SCB, make this the start/end of the network burst
                            if burst[0]<outer_start: outer_start=burst[0]
                            if burst[1]>outer_end: outer_end=burst[1]
                            # Add the identifiers (Network_burst ID, electrode, burst ID) to the list
                            participating_bursts.append([i, j, int(burst[4])])
                # Add start+end to network burst cores list
                network_burst_cores[i].append(outer_start)
                network_burst_cores[i].append(outer_end)
                
                # Give each network burst a number (identifier)
                network_burst_cores[i].append(i)
            
            # Remove overlapping bursts
            overlapping_bursts=[]
            # Loop through all the bursts
            for i in range(len(network_burst_cores)-1):
                # Check if there is overlap with the next burst
                if overlap(network_burst_cores[i][2:4], network_burst_cores[i+1][2:4]):
                    # Mark the shortest burst for removal
                    if (network_burst_cores[i][1]-network_burst_cores[i][0])<(network_burst_cores[i+1][1]-network_burst_cores[i+1][0]):
                        overlapping_bursts.append(i)
                    else:
                        overlapping_bursts.append(i+1)
            if len(overlapping_bursts)>0:
                # Remove the overlapping bursts (we do this outside of the loop to not disturb the loop)
                for j in sorted(overlapping_bursts, reverse=True):
                    del network_burst_cores[j]
                # Remove participating bursts with the removed ID
                participating_bursts=np.array(participating_bursts)[~np.isin(np.array(participating_bursts)[:,0],overlapping_bursts)]

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
            path = f'{outputpath}/network_data'
            np.savetxt(f'{path}/well_{well}_network_bursts.csv', total_network_burst, delimiter = ",")
            np.save(f'{path}/well_{well}_network_bursts', total_network_burst)
            np.savetxt(f'{path}/well_{well}_participating_bursts.csv', participating_bursts, delimiter = ",")
            np.save(f'{path}/well_{well}_participating_bursts', participating_bursts)
    if save_figures:
        # Save the figure
        path=f"{outputpath}/figures/well_{well}"
        fig.savefig(path, dpi=120)  # Full HD resolution (1920x1080)
    if plot_electrodes:
        return fig
    else:
        plt.cla()
        plt.clf()
        plt.close()