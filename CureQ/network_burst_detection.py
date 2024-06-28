import numpy as np
import os
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from skimage import filters

'''Takes two burst values and calculates whether they overlap or not'''
def overlap(burst1, burst2):
    if burst1[1]<burst2[0] or burst2[1]<burst1[0]:
        return False
    else:
        return True

def network_burst_detection(outputpath, wells, electrode_amnt, measurements, hertz, min_channels, threshold_method, plot_electrodes, savedata=True):
    spikepath=f'{outputpath}/spike_values'
    burstpath=f'{outputpath}/burst_values'

    min_channels=round(min_channels*electrode_amnt)
    #print(f"Minimal amount of channels required for network burst: {min_channels}")
    total_network_burst=[]
    participating_bursts=[]
    
    for well in wells:
        fig, ax = plt.subplots(4,1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]}, figsize=(16,9))
        ax[1].set_ylabel("Active\nbursts")
        ax[0].set_xlim([0,measurements/hertz])
        ax[0].set_ylim([0,electrode_amnt+1])
        ax[0].set_yticks(np.arange(1, electrode_amnt+1, 1))
        ax[3].set_xlabel("Time in seconds")
        ax[1].set_ylim([0, electrode_amnt+1])
        ax[0].set_ylabel("Electrode")
        well_spikes=[]
        burst_spikes=[]
        spikes_for_kde=np.array([])
        burst_spikes_for_kde=np.array([])
        for electrode in range(1,electrode_amnt+1):
            # Load in spikedata
            spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
            burstdata=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_spikes.npy')
            well_spikes.append(spikedata[:,0])
            spikes_for_kde = np.append(spikes_for_kde, spikedata[:, 0])
            if len(burstdata)>0:
                burst_spikes.append(burstdata[:,0])
                burst_spikes_for_kde = np.append(burst_spikes_for_kde, burstdata[:,0])
            else:
                burst_spikes.append([])
        lineoffsets1=np.arange(1,electrode_amnt+1)
        ax[0].eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        ax[0].eventplot(burst_spikes, alpha=0.25, color='red', lineoffsets=lineoffsets1)

        # Plot the burst frequency
        time_seconds=np.arange(0, measurements/hertz, 1/hertz)
        KDE_grid=np.arange(0, measurements/hertz, (1/hertz)*10)
        burst_freq=np.zeros(measurements)

        burst_cores_list=[]     # List that contains all the burst of an electrode

        # Load in burst core data
        for electrode in range(1,electrode_amnt+1):
            # Load in spikedata
            burst_cores=np.load(f'{burstpath}/well_{well}_electrode_{electrode}_burst_cores.npy')
            burst_cores_list.append(burst_cores)
            for burstcore in burst_cores:
                burst_freq[int(burstcore[2]):int(burstcore[3])]+=1
                    
        ax[1].plot(time_seconds, burst_freq)
        ax[0].set_title(f"Well {well}")

        data_time=measurements/hertz
        # Plot the spike frequency using kernel density estimate
        if len(spikes_for_kde)>0:
            y = FFTKDE(bw=0.1, kernel='gaussian').fit(spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            ax[2].clear()
            ax[2].plot(x,y)

        
        if len(burst_spikes_for_kde)>0:     # Check if there are bursts
            # Plot the spike frequency of spikes contained in bursts
            y = FFTKDE(bw=0.1, kernel='gaussian').fit(burst_spikes_for_kde).evaluate(grid_points=KDE_grid)
            x=KDE_grid
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

            ax[3].clear()
            # Determine the threshold
            if threshold_method=='yen' or threshold_method=='Yen':
                threshold=filters.threshold_yen(y, nbins=1000)
            elif threshold_method=='otsu' or threshold_method=='Otsu':
                threshold=filters.threshold_otsu(y, nbins=1000)
            else:
                raise ValueError(f"\"{threshold_method}\" is not a valid thresholding method")
            #print(f"Threshold using {threshold_method} method set at: {threshold}")
            ax[3].plot(x,y)
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
                    while above_threshold[i] and i+1<len(above_threshold):
                        i+=1
                    # When this while loop stops, the line has dropped below the threshold
                    end=x[i]
                    network_burst_cores.append([start, end])
                i+=1

            # Validate the network bursts by checking how many channels are participating
            # At the same time, remove burst cores that are too close to the beginning or end of the measurement (buffer)
            # otherwise, these will disrupt the feature calculation
            buffer=1 #second
            unvalid_burst=[]
            for i in range(len(network_burst_cores)):
                channels_participating=0
                for channel in burst_cores_list:
                    channel_participates=False
                    for burst in channel:
                        if overlap(network_burst_cores[i], burst):
                            channel_participates=True
                    if channel_participates: channels_participating+=1
                too_close = network_burst_cores[i][0]<buffer or network_burst_cores[i][1]>(measurements/hertz)-buffer
                if channels_participating<min_channels or too_close:
                    unvalid_burst.append(i)
            # Now remove all the network bursts that are not valid
            for j in sorted(unvalid_burst, reverse=True):
                del network_burst_cores[j]

            # Calculate which SCBs were participating in the network bursts, and add their extremes as the outer edges of the network burst   
            participating_bursts=[]
            for i in range(len(network_burst_cores)):
                outer_start=network_burst_cores[i][0]
                outer_end=network_burst_cores[i][1]
                for j in range(len(burst_cores_list)):
                    for burst in burst_cores_list[j]:
                        if overlap(network_burst_cores[i], burst):
                            if burst[0]<outer_start: outer_start=burst[0]
                            if burst[1]>outer_end: outer_end=burst[1]
                            # Add the identifiers (Network_burst ID, electrode, burst ID) to the list
                            participating_bursts.append([i, j, int(burst[4])])
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
            # Highlight the burst cores in every graph except te top one
            for graph in ax[:]:
                for network_burst in network_burst_cores:
                    temp=np.array(network_burst[:4])
                    graph.axvspan(temp[0], temp[1], facecolor='green', alpha=0.75)
                    graph.axvspan(temp[2], temp[3], facecolor='green', alpha=0.5)
                  
            ax[2].set_ylabel("Spike\nactivity\n(KDE)")
            ax[3].set_ylabel("Burst\nspike\nactivity\n(KDE)")
        else:
            total_network_burst=[]
            participating_bursts=[]
        if savedata:
            path = f'{outputpath}/network_data'
            np.savetxt(f'{path}/well_{well}_network_bursts.csv', total_network_burst, delimiter = ",")
            np.save(f'{path}/well_{well}_network_bursts', total_network_burst)
            np.savetxt(f'{path}/well_{well}_participating_bursts.csv', participating_bursts, delimiter = ",")
            np.save(f'{path}/well_{well}_participating_bursts', participating_bursts)
    if plot_electrodes:
        # Save the figure
        path=f"{outputpath}/figures/well_{well}"
        fig.savefig(path, dpi=120)
        return fig
    else:
        plt.cla()
        plt.clf()