import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

def burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, outputpath, plot_electrodes, savedata=True):
    # Colors
    rawdatacolor='#bb86fc'
    burstcolor='#cf6679'
    
    electrode_number=electrode
    # Calculate the well and electrode values to load in the spikedata
    well = round(electrode_number / electrode_amnt + 0.505)
    electrode = electrode_number % electrode_amnt + 1

    # Get the correct foldername to read and store values
    path=f'{outputpath}/spike_values'
    spikedata=np.load(f'{path}/well_{well}_electrode_{electrode}_spikes.npy')

    fig=None
    fig2=None
    # Calculate the inter-spike intervals
    ISI=[]
    # There are no intervals when there is only one spike
    if spikedata.shape[0]<2:
        pass
    else:
        # Iterate over all the spikes
        for i in range(spikedata.shape[0]-1):
            # Calculate the interval to the next spike and add them to an array
            time_to_next_spike = (spikedata[i+1][0]) - (spikedata[i][0])
            ISI.append(time_to_next_spike)
        ISI=np.array(ISI)
        # Convert ISIs to miliseconds
        ISI=ISI*1000
    # Only run the spike detection when there are enough spikes to form a burst
    if spikedata.shape[0]>=minspikes_burst:
        # Reset variables to the base values
        use_ISIth2=False
        ISIth2_max=maxISI_outliers
        use_chiappalone=False
        use_pasquale=False

        # Plot a smooth histogram using kernel density estimation
        output=sns.displot(ISI, alpha=0.2, edgecolor=None, kde=True, color='blue', log_scale=True, kde_kws={'gridsize': 100, 'bw_adjust':kde_bandwidth})
        for ax in output.axes.flat:
            for line in ax.lines:
                x = (line.get_xdata())
                y = (line.get_ydata())
        plt.cla()
        plt.clf()
        plt.close()
        
        # Find the peaks
        peaks=[]
        # Loop through the data
        for i in range(0, len(x)):
            # Make sure smallerneighbours does not go out of bounds
            lower_boundary=i-smallerneighbours
            if lower_boundary<0: lower_boundary=0
            upper_boundary=i+smallerneighbours
            if upper_boundary>len(x): upper_boundary=len(x)
            # Check if this value is higher than its specified amount of neighbours
            if np.all(y[i]>y[lower_boundary:i]) and np.all(y[i]>y[i+1:upper_boundary]):
                peaks.append(i)

        # Remove peaks before 1ms, as these are very likely false
        too_low_peaks=[]
        for i in range(len(peaks)):
            if x[peaks[i]]<1:
                too_low_peaks.append(i)
        for j in sorted(too_low_peaks, reverse=True):
            #print(f"Removed peak with value: {x[peaks[j]]} because value is too low")
            del peaks[j]


        # If more than 2 peaks are found, check if there are peaks located close to the values the algorithm received, this way there is a hihgher chance the advanced burst detection algorithm can be used
        if len(peaks)>2:
            # If there is a peak in the vicinity of the default value, choose that one
            # For peak 1
            peak1=float('inf')
            found_peak1=False
            for j in range(len(peaks)):
                # Check if there is a peak detected under the default ISIth value less than half the default ISIth value away from it
                dist_from_default=abs(x[peaks[j]]-default_threshold)
                if dist_from_default<0.9*default_threshold:
                    # Check if the distance is lower than the previous lowest
                    if dist_from_default<abs(peak1-default_threshold):
                        peak1=peaks[j]
                        found_peak1=True
            # For peak 2
            peak2=float('inf')
            found_peak2=False
            for j in range(len(peaks)):
                dist_from_default=abs(x[peaks[j]]-maxISI_outliers)
                #print(f"dist potential peak = {dist_from_default}")
                if x[peaks[j]]>default_threshold:
                    # Check if the distance is lower than the previous lowest
                    if dist_from_default<abs(peak2-maxISI_outliers):
                        peak2=peaks[j]
                        found_peak2=True
            #print(found_peak1, found_peak2)
            if (found_peak1 and found_peak2) and peak1 !=peak2:
                peaks=[peak1, peak2]
                #print(f"More than 2 peaks found, but able to identify 2 peaks close to the default values at peak 1: {x[peak1]} and peak 2 {x[peak2]}")
            else:
                pass
                #print(f"More than 2 peaks found")

        # Check if 2 peaks were detected
        if len(peaks)!=2:
            ISIth1=default_threshold #miliseconds
            valid_peaks=False
            use_chiappalone=True
            #print(f"Less than 2 valid peaks detected in smoothed ISI histogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
        # Check whether peak 1 has happened before default_threshold
        elif x[peaks[0]]>default_threshold:
            ISIth1=default_threshold
            valid_peaks=False
            use_chiappalone=True
            #print(f"No valid peak detected before {default_threshold}ms in smoothed ISI hisogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
        else:
            valid_peaks=True
            # Calculate the minimum y value between the 2 peaks
            logvalley=int(np.mean(np.argmin(y[peaks[0]:peaks[1]])))+peaks[0]
            valley_x=x[logvalley]
            valley_y=y[logvalley]
            if x[logvalley]<default_threshold:
                ISIth1=x[logvalley]
                use_chiappalone=True
                use_ISIth2=False
                #print(f"Valley detected before {default_threshold}ms in smoothed ISI histogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
            else:
                ISIth2=x[logvalley]
                ISIth1=default_threshold
                if ISIth2>ISIth2_max:
                    ISIth2=ISIth2_max
                use_pasquale=True
                use_ISIth2=True
                #print(f"2 valid peaks detected in smoothed ISI histogram at well {well}, electrode {electrode}, using advanced burst detection algorithm with ISIth1 set at {ISIth1} and ISIth2 set at {ISIth2}")
            
        # Plot the smoothed histogram
        if plot_electrodes:
            fig=Figure(figsize=(1,1))
            KDE=fig.add_subplot(111)
            KDE.plot(x, y)
            KDE.scatter(x[peaks], y[peaks], color="red", zorder=1)
            if valid_peaks:
                KDE.scatter(valley_x, valley_y, zorder=1)
                if use_ISIth2:
                    KDE.axvline(ISIth2, color='blue')
            for peak in peaks:
                peakneighbours=np.arange(np.max([peak-smallerneighbours, 0]), np.min([peak+smallerneighbours+1, len(x)]))
                KDE.plot(x[peakneighbours], y[peakneighbours], color="red")
            KDE.axvline(ISIth1, color='green')
            KDE.set_xscale("log")
            KDE.title.set_text(f"Well: { well}, Electrode: {electrode}")
            KDE.set_xlabel("Inter-spike interval")
            KDE.set_ylabel("Amount")

        # Apply the threshold
        burst_cores=[]
        burst_spikes=[]
        burst_counter=0

        if use_chiappalone:
            min_spikes_burstcore=minspikes_burst
            i=0
            while i < len(spikedata)-minspikes_burst-1:
                start_burst = 0.0
                end_burst = 0.0
                # Check if enough closely spaced consecutive spikes are found to form a burst
                if np.max(ISI[i:i+min_spikes_burstcore-1])<=ISIth1:
                    # A burst has been found
                    start_burst=spikedata[i][0]
                    start_burst_index=spikedata[i][2]
                    # Add all the spikes in the burst to the burst_spikes array
                    for l in range(min_spikes_burstcore):
                        temp_burst=spikedata[i+l]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                    # Move the index to the end of the burst core
                    i+=minspikes_burst-1
                    # Loop through each spike and check if it should be added to the burst
                    # Keep increasing the steps (i) while doing this
                    while ISI[i]<ISIth1:
                        temp_burst=spikedata[i+1]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                        # If you have reached the end of the list, stop
                        if i+1==len(ISI):
                            i+=1
                            break
                        i+=1
                    # Add the found values to the list
                    end_burst=spikedata[i][0]
                    end_burst_index=spikedata[i][2]
                    burst_cores.append([start_burst, end_burst, start_burst_index, end_burst_index, burst_counter])
                    burst_counter+=1
                i+=1
        if use_pasquale:
            min_spikes_burstcore=minspikes_burst
            i=0
            while i < len(spikedata)-minspikes_burst-1:
                start_burst = 0.0
                end_burst = 0.0
                # Check if enough closely spaced consecutive spikes are found to form a burst
                if np.max(ISI[i:i+min_spikes_burstcore-1])<=ISIth1:
                    # A burst has been found
                    start_burst=spikedata[i][0]
                    start_burst_index=spikedata[i][2]
                    for l in range(min_spikes_burstcore):
                        temp_burst=spikedata[i+l]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                    # Start moving backwards to append any spikes distanced less than ISIth2
                    j=i-1
                    while j>=0 and ISI[j]<ISIth2:
                        start_burst=spikedata[j][0]
                        start_burst_index=spikedata[j][2]
                        temp_burst=spikedata[j]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                        j-=1
                    # Move the index to the end of the burst
                    i+=minspikes_burst-1
                    # Loop through each spike and check if it should be added to the burst
                    # Keep increasing the steps (i) while doing this
                    while ISI[i]<ISIth2:
                        temp_burst=spikedata[i+1]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                        # If you have reached the end of the list, stop
                        if i+1==len(ISI):
                            i+=1
                            break
                        i+=1
                    # Add the found values to the list
                    end_burst=spikedata[i][0]
                    end_burst_index=spikedata[i][2]
                    burst_cores.append([start_burst, end_burst, start_burst_index, end_burst_index, burst_counter])
                    burst_counter+=1
                i+=1

            # Calculate the average burst length
            burst_len=[]
            for k in burst_cores:
                burst_len.append(k[1]-k[0])
            if len(burst_len)==0:
                avg=0
            else:
                avg=np.mean(burst_len)
            #print(f"Average burst length: {avg}")
                                            

            # Connect bursts located close to each other
            #print(f"Bursts before combining: {len(burst_cores)}")
            # Convert ISIth2 to Ms
            ISIth2_ms=ISIth2/1000
            i=0
            while i<len(burst_cores)-1:
                # Compare that end of burst a to the beginning of burst b
                if (burst_cores[i+1][0]-burst_cores[i][1])<ISIth2_ms:
                    print(f"combining {burst_cores[i]}, {burst_cores[i+1]}")
                    # Enlarge burst a so it includes burst b
                    burst_cores[i][1]=burst_cores[i+1][1]
                    # Remove burst b
                    del burst_cores[i+1]
                else:
                    i+=1
            #print(f"Bursts after combining: {len(burst_cores)}")

        # Plot the data and mark the bursts
        if plot_electrodes:
            fig2=Figure(figsize=(3,1))
            rawburstplot=fig2.add_subplot(111)

            # Plot the data
            time_seconds = np.arange(0, data.shape[0]) / hertz
            rawburstplot.plot(time_seconds, data, linewidth=0.5, zorder=-1, color=rawdatacolor)
            
            # Plot the bursts
            for burst in range(len(burst_cores)):
                burst_startx=int(burst_cores[burst][0]*hertz)
                burst_endx=int(burst_cores[burst][1]*hertz)
                rawburstplot.plot(time_seconds[burst_startx:burst_endx], data[burst_startx:burst_endx], color=burstcolor, linewidth=0.5, alpha=1)
        
            spikes=np.zeros(data.shape[0], dtype=bool)
            spike_indexes=spikedata[:,2]
            spike_indexes=spike_indexes.astype(int)
            spikes[spike_indexes]=True
            rawburstplot.scatter(time_seconds[spikes], data[spikes], color='green', marker='o', s=3, zorder=1)

            # Plot the spikes included in bursts
            if len(burst_spikes)>0:
                burst_spikes=np.array(burst_spikes)
                rawburstplot.scatter(burst_spikes[:,0], burst_spikes[:,1], color='blue', marker='o', s=3, zorder=2)

            # Plot layout
            if use_ISIth2:
                thresholdtext=f"Advanced burst detection, ISIth1: {ISIth1}, ISIth2: {ISIth2}"
            else:
                thresholdtext=f"Default burst detection, ISIth1: {ISIth1}"
            rawburstplot.title.set_text(f"Well {well} - MEA electrode {electrode}, bursts detected: {len(burst_cores)}, {thresholdtext}")
            rawburstplot.set_xlabel("Time in seconds")
            rawburstplot.set_ylabel("Micro voltage")
            rawburstplot.set_xlim([time_seconds.min(), time_seconds.max()])
            rawburstplot.set_ylim([np.min(data)*1.5, np.max(data)*1.5])
    else:
        fig=Figure(figsize=(1,1))
        fillerplot=fig.add_subplot(111)
        fig2=Figure(figsize=(3,1))
        fillerplot2=fig2.add_subplot(111)
        fillerplot2.title.set_text(f"No burst detection possible for well {well}, electrode {electrode} - not enough values")
        fillerplot2.set_xlabel("Time in seconds")
        fillerplot2.set_ylabel("Micro voltage")
        burst_spikes=[]
        burst_cores=[]
    # Save the spike data to a .csv file
    if savedata:
        path = f'{outputpath}/burst_values'
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_burst_spikes.csv', burst_spikes, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_burst_spikes', burst_spikes)
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_burst_cores.csv', burst_cores, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_burst_cores', burst_cores)
    return fig, fig2