import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

def burst_detection(data,                   # Raw data
                    electrode,              # Which electrode is getting analysed
                    electrode_amnt,         # Amount of electrodes in a well
                    hertz,                  # Sampling frequency
                    kde_bandwidth,          # Bandwidth of kernel density estimate
                    smallerneighbours,      # Amount of smaller neighbouring values a peak in the KDE should have before being used as one
                    minspikes_burst,        # Minimal amount of spikes for a burst
                    max_threshold,          # Maximum acceptable distance between spikes in a burst
                    default_threshold,      # Default threshold used for burst detection
                    outputpath,             # Path where to save the burst information
                    plot_electrodes,        # If true, returns a matplotlib figure of the spike detection/validation process. Used for the GUI
                    savedata=True           # Whether to save the burst information or not
                    ):
    
    # Colors
    rawdatacolor='#bb86fc'
    burstcolor='#cf6679'
    
    electrode_number=electrode
    fig=None
    fig2=None

    # Calculate the well and electrode values to load in the spikedata
    well = round(electrode_number / electrode_amnt + 0.505)
    electrode = electrode_number % electrode_amnt + 1

    # Get the correct foldername to read and store values
    path=f'{outputpath}/spike_values'
    # Load in the spikedata
    spikedata=np.load(f'{path}/well_{well}_electrode_{electrode}_spikes.npy')

    # Calculate the inter-spike intervals
    ISI=[]
    # There are no intervals when there is only one spike
    if spikedata.shape[0]>1:
        # Iterate over all the spikes
        for i in range(spikedata.shape[0]-1):
            # Calculate the interval to the next spike and add them to an array
            time_to_next_spike = (spikedata[i+1][0]) - (spikedata[i][0])
            ISI.append(time_to_next_spike)
        ISI=np.array(ISI)
        # Convert ISIs to miliseconds
        ISI=ISI*1000

    '''Determine how to run the burst detection'''
    # Only run the spike detection when there are enough spikes to form a burst
    if spikedata.shape[0]>=minspikes_burst:
        # Reset variables to the base values
        use_ISIth2=False
        ISIth2_max=max_threshold

        # Chiappalone is the burst detection algorithm using only one threshold
        # used when there are more/less than 2 peaks found, or when the valley between the peaks is less than the default threshold
        use_chiappalone=False
        # Pasquale is the burst detection method using two thresholds
        use_pasquale=False

        # Plot a smooth histogram using kernel density estimation
        # This very specific task (logarithmic kernel density estimate) works suprisingly well in the seaborn library, so we plot a figure there, and just extract the line values
        # Instruct seaborn to use the "Agg" backend. If we do not do this, seaborn will try to create a GUI which will eventually crash the analysis if executed from within the GUI
        matplotlib.use("Agg")
        output=sns.displot(ISI, alpha=0.2, edgecolor=None, kde=True, color='blue', log_scale=True, kde_kws={'gridsize': 100, 'bw_adjust':kde_bandwidth})
        # Extract the line values
        for ax in output.axes.flat:
            for line in ax.lines:
                x = (line.get_xdata())
                y = (line.get_ydata())
        # Close the seaborn plot so plt does not cause any trouble later
        plt.cla()
        plt.clf()
        plt.close()
        
        # Find the peaks in the KDE
        peaks=[]
        # Loop through all the values in the KDE
        for i in range(0, len(x)):
            # Make sure smallerneighbours does not go out of bounds
            lower_boundary=i-smallerneighbours
            if lower_boundary<0: lower_boundary=0
            upper_boundary=i+smallerneighbours
            if upper_boundary>len(x): upper_boundary=len(x)

            # Check if this value is higher than its specified amount of neighbours
            if np.all(y[i]>y[lower_boundary:i]) and np.all(y[i]>y[i+1:upper_boundary]):
                # If it is, append it to the list
                peaks.append(i)

        # Remove peaks before 1ms, as these are very likely false
        too_low_peaks=[]
        for i in range(len(peaks)):
            if x[peaks[i]]<1:
                too_low_peaks.append(i)
        for j in sorted(too_low_peaks, reverse=True):
            del peaks[j]

        # If more than 2 peaks are found, check if there are 2 peaks located close to the default threshold, and max threshold,
        # this way there is a higher chance the advanced burst detection algorithm can be used
        if len(peaks)>2:
            # Check which peak is closest to the default threshold
            peak1=float('inf')
            found_peak1=False
            for j in range(len(peaks)):
                dist_from_default=abs(x[peaks[j]]-default_threshold)
                if dist_from_default<0.9*default_threshold:
                    if dist_from_default<abs(peak1-default_threshold):
                        peak1=peaks[j]
                        found_peak1=True
            # Check which peak is closest to the max threshold
            peak2=float('inf')
            found_peak2=False
            for j in range(len(peaks)):
                dist_from_default=abs(x[peaks[j]]-max_threshold)
                if x[peaks[j]]>default_threshold:
                    if dist_from_default<abs(peak2-max_threshold):
                        peak2=peaks[j]
                        found_peak2=True

            # If candidates have been found for both peaks, and they are not the same, we use these two peaks for the burst detection
            if (found_peak1 and found_peak2) and peak1 !=peak2:
                peaks=[peak1, peak2]
            else:
                pass

        # Check if 2 peaks were detected
        if len(peaks)!=2:
            # If more/less than two peaks are detected, we use the chiappalone method with the default threshold
            ISIth1=default_threshold
            valid_peaks=False
            use_chiappalone=True

        # Check whether peak 1 has happened after default_threshold
        elif x[peaks[0]]>default_threshold:
            # If the first peak is located after the default threshold, we use the chiappalone method with the default threshold
            ISIth1=default_threshold
            valid_peaks=False
            use_chiappalone=True

        # If both previous statements were false, we move on to determine the valley between the peaks
        else:
            valid_peaks=True

            # Calculate the minimum y value between the 2 peaks
            logvalley=int(np.mean(np.argmin(y[peaks[0]:peaks[1]])))+peaks[0]
            # Calculate the exact values of the valley
            valley_x=x[logvalley]
            valley_y=y[logvalley]

            # Check if the valley is located before the default threshold
            if x[logvalley]<default_threshold:
                # If this is the case, we use the Chiappalone method with the threshold set at the x (time) value of the valley
                ISIth1=x[logvalley]
                use_chiappalone=True
                use_ISIth2=False
                
            # If this was not the case, move on to use the advanced burst detection algorithm with threshold1=default threshold and threshold2=valley or max threshold
            else:
                ISIth2=x[logvalley]
                ISIth1=default_threshold
                # If the valley exceeds the max threshold, set the second threshold to the max threshold
                if ISIth2>ISIth2_max:
                    ISIth2=ISIth2_max
                use_pasquale=True
                use_ISIth2=True

        '''Run the burst detection'''
        # Initialize values
        burst_cores=[]
        burst_spikes=[]
        burst_counter=0

        # Based on what was previously calculated, apply the burst detection using the correct method and values
        if use_chiappalone:
            min_spikes_burstcore=minspikes_burst
            i=0

            # Loop through the spikes until it would not be possible to form a burst anymore
            while i < len(spikedata)-minspikes_burst-1:
                start_burst = 0.0
                end_burst = 0.0
                # Check if enough closely spaced consecutive spikes are found to form a burst
                if np.max(ISI[i:i+min_spikes_burstcore-1])<=ISIth1:
                    # A burst has been found - save the start of the burst
                    start_burst=spikedata[i][0]
                    start_burst_index=spikedata[i][2]

                    # Add all the spikes in the burst to the burst_spikes array
                    for l in range(min_spikes_burstcore):
                        temp_burst=spikedata[i+l]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                    
                    # Loop through each spike and check if it should be added to the burst
                    # Keep increasing the steps (i) while doing this
                    i+=minspikes_burst-1
                    # Keep looping while the interval to the next spike is short enough
                    while ISI[i]<ISIth1:
                        temp_burst=spikedata[i+1]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                        # If you have reached the end of the list, stop
                        if i+1==len(ISI):
                            i+=1
                            break
                        i+=1

                    # Full burst has been identified, add the found values to the list
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
                    # A burst core has been found, save the start of the burst
                    start_burst=spikedata[i][0]
                    start_burst_index=spikedata[i][2]

                    # Save all the spikes in the burst core
                    for l in range(min_spikes_burstcore):
                        temp_burst=spikedata[i+l]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                    # Start moving backwards to append any spikes distanced less than ISIth2
                    j=i-1

                    # Keep moving backwards while the interval to the previous spike is low enough
                    while j>=0 and ISI[j]<ISIth2:
                        # Keep moving the start point of the burst
                        start_burst=spikedata[j][0]
                        start_burst_index=spikedata[j][2]
                        temp_burst=spikedata[j]
                        temp_burst=np.append(temp_burst, burst_counter)
                        burst_spikes.append(temp_burst)
                        j-=1
                    # Move the index back to the end of the burst
                    i+=minspikes_burst-1

                    # Loop forward to check if the next spikes should be added to the burst
                    # Keep looping while the distance to the next spike is less than the threshold
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

           
        '''Visualization - used by the GUI'''       
        # Plot the smoothed histogram
        if plot_electrodes:
            # Create a matplotlib figure
            fig=Figure(figsize=(1,1))
            KDE=fig.add_subplot(111)
            # Plot the KDE
            KDE.plot(x, y)
            # Plot the peaks
            KDE.scatter(x[peaks], y[peaks], color="red", zorder=1)
            # Plot the valley and 2nd threshold
            if valid_peaks:
                KDE.scatter(valley_x, valley_y, zorder=1)
                if use_ISIth2:
                    KDE.axvline(ISIth2, color='blue')
            # Plot neighbouring values peaks
            for peak in peaks:
                peakneighbours=np.arange(np.max([peak-smallerneighbours, 0]), np.min([peak+smallerneighbours+1, len(x)]))
                KDE.plot(x[peakneighbours], y[peakneighbours], color="red")
            # Plot the 1st treshold
            KDE.axvline(ISIth1, color='green')
            # Plot layout
            KDE.set_xscale("log")
            KDE.title.set_text(f"Well: { well}, Electrode: {electrode}")
            KDE.set_xlabel("Inter-spike interval")
            # KDE.set_ylabel("Amount")

        # Plot the raw data with bursts
        if plot_electrodes:
            # Create a matplotlib figure
            fig2=Figure(figsize=(3,1))
            rawburstplot=fig2.add_subplot(111)

            # Plot the data
            time_seconds = np.arange(0, data.shape[0]) / hertz

            # Plot the raw voltage signal
            rawburstplot.plot(time_seconds, data, linewidth=0.5, zorder=-1, color=rawdatacolor)
            
            # Plot the bursts
            for burst in range(len(burst_cores)):
                burst_startx=int(burst_cores[burst][0]*hertz)
                burst_endx=int(burst_cores[burst][1]*hertz)
                rawburstplot.plot(time_seconds[burst_startx:burst_endx], data[burst_startx:burst_endx], color=burstcolor, linewidth=0.5, alpha=1)
        
            # Plot the spikes
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

    # If there are not enough spikes for the burst detection, wel fill the figures and output arrays with nothing
    else:
        if plot_electrodes:
            fig=Figure(figsize=(1,1))
            fillerplot=fig.add_subplot(111)
            fig2=Figure(figsize=(3,1))
            fillerplot2=fig2.add_subplot(111)
            fillerplot2.title.set_text(f"No burst detection possible for well {well}, electrode {electrode} - not enough values")
            fillerplot2.set_xlabel("Time in seconds")
            fillerplot2.set_ylabel("Micro voltage")
        else:
            fig=None
            fig2=None
        
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