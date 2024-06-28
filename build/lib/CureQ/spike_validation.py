import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

'''Spike validation - noisebased'''
def spike_validation(data, electrode, threshold, hertz, spikeduration, exit_time_s, 
                     amplitude_drop, plot_electrodes, electrode_amnt, heightexception, 
                     max_boxheight, outputpath, savedata=True, plot_rectangles=False):
    # Colors
    rawdatacolor='#bb86fc'
    
    i = electrode
    # Identify points above and beneath the threshold
    above_threshold = data > threshold
    beneath_threshold = data < -threshold

    # The spike duration in amount of samples, used to establish the window around a spike
    spikeduration_samples = int(spikeduration * hertz)
    measurements=data.shape[0]
    # Iterate over the data
    loopvalues=np.nonzero(np.logical_or(above_threshold, beneath_threshold))[0]
    for j in loopvalues:
        # Calculate the the upper and lower boundary
        lower_boundary = j-spikeduration_samples
        upper_boundary = j+spikeduration_samples
        # Make sure that the boundaries do not go out of bound of the dataset (e.g. when there is a spike in the first or last milisecond of the dataset)
        if lower_boundary < 0:
            lower_boundary = 0
        if upper_boundary > measurements:
            upper_boundary = measurements
        # Checks whether this is the absolute maximum value within the give timeframe, if it is not, the peak will be removed
        maxvalue=(np.max(abs(data[(lower_boundary):(upper_boundary)])))
        potential_spike=(abs(data[j]))
        if maxvalue>potential_spike:
            above_threshold[j]=False
            beneath_threshold[j]=False
    spikes=np.logical_or(above_threshold, beneath_threshold)
    loopvalues=np.nonzero(spikes)[0]
    # Final check to remove spikes with unreasonable intervals
    for j in loopvalues:
        # Make sure we do not go out of bounds
        buffer=j+spikeduration_samples
        if buffer>measurements-1:
            buffer=measurements-1
        spikes[j+1:buffer]=False
    spikes_before_DMP=spikes.copy()
    #time_seconds = np.arange(0, data.shape[0]) / hertz
    time_seconds=np.arange(0, data.shape[0]/hertz, 1/hertz)

    # Implement dynamic multi-phasic event detection method
    # The exit time in amount of samples, used to establish the window around a spike
    exit_time = round(exit_time_s * hertz)

    loopvalues=np.nonzero(spikes)[0]
    boxheights=[]
    for j in loopvalues:
        # Check if there is a window of data to be checked before and after the spike. If the spike happens too close to the start/end of the measurement-
        # it cannot be confirmed, and will be removed.
        if j+exit_time+spikeduration_samples>data.shape[0]:
            spikes[j]=False
        elif j-exit_time-spikeduration_samples<0:
            spikes[j]=False
        else:
            # Determine the amount that the spike has to drop, based on the noise level surrounding the spike
            noise_left=data[j-spikeduration_samples-exit_time:j-exit_time]
            noise_right=data[j+exit_time:j+spikeduration_samples+exit_time]
            noise_surround=np.append(noise_left, noise_right)
            drop_amount=amplitude_drop*np.std(noise_surround)
            drop_amount=amplitude_drop*(np.sqrt(np.mean(noise_surround**2)))

            # The amount a spike has to drop should not exceed x*threshold
            if drop_amount>max_boxheight*threshold: drop_amount=max_boxheight*threshold
            boxheights.append(drop_amount)
            # Check if the voltage has reached a minimal change value of 2*Treshold since the detected spike
            # For positive spikes
            if data[j]>0:
                if not(np.min(data[j-exit_time:j+exit_time+1])<(data[j]-drop_amount)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[j]>heightexception*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
            else:
            # For negative spikes
                if not(np.max(data[j-exit_time:j+exit_time+1])>(data[j]+drop_amount)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[j]<heightexception*-1*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
            
    # Calculate MEA electrode
    electrode = i % electrode_amnt + 1
    well = round(i / electrode_amnt + 0.505)

    #Plot the data of the entire electrode
    if plot_electrodes:
        # Plot the MEA signal
        fig=Figure(figsize=(3,1))
        rawdataplot=fig.add_subplot(111)
        #rawdataplot.set_facecolor('#000000')

        #time_seconds = np.arange(0, data.shape[0]) / hertz
        rawdataplot.plot(time_seconds, data, linewidth=0.5, zorder=-1, color=rawdatacolor)
        
        # Plot the threshold
        rawdataplot.axhline(y=threshold, color='#737373', linestyle='-', linewidth=1) 
        rawdataplot.axhline(y=-threshold, color='#737373', linestyle='-', linewidth=1) 

        # Plot red dots at rejected spikes
        rawdataplot.scatter(time_seconds[spikes_before_DMP], data[spikes_before_DMP], color='red', marker='o', s=3)

        # Plot green dots at accepted spikes
        rawdataplot.scatter(time_seconds[spikes], data[spikes], color='green', marker='o', s=3)

        if plot_rectangles:
            loopvalues=np.nonzero(spikes_before_DMP)[0]
            k=0
            for j in loopvalues:

                # Plot green boxes at the accepted spikes
                if data[j]>0:
                    boxheight=-boxheights[k]
                else:
                    boxheight=boxheights[k]
                if spikes[j]:
                    # Validation rectangle
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], data[j]), exit_time*(1/hertz)*2, boxheight, edgecolor='green', facecolor='none', linewidth=0.5, zorder=1))
                    # Noise window left
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], -threshold), -(spikeduration/2), threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.5))
                    # Noise window right
                    rawdataplot.add_patch(Rectangle((time_seconds[j+exit_time], -threshold), spikeduration/2, threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.5))
                # Plot red boxes at the rejected spikes
                else:
                    # Validation rectangle
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], data[j]), exit_time*(1/hertz)*2, boxheight, edgecolor='red', facecolor='none', linewidth=0.5, zorder=1))
                    # Noise window left
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], -threshold), -(spikeduration/2), threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.5))
                    # Noise window right
                    rawdataplot.add_patch(Rectangle((time_seconds[j+exit_time], -threshold), spikeduration/2, threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.5))
                k+=1
        # Plot layout
        rawdataplot.title.set_text(f"Well {well} - MEA electrode {electrode} - Threshold: {threshold} - Spikes detected before validation: {np.sum(spikes_before_DMP)}, after: {np.sum(spikes)}")
        rawdataplot.set_xlabel("Time in seconds")
        rawdataplot.set_ylabel("Micro voltage")
        rawdataplot.set_xlim([time_seconds.min(), time_seconds.max()])
        rawdataplot.set_ylim([np.min(data)*1.5, np.max(data)*1.5])
    else:
        fig=None
    
    # Save the spike data to a .csv file
    if savedata:
        path = f'{outputpath}/spike_values'
        spike_x_values = time_seconds[spikes]
        spike_y_values = data[spikes]
        spike_indexes=np.arange(data.shape[0])
        spike_indexes=spike_indexes[spikes]
        spike_output = np.column_stack((spike_x_values, spike_y_values, spike_indexes))
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_spikes.csv', spike_output, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_spikes', spike_output)
        #print(f'calculated well: {well}, electrode: {electrode}')
    
    # Return the figure
    return fig