import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import copy

'''Spike validation - noisebased'''
def spike_validation(data,                      # Raw data from particular electrode
                     electrode,                 # Which electrode is getting analysed
                     threshold,                 # Spike detection threshold value
                     hertz,                     # Sampling frequency
                     spikeduration,             # Refractory period of the spike
                     exit_time_s,               # Time a spike gets to drop/rise in amplitude
                     amplitude_drop,            # How much a spike will have to drop/rise
                     plot_electrodes,           # If true, returns a matplotlib figure of the spike detection/validation process. Used for the GUI
                     electrode_amnt,            # How many electrodes per well
                     max_drop_amount,           # Maximum amount a spike can be required to drop/rise
                     outputpath,                # Path where to save the spike information
                     savedata=True,             # Whether to save the spike information or not
                     plot_rectangles=False      # Whether to plot the spike validation in the matplotlib figure
                     ):
    # Colors for possible matplotlib figure
    rawdatacolor='#bb86fc'
    
    
    '''Spike detection - detect spikes using the threshold and refractory period'''
    i = electrode

    # Identify points above and beneath the threshold
    above_threshold = data > threshold
    beneath_threshold = data < -threshold

    # Using the sampling frequency, calculate how many samples the refractory period lasts
    spikeduration_samples = int(spikeduration * hertz)

    measurements=data.shape[0]
    
    # Iterate over the potential spikes
    loopvalues=np.nonzero(np.logical_or(above_threshold, beneath_threshold))[0]
    for j in loopvalues:

        # Calculate the the upper and lower boundary
        lower_boundary = j-spikeduration_samples
        upper_boundary = j+spikeduration_samples
        # Make sure that the boundaries do not go out of bound of the dataset (e.g. when there is a spike in the first or last milisecond of the dataset)
        # Otherwise this will give an out of bounds error
        if lower_boundary < 0:
            lower_boundary = 0
        if upper_boundary > measurements:
            upper_boundary = measurements
        # Checks whether a threshold crossing is the absolute maximum value within the give timeframe (+-refractory period), if it is not, the peak will be removed
        # Maximum value in range
        maxvalue=(np.max(abs(data[(lower_boundary):(upper_boundary)])))
        # Amplitude of potential spike
        potential_spike=(abs(data[j]))
        # If the spike is lower than the maximum value nearby, remove the spike
        if maxvalue>potential_spike:
            above_threshold[j]=False
            beneath_threshold[j]=False
    # Combine positive and negative spikes
    spikes=np.logical_or(above_threshold, beneath_threshold)
    
    # Final check to remove spikes with unreasonable intervals
    loopvalues=np.nonzero(spikes)[0]
    for j in loopvalues:
        # Make sure we do not go out of bounds
        buffer=j+spikeduration_samples
        if buffer>measurements-1:
            buffer=measurements-1
        # If a spike has been found, set all values for in the refractory period to false
        spikes[j+1:buffer]=False

    time_seconds=np.arange(0, data.shape[0]/hertz, 1/hertz)

    '''Spike validation - check if the threshold crossing is not just noise'''
    # The exit time in amount of samples, used to establish the window around a spike
    exit_time = round(exit_time_s * hertz)

    surrounding_noise_window=0.010  # 10 ms window around the spike
    surrounding_noise_window_samples=int(0.010*hertz)
    loopvalues=np.nonzero(spikes)[0]
    boxheights=[]

    spikes_before_DMP=copy.deepcopy(spikes)

    if amplitude_drop != 0:
        # Iterate over all the spikes
        for j in loopvalues:
            # If the spikes i stoo close to the start or end of the measurement, in cannot be validated and will be removed
            if j+exit_time+surrounding_noise_window_samples>data.shape[0]:
                spikes[j]=False
                boxheights.append(0)
            elif j-exit_time-surrounding_noise_window_samples<0:
                spikes[j]=False
                boxheights.append(0)
            # The spike does not happen too close to the beginning or end of the measurement, so we can validate it
            else:
                # Determine the amount that the spike has to drop, based on the noise level surrounding the spike
                # Calculate the noise level around the spike, done by taking a +-refractory period window around the spike 
                noise_left=data[j-surrounding_noise_window_samples-exit_time:j-exit_time]
                noise_right=data[j+exit_time:j+surrounding_noise_window_samples+exit_time]

                # Only use values between the threshold
                noise_surround=np.append(noise_left, noise_right)
                noise_surround = noise_surround[(noise_surround > -threshold) & (noise_surround < threshold)]

                # Calculate the RMS of the surrounding noise
                drop_amount=amplitude_drop*(np.sqrt(np.mean(noise_surround**2)))

                # The amount a spike has to drop should not exceed max_drop_amount*threshold
                if drop_amount>max_drop_amount*threshold:
                    drop_amount=max_drop_amount*threshold

                # Array for visualization later
                boxheights.append(drop_amount)

                # Apply the validation method by checking whether the spike signal has any values below or above the 'box'
                # Spikes that have an amplitude of heightexception*threshold, do not have to drop amplitude in a short time
                
                # For positive spikes
                if data[j]>0:
                    if not(np.min(data[j-exit_time:j+exit_time+1])<(data[j]-drop_amount)):
                        spikes[j]=False
                else:
                # For negative spikes
                    if not(np.max(data[j-exit_time:j+exit_time+1])>(data[j]+drop_amount)):
                        spikes[j]=False
                
    # Calculate MEA electrode
    electrode = i % electrode_amnt + 1
    well = round(i / electrode_amnt + 0.505)

    '''Visualization of the spike detection/validation process - used by the GUI'''
    if plot_electrodes:
        # Create matplotlib figure
        fig=Figure(figsize=(3,1))
        rawdataplot=fig.add_subplot(111)

        # Plot raw data
        rawdataplot.plot(time_seconds, data, linewidth=0.5, zorder=-1, color=rawdatacolor)
        
        # Plot the threshold line
        rawdataplot.axhline(y=threshold, color='#737373', linestyle='-', linewidth=1) 
        rawdataplot.axhline(y=-threshold, color='#737373', linestyle='-', linewidth=1) 

        # Plot red dots at rejected spikes
        rawdataplot.scatter(time_seconds[spikes_before_DMP], data[spikes_before_DMP], color='red', marker='o', s=3)

        # Plot green dots at accepted spikes
        rawdataplot.scatter(time_seconds[spikes], data[spikes], color='green', marker='o', s=3)

        # Plot the 'boxes' used for validation
        if plot_rectangles:
            loopvalues=np.nonzero(spikes_before_DMP)[0]
            k=0
            for j in loopvalues:
                # Different direction of boxes for positive or negative spikes
                if data[j]>0:
                    boxheight=-boxheights[k]
                else:
                    boxheight=boxheights[k]

                if spikes[j]:
                    # Plot green boxes at the accepted spikes
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], data[j]), exit_time*(1/hertz)*2, boxheight, edgecolor='green', facecolor='none', linewidth=0.5, zorder=1))
                    # Noise window left
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], -threshold), -(surrounding_noise_window), threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.05))
                    # Noise window right
                    rawdataplot.add_patch(Rectangle((time_seconds[j+exit_time], -threshold), surrounding_noise_window, threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.05))
                else:
                    # Plot red boxes at the rejected spikes
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], data[j]), exit_time*(1/hertz)*2, boxheight, edgecolor='red', facecolor='none', linewidth=0.5, zorder=1))
                    # Noise window left
                    rawdataplot.add_patch(Rectangle((time_seconds[j-exit_time], -threshold), -(surrounding_noise_window), threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.05))
                    # Noise window right
                    rawdataplot.add_patch(Rectangle((time_seconds[j+exit_time], -threshold), surrounding_noise_window, threshold*2, edgecolor='violet', facecolor='violet', linewidth=0.5, alpha=0.05))
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
        spike_x_values = time_seconds[spikes]   # Timestamps of spikes
        spike_y_values = data[spikes]           # Amplitudes of spikes
        spike_indexes=np.arange(data.shape[0])  # Indexes of spikes
        spike_indexes=spike_indexes[spikes]
        # Combine in a signle array
        spike_output = np.column_stack((spike_x_values, spike_y_values, spike_indexes))
        # Save in both .csv and .npy format
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_spikes.csv', spike_output, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_spikes', spike_output)
    
    # Return the figure
    return fig