# Version 0.0.6

'''This file contains the entire MEA_analyzer pipeline, contained in functions'''
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import norm
import h5py
import os
import seaborn as sns


'''Gives the function of the the library'''
def get_library_function(message="This library analyzes MEA files!"):
    return message


'''Test the library functions with a nice message'''
def get_nice_message(message="Have fun with all our functions by analyzing your MEA files!"):
    return message


'''Bandpass function'''
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Call this one
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


'''Threshold function - returns threshold value'''
def threshold(data, hertz, stdevmultiplier, RMSmultiplier):
    measurements=data.shape[0]
    # The amount is samples needed to for a 50ms window is calculated
    windowsize = 0.05 * hertz
    windowsize = int(windowsize)
    # Create a temporary list that will contain 50ms windows of data
    windows = []

    # Iterate over the electrode data and create x amount of windows containing *windowsize* samples each
    # For this it is pretty important that the data consists of a multitude of 50ms seconds
    for j in range(0, measurements, windowsize):
        windows.append(data[j:j+windowsize])
    windows = np.array(windows) #convert into np.array
    # Create an empty list where all the data identified as spike-free noise will be stored
    noise=[]

    # Now iterate over every 50ms time window and check whether it is "spike-free noise"
    for j in range(0,windows.shape[0]):
        # Calculate the mean and standard deviation
        mu, std = norm.fit(windows[j])
        # Check whether the minimum/maximal value lies outside the range of x (defined above) times the standard deviation - if this is not the case, this 50ms box can be seen as 'spike-free noise'
        if not(np.min(windows[j])<(-1*stdevmultiplier*std) or np.max(windows[j])>(stdevmultiplier*std)):
            # 50ms has been identified as noise and will be added to the noise paremeter
            noise = np.append(noise, windows[j][:])

    # Calculate the RMS
    RMS=np.sqrt(np.mean(noise**2))
    threshold_value=RMSmultiplier*RMS

    # Calculate the % of the file that was noise
    noisepercentage=noise.shape[0]/data.shape[0]
    print(f'{noisepercentage*100}% of data identified as noise')
    return threshold_value


'''Open the HDF5 file'''
def openHDF5_SCA1(adress):
    with h5py.File(adress, "r") as file_data:
        # Returns HDF5 dataset objects
        dataset = file_data["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]
        # Convert to numpy array: (Adding [:] returns a numpy array)
        data=dataset[:]
    return data


'''Open the HDF5 file from nature'''
def openHDF5_Nature(adress):
    with h5py.File(adress, "r") as file_data:
        dataset=[]
        # Returns HDF5 dataset objects
        # Iterate over every single electrode, and add them to the list
        for electrode in file_data["Data"]["A3"]:
            dataset.append(file_data["Data"]["A3"][electrode])
        # Convert to numpy array: (Adding [:] returns a numpy array)
        data=np.array(dataset)
        data=np.reshape(data, (data.shape[0], data.shape[1]))
        print(data.shape)
    return data


'''Spike validation - noisebased'''
def spike_validation_DMP_noisebased(data, electrode, threshold, hertz, spikeduration, exit_time_s, amplitude_drop, plot_validation, electrode_amnt, heightexception, max_boxheight, filename):
    i = electrode
    # Identify points above and beneath the threshold
    above_threshold = data[i] > threshold
    beneath_threshold = data[i] < -threshold

    # Half of the spike duration in amount of samples, used to establish the window around a spike
    half_spikeduration_samples = int((spikeduration * hertz)/2)

    # Iterate over the data
    for j in range(0, data[i].shape[0]):
    # Check whether a positive or negative spike is detected at this datapoint
        if above_threshold[j] or beneath_threshold[j]:
            # Calculate the the upper and lower boundary
            lower_boundary = j-half_spikeduration_samples
            upper_boundary = j+half_spikeduration_samples
            # Make sure that the boundaries do not go out of bound of the dataset (e.g. when there is a spike in the first or last milisecond of the dataset)
            if lower_boundary < 0:
                lower_boundary = 0
            if upper_boundary > data[i].shape[0]:
                upper_boundary = data[i].shape[0]
            # Checks whether this is the absolute maximum value within the give timeframe, if it is not, the peak will be removed
            if (np.max(abs(data[i][(lower_boundary):(upper_boundary)])))>(abs(data[i][j])):
                above_threshold[j]=False
                beneath_threshold[j]=False
    for j in range(0, data[i].shape[0]):
        # Remove cases where 2 consecutive values are exactly the same, leading to a single ap registering as 2
        if (above_threshold[j] and above_threshold[j+1]):
            above_threshold[j]=False
        if (beneath_threshold[j] and beneath_threshold[j+1]):
            beneath_threshold[j]=False
    spikes=np.logical_or(above_threshold, beneath_threshold)
    # Final check to remove spikes with unreasonable intervals
    for j in range(0, spikes.shape[0]):
        if spikes[j]:
            # If a spike has been detected, all values in the next 1ms will be set to false
            spikes[j+1:j+half_spikeduration_samples]=False
    spikes_before_DMP=spikes.copy()
    time_seconds = np.arange(0, data[i].shape[0]) / hertz

    # Implement dynamic multi-phasic event detection method
    # The exit time in amount of samples, used to establish the window around a spike
    exit_time = round(exit_time_s * hertz)

    for j in range(0, data[i].shape[0]):
        # Checks whether there is a spike detected here
        if spikes[j]:
            # Check if there is a window of data to be checked after the spike. If the spike happens too close to the end of the measurement-
            # it cannot be confirmed, and will be removed.
            if j+exit_time+half_spikeduration_samples>data[i].shape[0]:
                spikes[j]=False
            if j-exit_time-half_spikeduration_samples<0:
                spikes[j]=False
            # Determine the amount that the spike has to drop, based on the noise level surrounding the spike
            noise_left=data[i][j-exit_time:j-half_spikeduration_samples-exit_time]
            noise_right=data[i][j+exit_time:j+half_spikeduration_samples+exit_time]
            noise_surround=np.append(noise_left, noise_right)
            drop_amount=amplitude_drop*np.std(noise_surround)
            # The amount a spike has to drop should not exceed 2*threshold
            if drop_amount>max_boxheight*threshold: drop_amount=max_boxheight*threshold
            # Check if the voltage has reached a minimal change value of 2*Treshold since the detected spike
            # For positive spikes
            if data[i][j]>0:
                if not(np.min(data[i][j-exit_time:j+exit_time+1])<(data[i][j]-drop_amount)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[i][j]>heightexception*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
            else:
            # For negative spikes
                if not(np.max(data[i][j-exit_time:j+exit_time+1])>(data[i][j]+drop_amount)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[i][j]<heightexception*-1*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
            
    # print(np.sum(spikes), np.sum(spikes_before_DMP))
    # Calculate MEA electrode
    electrode = i % electrode_amnt + 1
    well = round(i / electrode_amnt + 0.505)

    #Plot the data of the entire electrode
    if plot_validation:
        #%matplotlib widget
        # Plot the MEA signal
        plt.cla()

        # Enlarge size of plot figure
        plt.rcParams["figure.figsize"] = (22,5)

        time_seconds = np.arange(0, data[i].shape[0]) / hertz
        plt.plot(time_seconds, data[i], linewidth=0.2, zorder=-1)
        
        # Plot the threshold
        plt.axhline(y=threshold, color='k', linestyle='-', linewidth=1) 
        plt.axhline(y=-threshold, color='k', linestyle='-', linewidth=1) 
        
        # Calculate MEA electrode
        electrode = i % 12 + 1
        well = round(i / 12 + 0.505)

        # Plot red dots at rejected spikes
        plt.scatter(time_seconds[spikes_before_DMP], data[i][spikes_before_DMP], color='red', marker='o', s=3)

        # Plot green dots at accepted spikes
        plt.scatter(time_seconds[spikes], data[i][spikes], color='green', marker='o', s=3)

        # Plot layout
        plt.title(f"Well {well} - MEA electrode {electrode} - Threshold: {threshold} - Spikes detected before DMP: {np.sum(spikes_before_DMP)}, after: {np.sum(spikes)}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Micro voltage")
        plt.xlim([time_seconds.min(), time_seconds.max()])
        plt.ylim([np.min(data[i])-100, np.max(data[i])+100])
        plt.show()
    
    # Save the spike data to a .csv file
    if True:
        mapname=(os.path.basename(filename).split('/')[-1])
        mapname=mapname[:-3]
        mapname=mapname+'_values'
        filename=filename[:-3]
        filename=filename+"_values"
        if not os.path.exists(filename):
            os.makedirs(filename)
        path = f'{filename}/spike_values_DMP_noisebased'
        if not os.path.exists(path):
            os.makedirs(path)
        spike_x_values = time_seconds[spikes]
        spike_y_values = data[i][spikes]
        spike_indexes=np.arange(data.shape[1])
        spike_indexes=spike_indexes[spikes]
        spike_output = np.column_stack((spike_x_values, spike_y_values, spike_indexes))
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_spikes.csv', spike_output, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_spikes', spike_output)
        print(f'calculated well: {well}, electrode: {electrode}')

'''Spike detection and validation'''
def spike_validation_DMP(data, electrode, threshold, hertz, spikeduration, exit_time_s, amplitude_drop, plot_validation, electrode_amnt, filename):
    # Identify points above and beneath the threshold
    above_threshold = data > threshold
    beneath_threshold = data < -threshold

    # Half of the spike duration in amount of samples, used to establish the window around a spike
    half_spikeduration_samples = int((spikeduration * hertz)/2)

    # Iterate over the data
    for j in range(0, data.shape[0]):
    # Check whether a positive or negative spike is detected at this datapoint
        if above_threshold[j] or beneath_threshold[j]:
            # Calculate the the upper and lower boundary
            lower_boundary = j-half_spikeduration_samples
            upper_boundary = j+half_spikeduration_samples
            # Make sure that the boundaries do not go out of bound of the dataset (e.g. when there is a spike in the first or last milisecond of the dataset)
            if lower_boundary < 0:
                lower_boundary = 0
            if upper_boundary > data.shape[0]:
                upper_boundary = data.shape[0]
            # Checks whether this is the absolute maximum value within the give timeframe, if it is not, the peak will be removed
            if (np.max(abs(data[(lower_boundary):(upper_boundary)])))>(abs(data[j])):
                above_threshold[j]=False
                beneath_threshold[j]=False
    for j in range(0, data.shape[0]):
        # Remove cases where 2 consecutive values are exactly the same, leading to a single ap registering as 2
        if (above_threshold[j] and above_threshold[j+1]):
            above_threshold[j]=False
        if (beneath_threshold[j] and beneath_threshold[j+1]):
            beneath_threshold[j]=False
    spikes=np.logical_or(above_threshold, beneath_threshold)
    # Final check to remove spikes with unreasonable intervals
    for j in range(0, spikes.shape[0]):
        if spikes[j]:
            # If a spike has been detected, all values in the next 1ms will be set to false
            spikes[j+1:j+half_spikeduration_samples]=False
    spikes_before_DMP=spikes.copy()
    time_seconds = np.arange(0, data.shape[0]) / hertz

    # Implement dynamic multi-phasic event detection method
    # The exit time in amount of samples, used to establish the window around a spike
    exit_time = round(exit_time_s * hertz)
    thresholdmultiplier=amplitude_drop
    heightexception=2
    for j in range(0, data.shape[0]):
        # Checks whether there is a spike detected here
        if spikes[j]:
            # Check if there is a window of data to be checked after the spike. If the spike happens too close to the start or end of the measurement-
            # it cannot be confirmed, and will be removed.
            if j+exit_time>data.shape[0]:
                spikes[j]=False
            if j-exit_time<0:
                spikes[j]=False
            # Check if the voltage has reached a minimal change value of 2*Treshold since the detected spike
            # For positive spikes
            if data[j]>0:
                if not(np.min(data[j-exit_time:j+exit_time+1])<(data[j]-thresholdmultiplier*threshold)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[j]>heightexception*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
            else:
            # For negative spikes
                if not(np.max(data[j-exit_time:j+exit_time+1])>(data[j]+thresholdmultiplier*threshold)):
                    # Spikes that have an amplitude of twice the threshold, do not have to drop amplitude in a short time
                    if not(data[j]<heightexception*-1*threshold):
                        # If not, the spike will be removed
                        spikes[j]=False
    if plot_validation:
        # Plot the MEA signal
        plt.cla()

        time_seconds = np.arange(0, data.shape[0]) / hertz
        plt.plot(time_seconds, data, linewidth=0.2, zorder=-1)
        
        # Plot the threshold
        plt.axhline(y=threshold, color='k', linestyle='-', linewidth=1) 
        plt.axhline(y=-threshold, color='k', linestyle='-', linewidth=1) 

        # Plot red dots at rejected spikes
        plt.scatter(time_seconds[spikes_before_DMP], data[spikes_before_DMP], color='red', marker='o', s=3)

        # Plot green dots at accepted spikes
        plt.scatter(time_seconds[spikes], data[spikes], color='green', marker='o', s=3)
        
        # Calculate MEA electrode
        electrode_nr = electrode % electrode_amnt + 1
        well = round(electrode / electrode_amnt + 0.505)

        # Plot layout
        # Enlarge size of plot figure
        plt.rcParams["figure.figsize"] = (10,5)
        plt.title(f"Dataset: Nature - Well: {well} - Electrode: {electrode_nr} - Threshold: {threshold} - Spikes detected before DMP: {np.sum(spikes_before_DMP)}, after: {np.sum(spikes)}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Micro voltage")
        plt.xlim([time_seconds.min(), time_seconds.max()])
        plt.ylim([np.min(data)*1.2, np.max(data)*1.2])
        plt.show()
    # Save the spike data to a .csv file
    if True:
        filename=filename[:-3]
        filename=filename+"_values"
        if not os.path.exists(filename):
            os.makedirs(filename)
        path = f'{filename}/spike_values_DMP'
        if not os.path.exists(path):
            os.makedirs(path)
        spike_x_values = time_seconds[spikes]
        spike_y_values = data[spikes]
        spike_indexes=np.arange(data.shape[0])
        spike_indexes=spike_indexes[spikes]
        spike_output = np.column_stack((spike_x_values, spike_y_values, spike_indexes))
        electrode = electrode % electrode_amnt + 1
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_spikes.csv', spike_output, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_spikes', spike_output)
        print(f'calculated well: {well}, electrode: {electrode}')
    return spikes
                

def burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, validation_method, filename):
    electrode_number=electrode
    # Calculate the well and electrode values to load in the spikedata
    well = round(electrode_number / electrode_amnt + 0.505)
    electrode = electrode_number % electrode_amnt + 1

    
    filename=filename[:-3]
    filename=filename+"_values"
    path=f'{filename}/spike_values_{validation_method}'
    spikedata=np.load(f'{path}/well_{well}_electrode_{electrode}_spikes.npy')
    # Calculate the inter-spike intervals
    ISI=[]
    if spikedata.shape[0]<2:
        pass
    else:
        for i in range(spikedata.shape[0]-1):
            time_to_next_spike = (spikedata[i+1][0]) - (spikedata[i][0])
            ISI.append(time_to_next_spike)
        ISI=np.array(ISI)
        # Convert ISIs to miliseconds
        ISI=ISI*1000
    if len(ISI)>2:
        use_ISIth2=False
        ISIth2_max=maxISI_outliers
        use_chiappalone=False
        use_pasquale=False

        # Plot a smooth histogram using kernel density estimation
        plt.cla()
        plt.clf()
        output=sns.displot(ISI, alpha=0.2, edgecolor=None, kde=True, color='blue', log_scale=True, kde_kws={'gridsize': 100, 'bw_adjust':kde_bandwidth})
        for ax in output.axes.flat:
            for line in ax.lines:
                x = (line.get_xdata())
                y = (line.get_ydata())
        # line = output.get_lines()[0]
        # x, y = line.get_data()
        x=np.array(x)
        y=np.array(y)
        plt.cla
        plt.clf
        
        # Find the peaks
        peaks=[]
        for i in range(smallerneighbours, len(x)-smallerneighbours):
            # Check if this value is higher than its specified amount of neighbours
            if np.all(y[i]>y[i-smallerneighbours:i]) and np.all(y[i]>y[i+1:i+smallerneighbours+1]):
                peaks.append(i)
        for peak in peaks:
            peakneighbours=np.arange(peak-smallerneighbours, peak+smallerneighbours+1)
            plt.plot(x[peakneighbours], y[peakneighbours], color="red")

        # Check if 2 peaks were detected
        if len(peaks)!=2:
            ISIth1=default_threshold #miliseconds
            valid_peaks=False
            use_chiappalone=True
            print(f"More/less than 2 valid peaks detected in smoothed ISI histogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
        # Check whether peak 1 has happened before default_threshold
        elif x[peaks[0]]>default_threshold:
            ISIth1=default_threshold
            valid_peaks=False
            use_chiappalone=True
            print(f"No valid peak detected before {default_threshold}ms in smoothed ISI hisogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
        else:
            valid_peaks=True
            # Calculate the minimum y value between the 2 peaks
            g_min=np.min(y[peaks[0]:peaks[1]])
            # Retrieve the x value for this valley
            peak1=y[peaks[0]]
            peak2=y[peaks[1]]      
            # Calculate the void parameter
            void=1-(g_min/(np.sqrt(peak1*peak2)))
            print(f"Void parameter: {void}")
            valley_x=x[np.where(y==g_min)]
            logvalley=np.mean(valley_x)
            if logvalley<default_threshold:
                ISIth1=logvalley
                use_chiappalone=True
                use_ISIth2=False
                print(f"Valley detected before {default_threshold}ms in smoothed ISI histogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
            else:
                ISIth2=logvalley
                ISIth1=default_threshold
                if ISIth2>ISIth2_max:
                    ISIth2=ISIth2_max
                use_pasquale=True
                use_ISIth2=True
                print(f"2 valid peaks detected in smoothed ISI histogram at well {well}, electrode {electrode}, using advanced burst detection algorithm with ISIth1 set at {ISIth1} and ISIth2 set at {ISIth2}")
            
        # Plot the smoothed histogram
        if True:
            plt.scatter(x[peaks], y[peaks], color="red")
            if valid_peaks:
                plt.scatter(valley_x, y[np.where(y==g_min)])
                if use_ISIth2:
                    plt.axvline(ISIth2, color='blue')
            plt.axvline(ISIth1, color='green')
            plt.scatter
            plt.gca().set_xscale("log")
            plt.title(f"Well: { well}, Electrode: {electrode}")
            plt.xlabel("Inter-spike interval")
            plt.ylabel("Amount")
            plt.show()

        # Apply the threshold
        burst_cores=[]
        burst_spikes=[]

        # The seemingly random +1 and -1 in the following code
        # are because the ISI list is not equal in length to the spike_values list
        # The ISIs are 1 value shorter because an ISI is calculated using 2 spikes.
        # This makes comparing and indexing the 2 a little bit difficult.
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
                    # Add all the spikes in the burst to the burst_spikes array
                    for l in range(min_spikes_burstcore):
                        burst_spikes.append(spikedata[i+l])
                    # Move the index to the end of the burst core
                    i+=minspikes_burst-1
                    # Loop through each spike and check if it should be added to the burst
                    # Keep increasing the steps (i) while doing this
                    while ISI[i]<ISIth1:
                        burst_spikes.append(spikedata[i+1])
                        # If you have reached the end of the list, stop
                        if i+1==len(ISI):
                            i+=1
                            break
                        i+=1
                    # Add the found values to the list
                    end_burst=spikedata[i][0]
                    burst_cores.append([start_burst, end_burst])
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
                    for l in range(min_spikes_burstcore):
                        burst_spikes.append(spikedata[i+l])
                    # Start moving backwards to append any spikes distanced less than ISIth2
                    j=i-1
                    while j>=0 and ISI[j]<ISIth2:
                        start_burst=spikedata[j][0]
                        burst_spikes.append(spikedata[j])
                        j-=1
                    # Move the index to the end of the burst
                    i+=minspikes_burst-1
                    # Loop through each spike and check if it should be added to the burst
                    # Keep increasing the steps (i) while doing this
                    while ISI[i]<ISIth2:
                        burst_spikes.append(spikedata[i+1])
                        # If you have reached the end of the list, stop
                        if i+1==len(ISI):
                            i+=1
                            break
                        i+=1
                    # Add the found values to the list
                    end_burst=spikedata[i][0]
                    burst_cores.append([start_burst, end_burst])
                i+=1

            # Calculate the average burst length
            burst_len=[]
            for k in burst_cores:
                burst_len.append(k[1]-k[0])
            if len(burst_len)==0:
                avg=0
            else:
                avg=np.mean(burst_len)
            print(f"Average burst length: {avg}")
                                            

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
        if True:
            #%matplotlib widget
            plt.cla()
            plt.clf()

            # Plot the data
            time_seconds = np.arange(0, data[electrode_number].shape[0]) / hertz
            plt.plot(time_seconds, data[electrode_number], linewidth=0.2, zorder=-1)
            
            # Enlarge size of plot figure
            plt.rcParams["figure.figsize"] = (22,5)

            # Plot the bursts
            for burst in range(len(burst_cores)):
                burst_startx=int(burst_cores[burst][0]*hertz)
                burst_endx=int(burst_cores[burst][1]*hertz)
                plt.plot(time_seconds[burst_startx:burst_endx], data[electrode_number][burst_startx:burst_endx], color='red', linewidth=0.2, alpha=0.5)
        
            spikes=np.zeros(data.shape[1], dtype=bool)
            spike_indexes=spikedata[:,2]
            spike_indexes=spike_indexes.astype(int)
            spikes[spike_indexes]=True
            plt.scatter(time_seconds[spikes], data[electrode_number][spikes], color='green', marker='o', s=3, zorder=1)

            # Plot the spike included in bursts
            if len(burst_spikes)>0:
                burst_spikes=np.array(burst_spikes)
                plt.scatter(burst_spikes[:,0], burst_spikes[:,1], color="purple", marker='o', s=3, zorder=2)

            # Plot layout
            if use_ISIth2:
                thresholdtext=f"Advanced burst detection, ISIth1: {ISIth1}, ISIth2: {ISIth2}"
            else:
                thresholdtext=f"Default burst detection, ISIth1: {ISIth1}"
            plt.title(f"Well {well} - MEA electrode {electrode}, bursts detected: {len(burst_cores)}, index: {electrode_number}, min_spikes: {minspikes_burst}, {thresholdtext}")
            plt.xlabel("Time in seconds")
            plt.ylabel("Micro voltage")
            plt.xlim([time_seconds.min(), time_seconds.max()])
            plt.ylim([np.min(data[electrode_number])-100, np.max(data[electrode_number])+100])
            plt.show()
    else:
        print(f"No burst detection possible for well {well}, electrode {electrode} - not enough values")
        burst_spikes=[]
        burst_cores=[[]]
    # Save the spike data to a .csv file
    if True:
        if not os.path.exists(filename):
            os.makedirs(filename)
        path = f'{filename}/burst_values'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_burst_spikes.csv', burst_spikes, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_burst_spikes', burst_spikes)
        np.savetxt(f'{path}/well_{well}_electrode_{electrode}_burst_cores.csv', burst_cores, delimiter = ",")
        np.save(f'{path}/well_{well}_electrode_{electrode}_burst_cores', burst_cores)
        print(f'calculated well: {well}, electrode: {electrode}')

''' Get all of the MEA files in a folder '''
def get_files(MEA_folder):
    # Get all files from MEA folder 
    all_files = os.listdir(MEA_folder)

    MEA_files = []
    # Get all HDF5 files
    for file in all_files:
        # Convert file to right format
        file = "{0}/{1}".format(MEA_folder, file)

        # Check if file is HDF5 file
        if not file.endswith(".h5"):
            print("'{0}' is no HDF5 file!".format(file))
            continue

        # Check if HDF5 file can be opened
        try:
            h5_file = h5py.File(file, "r")
        except:
            print("'{0}' can not be opened as HDF5 file!".format(file))
            continue

        # Check if HDF5 MEA dataset object exist
        try:
            h5_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]
        except:
            print("'{0}' has no MEA dataset object!".format(file))
            continue

        # Create list with all MEA files
        MEA_files.append(file)

    # Print all HDF5 MEA files
    print("\nList with all HDF5 MEA files:")
    for file in MEA_files:
        print(file)
    
    return MEA_files

def raster(electrodes, electrode_amnt, samples, hertz, validation_method, filename):
    print("starting")
    # Check which electrodes are given, and how these will be plotted
    i=0
    
    filename=filename[:-3]
    filename=filename+"_values"
    while i < len(electrodes):
        well_spikes=[]
        burst_spikes=[]
        # Collect all the data from a single well
        while i<len(electrodes): # and ((electrodes[i])%electrode_amnt)!=0:
            electrode = electrodes[i] % electrode_amnt + 1
            well = round(electrodes[i] / electrode_amnt + 0.505)
            path=f'spike_values_{validation_method}'
            spikedata=np.load(f'{filename}/{path}/well_{well}_electrode_{electrode}_spikes.npy')
            path='burst_values'
            burstdata=np.load(f'{filename}/{path}/well_{well}_electrode_{electrode}_burst_spikes.npy')
            well_spikes.append(spikedata[:,0])
            if len(burstdata)>0:
                burst_spikes.append(burstdata[:,0])
            else:
                burst_spikes.append([])
            if ((electrodes[i]+1)%electrode_amnt)==0: break
            i+=1
        amount_of_electrodes=len(burst_spikes)
        plt.cla()
        plt.rcParams["figure.figsize"] = (22,5)
        end_electrode=((electrodes[i-1]+1)%12)+1
        start_electrode = end_electrode-amount_of_electrodes
        lineoffsets1=np.arange(start_electrode+1, end_electrode+1)
        plt.eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        plt.eventplot(burst_spikes, alpha=0.5, color='red', lineoffsets=lineoffsets1)
        plt.xlim([0,samples/hertz])
        plt.ylim([start_electrode, end_electrode+1])
        plt.yticks(lineoffsets1)
        plt.title(f"Well {well}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Electrode")
        plt.show()
        i+=1

'''Analyse a single electrode'''
def analyse_electrode(filename,                             # Where is the data file stored
                      electrodes,                           # Which electrodes do you want to analyze
                      hertz,                                # What is the sampling frequency of the MEA
                      validation_method="DMP_noisebased",   # Which validation method do you want to use, possible: "DMP", "DMP_noisebased"
                      low_cutoff=200,                       # The low_cutoff for the bandpass filter
                      high_cutoff=3500,                     # The high_cutoff for the bandpass filter
                      order=2,                              # The order for the bandpass filter
                      spikeduration=0.002,                  # The amount of time only 1 spike should be registered, aka refractory period
                      exit_time_s=0.00024,                  # The amount of time a spike gets to drop amplitude in the validation methods
                      amplitude_drop_threshold=2,           # The amount of amplitude the spikes has to drop, this value will be used in the DMP method and multiplied with the threshold
                      plot_validation=True,                 # Do you want to plot the detected spikes
                      well_amnt=24,                         # The amount of wells present in the MEA
                      electrode_amnt=12,                    # The amount of electrodes per well
                      kde_bandwidth=1,                      # The bandwidth of the kernel density estimate
                      smallerneighbours=10,                 # The amount of smaller neighbours a peak should have before being considered as one
                      minspikes_burst=5,                    # The minimal amount of spikes a burst can have
                      maxISI_outliers=1000,                 # The maximal ISIth2 that can be used in burst detection
                      default_threshold=100,                # The default value for ISIth1
                      heightexception=2,                    # Multiplied with the spike detection threshold, if a spike exceeds this value, it does not have to drop amplitude fast to be validated
                      max_boxheight=2,                      # Multiplied with the spike detection threshold, will be the maximal value the box for DMP_noisebased validation can be
                      amplitude_drop_sd=5,                  # Multiplied with the SD of surrounding noise, will be the boxheight for DMP_noisebased validation
                      stdevmultiplier=5,                    # The amount of SD a value needs to be from the mean to be considered a possible spike in the threshold detection
                      RMSmultiplier=5                       # Multiplied with the RMS of the spike-free noise, used to determine the threshold
                      ):                 
    # Open the data file
    data=openHDF5_SCA1(filename)
    print("MEA-analyzer. Developed by Jesse Antonissen and Joram van Beem for the CureQ research consortium")
    for electrode in electrodes:
        print(f'Analyzing {filename} - Electrode: {electrode}')
        # Filter the data
        print("Applying bandpass filter")
        data[electrode]=butter_bandpass_filter(data[electrode], low_cutoff, high_cutoff, hertz, order)
        # Calculate the threshold
        print(f'Calculating the threshold')
        threshold_value=threshold(data[electrode], hertz, stdevmultiplier, RMSmultiplier)
        print(f'Threshold for electrode {electrode} set at: {threshold_value}')
        # Calculate spike values
        print('Validating the spikes')
        if validation_method=="DMP":
            spike_validation_DMP(data[electrode], electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_threshold, plot_validation, electrode_amnt, filename)
        elif validation_method=="DMP_noisebased":
            spike_validation_DMP_noisebased(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_validation, electrode_amnt, heightexception, max_boxheight, filename)
        else:
            raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
        print("Detecting bursts")
        burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, validation_method, filename)
    raster(electrodes, electrode_amnt, data.shape[1], hertz, validation_method, filename)

'''Analyse an entire well, this allows for raster plots to be created'''
def analyse_well(filename,                                  # Where is the data file stored
                      well,                                 # Which wells do you want to analyze
                      hertz,                                # What is the sampling frequency of the MEA
                      validation_method="DMP_noisebased",   # Which validation method do you want to use, possible: "DMP", "DMP_noisebased"
                      low_cutoff=200,                       # The low_cutoff for the bandpass filter
                      high_cutoff=3500,                     # The high_cutoff for the bandpass filter
                      order=2,                              # The order for the bandpass filter
                      spikeduration=0.002,                  # The amount of time only 1 spike should be registered, aka refractory period
                      exit_time_s=0.00024,                  # The amount of time a spike gets to drop amplitude in the validation methods
                      amplitude_drop_threshold=2,           # The amount of amplitude the spikes has to drop, this value will be used in the DMP method and multiplied with the threshold
                      plot_validation=True,                 # Do you want to plot the detected spikes
                      well_amnt=24,                         # The amount of wells present in the MEA
                      electrode_amnt=12,                    # The amount of electrodes per well
                      kde_bandwidth=1,                      # The bandwidth of the kernel density estimate
                      smallerneighbours=10,                 # The amount of smaller neighbours a peak should have before being considered as one
                      minspikes_burst=5,                    # The minimal amount of spikes a burst can have
                      maxISI_outliers=1000,                 # The maximal ISIth2 that can be used in burst detection
                      default_threshold=100,                # The default value for ISIth1
                      heightexception=2,                    # Multiplied with the spike detection threshold, if a spike exceeds this value, it does not have to drop amplitude fast to be validated
                      max_boxheight=2,                      # Multiplied with the spike detection threshold, will be the maximal value the box for DMP_noisebased validation can be
                      amplitude_drop_sd=5,                  # Multiplied with the SD of surrounding noise, will be the boxheight for DMP_noisebased validation
                      stdevmultiplier=5,                    # The amount of SD a value needs to be from the mean to be considered a possible spike in the threshold detection
                      RMSmultiplier=5                       # Multiplied with the RMS of the spike-free noise, used to determine the threshold
                      ):                 
    electrodes=np.arange((well-1)*12, well*12)
    print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")
    analyse_electrode(filename, electrodes, hertz, validation_method, low_cutoff, high_cutoff, order, spikeduration, exit_time_s, amplitude_drop_threshold, plot_validation, well_amnt, electrode_amnt, kde_bandwidth, smallerneighbours,
                      minspikes_burst, maxISI_outliers, default_threshold, heightexception, max_boxheight, amplitude_drop_sd, stdevmultiplier, RMSmultiplier)
    
