# Version 0.0.9

'''This file contains the entire MEA_analyzer pipeline, contained in functions'''
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import norm
import h5py
import os
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels
from skimage import filters
import plotly.graph_objects as go
from functools import partial
import time
from KDEpy import FFTKDE


'''Test the library functions with a nice message'''
def get_nice_message(message="Enjoy your day!"):
    print(message)
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
def threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion):
    measurements=data.shape[0]
    # The amount is samples needed to for a 50ms window is calculated
    windowsize = 0.05 * hertz
    windowsize = int(windowsize)
    # Create a temporary list that will contain 50ms windows of data
    windows = []

    # Iterate over the electrode data and create x amount of windows containing *windowsize* samples each
    # For this it is pretty important that the data consists of a multitude of 50ms seconds
    for j in range(0, int(measurements), windowsize):
        windows.append(data[j:j+windowsize])
    windows = np.array(windows) #convert into np.array
    # Create an empty list where all the data identified as spike-free noise will be stored
    noise=[]

    # Now iterate over every 50ms time window and check whether it is "spike-free noise"
    # If all of the electrode should be used to determine the noise window
    if threshold_portion==1:
        loopvalues=range(0,windows.shape[0])
    # If this is not the case, only use a portion of the data to determine the noise level
    else:
        loopvalues=range(0,windows.shape[0])
        step=int(1/threshold_portion)
        loopvalues=loopvalues[::step]
    for j in loopvalues:
        # Calculate the mean and standard deviation
        #mu, std = norm.fit(windows[j])
        std=np.std(windows[j])
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

def fast_threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion):
    measurements=data.shape[0]
    # The stepsize needed for a 50ms window is calculated
    stepsize = int(0.05 * hertz)
    total_steps=int(measurements/stepsize)

    loopvalues=range(0,total_steps)
    step=int(1/threshold_portion)
    loopvalues=loopvalues[::step]
    noise=np.array([])

    for j in loopvalues:
        # Calculate the mean and standard deviation
        #mu, std = norm.fit(windows[j])
        std=np.std(data[j*stepsize:(j+1)*stepsize])
        # Check whether the minimum/maximal value lies outside the range of x (defined above) times the standard deviation - if this is not the case, this 50ms box can be seen as 'spike-free noise'
        if not(np.min(data[j*stepsize:(j+1)*stepsize])<(-1*stdevmultiplier*std) or np.max(data[j*stepsize:(j+1)*stepsize])>(stdevmultiplier*std)):
            # 50ms has been identified as noise and will be added to the noise paremeter
            noise = np.append(noise, data[j*stepsize:(j+1)*stepsize])

    # Calculate the RMS
    RMS=np.sqrt(np.mean(noise**2))
    threshold_value=RMSmultiplier*RMS

    # Calculate the % of the file that was noise
    noisepercentage=noise.shape[0]/measurements
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

'''Cut the data into smaller parts'''
def cut_data(data, parts, electrodes_per_well):
    total_electrodes=data.shape[0]
    total_wells=int(total_electrodes/electrodes_per_well)
    measurements=data.shape[1]
    partsize=int(measurements/10)
    new_data=[]
    for well in range(total_wells):
        for part in range(parts):
            for electrode in range(electrodes_per_well):
                electrode_id=well*12+electrode
                #print(f"Creating new well from electrode_id {electrode_id} part {part}")
                new_data.append(data[electrode_id, part*partsize:(part+1)*partsize])
    return np.array(new_data)



'''Spike validation - noisebased'''
def spike_validation_DMP_noisebased(data, electrode, threshold, hertz, spikeduration, exit_time_s, amplitude_drop, plot_electrodes, electrode_amnt, heightexception, max_boxheight, filename):
    i = electrode
    # Identify points above and beneath the threshold
    above_threshold = data[i] > threshold
    beneath_threshold = data[i] < -threshold

    # Half of the spike duration in amount of samples, used to establish the window around a spike
    half_spikeduration_samples = int((spikeduration * hertz)/2)
    measurements=data[i].shape[0]
    # Iterate over the data
    loopvalues=np.nonzero(np.logical_or(above_threshold, beneath_threshold))[0]
    for j in loopvalues:
        # Check whether a positive or negative spike is detected at this datapoint
        # Calculate the the upper and lower boundary
        lower_boundary = j-half_spikeduration_samples
        upper_boundary = j+half_spikeduration_samples
        # Make sure that the boundaries do not go out of bound of the dataset (e.g. when there is a spike in the first or last milisecond of the dataset)
        if lower_boundary < 0:
            lower_boundary = 0
        if upper_boundary > measurements:
            upper_boundary = measurements
        # Checks whether this is the absolute maximum value within the give timeframe, if it is not, the peak will be removed
        if (np.max(abs(data[i][(lower_boundary):(upper_boundary)])))>(abs(data[i][j])):
            above_threshold[j]=False
            beneath_threshold[j]=False
    spikes=np.logical_or(above_threshold, beneath_threshold)
    loopvalues=np.nonzero(spikes)[0]
    # Final check to remove spikes with unreasonable intervals
    for j in loopvalues:
        # If a spike has been detected, all values in the next 1ms will be set to false
        # Make sure we do not go out of bounds
        buffer=j+half_spikeduration_samples
        if buffer>measurements-1:
            buffer=measurements-1
        spikes[j+1:buffer]=False
    spikes_before_DMP=spikes.copy()
    time_seconds = np.arange(0, data[i].shape[0]) / hertz

    # Implement dynamic multi-phasic event detection method
    # The exit time in amount of samples, used to establish the window around a spike
    exit_time = round(exit_time_s * hertz)

    loopvalues=np.nonzero(spikes)[0]
    for j in loopvalues:
        # Check if there is a window of data to be checked before and after the spike. If the spike happens too close to the start/end of the measurement-
        # it cannot be confirmed, and will be removed.
        if j+exit_time+half_spikeduration_samples>data[i].shape[0]:
            spikes[j]=False
        elif j-exit_time-half_spikeduration_samples<0:
            spikes[j]=False
        else:
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
            
    # Calculate MEA electrode
    electrode = i % electrode_amnt + 1
    well = round(i / electrode_amnt + 0.505)

    #Plot the data of the entire electrode
    #if plot_validation:
    if plot_electrodes:
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
        path = f'{filename}/spike_values'
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
def spike_validation_DMP(data, electrode, threshold, hertz, spikeduration, exit_time_s, amplitude_drop, plot_electrodes, electrode_amnt, filename):
    # Identify points above and beneath the threshold
    above_threshold = data > threshold
    beneath_threshold = data < -threshold

    # Half of the spike duration in amount of samples, used to establish the window around a spike
    half_spikeduration_samples = int((spikeduration * hertz)/2)

    # Iterate over the data
    loopvalues=np.argwhere(np.logical_or(above_threshold, beneath_threshold)==True)
    for j in loopvalues:
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
    if plot_electrodes:
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
        path = f'{filename}/spike_values'
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
                

def burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, filename, plot_electrodes):
    electrode_number=electrode
    # Calculate the well and electrode values to load in the spikedata
    well = round(electrode_number / electrode_amnt + 0.505)
    electrode = electrode_number % electrode_amnt + 1

    # Get the correct foldername to read and store values
    filename=filename[:-3]
    filename=filename+"_values"
    path=f'{filename}/spike_values'
    spikedata=np.load(f'{path}/well_{well}_electrode_{electrode}_spikes.npy')

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
    if len(ISI)>=minspikes_burst:
        # Reset variables to the base values
        use_ISIth2=False
        ISIth2_max=maxISI_outliers
        use_chiappalone=False
        use_pasquale=False

        # Plot a smooth histogram using kernel density estimation
        plt.cla()
        plt.clf()
        plt.close()
        output=sns.displot(ISI, alpha=0.2, edgecolor=None, kde=True, color='blue', log_scale=True, kde_kws={'gridsize': 100, 'bw_adjust':kde_bandwidth})
        for ax in output.axes.flat:
            for line in ax.lines:
                x = (line.get_xdata())
                y = (line.get_ydata())
        x=np.array(x)
        y=np.array(y)
        
        # Find the peaks
        peaks=[]
        for i in range(smallerneighbours, len(x)-smallerneighbours):
            # Check if this value is higher than its specified amount of neighbours
            if np.all(y[i]>y[i-smallerneighbours:i]) and np.all(y[i]>y[i+1:i+smallerneighbours+1]):
                peaks.append(i)
        for peak in peaks:
            peakneighbours=np.arange(peak-smallerneighbours, peak+smallerneighbours+1)
            plt.plot(x[peakneighbours], y[peakneighbours], color="red")

        # Remove peaks before 10ms, as these are very likely false
        too_low_peaks=[]
        for i in range(len(peaks)):
            if x[peaks[i]]<10:
                too_low_peaks.append(i)
        for j in sorted(too_low_peaks, reverse=True):
            print(f"Removed peak with value: {x[peaks[j]]} because value is too low")
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
            print(found_peak1, found_peak2)
            if (found_peak1 and found_peak2) and peak1 !=peak2:
                peaks=[peak1, peak2]
                print(f"More than 2 peaks found, but able to identify 2 peaks close to the default values at peak 1: {x[peak1]} and peak 2 {x[peak2]}")
            else:
                print(f"More than 2 peaks found")

        # Check if 2 peaks were detected
        if len(peaks)!=2:
            ISIth1=default_threshold #miliseconds
            valid_peaks=False
            use_chiappalone=True
            print(f"Less than 2 valid peaks detected in smoothed ISI histogram at well {well}, electrode {electrode}, using default burst detection algorithm with ISIth1 set at {ISIth1}")
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
            # void=1-(g_min/(np.sqrt(peak1*peak2)))
            # print(f"Void parameter: {void}")
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
        if plot_electrodes:
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
        else:
            plt.cla()
            plt.clf()
            plt.close()

        # Apply the threshold
        burst_cores=[]
        burst_spikes=[]
        burst_counter=0

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
        burst_cores=[]
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

'''Create raster plots for all the electrodes'''
def raster(electrodes, electrode_amnt, samples, hertz, filename):
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
            path=f'spike_values'
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

'''Takes two burst values and calculates whether they overlap or not'''
def overlap(burst1, burst2):
    if burst1[1]<burst2[0] or burst2[1]<burst1[0]:
        return False
    else:
        return True

def network_burst_detection(filename, wells, electrode_amnt, measurements, hertz, min_channels, threshold_method):
    # Convert the filename so the algorithm can look through the folders
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
    burstpath=f'{filename}/burst_values'

    min_channels=round(min_channels*electrode_amnt)
    print(f"Minimal amount of channels required for network burst: {min_channels}")
    total_network_burst=[]
    participating_bursts=[]
    
    for well in wells:
        fig, ax = plt.subplots(4,1, sharex=True, figsize=(16,5), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        ax[1].set_ylabel("Burst\namount")
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
        lineoffsets1=np.arange(1,13)
        ax[0].eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        ax[0].eventplot(burst_spikes, alpha=0.25, color='red', lineoffsets=lineoffsets1)

        # Plot the burst frequency
        time_seconds = np.arange(0, measurements) / hertz
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
        if len(burst_spikes_for_kde)>0:
            y = FFTKDE(bw=0.05, kernel='gaussian').fit(burst_spikes_for_kde).evaluate(grid_points=np.arange(0, data_time, 0.001))
            x=np.arange(0, data_time, 0.001)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            ax[2].clear()
            ax[2].plot(x,y)

        
        if len(burst_spikes_for_kde)>0:     # Check if there are bursts
            # Plot the spike frequency of spikes contained in bursts
            y = FFTKDE(bw=0.05, kernel='gaussian').fit(burst_spikes_for_kde).evaluate(grid_points=np.arange(0, data_time, 0.001))
            x=np.arange(0, data_time, 0.001)
            ax[3].clear()
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            # Determine the threshold
            if threshold_method=='yen':
                threshold=filters.threshold_yen(y, nbins=1000)
            elif threshold_method=='otsu':
                threshold=filters.threshold_otsu(y, nbins=1000)
            else:
                raise ValueError(f"\"{threshold_method}\" is not a valid thresholding method")
            print(f"Threshold using {threshold_method} method set at: {threshold}")
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
                    #print(f"found overlap with burst {i} and {i+1}")
                    # Mark the shortest burst for removal
                    if (network_burst_cores[i][1]-network_burst_cores[i][0])<(network_burst_cores[i+1][1]-network_burst_cores[i+1][0]):
                        overlapping_bursts.append(i)
                    else:
                        overlapping_bursts.append(i+1)
            if len(overlapping_bursts)>0:
                # Remove the overlapping bursts (we do this outside of the loop to not disturb the loop)
                for j in sorted(overlapping_bursts, reverse=True):
                    del network_burst_cores[j]
                    #print(f"removed NB {j}")
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
        plt.show()
        path = f'{filename}/network_data'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(f'{path}/well_{well}_network_bursts.csv', total_network_burst, delimiter = ",")
        np.save(f'{path}/well_{well}_network_bursts', total_network_burst)
        np.savetxt(f'{path}/well_{well}_participating_bursts.csv', participating_bursts, delimiter = ",")
        np.save(f'{path}/well_{well}_participating_bursts', participating_bursts)
        return total_network_burst

'''Calculate the features for all the electrodes'''
def electrode_features(filename, electrodes, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes):
    # Define all lists for the features
    wells_df, electrodes_df, spikes, mean_firingrate, mean_ISI, median_ISI, ratio_median_over_mean, IIV, CVI, PACF  = [], [], [], [], [], [], [], [], [], []
    burst_amnt, avg_burst_len, burst_var, burst_CV, mean_IBI, IBI_var, IBI_CV= [], [], [], [], [], [], []
    intraBFR, interBFR, mean_spikes_per_burst, isolated_spikes, MAD_spikes_per_burst = [], [], [], [], []
    SCB_rate = []
    
    # Convert the filename so the algorithm can look through the folders
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
    burstpath=f'{filename}/burst_values'
    networkpath=f'{filename}/network_data'
    
    # Loop through all the measured electrodes
    for electrode in electrodes:
        # Calculate current well number and append to list
        well_nr = round(electrode / electrode_amnt + 0.505)
        wells_df.append(well_nr)

        # Calculate current MEA electrode number and append to list
        electrode_nr = electrode % electrode_amnt + 1
        electrodes_df.append(electrode_nr)

        # Load in the spikedata
        spikedata=np.load(f'{spikepath}/well_{well_nr}_electrode_{electrode_nr}_spikes.npy')

        # Calculate the total amount of spikes
        spike=spikedata.shape[0]
        spikes.append(spike)

        # Calculate the average firing frequency frequency
        firing_frequency=spike/(measurements/hertz)
        mean_firingrate.append(firing_frequency)

        # Calculate mean inter spike interval
        spikeintervals = []
        # Iterate over every registered spike
        if spikedata.shape[0]<2:
            mean_ISI.append(float("NaN"))
            median_ISI.append(float("NaN"))
            ratio_median_over_mean.append(float("NaN"))
        else:
            for i in range(spikedata.shape[0]-1):
                time_to_next_spike = (spikedata[i+1][0]) - (spikedata[i][0])
                spikeintervals.append(time_to_next_spike)
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
        if len(spikeintervals)<3:
            PACF.append(float("NaN"))
        else:
            #sm.graphics.tsa.plot_pacf(spikeintervals, lags=10, method="ywm")
            PACF.append(statsmodels.tsa.stattools.pacf(spikeintervals, nlags=1, method='yw')[1])

        # Open the burstfiles
        burst_cores=np.load(f'{burstpath}/well_{well_nr}_electrode_{electrode_nr}_burst_cores.npy')
        burst_spikes=np.load(f'{burstpath}/well_{well_nr}_electrode_{electrode_nr}_burst_spikes.npy')

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
            SCB_rate.append(len(burst_cores)/(measurements/hertz))
        else:
            SCB_rate.append(float("NaN"))


        
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
        "Partial_Autocorrelaction_Function": PACF,
        "Total_number_of_bursts": burst_amnt,
        "Average_length_of_bursts": avg_burst_len,
        "Burst_length_variance": burst_var,
        "Coefficient_of_variation_burst_length": burst_CV,
        "Mean_interburst_interval": mean_IBI,
        "Variance_interburst_interval": IBI_var,
        "Coefficient_of_variation_IBI": IBI_CV,
        "Mean_intra_burst_firing_rate": intraBFR,
        "Mean_spikes_per_burst": mean_spikes_per_burst,
        "MAD_spikes_per_burst": MAD_spikes_per_burst,
        "Isolated_spikes": isolated_spikes,
        "Single_channel_burst_rate": SCB_rate
    })
    if remove_inactive_electrodes:
        # Remove electrodes that do not have enough activity
        features_df=features_df[features_df["Mean_FiringRate"]>activity_threshold]
    # Calculate how many electrodes were active
    active_electrodes=len(features_df)/electrode_amnt
    # If none of the electrodes have enough activity, make sure we retain the well value
    if len(features_df)==0:
        features_df["Well"]=[well_nr]
    features_df.insert(1, "Active_electrodes", [active_electrodes]*len(features_df))
    return features_df

'''Calculate the features per well'''
def well_features(filename, well, electrode_amnt, measurements, hertz):
    # Convert the filename so the algorithm can look through the folders
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
    burstpath=f'{filename}/burst_values'
    networkpath=f'{filename}/network_data'

    # Load in the network burst data
    network_cores=np.load(f"{networkpath}/well_{well}_network_bursts.npy")
    
    network_bursts, network_burst_duration, network_burst_core_duration, network_IBI, NB_NBc_ratio = [], [], [], [], []
    nIBI_var, nIBI_CV, NB_NBC_ratio_left, NB_NBC_ratio_right, lr_NB_ratio = [], [], [], [], []

    # Calculate the total amount of network bursts
    network_bursts.append(len(network_cores))
    #print(network_bursts)

    # Fill the features with NaN values if there are no network bursts
    if len(network_cores)==0:
        #print("No network bursts")
        network_burst_duration.append(float("NaN"))
        network_burst_core_duration.append(float("NaN"))
        NB_NBc_ratio.append(float("NaN"))
        NB_NBC_ratio_left.append(float("NaN"))
        NB_NBC_ratio_right.append(float("NaN"))
        lr_NB_ratio.append(float("NaN"))
        network_IBI.append(float("NaN"))
        nIBI_var.append(float("NaN"))
        nIBI_CV.append(float("NaN"))
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
            left_ratios.append((network_cores[i,0]-network_cores[i,2])/(network_cores[i,1]-network_cores[i,0]))
            right_ratios.append((network_cores[i,3]-network_cores[i,1])/(network_cores[i,1]-network_cores[i,0]))
            lr_ratios.append((network_cores[i,0]-network_cores[i,2])/(network_cores[i,3]-network_cores[i,1]))
        NB_NBC_ratio_left.append(np.mean(left_ratios))
        NB_NBC_ratio_right.append(np.mean(right_ratios))
        lr_NB_ratio.append(np.mean(lr_ratios))

    # Create pandas dataframe with all features as columns 
    well_features_df = pd.DataFrame({
        "Well":[well],
        "Network_bursts": network_bursts,
        "Network_burst_duration": network_burst_duration,
        "Network_burst_core_duration": network_burst_core_duration,
        "Network_interburst_interval": network_IBI,
        "NB to NBc ratio": NB_NBc_ratio,
        "Network_IBI_variance": nIBI_var,
        "Network_IBI_coefficient_of_variation": nIBI_CV, 
        "Ratio_left_outer_burst_over_core": NB_NBC_ratio_left,
        "Ratio_right_outer_burst_over_core": NB_NBC_ratio_right,
        "Ratio_left_outer_right_outer": lr_NB_ratio
    })
    return well_features_df

'''This function will calculate a 3d guassian kernel'''
def K(x, H):
    # unpack two dimensions
    x1, x2 = x
    # extract four components from the matrix inv(H)
    a, b, c, d = np.linalg.inv(H).flatten()
    
    scale = 2*np.pi*np.sqrt( np.linalg.det(H))
    return np.exp(-(a*x1**2 + d*x2**2 + (b+c)*x1*x2)/2) / scale

'''This function will insert a 3d gaussian kernel at the location of every datapoint/spike'''
def KDE(x, H, data):
    # unpack two dimensions
    x1, x2 = x
    # prepare the grid for output values
    output = np.zeros_like(x1)
    # process every sample
    for sample in data:
        output += K([x1-sample[0], x2-sample[1]], H)
    return output

'''Create a 3D view of a single well. Every spike will be represented as a 3D gaussian kernel'''
def fancyplot(filename, wells, electrode_amnt, measurements, hertz, resolution, kernel_size, aspectratios, colormap):
    # Convert the filename so the algorithm can look through the folders
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
    for well in wells:
        kdedata=[]
        for electrode in range(1,electrode_amnt+1):
            # Load in spikedata
            spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
            spikedata=spikedata[:,:2]
            spikedata[:,1]=electrode
            for spike in spikedata:
                kdedata.append(spike)
        time=measurements/hertz

        # covariance matrix
        # This determines the shape of the gaussian kernel
        H = [[1, 0],
            [0, 1]]

        data = np.array(kdedata)
        # fix arguments 'H' and 'data' of KDE function for further calls
        KDE_partial = partial(KDE, H=kernel_size*np.array(H), data=data)

        # draw contour and surface plots
        func=KDE_partial

        # create a np xy grid using the dimensions of the data
        yres=int(resolution*time)
        xres=int(resolution*electrode_amnt)
        y_range = np.linspace(start=0, stop=time, num=yres)
        x_range = np.linspace(start=0, stop=electrode_amnt, num=xres)
        print(f"Creating 3D plot, resolution x: {xres} by y: {yres}")

        y_grid, X_grid = np.meshgrid(y_range, x_range)
        Z_grid = func([y_grid, X_grid])
        
        fig = go.Figure(data=[go.Surface(y=y_grid, x=X_grid, z=Z_grid, colorscale=colormap)])
        fig.update_layout(
        scene = dict(
            xaxis = dict(range=[1,electrode_amnt]),
            yaxis = dict(range=[0,time]),
            ),
        margin=dict(r=20, l=10, b=10, t=10))

        # change the aspect ratio's of the plot
        fig.update_layout(scene_aspectmode='manual',
                            title=f"Well: {well}",
                            scene_aspectratio=dict(x=(electrode_amnt/10)*aspectratios[0],
                                           y=(time/10)*aspectratios[1],
                                           z=1*aspectratios[2]))
        fig.update_layout(scene = dict(
                    xaxis_title='Electrode',
                    yaxis_title='Time in seconds',
                    zaxis_title='KDE'),)
        fig.show()
        fig.write_html(f"{filename}/well_{well}_graph.html")

'''Plots the KDE of all electrodes (x) of a well in x amount of subplots'''
def well_electrodes_kde(filename, well, electrode_amnt, measurements, hertz):
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
    data_time=measurements/hertz
    fig, ax = plt.subplots(electrode_amnt, 1, sharex=True, sharey=True, figsize=(16,electrode_amnt))
    for electrode in range(electrode_amnt):
        spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode+1}_spikes.npy')
        if len(spikedata[:,0])>0:
            y = FFTKDE(bw=1, kernel='gaussian').fit(spikedata[:,0]).evaluate(grid_points=np.arange(0, data_time, 0.001))
            y=np.array(y)*len(spikedata[:,0])
            x=np.arange(0, data_time, 0.001)
        else:
            x=np.arange(0, data_time, 0.001)
            y=np.zeros(len(x))
        ax[electrode].plot(x,y)
        ax[electrode].set_ylabel(f"Electrode\n{electrode+1}")
    ax[-1].set_xlabel("Time in seconds")
    ax[0].set_xlim([0, measurements/hertz])
    ax[0].set_title(f"Well: {well} activity")
    plt.show()


'''This function will take the electrode and well features, clean them up an give back the output'''
def feature_output(electrode_features, well_features, electrode_amnt):
    active_electrodes=len(electrode_features)

    # Take the average of all the single-electrode features
    avg_electrode_features=pd.DataFrame(electrode_features.mean()).transpose()

    # Remove unnecessary columns
    avg_electrode_features=avg_electrode_features.drop(columns=["Electrode"])
    well_features=well_features.drop(columns=["Well"])

    # Combine the dataframes
    avg_electrode_features = pd.concat([avg_electrode_features, well_features], axis=1, join='inner')
    return avg_electrode_features


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
                      minspikes_burst=5,                    # The minimal amount of spikes a burst should have
                      maxISI_outliers=1000,                 # The maximal ISIth2 that can be used in burst detection
                      default_threshold=100,                # The default value for ISIth1
                      heightexception=1.5,                  # Multiplied with the spike detection threshold, if a spike exceeds this value, it does not have to drop amplitude fast to be validated
                      max_boxheight=2,                      # Multiplied with the spike detection threshold, will be the maximal value the box for DMP_noisebased validation can be
                      amplitude_drop_sd=5,                  # Multiplied with the SD of surrounding noise, will be the boxheight for DMP_noisebased validation
                      stdevmultiplier=5,                    # The amount of SD a value needs to be from the mean to be considered a possible spike in the noise detection
                      RMSmultiplier=5,                      # Multiplied with the RMS of the spike-free noise, used to determine the threshold
                      raster_plot=True,                     # Whether a raster plot should be shown
                      activity_threshold=0.1,               # The lowest frequency an electrode can have before being removed from the analysis
                      threshold_portion=1
                      ):                 
    # Open the data file
    print(f"Analyzing electrodes: {electrodes}")

    # Open  the datafile
    data=openHDF5_SCA1(filename)

    # Loop through all the electrodes
    for electrode in electrodes:
        print(f'Analyzing {filename} - Electrode: {electrode}')

        # Filter the data
        print("Applying bandpass filter")
        data[electrode]=butter_bandpass_filter(data[electrode], low_cutoff, high_cutoff, hertz, order)
        
        # Calculate the threshold
        print(f'Calculating the threshold')
        threshold_value=threshold(data[electrode], hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
        print(f'Threshold for electrode {electrode} set at: {threshold_value}')
        
        # Calculate spike values
        print('Validating the spikes')
        if validation_method=="DMP":
            spike_validation_DMP(data[electrode], electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_threshold, plot_validation, electrode_amnt, filename)
        elif validation_method=="DMP_noisebased":
            spike_validation_DMP_noisebased(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_validation, electrode_amnt, heightexception, max_boxheight, filename)
        else:
            raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
        
        # Detect the bursts
        print("Detecting bursts")
        burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, filename)
    
    # Create raster plots
    if raster_plot:
        print("Creating raster plots")
        raster(electrodes, electrode_amnt, data.shape[1], hertz, filename)

    # Calculate the features
    print("Calculating features")
    features_df=electrode_features(filename, electrodes, electrode_amnt, data.shape[1], hertz, activity_threshold)
    return features_df, data.shape[1]

'''Analyse an entire well'''
def analyse_well(filename,                                  # Where is the data file stored
                      wells,                                 # Which wells do you want to analyze
                      hertz,                                # What is the sampling frequency of the MEA
                      validation_method="DMP_noisebased",   # Which validation method do you want to use, possible: "DMP", "DMP_noisebased"
                      low_cutoff=200,                       # The low_cutoff for the bandpass filter
                      high_cutoff=3500,                     # The high_cutoff for the bandpass filter
                      order=2,                              # The order for the bandpass filter
                      spikeduration=0.002,                  # The amount of time only 1 spike should be registered, aka refractory period
                      exit_time_s=0.00024,                  # The amount of time a spike gets to drop amplitude in the validation methods
                      amplitude_drop_threshold=2,           # The amount of amplitude the spikes has to drop, this value will be used in the DMP method and multiplied with the threshold
                      plot_electrodes=True,                 # Plot the single electrode visualizations
                      well_amnt=24,                         # The amount of wells present in the MEA
                      electrode_amnt=12,                    # The amount of electrodes per well
                      kde_bandwidth=1,                      # The bandwidth of the kernel density estimate
                      smallerneighbours=10,                 # The amount of smaller neighbours a peak should have before being considered as one
                      minspikes_burst=5,                    # The minimal amount of spikes a burst should have
                      maxISI_outliers=1000,                 # The maximal ISIth2 that can be used in burst detection
                      default_threshold=100,                # The default value for ISIth1
                      heightexception=1.5,                  # Multiplied with the spike detection threshold, if a spike exceeds this value, it does not have to drop amplitude fast to be validated
                      max_boxheight=2,                      # Multiplied with the spike detection threshold, will be the maximal value the box for DMP_noisebased validation can be
                      amplitude_drop_sd=5,                  # Multiplied with the SD of surrounding noise, will be the boxheight for DMP_noisebased validation
                      stdevmultiplier=5,                    # The amount of SD a value needs to be from the mean to be considered a possible spike in the threshold detection
                      RMSmultiplier=5,                      # Multiplied with the RMS of the spike-free noise, used to determine the threshold
                      min_channels=0.5,                     # Minimal % of channels that should participate in a burst
                      threshold_method='yen',               # Threshold method to decide whether activity is a network burst or not - possible: 'yen', 'otsu'
                      
                      # Parameters for the 3D plot
                      resolution=5,                         # Creates a resolution*electrode_amnt by resolution*time_seconds grid for the 3D plot. E.g. a measurement time of 150s with 12 electrodes and resolution 10 would give a 1500*120 grid. Higher resolution means longer computing times
                      kernel_size=1,                        # The size of the 3D gaussian kernel
                      aspectratios=[0.5,0.25,0.5],          # The aspect ratios of the plot. multiplied with the length of the xyz axis'.
                      colormap="deep",                      # Colormap of the 3D plot

                      spikes_are_analysed=False,            # If this values is true, the algorithm will search for the spikefiles instead of recalculating these values
                      activity_threshold=0.1,               # The lowest frequency an electrode can have before being removed from the analysis
                      threshold_portion=1,                  # How much of the electrode do you want to use to calculate the threshold. Higher number = higher runtime
                      remove_inactive_electrodes=True,      # Whether you want to remove inactive electrodes

                      # Parameters for cutting up the data
                      cut_data_bool=False,
                      parts=10,
                      recordingtime=120
                      ):
    print("MEA-analyzer. Developed by Joram van Beem and Jesse Antonissen for the CureQ research consortium")
    if not spikes_are_analysed:
        print("Opening the data")
        data=openHDF5_SCA1(filename)
    if cut_data_bool:
        data=cut_data(data, parts, electrode_amnt)
        # If you decide to cut the data, all wells will be analyzed
        #wells=int(data.shape[0]/electrode_amnt)
        new_wells=np.array([])
        for well in wells:
            temp=(np.arange((well-1)*parts+1, (well)*(parts)+1))
            #print(temp)
            new_wells=np.append(new_wells, temp)
        wells=new_wells.astype(int)

    first_iteration=True
    for well in wells:
        start=time.time()
        electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
        print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

        # We don't have to analyse the spikes again if the files are already there
        if not spikes_are_analysed:
            # Loop through all the electrodes
            for electrode in electrodes:
                print(f'Analyzing {filename} - Electrode: {electrode}')

                # Filter the data
                print("Applying bandpass filter")
                data[electrode]=butter_bandpass_filter(data[electrode], low_cutoff, high_cutoff, hertz, order)
                
                # Calculate the threshold
                print(f'Calculating the threshold')
                threshold_value=fast_threshold(data[electrode], hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
                print(f'Threshold for electrode {electrode+1} set at: {threshold_value}')
                
                # Calculate spike values
                print('Validating the spikes')
                if validation_method=="DMP":
                    spike_validation_DMP(data[electrode], electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_threshold, plot_electrodes, electrode_amnt, filename)
                elif validation_method=="DMP_noisebased":
                    spike_validation_DMP_noisebased(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, heightexception, max_boxheight, filename)
                elif validation_method=='none':
                    spike_validation_DMP_noisebased(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, heightexception, 0.1, filename)
                else:
                    raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
                # Detect the bursts
                print("Detecting bursts")
                burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, filename, plot_electrodes)
        if spikes_are_analysed:
            measurements=recordingtime*hertz
        else:
            measurements=data.shape[1]

        # Calculate the network bursts
        print(f"Calculating network bursts of well: {well}")
        network_burst_detection(filename, [well], electrode_amnt, measurements, hertz, min_channels, threshold_method)
        # Create plots

        well_electrodes_kde(filename, well, electrode_amnt, measurements, hertz)
        #fancyplot(filename, [well], electrode_amnt, measurements, hertz, resolution, kernel_size, aspectratios, colormap)
        # Calculate the features
        print("Calculating features")
        features_df=electrode_features(filename, electrodes, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
        well_features_df=well_features(filename, well, electrode_amnt, measurements, hertz)
        print(features_df)
        print(well_features_df)
        if first_iteration:
            first_iteration=False
            output=feature_output(features_df, well_features_df, electrode_amnt)
        else:
            output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0)
        end=time.time()
        print(f"It took {end-start} seconds to analyse well: {well}")
    outputpath=filename[:-3]
    outputpath=outputpath+"_values"
    name = os.path.basename(outputpath)
    name = os.path.splitext(name)
    output.to_csv(f"{outputpath}/{name[0]}_Features.csv")
    return output