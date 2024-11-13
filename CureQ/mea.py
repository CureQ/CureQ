# Imports
import time
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import gc
import json
from datetime import date
from datetime import datetime
import os
from importlib.metadata import version

# External libraries
import numpy as np
import pandas as pd
import h5py

# Import MEA functions
try:
    from ._bandpass import *
    from ._burst_detection import *
    from ._features import *
    from ._network_burst_detection import *
    from ._open_file import *
    from ._plotting import *
    from ._spike_validation import *
    from ._threshold import *
    from ._utilities import *
except:
    from _bandpass import *
    from _burst_detection import *
    from _features import *
    from _network_burst_detection import *
    from _open_file import *
    from _plotting import *
    from _spike_validation import *
    from _threshold import *
    from _utilities import *

'''Analyse electrode as subprocess
This is the subproces that gets called when multiprocessing is turned on'''
def _electrode_subprocess(outputpath, memory_id, shape, _type, electrode, hertz, low_cutoff, high_cutoff, order, stdevmultiplier,
                        RMSmultiplier, threshold_portion, spikeduration, exit_time_s,
                        amplitude_drop_sd, plot_electrodes, electrode_amnt,
                        max_drop_amount, kde_bandwidth, smallerneighbours,
                        minspikes_burst, max_threshold, default_threshold,
                        validation_method, progressfile):
    # Load in the data from the shared memory block in the RAM
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=memory_id, create=False)
    funcdata=np.ndarray(shape, _type, buffer=existing_shm.buf)

    # From all the data, select the electrode
    data=funcdata[electrode % electrode_amnt]

    # Filter the data
    data=butter_bandpass_filter(data, low_cutoff, high_cutoff, hertz, order)
    
    # Calculate the threshold
    threshold_value=fast_threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
    
    # Calculate spike values
    if validation_method=="DMP_noisebased":
        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, max_drop_amount, outputpath)
    elif validation_method=='none':
        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, 0, plot_electrodes, electrode_amnt, max_drop_amount, outputpath)
    else:
        raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
    
    # Detect the bursts
    burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, max_threshold, default_threshold, outputpath, plot_electrodes)

    data=None
    existing_shm.close()
    print(f"Calculated electrode: {electrode}")

'''Analyse an entire well'''
def analyse_wells(    fileadress,                                # Where is the data file stored
                      hertz,                                # What is the sampling frequency of the MEA
                      electrode_amnt,                       # The amount of electrodes per well
                      wells='all',                          # Which wells do you want to analyze
                      validation_method="DMP_noisebased",   # Which validation method do you want to use, possible: "none", "DMP_noisebased"
                      low_cutoff=200,                       # The low_cutoff for the bandpass filter
                      high_cutoff=3500,                     # The high_cutoff for the bandpass filter
                      order=2,                              # The order for the bandpass filter
                      spikeduration=0.001,                  # The amount of time only 1 spike should be registered, aka refractory period
                      exit_time_s=0.001,                    # The amount of time a spike gets to drop amplitude in the validation methods
                      bd_kde_bandwidth=1,                   # The bandwidth of the kernel density estimate
                      smallerneighbours=10,                 # The amount of smaller neighbours a peak should have before being considered as one
                      minspikes_burst=5,                    # The minimal amount of spikes a burst should have
                      max_threshold=1000,                   # The maximal ISIth2 that can be used in burst detection
                      default_threshold=100,                # The default value for ISIth1
                      max_drop_amount=2,                    # Multiplied with the spike detection threshold, will be the maximal value the box for DMP_noisebased validation can be
                      amplitude_drop_sd=5,                  # Multiplied with the SD of surrounding noise, will be the boxheight for DMP_noisebased validation
                      stdevmultiplier=5,                    # The amount of SD a value needs to be from the mean to be considered a possible spike in the threshold detection
                      RMSmultiplier=5,                      # Multiplied with the RMS of the spike-free noise, used to determine the threshold
                      min_channels=0.5,                     # Minimal % of channels that should participate in a burst
                      threshold_method='yen',               # Threshold method to decide whether activity is a network burst or not
                      nbd_kde_bandwidth=0.05,                # Bandwidth for network burst detection KDE
                      activity_threshold=0.1,               # The lowest frequency an electrode can have before being removed from the analysis
                      threshold_portion=0.1,                # How much of the electrode do you want to use to calculate the threshold. Higher number = higher runtime
                      remove_inactive_electrodes=True,      # Whether you want to remove inactive electrodes
                      use_multiprocessing=False             # Whether to use multiprocessing
                      ):
    analysis_time=time.time()
    plot_electrodes=False

    lib_version=version('CureQ')
    print(f"CureQ MEA Library - Version: {lib_version}")
    print(f"Analyzing: {fileadress}")
    
    # Create a directory which will contain the output
    outputpath=os.path.splitext(fileadress)[0]#remove the file extension
    outputpath=outputpath+"_output"

    # Add current datetime
    outputpath=outputpath+f"_{date.today()}"
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    outputpath=outputpath+f"_{current_time}"
    os.makedirs(outputpath)

    # Create different folders for the output
    os.makedirs(f"{outputpath}/burst_values")
    os.makedirs(f"{outputpath}/spike_values")
    os.makedirs(f"{outputpath}/network_data")
    os.makedirs(f"{outputpath}/figures")

    # Create a file to commmunicate the progress with the GUI
    progressfile=f'{os.path.split(fileadress)[0]}/progress.npy'
    np.save(progressfile, ['starting'])
    
    # Call the freeze_support function to make sure multiprocessing still works properly if the algorithm is frozen
    multiprocessing.freeze_support()

    # Open the raw data
    print("Opening the data")
    with h5py.File(fileadress, 'r') as h5file:
        dataset_chunks=h5file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].chunks
        # Check if dataset_chunks is not None
        if dataset_chunks:
            if dataset_chunks != 1:
                print("Data is not correctly chunked yet.\nRechunking the data will allow the tool to quickly analyze large files on limited amount of RAM")
                fileadress=rechunk_dataset(fileadress=fileadress, compression_method='lzf')
            else:
                print("Data is already correctly chunked")
        else:
            print("Data is not correctly chunked yet.\nRechunking the data will allow the tool to quickly analyze large files on limited amount of RAM")
            fileadress=rechunk_dataset(fileadress=fileadress, compression_method='lzf')
        datashape=h5file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].shape

    # Check if the electrode_amnt parameter is set properly
    if datashape[0]%electrode_amnt != 0:
        raise ValueError(f"The total amount of electrodes ({datashape[0]}) is not divisible by the number of electrodes per well ({electrode_amnt})")

    # If all wells should be analysed, generate a list of wells
    if wells=='all':
        wells=list(range(1,int(datashape[0]/electrode_amnt)+1))

    # Flag for if it is the first iteration
    first_iteration=True

    # Save the parameters that have been given in a JSON file 
    parameters={
        'file adress' : fileadress,
        'wells' : wells,
        'sampling rate' : hertz,
        'electrode amount' : electrode_amnt,
        'low cutoff' : low_cutoff,
        'high cutoff' : high_cutoff,
        'order' : order,
        'threshold portion' : threshold_portion,
        'standard deviation multiplier' : stdevmultiplier,
        'rms multiplier' : RMSmultiplier,
        'refractory period' : spikeduration,
        'spike validation method' : validation_method,
        'exit time' : exit_time_s,
        'drop amplitude' : amplitude_drop_sd,
        'max drop amount' : max_drop_amount,
        'minimal amount of spikes' : minspikes_burst,
        'default interval threshold' : default_threshold,
        'max interval threshold' : max_threshold,
        'KDE bandwidth' : bd_kde_bandwidth,
        'smaller neighbours' : smallerneighbours,
        'min channels' : min_channels,
        'thresholding method' : threshold_method,
        'nbd_kde_bandwidth' : nbd_kde_bandwidth,
        'remove inactive electrodes' : remove_inactive_electrodes,
        'activity threshold' : activity_threshold,
        'use multiprocessing' : use_multiprocessing,
        'measurements' : datashape[1],
        'library version' : lib_version
    }
    with open(f"{outputpath}/parameters.json", 'w') as outfile:
        json.dump(parameters, outfile)

    # With multiprocessing
    if use_multiprocessing:
        # Save the data in shared memory
        print("Loading data into shared memory")
        # Create space on the RAM
        with h5py.File(fileadress, 'r') as hdf_file:
            # Access the dataset
            dataset = hdf_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]
            np_size=dataset[:electrode_amnt].nbytes
            shape=dataset[:electrode_amnt].shape
            _type=dataset.dtype
            measurements=dataset.shape[1]
        sharedmemory=multiprocessing.shared_memory.SharedMemory(create=True, size=np_size)
        # Communicate file size with GUI
        np.save(progressfile, [(0)*electrode_amnt, datashape[0]])
        # Create a np array in the shared memory
        data_shared=np.ndarray(shape, dtype=_type, buffer=sharedmemory.buf)
        # Get the memory ID
        memory_id=sharedmemory.name
        # Clear up memory
        data=None

        # Start up a process for every single electrode
        print("Initializing processes")
        with multiprocessing.Pool(processes=electrode_amnt) as pool:
            # Iterate over all wells
            for well in wells:
                start=time.time()
                # Calculate which electrodes belong to this well
                electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
                print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

                readtime=time.time()
                # Read in the data of the well and put it into the shared memory block
                with h5py.File(fileadress, 'r') as hdf_file:
                    dataset = hdf_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]   
                    data_shared[:]=dataset[electrodes]
                print(f"readtime: {time.time()-readtime}")

                # Divide the tasks to the processes
                args=[(outputpath, memory_id, shape, _type, electrode, hertz, low_cutoff, high_cutoff, order, stdevmultiplier, RMSmultiplier, threshold_portion, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, max_drop_amount, bd_kde_bandwidth, smallerneighbours, minspikes_burst, max_threshold, default_threshold, validation_method, progressfile) for electrode in electrodes]
                pool.starmap(_electrode_subprocess, args)

                # Calculate the network bursts
                network_burst_detection(outputpath=outputpath, wells=[well], electrode_amnt=electrode_amnt, measurements=measurements, hertz=hertz, min_channels=min_channels, threshold_method=threshold_method, bandwidth=nbd_kde_bandwidth, plot_electrodes=plot_electrodes, save_figures=True)
                print(f"Calculated network bursts well: {well}")

                # Calculate electrode and well features
                features_df=electrode_features(outputpath, well, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
                well_features_df=well_features(outputpath, well, electrode_amnt, measurements, hertz)
                print(f"Calculated features well: {well}")

                # If its the first iteration, create the dataframe
                if first_iteration:
                    first_iteration=False
                    output=feature_output(features_df, well_features_df, electrode_amnt)
                # If its not the first iteration, keep appending to the dataframe
                else:
                    output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0)
                end=time.time()
                print(f"It took {end-start} seconds to analyse well: {well}")

                # Communicate progression with GUI
                np.save(progressfile, [(well)*electrode_amnt, datashape[0]])
        # Clean up the shared memory
        sharedmemory.close()
        sharedmemory.unlink()
    # Without multiprocessing
    else:
        with h5py.File(fileadress, 'r') as hdf_file:
            dataset = hdf_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]  
            for well in wells:
                start=time.time()
                # Calculate which electrodes belong to this well
                electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
                print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

                # Loop through all the electrodes
                for electrode in electrodes:
                    readtime=time.time() 
                    data=dataset[electrode]
                    print(f"readtime: {time.time()-readtime}")
                    # Filter the data
                    data=butter_bandpass_filter(data, low_cutoff, high_cutoff, hertz, order)
                    
                    # Calculate the threshold
                    threshold_value=fast_threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
                    
                    # Calculate spike values
                    if validation_method=="DMP_noisebased":
                        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, max_drop_amount, outputpath)
                    elif validation_method=='none':
                        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, 0, plot_electrodes, electrode_amnt, max_drop_amount, outputpath)
                    else:
                        raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
                    
                    # Detect the bursts
                    burst_detection(data, electrode, electrode_amnt, hertz, bd_kde_bandwidth, smallerneighbours, minspikes_burst, max_threshold, default_threshold, outputpath, plot_electrodes)
                    # Communicate with GUI
                    np.save(progressfile, [electrode+1, datashape[0]])
                    print(f"Calculated electrode: {electrode}")
                measurements=datashape[1]

                # Detect network bursts
                network_burst_detection(outputpath=outputpath, wells=[well], electrode_amnt=electrode_amnt, measurements=measurements, hertz=hertz, min_channels=min_channels, threshold_method=threshold_method, bandwidth=nbd_kde_bandwidth, plot_electrodes=plot_electrodes, save_figures=True)
                print(f"Calculated network bursts well: {well}")

                # Calculate electrode and well features
                features_df=electrode_features(outputpath, well, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
                well_features_df=well_features(outputpath, well, electrode_amnt, measurements, hertz)
                print(f"Calculated features well: {well}")

                # If its the first iteration, create the dataframe
                if first_iteration:
                    first_iteration=False
                    output=feature_output(features_df, well_features_df, electrode_amnt)
                # If its not the first iteration, keep appending to the dataframe
                else:
                    output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0, ignore_index=False)
                end=time.time()
                print(f"It took {end-start} seconds to analyse well: {well}")
        
        # Free up RAM
        data=None
        del data
        gc.collect()

    # Save the output
    output.to_csv(f"{outputpath}/Features.csv", index=False)
    
    # Close the analysis
    np.save(progressfile, ['done'])
    print(f"It took {time.time()-analysis_time} seconds to analyse {fileadress}")
    print("Done")
    return output