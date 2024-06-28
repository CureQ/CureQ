from CureQ.bandpass import *
from CureQ.burst_detection import *
from CureQ.features import *
from CureQ.network_burst_detection import *
from CureQ.open_file import *
from CureQ.plotting import *
from CureQ.spike_validation import *
from CureQ.threshold import *
import time
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
import gc
import json
from datetime import date
from datetime import datetime

'''Analyse electrode as subprocess'''
def _electrode_subprocess(outputpath, memory_id, shape, _type, electrode, hertz, low_cutoff, high_cutoff, order, stdevmultiplier,
                        RMSmultiplier, threshold_portion, spikeduration, exit_time_s,
                        amplitude_drop_sd, plot_electrodes, electrode_amnt,
                        heightexception, max_boxheight, kde_bandwidth, smallerneighbours,
                        minspikes_burst, maxISI_outliers, default_threshold,
                        validation_method, progressfile):
    # Attach to the existing shared memory block
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=memory_id)
    funcdata=np.ndarray(shape, _type, buffer=existing_shm.buf)
    data=funcdata[electrode]

    # Filter the data
    data=butter_bandpass_filter(data, low_cutoff, high_cutoff, hertz, order)
    
    # Calculate the threshold
    threshold_value=fast_threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
    
    # Calculate spike values
    if validation_method=="DMP_noisebased":
        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, heightexception, max_boxheight, outputpath)
    elif validation_method=='none':
        spike_validation(data, electrode, threshold_value, hertz, spikeduration, exit_time_s, 0, plot_electrodes, electrode_amnt, heightexception, max_boxheight, outputpath)
    else:
        raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
    
    # Detect the bursts
    burst_detection(data, electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, outputpath, plot_electrodes)
    print(f"Calculated electrode: {electrode}")

'''Analyse an entire well'''
def analyse_well(fileadress,                                # Where is the data file stored
                      hertz,                                # What is the sampling frequency of the MEA
                      electrode_amnt,                       # The amount of electrodes per well
                      wells='all',                          # Which wells do you want to analyze
                      validation_method="DMP_noisebased",   # Which validation method do you want to use, possible: "DMP", "DMP_noisebased"
                      low_cutoff=200,                       # The low_cutoff for the bandpass filter
                      high_cutoff=3500,                     # The high_cutoff for the bandpass filter
                      order=2,                              # The order for the bandpass filter
                      spikeduration=0.002,                  # The amount of time only 1 spike should be registered, aka refractory period
                      exit_time_s=0.00024,                  # The amount of time a spike gets to drop amplitude in the validation methods
                      plot_electrodes=True,                 # Plot the single electrode visualizations
                      well_amnt=24,                         # The amount of wells present in the MEA
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
                    #   resolution=5,                         # Creates a resolution*electrode_amnt by resolution*time_seconds grid for the 3D plot. E.g. a measurement time of 150s with 12 electrodes and resolution 10 would give a 1500*120 grid. Higher resolution means longer computing times
                    #   kernel_size=1,                        # The size of the 3D gaussian kernel
                    #   aspectratios=[0.5,0.25,0.5],          # The aspect ratios of the plot. multiplied with the length of the xyz axis'.
                    #   colormap="deep",                      # Colormap of the 3D plot

                      spikes_are_analysed=False,            # If this values is true, the algorithm will search for the spikefiles instead of recalculating these values
                      activity_threshold=0.1,               # The lowest frequency an electrode can have before being removed from the analysis
                      threshold_portion=0.1,                  # How much of the electrode do you want to use to calculate the threshold. Higher number = higher runtime
                      remove_inactive_electrodes=True,      # Whether you want to remove inactive electrodes

                      # Parameters for cutting up the data
                      cut_data_bool=False,
                      parts=10,
                      recordingtime=120,

                      use_multiprocessing=False
                      ):

    # Advertisement
    print("MEA-analyzer. Developed by Joram van Beem and Jesse Antonissen for the CureQ research consortium")
    
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

    if not spikes_are_analysed:
        print("Opening the data")
        data=openHDF5_SCA1(fileadress)
    if wells=='all':
        wells=list(range(1,int(data.shape[0]/electrode_amnt)+1))
    if cut_data_bool:
        print("Dividing the data")
        data=cut_data(data, parts, electrode_amnt)
        new_wells=np.array([])
        for well in wells:
            temp=(np.arange((well-1)*parts+1, (well)*(parts)+1))
            new_wells=np.append(new_wells, temp)
        wells=new_wells.astype(int)
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
        'height exception' : heightexception,
        'max drop amount' : max_boxheight,
        'minimal amount of spikes' : minspikes_burst,
        'default interval threshold' : default_threshold,
        'max interval threshold' : maxISI_outliers,
        'KDE bandwidth' : kde_bandwidth,
        'smaller neighbours' : smallerneighbours,
        'min channels' : min_channels,
        'thresholding method' : threshold_method,
        'remove inactive electrodes' : remove_inactive_electrodes,
        'activity threshold' : activity_threshold,
        'split data' : cut_data_bool,
        'parts' : parts,
        'use multiprocessing' : use_multiprocessing,
        'measurements' : data.shape[1]
    }
    with open(f"{outputpath}/parameters.json", 'w') as outfile:
        json.dump(parameters, outfile)

    # With multiprocessing
    if use_multiprocessing:
        # Save the data in shared memory
        sharedmemory=multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
        shape=data.shape
        np.save(progressfile, [(0)*electrode_amnt, shape[0]])
        _type=data.dtype
        data_shared=np.ndarray(shape, dtype=_type, buffer=sharedmemory.buf)
        data_shared[:]=data[:]
        memory_id=sharedmemory.name
        measurements=data.shape[1]
        data=[] # hopefully this clears up memory space

        # Start up a process for every single electrode
        with multiprocessing.Pool(processes=electrode_amnt) as pool:
            for well in wells:
                start=time.time()
                electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
                print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

                # Divide the tasks to the processes
                args=[(outputpath, memory_id, shape, _type, electrode, hertz, low_cutoff, high_cutoff, order, stdevmultiplier, RMSmultiplier, threshold_portion, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, heightexception, max_boxheight, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, validation_method, progressfile) for electrode in electrodes]
                pool.starmap(_electrode_subprocess, args)

                # Calculate the network bursts
                network_burst_detection(outputpath, [well], electrode_amnt, measurements, hertz, min_channels, threshold_method, plot_electrodes)

                features_df=electrode_features(outputpath, electrodes, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
                well_features_df=well_features(outputpath, well, electrode_amnt, measurements, hertz)
                if first_iteration:
                    first_iteration=False
                    output=feature_output(features_df, well_features_df, electrode_amnt)
                else:
                    output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0)
                end=time.time()
                print(f"It took {end-start} seconds to analyse well: {well}")
                np.save(progressfile, [(well)*electrode_amnt, shape[0]])
        # Clean up the shared memory
        sharedmemory.close()
        sharedmemory.unlink()
    # Without multiprocessing
    else:
        for well in wells:
            start=time.time()
            electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
            print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

            # We don't have to analyse the spikes again if the files are already there
            if not spikes_are_analysed:
                # Loop through all the electrodes
                for electrode in electrodes:
                    # Filter the data
                    data[electrode]=butter_bandpass_filter(data[electrode], low_cutoff, high_cutoff, hertz, order)
                    
                    # Calculate the threshold
                    threshold_value=fast_threshold(data[electrode], hertz, stdevmultiplier, RMSmultiplier, threshold_portion)
                    
                    # Calculate spike values
                    if validation_method=="DMP_noisebased":
                        spike_validation(data[electrode], electrode, threshold_value, hertz, spikeduration, exit_time_s, amplitude_drop_sd, plot_electrodes, electrode_amnt, heightexception, max_boxheight, outputpath)
                    elif validation_method=='none':
                        spike_validation(data[electrode], electrode, threshold_value, hertz, spikeduration, exit_time_s, 0, plot_electrodes, electrode_amnt, heightexception, max_boxheight, outputpath)
                    else:
                        raise ValueError(f"\"{validation_method}\" is not a valid spike validation method")
                    
                    # Detect the bursts
                    burst_detection(data[electrode], electrode, electrode_amnt, hertz, kde_bandwidth, smallerneighbours, minspikes_burst, maxISI_outliers, default_threshold, outputpath, plot_electrodes)
                    np.save(progressfile, [electrode+1, data.shape[0]])
                    print(f"Calculated electrode: {electrode}")
            if spikes_are_analysed:
                measurements=recordingtime*hertz
            else:
                measurements=data.shape[1]

            network_burst_detection(outputpath, [well], electrode_amnt, measurements, hertz, min_channels, threshold_method, plot_electrodes)
            print(f"Calculated network bursts well: {well}")

            features_df=electrode_features(outputpath, electrodes, electrode_amnt, measurements, hertz, activity_threshold, remove_inactive_electrodes)
            well_features_df=well_features(outputpath, well, electrode_amnt, measurements, hertz)
            if first_iteration:
                first_iteration=False
                output=feature_output(features_df, well_features_df, electrode_amnt)
            else:
                output=pd.concat([output, feature_output(features_df, well_features_df, electrode_amnt)], axis=0, ignore_index=False)
            end=time.time()
            print(f"It took {end-start} seconds to analyse well: {well}")
        
        # Attempt to free up RAM
        data=[]
        del data
        gc.collect()
    name = os.path.basename(outputpath)
    name = os.path.splitext(name)
    output.to_csv(f"{outputpath}/Features.csv", index=False)
    np.save(progressfile, ['done'])
    return output