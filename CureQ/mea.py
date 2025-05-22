# Imports
import time
import multiprocessing
import gc
import json
from datetime import date
from datetime import datetime
import os
from importlib.metadata import version
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

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
    from ._plotting import *
    from ._spike_validation import *
    from ._threshold import *
    from ._utilities import *
    from ._heatmap import *
    from ._machine_learning import *
except:
    from _bandpass import *
    from _burst_detection import *
    from _features import *
    from _network_burst_detection import *
    from _plotting import *
    from _spike_validation import *
    from _threshold import *
    from _utilities import *
    from _heatmap import *
    from _machine_learning import *

def get_default_parameters():
    """
    Store and retrieve global parameters
    
    Returns
    -------
    parameters : dict
        Default parameters for the library.
    
    """
    parameters={
        'low cutoff' : 200,
        'high cutoff' : 3500,
        'order' : 2,
        'threshold portion' : 0.1,
        'standard deviation multiplier' : 5,
        'rms multiplier' : 5,
        'refractory period' : 0.001,
        'spike validation method' : "Noisebased",
        'exit time' : 0.001,
        'drop amplitude' : 5,
        'max drop' : 2,
        'minimal amount of spikes' : 5,
        'default interval threshold' : 100,
        'max interval threshold' : 1000,
        'burst detection kde bandwidth' : 1,
        'min channels' : 0.5,
        'thresholding method' : 'Yen',
        'nbd kde bandwidth' : 0.05,
        'remove inactive electrodes' : True,
        'activity threshold' : 0.1,
        'use multiprocessing' : False
    }

    return parameters

def _electrode_subprocess(memory_id, shape, _type, electrode, parameters):
    """
    Function that can be called to analyse a single electrode as a subprocess when using multiprocessing.

    Parameters
    ----------
    memory_id : str
        The ID of the shared memory block.
    shape : tuple
        Shape of the shared array.
    _type : type
        Type of the shared array.
    electrode : int
        The electrode to be analysed.
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed.
    
    """

    # Load in the data from the shared memory block in the RAM
    existing_shm = multiprocessing.shared_memory.SharedMemory(name=memory_id, create=False)
    funcdata=np.ndarray(shape, _type, buffer=existing_shm.buf)

    # From all the data, select the electrode
    data=funcdata[electrode % parameters['electrode amount']]

    # Filter the data
    data=butter_bandpass_filter(data, parameters)
    
    # Calculate the threshold
    threshold_value=fast_threshold(data, parameters)
    
    # Calculate spike values
    if parameters['spike validation method']=="Noisebased":
        pass
    elif parameters['spike validation method']=='none':
        parameters['drop amplitude']=0
    else:
        raise ValueError(f"\"{parameters['spike validation method']}\" is not a valid spike validation method")
    spike_validation(data, electrode, threshold_value, parameters)
    
    # Detect the bursts
    burst_detection(data, electrode, parameters)

    data=None
    existing_shm.close()
    print(f"Calculated electrode: {electrode}")

def analyse_wells(fileadress, sampling_rate, electrode_amnt, parameters={}):
    """
    Analyse an entire MEA experiment, main function of the library.

    Parameters
    ----------
    fileadress : str
        The file location of the file that is to be analyzed.
    sampling_rate : int
        The sampling rate of the MEA experiment.
    electrode_amnt : int
        The amount of electrodes per well of the MEA experiment
    parameters : dict, optional
        Parameters that can alter the analysis, if left empty, will use default parameters.

    Returns
    -------
    output : pd.dataframe
        Pandas dataframe containing features for the different wells.

    Notes
    -----
    This function calls all other functions to perform the full analysis on a MEA-dataset.
    Besides this, it also communicates the progress with the GUI.

    Always launch the function with an “if __name__ == ‘__main__’:” guard.

    Examples
    --------
    >>> from CureQ.mea import analyse_wells, get_default_parameters
    >>> 
    >>> if __name__ == '__main__':
    ...     parameters = get_default_parameters()
    ...     fileadress = 'H:/MEA_data/mea_experiment.h5'
    ...     sampling_rate = 20000
    ...     electrode_amount = 12
    ...     analyse_wells(
    ...         fileadress=fileadress,
    ...         sampling_rate=sampling_rate,
    ...         electrode_amnt=electrode_amount,
    ...         parameters=parameters
    ...     )

    """


    analysis_time=time.time()

    lib_version=version('CureQ')
    print(f"MEA Analysis Tool - Version: {lib_version}")
    print(f"Analyzing: {fileadress}")
    
    path_obj = Path(fileadress)
    parent_dir=path_obj.parent

    # Create a directory which will contain the output
    output_folder=path_obj.stem
    output_folder+="_output"
    output_folder+=f"_{date.today()}"
    output_folder+=f"_{datetime.now().strftime('%H_%M_%S')}"

    outputpath=os.path.join(parent_dir, output_folder)
    os.makedirs(outputpath)

    # Create a file to commmunicate the progress with the GUI
    progressfile=f'{os.path.split(fileadress)[0]}/progress.npy'
    np.save(progressfile, ['starting'])
    
    # Call the freeze_support function to make sure multiprocessing still works properly if the algorithm is frozen
    multiprocessing.freeze_support()

    # Open the raw data
    print("Opening the data")
    rechunk_data=False
    with h5py.File(fileadress, 'r') as h5file:
        dataset_chunks=h5file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].chunks

        datashape=h5file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].shape
        # Check if the electrode_amnt parameter is set properly
        if datashape[0]%electrode_amnt != 0:
            raise ValueError(f"The total amount of electrodes ({datashape[0]}) is not divisible by the number of electrodes per well ({electrode_amnt})")

        # Check if dataset_chunks is not None
        if dataset_chunks:
            if dataset_chunks[0] != 1:
                rechunk_data=True
            else:
                print("Data is already correctly chunked")
                rechunk_data=False
        else:
            rechunk_data=True
            
        if rechunk_data:
            print("Data is not correctly chunked yet.\nRechunking the data will allow the tool to quickly analyze large files on limited amount of RAM")
            np.save(progressfile, ['rechunking'])
            fileadress=rechunk_dataset(fileadress=fileadress, compression_method='lzf')

    # Create figures folder for output
    os.makedirs(f"{outputpath}/figures")
    
    # Create hdf5 file for output
    output_hdf_file=f"{outputpath}/output_values.h5"
    with h5py.File(output_hdf_file, 'w') as f:
        f.create_group("spike_values")
        f.create_group("burst_values")
        f.create_group("network_values")

    # Load default parameters if none were given
    if parameters=={}:
        parameters=get_default_parameters()
        
    # Save the parameters that have been given in a JSON file 
    new_values={'output path' : outputpath,
                'output hdf file' : output_hdf_file,
                'file adress' : fileadress,
                'sampling rate' : sampling_rate,
                'electrode amount' : electrode_amnt,
                'measurements' : datashape[1],
                'library version' : lib_version}
    parameters.update(new_values)

    with open(f"{outputpath}/parameters.json", 'w') as outfile:
        json.dump(parameters, outfile, indent=4)

    # Create a list of all wells
    wells=list(range(1,int(datashape[0]/electrode_amnt)+1))

    # Flag for the first iteration
    first_iteration=True

    # With multiprocessing
    if parameters['use multiprocessing']:
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
                print(f"Readtime: {time.time()-readtime}")

                # Divide the tasks to the processes
                args=[(memory_id, shape, _type, electrode, parameters) for electrode in electrodes]
                pool.starmap(_electrode_subprocess, args)

                # Calculate the network bursts
                network_burst_detection([well], parameters)
                print(f"Calculated network bursts well: {well}")

                # Calculate electrode and well features
                electrode_features_df=electrode_features(well, parameters)
                well_features_df=well_features(well, parameters)
                print(f"Calculated features well: {well}")

                # If its the first iteration, create the dataframe
                if first_iteration:
                    first_iteration=False
                    electrode_features_output=copy.deepcopy(electrode_features_df)
                    output=feature_output(electrode_features_df, well_features_df)
                # If its not the first iteration, keep appending to the dataframe
                else:
                    electrode_features_output=pd.concat([electrode_features_output, copy.deepcopy(electrode_features_df)], ignore_index=False)
                    output=pd.concat([output, feature_output(electrode_features_df, well_features_df)], axis=0, ignore_index=False)
                end=time.time()
                print(f"It took {end-start} seconds to analyse well: {well}")

                # Check if the user wants to exit the analysis
                progressdata=np.load(progressfile)
                if progressdata[0]=="abort":
                    # Clean up the shared memory
                    sharedmemory.close()
                    sharedmemory.unlink()
                    pool.terminate()
                    pool.join()
                    data=None
                    del data
                    gc.collect()
                    np.save(progressfile, ["stopped"])
                    print("stopped analysis")
                    return
                
                # Communicate progression with GUI
                np.save(progressfile, [(well)*electrode_amnt, datashape[0]])

        # Clean up the shared memory
        sharedmemory.close()
        sharedmemory.unlink()

    # Without multiprocessing
    else:  
        for well in wells:
            start=time.time()
            # Calculate which electrodes belong to this well
            electrodes=np.arange((well-1)*electrode_amnt, well*electrode_amnt)
            print(f"Analyzing well: {well}, consisting of electrodes: {electrodes}")

            # Loop through all the electrodes
            for electrode in electrodes:
                with h5py.File(fileadress, 'r') as hdf_file:
                    dataset = hdf_file["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]
                    # Read in the data
                    data=dataset[electrode]
                # Filter the data
                data=butter_bandpass_filter(data, parameters)
                
                # Calculate the threshold
                threshold_value=fast_threshold(data, parameters)
                
                # Calculate spike values
                if parameters['spike validation method']=="Noisebased":
                    pass
                elif parameters['spike validation method']=='none':
                    parameters['drop amplitude']=0
                else:
                    raise ValueError(f"\"{parameters['spike validation method']}\" is not a valid spike validation method")
                spike_validation(data, electrode, threshold_value, parameters)

                # Detect the bursts
                burst_detection(data, electrode, parameters)
                
                # Check if the user wants to exit the analysis
                progressdata=np.load(progressfile)
                if progressdata[0]=="abort":
                    data=None
                    del data
                    gc.collect()
                    np.save(progressfile, ["stopped"])
                    print("stopped analysis")
                    return

                # Communicate with GUI
                np.save(progressfile, [electrode+1, datashape[0]])
                print(f"Calculated electrode: {electrode}")
            measurements=datashape[1]

            # Detect network bursts
            network_burst_detection([well], parameters, save_figures=True)
            print(f"Calculated network bursts well: {well}")

            # Calculate electrode and well features
            electrode_features_df=electrode_features(well, parameters)
            well_features_df=well_features(well, parameters)
            print(f"Calculated features well: {well}")

            # If its the first iteration, create the dataframe
            if first_iteration:
                first_iteration=False
                electrode_features_output=copy.deepcopy(electrode_features_df)
                output=feature_output(electrode_features_df, well_features_df)
            # If its not the first iteration, keep appending to the dataframe
            else:
                electrode_features_output=pd.concat([electrode_features_output, copy.deepcopy(electrode_features_df)], ignore_index=False)
                output=pd.concat([output, feature_output(electrode_features_df, well_features_df)], axis=0, ignore_index=False)
            end=time.time()
            print(f"It took {end-start} seconds to analyse well: {well}")
        
        # Free up RAM
        data=None
        del data
        gc.collect()

    # Save the output
    output.to_csv(f"{outputpath}/{output_folder}_Features.csv", index=False)
    electrode_features_output.to_csv(f"{outputpath}/{output_folder}_Electrode_Features.csv", index=False)
    
    # Close the analysis
    np.save(progressfile, ['done'])
    print(f"It took {time.time()-analysis_time} seconds to analyse {fileadress}")
    print("Done")
    return output