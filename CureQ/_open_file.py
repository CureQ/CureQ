import numpy as np
import h5py
import os

'''Open the HDF5 file'''
def openHDF5(adress):
    with h5py.File(adress, "r") as file_data:
        # Retrieve the data from the correct location in the hdf5 file
        dataset = file_data["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"]
        # Convert to numpy array
        data=dataset[:]
    return data

'''Cut the data into smaller parts'''
def cut_data(data, parts, electrodes_per_well):
    shape_before=data.shape
    total_electrodes=data.shape[0]
    total_wells=int(total_electrodes/electrodes_per_well)
    measurements=data.shape[1]
    partsize=int(measurements/parts)
    new_data=[]
    for well in range(total_wells):
        for part in range(parts):
            for electrode in range(electrodes_per_well):
                electrode_id=well*electrodes_per_well+electrode
                new_data.append(data[electrode_id, part*partsize:(part+1)*partsize])
    new_data=np.array(new_data)
    print(f"Data converted from shape: {shape_before} to shape: {new_data.shape}")
    return new_data

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