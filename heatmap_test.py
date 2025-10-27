#%% imports
import pandas as pd
import seaborn as sn
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import h5py


#%% Data prepping functions

def load_impedance(fileadress): # Load impedance data from h5file
    cols = ["1000", 
            "10000", 
            "41500"
            ]
    with h5py.File(fileadress, 'r') as h5file:
        impedance = pd.DataFrame(np.array(h5file["Data/impedance"]).astype(str), columns = cols)
        impedance.index = impedance.index + 1
    return impedance

def load_mapping(fileadress): # Load channelmapping from h5file
    cols =  ["Index", 
             "WellRow", 
             "WellColumn", 
             "ElectrodeColumn", 
             "ElectrodeRow"
             ]
    with h5py.File(fileadress, 'r') as h5file:
        mapping = pd.DataFrame(np.array(h5file["Data/channel_map"]).astype(int), columns = cols)
        mapping.set_index("Index", inplace=True)
    return mapping

def calculate_modulus(df): # Calculate modulus to transfer impedance from complex numbers to real numbers
    moduluses = []
    for index, row in df.iterrows():
        imp = row['41500'] # Collect complex impedance at 41500Hz
        if (type(imp) == str) and (imp != ""): # If there is a value: calculate the modulus
            real, imag = imp.split('-') # split into real and imaginary parts
            real = float(real) # Convert real to float
            imag = float(imag[:-1]) # Remove the 'i' and convert imaginary to float
            mod = (real**2 + imag**2)**0.5 # Calculate modulus of complex number
        else: # If no value: set value to 0
            mod = 0
        moduluses.append(mod)
    return moduluses

def normalise_data(data): # Normalise the impedance based on background and baseline
    background = 12000
    baseline = max(data)
    out = []
    for mod in data:
        norm = (mod - background) / (baseline - background)
        out.append(norm)
    return out

#%% Heatmap functions

def calculate_well_dimensions(mapping): # Calculate the dimensions of a well as tuple (Rows, Columns)
    return (mapping["WellRow"].max(), mapping["WellColumn"].max())

def cmap_creation(): # Create a colormap for the heatmap
    colours = [
        (0.0, "black"),   
        (1/4, "blue"),   
        (2/4, "cyan"),
        (3/4, "green"),   
        (1, "yellow")
    ]
    smooth_cmap = mcolors.LinearSegmentedColormap.from_list("smooth_heatmap", colours, N=256)
    return smooth_cmap


def reshape_wells(data, mapping): 
    n_Electrodes =  mapping['ElectrodeRow'].max() * mapping['ElectrodeColumn'].max()
    n_Wells = mapping['WellRow'].max() * mapping['WellColumn'].max()

    Size = 6  # Size of the expanded grid (including padding) Hardcoded for 4x4 electrodes

    # Creating empty grids for every well
    well_grids = []
    for i in range(n_Wells):
        temp_well = gaussian_filter(np.zeros((Size,Size)), sigma=0.1, mode='nearest')
        well_grids.append(temp_well)
    
    # fill grids with correct data
    for index, row in mapping.iterrows():
        impedance = data[index - 1]
        wr = row['WellRow']
        wc = row['WellColumn']
        er = row['ElectrodeRow']
        ec = row['ElectrodeColumn']
        well_number = ((wr - 1) * mapping['WellColumn'].max()) + wc
        well_grids[well_number - 1][er,ec] = impedance

    for i in range(len(well_grids)):
        temp = np.flipud(well_grids[i])
        well_grids[i] = temp

    return well_grids

def create_viability_heatmap(well_data, well_dims):
    fig, axs = plt.subplots(well_dims[0],well_dims[1], figsize=(8, 6))
    fig.subplots_adjust(hspace = 0.009, wspace = 0.009)
    axs = axs.ravel()

    for index, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 5)
        ax.set_ylim(5,0)

        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(0.5)
        
        hm = ax.imshow(
            well_data[index],
            cmap = cmap_creation(),
            interpolation = 'gaussian',
            vmax = 1,
            vmin = 0,
            origin = 'upper'
        )
    fig.patch.set_facecolor("black")
    return fig

def test_data_set(mapping): # TODO this is a testfunc to create a dataset for plotting
    values = []
    with open("testdata.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        for index, row in mapping.iterrows():
            if row["ElectrodeRow"] == 1 and row["ElectrodeColumn"] == 1:
                spamwriter.writerow([1])
            elif row["ElectrodeRow"] == 4 and row["ElectrodeColumn"] == 4:
                spamwriter.writerow([0.5])
            else:
                spamwriter.writerow([0])
        return values

def load_test_data(path):
    test_data = []
    with open("testdata.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            test_data.append(float(line[0]))
    return test_data

#%% Code
if __name__ == "__main__":
    # Load data
    fileadress = "D:/mea_data/2025_44_dagen_iv/Bow_div44.h5"
    impedance = load_impedance(fileadress)
    mapping = load_mapping(fileadress)

    # Prep data
    moduluses = calculate_modulus(impedance)
    normalised = normalise_data(moduluses)

    well_dims = calculate_well_dimensions(mapping)
    well_data = reshape_wells(normalised, mapping)

    test_data_set(mapping)
    test_data = load_test_data("testdata.csv")

    test_data = reshape_wells(test_data, mapping)
    #fig = create_viability_heatmap(test_data, well_dims)
    #fig.show()
    #t = input("press something to stop")

    # Create heatmap
    fig = create_viability_heatmap(well_data, well_dims)
    fig.show()
    t = input("Press Enter to close...")
