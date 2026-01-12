#%% imports
import pandas as pd
import seaborn as sn
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import csv
import h5py
import math


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

def calculate_modulus(df, frequency):
    """
    Calculate the modulus to transfer impedance 
    from complex number to decimal

    Parameters
    --------
    df: Dataframe with impedance data
    frequency: Which of the impedance data is used in Hz ('1000', '10000' or '41500') # TODO
    """

    # Create empty list for output values
    moduluses = []

    # Loop trough electrodes to calculate modulus
    for index, row in df.iterrows():

        # Collect complex impedance at set frequency
        imp = row[frequency]

        # Set value to 0 and skip the calculation if no impedance is measured
        if (type(imp) != str) or (imp == ""):
            moduluses.append(0)
            continue

        # Split compelx number into real and imaginary parts
        real, imag = imp.split('-') 
        
        # Convert values to float and remove the i from imaginary part
        real = float(real)
        imag = float(imag[:-1])

        # Calculate modulus of complex number
        mod = (real**2 + imag**2)**0.5 

        # Save the modulus
        moduluses.append(mod)

    return moduluses

def calculate_real(df, frequency):
    reals = []
    for index, row in df.iterrows():
        imp = row[frequency]
        if (type(imp) == str) and (imp != ""):
            real, imag = imp.split('-')
            real = float(real)
        else:
            real = 0
        reals.append(real)
    return reals



def normalise_data(data, background = - 1, baseline = -1): # Normalise the impedance based on background and baseline
    if background == -1: background = min(data)
    if baseline == -1: baseline = max(data)
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
        (0/5, "#000000"),
        (1/5, "#4235FC"),
        (2/5, "#15cbeb"),   
        (3/5, "#68C06A"),
        (4/5, "#F7A548"),
        (5/5, "#EFFF3B")
    ]
    smooth_cmap = mcolors.LinearSegmentedColormap.from_list("smooth_heatmap", colours, N=256)
    return smooth_cmap


def reshape_wells(data, mapping): # Reshape data to list of wells with each well being array of electrodes
    n_Electrodes =  mapping['ElectrodeRow'].max() * mapping['ElectrodeColumn'].max()
    n_Wells = mapping['WellRow'].max() * mapping['WellColumn'].max()

    Size = 6  # Size of the expanded grid (including padding) Hardcoded for 4x4 electrodes

    # Creating empty array for every well
    well_grids = []
    for i in range(n_Wells):
        temp_well = np.zeros((Size,Size)) #gaussian_filter(np.zeros((Size,Size)), sigma=0.1, mode='nearest')
        well_grids.append(temp_well)
    
    # fill arrays with correct data
    for index, row in mapping.iterrows():
        impedance = data[index - 1]
        wr = row['WellRow']
        wc = row['WellColumn']
        er = row['ElectrodeRow']
        ec = row['ElectrodeColumn']
        well_number = ((wr - 1) * mapping['WellColumn'].max()) + wc
        well_grids[well_number - 1][er,ec] = impedance

    for i in range(len(well_grids)):
        temp = np.flipud(well_grids[i]) # Electrode rows are counted from bottom
        well_grids[i] = temp

    return well_grids

def electrode_mesh(mapping): # TODO toggle functie
    rows = mapping["ElectrodeRow"].max()
    cols = mapping["ElectrodeColumn"].max()
    rowindexes = []
    colindexes = []
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            rowindexes.append(i)
            colindexes.append(j)
    return (rowindexes, colindexes)

def create_viability_heatmap(well_data, well_dims): # Create heatmap with the correct data
    fig, axs = plt.subplots(well_dims[0],well_dims[1], figsize=(8, 6))
    fig.subplots_adjust(hspace = 0.009, wspace = 0.009)
    axs = axs.ravel()

    e_mesh = electrode_mesh(mapping)

    for index, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 5)
        ax.set_ylim(5,0)
       
        ax.scatter( # Scatter dots to represent electrodes
            e_mesh[0], e_mesh[1],
            color='red',
            s=5,          # Scale of dots
            marker='o',
            edgecolors='black'
        )
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

    pos = axs[-1].get_position()
    norm = mcolors.Normalize(10000, 55000)
    sm = cm.ScalarMappable(cmap = cmap_creation(), norm=norm)
    cbar = fig.colorbar(sm, ax=axs.tolist(), cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height * well_dims[0]]))
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(axis='y', colors='white')
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
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            test_data.append(float(line[0]))
    return test_data

def make_viability_heatmap_img(impedances, mapping):  
    n_Electrodes = max(mapping["ElectrodeRow"]) * max(mapping["ElectrodeColumn"])
    n_Wells = max(mapping["WellRow"]) * max(mapping["WellColumn"])
    fig, axs = plt.subplots(max(mapping["WellRow"]), max(mapping["WellColumn"]), figsize = (8, 6))
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()

    fig.suptitle(f"Total impedance per electrode", fontsize=14, color="white")

    i = 0
    for i, ax in enumerate(axs):
        well_values = impedances[i][1:5, 1:5] // 1000
        vmin = 10
        vmax = 55
        sn.heatmap(well_values, ax=ax, cmap='plasma', vmin = vmin, 
                   vmax = vmax, cbar=False, square = True, xticklabels=False, yticklabels=False, 
                   annot=True, fmt=".0f", annot_kws={"size": 6}, mask=np.isnan(well_values))

        ax.set_frame_on(True)

        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(10)

    fig.patch.set_facecolor("black")
    return fig
#%% Code
if __name__ == "__main__":
    # Load data
    fileadress = "D:/mea_data/2025_44_dagen_iv/Bow_div44_inclus.h5"
    impedance = load_impedance(fileadress)
    mapping = load_mapping(fileadress)

    # Prep data
    moduluses = calculate_modulus(impedance, '41500')
    normalised = normalise_data(moduluses, 10000, 55000) # Normalise with 10k background and 55k baseline

    well_dims = calculate_well_dimensions(mapping)
    well_data = reshape_wells(moduluses, mapping)

    real = calculate_real(impedance, '41500')
    real_normalised = normalise_data(real, 10000, 55000)
    wells_real = reshape_wells(real, mapping)
    wells_norm = reshape_wells(real_normalised, mapping)

    #Create heatmap
    fig = create_viability_heatmap(wells_norm, well_dims)
    fig.show()
    t = input("Press Enter to close...")



#%% heatmap img
    #fig = make_viability_heatmap_img(wells_real, mapping)
    #fig.show()
    #t = input("press enter to close...")



#%% Testdata
    #test_data_set(mapping)
    #test_data = load_test_data("testdata.csv")
#
    #test_data = reshape_wells(test_data, mapping)
    #fig = create_viability_heatmap(test_data, well_dims)
    #fig.show()
    #t = input("press something to stop")

