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

#TODO Kijk of je functies van andere bestanden kan gebruiken voor efficiëntie


#%% Data prepping functions

def load_impedance(fileadress):
    """
    Loads the impedance data from the h5 file
    
    Parameters
    ---------
    file adress: The path to the hdf5 file 
    (Note: Impedance and mapping should be included through matlab script)
    """

    # Set column names for the 3 different impedances
    cols = ["1000", "10000", "41500"]

    # Open the hdf5 file
    with h5py.File(fileadress, 'r') as h5file:
        
        # Save the impedance data as strings in a dataframe
        impedance = pd.DataFrame(np.array(h5file["Data/impedance"]).astype(str), columns = cols)
        
        # Set indexing to 1-based instead of 0-based to match mapping
        impedance.index = impedance.index + 1

    return impedance


def load_mapping(fileadress):
    """
    Loads the mapping data from the h5 file. Mapping can be used
    to find the exact location of an electrode based on its index
    
    Parameters
    ---------
    file adress: The path to the hdf5 file 
    (Note: Impedance and mapping should be included through matlab script)
    """

    # Set the column names
    cols =  ["Index", "WellRow", "WellColumn", "ElectrodeColumn", "ElectrodeRow"]

    # Open the hdf5 file
    with h5py.File(fileadress, 'r') as h5file:

        # Save mapping data as integers in a dataframe 
        mapping = pd.DataFrame(np.array(h5file["Data/channel_map"]).astype(int), columns = cols)

        # Set indexing to 1-based #TODO CHECK HOE DIT IN ELKAAR STEEKT
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


def normalise_data(impedance, background = - 1, baseline = -1):
    """
    Normalise the impedance to a value between 0 and 1 
    based on the background noise and baseline

    Parameters
    --------
    impedance: The impedance data as a list
    background: the background impedance. If not given, its the minimal impedance
    baseline: The expected impedance of fully covered electrode. If not given: maximum
    """

    # Set background to minimum and baseline to maximum if these are not given
    if background == -1: background = min(impedance)
    if baseline == -1: baseline = max(impedance)

    # Create an empty list for the outputs
    out = []

    # Loop through each electrode in the impendance data
    for electrode in impedance:

        # Normalise the impedance for this electrode
        norm = (electrode - background) / (baseline - background)

        # Save the normalised impedance
        out.append(norm)

    return out

 #%% Heatmap functions

def calculate_well_dimensions(mapping): 
    """ 
    Calculate the dimensions of a well as tuple (Rows, Columns)

    Parameters
    -------
    mapping: The mapping dataframe
    """
    return (mapping["WellRow"].max(), mapping["WellColumn"].max())

def cmap_creation(): # TODO msch andere naam? want zelfde als andere hm
    """
    Create a colormap for the heatmap. Colors are based on axion heatmap
    """

    # Set the colors
    colours = [
        (0/5, "#000000"),
        (1/5, "#4235FC"),
        (2/5, "#15cbeb"),   
        (3/5, "#68C06A"),
        (4/5, "#F7A548"),
        (5/5, "#EFFF3B")
    ]

    # Create a colormap
    smooth_cmap = mcolors.LinearSegmentedColormap.from_list("smooth_heatmap", colours, N=256)
    return smooth_cmap


def reshape_wells(impedance, mapping):
    """
    Reshape the impedance data to a list of wells,
    each well being an array of electrodes. 
    This allows the data to be plotted in a heatmap correctly.

    Parameters:
    -------
    impedance: The normalised impedance data as a list
    mapping: The mapping dataframe
    """
    
    # Calculate the amount of wells 
    n_Wells = mapping['WellRow'].max() * mapping['WellColumn'].max()
    n_Electrodes =  mapping['ElectrodeRow'].max() * mapping['ElectrodeColumn'].max() # TODO

    # Set the size of a well including padding (Hardcoded for 4x4 electrodes)
    Size = 6

    # Create a list for the wells as output
    well_grids = []

    # Create an empty array for every well, where each value is 0
    for i in range(n_Wells):
        temp_well = np.zeros((Size,Size))
        well_grids.append(temp_well)
    
    # Loop trough every electrode in mapping
    for index, electrode in mapping.iterrows():

        # Colect the impedance
        imp = impedance[index - 1]

        # Collect the row and column of electrode and well
        wr = electrode['WellRow']
        wc = electrode['WellColumn']
        er = electrode['ElectrodeRow']
        ec = electrode['ElectrodeColumn']

        # Calculate the well number based on row and column, counting from upper left to botom right
        well_number = ((wr - 1) * mapping['WellColumn'].max()) + wc

        # Set the impedance to corresponding electrode in specific wellnumber
        well_grids[well_number - 1][er,ec] = imp

    # Flip each well upside down, because elctrode rows are counted from bottom
    for i in range(len(well_grids)):
        temp = np.flipud(well_grids[i])
        well_grids[i] = temp

    return well_grids

def electrode_mesh(mapping): # TODO toggle functie
    """
    Create a mesh for the electrode to plot dots to represent

    Parameters
    -------
    mapping: The mapping dataframe
    """

    # Calculate amount of rows and columns of electrodes
    rows = mapping["ElectrodeRow"].max()
    cols = mapping["ElectrodeColumn"].max()

    # Create empty lists for rows and columns
    rowindexes = []
    colindexes = []

    # Fill the to lists to represent every electrode from 1,1 to 4,4, including 1,4 and 4,1
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            rowindexes.append(i)
            colindexes.append(j)
    return (rowindexes, colindexes)


def create_viability_heatmap(well_data, well_dims): # Create heatmap with the correct data
    """
    Create a impedance heatmap with reshaped data

    Parameters
    -------
    well_data: reshaped and normalised impedance data
    well_dims: the dimensions of the wellplate
    """

    # Create a plot of subplots with the size of the wellplate
    fig, axs = plt.subplots(well_dims[0],well_dims[1], figsize=(8, 6))

    # Adjust the space between each well
    fig.subplots_adjust(hspace = 0.009, wspace = 0.009)

    # Set the subplots in a list
    axs = axs.ravel()

    # Create the mesh to represent electrodes
    e_mesh = electrode_mesh(mapping)

    # Loop through the wells
    for index, ax in enumerate(axs):

        # Remove the numbers of each plot and set sizes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 5)
        ax.set_ylim(5,0)
       
        # Represent every electrode with a red dot # TODO set in electrode_mesh functie (EN TESTEN)
        ax.scatter( 
            e_mesh[0], e_mesh[1], 
            color='red',  # Set color of electrodes to red
            s=5,          # Set the scale of the dots
            marker='o',   # Set the shape to circle
            edgecolors='black' # Set the edge to black
        )

        # Color the lines around each well red
        for spine in ax.spines.values():
            spine.set_color("red")
            spine.set_linewidth(0.5)
        
        # Create the heatmap for each well
        hm = ax.imshow(
            well_data[index], # Collect reshaped data for well
            cmap = cmap_creation(), # Collect colormap
            interpolation = 'gaussian', # Smoothen colors between electrodes
            vmax = 1, # Set max to 1 as normalised max
            vmin = 0, # Set min to 0 as normalised min
            origin = 'upper' # Locate heatmap correctly
        )
    
    # Set background of figure to black
    fig.patch.set_facecolor("black")

    # Get position of bottom right well for colorbar
    pos = axs[-1].get_position()

    # Create scaling colors
    norm = mcolors.Normalize(10000, 55000) # TODO moet hiervoor deze waarden?? is visueel voor cbar
    sm = cm.ScalarMappable(cmap = cmap_creation(), norm=norm)

    # Create colorbar and set text to white
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
    moduluses = calculate_modulus(impedance, '41500')
    normalised = normalise_data(moduluses, 10000, 55000) # Normalise with 10k background and 55k baseline

    well_dims = calculate_well_dimensions(mapping)
    well_data = reshape_wells(normalised, mapping)

    # Create heatmap
    fig = create_viability_heatmap(well_data, well_dims)
    fig.show()
    t = input("Press Enter to close...")

    test_data_set(mapping)
    test_data = load_test_data("testdata.csv")

    test_data = reshape_wells(test_data, mapping)
    #fig = create_viability_heatmap(test_data, well_dims)
    #fig.show()
    #t = input("press something to stop")
