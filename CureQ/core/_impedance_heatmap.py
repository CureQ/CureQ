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
import io
from PIL import Image

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

        # Set indexing to 1-based
        mapping.set_index("Index", inplace=True)

    return mapping

def calculate_real(df):
    """
    Calculate the real part of the impedance by splitting the text and turning it in to a float.

    Parameters
    ---------
    df: the impedane in the form of a dataframe with 3 cols of frequencies. Created by function load_impedance
    """
    reals = []
    for index, row in df.iterrows():
        imp = row['41500']
        if (type(imp) == str) and (imp != ""):
            real, imag = imp.split('-')
            real = float(real)
        else:
            real = 0
        reals.append(real)
    return reals


def normalise_data(impedance, background, baseline):
    """
    Normalise the impedance to a value between 0 and 1 
    based on the background noise and baseline

    Parameters
    --------
    impedance: The impedance data as a list, gotten from calculate real
    background: the background impedance. If -1 is given, its the minimal impedance
    baseline: The expected impedance of fully covered electrode. If -1 is given: maximum
    """

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

def cmap_creation_impedance():
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
    mapping: The channelmapping dataframe
    """
    
    # Calculate the amount of wells 
    n_Wells = mapping['WellRow'].max() * mapping['WellColumn'].max()
    n_Electrodes =  mapping['ElectrodeRow'].max() * mapping['ElectrodeColumn'].max()

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

def electrode_mesh(mapping):
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

def fig2img(fig):
    """
    convert a figure to an image without saving it

    Parameters
    ---------
    fig: The figure or plot to be converted to an image
    """
    # Save figure in newly created buffer
    buf = io.BytesIO()
    fig.savefig(buf)

    # Go back to start of buffer
    buf.seek(0)

    # Open the image from buffer
    img = Image.open(buf)

    return img

def create_viability_heatmap(well_data, mapping, background, baseline, show_electrodes = True):
    """
    Create a impedance heatmap with reshaped data

    Parameters
    -------
    well_data: reshaped and normalised impedance data
    mapping: the mapping information of the wellplate
    background: The lowest value shown in heatmap. If -1: min value selected
    baseline: The highest value shows in heatmap. If -1: max value selected
    show_electrodes: Toggle whether the electrodes are shown as red dots
    """
    # Calculate well dimensions
    well_dims = calculate_well_dimensions(mapping)

    # Create a plot of subplots with the size of the wellplate
    height = 6
    size = ((well_dims[1]/well_dims[0] * height), height)
    
    fig, axs = plt.subplots(well_dims[0],well_dims[1], figsize=size)

    # Adjust the space between each well
    fig.subplots_adjust(hspace = 0.009, wspace = 0.009, right=0.88)

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
       
        # Represent every electrode with a red dot als show electrodes true is
        if show_electrodes:
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
            cmap = cmap_creation_impedance(), # Collect colormap
            interpolation = 'gaussian', # Smoothen colors between electrodes
            vmax = 1, # Set max to 1 as normalised max
            vmin = 0, # Set min to 0 as normalised min
            origin = 'upper' # Locate heatmap correctly
        )
        ax.margins(0)
    
    # Set background of figure to transparent
    fig.patch.set_alpha(0.0)

    # Get position of bottom right well for colorbar
    pos = axs[-1].get_position()

    # Create scaling colors
    norm = mcolors.Normalize(background, baseline)
    sm = cm.ScalarMappable(cmap = cmap_creation_impedance(), norm=norm)

    ## Create colorbar and set text to white
    cbar = fig.colorbar(sm, cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height * well_dims[0]]))
    cbar.ax.tick_params(labelsize=8) 
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(axis='y', colors='white')

    pic = fig2img(fig)

    return pic

def viability_heatmap_handler(hdf5file, background = -1, baseline = -1, show_electrodes = True):
    """
    Main function to call in GUI. Handles every step by combining all of the functions above.

    Parameters:
    ---------
    hdf5file: The file with all the data (Note: make sure impedance is saved in here)
    background: The lowest value shown in heatmap. If -1: min value selected
    baseline: The highest value shows in heatmap. If -1: max value selected
    show_electrodes: Toggle whether the electrodes are shown as red dots
    """  

    # Load data
    impedance = load_impedance(hdf5file)
    mapping = load_mapping(hdf5file)

    # prepare data:
    imp_real = calculate_real(impedance)

    if background == -1:
        background = min(imp_real)
    if baseline == -1:
        baseline = max(imp_real)

    imp_norm = normalise_data(imp_real, background, baseline)
    imp_resh = reshape_wells(imp_norm, mapping)

    # Create heatmap
    fig = create_viability_heatmap(imp_resh, mapping = mapping, background = background, 
                                   baseline = baseline, show_electrodes = show_electrodes)
    return fig