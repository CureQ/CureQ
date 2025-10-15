import h5py
import re
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib import patheffects
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import seaborn as sns

def cmap_creation():
    """
    Creates a custom colour map, to be used later while generating the heatmap.

    Parameters
    ----------
    Returns
    -------
    smooth_cmap : Custom cmap
        This enables the cmap to be calles from anywhere in the code by referencing the dictionary.

    Notes
    -----
    The colourmap is based on the live axion biosystem heatmap.
    """
    colours = [
        (0.0, "black"),   
        (1/6, "blue"),   
        (2/6, "cyan"),
        (3/6, "green"),   
        (4/6, "yellow"), 
        (5/6, "red"),     
        (1.0, "white")
    ]

    # Create a smooth LinearSegmentedColormap
    smooth_cmap = mcolors.LinearSegmentedColormap.from_list("smooth_heatmap", colours, N=256)
    return smooth_cmap

def truesort(name):
    """
    Sorts through names and ensures it's in the right order (e.g. 2 before 11).

    Parameters
    ----------
    name : str
        Name of the electrode held in the hdf5 file.

    Returns
    -------
    (well_num, electrode_num) : int, int
        These are the numbers of the wells and the numbers of the electrode, sorted.   

    Notes
    -----
    The sorted numbers allow the program to call on the right wells and electrodes to sort them in a natural way (for humans, so 2 goes before 11).
    """
    match = re.search(r'well_(\d+)_electrode_(\d+)_spikes', name)
    if match:
        well_num = int(match.group(1))         # Extract well number
        electrode_num = int(match.group(2))    # Extract electrode number
        return (well_num, electrode_num)       # Sort by well first, then electrode
    return (float('inf'), float('inf'))        # Fallback in case of mismatch

def reshape_wells(data, n_Wells, n_Electrodes, Size):
    """
    Reshapes the individual wells for easier processing.

    Parameters
    ----------
    data : df
        The data that needs to be reshaped. This is 1 row of the data. 
    n_Wells : int
        The number of wells in a plate, typically 24, 48 or 96.
    n_Electrodes : int
        The number of electrodes in a well, typically 12 of 16.
    Size : int
        Size is to give the wells a slight padding to prevent the wells from being in the corners and allowing for a better looking gaussian smoothing.

    Returns
    -------
    well_grids : list
        This is a list of the wells, so every well gets an entry, which is a 6x6 shape.

    Notes
    -----
    Currently only allows for analysis of 16 electrodes. For any other number it needs to be edited.
    """
    data_array = data.to_numpy().flatten()
    well_grids = []
    for well in range(n_Wells):
        start = well * n_Electrodes
        end = start + n_Electrodes
        # Extract and reshape single well data
        well_data = data_array[start:end].reshape(4, 4)
        expanded_well = np.zeros((Size, Size))
        expanded_well[1:(Size-1), 1:(Size-1)] = well_data
        smoothed_well = gaussian_filter(expanded_well, sigma=0.1, mode='nearest')  
        well_grids.append(smoothed_well)
    return well_grids

def data_prepper(h5_data, parameters, Vars):
    """
    Prepares the data for the heatmaps.

    Parameters
    ----------
    h5_data : str
        Path to the hdf5 file.
    parameters : dict
        This is the dictionary containing the information about the measurement.
    Vars : dict
        This is a dictionary in which all relevant variables are stored.

    Returns
    -------
    precomputed_data : df
        Dataframe that contains one row per 0.1s of measurements, this entry is the number of spikes detected in a 1s window around the row.
    n_Wells : int
        The number of wells in the plate.
    n_Electrodes : int
        The number of electrodes in one well.
    v_max : list
        The higest amount of spikes detected in the 1s window of that measurement.
    Size : int
        The size of one side of the well that should be plotted (for 16 electrodes (4x4) it's 6, leading to a well of 6x6).
    sum_final_dataframe : list
        This is a list containing the total amount of spikes per electrode.
    max_df : list
        This list contains the highest number of detected spikes in a 1s window of each electrode.
    Notes
    -----
    This functions prepares all data for the three different heatmaps.
    """
    # Reads in the data and extracts the amount of spikes in a one second window for every 0.1s of measurements
    time_df = np.arange(0,int(parameters['measurements'])/int(parameters['sampling rate']),1/Vars["fps"])

    with h5py.File(h5_data, "r") as file:
        dataset = file["spike_values"]
        sorted_matches = sorted(dataset, key=truesort)
        
        # Temp dataframe to append values of every electrode to later
        df_list = []
        df_sum_list = []  
        for filename in sorted_matches:
            well_num, electrode_num = truesort(filename)
            col_name = f"W{well_num}_E{electrode_num}"

            # Read spike timestamps
            data = file["spike_values"][filename][:, 0]
            timestamps_array = np.round(data, 6)  # Round to match time_df precision
            n_spikes = len(data)
            # Temp dataframe to append the n spikes in a frame for one electrode
            frame_counts = []

            for frame_center in time_df:
                # Define the window: from (frame_center - 0.5s) to (frame_center + 0.5s)
                window_start = frame_center - 0.5
                window_end = frame_center + 0.5

                # Count the spikes within the 1s window
                count = np.sum((timestamps_array >= window_start) & (timestamps_array < window_end))

                # Append the 'count' for the current window
                frame_counts.append(count)

            # Store processed data for this electrode
            df_list.append(pd.Series(frame_counts, name=col_name))
            df_sum_list.append(pd.Series(n_spikes, name=col_name))

    # Makes df of the n spikes in a one second frame for every 0.1s of measurements
    final_dataframe = pd.concat(df_list, axis=1)
    sum_final_dataframe = pd.concat(df_sum_list, axis=1)
    v_max = []

    for frame_center in time_df:
        # Define the window: from (frame_center - 0.5s) to (frame_center + 0.5s)
        window_start = int(frame_center - 50)
        window_end = int(frame_center + 50)
        if window_start < 0:
            window_start = 0
        if window_end > len(final_dataframe):
            window_end = int(len(final_dataframe))
        
        # Finds the vmax in a 5s window around the 'frame center', so it decides the vmax based on the maximum of the frame_center
        vmax_index = final_dataframe.iloc[window_start:window_end].max().max()

        # Append the 'vmax_index' for the current window
        v_max.append(vmax_index)


    Final_match = re.findall(r'\d+', sorted_matches[len(sorted_matches)-1])
    n_Wells = int(Final_match[0])
    n_Electrodes = int(Final_match[1])
    Size = int(np.sqrt(n_Electrodes) + 2)

    max_values = final_dataframe.max(axis=0)
    max_df = pd.DataFrame(max_values).T

    # Precompute data to load in easier later
    precomputed_data = [reshape_wells(final_dataframe.iloc[frame], n_Wells, n_Electrodes, Size) for frame in range(Vars["Num frames"])]
    return precomputed_data, n_Wells, n_Electrodes, v_max, Size, sum_final_dataframe, max_df

def make_hm(Vars, Classes, Colour_classes):    
    """
    Prepares the figure and update function for the heatmap animation.
    It does NOT create the animation object itself.
    """    
    df = Vars["df"]
    vmin = 0
    vmax = (np.array(Vars["v_max"]).max()) * 0.5
    
    # Use Figure constructor directly
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)

    axs = axs.ravel()

    heatmaps = []
    i = 0
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        
        for spine in ax.spines.values():
            for class_name, indices in Classes.items():
                if (i+1) in indices:
                    color = Colour_classes.get(class_name, 'grey')
                    spine.set_color(mcolors.to_rgba(color, alpha=0.6))
                    spine.set_linewidth(0.5)
                    break  
        
        hm = ax.imshow(
            np.zeros((4,4)), 
            cmap=Vars["cmap"],
            interpolation='bicubic',
            vmax=vmax,
            vmin=vmin,
            extent=[-0.5, 3.5, 3.5, -0.5],
            origin='upper'
        )
        heatmaps.append(hm)

    pos = axs[-1].get_position()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=Vars["cmap"], norm=norm)
    cbar = fig.colorbar(sm, ax=axs.tolist(), cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, (pos.height * Vars["Rows"])]))
    cbar.set_label("Activity Level", fontsize=10, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(axis='y', colors='white')

    title = fig.suptitle("", color='white', fontsize=16, weight='bold', y=0.95)
    title.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])

    # This is the function the animation will call
    def update(frame):
        well_grids = df[frame]
        for hm, well_data in zip(heatmaps, well_grids):
            hm.set_array(well_data)

        time_seconds = frame / Vars["fps"]
        fig.suptitle(f"Spike rate (Spikes/Sec), Time: {time_seconds:.1f}s")
        # Return the artists that were changed
        return heatmaps + [title]

    # Return the figure and the update function, but NOT the animation object
    return fig, update


def make_hm_img(Vars, Classes, Colour_classes):  
    """
    Makes a heatmap image, showing the total number of spikes in an electrode.

    Parameters
    ----------
    Vars : dict
        This is a dictionary in which all relevant variables are stored.
    Classes : dict
        This is a dictionary in which all classes (names of cell cultures) are stored.
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure on which everything is drawn.
    Notes
    -----
    This is a heatmap showcasing only the total activity of each electrode. It is an image, not an animation.
    """    
    df = Vars["Precomputed Max"]
    n_Electrodes = Vars["n_Electrodes"]
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()

    fig.suptitle(f"Total spikes per electrode", fontsize=14, color="white")
    i = 0
    vmax = df.max().max()

    # Make well plots
    for i, ax in enumerate(axs):
        well_values = df.iloc[0, i * n_Electrodes : (i + 1) * n_Electrodes].to_numpy().reshape(4, 4)

        sns.heatmap(well_values, ax=ax, cmap='plasma', vmin=0, vmax=vmax, cbar=False, square=True, xticklabels=False, yticklabels=False, annot=True, fmt="d", annot_kws={"size": 6})
        
        ax.set_frame_on(True)

        for spine in ax.spines.values():
            for class_name, indices in Classes.items():
                if (i+1) in indices:
                    color = Colour_classes.get(class_name, 'grey')  
                    spine.set_visible(True)
                    spine.set_color(mcolors.to_rgba(color, alpha=0.8))
                    spine.set_linewidth(1.5)
                    break  

    fig.patch.set_facecolor(Vars["background_color"])
    return fig  

def make_hm_max_activity(Vars, Classes, Colour_classes):  
    """
    Makes a heatmap image, showing the maximum number of detected spikes in a 1 second timeframe of all electrodes.

    Parameters
    ----------
    Vars : dict
        This is a dictionary in which all relevant variables are stored.
    Classes : dict
        This is a dictionary in which all classes (names of cell cultures) are stored.
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure on which everything is drawn.
    Notes
    -----
    This is a heatmap showcasing only the maximum activity of each electrode in a one second window. It is an image, not an animation.
    """
    n_Electrodes = Vars["n_Electrodes"]
    max_values = Vars["max_df"].max(axis=0)
    df = pd.DataFrame(max_values).T
    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()

    fig.suptitle(f"Highest amount of spikes in 1 second per electrode", fontsize=14, color="white")
    i = 0
    vmax = df.max().max()

    # Make well plots
    for i, ax in enumerate(axs):
        well_values = df.iloc[0, i * n_Electrodes : (i + 1) * n_Electrodes].to_numpy().reshape(4, 4)

        sns.heatmap(well_values, ax=ax, cmap='plasma', vmin=0, vmax=vmax, cbar=False, square=True, xticklabels=False, yticklabels=False, annot=True, fmt="d", annot_kws={"size": 6})
        
        ax.set_frame_on(True)
        
        for spine in ax.spines.values():
            for class_name, indices in Classes.items():
                if (i+1) in indices:
                    color = Colour_classes.get(class_name, 'grey')  
                    spine.set_visible(True)
                    spine.set_color(mcolors.to_rgba(color, alpha=0.8))
                    spine.set_linewidth(1.5)
                    break  

    fig.patch.set_facecolor(Vars["background_color"])
    return fig  

def create_placeholder_figure(Vars):
    """
    Makes a empty heatmap image, showing only the layout of the plate.

    Parameters
    ----------
    Vars : dict
        This is a dictionary in which all relevant variables are stored.
    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure on which everything is drawn.
    Notes
    -----
    """    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()  

    for ax in axs:
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        
        for spine in ax.spines.values():            
            spine.set_color(mcolors.to_rgba('grey', alpha=0.8))
            spine.set_linewidth(1)

    pos = axs[-1].get_position()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=Vars["cmap"], norm=norm)
    cbar = fig.colorbar(sm, 
                        ax=axs.tolist(), 
                        cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, (pos.height * Vars["Rows"])]))
    cbar.set_label("Activity Level", fontsize=10, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(axis='y', colors='white')

    title = fig.suptitle("Spike rate (Spikes/Sec)", color='white', fontsize=16, weight='bold', y=0.95)
    title.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])
    return fig