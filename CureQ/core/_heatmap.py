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
    colours = [
        (0.0, "black"),   
        (1/6, "blue"),   
        (2/6, "cyan"),
        (3/6, "green"),   
        (4/6, "yellow"), 
        (5/6, "red"),     
        (1.0, "white")
    ]

    smooth_cmap = mcolors.LinearSegmentedColormap.from_list("smooth_heatmap", colours, N=256)
    return smooth_cmap

def truesort(name):
    match = re.search(r'well_(\d+)_electrode_(\d+)_spikes', name)
    if match:
        well_num = int(match.group(1))
        electrode_num = int(match.group(2))
        return (well_num, electrode_num)
    return (float('inf'), float('inf'))

def reshape_wells(data, n_Wells, n_Electrodes, Size, electrode_grid):
    data_array = data.to_numpy().flatten()
    well_grids = []
    
    for well in range(n_Wells):
        start = well * n_Electrodes
        end = start + n_Electrodes
        well_data_flat = data_array[start:end]
        
        # Use electrode_grid to place data in correct positions
        well_data = np.zeros((Size - 2, Size - 2))
        electrode_idx = 0
        for i in range(Size - 2):
            for j in range(Size - 2):
                if electrode_grid[i, j]:
                    well_data[i, j] = well_data_flat[electrode_idx]
                    electrode_idx += 1
        
        expanded_well = np.zeros((Size, Size))
        expanded_well[1:(Size-1), 1:(Size-1)] = well_data
        smoothed_well = gaussian_filter(expanded_well, sigma=0.1, mode='nearest')  
        well_grids.append(smoothed_well)
    return well_grids

def data_prepper(h5_data, parameters, Vars, calculate_electrode_grid):
    time_df = np.arange(0,int(parameters['measurements'])/int(parameters['sampling rate']),1/Vars["fps"])

    with h5py.File(h5_data, "r") as file:
        dataset = file["spike_values"]
        sorted_matches = sorted(dataset, key=truesort)
        
        df_list = []
        df_sum_list = []  
        for filename in sorted_matches:
            well_num, electrode_num = truesort(filename)
            col_name = f"W{well_num}_E{electrode_num}"

            data = file["spike_values"][filename][:, 0]
            timestamps_array = np.round(data, 6)
            n_spikes = len(data)
            frame_counts = []

            for frame_center in time_df:
                window_start = frame_center - 0.5
                window_end = frame_center + 0.5

                count = np.sum((timestamps_array >= window_start) & (timestamps_array < window_end))

                frame_counts.append(count)

            df_list.append(pd.Series(frame_counts, name=col_name))
            df_sum_list.append(pd.Series(n_spikes, name=col_name))

    final_dataframe = pd.concat(df_list, axis=1)
    sum_final_dataframe = pd.concat(df_sum_list, axis=1)
    v_max = []

    for frame_center in time_df:
        window_start = int(frame_center - 50)
        window_end = int(frame_center + 50)
        if window_start < 0:
            window_start = 0
        if window_end > len(final_dataframe):
            window_end = int(len(final_dataframe))
        
        vmax_index = final_dataframe.iloc[window_start:window_end].max().max()

        v_max.append(vmax_index)

    Final_match = re.findall(r'\d+', sorted_matches[len(sorted_matches)-1])
    n_Wells = int(Final_match[0])
    n_Electrodes = int(Final_match[1])
    
    electrode_grid = calculate_electrode_grid(n_Electrodes)
    grid_size = electrode_grid.shape[0]
    Size = grid_size + 2  # Add padding
    
    # Store electrode_grid in Vars for later use
    Vars["electrode_grid"] = electrode_grid

    max_values = final_dataframe.max(axis=0)
    max_df = pd.DataFrame(max_values).T

    # Pass electrode_grid to reshape_wells
    precomputed_data = [reshape_wells(final_dataframe.iloc[frame], n_Wells, n_Electrodes, Size, electrode_grid) for frame in range(Vars["Num frames"])]
    return precomputed_data, n_Wells, n_Electrodes, v_max, Size, sum_final_dataframe, max_df

def make_hm(Vars, Classes, Colour_classes):    
    df = Vars["df"]
    vmin = 0
    vmax = (np.array(Vars["v_max"]).max()) * 0.5
    
    # Get electrode grid dimensions
    electrode_grid = Vars["electrode_grid"]
    grid_size = electrode_grid.shape[0]
    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)

    axs = axs.ravel()

    heatmaps = []
    i = 0
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        # Use dynamic grid size
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        
        for spine in ax.spines.values():
            for class_name, indices in Classes.items():
                if (i+1) in indices:
                    color = Colour_classes.get(class_name, 'grey')
                    spine.set_color(mcolors.to_rgba(color, alpha=0.6))
                    spine.set_linewidth(0.5)
                    break  
        
        hm = ax.imshow(
            np.zeros((grid_size, grid_size)), 
            cmap=Vars["cmap"],
            interpolation='bicubic',
            vmax=vmax,
            vmin=vmin,
            extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5],
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

    def update(frame):
        well_grids = df[frame]
        for hm, well_data in zip(heatmaps, well_grids):
            hm.set_array(well_data)

        time_seconds = frame / Vars["fps"]
        fig.suptitle(f"Spike rate (Spikes/Sec), Time: {time_seconds:.1f}s")
        return heatmaps + [title]

    return fig, update


def make_hm_img(Vars, Classes, Colour_classes):  
    df = Vars["Precomputed Max"]
    n_Electrodes = Vars["n_Electrodes"]
    electrode_grid = Vars["electrode_grid"]
    grid_size = electrode_grid.shape[0]
    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()

    fig.suptitle(f"Total spikes per electrode", fontsize=14, color="white")
    i = 0
    vmax = df.max().max()

    for i, ax in enumerate(axs):
        well_values_flat = df.iloc[0, i * n_Electrodes : (i + 1) * n_Electrodes].to_numpy()
        
        # Use electrode_grid to place values in correct positions
        well_values = np.zeros((grid_size, grid_size))
        electrode_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if electrode_grid[row, col]:
                    well_values[row, col] = well_values_flat[electrode_idx]
                    electrode_idx += 1
                else:
                    well_values[row, col] = np.nan  # NaN for non-electrode positions

        sns.heatmap(well_values, ax=ax, cmap='plasma', vmin=0, vmax=vmax, cbar=False, square=True, xticklabels=False, yticklabels=False, annot=True, fmt=".0f", annot_kws={"size": 6}, mask=np.isnan(well_values))
        
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
    n_Electrodes = Vars["n_Electrodes"]
    max_values = Vars["max_df"].max(axis=0)
    df = pd.DataFrame(max_values).T
    electrode_grid = Vars["electrode_grid"]
    grid_size = electrode_grid.shape[0]
    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()

    fig.suptitle(f"Highest amount of spikes in 1 second per electrode", fontsize=14, color="white")
    i = 0
    vmax = df.max().max()

    for i, ax in enumerate(axs):
        well_values_flat = df.iloc[0, i * n_Electrodes : (i + 1) * n_Electrodes].to_numpy()
        
        # Use electrode_grid to place values in correct positions
        well_values = np.zeros((grid_size, grid_size))
        electrode_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if electrode_grid[row, col]:
                    well_values[row, col] = well_values_flat[electrode_idx]
                    electrode_idx += 1
                else:
                    well_values[row, col] = np.nan  # NaN for non-electrode positions

        cmap = cm.get_cmap('plasma').copy()
        cmap.set_bad(color='#1a1a1a')
        
        sns.heatmap(well_values, ax=ax, cmap=cmap, vmin=0, vmax=vmax, cbar=False, square=True, xticklabels=False, yticklabels=False, annot=True, fmt=".0f", annot_kws={"size": 6}, mask=np.isnan(well_values))
        
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
    # Get electrode grid dimensions if available
    electrode_grid = Vars.get("electrode_grid")
    if electrode_grid is not None:
        grid_size = electrode_grid.shape[0]
    else:
        grid_size = 4  # Default fallback
    
    fig = Figure(figsize=(12, 9), facecolor=Vars["background_color"])
    axs = fig.subplots(Vars["Rows"], Vars["Cols"])
    fig.subplots_adjust(wspace=0.009, hspace=0.009)
    axs = axs.ravel()  

    for ax in axs:
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])

        # Use dynamic grid size
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        
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