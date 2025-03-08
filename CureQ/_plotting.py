import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
import json
import re
from scipy.stats import sem
import h5py


def well_electrodes_kde(outputpath, well, parameters, bandwidth=1):
    """
    Plot a kernel density estimate of all electrodes in a single plot.

    Parameters
    ----------
    outputpath : str
        The folder in where to find the data required to create the KDEs
    well : int
        The well number which is to be used for the KDE
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed.
    bandwidth : float, optional
        The bandwidth used for the KDE, passed to KDEpy

    Returns
    -------
    fig : matplotlib Figure

    """

    electrode_amnt = parameters['electrode amount']
    measurements = parameters['measurements']
    hertz = parameters['sampling rate']

    # Where to find the spike-data
    spikepath=f'{outputpath}/spike_values'
    data_time=measurements/hertz

    # Create matplotlib figure
    fig = Figure()
    
    gs = GridSpec(electrode_amnt, 1, figure=fig)
    axes=[]

    first_iteration=True
    # Loop through all electrodes
    for electrode in range(1, electrode_amnt+1):
        if first_iteration:
            ax = fig.add_subplot(gs[electrode_amnt-electrode])
            first_iteration=False
        else:
            ax = fig.add_subplot(gs[electrode_amnt-electrode], sharex=axes[0], sharey=axes[0])
        
        # Load in the spike data and create a KDE using KDEpy for each well
        output_hdf_file=parameters['output hdf file']

        with h5py.File(output_hdf_file, "r") as f:
            dataset=f[f"spike_values/well_{well}_electrode_{electrode}_spikes"]
            spikedata=dataset[:]

        if len(spikedata[:,0])>0:
            y = FFTKDE(bw=bandwidth, kernel='gaussian').fit(spikedata[:,0]).evaluate(grid_points=np.arange(0, data_time, 0.001))
            y=np.array(y)*len(spikedata[:,0])
            x=np.arange(0, data_time, 0.001)
        else:
            x=np.arange(0, data_time, 0.001)
            y=np.zeros(len(x))
        # Plot the KDE and give the subplot a label
        ax.plot(x,y)
        ax.set_ylabel(f"E: {electrode}")
        ax.set_xlim([0, measurements/hertz])
        axes.append(ax)
    # Plot layout
    axes[0].set_xlabel("Time (s)")
    axes[0].set_xlim([0, measurements/hertz])
    axes[-1].set_title(f"Well: {well} activity")
    
    return fig

def get_defaultcolors():
    """
    Get default colors used for plotting.

    Returns
    -------
    default_colors : list
        List of hex codes of default colors.
    
    """

    default_colors = [
    "#800000",  # maroon
    "#008080",  # teal
    "#32CD32",  # limegreen
    "#4B0082",  # indigo
    "#FFA500",  # orange
    "#FF6347",  # tomato
    "#EE82EE",  # violet
    "#00FFFF",  # cyan
    "#808080",  # gray
    "#9400D3",  # darkviolet
    "#6A5ACD",  # slateblue
    "#2E8B57"   # seagreen
]

    return default_colors


def feature_boxplots(features, labels, output_fileadress, colors=None, show_datapoints=True, discern_wells=False, well_amnt=None):
    """
    Creates a pdf file with boxplots for all different features and labels

    Parameters
    ----------
    features : pandas dataframe
        Dataframe containing features from a MEA experiment
    labels : dict
        Information about the contents of the well in the following format: {'Control':[1, 2, 3], 'Mutated':[4, 5, 6]}
    output_fileadress : str
        Location where the file will be saved, must be a .pdf file
    colors : list, optional
        List of colors to be used for the boxplots. Will use default colors if left empty
    show_datapoints : bool, optional
        Whether to show the individual datapoints on the boxplot
    discern_wells : bool, optional
        If True, will assign a random colour to datapoints from each well
    well_amnt : int, optional
        The amount of wells in the MEA-plate, this parameter is required if discern_wells = True

    Returns
    -------
    pdf_path : str
        Path of output pdf-file

    """

    if well_amnt is not None:
        rainbow_colors = [f"#{np.random.randint(0, 0xFFFFFF):06X}" for _ in range(well_amnt)]

    not_features=["Well", "Active_electrodes"]

    pdf_path=output_fileadress
    pdf = PdfPages(pdf_path)

    # Set default colors
    defaultcolors = get_defaultcolors()
    if colors==None:
        colors = defaultcolors
        if len(labels)>len(colors):
            raise ValueError(f"You have supplied more labels ({len(labels)}) than there are colors available ({len(colors)}). Please supply colors manually using the 'colors' parameter.")
    else:
        if len(labels)>len(colors):
            raise ValueError(f"You have supplied more labels ({len(labels)}) than colors ({len(colors)}). Please supply more color values, or use the default colors ({len(defaultcolors)} colors)")

    for feature in features.columns:
        if feature not in not_features:
            fig, ax = plt.subplots()
            featuredata=features[feature]
            plotdata=[]
            for key_index, key in enumerate(labels.keys()):
                wells=labels[key]
                # -1 to turn wells into indexes
                temp_data=featuredata[(np.array(labels[key])-1)]
                plotdata.append(temp_data)
                temp_data=list(temp_data)
                if show_datapoints:
                    for i, well in enumerate(wells):
                        x = np.random.normal(key_index+1, 0.07, size=1)
                        if discern_wells and (well_amnt is not None):
                            color=rainbow_colors[well%len(rainbow_colors)]
                        else:
                            color='black'
                        plt.scatter(x[0], temp_data[i], alpha=0.5, color=color, zorder=10, s=1)
            # Remove NaNs
            plotdata=[d[~np.isnan(d)] for d in plotdata]
            bplot=ax.boxplot(plotdata, vert=True, labels=labels.keys(), patch_artist=True, widths=0.5)
            ax.set_title(f"{feature}")
            ax.set_ylabel(feature)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bplot[element], color='black')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    pdf.close()
    return pdf_path


def combined_feature_boxplots(folder, labels, output_fileadress, colors=None, show_datapoints=True, discern_wells=False, well_amnt=None):
    """
    Combine the features of multiple measurements together

    Parameters
    ----------
    folder : str
        folder in which to look for featurefiles
    labels : dict
        Information about the contents of the well in the following format: {'Control':[1, 2, 3], 'Mutated':[4, 5, 6]}
    output_fileadress : str
        Location where the file will be saved, must be a .pdf file
    colors : list, optional
        List of colors to be used for the boxplots. Will use default colors if left empty
    show_datapoints : bool, optional
        Whether to show the individual datapoints on the boxplot
    discern_wells : bool, optional
        If True, will assign a random colour to datapoints from each well
    well_amnt : int, optional
        The amount of wells in the MEA-plate, this parameter is required if discern_wells = True

    Returns
    -------
    pdf_path : str
        Path of output pdf-file

    """
    
    # Find all featurefiles
    featurefiles = []
    print("Extracting features from:")
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("Features.csv"):
                featurefiles.append(os.path.join(root, file))
                print(os.path.join(root, file))

    # Combine the data in one large dataframe
    dataframes=[]
    for featurefile in range(len(featurefiles)):
        dataframes.append(pd.read_csv(featurefiles[featurefile]))
    original_rows=dataframes[0].shape[0]
    dataframe=pd.concat(dataframes, ignore_index=True)

    # Update the labels
    combined_labels=deepcopy(labels)
    for i in range(1, len(featurefiles)):
        for key in labels.keys():
            combined_labels[key]=((np.append(combined_labels[key], (np.array(labels[key]))+(original_rows*i))).astype(int)).tolist()

    # Save the data
    with open(f"{folder}/combined_labels.json", 'w') as outfile:
        json.dump(combined_labels, outfile)
    dataframe.to_csv(f"{folder}/combined_dataframe.csv")

    # Plot the data
    pdf_path=feature_boxplots(dataframe, combined_labels, output_fileadress=output_fileadress, colors=colors, show_datapoints=show_datapoints, discern_wells=discern_wells, well_amnt=well_amnt)
    return pdf_path



def features_over_time(folder, labels, div_prefix, output_fileadress, colors=None, show_datapoints=False):
    """
    Show the different features between groups over time.
    The error bars represent the standard error of the mean.

    Parameters
    ----------
    folder : str
        folder in which to look for featurefiles
    labels : dict
        Information about the contents of the well in the following format: {'Control':[1, 2, 3], 'Mutated':[4, 5, 6]}
    div_prefix : str
        The prefix used to indicate the age of the cells (e.g. DIV, t)
    output_fileadress : str
        Location where the file will be saved, must be a .pdf file
    colors : list, optinal
        List of colors (hex codes) to be used for the boxplots. Will use default colors if left empty
    show_datapoints : bool, optional
        Whether to show the individual datapoints on the boxplot

    Returns
    -------
    pdf_path : str
        Path of output pdf-file

    """

    # Set default colors
    defaultcolors = get_defaultcolors()
    if colors==None:
        colors = defaultcolors
        if len(labels)>len(colors):
            raise ValueError(f"You have supplied more labels ({len(labels)}) than there are colors available ({len(colors)}). Please supply colors manually using the 'colors' parameter.")
    else:
        if len(labels)>len(colors):
            raise ValueError(f"You have supplied more labels ({len(labels)}) than colors ({len(colors)}). Please supply more color values, or use the default colors ({len(defaultcolors)} colors)")

    # Function to sort the filenames by DIV
    def sort_filenames_by_number(filenames, prefix):
        sorted_filenames = []
        for filename in filenames:
            match = re.search(f"{prefix}(\d+)", filename)
            if match:
                num_value = int(match.group(1))
                sorted_filenames.append((num_value, filename))
        sorted_filenames.sort()
        numbers, filenames = zip(*sorted_filenames) if sorted_filenames else ([], [])
        return list(numbers), sorted_filenames

    # Identify all featurefiles in the main folder
    print("Extracting features from:")
    featurefiles = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("Features.csv"):
                featurefiles.append(os.path.join(root, file))
                print(os.path.join(root, file))

    nums, sorted_featurefiles = sort_filenames_by_number(featurefiles, div_prefix)

    # Read the featurefiles as dataframes
    sorted_dataframes=[]
    all_nums=[]
    for featurefile in range(len(sorted_featurefiles)):
        sorted_dataframes.append((sorted_featurefiles[featurefile][0], pd.read_csv(sorted_featurefiles[featurefile][1])))
        all_nums.append(sorted_featurefiles[featurefile][0])
    nums=np.sort(np.unique(all_nums))

    not_features=["Well", "Active_electrodes"]
    features = sorted_dataframes[0][1].columns
    graphlabels=np.array(nums).astype(str)
    for i in range(len(graphlabels)):
        graphlabels[i]=f"{div_prefix} {graphlabels[i]}"

    # Create pdf
    pdf = PdfPages(output_fileadress)

    # Convert labels from wells to indexes (subtract 1)
    for label in labels.keys():
        for value in range(len(labels[label])):
            labels[label][value]= labels[label][value]-1

    # Loop over all features
    for i in range(len(features)):
        # Check if these are the features we want
        if features[i] not in not_features:      
            fig, ax = plt.subplots(figsize=(8,4.5))
            means_list=[]
            errors_list=[]
            # Loop over all groups
            for index, key in enumerate(labels.keys()):
                means=[]
                errors=[]
                # Loop over all the DIVs
                for pos, num in enumerate(nums):
                    temp_data=[]
                    found=0
                    # For each DIV, loop over all measurement with that DIV
                    for dataframe in range(len(sorted_dataframes)):
                        if sorted_dataframes[dataframe][0]==num:
                            temp_data.append((sorted_dataframes[dataframe][1][features[i]][labels[key]]))
                            found+=1
                    temp_data=np.array(temp_data)
                    temp_data = temp_data[~np.isnan(temp_data)]
                    # Calculate the mean and std
                    means.append(np.nanmean(temp_data))
                    errors.append(sem(temp_data))
                    if show_datapoints:
                        # Add some jitter to the datapoints
                        plt.scatter(x=np.array([pos]*len(temp_data))+np.random.uniform(-0.2, 0.2, len(temp_data)), y=temp_data, color=colors[index], s=5, alpha=0.5)
                means_list.append(means)
                errors_list.append(errors)
                plt.errorbar(x=range(len(nums)), y=means, yerr=errors, capsize=5, label=key, color=colors[index])
            plt.title(features[i])
            plt.ylabel(features[i])
            plt.legend()
            plt.xticks(ticks=range(len(nums)), labels=graphlabels, rotation=45, ha="right")
            pdf.savefig()
            plt.close()
    pdf.close()

    return output_fileadress