#%% Imports
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

#%% Setup
# The path to the counted density in ILastik
density_path = "D:/mea_data/cellcountvalidation/counted_cells.csv"

# The paths to the hdf5 files for the different impedances
path_09 = "D:/mea_data/cellcountvalidation/09_impedance.txt"
path_10 = "D:/mea_data/cellcountvalidation/10_impedance.txt"
path_12 = "D:/mea_data/cellcountvalidation/12_impedance.txt"
path_16 = "D:/mea_data/cellcountvalidation/16_impedance.txt"

mapping_path = "D:/mea_data/cellcountvalidation/mapping.txt"
              

#%% Functions

def load_impedance_txt(fileadress):
    outlist = []
    with open(fileadress, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 0:
                value = 0
            else:
                value = float(parts[2].split('-')[0])
            outlist.append(value)
    return outlist
    
def load_mapping_txt(fileadress):
    cols =  ["Index", "WellRow", "WellColumn", "ElectrodeColumn", "ElectrodeRow"]
    mapping = pd.DataFrame(columns = cols)
    with open(fileadress, 'r') as f:
        i = 0
        for line in f:
            parts = line.split()
            temp_df = pd.DataFrame({"Index": [int(float(parts[0]))],
                                    "WellRow": [int(float(parts[1]))],
                                    "WellColumn":[int(float(parts[2]))],
                                    "ElectrodeColumn":[int(float(parts[3]))],
                                    "ElectrodeRow":[int(float(parts[4]))]
                                                })
            mapping = pd.concat([mapping, temp_df])
            i += 1
        mapping.set_index("Index", inplace=True)
        return mapping

def normalise_impedance(input):
    # Create an empty list for the outputs
    out = []
    maxvalue = max(input)
    minvalue = min(input)
    # Loop through each electrode in the impendance data
    for value in input:

        # Normalise the impedance for this electrode
        norm = (value - minvalue) / (maxvalue - minvalue)

        # Save the normalised impedance
        out.append(norm)

    return out

def normalise_density(input):
    # Create an empty list for the outputs
    out = []
    maxvalue = max(input)
    # Loop through each electrode in the impendance data
    for value in input:

        # Normalise the impedance for this electrode
        norm = value / maxvalue

        # Save the normalised impedance
        out.append(norm)

    return out

def plot_correlation(density, impedance, title = ""):
    d = {'density': density, 'impedance': impedance}
    df = pd.DataFrame(data=d)
    df = df[(df["density"] !=0) & (df["impedance"] !=0)]
    sns.regplot(data = df, x = 'density', y = 'impedance', ci = 95, scatter_kws ={"alpha":0.6}, line_kws ={"color":"red"})
    plt.xlabel("density")
    plt.ylabel("impedantie")
    plt.title(title)
    print(title)
    ps_test = scipy.stats.pearsonr(density, impedance, alternative='two-sided')
    print(ps_test)
    plt.show()

#%% Data retrieval impedance
impedance_09 = load_impedance_txt(path_09)
impedance_10 = load_impedance_txt(path_10)
impedance_12 = load_impedance_txt(path_12)
impedance_16 = load_impedance_txt(path_16)

norm_impedance_09 = normalise_impedance(impedance_09)
norm_impedance_10 = normalise_impedance(impedance_10)
norm_impedance_12 = normalise_impedance(impedance_12)
norm_impedance_16 = normalise_impedance(impedance_16)


# Load mapping
mapping = load_mapping_txt(mapping_path)



#%% Data retrieval density
density_df = pd.read_csv(density_path, header = None)
density_df = density_df[0].str.split(',', expand=True)
density_df.columns = ['electrode', 'value']

# Make list of 0 with length of amount of electrodes (384) for every day measured
density_09 = [0 for i in range(384)]
density_10 = [0 for i in range(384)]
density_12 = [0 for i in range(384)]
density_16 = [0 for i in range(384)]

# Loop trough density dataframe and get day and location on plate from mapping.
# This will make the density list have the same structure as the impedance.
# The impedance lists are already based on mapping, due to Axion's workflow.
for index, row in density_df.iterrows():
    value = float(row["value"])
    electrode_name = row["electrode"] # 16_D6_0_r4k4 is an example of a name (sorry for the crypticness here) (timestress)
    day = int(electrode_name.split("_")[0]) # Day
    wc = int(electrode_name.split("_")[1][1]) # Well col
    wr = int(ord(electrode_name.split("_")[1][0])) - 64 # Well row
    er = int(electrode_name.split("_")[3][1]) # Electrode row
    ec = int(electrode_name.split("_")[3][3]) # Electrode col

    # Get index from mapping
    index = mapping.loc[(mapping['WellRow'] == wr) & 
                        (mapping['WellColumn'] == wc) & 
                        (mapping['ElectrodeColumn'] == ec) & 
                        (mapping['ElectrodeRow'] == er)
                        ].index[0]
    
    if day == 9:
        density_09[index - 1] = value
    elif day == 10:
        density_10[index - 1] = value
    elif day == 12:
        density_12[index - 1] = value
    elif day == 16:
        density_16[index - 1] = value
    else:
        print("ERROR: DAY NOT FOUND")

norm_density_09 = normalise_density(density_09)
norm_density_10 = normalise_density(density_10)
norm_density_12 = normalise_density(density_12)
norm_density_16 = normalise_density(density_16)


# Execute pearson correlation test
plot_correlation(norm_density_09, norm_impedance_09, "Correlation on 9th of april")
plot_correlation(norm_density_10, norm_impedance_10, "Correlation on 10th of april")
plot_correlation(norm_density_12, norm_impedance_12, "Correlation on 12th of april")
plot_correlation(norm_density_16, norm_impedance_16, "Correlation on 16th of april")
