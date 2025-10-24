#%% imports
import pandas as pd
import seaborn as sn
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import h5py


#%% Functions

def cmap_creation(): # Create a colormap for the heatmap
    colours = [
        (0.0, "black"),   
        (1/4, "blue"),   
        (2/4, "cyan"),
        (3/4, "green"),   
        (1, "yellow")
    ]

    # Create a smooth LinearSegmentedColormap
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
    
    for index, row in mapping.iterrows():
        viability = data[index]
        wr = row['WellRow']  # Convert to 0-based index
        wc = row['WellColumn']  # Convert to 0-based index
        er = row['ElectrodeRow']  # Convert to 0-based index
        ec = row['ElectrodeColumn']  # Convert to 0-based
        well_number = ((wr - 1) * mapping['WellColumn'].max()) + wc
        well_grids[well_number - 1][er,ec] = viability

    for i in range(len(well_grids)):
        temp = np.flipud(well_grids[i])
        well_grids[i] = temp

    return well_grids

def calculate_modulus(df):
    moduluses = []
    for index, row in df.iterrows():
        imp = row['41500']
        if (type(imp) == str) and (imp != ""): # If there is a value: calculate the modulus
            real, imag = imp.split('-') # split into real and imaginary parts
            real = float(real) # Convert real to float
            imag = float(imag[:-1]) # Remove the 'i' and convert to float
            mod = (real**2 + imag**2)**0.5 # Calculate modulus of complex number
        else: # If no value: set value to 0
            mod = 0
        moduluses.append(mod)
    return moduluses

def normalise_data(data):
    background = 12000
    baseline = max(data) # TODO aanpassen
    out = []

    for mod in data:
        norm = (mod - background) / (baseline - background)
        out.append(norm)
    return out

def test_data_set(mapping): # TODO this is a testfunc to create a dataset for plotting
    values = []
    with open("testdata.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        for index, row in mapping.iterrows():
            print(f"Row: {row['ElectrodeRow']}, col: {row['ElectrodeColumn']}")
            if row["ElectrodeRow"] == 1 and row["ElectrodeColumn"] == 1:
                spamwriter.writerow([index + 1, 1])
            elif row["ElectrodeRow"] == 4 and row["ElectrodeColumn"] == 4:
                spamwriter.writerow([index + 1, 0.5])
            else:
                spamwriter.writerow([index + 1, 0])
        return values



# Code
fileadress = "D:/mea_data/2024_04_16_GLS_CTRL/20240416_GLS_CTRL.h5"

print("Loading data...")
with h5py.File(fileadress, 'r') as h5file:
    mapping = pd.DataFrame(np.array(h5file["Data/channel_map"]).astype(int), columns = ["Index", "WellRow", "WellColumn", "ElectrodeColumn", "ElectrodeRow"])
    impedance = pd.DataFrame(np.array(h5file["Data/impedance"]).astype(str), columns = ["1000", "10000", "41500"])

print(impedance)
print("Calculating moduluses...")

# Calculate wells and electrodes as tuples
wells = (mapping["WellRow"].max(), mapping["WellColumn"].max())

moduluses = calculate_modulus(impedance)

normalised = normalise_data(moduluses)


print("Reshaping data...")
well_data = reshape_wells(normalised, mapping)

#test_well_data = reshape_wells(impedance['values'], mapping)

print("Creating plot...")
fig, axs = plt.subplots(wells[0],wells[1], figsize=(8, 6))
fig.subplots_adjust(hspace = 0.009, wspace=0.009)

axs = axs.ravel()

print("plotting heatmaps...")
for index, ax in enumerate(axs):

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)

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
fig.show()
t = input("Press Enter to close...")


# %%
