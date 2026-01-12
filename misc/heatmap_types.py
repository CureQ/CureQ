from heatmap_test import *

"""
This algorithm can be used to create a 12 heatmaps for each electrode.
These heatmaps consist of the 3 frequencies (1kHz, 10kHz and 41.5kHz) 
and 4 different types: Real part, Imaginary part, Modulus and Angle
"""  

def real_imag(df, frequence): # Return two lists: real and imaginary numbers
    reals = []
    imags = []
    for index, row in df.iterrows():
        imp = row [frequence]
        if (type(imp) == str) and (imp != ""): # If there is a value: calculate the modulus
            real, imag = imp.split('-') # split into real and imaginary parts
            real = float(real) # Convert real to float
            imag = float(imag[:-1]) # Remove the 'i' and convert imaginary to float
            reals.append(real)
            imags.append(imag)
        else:
            reals.append(0)
            imags.append(0)
    return reals, imags

def calculate_angles(df, frequence): # Calculate angles of complex numbers
    angles = []
    for index, row in df.iterrows():
        imp = row[frequence] # Collect complex impedance at 41500Hz
        if (type(imp) == str) and (imp != ""): # If there is a value: calculate the modulus
            real, imag = imp.split('-') # split into real and imaginary parts
            real = float(real) # Convert real to float
            imag = float(imag[:-1]) # Remove the 'i' and convert imaginary to float
            angle = math.atan(imag/real) # Calculate angle in radians
        else: # If no value: set value to 0
            angle = 0
        angles.append(angle)
    return angles

def create_heatmaps_dif(dt, well_dims, mapping, outputmap): # Create heatmap with the correct data
    
    e_mesh = electrode_mesh(mapping)

    for welln in range(len(r1)):
        wellrow = ((welln) / max(mapping["WellColumn"]) + 1)
        wellcolumn = (welln) % max(mapping["WellColumn"]) + 1
        wellname = f"{chr(int(wellrow + 64))}{wellcolumn}"
        fig, axs = plt.subplots(3, 4, figsize=(8, 8))
        fig.subplots_adjust(hspace = 0.009, wspace = 0.009)
        axs = axs.ravel()
        try:
            wells = [dt[0][welln],
                    dt[1][welln],
                    dt[2][welln],
                    dt[3][welln],
                    dt[4][welln],
                    dt[5][welln],
                    dt[6][welln],
                    dt[7][welln],
                    dt[8][welln],
                    dt[9][welln],
                    dt[10][welln],
                    dt[11][welln]]
        except:
            return

        for index, ax in enumerate(axs):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 5)
            ax.set_ylim(5,0)
            ax.scatter(
                e_mesh[0], e_mesh[1],
                color='red',
                s=5,          # grootte van de punten
                marker='o',    # type marker
                edgecolors='black'
            )
            for spine in ax.spines.values():
                spine.set_color("red")
                spine.set_linewidth(0.5)
        
            hm = ax.imshow(
                wells[index],
                cmap = cmap_creation(),
                interpolation = 'gaussian',
                vmax = 1,
                vmin = 0,
                origin = 'upper'
            )

        fig.patch.set_facecolor("black")

        pos = axs[-1].get_position()
        norm = mcolors.Normalize(0, 1.0)
        sm = cm.ScalarMappable(cmap = cmap_creation(), norm=norm)
        cbar = fig.colorbar(sm, ax=axs.tolist(), cax=fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height * 3]))
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.tick_params(axis='y', colors='white')

        col_titles = ["real", "imaginary", "modulus", "angle"]
        for ax, col in zip(axs[0:4], col_titles):
            ax.set_title(col, fontsize=12, pad=10, color='white')

        row_titles = ["1Khz", "10kHz", "41.5kHz"]
        for ax, row in zip([axs[0], axs[4], axs[8]], row_titles):
            ax.set_ylabel(row, fontsize=12, labelpad=10, color='white')

        fig.savefig(f"{outputmap}/{wellname}.png", dpi=2000, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

"""
inputfile = "D:/mea_data/2025_44_dagen_iv/Bow_div44.h5"
outputmap = "D:/mea_data/2025_44_dagen_iv/heatmaps_norm_per"
"""


inputfile = "D:/mea_data/2024_04_09_GLS_CTRL/20240409_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_09_GLS_CTRL/heatmaps"

inputfile = "D:/mea_data/2024_04_10_GLS_CTRL/20240410_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_10_GLS_CTRL/heatmaps"

inputfile = "D:/mea_data/2024_04_11_GLS_CTRL/20240411_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_11_GLS_CTRL/heatmaps"

inputfile = "D:/mea_data/2024_04_12_GLS_CTRL/20240412_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_12_GLS_CTRL/heatmaps"

inputfile = "D:/mea_data/2024_04_15_GLS_CTRL/20240415_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_15_GLS_CTRL/heatmaps"

inputfile = "D:/mea_data/2024_04_16_GLS_CTRL/20240416_GLS_CTRL.h5"
outputmap = "D:/mea_data/2024_04_16_GLS_CTRL/heatmaps"

numbers = ["10", "11", "12", "15", "16"]
for number in numbers: 
    print("Creating heatmaps for day: ", number)
    inputfile = f"D:/mea_data/2024_04_{number}_GLS_CTRL/202404{number}_GLS_CTRL.h5"
    outputmap = f"D:/mea_data/2024_04_{number}_GLS_CTRL/heatmaps"

    impedance = load_impedance(inputfile)
    mapping = load_mapping(inputfile)


    # Real and imag
    r1, i1 = real_imag(impedance, '1000')
    rn1 = reshape_wells(normalise_data(r1, 10000, max(r1)), mapping)
    in1 = reshape_wells(normalise_data(i1, 10000, max(i1)), mapping)
    moduluses_1 = calculate_modulus(impedance, '1000')
    mn1 = reshape_wells(normalise_data(moduluses_1, 10000, max(moduluses_1)), mapping)
    a1 = reshape_wells(calculate_angles(impedance, "1000"), mapping)

    r10, i10 = real_imag(impedance, '10000')
    rn10 = reshape_wells(normalise_data(r10, 10000, max(r10)), mapping)
    in10 = reshape_wells(normalise_data(i10, 10000, max(i10)), mapping)
    moduluses_10 = calculate_modulus(impedance, '10000')
    mn10 = reshape_wells(normalise_data(moduluses_10, 10000, max(moduluses_10)), mapping)
    a10 = reshape_wells(calculate_angles(impedance, "10000"), mapping)

    r41, i41 = real_imag(impedance, '41500')
    rn41 = reshape_wells(normalise_data(r41, 10000, max(r41)), mapping)
    in41 = reshape_wells(normalise_data(i41, 10000, max(i41)), mapping)
    moduluses_41 = calculate_modulus(impedance, '41500')
    mn41 = reshape_wells(normalise_data(moduluses_41,10000, max(moduluses_41)), mapping)
    a41 = reshape_wells(calculate_angles(impedance, "41500"), mapping)

    well_dims = calculate_well_dimensions(mapping)

    dt = [rn1, in1, mn1, a1, rn10, in10, mn10, a10, rn41, in41, mn41, a41]
    create_heatmaps_dif(dt, well_dims, mapping, outputmap) # Create heatmap with the correct data


