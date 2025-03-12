"Code gebasseerd op Axion Biosystem Cross-correlation"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

# Inladen van data wat normaal er al is
H5_data = r"C:\Users\lucap\OneDrive\Documents\BMT\Jaar 4\Stage\Data\Raw_microelectrode_array_data_workshop (2)_output_2025-03-10_13_52_55\output_values.h5"


#%%

def netwerk_graph(dataframe,amnt_elektrodes, label, well, normalisation, color_nodes, show=False, ax=False, title=None): # wordt parameters
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
    # Maak een netwerk
    G = nx.Graph()
    
    # Add Nodes (represent elektrodes)
    for elektro in amnt_elektrodes:
        G.add_node(elektro)
    
    # Add edges as value from dataframe
    for i in range(len(amnt_elektrodes)):
        for j in range(i+1, len(amnt_elektrodes)):  # Alleen de bovenste driehoek om dubbele verbindingen te vermijden
            data = dataframe.iloc[i, j]
        #    if synchroniciteit > 0.00005:  # Je kunt een drempel instellen
            G.add_edge(amnt_elektrodes[i], amnt_elektrodes[j], weight=data)
    
    # Handmatige posities voor een 4x4 grid zonder knopen in de hoeken
    # We maken een 4x4 grid, maar de hoeken (1, 4, 13, 16) worden weggelaten
    grid_positions = [
        (1, 3), (2, 3),     
        (0, 2), (1, 2), (2, 2), (3, 2),    # 2nd row
        (0, 1), (1, 1), (2, 1), (3, 1),    # 3rd row
        (1, 0), (2, 0)    # 4th row 
    ]

    # Give electrodes positions as nodes
    
    pos = {}  # Create an empty dictionary

    for i in range(len(amnt_elektrodes)):  
        electrode = amnt_elektrodes[i]  # Get the electrode number
        position = grid_positions[i]    # Get the corresponding position
        pos[electrode] = position       # Assign to dictionary
 
    # Bepaal de dikte van de randen op basis van het gewicht
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Teken het netwerk
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color= color_nodes, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='white', ax=ax)
    nx.draw_networkx_edges(G, pos, width=np.array(edge_weights)* 250, alpha=0.6, edge_color='black', ax=ax)
    
    if title is False:
        ax.set_title(f"Netwerk of {label} between electrodes off well {well} \n Times normalisation {normalisation}")
    ax.axis('off')

    
    return G


def cross_synchronisation (data, illness): # add parameters value
    
    # Parameters
    electrode_amnt = 12
    electrodes = range(1, electrode_amnt + 1, 1)
    measurements = 1200000
    hertz = 20000
    wells = range(1, 12 + 1, 1) # Assuming 12 wells # Dit moet anders 
    # Calculating date_time
    data_time = measurements/hertz
    
    # Dictionarys
    spike_data = {}
    spike_times = []
    spike_indices = []
    
    # Initialize plot
    ncols = int(np.ceil(np.sqrt(len(wells))))  
    nrows = int(np.ceil(len(wells) / ncols))   
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    
    # Flatten axes if necessary to ensure consistent indexing
    axes = axes.flatten()  
    fig.suptitle("Synchronity Network Analysis for Wells", fontsize=16, fontweight='bold', y=1.05)
    
    with h5py.File(H5_data, 'r') as f:
        for idx, well in enumerate(wells):
           # Select appropriate subplot based on well index
            if idx < len(axes):
                ax = axes[idx]
                print(f"Analysing well: {well}.")
            
            for electrode in electrodes:
                dataset_path = f"/spike_values/well_{well}_electrode_{electrode}_spikes"
                
                if dataset_path in f:
                    # Load spike data
                    spike_values = f[dataset_path][()]
                    
                    # Get correct columns: Time (column 0), amplitude (column 2), index (column 2)
                    spike_times = spike_values[:, 0]  
                    spike_indices = spike_values[:, 2]  
                    
                    # Set in dataframe and use original index als index.
                    df = pd.DataFrame({'Time': spike_times}, index=spike_indices)
                    
                    # Save in dictionary with key (well, electrode)
                    spike_data[(well, electrode)] = df
    
            # All indeces behind eachoter
            all_indices = sorted(set(idx for df in spike_data.values() for idx in df.index))
            max_index = max(all_indices)  
    
            # Make a dataframe with rows from 0 to max_index and columns for each electrode in the well.
            df_presence = pd.DataFrame(index=np.arange(0, max_index + 1))
            
            for (well, electrode), df in spike_data.items():
                col_name = f"electrode_{electrode}"  # Culumn name: 'electrode_1', 'electrode_2', etc.
                df_presence[col_name] = df_presence.index.isin(df.index).astype(int)
           
            # Cross correlation of dataframe
            cross_corr = df_presence.corr()
            cross_corr_df = pd.DataFrame(cross_corr)
            
            # Sum Calculation autocorrelation for each electrode
            auto_corr = df_presence.apply(lambda col: col.autocorr(lag=1))
            auto_corr_df = pd.DataFrame(auto_corr)
            
            # Calculation synchronity
            synchronity_df = cross_corr_df.copy()
            for electrode1 in cross_corr_df.index:
                for electrode2 in cross_corr_df.columns:
                    # Get autocorrelation of the 2 electrodes
                    autocorr_electrode1 = auto_corr_df.loc[electrode1, 0]
                    autocorr_electrode2 = auto_corr_df.loc[electrode2, 0]
                    
                    # Calcute synchronity from function from Axion Biosystem Methods 
                    synchronity = cross_corr_df.loc[electrode1, electrode2] / np.sqrt(autocorr_electrode1 * autocorr_electrode2)
                    synchronity_df.loc[electrode1, electrode2] = synchronity    # Add to dataframe
                    
            # Normalise dataframe between 0 en 1
            scaler = MinMaxScaler()  
            syn_norm = pd.DataFrame(scaler.fit_transform(synchronity_df), columns=synchronity_df.columns)
            syn_norm_df = pd.DataFrame(syn_norm)
            
            ax.set_title(f"Well {well}", fontsize=12, fontweight='bold')
            
            if well in illness:
                netwerk_graph(syn_norm_df, electrodes, "synchronity", well, "250",'cyan', ax=ax)  # Plot each graph on its axis
            else:
                netwerk_graph(syn_norm_df, electrodes, "synchronity", well, "250",'magenta', ax=ax)
                
        plt.tight_layout()
        plt.show()          
                    
                
    return synchronity_df, syn_norm_df

# Wells with illness
kleefstra = np.array([3, 5, 6, 9, 11])
# Calculations synchronicity
synchronity_df, syn_norm_df = cross_synchronisation(H5_data, kleefstra)

#%%
# Calculations synchronicity
synchronity_df, syn_norm_df = cross_synchronisation( H5_data)

# Variations normalisation
syn_norm_df_times = 250 * syn_norm_df

electrode_amnt = 12
electrodes = range(1, electrode_amnt + 1, 1)
netwerk_graph(syn_norm_df_times, electrodes, "synchronity", "1","250" )


#%% Heatmap

# Heatmap plotten met aangepaste kleurenschaal
fig, ax = plt.subplots(figsize=(8, 6))  # We maken 1 enkele plot

# Minimum en maximum waarden voor betere zichtbaarheid
vmin, vmax = 0, 1   

# Synchronie heatmap
sns.heatmap(syn_norm_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={'label': 'Synchroniteit'})
ax.set_title("Synchroniteit tussen elektroden")  # Aangepaste titel

# Axes labels aanpassen
ax.set_xlabel("Electroden")
ax.set_ylabel("Electroden")

# Verbeteren van de layout
plt.tight_layout()
plt.show()

#%%

        