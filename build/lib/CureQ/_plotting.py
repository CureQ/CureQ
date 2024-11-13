import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from functools import partial
from KDEpy import FFTKDE
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages

'''This function will calculate a 3d guassian kernel'''
def K(x, H):
    # unpack two dimensions
    x1, x2 = x
    # extract four components from the matrix inv(H)
    a, b, c, d = np.linalg.inv(H).flatten()
    
    scale = 2*np.pi*np.sqrt( np.linalg.det(H))
    return np.exp(-(a*x1**2 + d*x2**2 + (b+c)*x1*x2)/2) / scale

'''This function will insert a 3d gaussian kernel at the location of every datapoint/spike'''
def KDE(x, H, data):
    # unpack two dimensions
    x1, x2 = x
    # prepare the grid for output values
    output = np.zeros_like(x1)
    # process every sample
    for sample in data:
        output += K([x1-sample[0], x2-sample[1]], H)
    return output

'''Create a 3D view of a single well. Every spike will be represented as a 3D gaussian kernel'''
def fancyplot(outputpath,       # Where to find the spikedata
              wells,            # Which wells to generate a plot for
              electrode_amnt,   # Amount of electrodes in a well
              measurements,     # Measurements done (time*sampling rate)
              hertz,            # Sampling rate
              resolution,       # Resolution of the 3d graph - higher resolution lead to significantly more processing time
              kernel_size,      # Size of the gaussian kernel
              aspectratios,     # Ratios of the 3d plot
              colormap          # Colourmap used - can be any colourmap supported by plotly
              ):

    # Example usage:
    # fancyplot("C:/MEA_data/output_folder_of_experiment", [15], 12, 3000000, 20000, 5, 1, [0.5,0.25,0.5], "deep")

    spikepath=f'{outputpath}/spike_values'
    for well in wells:
        kdedata=[]
        for electrode in range(1,electrode_amnt+1):
            # Load in spikedata
            spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
            spikedata=spikedata[:,:2]
            spikedata[:,1]=electrode
            for spike in spikedata:
                kdedata.append(spike)
        time=measurements/hertz

        # covariance matrix
        # This determines the shape of the gaussian kernel
        H = [[1, 0],
            [0, 1]]

        data = np.array(kdedata)
        # fix arguments 'H' and 'data' of KDE function for further calls
        KDE_partial = partial(KDE, H=kernel_size*np.array(H), data=data)

        # draw contour and surface plots
        func=KDE_partial

        # create a np xy grid using the dimensions of the data
        yres=int(resolution*time)
        xres=int(resolution*electrode_amnt)
        y_range = np.linspace(start=0, stop=time, num=yres)
        x_range = np.linspace(start=0, stop=electrode_amnt, num=xres)
        print(f"Creating 3D plot, resolution x: {xres} by y: {yres}")

        y_grid, X_grid = np.meshgrid(y_range, x_range)
        Z_grid = func([y_grid, X_grid])
        
        fig = go.Figure(data=[go.Surface(y=y_grid, x=X_grid, z=Z_grid, colorscale=colormap)])
        fig.update_layout(
        scene = dict(
            xaxis = dict(range=[1,electrode_amnt]),
            yaxis = dict(range=[0,time]),
            ),
        margin=dict(r=20, l=10, b=10, t=10))

        # change the aspect ratio's of the plot
        fig.update_layout(scene_aspectmode='manual',
                            title=f"Well: {well}",
                            scene_aspectratio=dict(x=(electrode_amnt/10)*aspectratios[0],
                                           y=(time/10)*aspectratios[1],
                                           z=1*aspectratios[2]))
        fig.update_layout(scene = dict(
                    xaxis_title='Electrode',
                    yaxis_title='Time in seconds',
                    zaxis_title='KDE'),)
        fig.write_html(f"{outputpath}/figures/well_{well}_3dgraph.html")

'''Plots the KDE of all electrodes (x) of a well in x amount of subplots'''
def well_electrodes_kde(outputpath,         # Where to find the spike information
                        well,               # Which well to analyse
                        electrode_amnt,     # Amount of electrodes per well
                        measurements,       # Measurements done (time*sampling rate)
                        hertz,              # Sampling frequency
                        bandwidth=1         # Bandwidth of the KDE
                        ):
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
        spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode}_spikes.npy')
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
    axes[-1].set_xlabel("Time in seconds")
    axes[0].set_xlim([0, measurements/hertz])
    axes[0].set_title(f"Well: {well} activity")
    
    return fig