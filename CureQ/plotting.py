import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from functools import partial
from KDEpy import FFTKDE
import seaborn

'''Create raster plots for all the electrodes'''
def raster(electrodes, electrode_amnt, samples, hertz, filename):
    # Check which electrodes are given, and how these will be plotted
    i=0
    
    filename=filename[:-3]
    filename=filename+"_values"
    while i < len(electrodes):
        well_spikes=[]
        burst_spikes=[]
        # Collect all the data from a single well
        while i<len(electrodes): # and ((electrodes[i])%electrode_amnt)!=0:
            electrode = electrodes[i] % electrode_amnt + 1
            well = round(electrodes[i] / electrode_amnt + 0.505)
            path=f'spike_values'
            spikedata=np.load(f'{filename}/{path}/well_{well}_electrode_{electrode}_spikes.npy')
            path='burst_values'
            burstdata=np.load(f'{filename}/{path}/well_{well}_electrode_{electrode}_burst_spikes.npy')
            well_spikes.append(spikedata[:,0])
            if len(burstdata)>0:
                burst_spikes.append(burstdata[:,0])
            else:
                burst_spikes.append([])
            if ((electrodes[i]+1)%electrode_amnt)==0: break
            i+=1
        amount_of_electrodes=len(burst_spikes)
        plt.cla()
        plt.rcParams["figure.figsize"] = (22,5)
        end_electrode=((electrodes[i-1]+1)%12)+1
        start_electrode = end_electrode-amount_of_electrodes
        lineoffsets1=np.arange(start_electrode+1, end_electrode+1)
        plt.eventplot(well_spikes, alpha=0.5, lineoffsets=lineoffsets1)
        plt.eventplot(burst_spikes, alpha=0.5, color='red', lineoffsets=lineoffsets1)
        plt.xlim([0,samples/hertz])
        plt.ylim([start_electrode, end_electrode+1])
        plt.yticks(lineoffsets1)
        plt.title(f"Well {well}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Electrode")
        plt.show()
        i+=1

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
def fancyplot(filename, wells, electrode_amnt, measurements, hertz, resolution, kernel_size, aspectratios, colormap):
    # Convert the filename so the algorithm can look through the folders
    filename=filename[:-3]
    filename=filename+"_values"
    spikepath=f'{filename}/spike_values'
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
        return fig
        # fig.show()
        # fig.write_html(f"{filename}/well_{well}_graph.html")

'''Plots the KDE of all electrodes (x) of a well in x amount of subplots'''
def well_electrodes_kde(outputpath, well, electrode_amnt, measurements, hertz, bandwidth=1):
    spikepath=f'{outputpath}/spike_values'
    data_time=measurements/hertz
    fig, ax = plt.subplots(electrode_amnt, 1, sharex=True, sharey=True)
    for electrode in range(electrode_amnt):
        spikedata=np.load(f'{spikepath}/well_{well}_electrode_{electrode+1}_spikes.npy')
        if len(spikedata[:,0])>0:
            y = FFTKDE(bw=bandwidth, kernel='gaussian').fit(spikedata[:,0]).evaluate(grid_points=np.arange(0, data_time, 0.001))
            y=np.array(y)*len(spikedata[:,0])
            x=np.arange(0, data_time, 0.001)
        else:
            x=np.arange(0, data_time, 0.001)
            y=np.zeros(len(x))
        ax[electrode].plot(x,y)
        ax[electrode].set_ylabel(f"E: {electrode+1}")
    ax[-1].set_xlabel("Time in seconds")
    ax[0].set_xlim([0, measurements/hertz])
    ax[0].set_title(f"Well: {well} activity")
    return fig