import numpy as np

'''Threshold function - returns threshold value'''
def threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion):
    measurements=data.shape[0]
    # The amount is samples needed to for a 50ms window is calculated
    windowsize = 0.05 * hertz
    windowsize = int(windowsize)
    # Create a temporary list that will contain 50ms windows of data
    windows = []

    # Iterate over the electrode data and create x amount of windows containing *windowsize* samples each
    # For this it is pretty important that the data consists of a multitude of 50ms seconds
    for j in range(0, int(measurements), windowsize):
        windows.append(data[j:j+windowsize])
    windows = np.array(windows) #convert into np.array
    # Create an empty list where all the data identified as spike-free noise will be stored
    noise=[]

    # Now iterate over every 50ms time window and check whether it is "spike-free noise"
    # If all of the electrode should be used to determine the noise window
    if threshold_portion==1:
        loopvalues=range(0,windows.shape[0])
    # If this is not the case, only use a portion of the data to determine the noise level
    else:
        loopvalues=range(0,windows.shape[0])
        step=int(1/threshold_portion)
        loopvalues=loopvalues[::step]
    for j in loopvalues:
        # Calculate the mean and standard deviation
        #mu, std = norm.fit(windows[j])
        std=np.std(windows[j])
        # Check whether the minimum/maximal value lies outside the range of x (defined above) times the standard deviation - if this is not the case, this 50ms box can be seen as 'spike-free noise'
        if not(np.min(windows[j])<(-1*stdevmultiplier*std) or np.max(windows[j])>(stdevmultiplier*std)):
            # 50ms has been identified as noise and will be added to the noise paremeter
            noise = np.append(noise, windows[j][:])

    # Calculate the RMS
    RMS=np.sqrt(np.mean(noise**2))
    threshold_value=RMSmultiplier*RMS

    # Calculate the % of the file that was noise
    noisepercentage=noise.shape[0]/data.shape[0]
    #print(f'{noisepercentage*100}% of data identified as noise')
    return threshold_value

def fast_threshold(data, hertz, stdevmultiplier, RMSmultiplier, threshold_portion):
    measurements=data.shape[0]
    # The stepsize needed for a 50ms window is calculated
    stepsize = int(0.05 * hertz)
    total_steps=int(measurements/stepsize)

    loopvalues=range(0,total_steps)
    step=int(1/threshold_portion)
    loopvalues=loopvalues[::step]
    noise=np.array([])

    for j in loopvalues:
        # Calculate the mean and standard deviation
        #mu, std = norm.fit(windows[j])
        std=np.std(data[j*stepsize:(j+1)*stepsize])
        # Check whether the minimum/maximal value lies outside the range of x (defined above) times the standard deviation - if this is not the case, this 50ms box can be seen as 'spike-free noise'
        if not(np.min(data[j*stepsize:(j+1)*stepsize])<(-1*stdevmultiplier*std) or np.max(data[j*stepsize:(j+1)*stepsize])>(stdevmultiplier*std)):
            # 50ms has been identified as noise and will be added to the noise paremeter
            noise = np.append(noise, data[j*stepsize:(j+1)*stepsize])

    # Calculate the RMS
    RMS=np.sqrt(np.mean(noise**2))
    threshold_value=RMSmultiplier*RMS

    # Calculate the % of the file that was noise
    noisepercentage=noise.shape[0]/measurements
    #print(f'{noisepercentage*100}% of data identified as noise')
    return threshold_value