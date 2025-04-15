import numpy as np

def fast_threshold(data, parameters):
    """
    Calculate the threshold for a single electrode

    Parameters
    ----------
    data : list, np.ndarray
        Raw single electrode data.
    parameters : dict
        Dictionary containing global paramaters. The function will extract the values needed.

    Returns
    -------
    threshold_values : float
        Threshold for the electrode.

    Notes
    -----
    This function will first determine periods of noise, then use these periods to calculate a threshold value.
    
    """  
        
    measurements=data.shape[0]

    # The stepsize needed for a 50ms window is calculated
    stepsize = int(0.05 * parameters['sampling rate'])
    # Calculate how many steps there are in the measurement
    total_steps=int(measurements/stepsize)

    # Determine which parts of the data the algorithm will use for calculating the threshold
    loopvalues=range(0,total_steps)
    step=int(1/parameters['threshold portion'])
    loopvalues=loopvalues[::step]

    noise=np.array([])

    # Loop over certain position in the raw data
    for j in loopvalues:
        # Calculate the standard deviation
        std=np.std(data[j*stepsize:(j+1)*stepsize])
        # Check whether the minimum/maximal value lies outside the range of *stdevmultiplier* times the standard deviation - if this is not the case, this 50ms box can be seen as 'spike-free noise'
        if not(np.min(data[j*stepsize:(j+1)*stepsize])<(-1*parameters['standard deviation multiplier']*std) or np.max(data[j*stepsize:(j+1)*stepsize])>(parameters['standard deviation multiplier']*std)):
            # 50ms has been identified as noise and will be added to the noise paremeter
            noise = np.append(noise, data[j*stepsize:(j+1)*stepsize])

    # Calculate the RMS of all the noise
    RMS=np.sqrt(np.mean(noise**2))
    # Calculate the threshold
    threshold_value=parameters['rms multiplier']*RMS

    return threshold_value