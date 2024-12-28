from scipy.signal import butter, lfilter

'''Bandpass function'''
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, parameters):
    b, a = butter_bandpass(parameters['low cutoff'], parameters['high cutoff'], parameters['sampling rate'], order=parameters['order'])
    y = lfilter(b, a, data)
    return y