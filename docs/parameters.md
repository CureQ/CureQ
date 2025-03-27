---
layout: default
title: Parameters
permalink: /parameters
---

The MEAlytics package contains a wide range of parameters that can be used to alter the analysis pipeline. Described below are all the parameters, and how they affect the analysis pipeline.

The default parameter values are stored as a dictionary in the library, and can be accessed as follows:

```python
from CureQ.mea import get_default_parameters
print(get_default_parameters())
```

Default parameters are automatically loaded into the MEAlytics GUI

### Bandpass filter
#### Low cutoff
Low cutoff frequency for the Butterworth bandpass filter.<br>
Default value: 200 Hz

#### High cutoff
High cutoff frequency for the Butterworth bandpass filter.<br>
Default value: 3500 Hz

#### Order
The filter order used for the Butterworth bandpass filter.<br>
Default value: 2

### Threshold
#### Threshold portion
Portion of the data that is used to calculate the threshold value. Higher values can give a more accurate estimate of the background noise level, but takes longer to compute. <br>
Default value: 0.1

#### Standard deviation multiplier
Value that is multiplied with the standard deviation of a portion of the data to create a threshold that determines whether this portion of the data contains purely noise or potential spikes.<br>
Default value: 5

#### RMS multiplier
Value that is multiplied with the root mean square (RMS) of the background noise to determine the threshold for the spike detection.<br>
Default value: 5

### Spike detection
#### Refractory period
The refractory period of the spike. The time in which only one spike can be detected. Value should be given in seconds.<br>
Default value: 0.001

### Spike validation
#### Spike validation method
The spike validation method used to determine whether a threshold crossing should be considered a neuronal signal or discarded.<br>
Possible options: Noisebased, none.<br>
Default value: Noisebased

#### Exit time
The time the signal must drop/rise below a certain amplitude before/after it has reached its absolute peak.
Value should be given in seconds. <br>
Default value: 0.001

#### Drop amplitude
Multiplied with the surrounding background noise of the spike to determine the amplitude the signal must drop.<br>
Default value: 5

#### Max drop
Value multiplied with the threshold value of the electrode to determine the maximum amplitude the signal can be required to drop.<br>
Default value: 2

### Burst detection
#### Minimal amount of spikes
The smallest amount of spikes that can form a single channel burst.<br>
Default value: 5

#### Default interval threshold
The default inter-spike interval (ISI) threshold used for burst detection. Value should be given in milliseconds. <br>
Default value: 100

#### Max interval threshold
The maximum ISI threshold that can be used for burst detection. Values should be given in milliseconds.<br>
Default value: 1000

#### KDE bandwidth
The bandwidth of the Kernel Density Estimate (KDE) used for determining the burst detection parameters.<br>
Default value: 1

### Network burst detection
#### Min channels
The minimal amount of channels that should be participating in a network burst to consider it as one. Value should be given as a percentage.<br>
Default value: 0.5

#### Thresholding method
The method used to automatically calculate the threshold used for determining high-activity periods.<br>
Possible values: Yen, Otsu, Li, Isodata, Mean, Minimum, Triangle. See [scikit-image filters](https://scikit-image.org/docs/0.24.x/api/skimage.filters.html#).
<br>
Default value: Yen

#### KDE bandwidth
The bandwidth used for the kernel density estimate representing spike activity.<br>
Default value: 0.05
