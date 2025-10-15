MEAlytics calculates a large variety of features. Electrode features are first calculated seperately, and then averaged over the well. Below you can find an overview of all the features the package calculates.

It is possible to add your own custom features if you have knowledge of Python programming. Instructions for this can be found in ```_features.py``` in the package files.

Have a specific feature in mind that has not been added yet, and might benefit others? Please do not hesitate to contact us.

## Spike features

| Feature name | Description |
|-------------|------------|
| Spikes | The total amount of spikes recorded in a single electrode (spikes) |
| Mean Firing Rate | The average activity of the electrode, calculated as the average amount of spikes per second. (spikes/s) |
| Mean Inter-Spike interval | Mean inter-spike interval. The average amount of time between two successive spikes. (s) |
| Median Inter-Spike interval | The median of the inter-spike intervals. (s) |
| Ratio median ISI over mean ISI | The ratio of the median ISI over mean ISI. Calculated by dividing the median by the mean. |
| Inter-spike interval variance | The average of the square deviations from the mean from the inter-spike intervals. (s) |
| Coefficient of variation ISI | The coefficient of variation of the inter-spike interval. (s) |
| Partial Autocorrelaction Function ISI | The partial autocorrelation of lag 1 of the ISIs. This value is calculated using the “statsmodels” python library (Seabold & Perktold, 2010) |
| Mean Absolute Spike Amplitude | The average absolute amplitude of the spikes. Spike amplitude is measured from 0 to the absolute top of the spike |
| Median Absolute Spike Amplitude | The median absolute amplitude of the spikes |
| Coefficient of Variation Absolute Spike Amplitude | The coefficient of variation of the absolute spike amplitude |

## Burst features

| Feature name | Description |
|-------------|------------|
| Bursts | The amount of burst that are detected in a single electrode. (bursts) |
| Mean Burst Length | The average length of a burst in a single electrode. (s) |
| Burst Length Variance | The variance in the length of the bursts. (s) |
| Burst Length Coefficient of Variation | The coefficient of variation of the burst length. (s) |
| Mean Inter-Burst Interval | The average amount of time between two bursts. (s) |
| Variance Inter-Burst Interval | The variance of the inter-burst interval. (s) |
| Coefficient of Variation IBI | The coefficient of variation of the inter-burst interval. (s) |
| Partial Autocorrelation Function IBI | The partial autocorrelation of lag 1 of the inter burst interval. |
| Mean Intra-Burst Firing Rate | The average firing rate in a burst. (spikes/s) |
| Mean Spikes per Burst | The average amount of spikes per burst. (spikes) |
| Mean Absolute Deviation Spikes per Burst | The mean absolute deviation of the amount of spikes per burst. (spikes) |
| Isolated Spikes | The portion of spikes that are isolated (not part of a burst). (value from 0 to 1) |
| Mean Burst Rate | The average amount of bursts per second. (bursts/s) |

## Network Burst Features
Here, "Network Burst" means the entire network burst, including the outer edges. If a calculation is made using the network burst core, it will be explicitly stated.

| Feature name | Description |
|-------------|------------|
| Network Bursts | The total amount of network bursts in a well (network bursts) |
| Mean Network Burst Duration | The average duration of a network bursts. Calculated as the distance between the outer edges of the network burst. (s) |
| Mean Network Burst Core Duration | The average duration of the network burst core. (s) |
| Coefficient of Variation Network Burst Core Duration | The coefficient of variation of the network burst core durations. (s) |
| Network Inter-Burst Interval | The average time between the network bursts (s). This values is calculated as the average time between the outer edges of the network bursts. |
| Partial Autocorrelation Function NIBI | The partial autocorrelation of lag 1 of the network inter burst intervals |
| Network Burst to Network Burst Core ratio | The average ratio between the length of the network burst and network burst core. |
| NIBI Variance | The variation between the network interburst intervals (s) |
| Coefficient of Variation NIBI | The coefficient of variation between the network interburst intervals (s) |
| Mean Spikes per Network Burst | The average amount of spikes in a network burst |
| Network Burst Firing Rate | The average firing rate in the network bursts (spikes/s) |
| Network Burst ISI | The average inter spike interval in the network bursts (s) |
| Portion of Spikes in Network Bursts | The portion of spikes that are participating in network bursts. |
| Portion of Bursts in Network Bursts | The portion of bursts that are participating in network bursts. |
| Ratio Left Outer NB over NB Core | The average ratio of the left outer burst compared to the burst core |
| Ratio Right Outer NB over NB Core | The average ratio of the left outer burst compared to the burst core |
| Ratio Left Outer over Right Outer NB | The ratio of the left outer burst compared to the right outer burst |
| Participating Electrodes | The average amount of channels that are actively bursting during network bursts |