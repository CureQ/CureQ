# CureQ

This is the repository of the CureQ consortium.<br>
This repository contains a library with functions for analyzing MEA files.<br>
This repository is maintained by the Amsterdam University of Applied Sciences (AUMC).<br>
More information: https://cureq.nl/

___

## Install the library
Open a terminal and navigate to your home folder.

#### Install library with pip
Install the MEA analyzer with the following command when you are using Pip:
```shell
pip install CureQ 
```

#### Install library with conda
Install the MEA analyzer with the following command when you are using Conda:
```shell
conda install CureQ::CureQ
```

---

## Library usage
Now you can try the CureQ library functions in your Python environment. <br>
Import the function you need, call this function and watch how the pipeline analyzes your MEA file!

#### Example for analyzing MEA electrodes
```python
from CureQ.mea import analyse_electrode          # Library function for analyzing electrodes

file_path = 'path/to/your/mea_file.h5'           # Path to your MEA file
hertz = 20000                                    # Sampling frequency of MEA system
electrodes = [109, 110]                          # Electrodes to be analyzed
analyse_electrode(file_path, electrodes, hertz)  # Analyzes specified electrodes in the MEA file
```

#### Example for analyzing a MEA well
```python
from CureQ.mea import analyse_well               # Library function for analyzing wells

file_path = 'path/to/your/mea_file.h5'           # Path to your MEA file
hertz = 20000                                    # Sampling frequency of MEA system
wells = [10]                                     # Wells to be analyzed
analyse_well(filename, wells, hertz)             # Analyzes specified wells in the MEA file
```

---

## Example visualisations of the MEA analysis pipeline

#### Spike detection
![Spike detection well 10 electrode 2](Example_visualisations/Spike_detection_well_10_electrode_2.png)

#### Inter-spike interval
![Inter-spike interval well 15 electrode 3](Example_visualisations/Inter-spike_interval_well_15_electrode_3.png)

#### Single burst detection
![Single burst detection well 15 electrode 3](Example_visualisations/Single_burst_detection_well_15_electrode_3.png)

#### Network burst detection
![Network burst detection well 10](Example_visualisations/Network_burst_detection_well_10.png)

<!--
**CureQ/CureQ** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
