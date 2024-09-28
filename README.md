# CureQ

This is the repository of the CureQ consortium.<br>
This repository contains a library with functions for analyzing MEA files.<br>
This repository is maintained by the Amsterdam University of Applied Sciences (AUMC).<br>
For more information about the analysis or how to use the library, check out the "CureQ MEA-analysis library User Guide.pdf" file.<br>
For more information about the CureQ project, visit https://cureq.nl/

___

## Install the library

First, make sure you have installed a version of python on your machine, all python releases can be found here: https://www.python.org/downloads/ <br>
Secondly, make sure you have installed a package manager, like pip or conda.
Next, open a terminal and navigate to your home folder.

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

#### Example for analyzing all MEA wells
```python
from CureQ.mea import analyse_well               # Library function for analyzing wells

file_path = 'path/to/your/mea_file.h5'           # Path to your MEA file
hertz = 20000                                    # Sampling frequency of MEA system
electrodes = 12                                  # Electrode amount per well

# Analyzes all wells in the MEA file
if __name__=='__main__':
   analyse_well(fileadress=file_path, hertz=hertz, electrode_amnt=electrodes)
```

---

## Example visualisations of the MEA analysis pipeline

#### Spike detection
![Spike detection](https://github.com/CureQ/CureQ/blob/main/Example_visualisations/spike_detection.png)

#### Single burst detection
![Burst detection](https://github.com/CureQ/CureQ/blob/main/Example_visualisations/burst_detection.PNG)

#### Network burst detection
![Network burst detection](https://github.com/CureQ/CureQ/blob/main/Example_visualisations/network_burst_detection.PNG)

<!--
**CureQ/CureQ** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
