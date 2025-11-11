#### This project is still in development.

# MEAlytics

This repository contains a python package for analyzing Micro-Electrode Array (MEA) experiments.<br>
This repository is maintained by the Amsterdam University of Applied Sciences (AUAS).<br>
MEAlytics offers high speed spike, burst and network burst detection followed by feature extraction.<br>

#### [For more information about the analysis or how to use the library, check out the MEAlytics Github pages](https://cureq.github.io/CureQ/).

## CureQ
This tool was created for the CureQ consortium.<br>
For more information about the CureQ project, visit https://cureq.nl/
___

## Install the library

1. **Download and install Python** (if not already installed) from [python.org](https://python.org/downloads/)
2. **Open a terminal or command prompt with installed Python library** <br> Navigate to your home folder.
   ```
   cd /path/to/home/folder
   ```
3. Make sure you downloaded **Python**!<br>
> Check this with ```python --version``` in your CMD. <br>
> <i>If you don't have Python installed yet, install it by typing ```python``` in your CMD.<br> 
This will automatically install pip.</i><br>
> Check this with ```pip --version``` in your CMD.<br>
> <i>If you don't have pip installed yet, install it by typing ```get-pip.py``` in your CMD.</i><br>
4. **Install library with pip**<br>
Install the MEA analyzer with the following command when you are using Pip:
```shell
pip install CureQ 
```

More elaborate and simple installation instructions can be found in the [User Guide](https://cureq.github.io/CureQ/installation.html).<br>

#### Please note that this project is still in development. Certain aspects of the package might change in the future.

---

## Library usage
Now you can try the CureQ library functions in your Python environment. <br>
Import the function you need, call this function and watch how the pipeline analyzes your MEA file!

#### Example for analyzing all MEA wells
```python
from CureQ.mea import analyse_wells, get_default_parameters

fileadress='C:/mea_data/mea_experiment.h5'
sampling_rate=20000
electrode_amount=12

# Get and edit parameters
parameters = get_default_parameters()
parameters['use multiprocessing'] = True

if __name__ == '__main__':
    analyse_wells(fileadress=fileadress,
                  sampling_rate=sampling_rate,
                  electrode_amnt=electrode_amount,
                  parameters=parameters
                  )
```

---

## MEA GUI
Addionally, the CureQ mea library contains a graphical user interface that can be used to perform the analysis. <br>
The GUI is also used to inspect the raw MEA data and the analysis results. Example visualisations can be found below. <br>
The GUI can be opened as follows:

#### Opening the GUI
```python
from CureQ.GUI.mea_analysis_tool import MEA_GUI

if __name__=="__main__":
    MEA_GUI()
```

#### Launch from command prompt
Additionally, he GUI can be launched from the command prompt. Simply open the command prompt, and enter “cureq”.
```shell
C:\Users>cureq
Successfully launched MEAlytics GUI
```
The output should look like this, and the GUI should appear on your screen.

#### Create shortcuts
This process can be simplified by creating shortcuts that in essence perform the same process. In the command prompt, enter “cureq –create-shortcut”.

```shell
C:\Users>cureq --create-shortcut
Desktop shortcut created at C:\Users\Desktop\CureQ.lnk
```
The output should look like this, and a shortcut should appear on your desktop:

---

## MEAlytics functionality

### Parameters
MEAlytics offers a wide range of parameters that can be used to alter the analysis! However, all parameters have default values that are backed by literature.

![Parameters](./Example_visualisations/parameters.png)

### MEA Analysis

MEAlytics is capable of using **multiprocessing** to make optimal use of all available resources, speeding up the analysis!

![](./Example_visualisations/process.png)

### Spike detection

After performing the analysis, the user can inspect the results using the GUI!<br>
The user can alter all parameters regarding spike, burst and network burst detection and immediately apply these changes to see how they alter the analysis. This allows you to quickly see the effect of parameter changes, without having to redo the entire analysis. <br>

![Spike detection](./Example_visualisations/spike_detection.png)

Additonally, the user can zoom in on the data to view the smaller timeframes.

![](./Example_visualisations/spike_detection_zoomed.png)

### Single channel burst detection

Burst detection is performed using the logISI method, meaning that the thresholds adapt to the input data!

![Burst detection](./Example_visualisations/burst_detection.png)

### Network burst detection

Network burst detection is performed by looking for high activity periods on multiple channels.

![Network burst detection](./Example_visualisations/network_burst_detection.png)

### Batch processing

Perform high-throughput analysis using the batch processing module!

![Batch processing](./Example_visualisations/batch_processing.png)

### Features

MEAlytics calculates over 40 descriptive well and electrode features and saves them in a csv file. These can then be read by other applications such as excel!

![Features](./Example_visualisations/features.png)

### Group comparison

You can combine the outcome features of multiple experiments to compare the differences between two groups using the plotting module!

![Group comparison](./Example_visualisations/group_comparison.png)

Compare two groups with each other and create visualisations for all features. Output is saved in pdf format.

![Boxplots](./Example_visualisations/boxplot.png)

Visualise the development of features over time by adding a prefix to your feature files!

![Features over time](./Example_visualisations/features_over_time.png)


---

## Run the ipywidgets UI (Jupyter)

An experimental ipywidgets-based UI is available for use in Jupyter notebooks or with Voilà.

- Install optional dependencies:

  - With pip: `pip install "CureQ[widgets]"`
  - Or add separately: `pip install ipywidgets voila`
  - In JupyterLab, also enable widgets if needed.

- Launch in a notebook:
  ```python
  from CureQ.ipywidgets_ui import CureQApp
  app = CureQApp()
  app.show()  # or display(app.widget())
  ```

Pages currently supported:

- Analyze file: run a single-file analysis with progress and abort.
- Parameters: view/edit/save analysis parameters.
- Batch processing: select a folder, choose .h5 files, and run sequentially with per-file progress.
- Compress/Rechunk: compress one file or all files in a folder (LZF/GZIP).
- Other pages are stubbed and will arrive in future updates.

Tip: For large files, prefer the "File path" field instead of uploading via the widget.

<!--
**CureQ/CureQ** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
