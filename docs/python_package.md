Besides the GUI, the MEA analysis tool can also be called as a python library, and has a few functions that can be of use to the user. Let’s walk through an example to fully analyse a MEA file.

### Analysing MEA file

Firstly, import the necessary functions:

```python
from CureQ.mea import analyse_wells, get_default_parameters
```

Next, we define some variables that we later need to pass to the function.

```python
fileadress='C:/mea_data/mea_experiment.h5'
sampling_rate=20000
electrode_amount=12
```

Then, we retrieve the dictionary containing the default parameters so we can alter the analysis. In this case we turn on multiprocessing to speed up the analysis.

```python
parameters = get_default_parameters()
parameters['use multiprocessing'] = True
```

Finally, pass all the arguments to the analyse_wells function to initiate the analysis. Because we turned on multiprocessing, we must use and `“if __name__ == ‘__main__’: “` guard here. Otherwise, the application will eventually create an infinite number of processes and eventually crash.

```python
if __name__ == '__main__':
    analyse_wells(
        fileadress=fileadress,
        sampling_rate=sampling_rate,
        electrode_amnt=electrode_amount,
        parameters=parameters
    )
```

In the end, it should look like this:

```python
from CureQ.mea import analyse_wells, get_default_parameters

fileadress='C:/mea_data/mea_experiment.h5'
sampling_rate=20000
electrode_amount=12

parameters = get_default_parameters()
parameters['use multiprocessing'] = True

if __name__ == '__main__':
    analyse_wells(fileadress=fileadress,
                  sampling_rate=sampling_rate,
                  electrode_amnt=electrode_amount,
                  parameters=parameters
                  )
```
