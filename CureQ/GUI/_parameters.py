# Imports
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from functools import partial
import os
import json
import traceback

# External imports
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from CTkToolTip import *

class parameter_frame(ctk.CTkFrame):
    """
    Allows the user to set the different parameters for the analysis.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent
        self.tooltipwraplength=parent.tooltipwraplength

        # Weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        """Filter parameters"""
        # Filter parameters frame
        filterparameters=ctk.CTkFrame(self)
        filterparameters.grid(row=0, column=0, padx=10, pady=10, sticky='nesw')

        filterparameters.grid_columnconfigure(0, weight=1)

        filterparameters_label=ctk.CTkLabel(filterparameters, text="Filter Parameters", font=ctk.CTkFont(size=25))
        filterparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=2)

        # Low cutoff
        lowcutofflabel=ctk.CTkLabel(master=filterparameters, text="Low cutoff:")
        lowcutofflabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        lowcutofftooltip = CTkToolTip(lowcutofflabel, message='Define the low cutoff value for the butterworth bandpass filter. Values should be given in hertz')
        lowcutoffinput=ctk.CTkEntry(master=filterparameters)
        lowcutoffinput.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        # High cutoff
        highcutofflabel=ctk.CTkLabel(master=filterparameters, text="High cutoff:")
        highcutofflabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        highcutofftooltip = CTkToolTip(highcutofflabel, message='Define the high cutoff value for the butterworth bandpass filter. Values should be given in hertz')
        highcutoffinput=ctk.CTkEntry(master=filterparameters)
        highcutoffinput.grid(row=2, column=1, padx=10, pady=10, sticky='e')

        # Filter order
        orderlabel=ctk.CTkLabel(master=filterparameters, text="Filter order:")
        orderlabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        ordertooltip = CTkToolTip(orderlabel, message='The filter order for the butterworth filter')
        orderinput=ctk.CTkEntry(master=filterparameters)
        orderinput.grid(row=3, column=1, padx=10, pady=10, sticky='e')

        """Spike detection parameters"""
        # Set up all the spike detection parameters
        spikedetectionparameters=ctk.CTkFrame(self)
        spikedetectionparameters.grid(row=0, column=1, padx=10, pady=10, sticky='nsew', rowspan=2)

        spikedetectionparameters.grid_columnconfigure(0, weight=1)
        spikedetectionparameters.grid_rowconfigure(1, weight=1)
        spikedetectionparameters.grid_rowconfigure(2, weight=1)

        filterparameters_label=ctk.CTkLabel(spikedetectionparameters, text="Spike Detection Parameters", font=ctk.CTkFont(size=25))
        filterparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        # Threshold parameters
        thresholdparameters=ctk.CTkFrame(spikedetectionparameters)
        thresholdparameters.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        thresholdparameters.grid_columnconfigure(0, weight=1)

        thresholdparameters_label=ctk.CTkLabel(thresholdparameters, text="Threshold Parameters", font=ctk.CTkFont(size=15))
        thresholdparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=2)

        # Threshold portion
        thresholdportionlabel=ctk.CTkLabel(master=thresholdparameters, text="Threshold portion:")
        thresholdportionlabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        thresholdportiontooltip = CTkToolTip(thresholdportionlabel, message='Define the portion of the electrode data that is used for determining the threshold. A higher values will give a better estimate of the background noise, but will take longer to compute Ranges from 0 to 1.', wraplength=self.tooltipwraplength)
        thresholdportioninput=ctk.CTkEntry(master=thresholdparameters)
        thresholdportioninput.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # stdevmultiplier
        stdevmultiplierlabel=ctk.CTkLabel(master=thresholdparameters, text="Standard Deviation Multiplier:")
        stdevmultiplierlabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        stdevmultipliertooltip = CTkToolTip(stdevmultiplierlabel, message='Define when beyond which point values are seen as outliers (spikes) when identifying spike-free noise. A higher value will identify more data as noise', wraplength=self.tooltipwraplength)
        stdevmultiplierinput=ctk.CTkEntry(master=thresholdparameters)
        stdevmultiplierinput.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # RMSmultiplier
        RMSmultiplierlabel=ctk.CTkLabel(master=thresholdparameters, text="RMS Multiplier:")
        RMSmultiplierlabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        RMSmultipliertooltip = CTkToolTip(RMSmultiplierlabel, 'Define the multiplication factor of the root mean square (RMS) of the background noise. A higher number will lead to a higher threshold', wraplength=self.tooltipwraplength)
        RMSmultiplierinput=ctk.CTkEntry(master=thresholdparameters)
        RMSmultiplierinput.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Spike validation parameters
        validationparameters=ctk.CTkFrame(spikedetectionparameters)
        validationparameters.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')

        validationparameters.grid_columnconfigure(0, weight=1)

        validationparameters_label=ctk.CTkLabel(validationparameters, text="Spike Validation Parameters", font=ctk.CTkFont(size=15))
        validationparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=2)

        # Refractory period
        refractoryperiodlabel=ctk.CTkLabel(master=validationparameters, text="Refractory Period:")
        refractoryperiodlabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        refractoryperiodtooltip = CTkToolTip(refractoryperiodlabel, message='Define the refractory period in the spike detection In this period after a spike, no other spike can be detected. Value should be given in seconds, so 1 ms = 0.001 s', wraplength=self.tooltipwraplength)
        refractoryperiodinput=ctk.CTkEntry(master=validationparameters)
        refractoryperiodinput.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        # Dropdown menu where the user selects the validation method
        def option_selected(choice):
            validation_method = choice
            if validation_method=='Noisebased':
                exittimeinput.configure(state="normal")
                maxheightinput.configure(state="normal")
                amplitudedropinput.configure(state="normal")
            else:
                exittimeinput.configure(state="disabled")
                maxheightinput.configure(state="disabled")
                amplitudedropinput.configure(state="disabled")
            
        
        options = ['Noisebased', 'none']    
        dropdown_var = ctk.StringVar(value=options[0])
        dropdownlabel = ctk.CTkLabel(master=validationparameters, text="Spike validation method:")
        dropdownlabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        dropdowntooltip = CTkToolTip(dropdownlabel, 'Select the spike validation method. \'Noisebased\' will perform spike validation using surrounding noise, \'none\' will not perform any spike validation.', wraplength=self.tooltipwraplength)
        dropdown_menu = ctk.CTkComboBox(validationparameters, variable=dropdown_var, values=options, command=option_selected)
        dropdown_menu.grid(row=2, column=1, padx=10, pady=10, sticky='nes')

        exittimelabel=ctk.CTkLabel(master=validationparameters, text="Exit time:")
        exittimelabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        exittimetooltip = CTkToolTip(exittimelabel, 'Define the time in which the signal must drop/rise a certain amplitude before/after a spike has been detected to be validated. Value should be given in seconds, so 1 ms is 0.001s', wraplength=self.tooltipwraplength)
        exittimeinput=ctk.CTkEntry(master=validationparameters)
        exittimeinput.grid(row=3, column=1, padx=10, pady=10, sticky='e')

        amplitudedroplabel=ctk.CTkLabel(master=validationparameters, text="Drop amplitude:")
        amplitudedroplabel.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        amplitudedroptooltip = CTkToolTip(amplitudedroplabel, 'Multiplied with the root mean square of the surrounding noise. This is the height the signal must drop/rise in amplitude to be validated.', wraplength=self.tooltipwraplength)
        amplitudedropinput=ctk.CTkEntry(master=validationparameters)
        amplitudedropinput.grid(row=4, column=1, padx=10, pady=10, sticky='e')

        maxheightlabel=ctk.CTkLabel(master=validationparameters, text="Max drop:")
        maxheightlabel.grid(row=5, column=0, padx=10, pady=10, sticky='w')
        maxheighttooltip = CTkToolTip(maxheightlabel, 'Multiplied with the threshold value of the electrode. The maximum height a spike can be required to drop in amplitude in the set timeframe', wraplength=self.tooltipwraplength)
        maxheightinput=ctk.CTkEntry(master=validationparameters)
        maxheightinput.grid(row=5, column=1, padx=10, pady=10, sticky='e')

        """Burst detection parameters"""
        # Set up all the burst detection parameters
        burstdetectionparameters=ctk.CTkFrame(self)
        burstdetectionparameters.grid(row=0, column=2, padx=10, pady=10, sticky='nesw')

        burstdetectionparameters_label=ctk.CTkLabel(burstdetectionparameters, text="Burst Detection Parameters", font=ctk.CTkFont(size=25))
        burstdetectionparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=4)
        burstdetectionparameters.grid_columnconfigure(0, weight=1)

        # Setup up the minimal amount of spikes for a burst
        minspikeslabel=ctk.CTkLabel(master=burstdetectionparameters, text="Minimal amount of spikes:")
        minspikeslabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        minspikestooltip = CTkToolTip(minspikeslabel, message='Define the minimal amount of spikes a burst should have before being considered as one.', wraplength=self.tooltipwraplength)
        minspikesinput=ctk.CTkEntry(master=burstdetectionparameters)
        minspikesinput.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # Setup up the default threshold
        defaultthlabel=ctk.CTkLabel(master=burstdetectionparameters, text="Default interval threshold:")
        defaultthlabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        defaultthtooltip = CTkToolTip(defaultthlabel, 'Define the default inter-spike interval threshold that is used for burst detection. Value should be given in miliseconds.', wraplength=self.tooltipwraplength)
        defaultthinput=ctk.CTkEntry(master=burstdetectionparameters)
        defaultthinput.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Setup up the max threshold
        maxisilabel=ctk.CTkLabel(master=burstdetectionparameters, text="Max interval threshold:")
        maxisilabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        maxisitooltip = CTkToolTip(maxisilabel, message='Define the maximum value the inter-spike interval threshold can be when 2 valid peaks have been detected in the ISI graph. Value should be given in miliseconds.', wraplength=self.tooltipwraplength)
        maxisiinput=ctk.CTkEntry(master=burstdetectionparameters)
        maxisiinput.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Setup the KDE bandwidth
        isikdebwlabel=ctk.CTkLabel(master=burstdetectionparameters, text="KDE bandwidth:")
        isikdebwlabel.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        isikdebwtooltip = CTkToolTip(isikdebwlabel, message='Define the bandwidth that is used when calculating the kernel density estimate of the inter-spike intervals.', wraplength=self.tooltipwraplength)
        isikdebwinput=ctk.CTkEntry(master=burstdetectionparameters)
        isikdebwinput.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        """Network burst detection parameters"""
        networkburstdetectionparameters=ctk.CTkFrame(self)
        networkburstdetectionparameters.grid(row=1, column=2, padx=10, pady=10, sticky='nesw')
        networkburstdetectionparameters_label=ctk.CTkLabel(networkburstdetectionparameters, text="Network Burst Detection Parameters", font=ctk.CTkFont(size=25))
        networkburstdetectionparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=2)
        networkburstdetectionparameters.grid_columnconfigure(0, weight=1)

        # Setup the minimum amount of channels participating
        minchannelslabel=ctk.CTkLabel(master=networkburstdetectionparameters, text="Min channels:")
        minchannelslabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        minchannelstooltip = CTkToolTip(minchannelslabel, 'Define the minimal percentage of channels that should be active in a network burst, values ranges from 0 to 1. For example, a value of 0.5 requires half of the channels to be actively bursting during a network burst.', wraplength=self.tooltipwraplength)
        minchannelsinput=ctk.CTkEntry(master=networkburstdetectionparameters)
        minchannelsinput.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # Setup the thresholding method
        nwthoptions = ['Yen', 'Otsu', 'Li', 'Isodata', 'Mean', 'Minimum', 'Triangle']
        networkth_var = ctk.StringVar(value=nwthoptions[0])
        nbd_dropdownlabel = ctk.CTkLabel(master=networkburstdetectionparameters, text="Thresholding method:")
        nbd_dropdownlabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        nbd_dropdownlabel = CTkToolTip(nbd_dropdownlabel, 'The application offers multiple methods to automatically calculate the network burst detection activity threshold. Methods are derived from the scikit-image filters library.', wraplength=self.tooltipwraplength)
        dropdown_menu = ctk.CTkComboBox(networkburstdetectionparameters, variable=networkth_var, values=nwthoptions)
        dropdown_menu.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Setup the network burst detection KDE bandwidth
        nbd_kde_bandwidth_label=ctk.CTkLabel(master=networkburstdetectionparameters, text="KDE Bandwidth:")
        nbd_kde_bandwidth_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        nbd_kde_bandwidth_tooltip = CTkToolTip(nbd_kde_bandwidth_label, 'Define the bandwidth value that should be used when creating the kernel density estimate for the network burst detection.', wraplength=self.tooltipwraplength)
        nbd_kde_bandwidth_input=ctk.CTkEntry(master=networkburstdetectionparameters)
        nbd_kde_bandwidth_input.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        """Other parameters"""
        otherparameters=ctk.CTkFrame(self)
        otherparameters.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        otherparameters_label=ctk.CTkLabel(otherparameters, text="Other Parameters", font=ctk.CTkFont(size=25))
        otherparameters_label.grid(row=0, column=0, padx=10, pady=10, sticky='w', columnspan=2)

        def removeinactivefunc():
            if removeinactivevar.get():
                activitythinput.configure(state='normal')
            else:
                activitythinput.configure(state='disabled')

        # Use multiprocessing
        multiprocessinglabel=ctk.CTkLabel(otherparameters, text="Use multiprocessing:")
        multiprocessinglabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        multiprocessingtooltip = CTkToolTip(multiprocessinglabel, message='Using multiprocessing means the electrodes will be analyzed in parallel, generally speeding up the analysis. Multiprocessing might not work properly if the device you\'re using does not have sufficient RAM/CPU-cores', wraplength=self.tooltipwraplength)
        multiprocessingvar=ctk.BooleanVar()
        multiprocessinginput=ctk.CTkCheckBox(otherparameters, onvalue=True, offvalue=False, variable=multiprocessingvar, text='')
        multiprocessinginput.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # Select synchronicity method
        sync_method_label = ctk.CTkLabel(master=otherparameters, text="Synchronicity method:")
        sync_method_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        sync_method_tooltip = CTkToolTip(sync_method_label, message="Choose the method to compute synchronicity between spike trains.", wraplength=self.tooltipwraplength)

        sync_options = ["ISI-distance", "Adaptive ISI-distance",  "SPIKE-distance", "Adaptive SPIKE-distance"]
        sync_method_var = ctk.StringVar(value="SPIKE-distance") 
        sync_method_dropdown = ctk.CTkComboBox(master=otherparameters, variable=sync_method_var, values=sync_options)
        sync_method_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Remove inactive electrodes
        removeinactivelabel=ctk.CTkLabel(otherparameters, text="Remove inactive electrodes:")
        removeinactivelabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        removeinactivetooltip = CTkToolTip(removeinactivelabel, message='Remove inactive electrodes from the spike, burst and network burst feature calculations.', wraplength=self.tooltipwraplength)
        removeinactivevar=ctk.IntVar()
        removeinactiveinput=ctk.CTkCheckBox(otherparameters, onvalue=True, offvalue=False, variable=removeinactivevar, command=removeinactivefunc, text='')
        removeinactiveinput.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Setup the activity threshold
        activitythlabel=ctk.CTkLabel(master=otherparameters, text="Activity threshold:")
        activitythlabel.grid(row=4, column=0, padx=10, pady=10, sticky='w')
        activitythtooltip = CTkToolTip(activitythlabel, message='Define the minimal activity a channel must have, to be used in calculating features. Value should be given in hertz, so a value of 0.1 would mean any channel with less that 1 spike per 10 seconds will be removed', wraplength=self.tooltipwraplength)
        activitythinput=ctk.CTkEntry(otherparameters)
        activitythinput.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        """Buttons and functions for saving, resetting and importing parameters"""
        def set_parameters(parameters):

            # Make sure every entry is set to 'normal'
            dropdown_var.set("Noisebased")
            removeinactivevar.set(True)
            removeinactivefunc()
            option_selected(dropdown_var.get())

            lowcutoffinput.delete(0, END)
            lowcutoffinput.insert(0, parameters["low cutoff"])
            highcutoffinput.delete(0, END)
            highcutoffinput.insert(0, parameters["high cutoff"])
            orderinput.delete(0, END)
            orderinput.insert(0, parameters["order"])
            thresholdportioninput.delete(0, END)
            thresholdportioninput.insert(0, parameters["threshold portion"])
            stdevmultiplierinput.delete(0, END)
            stdevmultiplierinput.insert(0, parameters["standard deviation multiplier"])
            RMSmultiplierinput.delete(0, END)
            RMSmultiplierinput.insert(0, parameters["rms multiplier"])
            refractoryperiodinput.delete(0, END)
            refractoryperiodinput.insert(0, parameters["refractory period"])
            dropdown_var.set(parameters["spike validation method"])
            exittimeinput.delete(0, END)
            exittimeinput.insert(0, parameters["exit time"])
            amplitudedropinput.delete(0, END)
            amplitudedropinput.insert(0, parameters["drop amplitude"])
            maxheightinput.delete(0, END)
            maxheightinput.insert(0, parameters["max drop"])
            minspikesinput.delete(0, END)
            minspikesinput.insert(0, parameters["minimal amount of spikes"])
            defaultthinput.delete(0, END)
            defaultthinput.insert(0, parameters["default interval threshold"])
            maxisiinput.delete(0, END)
            maxisiinput.insert(0, parameters["max interval threshold"])
            isikdebwinput.delete(0, END)
            isikdebwinput.insert(0, parameters["burst detection kde bandwidth"])
            minchannelsinput.delete(0, END)
            minchannelsinput.insert(0, parameters["min channels"])
            networkth_var.set(parameters["thresholding method"])
            removeinactivevar.set(bool(parameters["remove inactive electrodes"]))
            activitythinput.delete(0, END)
            activitythinput.insert(0, parameters["activity threshold"])
            multiprocessingvar.set(bool(parameters["use multiprocessing"]))
            sync_method_var.set(parameters.get("synchronicity method", "SPIKE-distance")) #SPIKE-distance 
            nbd_kde_bandwidth_input.delete(0, END)
            nbd_kde_bandwidth_input.insert(0, parameters["nbd kde bandwidth"])

            # Update other parameter availability
            removeinactivefunc()
            option_selected(dropdown_var.get())

        def import_parameters():
            parametersfile = filedialog.askopenfilename(filetypes=[("Parameter file", "*.json")])
            if parametersfile == '':
                return
            parameters=json.load(open(parametersfile))
            
            set_parameters(parameters)

        def save_parameters():
            try:
                self.parent.parameters['low cutoff']=int(lowcutoffinput.get())
                self.parent.parameters['high cutoff']=int(highcutoffinput.get())
                self.parent.parameters['order']=int(orderinput.get())
                self.parent.parameters['refractory period']=float(refractoryperiodinput.get())
                self.parent.parameters['exit time']=float(exittimeinput.get())
                self.parent.parameters['burst detection kde bandwidth']=float(isikdebwinput.get())
                self.parent.parameters['minimal amount of spikes']=int(minspikesinput.get())
                self.parent.parameters['max interval threshold']=float(maxisiinput.get())
                self.parent.parameters['default interval threshold']=float(defaultthinput.get())
                self.parent.parameters['max drop']=float(maxheightinput.get())
                self.parent.parameters['drop amplitude']=float(amplitudedropinput.get())
                self.parent.parameters['standard deviation multiplier']=float(stdevmultiplierinput.get())
                self.parent.parameters['rms multiplier']=float(RMSmultiplierinput.get())
                self.parent.parameters['min channels']=float(minchannelsinput.get())
                self.parent.parameters['thresholding method']=networkth_var.get()
                self.parent.parameters['nbd kde bandwidth']=float(nbd_kde_bandwidth_input.get())
                self.parent.parameters['remove inactive electrodes']=bool(removeinactivevar.get())
                self.parent.parameters['activity threshold']=float(activitythinput.get())
                self.parent.parameters['threshold portion']=float(thresholdportioninput.get())
                self.parent.parameters['spike validation method']=str(dropdown_var.get())
                self.parent.parameters['use multiprocessing']=bool(multiprocessingvar.get())
                self.parent.parameters['synchronicity method'] = str(sync_method_var.get())
                self.parent.show_frame(self.parent.home_frame)

            except Exception as error:
                traceback.print_exc()
                CTkMessagebox(title="Error",
                              message=f'Certain parameters could not be converted to the correct datatype (e.g. int or float). Please check if every parameter has the correct values\n\n{error}',
                              icon="cancel",
                              wraplength=400)
        
        set_parameters(self.parent.parameters)

        def set_default_parameters():
            set_parameters(self.parent.default_parameters)

        default_parameters=ctk.CTkButton(master=self, text="Restore default parameters", command=set_default_parameters)
        default_parameters.grid(row=3, column=1, padx=10, pady=10, sticky='nsew')

        import_parameters_button=ctk.CTkButton(master=self, text="Import parameters", command=import_parameters)
        import_parameters_button.grid(row=3, column=2, padx=10, pady=10, sticky='nsew')

        save_parameters_button=ctk.CTkButton(master=self, text="Save parameters and return", command=save_parameters)
        save_parameters_button.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')