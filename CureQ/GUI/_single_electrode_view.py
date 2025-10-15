# Imports
import os
from functools import partial
import json
import copy
from tkinter import *

# External libraries
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 
import h5py
import customtkinter as ctk
from CTkToolTip import *

# Package imports
from ..core._bandpass import butter_bandpass_filter
from ..core._threshold import fast_threshold
from ..core._spike_validation import spike_validation
from ..core._burst_detection import burst_detection

class single_electrode_view(ctk.CTkToplevel):
    """
    Allows the user to inpect the spike and burst detection on a single electrode.
    """
    def __init__(self, parent, folder, rawfile, well, electrode):
        super().__init__(parent)
        self.title(f"Well: {well}, Electrode: {electrode}")

        self.tab_frame=ctk.CTkTabview(self, anchor='nw')
        self.tab_frame.pack(fill='both', expand=True, pady=10, padx=10)

        self.grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Spike Detection")
        self.tab_frame.tab("Spike Detection").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Spike Detection").grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Burst Detection")
        self.tab_frame.tab("Burst Detection").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Burst Detection").grid_rowconfigure(0, weight=1)

        # Set the icon with a little delay, otherwise it does not work
        try:
            self.after(250, lambda: self.iconbitmap(os.path.join(parent.icon_path)))
        except Exception as error:
            print(error)

        self.parameters=open(f"{folder}/parameters.json")
        self.parameters=json.load(self.parameters)
        self.parameters["output hdf file"] = os.path.join(folder, "output_values.h5")
        self.electrode_nr=(well-1)*self.parameters["electrode amount"]+electrode-1

        self.rawfile=rawfile
        self.folder=folder
        
        """Spike detection"""
        # Create a frame for the plots
        self.electrode_plot_frame=ctk.CTkFrame(master=self.tab_frame.tab("Spike Detection"))
        self.electrode_plot_frame.grid(row=0, column=0, sticky='nesw')
        self.electrode_plot_frame.grid_columnconfigure(0, weight=1)
        self.electrode_plot_frame.grid_rowconfigure(0, weight=1)

        # Create a frame for the settings
        electrode_settings_frame=ctk.CTkFrame(master=self.tab_frame.tab("Spike Detection"))
        electrode_settings_frame.grid(row=1, column=0)
        electrode_settings_frame.grid_columnconfigure(0, weight=1)
        electrode_settings_frame.grid_columnconfigure(1, weight=1)
        electrode_settings_frame.grid_columnconfigure(2, weight=1)

        self.tab_frame.tab("Spike Detection").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Spike Detection").grid_rowconfigure(0, weight=1)

        # Bandpass options
        bp_options_ew_frame = ctk.CTkFrame(master=electrode_settings_frame)
        bp_options_ew_frame.grid(row=0, column=0, pady=10, padx=10)
        bandpass_options_label=ctk.CTkLabel(master=bp_options_ew_frame, text='Bandpass Parameters', font=ctk.CTkFont(size=25)).grid(row=0, column=0, pady=10, padx=10, sticky='w', columnspan=2)

        lowcut_label=ctk.CTkLabel(master=bp_options_ew_frame, text='Low cutoff').grid(row=1, column=0, pady=10, padx=10, sticky='w')
        self.lowcut_ew_entry=ctk.CTkEntry(master=bp_options_ew_frame)
        self.lowcut_ew_entry.grid(row=1, column=1, pady=10, padx=10, sticky='w')

        highcut_label=ctk.CTkLabel(master=bp_options_ew_frame, text='High cutoff').grid(row=2, column=0, sticky='w', pady=10, padx=10)
        self.highcut_ew_entry=ctk.CTkEntry(master=bp_options_ew_frame)
        self.highcut_ew_entry.grid(row=2, column=1, pady=10, padx=10, sticky='w')

        order_label=ctk.CTkLabel(master=bp_options_ew_frame, text='Order').grid(row=3, column=0, sticky='w', pady=10, padx=10)
        self.order_ew_entry=ctk.CTkEntry(master=bp_options_ew_frame)
        self.order_ew_entry.grid(row=3, column=1, pady=10, padx=10, sticky='w')

        # Threshold options
        th_options_ew_frame = ctk.CTkFrame(master=electrode_settings_frame)
        th_options_ew_frame.grid(row=0, column=1, pady=10, padx=10)
        threshold_options_label=ctk.CTkLabel(master=th_options_ew_frame, text='Threshold Parameters', font=ctk.CTkFont(size=25)).grid(row=0, column=0, pady=10, padx=10, sticky='w', columnspan=2)
        
        stdevmultiplier_label=ctk.CTkLabel(master=th_options_ew_frame, text='Standard deviation multiplier').grid(row=1, column=0, pady=10, padx=10, sticky='w')
        self.stdevmultiplier_ew_entry=ctk.CTkEntry(master=th_options_ew_frame)
        self.stdevmultiplier_ew_entry.grid(row=1, column=1, pady=10, padx=10, sticky='w')

        RMSmultiplier_label=ctk.CTkLabel(master=th_options_ew_frame, text='RMS multiplier').grid(row=2, column=0, sticky='w', pady=10, padx=10)
        self.RMSmultiplier_ew_entry=ctk.CTkEntry(master=th_options_ew_frame)
        self.RMSmultiplier_ew_entry.grid(row=2, column=1, pady=10, padx=10, sticky='w')

        thpn_label=ctk.CTkLabel(master=th_options_ew_frame, text='Threshold portion').grid(row=3, column=0, sticky='w', pady=10, padx=10)
        self.thpn_ew_entry=ctk.CTkEntry(master=th_options_ew_frame)
        self.thpn_ew_entry.grid(row=3, column=1, pady=10, padx=10, sticky='w')

        # Spike validation options
        val_options_ew_frame = ctk.CTkFrame(master=electrode_settings_frame)
        val_options_ew_frame.grid(row=0, column=2, pady=10, padx=10)
        spike_val_options_label=ctk.CTkLabel(master=val_options_ew_frame, text='Spike Detection Parameters', font=ctk.CTkFont(size=25)).grid(row=0, column=0, pady=10, padx=10, sticky='w', columnspan=4)

        def ew_option_selected(event):
            self.set_states()
                
        validation_options = ['Noisebased', 'none']
        self.validation_method_var = ctk.StringVar(value=validation_options[0])
        validation_method_label = ctk.CTkLabel(master=val_options_ew_frame, text="Spike validation method:")
        validation_method_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.validation_method_entry = ctk.CTkOptionMenu(val_options_ew_frame, variable=self.validation_method_var, values=validation_options, command=ew_option_selected)
        self.validation_method_entry.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        rfpd_label=ctk.CTkLabel(master=val_options_ew_frame, text='Refractory period').grid(row=2, column=0, pady=10, padx=10, sticky='w')
        self.rfpd_ew_entry=ctk.CTkEntry(master=val_options_ew_frame)
        self.rfpd_ew_entry.grid(row=2, column=1, pady=10, padx=10, sticky='w')

        exittime_label=ctk.CTkLabel(master=val_options_ew_frame, text='Exit time').grid(row=3, column=0, pady=10, padx=10, sticky='w')
        self.exittime_ew_entry=ctk.CTkEntry(master=val_options_ew_frame)
        self.exittime_ew_entry.grid(row=3, column=1, pady=10, padx=10, sticky='w')

        dropamplitude_label=ctk.CTkLabel(master=val_options_ew_frame, text='Drop amplitude').grid(row=1, column=2, pady=10, padx=10, sticky='w')
        self.dropamplitude_ew_entry=ctk.CTkEntry(master=val_options_ew_frame)
        self.dropamplitude_ew_entry.grid(row=1, column=3, pady=10, padx=10, sticky='w')

        maxdrop_label=ctk.CTkLabel(master=val_options_ew_frame, text='Max drop').grid(row=2, column=2, pady=10, padx=10, sticky='w')
        self.maxdrop_ew_entry=ctk.CTkEntry(master=val_options_ew_frame)
        self.maxdrop_ew_entry.grid(row=2, column=3, pady=10, padx=10, sticky='w')

        plot_rectangle_label=ctk.CTkLabel(master=val_options_ew_frame, text='Plot validation')
        plot_rectangle_label.grid(row=3, column=2, pady=10, padx=10, sticky='w')
        plot_rectangle_tooltip = CTkToolTip(plot_rectangle_label, message='Display the rectangles that have been used to validate the spikes. Warning: Plotting these is computationally expensive and might take a while', wraplength=parent.tooltipwraplength)
        self.plot_rectangle=ctk.BooleanVar(value=False)
        self.plot_rectangle.set(False)
        self.plot_rectangle_ew_entry=ctk.CTkCheckBox(master=val_options_ew_frame, variable=self.plot_rectangle, text='')
        self.plot_rectangle_ew_entry.grid(row=3, column=3, pady=10, padx=10, sticky='w')

        # Set values and create initial plot
        self.reset(parent)

        # Buttons
        update_plot_button=ctk.CTkButton(master=electrode_settings_frame, text='Update plot', command=partial(self.update_plot, parent))
        update_plot_button.grid(row=1, column=0, pady=10, padx=10, sticky='nesw')
        electrode_plot_disclaimer = CTkToolTip(update_plot_button, y_offset=-100, wraplength=400, message='These settings are for visualisation purposes only, they will not affect the current analysis outcomes, or further steps such as burst or network burst detection. These options are solely here to show how they could alter the analysis.')

        reset_button = ctk.CTkButton(master=electrode_settings_frame, text='Reset', command=partial(self.reset, parent))
        reset_button.grid(row=1, column=1, pady=10, padx=10, sticky='nesw')

        
        """Burst Detection"""
        # Create the initial burst plot
        self.burstplotsframe=ctk.CTkFrame(master=self.tab_frame.tab("Burst Detection"))
        self.burstplotsframe.grid(row=0, column=0, sticky='nesw')

        burstsettingsframe=ctk.CTkFrame(master=self.tab_frame.tab("Burst Detection"), fg_color=parent.gray_6)
        burstsettingsframe.grid(row=1, column=0, pady=10, padx=10)

        # Burst detection settings
        burst_options_label=ctk.CTkLabel(master=burstsettingsframe, text='Burst Detection Parameters', font=ctk.CTkFont(size=25)).grid(row=0, column=0, pady=10, padx=10, sticky='w', columnspan=4)
        minspikes_bw_label=ctk.CTkLabel(master=burstsettingsframe, text='Minimal amount of spikes').grid(row=1, column=0, pady=10, padx=10, sticky='w')
        self.minspikes_bw_entry=ctk.CTkEntry(master=burstsettingsframe)
        self.minspikes_bw_entry.grid(row=1, column=1, pady=10, padx=10, sticky='w')
        def_iv_bw_label=ctk.CTkLabel(master=burstsettingsframe, text='Default interval threshold').grid(row=2, column=0, pady=10, padx=10, sticky='w')
        self.def_iv_bw_entry=ctk.CTkEntry(master=burstsettingsframe)
        self.def_iv_bw_entry.grid(row=2, column=1, pady=10, padx=10, sticky='w')
        max_iv_bw_label=ctk.CTkLabel(master=burstsettingsframe, text='Max interval threshold').grid(row=1, column=2, pady=10, padx=10, sticky='w')
        self.max_iv_bw_entry=ctk.CTkEntry(master=burstsettingsframe)
        self.max_iv_bw_entry.grid(row=1, column=3, pady=10, padx=10, sticky='w')
        kde_bw_bw_label=ctk.CTkLabel(master=burstsettingsframe, text='KDE bandwidth').grid(row=2, column=2, pady=10, padx=10, sticky='w')
        self.kde_bw_bw_entry=ctk.CTkEntry(master=burstsettingsframe)
        self.kde_bw_bw_entry.grid(row=2, column=3, pady=10, padx=10, sticky='w')

        # Burst buttons
        update_burst_plot_button=ctk.CTkButton(master=burstsettingsframe, text='Update plot', command=partial(self.update_burst_plot, parent))
        update_burst_plot_button.grid(row=3, column=0, pady=10, padx=10, sticky='nesw', columnspan=2)
        burst_plot_disclaimer = CTkToolTip(update_burst_plot_button, y_offset=-100, wraplength=400, message='These settings are for visualisation purposes only, they will not affect the current analysis outcomes, or further steps such as network burst detection. These options are solely here to show how they could alter the analysis.')

        reset_burst_button = ctk.CTkButton(master=burstsettingsframe, text='Reset', command=partial(self.burst_reset, parent))
        reset_burst_button.grid(row=3, column=2, pady=10, padx=10, sticky='nesw', columnspan=2)

        # Create first plot
        self.burst_reset(parent)

    def set_states(self):
        validation_method = self.validation_method_var.get()
        if validation_method=='Noisebased':
            self.exittime_ew_entry.configure(state="normal")
            self.maxdrop_ew_entry.configure(state="normal")
            self.dropamplitude_ew_entry.configure(state="normal")
            self.plot_rectangle_ew_entry.configure(state="normal")
        else:
            self.exittime_ew_entry.configure(state="disabled")
            self.maxdrop_ew_entry.configure(state="disabled")
            self.dropamplitude_ew_entry.configure(state="disabled")
            self.plot_rectangle.set(False)
            self.plot_rectangle_ew_entry.configure(state="disabled")

    def plot_single_electrode(self, parent, parameters):
        with h5py.File(self.rawfile, 'r') as hdf_file:
            dataset=hdf_file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
            raw_data=dataset[self.electrode_nr]
        electrode_data=butter_bandpass_filter(raw_data, parameters)
        threshold=fast_threshold(electrode_data, parameters)
        fig=spike_validation(data=electrode_data, electrode=self.electrode_nr, threshold=threshold, parameters=parameters, plot_electrodes=True, savedata=False, plot_rectangles=self.plot_rectangle.get())
        
        # Check which colorscheme we have to use
        axiscolour=parent.text_color
        bgcolor=parent.gray_4

        # Set the plot background
        fig.set_facecolor(bgcolor)
        ax=fig.axes[0]
        ax.set_facecolor(bgcolor)

        # Change the other colours
        ax.xaxis.label.set_color(axiscolour)
        ax.yaxis.label.set_color(axiscolour)
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color(axiscolour)
        ax.tick_params(axis='x', colors=axiscolour)
        ax.tick_params(axis='y', colors=axiscolour)
        ax.set_title(label=ax.get_title(),color=axiscolour)

        plot_canvas = FigureCanvasTkAgg(fig, master=self.electrode_plot_frame)  
        plot_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbarframe=ctk.CTkFrame(master=self.electrode_plot_frame, fg_color=parent.primary_1)
        toolbarframe.grid(row=1, column=0, sticky='s')
        toolbar = NavigationToolbar2Tk(plot_canvas, toolbarframe)
        toolbar.config(background=parent.primary_1)
        toolbar._message_label.config(background=parent.primary_1)
        for button in toolbar.winfo_children():
            button.config(background=parent.primary_1)
        toolbar.update()
        plot_canvas.draw()

    def update_plot(self, parent):
        # Get the new parameters from the entry widgets
        # Create a temporary dict we will give to the plot electrode function
        temp_parameters=copy.deepcopy(self.parameters)
        temp_parameters['low cutoff']=int(self.lowcut_ew_entry.get())
        temp_parameters['high cutoff']=int(self.highcut_ew_entry.get())
        temp_parameters['order']=int(self.order_ew_entry.get())
        temp_parameters['standard deviation multiplier']=float(self.stdevmultiplier_ew_entry.get())
        temp_parameters['rms multiplier']=float(self.RMSmultiplier_ew_entry.get())
        temp_parameters['threshold portion']=float(self.thpn_ew_entry.get())
        temp_parameters['refractory period']=float(self.rfpd_ew_entry.get())
        if str(self.validation_method_var.get())=='none':
            temp_parameters['drop amplitude']=0
        else:
            temp_parameters['exit time']=float(self.exittime_ew_entry.get())
            temp_parameters['drop amplitude']=float(self.dropamplitude_ew_entry.get())
            temp_parameters['max drop']=float(self.maxdrop_ew_entry.get())
        
        # Update the output folder path, as this might have changed since the original analysis
        temp_parameters['output path']=self.folder

        # Plot the electrode with the new parameters
        self.plot_single_electrode(parent, temp_parameters)

    def default_values(self):
        # Bandpass
        self.lowcut_ew_entry.delete(0,END)
        self.lowcut_ew_entry.insert(0,self.parameters["low cutoff"])
        self.highcut_ew_entry.delete(0,END)
        self.highcut_ew_entry.insert(0,self.parameters["high cutoff"])
        self.order_ew_entry.delete(0,END)
        self.order_ew_entry.insert(0,self.parameters["order"])
        # Threshold
        self.stdevmultiplier_ew_entry.delete(0,END)
        self.stdevmultiplier_ew_entry.insert(0,self.parameters["standard deviation multiplier"])
        self.RMSmultiplier_ew_entry.delete(0,END)
        self.RMSmultiplier_ew_entry.insert(0,self.parameters["rms multiplier"])
        self.thpn_ew_entry.delete(0,END)
        self.thpn_ew_entry.insert(0,self.parameters["threshold portion"])
        # Spike validation
        self.rfpd_ew_entry.delete(0,END)
        self.rfpd_ew_entry.insert(0,self.parameters["refractory period"])
        self.exittime_ew_entry.delete(0,END)
        self.exittime_ew_entry.insert(0,self.parameters["exit time"])
        self.dropamplitude_ew_entry.delete(0,END)
        self.dropamplitude_ew_entry.insert(0,self.parameters["drop amplitude"])
        self.maxdrop_ew_entry.delete(0,END)
        self.maxdrop_ew_entry.insert(0,self.parameters["max drop"])
        self.plot_rectangle.set(False)
        self.validation_method_var.set(self.parameters['spike validation method'])
        self.set_states()
    
    def reset(self, parent):
        # First, enable all the possibly disabled entries so we can alter the values
        self.exittime_ew_entry.configure(state="normal")
        self.maxdrop_ew_entry.configure(state="normal")
        self.dropamplitude_ew_entry.configure(state="normal")
        self.plot_rectangle_ew_entry.configure(state="normal")
        # Insert the new values
        self.default_values()
        # Update the availability of certain entries
        self.set_states()
        # Update the plot
        self.update_plot(parent)

    def plot_burst_detection(self, parent, parameters):
        with h5py.File(self.rawfile, 'r') as hdf_file:
            dataset=hdf_file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
            raw_data=dataset[self.electrode_nr]
        electrode_data=butter_bandpass_filter(raw_data, parameters)
        KDE_fig, burst_fig = burst_detection(data=electrode_data, electrode=self.electrode_nr, parameters=parameters, plot_electrodes=True, savedata=False)
        
        # Check which colorscheme we have to use
        axiscolour=parent.text_color
        bgcolor=parent.gray_4

        for fig in [KDE_fig, burst_fig]:
            fig.set_facecolor(bgcolor)
            ax=fig.axes[0]
            ax.set_facecolor(bgcolor)
            # Change the other colours
            ax.xaxis.label.set_color(axiscolour)
            ax.yaxis.label.set_color(axiscolour)
            for side in ['top', 'bottom', 'left', 'right']:
                ax.spines[side].set_color(axiscolour)
            ax.tick_params(axis='x', colors=axiscolour)
            ax.tick_params(axis='y', colors=axiscolour)
            ax.set_title(label=ax.get_title(),color=axiscolour)
        
        # Plot the raw burst plot
        burst_canvas = FigureCanvasTkAgg(burst_fig, master=self.burstplotsframe)  
        burst_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbarframe=ctk.CTkFrame(master=self.burstplotsframe)
        toolbarframe.grid(row=1, column=0, sticky='s', columnspan=2)
        toolbar = NavigationToolbar2Tk(burst_canvas, toolbarframe)
        toolbar.config(background=parent.primary_1)
        toolbar._message_label.config(background=parent.primary_1)
        for button in toolbar.winfo_children():
            button.config(background=parent.primary_1)
        toolbar.update()
        burst_canvas.draw()

        # Plot the KDE plot
        KDE_canvas = FigureCanvasTkAgg(KDE_fig, master=self.burstplotsframe)  
        KDE_canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self.burstplotsframe.grid_columnconfigure(0, weight=3)
        self.burstplotsframe.grid_columnconfigure(1, weight=1)
        self.burstplotsframe.grid_rowconfigure(0, weight=1)
        KDE_canvas.draw()

    def update_burst_plot(self, parent):
        # Get the new parameters from the entry widgets
        temp_parameters=copy.deepcopy(self.parameters)
        temp_parameters["minimal amount of spikes"]=int(self.minspikes_bw_entry.get())
        temp_parameters["default interval threshold"]=float(self.def_iv_bw_entry.get())
        temp_parameters["max interval threshold"]=float(self.max_iv_bw_entry.get())
        temp_parameters["burst detection kde bandwidth"]=float(self.kde_bw_bw_entry.get())

        # Update the output folder path, as this might have changed since the original analysis
        temp_parameters['output path']=self.folder

        self.plot_burst_detection(parent, temp_parameters)

    def default_values_burst(self):
        # Reset the values to the ones in the JSON file
        self.minspikes_bw_entry.delete(0,END)
        self.minspikes_bw_entry.insert(0,self.parameters["minimal amount of spikes"])
        self.def_iv_bw_entry.delete(0,END)
        self.def_iv_bw_entry.insert(0,self.parameters["default interval threshold"])
        self.max_iv_bw_entry.delete(0,END)
        self.max_iv_bw_entry.insert(0,self.parameters["max interval threshold"])
        self.kde_bw_bw_entry.delete(0,END)
        self.kde_bw_bw_entry.insert(0,self.parameters["burst detection kde bandwidth"])

    def burst_reset(self, parent):
        self.default_values_burst()
        self.update_burst_plot(parent)