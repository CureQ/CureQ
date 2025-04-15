# Imports
import os
import threading
from functools import partial
import json
import copy
import math
import webbrowser
import sys
from pathlib import Path
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from importlib.metadata import version
import traceback

# External libraries
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 
import h5py
import customtkinter as ctk
from CTkToolTip import *
from CTkMessagebox import CTkMessagebox
from CTkColorPicker import *
import requests

# Package imports
from ..core._network_burst_detection import network_burst_detection
from ..core._plotting import well_electrodes_kde

class whole_well_view(ctk.CTkToplevel):
    """
    Allows the user to inspect the network burst detection of a single well.
    """
    def __init__(self, parent, folder, well):
        super().__init__(parent)
        self.title(f"Well: {well}")

        self.tab_frame=ctk.CTkTabview(self, anchor='nw')
        self.tab_frame.pack(fill='both', expand=True, pady=10, padx=10)

        self.grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Network Burst Detection")
        self.tab_frame.tab("Network Burst Detection").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Network Burst Detection").grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Well Activity")
        self.tab_frame.tab("Well Activity").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Well Activity").grid_rowconfigure(0, weight=1)

        # Set the icon with a little delay, otherwise it does not work
        try:
            self.after(250, lambda: self.iconbitmap(os.path.join(parent.icon_path)))
        except Exception as error:
            print(error)

        self.parameters=open(f"{folder}/parameters.json")
        self.parameters=json.load(self.parameters)
        self.parameters["output hdf file"] = os.path.join(folder, "output_values.h5")

        self.folder=folder
        self.well=well
        self.parent=parent

        """Network burst detection plot"""
        # Create a frame for the plots
        self.nbd_plot_frame=ctk.CTkFrame(master=self.tab_frame.tab("Network Burst Detection"))
        self.nbd_plot_frame.grid(row=0, column=0, sticky='nesw')
        self.nbd_plot_frame.grid_columnconfigure(0, weight=1)
        self.nbd_plot_frame.grid_rowconfigure(0, weight=1)

        # Create a frame for the settings
        nbd_settings_frame=ctk.CTkFrame(master=self.tab_frame.tab("Network Burst Detection"), fg_color=parent.gray_6)
        nbd_settings_frame.grid(row=1, column=0)

        # Network burst detection settings
        # NB settings
        burst_options_label=ctk.CTkLabel(master=nbd_settings_frame, text='Network Burst Detection Parameters', font=ctk.CTkFont(size=25)).grid(row=0, column=0, pady=10, padx=10, sticky='w', columnspan=2)
        min_channels_nb_label=ctk.CTkLabel(master=nbd_settings_frame, text="Min channels (%)").grid(row=1, column=0, sticky='w', padx=10, pady=10)
        self.min_channels_nb_entry=ctk.CTkEntry(master=nbd_settings_frame)
        self.min_channels_nb_entry.grid(row=1, column=1, sticky='w', padx=10, pady=10)
        
        self.th_method_nb_var = ctk.StringVar(value=nbd_settings_frame)
        self.nwthoptions_nb = ['Yen', 'Otsu', 'Li', 'Isodata', 'Mean', 'Minimum', 'Triangle']
        th_method_nb = ctk.CTkLabel(master=nbd_settings_frame, text="Thresholding method:")
        th_method_nb.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.th_method_dropdown_nb = ctk.CTkOptionMenu(nbd_settings_frame, variable=self.th_method_nb_var, values=self.nwthoptions_nb)
        self.th_method_dropdown_nb.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        nbd_kde_bandwidth_nb_label=ctk.CTkLabel(master=nbd_settings_frame, text="KDE Bandwidth:").grid(row=3, column=0, sticky='w', padx=10, pady=10)
        self.nbd_kde_bandwidth_nb_entry=ctk.CTkEntry(master=nbd_settings_frame)
        self.nbd_kde_bandwidth_nb_entry.grid(row=3, column=1, sticky='w', padx=10, pady=10)

        # Buttons
        nb_update_plot_button=ctk.CTkButton(master=nbd_settings_frame, text="Update plot", command=self.update_plot)
        nb_update_plot_button.grid(row=4, column=0, sticky='nesw', padx=10, pady=10)
        nb_plot_disclaimer = CTkToolTip(nb_update_plot_button, y_offset=-100, wraplength=400, message='These settings are for visualisation purposes only, they will not affect the current analysis outcomes, or further steps such as feature calculation. These options are solely here to show how they could alter the analysis.')

        nb_reset_button=ctk.CTkButton(master=nbd_settings_frame, text="Reset", command=self.reset)
        nb_reset_button.grid(row=4, column=1, sticky='nesw', padx=10, pady=10)

        # Create initial plot
        self.reset()

        '''Electrode activity'''
        self.electrode_activity_plot_frame=ctk.CTkFrame(master=self.tab_frame.tab("Well Activity"))
        self.electrode_activity_plot_frame.grid(row=0, column=0, sticky='nsew')
        self.electrode_activity_plot_frame.grid_columnconfigure(0, weight=1)
        self.electrode_activity_plot_frame.grid_rowconfigure(0, weight=1)
        electrode_activity_settings=ctk.CTkFrame(master=self.tab_frame.tab("Well Activity"), fg_color=parent.gray_6)
        electrode_activity_settings.grid(row=1, column=0)

        self.def_bw_value=0.1

        el_act_bw_label=ctk.CTkLabel(master=electrode_activity_settings, text="KDE bandwidth")
        el_act_bw_label.grid(row=0, column=0, sticky='w', padx=10, pady=10)
        self.el_act_bw_entry=ctk.CTkEntry(master=electrode_activity_settings)
        self.el_act_bw_entry.grid(row=0, column=1, sticky='nesw', pady=10, padx=10)
        self.el_act_bw_entry.insert(0, self.def_bw_value)

        # Buttons
        el_act_update_plot=ctk.CTkButton(master=electrode_activity_settings, text='Update plot', command=self.well_activity_update_plot)
        el_act_update_plot.grid(row=1, column=0, sticky='nesw', padx=10, pady=10)

        el_act_reset=ctk.CTkButton(master=electrode_activity_settings, text='Reset', command=self.reset_electrode_activity)
        el_act_reset.grid(row=1, column=1, sticky='nesw', padx=10, pady=10)

        # Create initial plot
        self.reset_electrode_activity()

    def plot_network_bursts(self, parameters):
        fig=network_burst_detection(wells=[self.well], parameters=parameters, plot_electrodes=True, savedata=False, save_figures=False)
        
        # Check which colorscheme we have to use
        axiscolour=self.parent.text_color
        bgcolor=self.parent.gray_4

        fig.set_facecolor(bgcolor)
        for ax in fig.axes:
            ax.set_facecolor(bgcolor)
            # Change the other colours
            ax.xaxis.label.set_color(axiscolour)
            ax.yaxis.label.set_color(axiscolour)
            for side in ['top', 'bottom', 'left', 'right']:
                ax.spines[side].set_color(axiscolour)
            ax.tick_params(axis='x', colors=axiscolour)
            ax.tick_params(axis='y', colors=axiscolour)
            ax.set_title(label=ax.get_title(),color=axiscolour)

        plot_canvas = FigureCanvasTkAgg(fig, master=self.nbd_plot_frame)  
        plot_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbarframe=ttk.Frame(master=self.nbd_plot_frame)
        toolbarframe.grid(row=1, column=0, sticky='s')
        toolbar = NavigationToolbar2Tk(plot_canvas, toolbarframe)
        toolbar.config(background=self.parent.primary_1)
        toolbar._message_label.config(background=self.parent.primary_1)
        for button in toolbar.winfo_children():
            button.config(background=self.parent.primary_1)
        toolbar.update()
        plot_canvas.draw()

    def default_values(self):
        self.th_method_nb_var.set(self.parameters["thresholding method"])
        self.min_channels_nb_entry.delete(0,END)
        self.min_channels_nb_entry.insert(0,self.parameters["min channels"])
        self.nbd_kde_bandwidth_nb_entry.delete(0, END)
        self.nbd_kde_bandwidth_nb_entry.insert(0, self.parameters["nbd kde bandwidth"])

    def update_plot(self):
        temp_parameters=copy.deepcopy(self.parameters)
        temp_parameters["min channels"]=float(self.min_channels_nb_entry.get())
        temp_parameters["thresholding method"]=str(self.th_method_nb_var.get())
        temp_parameters["nbd kde bandwidth"]=float(self.nbd_kde_bandwidth_nb_entry.get())
        
        # Update the output folder path, as this might have changed since the original analysis
        temp_parameters['output path']=self.folder

        self.plot_network_bursts(parameters=temp_parameters)

    def reset(self):
        self.default_values()
        self.update_plot()

    def plot_well_activity(self):
        fig=well_electrodes_kde(outputpath=self.folder, well=self.well, parameters=self.parameters, bandwidth=float(self.el_act_bw_entry.get()))
        
        # Check which colorscheme we have to use
        axiscolour="#586d97"
        bgcolor=self.parent.gray_4
        
        fig.set_facecolor(bgcolor)
        for ax in fig.axes:
            ax.set_facecolor(bgcolor)
            # Change the other colours
            ax.xaxis.label.set_color(axiscolour)
            ax.yaxis.label.set_color(axiscolour)
            for side in ['top', 'bottom', 'left', 'right']:
                ax.spines[side].set_color(axiscolour)
            ax.tick_params(axis='x', colors=axiscolour)
            ax.tick_params(axis='y', colors=axiscolour)
            ax.set_title(label=ax.get_title(),color=axiscolour)

        plot_canvas = FigureCanvasTkAgg(fig, master=self.electrode_activity_plot_frame)  
        plot_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbarframe=ttk.Frame(master=self.electrode_activity_plot_frame)
        toolbarframe.grid(row=1, column=0)
        toolbar = NavigationToolbar2Tk(plot_canvas, toolbarframe)
        toolbar.config(background=self.parent.primary_1)
        toolbar._message_label.config(background=self.parent.primary_1)
        for button in toolbar.winfo_children():
            button.config(background=self.parent.primary_1)
        toolbar.update()
        plot_canvas.draw()

    def well_activity_update_plot(self):
        self.plot_well_activity()

    def reset_electrode_activity(self):
        self.el_act_bw_entry.delete(0, END)
        self.el_act_bw_entry.insert(0, self.def_bw_value)
        self.well_activity_update_plot()