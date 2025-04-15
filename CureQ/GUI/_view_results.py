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

# GUI components
from ._single_electrode_view import single_electrode_view
from ._whole_well_view import whole_well_view

class select_folder_frame(ctk.CTkFrame):
    """
    Allows the user to select a dataset and outputfile to inspect the results.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.folder_path=''
        self.data_path=''

        datalocation=ctk.CTkFrame(self)
        datalocation.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        datalocation.grid_columnconfigure(1, weight=1)

        folderlabel=ctk.CTkLabel(master=datalocation, text="Folder:")
        folderlabel.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.btn_selectfolder=ctk.CTkButton(master=datalocation, text="Select a folder", command=self.openfolder)
        self.btn_selectfolder.grid(row=0, column=1, padx=10, pady=10, sticky='nesw')

        rawfilelabel=ctk.CTkLabel(master=datalocation, text="File:")
        rawfilelabel.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.btn_selectrawfile=ctk.CTkButton(master=datalocation, text="Select a file", command=self.openrawfile)
        self.btn_selectrawfile.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        view_results_button=ctk.CTkButton(self, text="View results", command= lambda: self.go_to_results(parent))
        view_results_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        return_button=ctk.CTkButton(self, text="Return", command= lambda: parent.show_frame(self.parent.home_frame))
        return_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

    def openfolder(self):
            resultsfolder = filedialog.askdirectory()
            if resultsfolder != '':
                self.btn_selectfolder.configure(text=resultsfolder)
                self.folder_path=resultsfolder
    
    def openrawfile(self):
            selectedfile = filedialog.askopenfilename(filetypes=[("MEA data", "*.h5")])
            if len(selectedfile)!=0:
                self.btn_selectrawfile.configure(text=os.path.split(selectedfile)[1])
                self.data_path=selectedfile

    # Check if the correct folder/file has been selected
    def go_to_results(self, parent):
        if self.folder_path == '' or self.data_path == '':
            CTkMessagebox(title="Error", message='Please select an output folder and raw datafile', icon="cancel", wraplength=400)
            return
        
        # Check for parameters json
        try:
            parameters=open(f"{self.folder_path}/parameters.json")
            parameters=json.load(parameters)
        except Exception as error:
            traceback.print_exc()
            CTkMessagebox(title='Error', message='Could not load in the results, please make sure you have selected the correct folder', icon='cancel', wraplength=400)
            return
        
        # Check the raw datafile
        try:
            with h5py.File(self.data_path, 'r') as hdf_file:
                datashape=hdf_file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].shape
        except Exception as error:
            traceback.print_exc()
            CTkMessagebox(title='Error', message='Could not load in the raw data, please make sure you have selected the correct file', icon='cancel', wraplength=400)
            return

        # If everything is good, go to the next frame
        parent.show_frame(view_results, folder=self.folder_path, rawfile=self.data_path)


class view_results(ctk.CTkFrame):
    """
    Allow the user to view the results of the MEA analysis
    """
    def __init__(self, parent, folder, rawfile):
        super().__init__(parent)

        self.folder=folder
        self.rawfile=rawfile
        self.parent=parent

        self.parent.title(f"MEAlytics - Version: {version('CureQ')} - {self.folder}")

        self.tab_frame=ctk.CTkTabview(self, anchor='nw')
        self.tab_frame.grid(column=0, row=0, sticky='nesw', pady=10, padx=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Single Electrode View")
        self.tab_frame.tab("Single Electrode View").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Single Electrode View").grid_rowconfigure(0, weight=1)

        self.tab_frame.add("Whole Well View")
        self.tab_frame.tab("Whole Well View").grid_columnconfigure(0, weight=1)
        self.tab_frame.tab("Whole Well View").grid_rowconfigure(0, weight=1)

        # sev = single electrode view
        # wwv = whole well view

        # Load files
        parameters=open(f"{folder}/parameters.json")
        parameters=json.load(parameters)
        with h5py.File(rawfile, 'r') as hdf_file:
            datashape=hdf_file["Data/Recording_0/AnalogStream/Stream_0/ChannelData"].shape
        well_amnt = datashape[0]/parameters["electrode amount"]

        """Single electrode view"""
        self.selected_well=1
        sev_well_button_frame=ctk.CTkFrame(self.tab_frame.tab("Single Electrode View"))
        sev_well_button_frame.grid(row=0, column=0, pady=10, padx=10, sticky='nesw')

        sev_electrode_button_frame=ctk.CTkFrame(self.tab_frame.tab("Single Electrode View"))
        sev_electrode_button_frame.grid(row=0, column=1, pady=10, padx=10, sticky='nesw')
        
        # Wellbuttons
        xwells, ywells = parent.calculate_well_grid(well_amnt)
        self.sev_wellbuttons=[]
        i=1

        for y in range(ywells):
            for x in range(xwells):
                well_btn=ctk.CTkButton(master=sev_well_button_frame, text=i, command=partial(self.set_selected_well, i), height=100, width=100, font=ctk.CTkFont(size=25))
                well_btn.grid(row=y, column=x, sticky='nesw')
                self.sev_wellbuttons.append(well_btn)
                i+=1

        # Electrode buttons
        electrode_amnt = parameters["electrode amount"]

        electrode_layout = parent.calculate_electrode_grid(electrode_amnt)

        i = 1
        electrodebuttons=[]
        for x in range(electrode_layout.shape[0]):
            for y in range(electrode_layout.shape[1]):
                if electrode_layout[x,y]:
                    electrode_btn=ctk.CTkButton(master=sev_electrode_button_frame, text=i, command=partial(self.open_sev_tab, i), height=100, width=100, font=ctk.CTkFont(size=25))
                    electrode_btn.grid(row=x, column=y, sticky='nesw')
                    electrodebuttons.append(electrode_btn)
                    i+=1

        """Whole well view"""
        wwv_well_button_frame=ctk.CTkFrame(self.tab_frame.tab("Whole Well View"))
        wwv_well_button_frame.grid(row=0, column=0, pady=10, padx=10, sticky='nesw')

        wwv_wellbuttons=[]
        i=1

        for y in range(ywells):
            for x in range(xwells):
                well_btn=ctk.CTkButton(master=wwv_well_button_frame, text=i, command=partial(self.open_wwv_tab, i), height=100, width=100, font=ctk.CTkFont(size=25))
                well_btn.grid(row=y, column=x)
                wwv_wellbuttons.append(well_btn)
                i+=1

        # Button to return to main menu
        return_to_main = ctk.CTkButton(master=self, text="Return to main menu", command=lambda: self.parent.show_frame(self.parent.home_frame), fg_color=parent.gray_1)
        return_to_main.grid(row=1, column=0, pady=10, padx=10)
    
    def set_selected_well(self, i):
        self.selected_well=i
        for j in range(len(self.sev_wellbuttons)):
            self.sev_wellbuttons[j].configure(fg_color=self.parent.theme["CTkButton"]["fg_color"][1])
        self.sev_wellbuttons[i-1].configure(fg_color=self.parent.theme["CTkButton"]["hover_color"][1])


    def open_sev_tab(self, electrode):
        single_electrode_view(self.parent, self.folder, self.rawfile, self.selected_well, electrode)

    def open_wwv_tab(self, well):
        whole_well_view(self.parent, self.folder, well)