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
import time

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
from ..mea import analyse_wells

class process_file_frame(ctk.CTkFrame):
    """
    Allows the user to perform analysis on a single MEA experiment.
    """
    def __init__(self, parent):
        super().__init__(parent)  # Initialize the parent class

        self.parent=parent

        self.crashed=False
        self.progressfile=''

        self.grid_columnconfigure(0, weight=1)

        # Values
        self.selected_file=''

        self.select_file_button = ctk.CTkButton(master=self, text="Select a file", command=self.select_file)
        self.select_file_button.grid(row=0, column=0, pady=5, padx=5, sticky='nesw')

        parametersframe = ctk.CTkFrame(master=self)
        parametersframe.grid(row=1, column=0, padx=5, pady=5, sticky='nesw')

        parametersframe.grid_columnconfigure(0, weight=1)
        parametersframe.grid_columnconfigure(1, weight=1)

        sampling_rate_label = ctk.CTkLabel(master=parametersframe, text="Sampling Rate:")
        sampling_rate_label.grid(row=0, column=0, pady=10, padx=10, sticky='w')

        self.sampling_rate_entry = ctk.CTkEntry(master=parametersframe)
        self.sampling_rate_entry.grid(row=0, column=1, sticky='nesw', pady=10, padx=10)

        electrode_amnt_label = ctk.CTkLabel(master=parametersframe, text="Electrode amount:")
        electrode_amnt_label.grid(row=1, column=0, pady=10, padx=10, sticky='w')

        self.electrode_amnt_entry = ctk.CTkEntry(master=parametersframe)
        self.electrode_amnt_entry.grid(row=1, column=1, sticky='nesw', pady=10, padx=10)

        self.start_button = ctk.CTkButton(master=self, text="Start Analysis", command=lambda: threading.Thread(target=self.start_analysis).start())
        self.start_button.grid(row=2, column=0, pady=5, padx=5, sticky='nesw')

        self.return_button = ctk.CTkButton(master=self, text="Return to Main Menu", command=lambda: parent.show_frame(self.parent.home_frame), fg_color=parent.gray_1)
        self.return_button.grid(row=3, column=0, padx=5, pady=5, sticky='nesw')

    def select_file(self):
        selectedfile = filedialog.askopenfilename(filetypes=[("MEA data", "*.h5")])
        if len(selectedfile)!=0:
            self.selected_file=selectedfile
            self.select_file_button.configure(text=selectedfile)

    def call_library(self):
        try:
            electrode_amnt=int(self.electrode_amnt_entry.get())
            sampling_rate=int(self.sampling_rate_entry.get())
            analyse_wells(self.selected_file, electrode_amnt=electrode_amnt, sampling_rate=sampling_rate, parameters=self.parent.parameters)
        except Exception as error:
            CTkMessagebox(title="Error", message=f'Something went wrong during the analysis:\n{error}', icon="cancel", wraplength=400)
            traceback.print_exc()
            self.crashed=True
            self.start_button.configure(state='normal')
            self.return_button.configure(state='normal')

    def abort_analysis(self):
        np.save(self.progressfile, ["abort"])
        print("Aborting analysis, please wait...")

    def start_analysis(self):
        # Check things
        if self.selected_file=='':
            CTkMessagebox(title="Error", message='Please select a file', icon="cancel", wraplength=400)
            return
        try:
            sampling_rate=int(self.sampling_rate_entry.get())
            electrode_amnt=int(self.electrode_amnt_entry.get())
        except:
            CTkMessagebox(title="Error", message='Please fill in the sampling rate and electrode amount.', icon="cancel", wraplength=400)
            return

        self.start_button.configure(state='disabled')
        self.return_button.configure(state='disabled')

        # Create a popup for the progress
        popup=ctk.CTkToplevel(self)
        popup.title('Progress')

        try:
            popup.after(250, lambda: popup.iconbitmap(os.path.join(self.parent.icon_path)))
        except Exception as error:
            print(error)

        popup.grid_columnconfigure(0, weight=1)

        popup.protocol("WM_DELETE_WINDOW", lambda: self.abort_analysis())

        progress_label = ctk.CTkLabel(master=popup, text="The MEA data is being processed, please wait for the application to finish.")
        progress_label.grid(row=0, column=0, pady=10, padx=10, sticky='nesw')

        progressbar=ctk.CTkProgressBar(master=popup, orientation='horizontal', mode='determinate', progress_color="#239b56", width=400)
        progressbar.grid(row=1, column=0, pady=10, padx=10, sticky='nesw')
        progressbar.set(0)

        progress_info = ctk.CTkLabel(master=popup, text='')
        progress_info.grid(row=2, column=0, pady=10, padx=10, sticky='nesw')

        path=os.path.split(self.selected_file)[0]
        self.progressfile=f"{path}/progress.npy"

        # Remove potential pre-existing progressfile
        try: 
            os.remove(self.progressfile)
        except:
            pass

        # And keep track of the progress
        start=time.time()

        # Start the analysis in a new thread
        process=threading.Thread(target=self.call_library)
        process.start()

        while True:
            currenttime=time.time()
            elapsed=round(currenttime-start,1)
            try:
                progress=np.load(self.progressfile)
            except:
                progress=['starting']
            if self.crashed:
                popup.destroy()
                os.remove(self.progressfile)
                self.start_button.configure(state='normal')
                self.return_button.configure(state='normal')
                return
            if progress[0]=='done':
                break
            elif progress[0]=='starting':
                progress_info.configure(text=f'Loading raw data, time elapsed: {elapsed} seconds')
            elif progress[0]=='rechunking':
                progress_info.configure(text=f'Rechunking data to new chunk shape, time elapsed: {elapsed} seconds')
            elif progress[0]=='abort':
                progress_info.configure(text=f'Aborting analysis, please wait...')
            elif progress[0]=='stopped':
                popup.destroy()
                os.remove(self.progressfile)
                self.start_button.configure(state='normal')
                self.return_button.configure(state='normal')
                return
            else:
                progressbar.set(progress[0]/progress[1])
                progress_info.configure(text=f"Analyzing data, channel: {progress[0]}/{progress[1]}, time elapsed: {elapsed} seconds")
            time.sleep(0.01)
            
        currenttime=time.time()
        elapsed=round(currenttime-start,2)
        os.remove(self.progressfile)
        progress_label.configure(text='')
        progress_info.configure(text=f"Analysis finished in {elapsed} seconds.")
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self.start_button.configure(state='normal')
        self.return_button.configure(state='normal')