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
from ..core._utilities import rechunk_dataset

class compress_files(ctk.CTkFrame):
    """
    Allows the user to compress/rechunk multiple files.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.select_file_button=ctk.CTkButton(master=self, text="Select a file", command=self.openfiles)
        self.select_file_button.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        self.to_main_frame_button=ctk.CTkButton(master=self, text="Return to main menu", command=lambda: parent.show_frame(self.parent.home_frame))
        self.to_main_frame_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        gzip_level_text=ctk.CTkLabel(master=self, text="GZIP compression level: 1")
        gzip_level_text.grid(row=2, column=0, padx=10, pady=10, sticky='nesw', columnspan=2)

        def show_value(value):
            gzip_level_text.configure(text=f"GZIP compression level: {int(value)}")

        self.slider_value = ctk.IntVar(value=1)
        self.gzip_level_slider=ctk.CTkSlider(master=self, from_=1, to=9, orientation='horizontal', variable=self.slider_value, width=200, number_of_steps=8, command=show_value)
        self.gzip_level_slider.grid(row=3, column=0, padx=10, pady=(0, 10), columnspan=2, sticky='nesw')
        self.gzip_level_slider.configure(state='disabled')

        def lzf_selected():
            self.gzip_var.set(False)
            self.gzip_level_slider.configure(state='disabled')
            self.compression_method='lzf'

        def gzip_selected():
            self.lzf_var.set(False)
            self.gzip_level_slider.configure(state='normal')
            self.compression_method='gzip'

        self.lzf_var=ctk.IntVar()
        lzf_button=ctk.CTkCheckBox(self, text='LZF', onvalue=True, offvalue=False, variable=self.lzf_var, command=lzf_selected)
        lzf_button.grid(row=1, column=0, padx=10, pady=10, sticky='nesw')

        self.gzip_var=ctk.IntVar()
        gzip_button=ctk.CTkCheckBox(self, text='GZIP', onvalue=True, offvalue=False, variable=self.gzip_var, command=gzip_selected)
        gzip_button.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        self.compress_all=ctk.BooleanVar()
        compress_all_button=ctk.CTkCheckBox(self, text='Compress all files in folder', onvalue=True, offvalue=False, variable=self.compress_all)
        compress_all_button.grid(row=4, column=0, padx=10, pady=10, sticky='nesw', columnspan=2)
        self.compress_all.set(False)

        # Default values
        self.lzf_var.set(True)
        self.selected_file=''
        self.compression_method='lzf'
        self.compression_level=1

        self.compress_files_button=ctk.CTkButton(master=self, text="Start compression", command=lambda: threading.Thread(target=self.compress_files_function).start())
        self.compress_files_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        self.abort_flag=False

    def openfiles(self):
        selectedfile = filedialog.askopenfilename(filetypes=[("MEA data", "*.h5")])
        if len(selectedfile)!=0:
            self.selected_file=selectedfile
            self.select_file_button.configure(text=selectedfile)

    def compress_files_function(self):
        if self.selected_file=='':
            return
        if self.compress_all.get():
            folder=os.path.dirname(self.selected_file)
            mea_files=[os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]
        else:
            mea_files=[self.selected_file]

        popup=ctk.CTkToplevel(self)
        popup.title('File Compression')
        try:
            popup.after(250, lambda: popup.iconbitmap(os.path.join(self.parent.icon_path)))
        except Exception as error:
            print(error)
        
        progressinfo=ctk.CTkLabel(master=popup, text=f'Compressing {len(mea_files)} file{"s" if len(mea_files) != 1 else ""}')
        progressinfo.grid(row=0, column=0, pady=10, padx=20)
        progressbarlength=300
        progressbar=ctk.CTkProgressBar(master=popup, orientation='horizontal', width=progressbarlength, mode='determinate', progress_color="#239b56")
        progressbar.grid(row=1, column=0, pady=10, padx=20)
        progressbar.set(0)
        info=ctk.CTkLabel(master=popup, text='')
        info.grid(row=2, column=0, pady=10, padx=20)
        finishedfiles=ctk.CTkLabel(master=popup, text='')
        finishedfiles.grid(row=3, column=0, pady=10, padx=20)

        # Aborting compression
        def abort_compression():
            self.abort_flag=True
            print("Aborting compression for further files...")
            abort_button.configure(text="Aborting compression for further files...")
            abort_button.configure(state='disabled')

        abort_button=ctk.CTkButton(master=popup, text="Abort compression", command=abort_compression)
        abort_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')
        popup.protocol("WM_DELETE_WINDOW", abort_compression)

        succesfiles=[]
        failedfiles=[]
    

        for file in range(len(mea_files)):
            if self.abort_flag:
                self.abort_flag=False
                popup.destroy()
                return
            info.configure(text=f"Compressing file {file+1} out of {len(mea_files)}")
            try:
                print(f"Compressing {mea_files[file]}")
                rechunk_dataset(fileadress=mea_files[file], compression_method=self.compression_method, compression_level=self.compression_level, always_compress_files=True)
                succesfiles.append(mea_files[file])
            except Exception as error:
                print(f"Could not compress {mea_files[file]}")
                traceback.print_exc()
                failedfiles.append(mea_files[file])
            currentprogress=((file+1)/len(mea_files))
            progressbar.set(currentprogress)
            finishedfiles_text=""
            if len(succesfiles) != 0:
                finishedfiles_text+="Compressed files:\n"
                for i in range(len(succesfiles)):
                    finishedfiles_text+=f"{succesfiles[i]}\n"
            if len(failedfiles) != 0:
                finishedfiles_text+="Failed files:\n"
                for i in range(len(failedfiles)):
                    finishedfiles_text+=f"{failedfiles[i]}\n"
            finishedfiles.configure(text=finishedfiles_text)
        
        abort_button.configure(state='disabled')
        info.configure(text="Finished compression")
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)