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

class batch_processing(ctk.CTkFrame):
    """
    Allows the user to process multiple MEA experiments.
    """
    def __init__(self, parent):
        super().__init__(parent)  # Initialize the parent class

        self.parent=parent

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Different main frames
        self.file_selection_frame=ctk.CTkFrame(self)
        self.file_selection_frame.grid(row=0, column=0, sticky='nsew')
        self.file_selection_frame.grid_columnconfigure(0, weight=1)
        self.file_selection_frame.grid_rowconfigure(0, weight=0)
        self.file_selection_frame.grid_rowconfigure(1, weight=1)
        self.file_selection_frame.grid_rowconfigure(2, weight=0)

        # List to store selected file paths
        self.selected_files = []

        """ Folder selection frame """
        self.folder_frame = ctk.CTkFrame(self.file_selection_frame, fg_color=parent.gray_3)
        self.folder_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        # Select folder button
        self.select_folder_btn = ctk.CTkButton(
            self.folder_frame, 
            text="Select Folder", 
            command=self.select_folder
        )
        self.select_folder_btn.pack(pady=10, padx=10, fill="x")

        # Treeview frame
        self.tree_frame = ctk.CTkFrame(self.file_selection_frame)
        self.tree_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)

        # Vertical Scrollbar
        self._vertical_scrollbar = ctk.CTkScrollbar(
            self.tree_frame, 
            orientation="vertical"
        )
        self._vertical_scrollbar.grid(row=0, column=1, padx=(0, 5), pady=5, sticky='ns')
        
        # Horizontal Scrollbar
        self._horizontal_scrollbar = ctk.CTkScrollbar(
            self.tree_frame, 
            orientation="horizontal"
        )
        self._horizontal_scrollbar.grid(row=1, column=0, padx=5, pady=(0, 5), sticky='ew')

        # Treeview Customisation
        bg_color = self.tree_frame.cget("fg_color")[1]
        text_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        selected_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        treestyle = ttk.Style()
        treestyle.theme_use('default')
        treestyle.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0)
        treestyle.map('Treeview', background=[('selected', bg_color)], foreground=[('selected', selected_color)])
        parent.bind("<<TreeviewSelect>>", lambda event: parent.focus_set())

        # Treeview with scrollbars
        self.tree = ttk.Treeview(
            self.tree_frame, 
            columns=("fullpath", "type"), 
            selectmode="extended",
            yscrollcommand=self._vertical_scrollbar.set,
            xscrollcommand=self._horizontal_scrollbar.set,
            show='tree'
        )
        self.tree.grid(row=0, column=0, sticky="nsew", pady=(5,0), padx=(5,0))

        # Link scrollbars to Treeview
        self._vertical_scrollbar.configure(command=self.tree.yview)
        self._horizontal_scrollbar.configure(command=self.tree.xview)

        # Configure treeview columns
        self.tree.heading("#0", text="Directory Structure")
        self.tree.column("#0", width=600, stretch=True)

        # Selected tag
        self.tree.tag_configure('selected', background='#627747')

        # Print selected files button
        self.print_files_btn = ctk.CTkButton(
            self.file_selection_frame, 
            text="Confirm Selection", 
            command=partial(self.confirm_selection, parent)
        )
        self.print_files_btn.grid(row=2, column=0, padx=10, pady=10, sticky='nesw')

        to_main_menu=ctk.CTkButton(self.file_selection_frame, 
            text="Return to Main Menu", 
            command=lambda: parent.show_frame(self.parent.home_frame)
            )
        to_main_menu.grid(row=3, column=0, padx=10, pady=10, sticky='news')

        # Bind events
        self.tree.bind("<<TreeviewOpen>>", self.open_folder)
        self.tree.bind("<<TreeviewSelect>>", self.item_selected)

        """ Process files frame """
        self.selected_files_labels = []
        self.progressbars = []
        self.analysis_crashed=False     # Flag to communicate if analysis has crashed with a file

        # Frames
        self.process_files_frame=ctk.CTkFrame(self)
        self.selected_files_frame = ctk.CTkFrame(self.process_files_frame, fg_color=parent.gray_3)
        self.selected_files_frame.grid(row=0, column=0, pady=5, padx=5, sticky='nesw')
        self.selected_files_frame.grid_columnconfigure(0, weight=1)
        self.selected_files_frame.grid_rowconfigure(0, weight=1)

        self.process_files_frame.grid_columnconfigure(0, weight=2)
        self.process_files_frame.grid_rowconfigure(0, weight=1)
        self.process_files_frame.grid_columnconfigure(1, weight=1)

        self.settings_and_progress_frame=ctk.CTkFrame(self.process_files_frame, fg_color="transparent")
        self.settings_and_progress_frame.grid(row=0, column=1, sticky='nesw')
        self.settings_and_progress_frame.grid_rowconfigure(1, weight=1)
        self.settings_and_progress_frame.grid_columnconfigure(0, weight=1)

        self.analysis_settings_frame=ctk.CTkFrame(self.settings_and_progress_frame, fg_color=parent.gray_3)
        self.analysis_settings_frame.grid(row=0, column=0, sticky='nesw', pady=5, padx=5)

        self.analysis_progress_frame=ctk.CTkFrame(self.settings_and_progress_frame, fg_color=parent.gray_3)
        self.analysis_progress_frame.grid(row=1, column=0, sticky='nesw', pady=5, padx=5)
        self.analysis_progress_frame.grid_columnconfigure(0, weight=1)
        
        # Analysis settings
        self.sampling_rate_label = ctk.CTkLabel(self.analysis_settings_frame, text="Sampling rate:")
        self.sampling_rate_label.grid(row=0, column=0, pady=5, padx=5, sticky='w')
        self.sampling_rate_input = ctk.CTkEntry(self.analysis_settings_frame)
        self.sampling_rate_input.grid(row=0, column=1, pady=5, padx=5, sticky='w')

        self.electrode_amnt_label = ctk.CTkLabel(self.analysis_settings_frame, text="Electrode amount:")
        self.electrode_amnt_label.grid(row=1, column=0, pady=5, padx=5, sticky='w')
        self.electrode_amnt_input = ctk.CTkEntry(self.analysis_settings_frame)
        self.electrode_amnt_input.grid(row=1, column=1, pady=5, padx=5, sticky='w')

        # Analysis and progress
        self.start_analysis_button = ctk.CTkButton(self.analysis_progress_frame, text="Initiate Analysis", command=lambda: threading.Thread(target=self.initiate_analysis).start())
        self.start_analysis_button.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)

        self.progressbar = ctk.CTkProgressBar(self.analysis_progress_frame, orientation='horizontal', mode='determinate', progress_color="#239b56")
        self.progressbar.grid(row=1, column=0, sticky='nesw', padx=5, pady=20)
        self.progressbar.set(0)

        self.abort_analysis_bool=False
        self.abort_analysis_button = ctk.CTkButton(self.analysis_progress_frame, text="Abort Analysis", command=self.abort_analysis_func)
        self.abort_analysis_button.grid(row=2, column=0, sticky='nesw', padx=5, pady=5)
        self.abort_analysis_button.configure(state='disabled')

        self.back_button = ctk.CTkButton(self.analysis_progress_frame, text="Return", command=lambda: parent.show_frame(self.parent.home_frame))
        self.back_button.grid(row=3, column=0, sticky='nesw', padx=5, pady=5)


    # Select parent folder
    def select_folder(self):
        # Clear existing tree
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.selected_files=[]

        # Open folder selection dialog
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Populate tree with root folder
            root_id = self.tree.insert("", "end", text=os.path.basename(folder_path), 
                                       values=(folder_path, "Folder"), open=True)
            self.populate_tree(root_id, folder_path)

    def populate_tree(self, parent, path):
        try:
            for name in sorted(os.listdir(path)):
                full_path = os.path.join(path, name)
                if os.path.isdir(full_path):
                    # Add folder
                    folder_id = self.tree.insert(parent, "end", text=name, 
                                                 values=(full_path, "Folder"))
                    # Add a dummy child to enable expansion
                    self.tree.insert(folder_id, "end", text="placeholder")
                else:
                    # Add file
                    if name.endswith(".h5"):
                        self.tree.insert(parent, "end", text=name, 
                                        values=(full_path, "File"))
        except PermissionError:
            print(f"Permission denied for {path}")

    def open_folder(self, event):
        # Remove placeholder and populate actual contents when folder is opened
        item = self.tree.focus()
        if self.tree.get_children(item) and \
           self.tree.item(self.tree.get_children(item)[0])['text'] == 'placeholder':
            # Remove placeholder
            self.tree.delete(self.tree.get_children(item)[0])
            # Populate actual contents
            self.populate_tree(item, self.tree.item(item, "values")[0])

    def deselect_all(self):
        for item in self.tree.selection():
            self.tree.selection_remove(item)

    def add_selection(self, item):
        if len(self.tree.item(item)['tags'])!=0 and self.tree.item(item)['tags'][0]=="selected":
            self.tree.item(item, tags=())
            self.selected_files.remove(item)
        else:
            self.tree.item(item, tags="selected")
            self.selected_files.append(item)

    def item_selected(self, event):
        for selected_item in self.tree.selection():
            values = self.tree.item(selected_item, "values")
            # If the selected item is a file
            if values[1]=="File":
                self.add_selection(selected_item)
            # If the selected item is a folder
            elif values[1]=="Folder":
                children = self.tree.get_children(selected_item)
                for child in children:
                    if self.tree.item(child, "values")[1] == "File" and self.tree.item(child, "values")[0].endswith(".h5"):
                        self.add_selection(child) 
        self.deselect_all()

    def confirm_selection(self, parent):
        self.selected_files_table = ctk.CTkScrollableFrame(self.selected_files_frame)
        self.selected_files_table.grid(row=0, column=0, padx=10, pady=10, sticky='nesw')
        self.selected_files_table.columnconfigure(0, weight=1)
        self.selected_files_table.columnconfigure(1, weight=1)

        for index, file in enumerate(self.selected_files):
            label=ctk.CTkLabel(self.selected_files_table, text=self.tree.item(file, "values")[0], corner_radius=20, width=300)
            label.grid(row=index, column=0, sticky='w')
            self.selected_files_labels.append(label)
            progressbar=ctk.CTkProgressBar(self.selected_files_table, orientation='horizontal', mode='determinate', progress_color="#239b56")
            progressbar.grid(row=index, column=1, sticky='w')
            progressbar.set(0)
            self.progressbars.append(progressbar)

        self.process_files_frame.grid(row=0, column=0, sticky='nsew')
        self.file_selection_frame.grid_forget()

    def call_library(self, file, index):
        try:
            electrode_amnt=int(self.electrode_amnt_input.get())
            sampling_rate=int(self.sampling_rate_input.get())
            analyse_wells(file, electrode_amnt=electrode_amnt, sampling_rate=sampling_rate, parameters=self.parent.parameters)
        except:
            traceback.print_exc()
            self.progressbars[index].configure(progress_color='#922b21')
            self.progressbars[index].set(1)
            self.analysis_crashed=True

    def initiate_analysis(self):
        self.start_analysis_button.configure(state='disabled')
        self.back_button.configure(state='disabled')
        self.abort_analysis_button.configure(state='normal')

        for index, file in enumerate(self.selected_files):
            if self.abort_analysis_bool:
                self.abort_analysis_button.configure(text="Aborted Analysis")
                break
            # Start the analysis in another thread

            file=self.tree.item(file, "values")[0]

            process=threading.Thread(target=self.call_library, args=(file, index))
            process.start()

            # Read out the progress
            progressfile=f'{os.path.split(file)[0]}/progress.npy'

            # Make sure there is no pre-existing progressfile
            try: os.remove(progressfile)
            except: pass

            while True:
                while True: 
                    if self.analysis_crashed:
                        break
                    try:
                        progress=np.load(progressfile)
                        break
                    except: pass
                try:
                    if progress[0] == 'done':
                        break
                except: pass
                if self.analysis_crashed:
                    self.analysis_crashed=False
                    break
                try:
                    currentprogress=(progress[0]/progress[1])
                    self.progressbars[index].set(currentprogress)
                except: pass
            self.progressbar.set((index+1)/len(self.selected_files_labels))
        
        self.abort_analysis_button.configure(state="disabled")
        self.back_button.configure(state='normal')

    def abort_analysis_func(self):
        self.abort_analysis_bool=True
        self.abort_analysis_button.configure(text="Aborting analysis after current file")
        self.abort_analysis_button.configure(state="disabled")