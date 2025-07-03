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
from ..core._plotting import get_defaultcolors, feature_boxplots, combined_feature_boxplots, features_over_time, plot_network_diagram, plot_3d

class plotting_window(ctk.CTkFrame):
    """
    Allows the user to combine multiple MEA experiments to generate plots and discern between healthy and diseased cultures.
    """
    def __init__(self, parent):
        super().__init__(parent)
        
        self.parent=parent

        # Weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Plot frames
        plotting_frame = ctk.CTkFrame(master=self, fg_color='transparent')
        plotting_frame.grid(row=1, column=0, sticky='nesw')

        select_file_frame = ctk.CTkFrame(master=self, fg_color=self.parent.gray_1)
        select_file_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nesw', columnspan=4)
        select_file_frame.grid_columnconfigure(0, weight=1)

        features_over_time_frame = ctk.CTkFrame(master=plotting_frame, fg_color=self.parent.gray_1)
        features_over_time_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nesw')
        features_over_time_frame.grid_columnconfigure(0, weight=1)
        features_over_time_frame.grid_columnconfigure(1, weight=1)

        boxplot_frame = ctk.CTkFrame(master=plotting_frame, fg_color=self.parent.gray_1)
        boxplot_frame.grid(row=2, column=0, sticky='nesw', padx=5, pady=5)
        boxplot_frame.grid_columnconfigure(0, weight=1)
        boxplot_frame.grid_columnconfigure(1, weight=1)

        features_over_time_frame.grid_columnconfigure(1, weight=1)

        # Selected files frame
        self.selected_files_frame = ctk.CTkScrollableFrame(master=self)
        self.selected_files_frame.grid(row=1, column=1, padx=5, pady=5, sticky='nesw')

        # Label frames
        self.assign_labels_frame = ctk.CTkFrame(master=self)
        self.assign_labels_frame.grid(row=1, column=2, padx=5, pady=5, sticky='new')

        label_main_frame=ctk.CTkFrame(master=self, fg_color='transparent')
        label_main_frame.grid(row=1, column=3, sticky='nesw')

        create_labels_frame = ctk.CTkFrame(master=label_main_frame, fg_color=self.parent.gray_1)
        create_labels_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nesw')
        create_labels_frame.grid_columnconfigure(0, weight=1)

        label_settings_frame=ctk.CTkFrame(label_main_frame, fg_color=self.parent.gray_1)
        label_settings_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nesw')
        label_settings_frame.grid_columnconfigure(0, weight=1)

        self.labels_frame = ctk.CTkFrame(master=label_main_frame, fg_color=self.parent.gray_1)
        self.labels_frame.grid(row=2, column=0, padx=5, pady=5, sticky='nesw')
        self.labels_frame.grid_columnconfigure(0, weight=1)

        label_main_frame.grid_rowconfigure(0, weight=0)
        label_main_frame.grid_rowconfigure(1, weight=0)
        label_main_frame.grid_rowconfigure(2, weight=0)
        label_main_frame.grid_rowconfigure(3, weight=0)

        # Values
        self.selected_folder = ''
        self.well_buttons=[]
        self.label_buttons=[]
        self.assigned_labels={}
        self.selected_label=[]
        self.default_colors=get_defaultcolors()
        self.well_amnt=None

        self.select_folder_button=ctk.CTkButton(master=select_file_frame, text='Select a folder', command=self.create_well_buttons)
        self.select_folder_button.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky='nesw')

        # Features over time
        fot_label=ctk.CTkLabel(master=features_over_time_frame, text="Features over time", font=ctk.CTkFont(size=15))
        fot_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='w')

        prefix_label=ctk.CTkLabel(master=features_over_time_frame, text="DIV Prefix:")
        prefix_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.prefix_entry=ctk.CTkEntry(master=features_over_time_frame)
        self.prefix_entry.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        show_datapoints_label=ctk.CTkLabel(master=features_over_time_frame, text='Show datapoints:')
        show_datapoints_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        self.show_datapoints_entry=ctk.CTkCheckBox(master=features_over_time_frame, text='')
        self.show_datapoints_entry.grid(row=2, column=1, padx=10, pady=10, sticky='nesw')

        create_plot_button=ctk.CTkButton(text="Plot Features over time", master=features_over_time_frame, command=self.create_plots)
        create_plot_button.grid(row=3, column=0, pady=10, padx=10, sticky='nesw', columnspan=2)

        # Combine measurements
        bp_label=ctk.CTkLabel(master=boxplot_frame, text="Boxplots", font=ctk.CTkFont(size=15))
        bp_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='w')

        bp_show_datapoints_label=ctk.CTkLabel(master=boxplot_frame, text='Show datapoints:')
        bp_show_datapoints_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.bp_show_datapoints_entry=ctk.CTkCheckBox(master=boxplot_frame, text='')
        self.bp_show_datapoints_entry.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        discern_wells_label=ctk.CTkLabel(master=boxplot_frame, text='Color wells:')
        discern_wells_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        self.discern_wells_entry=ctk.CTkCheckBox(master=boxplot_frame, text='')
        self.discern_wells_entry.grid(row=2, column=1, padx=10, pady=10, sticky='nesw')

        create_plot_button=ctk.CTkButton(text="Create Boxplots", master=boxplot_frame, command=self.create_boxplots)
        create_plot_button.grid(row=3, column=0, pady=10, padx=10, sticky='nesw', columnspan=2)


        # Frame synchrony frame
        sync_frame = ctk.CTkFrame(master=plotting_frame, fg_color=self.parent.gray_1)
        sync_frame.grid(row=3, column=0, sticky='nesw', padx=5, pady=5)

        sync_label = ctk.CTkLabel(master=sync_frame, text="Synchronicity plots", font=ctk.CTkFont(size=15))
        sync_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='w')

        self.network_diagram = ctk.CTkCheckBox(master=sync_frame, text='Network diagram')
        self.network_diagram.grid(row=1, column=1, padx=10, pady=5, sticky='w')

        self.plot_3D = ctk.CTkCheckBox(master=sync_frame, text='3D-plot')
        self.plot_3D.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        self.sync_discern_wells_entry = ctk.CTkCheckBox(master=sync_frame, text='Color wells')
        self.sync_discern_wells_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')

        sync_button = ctk.CTkButton(master=sync_frame, text="Create synchronicity Plot", command=self.create_sync_plot)
        sync_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        return_to_main = ctk.CTkButton(master=plotting_frame, text="Return to main menu", command=lambda: self.parent.show_frame(self.parent.home_frame), fg_color=parent.gray_1)
        return_to_main.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='nesw')

        # Create labels
        self.new_label_entry=ctk.CTkEntry(master=create_labels_frame, placeholder_text="Create a label; e.g.: \'control\'")
        self.new_label_entry.grid(row=0, column=0, pady=10, padx=10, sticky='nesw')

        new_label_button=ctk.CTkButton(master=create_labels_frame, text='Add Label', command=self.new_label)
        new_label_button.grid(row=1, column=0, padx=10, pady=5, sticky='nesw')

        save_label_button=ctk.CTkButton(master=label_settings_frame, text="Save Labels", command=self.save_labels)
        save_label_button.grid(row=2, column=0, padx=10, pady=(10,5), sticky='nesw')

        save_label_button=ctk.CTkButton(master=label_settings_frame, text="Import Labels", command=self.import_labels)
        save_label_button.grid(row=3, column=0, padx=10, pady=5, sticky='nesw')

        save_label_button=ctk.CTkButton(master=label_settings_frame, text="Reset Labels", command=self.reset_labels)
        save_label_button.grid(row=4, column=0, padx=10, pady=(5,10), sticky='nesw')

    def save_labels(self):
        file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Save JSON File"
        )

        if not file_path:
            return
        
        with open(file_path, "w") as json_file:
            json.dump(self.assigned_labels, json_file, indent=4)

        CTkMessagebox(message=f"Labels succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Labels")

    def import_labels(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],  # File type filters
            title="Open JSON File")

        if not file_path:
            return

        with open(file_path, "r") as json_file:
            imported_labels = json.load(json_file)
        self.set_labels(imported_labels)

    def set_labels(self, labels):
        # Reset labels
        for label_button in self.label_buttons:
            label_button.destroy()
        self.label_buttons=[]

        for label in labels.keys():
            self.create_label_button(label)
        self.assigned_labels=labels

        self.update_button_colours()

    def reset_labels(self):
        self.set_labels({})

    def create_plots(self):
        # Perform checks
        valid_groups=False
        for label in self.assigned_labels.keys():
            if len(self.assigned_labels[label]) > 0:
                valid_groups=True
        if not valid_groups:
            CTkMessagebox(title="Error",
                              message='Please create at least one label, and assign at least one well to it.',
                              icon="cancel",
                              wraplength=400)
            return
        if str(self.prefix_entry.get()) == '':
            CTkMessagebox(title="Error",
                              message='Please define the prefix that is used to indicate the age of the neurons, e.g. DIV, t, day',
                              icon="cancel",
                              wraplength=400)
            return

        # Call library
        try:
            file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save PDF File"
            )

            if not file_path:
                return

            pdf_path=features_over_time(folder=self.selected_folder, labels=copy.deepcopy(self.assigned_labels), div_prefix=str(self.prefix_entry.get()), output_fileadress=file_path, colors=self.default_colors, show_datapoints=bool(self.show_datapoints_entry.get()))
            CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")
            webbrowser.open(f"file://{pdf_path}")
        except Exception as error:
            CTkMessagebox(title="Error",
                              message='Something went wrong while creating the plots',
                              icon="cancel",
                              wraplength=400)
            
            traceback.print_exc()

    def create_boxplots(self):
        # check if there are groups that contain wells
        valid_groups=False
        for label in self.assigned_labels.keys():
            if len(self.assigned_labels[label]) > 0:
                valid_groups=True
        if not valid_groups:
            CTkMessagebox(title="Error",
                              message='Please create at least one label, and assign at least one well to it.',
                              icon="cancel",
                              wraplength=400)
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save PDF File"
            )

            if not file_path:
                return
            
            pdf_path=combined_feature_boxplots(folder=self.selected_folder, labels=copy.deepcopy(self.assigned_labels), output_fileadress=file_path, colors=self.default_colors, show_datapoints=bool(self.bp_show_datapoints_entry.get()), discern_wells=bool(self.discern_wells_entry.get()), well_amnt=self.well_amnt)
            webbrowser.open(f"file://{pdf_path}")
            CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")
        except Exception as error:
            CTkMessagebox(title="Error",
                              message='Something went wrong while creating the boxplots',
                              icon="cancel",
                              wraplength=400)
            traceback.print_exc()

    def create_sync_plot(self):
        # check if there are groups that contain wells
        valid_groups=False
        for label in self.assigned_labels.keys():
            if len(self.assigned_labels[label]) > 0:
                valid_groups=True
        if not valid_groups:
            CTkMessagebox(title="Error",
                              message='Please create at least one label, and assign at least one well to it.',
                              icon="cancel",
                              wraplength=400)
            return
        
        if self.plot_3D.get() == 1:
            try:
                file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save 3D-plot png File"
                )
                
                if not file_path:
                    return
                
                # Get parameters
                parameters_path = os.path.join(self.selected_folder, "parameters.json")
                if not os.path.exists(parameters_path):
                    CTkMessagebox(title="Missing file", message="parameters.json not found in the selected folder.", icon="cancel")
                    return

                with open(parameters_path, 'r') as f:
                    parameters = json.load(f)

                
                pdf_path = plot_3d(folder=self.selected_folder, labels=copy.deepcopy(self.assigned_labels), output_fileadress=file_path, well_amnt=self.well_amnt, parameters = parameters, diagnol = True)
                webbrowser.open(f"file://{pdf_path}")
                CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")

                CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"An error occurred while generating the plot:\n{str(e)}", icon="cancel")
                traceback.print_exc()

        elif self.network_diagram.get() == 1:
            try:
                file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save 3D-plot png File"
                )
                
                if not file_path:
                    return
                
                # Get parameters
                parameters_path = os.path.join(self.selected_folder, "parameters.json")
                if not os.path.exists(parameters_path):
                    CTkMessagebox(title="Missing file", message="parameters.json not found in the selected folder.", icon="cancel")
                    return

                with open(parameters_path, 'r') as f:
                    parameters = json.load(f)
                if self.network_diagram.get() == 1:
                    pdf_path = plot_network_diagram(folder=self.selected_folder,
                                        labels=copy.deepcopy(self.assigned_labels),
                                        output_fileadress=file_path,
                                        well_amount=self.well_amnt,
                                        parameters=parameters,
                                        asynchrony=0)
                    webbrowser.open(f"file://{pdf_path}")
                    CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")

                CTkMessagebox(message=f"Figures succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved Figures")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"An error occurred while generating the plot:\n{str(e)}", icon="cancel")
                traceback.print_exc()
                
                
            except Exception as error:
                CTkMessagebox(title="Error",
                                message='Something went wrong while creating the synchronicity plots',
                                icon="cancel",
                                wraplength=400)
                traceback.print_exc()
        else: 
            CTkMessagebox(title="No plot type selected", message="Please select a plot type (3D or Network diagram).", icon="warning")
        return

    def set_selected_label(self, label):
        self.selected_label=label

    def create_label_button(self, label):
        label_button=ctk.CTkButton(master=self.labels_frame, text=label, command=partial(self.set_selected_label, label), fg_color=self.default_colors[len(self.label_buttons)], hover_color=self.parent.adjust_color(self.default_colors[len(self.label_buttons)], 0.6))
        label_button.grid(row=len(self.label_buttons), column=0, pady=5, padx=10, sticky='nesw')
        self.label_buttons.append(label_button)
        self.assigned_labels[label]=[]
        self.new_label_entry.delete(0, END)

    def new_label(self):
        label = self.new_label_entry.get()
        if (label == '') or (label in self.assigned_labels.keys()):
            return
        self.create_label_button(label)

    def update_button_colours(self):
        for well_button in self.well_buttons:
            well_button.configure(fg_color=self.parent.theme["CTkButton"]["fg_color"][1], hover_color=self.parent.theme["CTkButton"]["hover_color"][1])
        for index, key in enumerate(self.assigned_labels.keys()):
            for well in self.assigned_labels[key]:
                self.well_buttons[well-1].configure(fg_color=self.default_colors[index], hover_color=self.parent.adjust_color(self.default_colors[index], 0.6))

    def well_button_func(self, button):
        # If the label was already selected, remove the selection
        if button in self.assigned_labels[self.selected_label]:
            self.assigned_labels[self.selected_label].remove(button)
        else:
            # First remove well from all labels
            for key in self.assigned_labels.keys():
                if button in self.assigned_labels[key]:
                    self.assigned_labels[key].remove(button)
            self.assigned_labels[self.selected_label].append(button)
        self.update_button_colours()

    def create_well_buttons(self):
        folder=filedialog.askdirectory()
        if folder == '':
            return
        well_amnts=[]
        file_names=[]
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith("Features.csv") and not "Electrode" in file:
                    data=pd.read_csv(os.path.join(root, file))
                    well_amnts.append(len(data))
                    file_names.append(Path(os.path.join(root, file)).stem)
        if len(well_amnts) < 1:
            CTkMessagebox(title="Error",
                              message='Not enough features-files found. Minimum amount is 1',
                              icon="cancel",
                              wraplength=400)
            return
        if not(np.min(well_amnts) == np.max(well_amnts)):
            CTkMessagebox(title="Error",
                              message='Not all experiments have the same amount of wells, please remove the exceptions from the folder.',
                              icon="cancel",
                              wraplength=400)
            return
        
        self.well_amnt=int(np.mean(well_amnts))
        self.selected_folder=folder
        self.select_folder_button.configure(text=self.selected_folder)
        width, height = self.parent.calculate_well_grid(np.mean(well_amnts))
        counter=1

        for h in range(height):
            for w in range(width):
                well_button=ctk.CTkButton(master=self.assign_labels_frame, text=counter, command=partial(self.well_button_func, counter), height=100, width=100, font=ctk.CTkFont(size=25))
                well_button.grid(row=h, column=w, sticky='nesw')
                self.well_buttons.append(well_button)
                counter+=1

        for i, file in enumerate(file_names):
            file_label=ctk.CTkLabel(master=self.selected_files_frame, text=file)
            file_label.grid(row=i, column=0, sticky='w', padx=10, pady=2)

        self.select_folder_button.configure(state='disabled')    