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
import requests
import subprocess
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

# Import the MEA library
try:
    from .mea import *
except:
    from mea import *


class MainApp(ctk.CTk):
    """
    Control frame selection and hold 'global' variables.
    """
    def __init__(self):
        # Initialize GUI
        super().__init__()

        # Get icon - works for both normal and frozen
        relative_path="cureq_icon.ico"
        try:
            self.base_path = sys._MEIPASS
        except Exception:
            source_path = Path(__file__).resolve()
            self.base_path = source_path.parent
        self.icon_path=os.path.join(self.base_path, relative_path)
        try:
            self.iconbitmap(self.icon_path)
        except Exception as error:
            print("Could not load in icon")
            print(error)

        # 'Global' variables
        self.tooltipwraplength=200

        # Colors
        self.gray_1 = '#333333'
        self.gray_2 = '#2b2b2b'
        self.gray_3 = "#3f3f3f"
        self.gray_4 = "#212121"
        self.gray_5 = "#696969"
        self.gray_6 = "#292929"
        self.entry_gray = "#565b5e"

        self.text_color = '#dce4ee'

        # Set theme from json
        theme_path=os.path.join(self.base_path, "theme.json")
        
        with open(theme_path, "r") as json_file:
            self.theme = json.load(json_file)

        ctk.set_default_color_theme(theme_path)
        ctk.set_appearance_mode("dark")

        base_color = self.theme["CTkButton"]["fg_color"][1]
        self.primary_1 = self.mix_color(base_color, self.gray_6, factor=0.9)
        self.primary_1 = self.adjust_color(self.primary_1, 1.5)

        # Initialize main frame
        self.show_frame(main_window)

        # The parent holds all the analysis parameters in a dict, which are here initialized with the default values
        self.parameters = get_default_parameters()
        self.default_parameters = get_default_parameters()

        print("Successfully launched MEA Analysis Tool")

    # Handle frame switching
    def show_frame(self, frame_class, *args, **kwargs):
        for widget in self.winfo_children():
            widget.destroy()
        frame = frame_class(self, *args, **kwargs)
        frame.pack(expand=True, fill="both") 

    # Function to calculate the optimal grid
    def calculate_optimal_grid(self, num_items):
        min_difference = num_items
        optimal_width = num_items
        optimal_height = 1
        for width in range(1, int(math.sqrt(num_items)) + 1):
            if num_items % width == 0:
                height = num_items // width
                difference = abs(width - height)
                if difference < min_difference:
                    min_difference = difference
                    optimal_width = width
                    optimal_height = height
        return int(optimal_width), int(optimal_height)
    
    def adjust_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:], 16)
        
        r = int(min(255, max(0, r * factor)))
        g = int(min(255, max(0, g * factor)))
        b = int(min(255, max(0, b * factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def mix_color(self, hex_color1, hex_color2, factor):
        # Convert hex colors to RGB
        hex_color1 = hex_color1.lstrip('#')
        hex_color2 = hex_color2.lstrip('#')
        
        # Original color RGB
        r1 = int(hex_color1[:2], 16)
        g1 = int(hex_color1[2:4], 16)
        b1 = int(hex_color1[4:], 16)
        
        # Gray color RGB
        r2 = int(hex_color2[:2], 16)
        g2 = int(hex_color2[2:4], 16)
        b2 = int(hex_color2[4:], 16)
        
        # Mix colors based on factor
        r = int(r1 * (1-factor) + r2 * factor)
        g = int(g1 * (1-factor) + g2 * factor)
        b = int(b1 * (1-factor) + b2 * factor)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def set_theme(self, base_color):
        theme_path=os.path.join(self.base_path, "theme.json")

        with open(theme_path, "r") as json_file:
            theme = json.load(json_file)
        
        # Edit all relevant widgets
        theme["CTkButton"]["fg_color"]=["#3a7ebf", base_color]
        theme["CTkButton"]["hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]

        theme["CTkCheckBox"]["fg_color"]=["#3a7ebf", base_color]
        theme["CTkCheckBox"]["hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]

        theme["CTkEntry"]["border_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.8)]

        theme["CTkComboBox"]["border_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        theme["CTkComboBox"]["button_color"]=["#325882", base_color]
        theme["CTkComboBox"]["button_hover_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]

        theme["CTkOptionMenu"]["fg_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        theme["CTkOptionMenu"]["button_color"]=["#325882", base_color]
        theme["CTkOptionMenu"]["button_hover_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        
        theme["CTkSlider"]["button_color"]=[base_color, base_color]
        theme["CTkSlider"]["button_hover_color"]=[self.adjust_color(base_color, factor=0.6), self.adjust_color(base_color, factor=0.6)]

        # Tabview buttons
        theme["CTkSegmentedButton"]["selected_color"]=["#3a7ebf", base_color]
        theme["CTkSegmentedButton"]["selected_hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]
        
        self.primary_1 = self.mix_color(base_color, self.gray_6, factor=0.9)
        self.primary_1 = self.adjust_color(self.primary_1, 1.5)

        with open(theme_path, 'w') as json_file:
            json.dump(theme, json_file, indent=4)
        ctk.set_default_color_theme(theme_path)

        self.theme=theme

        self.show_frame(main_window)

class main_window(ctk.CTkFrame):
    """
    Main window and landing page for the user.
    Allows the user to switch to different frames to perform different tasks.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        parent.title(f"MEA Analysis Tool - Version: {version('CureQ')}")

        # Weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame for the sidebar buttons
        sidebarframe=ctk.CTkFrame(self)
        sidebarframe.grid(row=0, column=1, padx=5, pady=5, sticky='nesw')

        # Switch themes
        theme_switch = ctk.CTkButton(sidebarframe, text="Theme", command=self.colorpicker)
        theme_switch.grid(row=0, column=0, sticky='nesw', pady=10, padx=10)
        self.selected_color=parent.theme["CTkButton"]["fg_color"][1]

        cureq_button=ctk.CTkButton(master=sidebarframe, text="CureQ project", command=lambda: webbrowser.open_new("https://cureq.nl/"))
        cureq_button.grid(row=1, column=0, sticky='nesw', pady=10, padx=10)

        pypi_button=ctk.CTkButton(master=sidebarframe, text="Library", command=lambda: webbrowser.open_new("https://pypi.org/project/CureQ/"))
        pypi_button.grid(row=2, column=0, sticky='nesw', pady=10, padx=10)

        github_button=ctk.CTkButton(master=sidebarframe, text="GitHub", command=lambda: webbrowser.open_new("https://github.com/CureQ"))
        github_button.grid(row=3, column=0, sticky='nesw', pady=10, padx=10)

        # Check for updates
        installed_version=self.get_installed_version()
        latest_version=self.get_latest_version()

        if installed_version is not None and latest_version is not None:
            if latest_version != installed_version:
                update_button=ctk.CTkButton(master=sidebarframe, text="A new version is available!\nClick here to install it.", command=self.update_package, fg_color="#1d5200", hover_color="#0f2b00")
                update_button.grid(row=4, column=0, sticky='nesw', pady=10, padx=10)

        # Main button frame
        main_buttons_frame=ctk.CTkFrame(self)
        main_buttons_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nesw')
        main_buttons_frame.grid_columnconfigure(0, weight=1)
        main_buttons_frame.grid_columnconfigure(1, weight=1)
        main_buttons_frame.grid_rowconfigure(0, weight=1)
        main_buttons_frame.grid_rowconfigure(1, weight=1)

        # Go to parameter_frame
        to_parameters_button=ctk.CTkButton(master=main_buttons_frame, text="Set Parameters", command=lambda: parent.show_frame(parameter_frame), height=90, width=160)
        to_parameters_button.grid(row=0, column=0, sticky='nesw', pady=10, padx=10)

        # View results
        view_results_button=ctk.CTkButton(master=main_buttons_frame, text="View Results", command=lambda: parent.show_frame(select_folder_frame), height=90, width=160)
        view_results_button.grid(row=1, column=0, sticky='nesw', pady=10, padx=10)

        # Batch processing
        batch_processing_button=ctk.CTkButton(master=main_buttons_frame, text="Batch Processing", command=lambda: parent.show_frame(batch_processing), height=90, width=160)
        batch_processing_button.grid(row=0, column=1, sticky='nesw', pady=10, padx=10)

        # single file processing
        process_file_button=ctk.CTkButton(master=main_buttons_frame, text="Process single file", command=lambda: parent.show_frame(process_file_frame), height=90, width=160)
        process_file_button.grid(row=1, column=1, sticky='nesw', pady=10, padx=10)

        # Utility/plotting buttons
        util_plot_button_frame=ctk.CTkFrame(master=self)
        util_plot_button_frame.grid(row=1, column=0, columnspan=2, sticky='nesw', pady=5, padx=5)

        compression_button=ctk.CTkButton(master=util_plot_button_frame, text="Compress/Rechunk Files", command=lambda: parent.show_frame(compress_files))
        compression_button.grid(row=0, column=0, sticky='nesw', padx=10, pady=10)
        
        features_over_time_button=ctk.CTkButton(master=util_plot_button_frame, text="Plotting", command=lambda: parent.show_frame(plotting_window))
        features_over_time_button.grid(row=0, column=1, sticky='nesw', padx=10, pady=10)

        for i in range(3):
            util_plot_button_frame.grid_rowconfigure(i, weight=1)

    def get_installed_version(self):
        try:
            return version("CureQ")
        except importlib.metadata.PackageNotFoundError:
            return None

    def get_latest_version(self):
        url = f"https://pypi.org/pypi/CureQ/json"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["info"]["version"]
        return None

    def update_package(self):
        latest_version = self.get_latest_version()
        if self.get_installed_version() != latest_version:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"CureQ=={latest_version}"])
                print(f"Successfully installed CureQ version {latest_version}.")
                CTkMessagebox(message=f"Successfully installed CureQ version {latest_version}. Please restart the application for the changes to take effect.", icon="check", option_1="Ok", title="Updated Package", wraplength=400)
            except Exception as error:
                print(f"Failed to install CureQ version {latest_version}.")
                traceback.print_exc()
                CTkMessagebox(title="Error", message=f"Failed to install CureQ version {latest_version}:\n{error}", icon="cancel", wraplength=400)

    def colorpicker(self):
        popup=ctk.CTkToplevel(self)
        popup.title('Theme Selector')

        try:
            popup.after(250, lambda: popup.iconbitmap(os.path.join(self.parent.icon_path)))
        except Exception as error:
            print(error)
        
        def set_theme():
            self.parent.set_theme(self.selected_color)
            popup.destroy()
            self.parent.show_frame(main_window)

        def set_color(color):
            self.selected_color=color

        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)
        colorpicker = CTkColorPicker(popup, width=350, command=lambda e: set_color(e), initial_color=self.selected_color)
        colorpicker.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        confirm_button=ctk.CTkButton(master=popup, text="Confirm", command=set_theme)
        confirm_button.grid(row=1, column=0, sticky='nesw', padx=5, pady=5)
    

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

        # Remove inactive electrodes
        removeinactivelabel=ctk.CTkLabel(otherparameters, text="Remove inactive electrodes:")
        removeinactivelabel.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        removeinactivetooltip = CTkToolTip(removeinactivelabel, message='Remove inactive electrodes from the spike, burst and network burst feature calculations.', wraplength=self.tooltipwraplength)
        removeinactivevar=ctk.IntVar()
        removeinactiveinput=ctk.CTkCheckBox(otherparameters, onvalue=True, offvalue=False, variable=removeinactivevar, command=removeinactivefunc, text='')
        removeinactiveinput.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Setup the activity threshold
        activitythlabel=ctk.CTkLabel(master=otherparameters, text="Activity threshold:")
        activitythlabel.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        activitythtooltip = CTkToolTip(activitythlabel, message='Define the minimal activity a channel must have, to be used in calculating features. Value should be given in hertz, so a value of 0.1 would mean any channel with less that 1 spike per 10 seconds will be removed', wraplength=self.tooltipwraplength)
        activitythinput=ctk.CTkEntry(otherparameters)
        activitythinput.grid(row=3, column=1, padx=10, pady=10, sticky='w')

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
                self.parent.show_frame(main_window)

            except Exception as error:
                traceback.print_exc()
                CTkMessagebox(title="Error",
                              message='Certain parameters could not be converted to the correct datatype (e.g. int or float). Please check if every parameter has the correct values',
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


class select_folder_frame(ctk.CTkFrame):
    """
    Allows the user to select a dataset and outputfile to inspect the results.
    """
    def __init__(self, parent):
        super().__init__(parent)

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

        return_button=ctk.CTkButton(self, text="Return", command= lambda: parent.show_frame(main_window))
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

        self.parent.title(f"MEA Analysis Tool - Version: {version('CureQ')} - {self.folder}")

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
        ywells, xwells=parent.calculate_optimal_grid(well_amnt)
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
        if electrode_amnt==12:
            # Preset for 12 electrode MEA
            electrode_layout=np.array([ [False, True, True, False],
                                        [True, True, True, True],
                                        [True, True, True, True],
                                        [False, True, True, False]])
        else:
            height, width = parent.calculate_optimal_grid(electrode_amnt)
            electrode_layout = np.ones((height, width), dtype=bool)
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
        return_to_main = ctk.CTkButton(master=self, text="Return to main menu", command=lambda: self.parent.show_frame(main_window), fg_color=parent.gray_1)
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

        self.return_button = ctk.CTkButton(master=self, text="Return to Main Menu", command=lambda: parent.show_frame(main_window), fg_color=parent.gray_1)
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

        # Start the analysis in a new thread
        process=threading.Thread(target=self.call_library)
        process.start()

        # And keep track of the progress
        start=time.time()
        path=os.path.split(self.selected_file)[0]
        self.progressfile=f"{path}/progress.npy"
        # Remove potential pre-existing progressfile
        try: 
            os.remove(self.progressfile)
        except:
            pass
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
            command=lambda: parent.show_frame(main_window)
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

        self.back_button = ctk.CTkButton(self.analysis_progress_frame, text="Return", command=lambda: parent.show_frame(main_window))
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

        self.to_main_frame_button=ctk.CTkButton(master=self, text="Return to main menu", command=lambda: parent.show_frame(main_window))
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


        return_to_main = ctk.CTkButton(master=plotting_frame, text="Return to main menu", command=lambda: self.parent.show_frame(main_window), fg_color=parent.gray_1)
        return_to_main.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='nesw')

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



    def set_selected_label(self, label):
        self.selected_label=label

    def create_label_button(self, label):
        label_button=ctk.CTkButton(master=self.labels_frame, text=label, command=partial(self.set_selected_label, label), fg_color=self.default_colors[len(self.label_buttons)])
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
            well_button.configure(fg_color=self.parent.theme["CTkButton"]["fg_color"][1])
        for index, key in enumerate(self.assigned_labels.keys()):
            for well in self.assigned_labels[key]:
                self.well_buttons[well-1].configure(fg_color=self.default_colors[index])

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
        print(self.assigned_labels)
        self.update_button_colours()

    def create_well_buttons(self):
        folder=filedialog.askdirectory()
        if folder == '':
            return
        well_amnts=[]
        file_names=[]
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith("Features.csv"):
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
        width, height = self.parent.calculate_optimal_grid(np.mean(well_amnts))
        counter=1

        for w in range(width):
            for h in range(height):
                well_button=ctk.CTkButton(master=self.assign_labels_frame, text=counter, command=partial(self.well_button_func, counter), height=100, width=100, font=ctk.CTkFont(size=25))
                well_button.grid(row=w, column=h, sticky='nesw')
                self.well_buttons.append(well_button)
                counter+=1

        for i, file in enumerate(file_names):
            file_label=ctk.CTkLabel(master=self.selected_files_frame, text=file)
            file_label.grid(row=i, column=0, sticky='w', padx=10, pady=2)

        self.select_folder_button.configure(state='disabled')

def MEA_GUI():
    """
    Launches the graphical user interface (GUI) of the MEA Analysis Tool.

    Always launch the function with an "if __name__ == '__main__':" guard as follows:
        if __name__ == "__main__":
            MEA_GUI()
    """

    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    MEA_GUI()