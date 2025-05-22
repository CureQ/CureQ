# Imports
import os
import json
import math
import webbrowser
import sys
from pathlib import Path
from tkinter import *
from importlib.metadata import version

# External libraries
import numpy as np
import customtkinter as ctk
from CTkToolTip import *
from CTkColorPicker import *
import requests

# Package imports
from ..mea import get_default_parameters
from ..core._utilities import gui_logger

# Import GUI components
from ._exclude_electrodes import recalculate_features_class
from ._parameters import parameter_frame
from ._view_results import select_folder_frame
from ._process_file import process_file_frame
from ._batch_processing import batch_processing
from ._compress_files import compress_files
from ._plotting import plotting_window

class MainApp(ctk.CTk):
    """
    Control frame selection and hold 'global' variables.
    """
    def __init__(self):
        # Initialize GUI
        super().__init__()

        # Get icon - works for both normal and frozen
        relative_path="MEAlytics_logo.ico"
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
        self.selected_color = "#125722"     # Green color to show something is selected
        self.unselected_color = "#570700"   # Red color to show somehting is not selected

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
        self.home_frame=main_window
        self.show_frame(self.home_frame)

        # The parent holds all the analysis parameters in a dict, which are here initialized with the default values
        self.parameters = get_default_parameters()
        self.default_parameters = get_default_parameters()

        print("Successfully launched MEAlytics GUI :)")

    # Handle frame switching
    def show_frame(self, frame_class, *args, **kwargs):
        for widget in self.winfo_children():
            widget.destroy()
        frame = frame_class(self, *args, **kwargs)
        frame.pack(expand=True, fill="both") 

    # Function to calculate the well grid
    def calculate_well_grid(self, num_items):
        """Calculate grid shape for displaying wells, contains preset for 24w plate"""
        if num_items == 24:
            return 6, 4

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

        # Height should always be the lowest value of the two
        # Width is returned first, then height
        return int(max(optimal_width, optimal_height)), int(min(optimal_width, optimal_height))

        # Function to calculate the electrode grid
    def calculate_electrode_grid(self, num_items):
        """Calculate grid shape for displaying electrodes, contains present for 12 and 16 electrode wells"""
        if num_items == 12:
            return np.array([   [False, True, True, False],
                                [True, True, True, True],
                                [True, True, True, True],
                                [False, True, True, False]])
        if num_items == 16:
            return np.array([   [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True],
                                [True, True, True, True]])

        width, height = self.calculate_well_grid(num_items)
        return np.ones(shape=(width, height))

    
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
        self.show_frame(self.home_frame)

class main_window(ctk.CTkFrame):
    """
    Main window and landing page for the user.
    Allows the user to switch to different frames to perform different tasks.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        parent.title(f"MEAlytics - Version: {version('CureQ')}")

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

        github_button=ctk.CTkButton(master=sidebarframe, text="User Guide", command=lambda: webbrowser.open_new("https://cureq.github.io/CureQ/"))
        github_button.grid(row=4, column=0, sticky='nesw', pady=10, padx=10)

        # Check for updates
        installed_version=self.get_installed_version()
        latest_version=self.get_latest_version()

        if installed_version is not None and latest_version is not None:
            if latest_version != installed_version:
                update_button=ctk.CTkButton(master=sidebarframe, text="A new version is available!\n Close the application and run\n\"pip install CureQ --upgrade\"\nto install it.", fg_color="#1d5200", hover_color="#0f2b00")
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

        recalculate_features_button = ctk.CTkButton(master=util_plot_button_frame, text='Exclude Electrodes', command=lambda: parent.show_frame(recalculate_features_class))
        recalculate_features_button.grid(row=0, column=2, pady=10, padx=10, sticky='nesw')

        for i in range(3):
            util_plot_button_frame.grid_rowconfigure(i, weight=1)

    def get_installed_version(self):
        try:
            return version("CureQ")
        except importlib.metadata.PackageNotFoundError:
            return None

    def get_latest_version(self):
        url = f"https://pypi.org/pypi/CureQ/json"
        try:
            response = requests.get(url)
        except:
            return None
        
        if response.status_code == 200:
            return response.json()["info"]["version"]
        return None

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

@gui_logger()
def MEA_GUI():
    """
    Launches the graphical user interface (GUI) of MEAlytics.

    Always launch the function with an "if __name__ == '__main__':" guard as follows:
        if __name__ == "__main__":
            MEA_GUI()
    """

    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    MEA_GUI()