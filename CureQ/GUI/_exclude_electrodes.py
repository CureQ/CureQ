# Imports
from tkinter import filedialog
from functools import partial
import os
import json
from pathlib import Path
import threading

# External imports
import customtkinter as ctk
import pandas as pd
import numpy as np
from CTkMessagebox import CTkMessagebox
from PIL import ImageGrab

# Imports from package
from ..core._features import recalculate_features

class recalculate_features_class(ctk.CTkFrame):
    """
    Recalculate features while excluding certain electrodes from wells
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.parent=parent

        # Weight configuration
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Control frame - load different experiments/initiate recalculation
        self.control_frame = ctk.CTkFrame(master=self)
        self.control_frame.grid(row=0, column=0, pady=10, padx=10, sticky='nesw')

        # Selected files frame - display files selected for recalculation
        self.selected_files_frame = ctk.CTkScrollableFrame(master=self)
        self.selected_files_frame.grid(row=0, column=1, pady=10, padx=10, sticky='nesw')

        # Control frame buttons
        self.select_folder_button = ctk.CTkButton(master=self.control_frame, text='Load folder', command=self.load_folder)
        self.select_folder_button.grid(row=0, column=0, pady=(10, 5), padx=10, sticky='nesw')
        
        self.edit_config_button = ctk.CTkButton(master=self.control_frame, text='Edit/New configuration', command=self.edit_configuration_func)
        self.edit_config_button.grid(row=1, column=0, pady=(5, 5), padx=10, sticky='nesw')

        self.recalculate_features_button = ctk.CTkButton(master=self.control_frame, text='Recalculate features', command=self.recalculate_features_button_func)
        self.recalculate_features_button.grid(row=2, column=0, pady=(5, 10), padx=10, sticky='nesw')

        # Config image
        self.config_image = ctk.CTkLabel(master=self.control_frame, text="")
        self.config_image.grid(row=3, column=0, pady=10, padx=10, sticky='nesw')

        return_to_main = ctk.CTkButton(master=self, text="Return to main menu", command=lambda: self.parent.show_frame(self.parent.home_frame), fg_color=parent.gray_1)
        return_to_main.grid(row=1, column=0, pady=10, padx=10, sticky='nesw')

        self.file_buttons = {}
        self.well_amnt = 0
        self.electrode_amnt = 0
        self.configuration = None
        self.config_selected = False

    def load_folder(self):
        """
        Walk through a folder and collect all files ending with 'Features.csv'
        """
        folder=filedialog.askdirectory()
        if folder == '':
            return
        well_amnts=[]
        electrode_amnts=[]
        file_names=[]

        # Walk through folder
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith("Features.csv") and not "Electrode" in file:
                    data=pd.read_csv(os.path.join(root, file))
                    well_amnts.append(len(data))
                    file_names.append(os.path.join(root, file))
                
                    # Also get parameters.json to get electrode amount
                    try:
                        with open(os.path.join(root, 'parameters.json'), "r") as json_file:
                            parameters = json.load(json_file)
                        electrode_amnts.append(parameters['electrode amount'])
                    except:
                        CTkMessagebox(title="Error",
                              message=f"Could not find a complementary 'parameters.json' file for '{file}'. Please make sure every feature file is accompanied by the original 'parameters.json' file.",
                              icon="cancel",
                              wraplength=400)
                        return
                    
        # Check if the well layout is the same for all experiments
        if not(np.min(well_amnts) == np.max(well_amnts)):
            CTkMessagebox(title="Error",
                              message='Not all experiments have the same amount of wells, please remove the exceptions from the folder.',
                              icon="cancel",
                              wraplength=400)
            return
        else:
            self.well_amnt = np.min(well_amnts)

        # Check if the electrode amount is the same for all experiments
        if not(np.min(electrode_amnts) == np.max(electrode_amnts)):
            CTkMessagebox(title="Error",
                              message='Not all experiments have the same amount of electrodes per well, please remove the exceptions from the folder.',
                              icon="cancel",
                              wraplength=400)
            return
        else:
            self.electrode_amnt = np.min(electrode_amnts)


        self.display_selected_files(file_names)

    def display_selected_files(self, files):
        """
        Display all selected featurefiles including tickbox
        """
        # Destroy existing buttons
        for btn in self.file_buttons.values():
            btn["button"].destroy()
        self.file_buttons = {}

        # Loop over files and create buttons
        for i, file in enumerate(files):
            pady=(2.5, 2.5)
            if i == 0: pady = (5, 2.5)
            if i == len(files): pady = (2.5, 5)

            file_button = ctk.CTkButton(master=self.selected_files_frame, text=file, anchor='w', fg_color=self.parent.selected_color, hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6), command=partial(self.toggle_button_state, file))
            file_button.grid(row=i, column=0, padx=5 , pady=pady, sticky='nesw')

            self.file_buttons[file] = {
                "button": file_button,
                "state": True
            }

    def toggle_button_state(self, file):
        """
        Toggle button state
        """

        if self.file_buttons[file]["state"]:
            fg_color=self.parent.theme["CTkButton"]["fg_color"]
            hover_color=self.parent.theme["CTkButton"]["hover_color"]
            self.file_buttons[file]["state"] = False
            self.file_buttons[file]["button"].configure(fg_color=fg_color, hover_color=hover_color)
        else:
            self.file_buttons[file]["state"] = True
            self.file_buttons[file]["button"].configure(fg_color=self.parent.selected_color, hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6))

    def edit_configuration_func(self):
        """Open window to edit configuration"""
        if len(self.file_buttons) == 0:
            CTkMessagebox(title="Error",
                              message='Please first select a folder before editing a configuration, the layout of the configuration is determined by the selected files.',
                              icon="cancel",
                              wraplength=400)
            return

        edit_configuration(parent=self.parent, main_window=self, wells=self.well_amnt, electrodes=self.electrode_amnt, existing_config=self.configuration)

    def recalculate_features_func(self):
        """Function recalculating the features, called as a thread"""
        failed = []
        finished = []
        errors = []
        outputtext=""

        for i, file in enumerate(self.selected_files):
            filepath = Path(file)

            try:
                # Retrieve neccesary parameters
                with open(os.path.join(filepath.parent, "parameters.json")) as json_file:
                    parameters = json.load(json_file)

                recalculate_features(outputfolder=filepath.parent, well_amnt=self.well_amnt, electrode_amnt=self.electrode_amnt, electrodes=self.configuration, sampling_rate=parameters["sampling rate"], measurements=parameters["measurements"])
            
                print(f"Recalculated features for: {filepath.parent}")
                finished.append(filepath.stem)
            except Exception as error:
                failed.append(filepath.stem)
                errors.append(error)
            self.progressbar.set((i+1)/len(self.selected_files))
    
        outputtext += "Finished files:\n" if len(finished) > 0 else "Did not finish any files\n"
        for file in finished:
            outputtext += f"{file}\n"

        outputtext += "Failed files:" if len(failed) > 0 else ""
        for i, file in enumerate(failed):
            outputtext += f"\n{file}:"
            outputtext += f"\n\t{errors[i]}\n"

        CTkMessagebox(message=outputtext, option_1="Ok", title="Recalculated features", width=800, wraplength=750)


        self.popup.destroy()

    def recalculate_features_button_func(self):
        """Feature called by the button, performs checkes and start 'recalculate_features_func' as a thread"""
        # Check if config has been selected
        if not self.config_selected:
            CTkMessagebox(title="Error", message="No configuration selected, please create a configuration using 'Edit/New configuration'", icon="cancel", wraplength=400)
            return

        # Retrieve all files that are selected
        self.selected_files = []

        for file in list(self.file_buttons.keys()):
            if self.file_buttons[file]["state"]:
                self.selected_files.append(file)
        
        # Check if any files are selected
        if len(self.selected_files) == 0:
            CTkMessagebox(title="Error", message='No files selected', icon="cancel")
            return
        
        # Initialize popup and progressbar
        self.popup=ctk.CTkToplevel(self)
        self.popup.title('Recalculating features')
        try:
            self.popup.after(250, lambda: self.popup.iconbitmap(os.path.join(self.parent.icon_path)))
        except Exception as error:
            print(error)

        self.progress_label = ctk.CTkLabel(master=self.popup, text="Recalculating features...")
        self.progress_label.grid(row=0, column=0, pady=(10, 5), padx=10, sticky='nesw')

        self.progressbar=ctk.CTkProgressBar(master=self.popup, orientation='horizontal', mode='determinate', progress_color="#239b56", width=400)
        self.progressbar.grid(row=1, column=0, pady=(5, 10), padx=10, sticky='nesw')
        self.progressbar.set(0)

        process=threading.Thread(target=self.recalculate_features_func)
        process.start()

class edit_configuration(ctk.CTkToplevel):
    """
    Create/save/edit electrode on/off configurations

    Configurations are saved in json format, specifying wells, electrodes and which electrodes are turned on/off using a boolean array
    """
    def __init__(self, main_window, parent, wells, electrodes, existing_config=None):
        super().__init__(parent)
        self.parent=parent
        self.main_window=main_window

        try:
            self.after(250, lambda: self.iconbitmap(os.path.join(parent.icon_path)))
        except Exception as error:
            print(error)

        self.title("Electrode configuration")

        # Weight configuration
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Control frame - load different experiments/initiate recalculation
        self.control_frame = ctk.CTkFrame(master=self)
        self.control_frame.grid(row=0, column=0, pady=0, padx=0, sticky='nesw')

        # Manage configuration buttons
        self.manage_config_frame = ctk.CTkFrame(master=self.control_frame)
        self.manage_config_frame.grid(row=0, column=0, pady=(10,5), padx=10, sticky='nesw')

        self.select_config_button = ctk.CTkButton(master=self.manage_config_frame, text='Load configuration', command=self.load_config)
        self.select_config_button.grid(row=0, column=0, pady=(10, 5), padx=10, sticky='nesw')
        
        self.save_config_button = ctk.CTkButton(master=self.manage_config_frame, text='Save configuration', command=self.save_config)
        self.save_config_button.grid(row=1, column=0, pady=(5, 5), padx=10, sticky='nesw')
        
        self.apply_config_button = ctk.CTkButton(master=self.manage_config_frame, text='Apply configuration', command=self.apply_config)
        self.apply_config_button.grid(row=2, column=0, pady=(5, 10), padx=10, sticky='nesw')

        # Control selection
        self.manage_config_frame = ctk.CTkFrame(master=self.control_frame)
        self.manage_config_frame.grid(row=1, column=0, pady=(5,10), padx=10, sticky='nesw')

        self.select_folder_button = ctk.CTkButton(master=self.manage_config_frame, text='Select all', command=self.select_all)
        self.select_folder_button.grid(row=0, column=0, pady=(10, 5), padx=10, sticky='nesw')
        
        self.edit_config_button = ctk.CTkButton(master=self.manage_config_frame, text='Deselect all', command=self.deselect_all)
        self.edit_config_button.grid(row=1, column=0, pady=(5, 10), padx=10, sticky='nesw')

        # Electrode configuration frame
        self.config_frame = ctk.CTkFrame(master=self)
        self.config_frame.grid(row=0, column=1, pady=10, padx=10)

        self.electrode_buttons = {}
        self.well_frame_list = []

        # Load buttons based on well/electrode config
        self.create_buttons(wells, electrodes)

        # If existing config was given, apply config
        if existing_config is not None:
            for i, value in enumerate(existing_config):
                if not value:
                    self.toggle_button_state(str(i+1))            

    def create_buttons(self, wells, electrodes):
        """Create buttons representing electrodes""" 

        # Create frames representing wells to hold buttons
        width, height = self.parent.calculate_well_grid(wells)
    
        self.electrode_buttons = {}
        self.well_frame_list = []

        counter = 1

        for h in range(height):
            for w in range(width):
                well_frame = ctk.CTkFrame(master=self.config_frame)
                well_frame.grid(row=h, column=w, pady=10, padx=10)
                self.well_frame_list.append(well_frame)

        electrode_layout = self.parent.calculate_electrode_grid(num_items = electrodes)
        electrode_button_size=20

        for well_frame in self.well_frame_list:
            for x in range(electrode_layout.shape[0]):
                for y in range(electrode_layout.shape[1]):
                    if electrode_layout[x,y]:
                        electrode_btn=ctk.CTkButton(master=well_frame,
                                                    text="",
                                                    height=electrode_button_size,
                                                    width=electrode_button_size,
                                                    fg_color=self.parent.selected_color,
                                                    hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6),
                                                    command=partial(self.toggle_button_state, str(counter))
                        )

                        electrode_btn.grid(row=x, column=y, sticky='nesw')
                        self.electrode_buttons[str(counter)] = {
                            "button": electrode_btn,
                            "state": True
                        }
                        counter+=1

    def toggle_button_state(self, electrode):
        """
        Toggle button state
        """

        if self.electrode_buttons[electrode]["state"]:
            self.electrode_buttons[electrode]["state"] = False
            self.electrode_buttons[electrode]["button"].configure(fg_color=self.parent.unselected_color, hover_color=self.parent.adjust_color(self.parent.unselected_color, factor=0.6))
        else:
            self.electrode_buttons[electrode]["state"] = True
            self.electrode_buttons[electrode]["button"].configure(fg_color=self.parent.selected_color, hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6))

    def select_all(self):
        for btn in self.electrode_buttons.values():
            btn["state"] = True
            btn["button"].configure(fg_color=self.parent.selected_color, hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6))

    def deselect_all(self): 
        for btn in self.electrode_buttons.values():
            btn["state"] = False
            btn["button"].configure(fg_color=self.parent.unselected_color, hover_color=self.parent.adjust_color(self.parent.unselected_color, factor=0.6))

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")],
            title="Save configuration file"
        )

        if not file_path:
            return
        
        # Load config in array
        config_list = []
        for btn in self.electrode_buttons.values(): config_list.append(btn["state"])

        np.save(file_path, np.array(config_list))

        CTkMessagebox(message=f"Labels succesfully saved at {file_path}", icon="check", option_1="Ok", title="Saved configuration")

    def load_config(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")],
            title="Save configuration file"
            )

        if not file_path:
            return

        try:
            config_list = np.load(file_path)
        except:
            CTkMessagebox(title="Error", message='Could not load in config file, please make sure you have selected the correct file.', icon="cancel")
            return

        if len(config_list) != len(self.electrode_buttons):
            CTkMessagebox(title="Error", message=f"The amount of electrodes in the selected file ({len(config_list)}) does not match with the amount of electrodes in the currently selected experiments ({len(self.electrode_buttons)}).", icon="cancel", wraplength=400)
            return

        for i, value in enumerate(config_list):
            if value:
                self.electrode_buttons[str(i+1)]["state"] = True
                self.electrode_buttons[str(i+1)]["button"].configure(fg_color=self.parent.selected_color, hover_color=self.parent.adjust_color(self.parent.selected_color, factor=0.6))
            else:
                self.electrode_buttons[str(i+1)]["state"] = False
                self.electrode_buttons[str(i+1)]["button"].configure(fg_color=self.parent.unselected_color, hover_color=self.parent.adjust_color(self.parent.unselected_color, factor=0.6))

    def apply_config(self):
        # Take a screenshot of the config to display on the main window
        widget = self.config_frame
        widget.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        w = widget.winfo_width()
        h = widget.winfo_height()
        
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(w*0.2, h*0.2))
        self.main_window.config_image.configure(image=ctk_image, text="")

        # Save the config
        config_list = []
        for btn in self.electrode_buttons.values(): config_list.append(btn["state"])
        self.main_window.configuration = np.array(config_list)
        self.main_window.config_selected = True

        # Destroy widget
        self.destroy()