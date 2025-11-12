# Imports

# Ecternal libraries
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Core components
from ..core._impedance_heatmap import *

class impedance_frame(ctk.CtkToplevel):
    def __init__(self, master, parent, tabwidget):
        super().__init__(master)

        self.title("Impedance")

        self.parent = parent
        self.tabwidget = tabwidget

        self.h5_file = "D:/mea_data/2025_44_dagen_iv/Bow_div44.h5"
        self.impedance = load_impedance(self.h5_file)
        self.mapping = load_mapping(self.h5_file)

        heatmap_button_frame = ctk.CTkFrame(master = self, width = 180)
        heatmap_button_frame.grid(row = 0, column = 1, padx=20, pady=20, sticky='ns')
        heatmap_button_frame.grid_propagate(False)
        heatmap_button_frame.pack_propagate(False)

        self.hm_data_generator = ctk.CTkButton(heatmap_button_frame, text="Process data", command=self.generate_data_handler)
        self.hm_data_generator.pack(pady=15)

        self.display_hm = ctk.CTkButton(heatmap_button_frame, text="Show heatmap", state="disabled", command=lambda: self.show_heatmap)
        self.display_hm.pack(pady=15)

        self.cmap = cmap_creation()

        # Placeholder figure
        #figure = SOMETHING
        #canvas = FigureCanvasTkAgg(figure, master=self)

    def show_heatmap(self): # TODO kijk of heatmap interactive kan.
        well_data = reshape_wells(self.impedance, self.mapping)
        well_dims = calculate_well_dimensions(self.mapping)
        fig = create_viability_heatmap(well_data, well_dims)


        canvas = FigureCanvasTkAgg(fig, master = self)