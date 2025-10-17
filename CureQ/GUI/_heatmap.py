# Imports
import os
import json
from tkinter import *
from tkinter import filedialog
import threading
import webbrowser

# External libraries
import customtkinter as ctk
from CTkToolTip import *
from CTkColorPicker import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Heatmap core components
from ..core._heatmap import *

class progressbar(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Preparing data")
        self.geometry("400x50")
        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text="Preparing data, please wait...", font=ctk.CTkFont(size=14))
        self.label.pack(pady=(10, 10))

        try:
            self.after(250, lambda: self.iconbitmap(os.path.join(parent.parent.icon_path)))
        except Exception as error:
            print(error)

        self.lift()

class heatmap_frame(ctk.CTkToplevel):
    def __init__(self, master, parent, datashape, parameters, xwells, ywells, folder, tabwidget):
        super().__init__(master)

        self.title("Heatmap")

        try:
            self.after(250, lambda: self.iconbitmap(os.path.join(parent.icon_path)))
        except Exception as error:
            print(error)

        self.parent = parent
        self.tabwidget = tabwidget
        self.parameters = parameters

        self.h5_file = os.path.join(folder, "output_values.h5")

        self.hm_Vars = {
            "fps": 10,
            "Num frames": None,
            "df": None,
            "n_Wells": None,
            "n_Electrodes": None,
            "v_max": None,
            "Size": None,
            "Precomputed Max": None,
            "Rows": None,
            "Cols": None,
            "cmap": None,
            "max_df": None,
            "last_hm": None,
            "background_color": '#1a1a1a'
        }
        self.Classes = {
            "Unknown": list(range(0, int(datashape[0]/parameters["electrode amount"])+1)),
        }
        self.Colour_classes = {
            "Unknown": "grey"
        }

        self.hm_Vars["Rows"] = ywells
        self.hm_Vars["Cols"] = xwells
        self.hm_Vars["Num frames"] = int(parameters['measurements'] / parameters['sampling rate'] * self.hm_Vars["fps"])
        heatmap_button_frame = ctk.CTkFrame(master=self, width=180)
        heatmap_button_frame.grid(row=0, column=1, padx=20, pady=20, sticky="ns")
        heatmap_button_frame.grid_propagate(False)
        heatmap_button_frame.pack_propagate(False)

        self.hm_data_generator = ctk.CTkButton(heatmap_button_frame, text="Process data", command=self.generate_data_handler)
        self.hm_data_generator.pack(pady=15)

        # self.add_hm_labels = ctk.CTkButton(heatmap_button_frame, text="Add labels", state ="disabled", command=lambda: self.add_labels(heatmap_button_frame))
        # self.add_hm_labels.pack(pady=15)

        self.display_hm = ctk.CTkButton(heatmap_button_frame, text="Show heatmap", state="disabled", command=lambda: self.start_animation(self.hm_Vars, self.Classes, self.Colour_classes))
        self.display_hm.pack(pady=15)  

        self.display_hm_full = ctk.CTkButton(heatmap_button_frame, text="Show total activity", state="disabled", command=self.show_full_heatmap_handler)
        self.display_hm_full.pack(pady=15)  

        # Comment out until functional
        # self.download_hm = ctk.CTkButton(heatmap_button_frame, text="Download", state="disabled", command=lambda: self.download_heatmap(self.hm_Vars, self.Classes, self.Colour_classes))
        # self.download_hm.pack(pady=15)

        self.animation = None
        self.slider = None
        self.from_label = None
        self.to_label = None
        # self.add_legend_to_frame(heatmap_button_frame, self.Colour_classes)
        self.hm_Vars["cmap"] = cmap_creation()

        placeholder_fig = create_placeholder_figure(self.hm_Vars)
        placeholder_canvas = FigureCanvasTkAgg(placeholder_fig, master=self)
        placeholder_canvas.draw()
        placeholder_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        self.progressbar = None

        layout_warning = ctk.CTkButton(master=self, text="Warning: The well/electrode layout is auto-generated and may not match the physical plate exactly. Click here for details.", command=lambda: webbrowser.open_new("https://cureq.github.io/CureQ/supported_plates/"), fg_color=parent.gray_1)
        layout_warning.grid(row=2 , column=0, pady=10, padx=10, sticky='nesw', columnspan=2)

    def generate_data_handler(self):
        self.generate_data_progressbar()
        threading.Thread(target=self.generate_data).start()

    def generate_data_progressbar(self):
        self.progressbar = progressbar(self)

    def generate_data(self):
        self.hm_data_generator.configure(state='disabled')
        self.hm_Vars["df"], self.hm_Vars["n_Wells"], self.hm_Vars["n_Electrodes"], self.hm_Vars["v_max"], self.hm_Vars["Size"], self.hm_Vars["Precomputed Max"], self.hm_Vars["max_df"] = data_prepper(self.h5_file, self.parameters, self.hm_Vars, self.parent.calculate_electrode_grid)
        # self.add_hm_labels.configure(state="normal")
        self.display_hm.configure(state="normal")  
        self.display_hm_full.configure(state="normal")
        # self.download_hm.configure(state="normal")  
        self.hm_data_generator.configure(state='normal')
        self.hm_data_generator.configure(state='disabled')
        self.progressbar.destroy()
        self.progressbar = None

    def add_labels(self, heatmap_button_frame):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Open JSON File")

        if not file_path:
            return

        with open(file_path, "r") as json_file:
            imported_labels = json.load(json_file)
        
        self.Classes = imported_labels

        colours = ['green', 'violet', 'gold', 'red', 'blue', 'orange', 'cyan', 'pink', 'brown', 'gray']
        self.Colour_classes = {key: colour for key, colour in zip(self.Classes.keys(), colours)}

        # self.add_legend_to_frame(heatmap_button_frame, self.Colour_classes)

    def start_animation(self, Vars, Classes, Colour_classes):
        if hasattr(self, 'animation') and self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

        if hasattr(self, 'canvas') and self.canvas.get_tk_widget().winfo_exists():
            self.canvas.get_tk_widget().destroy()

        if hasattr(self, 'slider') and self.slider and self.slider.winfo_exists():
            self.slider.destroy()
            self.from_label.destroy()
            self.to_label.destroy()

        fig, update_func = make_hm(Vars, Classes, Colour_classes)

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        self.animation = animation.FuncAnimation(
            fig,
            update_func,
            frames=Vars["Num frames"],
            interval=100,
            blit=False,
            repeat=True
        )
        
        self.slider = ctk.CTkSlider(
            master=self,
            from_=0,
            to=(Vars["Num frames"]-1),
            number_of_steps=(Vars["Num frames"]-1)
        )
        self.slider.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.is_dragging = False

        def on_slider_change(value):
            frame = int(float(value))
            update_func(frame)
            self.canvas.draw_idle()

        def animation_update_wrapper(frame):
            if not self.is_dragging:
                self.slider.set(frame)
            return update_func(frame)

        self.animation._func = animation_update_wrapper

        def on_slider_press(event):
            self.is_dragging = True
            self.animation.event_source.stop()

        def on_slider_release(event):
            self.is_dragging = False
            current_frame = int(self.slider.get())
            self.animation.frame_seq = iter(range(current_frame, Vars["Num frames"]))
            self.animation.event_source.start()

        self.slider.configure(command=on_slider_change)
        self.slider.bind("<ButtonPress-1>", on_slider_press)
        self.slider.bind("<ButtonRelease-1>", on_slider_release)

        self.from_label = ctk.CTkLabel(master=self, text="0.0s")
        self.from_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.to_label = ctk.CTkLabel(master=self, text=f"{Vars['Num frames']/10:.1f}s")
        self.to_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        Vars["last_hm"] = "animation_hm"
        
        self.canvas.draw()

    def show_full_heatmap_handler(self):
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.generate_data_progressbar()
        threading.Thread(target=self.show_full_heatmap).start()

    def show_full_heatmap(self):
        # self.add_hm_labels.configure(state="disabled")
        self.display_hm.configure(state="disabled")
        self.display_hm_full.configure(state="disabled")
        # self.download_hm.configure(state="disabled")
        
        if self.slider:
            self.slider.destroy()
        
        if self.from_label:
            self.from_label.destroy()
            self.to_label.destroy()

        fig = make_hm_img(self.hm_Vars, self.Classes, self.Colour_classes)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        self.canvas = canvas

        self.from_label = ctk.CTkLabel(master=self, text="")
        self.from_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.to_label = ctk.CTkLabel(master=self, text="")
        self.to_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        
        self.hm_Vars["last_hm"] = "total_hm"

        # self.add_hm_labels.configure(state="normal")
        self.display_hm.configure(state="normal")  
        self.display_hm_full.configure(state="normal")
        # self.download_hm.configure(state="normal")

        self.progressbar.destroy()
        self.progressbar = None

    def download_heatmap(self, Vars, Classes, Colour_classes):
        if Vars["last_hm"] is None:
            return

        from tkinter.filedialog import asksaveasfilename
        from matplotlib.animation import FFMpegWriter

        if Vars["last_hm"] == "total_hm":
            fig = make_hm_img(Vars, Classes, Colour_classes)
            file_path = asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG file", "*.png"), ("PDF file", "*.pdf"), ("All files", "*.*")]
            )
            if file_path:
                fig.savefig(file_path, dpi=300)
            plt.close(fig)

        elif Vars["last_hm"] == "animation_hm":
            if not hasattr(self, 'animation') or not self.animation:
                print("Error: Animation object not found. Please show the heatmap first.")
                return

            self.animation.event_source.stop()

            file_path = asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")]
            )
            
            if not file_path:
                if hasattr(self, 'canvas') and self.canvas.get_tk_widget().winfo_exists():
                    self.animation.event_source.start()
                return

            hm_saving_window = ctk.CTkToplevel(self)
            hm_saving_window.title("Saving Animation")
            hm_saving_window.geometry("300x100")
            hm_saving_window.resizable(False, False)
            hm_saving_window.transient(self)
            hm_saving_window.lift()

            hm_saving_window.label = ctk.CTkLabel(hm_saving_window, text="Preparing to save...")
            hm_saving_window.label.pack(pady=10)

            hm_saving_window.progress = ctk.CTkProgressBar(hm_saving_window, width=250)
            hm_saving_window.progress.set(0)
            hm_saving_window.progress.pack(pady=5)

            def update_progress(current_frame, total_frames):
                progress = (current_frame + 1) / total_frames
                hm_saving_window.progress.set(progress)
                hm_saving_window.label.configure(text=f"Saving frame {current_frame + 1} of {total_frames}")
                hm_saving_window.update_idletasks()

            try:
                self.animation.save(
                    file_path,
                    writer=FFMpegWriter(fps=10, extra_args=['-crf', '28', '-preset', 'fast']),
                    dpi=150,
                    progress_callback=update_progress
                )
            except Exception as e:
                print(f"Failed to save MP4: {e}")
            finally:
                hm_saving_window.destroy()
                if hasattr(self, 'canvas') and self.canvas.get_tk_widget().winfo_exists():
                    self.animation.event_source.start()

    # def add_legend_to_frame(self, frame, colour_dict):
    #     for child in frame.winfo_children():
    #         if getattr(child, 'is_legend_widget', False):
    #             child.destroy()
        
    #     legend_label = ctk.CTkLabel(frame, text="Legend", font=("Arial", 14, "bold"))
    #     legend_label.is_legend_widget = True
    #     legend_label.pack(pady=(15, 5))  

    #     for class_name, color in colour_dict.items():
    #         legend_row = ctk.CTkFrame(frame, fg_color="transparent")
    #         legend_row.is_legend_widget = True
    #         legend_row.pack(anchor="w", padx=10, pady=2, fill="x")

    #         color_box = ctk.CTkLabel(legend_row, text="", width=20, height=20, corner_radius=3)
    #         color_box.is_legend_widget = True
    #         color_box.configure(fg_color=color)
    #         color_box.pack(side="left")

    #         text_label = ctk.CTkLabel(legend_row, text=class_name, anchor="w")
    #         text_label.is_legend_widget = True
    #         text_label.pack(side="left", padx=10)