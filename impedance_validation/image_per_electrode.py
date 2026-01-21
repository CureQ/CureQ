import customtkinter as ctk
from skimage import io, draw
import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

"""
This algorithm can split every omni image into seperate images per elektrode.
"""

# Laad afbeelding met skimage
class main(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Select electrodes")
        self.geometry("600x400")
        self.minsize(600,400)

        self.map_path = "C:/Users/jveer/Desktop/hogeschool/Afstudeerstage BMT jaar 5/Programmas/CureQ/impedance_validation/bow_omni/wells"
        self.out_path = "C:/Users/jveer/Desktop/hogeschool/Afstudeerstage BMT jaar 5/Programmas/CureQ/impedance_validation/bow_omni/electrodes"
        self.image_paths = os.listdir(self.map_path)
        self.current_img = self.image_paths[0]
        self.image_number = 0
        self.image = io.imread((self.map_path + "/" + self.current_img))
        self.mask = None
        self.point = None
        self.electrodes = []
        self.image_cutoff = 0.2
        self.arm_length = 244

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Stel thema in
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.setup_ui()

    def setup_ui(self):
        # Maak frames
        self.image_frame = ctk.CTkFrame(master=self, width=600)
        self.image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.control_frame = ctk.CTkFrame(master=self, width=200)
        self.control_frame.pack(side="right", fill="y", padx=10, pady=10)
        
        # Matplotlib figuur in Tkinter canvas
        self.setup_matplotlib_canvas()
        
        # Controle knoppen
        self.setup_control_buttons()
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.control_frame, 
            text="Selecteer de linker boven elektrode",
            wraplength=180
        )
        self.status_label.pack(pady=10)
        
    def setup_matplotlib_canvas(self):
        # Maak matplotlib figuur
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        # Toon afbeelding
        self.cut_image()
        self.ax.imshow(self.image, cmap="gray")
        self.ax.set_title(f"file: {self.current_img}. {self.image_number} out of {len(self.image_paths)}\n Klik elektrode linksboven", color='white')
        self.ax.axis('off')
        
        # Integreer matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Verbind klik event
        self.canvas.mpl_connect('button_press_event', self.on_image_click)
        
    def setup_control_buttons(self):
        # Button: Remove point
        self.remove_btn = ctk.CTkButton(
            self.control_frame,
            text="remove point",
            command=self.remove_point,
            height=40
        )
        self.remove_btn.pack(pady=5, padx=10, fill="x")
        
        # Button: Show the borders
        self.border_btn = ctk.CTkButton(
            self.control_frame,
            text="Show borders",
            command=self.show_border,
            height=40
        )
        self.border_btn.pack(pady=5, padx=10, fill="x")

        # Button: Save electrodes and next image 
        self.export_btn = ctk.CTkButton(
            self.control_frame,
            text="Export and Next",
            command=self.export_and_next,
            height=40
        )
        self.export_btn.pack(pady=5, padx=10, fill="x")
    
    def cut_image(self):
        h, w = self.image.shape
        cx = w//2 # Calculate center x
        cy = h//2 # Calculate center y 
        dx = w * self.image_cutoff # Calculate delta x
        dy = h * self.image_cutoff # Calculate delta y
        x_start = round(cx - dx)
        x_end = round(cx + dx)
        y_start = round(cy - dy)
        y_end = round(cy + dy)
        self.image = self.image[y_start:y_end, x_start:x_end]
        self.mask = np.zeros(self.image.shape + (4,))
        self.mask[:, :, 3] = 0  # Volledig transparant

    def reload_image(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        self.ax.imshow(self.mask)
        self.ax.set_title(f"file: {self.current_img}\n Klik elektrode linksboven", color = 'white')
        self.ax.axis('off')
        self.canvas.draw()
         
    def on_image_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.remove_point()
            x, y = int(event.xdata), int(event.ydata)
            self.point = (x, y)
            
            # Teken rode stip op de afbeelding
            rr, cc = draw.disk((y, x), radius=40, shape=self.image.shape)
            self.mask[rr, cc] = [1, 0, 0, 1]  # Rode stip            
            self.reload_image()
            
            # Update status
            self.status_label.configure(text=f"Punt {self.point})")
            
            print(f'Punt: ({x}, {y})')
    

    def remove_point(self):
        """Reset alle geselecteerde punten"""
        self.point = None
        self.mask = np.zeros(self.image.shape + (4,))

        self.reload_image()
        
        self.status_label.configure(text="Selectie gereset. Klik opnieuw.")
        
    def export(self):
        """Sla de geselecteerde punten op"""
        if not self.point:
            self.status_label.configure(text="Electrode niet geselecteerd!")
            return

        if not os.path.isdir(self.out_path):
            output_dir = f"{self.map_path}/output"
            os.makedirs(output_dir)
            
        try:
            if not self.electrodes or len(self.electrodes) == 0:
                self.calc_electrodes
            i = 0
            for row in range(1,5):
                for col in range(1,5):
                    electrode = self.electrodes[i]
                    print(f"splitting electrode: {electrode}")

                    xstart = electrode[0] - self.arm_length//2
                    xend = electrode[0] + self.arm_length//2
                    ystart = electrode[1] - self.arm_length//2
                    yend = electrode[1] + self.arm_length//2
                    
                    temp_image = self.image[ystart: yend, xstart: xend]

                    filename = f"{self.current_img.split('.')[0]}_r{5 - row}k{col}"
                    filepath = f"{self.out_path}/{filename}.jpeg"
                    #if row != 4 or col != 4:
                    io.imsave(filepath, temp_image)
                    print(f"File made: {filename}")
                    i += 1

        except Exception as e:
            self.status_label.configure(text=f"Fout bij opslaan: {str(e)}")

    def next(self):
        self.image_number += 1
        self.current_img = self.image_paths[self.image_number]
        self.image = io.imread((self.map_path + "/" + self.current_img))
        self.cut_image()
        self.remove_point()
        self.reload_image()
    
    def export_and_next(self):
        self.export()
        self.next()

    
    def calc_electrodes(self):
        """Bereken op welke punten de electroden zitten """ 
        self.electrodes = []
        for row in range(4):
            #x = self.point[0] + row * 244
            y = self.point[1] + row * 244
            for column in range(4):
                #y = self.point[1] + column * 244
                x = self.point[0] + column * 244
                self.electrodes.append((x,y))

    def show_border(self):
        """Laat randen zien van alle elektroden"""
        print("Showing borders")
        thickness = 3
        self.calc_electrodes()

        for electrode in self.electrodes:
            for i in range(thickness):
                start = (electrode[1] - self.arm_length//2 + i, electrode[0] - self.arm_length//2 + i)
                end = (electrode[1] + self.arm_length//2 - i, electrode[0] + self.arm_length//2 - i)

                rr, cc = draw.rectangle_perimeter(start, end, shape=self.image.shape)
                self.mask[rr, cc] = [1,0,0,1]
        self.reload_image()
    
    def on_close(self):
        plt.close(self.fig) # Remove figure
        self.image_frame.destroy()
        self.destroy()




    
if __name__ == "__main__":
    app = main()
    app.mainloop()