import os
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

"""
This algorithm can combined all seperate omni images within one folder, or even within one image.
"""

sources = ["D:/mea_data/2024_04_09_GLS_CTRL/20240409_OMNI/unprocessed",
           "D:/mea_data/2024_04_10_GLS_CTRL/20240410_OMNI/unprocessed",
           "D:/mea_data/2024_04_12_GLS_CTRL/20240412_OMNI/unprocessed",
           "D:/mea_data/2024_04_16_GLS_CTRL/20240416_OMNI/unprocessed"
           ]
outputmap = "D:/mea_data/alle_omni" 

def same_map(source, outputmap):
    # Loop through all submaps
    day = source.split("/")[2].split("_")[2]
    for submap in os.listdir(source):
        submap_path = os.path.join(source, submap)

        if os.path.isdir(submap_path): # Check if it is a map
            for image in os.listdir(submap_path): # Find image in map
                image_path = os.path.join(submap_path, image)
                # Check if image exists
                if os.path.isfile(image_path):
                    new_image_path = os.path.join(outputmap, f"{day}_{image}")
                    # Copy image
                    shutil.copy2(image_path, new_image_path)
                    print(f"✅ {image} copied to {new_image_path}")

for source in sources:
    same_map(source, outputmap)


def load_pics(source):
    names = []
    pics = []
    for submap in os.listdir(source): # Loop through submaps
        submap_path = os.path.join(source, submap)
        if os.path.isdir(submap_path): # Check if map exists

            for image in os.listdir(submap_path): # Find image in map
                names.append(image[0:2]) # Save image_name (Letter - Number) # Note: Does not work on plates with 10 or more cols
                image_path = os.path.join(submap_path, image)
                img = mpimg.imread(image_path)
                pics.append(img)
    return pics, names



def same_pic(pics, names, outputfile):
    
        
    # Get wellplate size based on names
    rows = []
    cols = []
    for name in names:
        rows.append((ord(name[0]))-64)
        cols.append(int(name[1]))
    row_amount = max(rows)
    col_amount = max(cols)
    print(f"rows: {row_amount}, cols: {col_amount}")
    # Create picture of all images combined

    fig, axs = plt.subplots(row_amount ,col_amount, figsize=(8, 6))
    axs = axs.ravel()
    fig.subplots_adjust(hspace = 0.009, wspace = 0.009)

    for i, ax in enumerate(axs):
        img = pics[i]
        well = ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.patch.set_facecolor("black")
    fig.savefig(outputfile, dpi=2000, bbox_inches='tight', pad_inches=0)
    print("Image saved!")


#pics, names = load_pics(source)
#same_pic(pics, names, "D:/mea_data/2024_04_16_GLS_CTRL/20240416_OMNI/compact.png")