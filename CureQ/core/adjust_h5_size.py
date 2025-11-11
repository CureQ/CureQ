import h5py
import numpy as np

input_file = "c:/Users/jesse/Documents/HvA/CureQ/Mea/Kleefstra/nbasal_10000_Hz/20170306_29.3_nbasal1_001_mwd.h5"
output_file = "c:/Users/jesse/Documents/HvA/CureQ/CureQ_GUI/CCB/MEA/Kleefstra/MEA_Kleefstra_reduced.h5"

# --------------------

# Know how dataset is called
with h5py.File(input_file, "r") as f:
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name}  -->  shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP: {name}")
    f.visititems(print_attrs)

# ------------------

source_dataset_path = "Data/Recording_0/AnalogStream/Stream_0/ChannelData"

# Definieer de rijen (kanalen) die je wilt behouden
# Pas deze aan als je andere indeling hebt
# Maak een lijst van alle rijen (0–143) en verwijder die van well 2 en 12
all_rows = np.arange(144)
remove_rows = np.r_[12:24, 132:144]  # well 2 en 12
rows_to_keep = np.setdiff1d(all_rows, remove_rows)

# Open het originele bestand in read-only modus
with h5py.File(input_file, "r") as f_in:
    # Maak een nieuw HDF5-bestand aan
    with h5py.File(output_file, "w") as f_out:

        # Kopieer alle top-level groepen en datasets uit het originele bestand
        for key in f_in.keys():
            f_in.copy(key, f_out)
        
        # De dataset openen die we willen inkorten
        dset_in = f_in[source_dataset_path]
        
        # Inkorten naar 120 rijen en 1.000.000 kolommen
        subset = dset_in[rows_to_keep, :1_000_000]
        
        # Oud dataset verwijderen uit het nieuwe bestand
        del f_out[source_dataset_path]
        
        # Nieuwe (verkorte) dataset aanmaken met dezelfde eigenschappen
        f_out.create_dataset(
            source_dataset_path,
            data=subset,
            dtype=dset_in.dtype,
            compression=dset_in.compression,  # behoud compressie indien aanwezig
            chunks=True
        )
        
        print(f"Verkort dataset opgeslagen naar {output_file}")
        print(f"Nieuwe shape: {subset.shape}")