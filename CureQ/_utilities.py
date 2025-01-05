import h5py
import os

def rechunk_dataset(fileadress, compression_method='lzf', compression_level=1, always_compress_files=False):
    """
    Rechunk an existing hdf5 dataset.

    Parameters
    ----------
    fileadress : str
        Path to the hdf5 file
    compression_method : {'lzf', 'gzip'}, optional
        Compression method
    compression_level : int, optional
        Compression level when using gzip - ranges 1-9
    always_compress_files: bool, optional
        If set to 'True', the algorithm will always perform the rechunking and compression, even when the data is already correctly chunked.

    Returns
    -------
    outputfile : str
        Path of the new file    

    Notes
    -----
    MCS hdf5 dataset are inefficiently chunked.
    Rechunking the dataset will allow for python to indiviually extract electrode data without having to read the entire dataset.
    Besides rechunking, this function will also apply a compression algorithm to the dataset.
    """

    outputfile=f"{fileadress[:-3]}_rechunked.h5"
    with h5py.File(fileadress, 'r') as src, h5py.File(outputfile, 'w') as dst:
        dataset_to_rechunk="Data/Recording_0/AnalogStream/Stream_0/ChannelData"
        original_chunks=src[dataset_to_rechunk].chunks
        if original_chunks:
            print("Dataset is chunked with chunk shape:", original_chunks)
        else:
            print("Dataset is contiguous.")
        
        new_chunks=(1, src[dataset_to_rechunk].shape[1])

        if original_chunks==new_chunks:
            print("Dataset is already correctly chunked")
            if not always_compress_files:
                return
        
        print(f"Rechunking dataset to shape: {new_chunks}, this will create a new file")

        def copy_attributes(src_obj, dst_obj):
            for key, value in src_obj.attrs.items():
                dst_obj.attrs[key] = value
        
        def copy_item(name, obj):
            parent_path = os.path.dirname(name)
            if parent_path and parent_path not in dst:
                dst.create_group(parent_path)
            
            if isinstance(obj, h5py.Dataset):
                if name == dataset_to_rechunk:
                    if compression_method=='lzf':
                        chunks = new_chunks
                        dst_dataset = dst.create_dataset(
                            name,
                            shape=obj.shape,
                            dtype=obj.dtype,
                            chunks=chunks,
                            compression=compression_method
                        )
                    elif compression_method=='gzip':
                        chunks = new_chunks
                        dst_dataset = dst.create_dataset(
                            name,
                            shape=obj.shape,
                            dtype=obj.dtype,
                            chunks=chunks,
                            compression=compression_method,
                            compression_opts=compression_level
                        )
                    else:
                        raise ValueError(f"{compression_method} is not a valid compression method")

                    dst_dataset[:] = obj[:]
                    
                else:
                    dst_dataset = dst.create_dataset(
                        name,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        compression=obj.compression,
                        compression_opts=obj.compression_opts,
                        shuffle=obj.shuffle
                    )
                    dst_dataset[:] = obj[:]
        
                copy_attributes(obj, dst_dataset)
                
            elif isinstance(obj, h5py.Group):
                dst_group = dst.create_group(name)
                copy_attributes(obj, dst_group)

        src.visititems(copy_item)
        original_size=os.stat(fileadress).st_size/(1024**3)
        new_size=os.stat(outputfile).st_size/(1024**3)
        print(f"Original size: {round(original_size, 2)} GB\nNew size: {round(new_size, 2)} GB")
        print(f"Rechunking and compression succesful")
        return outputfile