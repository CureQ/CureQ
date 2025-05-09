---
layout: default
title: Convert Axion data
permalink: /axion
---

Data generated from an Axion Biosystems MEA will be saved in the .raw format. Unfortunately, it is not possible to open this format in Python. For the MEA-library to analyse these files, they must first be converted to the hdf5 format. The only way to do so, is to read the .raw files in MATLAB using the [AxionFileLoader](https://github.com/axionbio/AxionFileLoader). This repository must first be downloaded and added to the MATLAB path.

For the file conversion, a script is available in the [github repository](https://github.com/CureQ/CureQ/blob/main/raw_to_hdf5.m). This script will read the raw voltage data from the file, and save them in a new hdf5 file. After this, the hdf5 file can be analysed by MEAlytics. For this script to work, the [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) must be installed in MATLAB.

After the conversion has been completed, these files are not compressed yet, and might take up a lot of storage space. Hdf5 files can be compressed using the GUI (see [Compress/Rechunk files](compress_rechunk.html)) or using [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html)
