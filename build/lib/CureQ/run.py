from mea_analysis_tool import MEA_GUI

if __name__ == "__main__":
    MEA_GUI()

# from CureQ.mea import analyse_wells
# from mea import analyse_wells
# from os import listdir
# from os.path import isfile, join
# import warnings
# warnings.filterwarnings("ignore")

# path="G:/Axion_vs_MCS"
# path="G:/MEADATA/tsc"
# files = [f for f in listdir(path) if isfile(join(path, f))]
# print(files)

# # All MCS measurements are 120s - probably 20000 Hz

# if __name__ == "__main__":
#     for i in range(len(files)):
#         filename=files[i]
#         filepath=f"{path}/{files[i]}"
#         if "20DIV.h5" in filename:
#             print(f"Analysing: {filename}")
#             try:
#                 analyse_wells(fileadress=filepath, hertz=12500, electrode_amnt=16, use_multiprocessing=False, exit_time_s=0.001)
#             except Exception as error:
#                 print(f"Failed {filename}")
#                 print(error)

# # from CureQ.mea import analyse_wells
# from mea import analyse_wells
# from os import listdir
# from os.path import isfile, join
# # import warnings
# # warnings.filterwarnings("ignore")

# if __name__ == "__main__":
#     filepath="H:/107-5629 raws joran/Raws/t35.h5"
#     filepath="G:/CureQ/Kleefstra-data/20170306_29.3_nbasal1_001_mwd.h5"
#     analyse_wells(fileadress=filepath, hertz=10000, electrode_amnt=12, use_multiprocessing=True, exit_time_s=0.001)