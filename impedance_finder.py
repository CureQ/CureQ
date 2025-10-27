from heatmap_test import *

#%% Functions
def impedance_list(impedance, mapping, path):
    with open(path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        for index, imp in enumerate(impedance):

            wr = mapping.loc[index + 1, "WellRow"]
            wc = mapping.loc[index + 1, 'WellColumn']
            ec = mapping.loc[index + 1, 'ElectrodeColumn']
            er = mapping.loc[index + 1, 'ElectrodeRow']

            code = chr(64+wr) + str(wc) + str(ec) + str(er)
            spamwriter.writerow([code, imp])

#%% Code
path = "D:/mea_data/2025_44_dagen_iv/Bow_div44.h5"
impedance = load_impedance(path)
mapping = load_mapping(path)
impedance = calculate_modulus(impedance)

impedance_list(impedance, mapping, "output.csv")