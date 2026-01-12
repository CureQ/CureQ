# Imports
import csv

# Setup
paths = [
    "D:/mea_data/cellcountvalidation/counted_09.csv",
    "D:/mea_data/cellcountvalidation/counted_10.csv",
    "D:/mea_data/cellcountvalidation/counted_12.csv",
    "D:/mea_data/cellcountvalidation/counted_16.csv"
]

# Code

# Using a dict, to make sure no duplicates are saved
combined = {}
for path in paths:

    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        # Add rows to combined dict
        for row in spamreader:
            electrode, value = row[0].split(",")
            combined[electrode] = value

# Turning dict back into list
output_list = []
for electrode in combined.keys():
    value = combined[electrode]
    output_list.append(f"{electrode},{value}")


print(output_list)
with open("D:/mea_data/cellcountvalidation/counted_cells.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for row in output_list:
        writer.writerow([row])