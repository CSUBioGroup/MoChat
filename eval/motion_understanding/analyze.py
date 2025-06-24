import csv
import numpy as np

csv_file_path = 'scores.csv'
scores_list = [[] for _ in range(4)]  # create a list

with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # head
    for row in csvreader:
        for i in range(4):
            scores_list[i].append(float(row[i+1]))  # add scores

# convert list to numpy array
scores_array = [np.array(scores) for scores in scores_list]

def compute_stats(scores):
    return {
        'mean': np.mean(scores),
        'max': np.max(scores),
        'min': np.min(scores)
    }

# calculate
stats_list = [compute_stats(scores) for scores in scores_array]

# output
for i, stats in enumerate(stats_list):
    print(f"Stats for Score {i+1}:")
    print(stats)
