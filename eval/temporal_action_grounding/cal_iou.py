import os
import random
def calculate_iou(range1, range2):
    # calculate the intersection length
    intersection_start = max(range1[0], range2[0])
    intersection_end = min(range1[1], range2[1])
    intersection_length = max(0, intersection_end - intersection_start)

    # calculate the union length
    union_start = min(range1[0], range2[0])
    union_end = max(range1[1], range2[1])
    union_length = union_end - union_start

    # calculate IOU
    iou = intersection_length / union_length if union_length > 0 else 0
    return iou

def read_ranges_from_file(file_path):
    ranges = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('+++++')
            start = float(parts[0])
            end = float(parts[1])
            ranges.append((start, end))
    return ranges

def calculate_iou_for_files(file1, file2, output_file):
    ranges1 = read_ranges_from_file(file1)
    ranges2 = read_ranges_from_file(file2)
    
    if len(ranges1) != len(ranges2):
        print("Error: Lines in two files don't match!", file1)
        return
    
    with open(output_file, 'w') as out_file:
        for range1, range2 in zip(ranges1, ranges2):
            iou = calculate_iou(range1, range2)
            out_file.write(f"{iou}\n")
            #print(f"IOU: {iou}")


file1 = 'answers/baseline.txt' # Your answer file
file2 = 'real_ans_233.txt'  # correct answer file
output_file = "iou_" + file1.split("/")[-1]
calculate_iou_for_files(file1, file2, output_file)

with open (output_file, 'r') as f:
    lines = f.readlines()
    num3, num5, num7 = 0, 0, 0
    for line in lines:
        score = float(line.strip())
        if score >= 0.3:
            num3 += 1
        if score >= 0.5:
            num5 += 1
        if score >= 0.7:
            num7 += 1
    print(num3 / len(lines))
    print(num5 / len(lines))
    print(num7 / len(lines))
