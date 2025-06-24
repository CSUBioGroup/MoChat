import os

# read the missing inds
missing_ind = []
with open('fid_missing_ind.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        missing_ind.append(int(line.strip()))

# from 250 to 233
directory = './'

for txtf in os.listdir(directory):
    if txtf.startswith('time'):
        file_path = os.path.join(directory, txtf)

        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = [line for idx, line in enumerate(lines) if idx not in missing_ind]

        with open(file_path, 'w') as file:
            file.writelines(new_lines)


