import random
import os

files = []

for file in os.listdir('answers'):
    if not file.startswith('ref'):
        files.append(file)

random.shuffle(files)

for ind, file_name in enumerate(files):
    print(f"answer{ind+1}_list = get_ans_list('answers/{file_name}')")

