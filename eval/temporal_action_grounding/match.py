import re

# file paths
input_file_path = 'real_ans_full.txt'
output_file_path = 'real_ans_250.txt'

with open(input_file_path, 'r') as file:
    lines = file.readlines()

# re-match-pattern
pattern = re.compile(r"<frameid_(\d+)>")

output_lines = []
for ind,line in enumerate(lines):
    matches = re.findall(pattern, line)
    if len(matches) == 2:
        output_lines.append(f"{matches[0]}+++++{matches[1]}\n")
    else:
        print(ind)

with open(output_file_path, 'w') as file:
    file.writelines(output_lines)

