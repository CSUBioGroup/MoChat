import re
import os

def extract_rounded_integers_from_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            # from format like [[ 0.0, 1.0 ]]
            match = re.search(r'\[\[(\d+\.\d+), (\d+\.\d+)\]\]', line)
            if match:
                num1 = round(float(match.group(1)))
                num2 = round(float(match.group(2)))
                file.write(f"{num1}+++++{num2}\n")


input_file = 'answers/MoR.txt'
output_file = 'answers/MoR.txt'
extract_rounded_integers_from_file(input_file, output_file)
