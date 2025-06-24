import os
import json

with open ('choose_side_test.json', 'r') as jsf:
	data = json.load(jsf)

missing_ids = []
with open ('side_missing_ids.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		missing_ids.append(line.strip())

with open ('lr_missing_ind.txt', 'a') as file:
	for ind, item in enumerate(data):
		if item['id'] in missing_ids or item['id'].startswith('M'):
			file.write(str(ind)+"\n")
		
