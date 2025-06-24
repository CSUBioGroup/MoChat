import os

right_ans = []
with open ('lor_prompt_1182.txt', 'r') as file:
	lines = file.readlines()
	for line in lines:
		line = line.split("+++++")[-1]
		assert line.strip().lower() == 'left' or line.strip().lower() == 'right'
		right_ans.append(line.strip().lower())

with open ('acc_comp.txt', 'w') as f:
	for filen in os.listdir('answers'):
		path = os.path.join('answers', filen)
		num_right = 0
		with open (path, 'r') as ansf:
			for ind, ans in enumerate(ansf.readlines()):
				a = ans.strip().lower()
				if a != 'right' and a != 'left':
					if 'left' in a and 'right' not in a:
						a = 'left'
					elif 'right' in a and 'left' not in a:
						a = 'right'
					else:
						continue
				if a == right_ans[ind]:
					num_right +=1
		f.write(filen + "+++++" + str(num_right/1182) + "\n")

		
