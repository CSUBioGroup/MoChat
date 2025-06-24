import csv
import json

with open("output.txt", "r") as txt_f:
    with open("scores.csv", "w", newline='') as scores_f:
        csvwriter = csv.writer(scores_f)
        csvwriter.writerow(['q_id', 'score1', 'score2', 'score3'])
        for line in txt_f:
            data = json.loads(line)
            scores = data["score"]
            if len(scores) != 3:
                print(data['question_id'])
                continue
            score1, score2, score3= scores[0], scores[1], scores[2]
            csvwriter.writerow([data['question_id'], score1, score2, score3])

print('done!')

