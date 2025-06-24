##############################################
## From https://github.com/i-Eval/FairEval ###
##############################################

import argparse
import json
import os
import time
from openai import OpenAI


from tqdm import tqdm

MAX_API_RETRY = 10000
REQ_TIME_GAP = 0.5

parser = argparse.ArgumentParser()
# parser.add_argument("-q", "--question-file")
# parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
# parser.add_argument("-ma", "--ma-list", nargs="+", default=[])
# parser.add_argument('-o', '--output', help='Output file (defaults to stdout)')
parser.add_argument("-m", "--eval-model", default="gpt-4o")
parser.add_argument("-k", "--k", type=int, default=2)
parser.add_argument("-b", "--bpc", type=int, default=0)

args = parser.parse_args()

# if args.eval_model == "gpt-4o":
#     cost_per_promtp_token = 0.03 / 1000
#     cost_per_completion_token = 0.06 / 1000
# elif args.eval_model == "gpt-3.5-turbo-0301":
#     cost_per_promtp_token = 2/ 10**6
#     cost_per_completion_token = 2/ 10**6
# else:
#     raise ValueError("Invalid evaluator name")



api_key = "sk-"

def gen_prompt(ques, ma1, ma2, ma3, ass1, ass2, ass3):
    sys_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'

    prompt_template = """
    ## Questions
    {question}

    ## Manual Answers (correct answers)
    [The Start of the First Manual Answer]
    {ref_0_ans}
    [The End of the First Manual Answer]

    [The Start of the Second Manual Answer]
    {ref_1_ans}
    [The End of the Second Manual Answer]

    [The Start of the Third Manual Answer]
    {ref_2_ans}
    [The End of the Third Manual Answer]

    ## Assistants' Answers
    [The Start of the Assistant 1's Answer]
    {ass1}
    [The End of the Assistant 1's Answer]

    [The Start of the Assistant 2's Answer]
    {ass2}
    [The End of the Assistant 2's Answer]

    [The Start of the Assistant 3's Answer]
    {ass3}
    [The End of the Assistant 3's Answer]
    

    [System]

    {prompt}
    """

    default_prompt = """
    ### General
    We would like to request your feedback on the performance of three AI assistants in response_1st to the user question displayed above. What you need to do is only output the rates(1-10) of each answer.
    ### Guides
    First, read the question. The questions is about a series of motion about people, so you won't see the motion. Then, review the three manual answers provided by real people. These answers can be considered as the correct answers, or 10-point answers. You can combine them or choose one as the correct answer.
    Next, examine the three answers given by the 3 AI assistants: Assistant 1, Assistant 2, Assistant 3. Pay attention to whether each assistant's answer identifies the correct action, as indicated in the manual answers. They shouldn't have details that are even ignored in manual details. 
    Rate each assistant on a scale of 1 to 10, where a higher score indicates better overall performance. Ensure to avoid bias and that the order of responses does not affect your judgment.
	
    ### Output Format
    Remember to only output the evaluation score ,in the following json format. 
    [{"score1": "<Score of the first assistant>", "score2": "<Score of the second assistant>", "score3": "<Score of the third assistant>"}]
    DON'T OUTPUT ANY OTHER THINGS
    """


    return sys_prompt, prompt_template.format(question=ques, ref_0_ans=ma1, ref_1_ans=ma2, ref_2_ans=ma3, ass1=ass1, ass2=ass2, ass3= ass3, prompt=default_prompt)

def query_gpt(system_prompt, uer_prompt):
    client = OpenAI(
        base_url="",
        api_key=api_key
    )

    for i in range(MAX_API_RETRY):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": uer_prompt},
                ],
                temperature=1,
                max_tokens=100,
                n=args.k
            )
            return response
        # except RateLimitError:
        #     print('rate limit')
        #     time.sleep(30)
        except Exception as e:
            print('error')
    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")


def get_eval(i, idx, ques, ma1, ma2, ma3, ass1, ass2, ass3):
    system_prompt, user_prompt = gen_prompt(ques, ma1, ma2, ma3, ass1, ass2, ass3)
    response = query_gpt(system_prompt, user_prompt)
    all_scores = []
    contents = []
    contents_bpc = []
    print(len(response.choices))
    for choice in response.choices:
        with open(f'response/{i}-{idx}.txt', 'a', encoding='utf-8') as file:
            file.write(choice.message.content + '\n')
        content = choice.message.content
        score_3_list = parse_score_from_review(content)
        if any(score == -1 for score in score_3_list):
            continue
        all_scores.append(score_3_list)
        contents.append(content)


    
    # if args.bpc == 1:
    #     system_prompt, user_prompt_bpc = gen_prompt(ques, ans3, ans2, ans1)
    #     response_bpc = query_gpt(system_prompt, user_prompt_bpc)
    #     cost += response_bpc.usage.prompt_tokens * cost_per_promtp_token
    #     cost += response_bpc.usage.completion_tokens * cost_per_completion_token
    #     for choice in response_bpc.choices:
    #         content = choice.message.content
    #         score3, score2, score1 = parse_score_from_review(content)
    #         if score1 == -1 or score2 == -1 or score3 == -1:
    #             continue
    #         all_scores.append([score1, score2, score3])
    #         contents_bpc.append(content)
    try:
        average_scores = []
        for i in range(3):
            average_scores.append(sum([score[i] for score in all_scores]) / len(all_scores))

    except Exception as e:
        with open("err.txt", "a") as file:
            file.write(f'question: {ques}, str(e)')
    return contents, contents_bpc, cost, average_scores


def parse_score_from_review(review):
    try:
        review = review.strip()
        review = json.loads(review)
        scores = [float(int(review[0][f'score{i + 1}'])) for i in range(11)]

        return scores
    except:
        print(f'Failed to parse scores from {review}')
        return [-1] * 3

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = json.load(f)
    return json_list

def get_ans_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        ans_list = []
        for line in f:
            ans_list.append(line.strip())
    return ans_list


if __name__ == "__main__":
    
    question_jsons = get_json_list("evalchat_like.json")

    answer1_list = get_ans_list('answers/JGSE+CFT+Vicuna-13B_BMUDSDTGD.txt')
    answer2_list = get_ans_list('answers/Mochat-R.txt')
    answer3_list = get_ans_list('answers/GLTE+Vicuna-13B_BMUDSD.txt')


    ma1_list = get_ans_list("answers/ref0.txt")
    ma2_list = get_ans_list("answers/ref1.txt")
    ma3_list = get_ans_list("answers/ref2.txt")

    print(len(question_jsons))
    print(len(answer1_list))

    assert len(question_jsons) == len(answer1_list) == len(answer2_list) == len(answer3_list) == len(ma1_list) == len(ma2_list) == len(ma3_list)

    reviews = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))
    total_cost = 0

    with open(f"output.txt", "w") as output_review_file:
        for i in tqdm(question_idx_list):
            print(f"^^^^^^^^id: {question_jsons[i]['id']} ^^^^^^^^")
            ques = question_jsons[i]["conversations"][0]["value"].replace("\n", "").replace("<image>", "")

            anss = [answer1_list[i], answer2_list[i], answer3_list[i]]


            ma1 = ma1_list[i]
            ma2 = ma2_list[i]
            ma3 = ma3_list[i]
            idx = question_jsons[i]["id"]

            contents, contents_bpc, cost, scores = get_eval(i, idx, ques, ma1, ma2, ma3, *anss)
            results = {
                "question_id": question_jsons[i]["id"],
                "review": contents,
                "review_bpc": contents_bpc,
                "score": scores,
            }
            output_review_file.write(json.dumps(results) + "\n")



            time.sleep(REQ_TIME_GAP)



