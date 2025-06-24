from nlgeval import NLGEval

result_path = ''
with open(result_path) as f:
    cands = [line.strip() for line in f]

refs = []
for ref_path in ['ref0.txt', 'ref1.txt', 'ref2.txt']:
    with open(ref_path) as f:
        refs.append([line.strip() for line in f])

# 初始化 NLGEval 对象，忽略某些指标
nlg_eval = NLGEval(metrics_to_omit=[
    'EmbeddingAverageCosineSimilarity',
    'SkipThoughtCS',
    'VectorExtremaCosineSimilarity',
    'GreedyMatchingScore',
])

# 计算指标
metrics_dict = nlg_eval.compute_metrics(refs, cands)

print(result_path)
print(metrics_dict)



from bert_score import score
import torch

device = "cuda:0"

transposed_refs = list(map(list, zip(*refs))) #
P, R, F1 = score(cands, transposed_refs, 
                lang="en", 
                rescale_with_baseline=True,
                idf=True,
                device=device,
                verbose=True)

print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")
