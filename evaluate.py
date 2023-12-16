from metrics import corpus_bleu
from rouge_score import rouge_scorer
import sacrebleu
import json

# refs = []
predictions = []
with open('data/test.sug') as fp:
    refs = fp.readlines()
# with open('fix-data/test.pre') as fp:
#     predictions = fp.readlines()
with open('rl4lm_exps/rl4lm_experiment/epoch_50_test_split_predictions.json') as f:
    results = json.load(f)
    for res in results:
        # refs.append(res["ref_text"].replace("<START-1>", "").replace("<END-1>", ""))
        predictions.append(res["generated_text"])
# with open('../../Baselines/ChatGPT/chat.out') as fp:
#     predictions = fp.readlines()

for r, p in zip(refs, predictions):
    print(r)
    print(p)
    print('='*100)
hyps = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_1, rouge_2, rouge_l = [], [], []
for i, ref in enumerate(refs):
    hyp = predictions[i]
    scores = scorer.score(ref, hyp)
    rouge_1.append(scores['rouge1'].fmeasure)
    rouge_2.append(scores['rouge2'].fmeasure)
    rouge_l.append(scores['rougeL'].fmeasure)
    hyps.append(hyp)
bleu = sacrebleu.corpus_bleu(predictions, [refs])
print(f"BLEU score: {bleu}")
print(corpus_bleu(hyps, refs))
print(sum(rouge_1)/len(rouge_1), sum(rouge_2)/len(rouge_2), sum(rouge_l)/len(rouge_l))
