import os
import torch
import evaluate
import numpy as np
import math
import torch.nn.functional as F
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.metrics import GEval
from api_keys.openai import openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
_g_eval = GEval(
    name="dialogue_quality",
    criteria="Coherence — the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.INPUT],
)




bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def compute_perplexity(inputs, outputs, skip_token_ids=None):
    if skip_token_ids is None:
        skip_token_ids = []

    scores = outputs.scores
    probs = []
    for i, score_tensor in enumerate(scores):
        token_id = outputs.sequences[0][len(inputs[0]) + i]  # skip input prompt tokens

        if token_id in skip_token_ids:
            continue

        prob = torch.softmax(score_tensor, dim=-1)[0, token_id].item()
        prob = max(prob, 1e-12)
        probs.append(prob)

    if probs:
        avg_log_prob = sum(torch.log(torch.tensor(p)) for p in probs) / len(probs)
        perplexity = torch.exp(-avg_log_prob).item()
    else:
        perplexity = float("inf")


    return perplexity, probs



def compute_g_eval_metrics(questions_text_list, predictions_text_list, references_text_list, metrics_dict={}):
    g_eval_scores = []
    for q, pred, ref in zip(questions_text_list, predictions_text_list, references_text_list):
        tc = LLMTestCase(
            input=q,
            actual_output=pred,
            expected_output=ref
        )
        score = _g_eval.measure(test_case=tc)  # 0~1
        g_eval_scores.append(score)
    g_eval_mean = float(sum(g_eval_scores) / len(g_eval_scores))
    return metrics_dict | {
        "GEval Score": g_eval_mean,
    }



def compute_text_generation_metrics(predictions, references, metrics_dict={}):
    # BLEU Evaluation
    bleu_refs = [[ref] for ref in references]  # wrap each reference in a list
    bleu_result = bleu.compute(predictions=predictions, references=bleu_refs)
    print(f"BLEU Score: {bleu_result['bleu']:.4f}")

    # ROUGE Evaluation
    rouge_result = rouge.compute(predictions=predictions, references=references)
    # print(f"ROUGE-L Score: {rouge_result['rougeL']:.4f}")

    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    # print(f"Precision: {sum(results['precision']) / len(results['precision']):.4f}")
    # print(f"Recall:    {sum(results['recall']) / len(results['recall']):.4f}")
    # print(f"F1:        {sum(results['f1']) / len(results['f1']):.4f}")

    return metrics_dict | {
        "Bleu Score": bleu_result['bleu'],
        "ROUGE-L Score": rouge_result['rougeL'],
        "BERT Score(F1)": sum(results['f1']) / len(results['f1']),
    }





if __name__ == '__main__':
    print(compute_g_eval_metrics(["How old are you."], ["I'm 7."], ["Just turned 18."]))




