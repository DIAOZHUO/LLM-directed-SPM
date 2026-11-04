import gc
from unsloth import FastLanguageModel
from dataset.save_router_dataset import system_prompt
from spm_gpt.tokenize_functions import tokenize_function, ModelType
from spm_gpt.eval.cmd_eval_util import add_data_to_json
from spm_gpt.eval.generate_util import get_eval_data
from datasets import load_from_disk
import os


model_list = [
    ("Phi-4", "unsloth/Phi-4-unsloth-bnb-4bit", ModelType.Phi4),
    ("Mistral-v0.3", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", ModelType.Mistral),
    ("Llama-3.2", "unsloth/Llama-3.2-3B-Instruct", ModelType.Llama),
]


def compute_exact_match(predictions, references):
    assert len(predictions) == len(references)
    correct_flags = []

    for pred, ref in zip(predictions, references):
        correct = int(pred == ref)
        correct_flags.append(correct)

    accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    metrics = {
        "exact_match_accuracy": accuracy,
        "num_samples": len(correct_flags),
        "per_sample_correct": correct_flags,
    }
    return metrics





def run(model_index):
    max_seq_length = 5000  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_list[model_index][1],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="cuda",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    print("Load Finished")

    _dataset = load_from_disk("../finetune/spmrouter_test_dataset")
    # random choose

    print("dataset count:", len(_dataset))
    tokenized_datasets = _dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "model_type": model_list[model_index][2]},
        batched=True,
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

    test_dataset = tokenized_datasets

    questions, predictions, references = get_eval_data(
        model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        system_prompt=system_prompt,
        max_new_tokens=1
    )
    metrics_dict = compute_exact_match(predictions, references)

    #  間違えたサンプルだけを抜き出す処理を追加
    per_sample = metrics_dict["per_sample_correct"]
    wrong_indices = [i for i, c in enumerate(per_sample) if c == 0]

    wrong_cases = []
    for idx in wrong_indices:
        wrong_cases.append({
            "index_0_based": idx,
            "index_1_based": idx + 1,
            "question": questions[idx],
            "prediction": predictions[idx],
            "ground_truth": references[idx],
        })
    #
    result_dict = {
        "questions": questions,
        "predictions": predictions,
        "ground truth": references,
        "wrong_indices_0_based": wrong_indices,  #
        "wrong_cases": wrong_cases,  #
    }
    result_dict.update(metrics_dict)

    os.makedirs("eval_results_router", exist_ok=True)
    add_data_to_json(
        f"eval_results_router/{model_list[model_index][0]}.json",
        result_dict,
    )

    del model, tokenizer
    gc.collect()


if __name__ == '__main__':
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    for it in range(len(model_list)):
        run(it)
