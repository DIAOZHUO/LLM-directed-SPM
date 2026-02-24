import gc
import re

from unsloth import FastLanguageModel
from dataset.save_cmd_dataset import system_prompt, command_str_csv
from spm_gpt.tokenize_functions import tokenize_function, ModelType
from datasets import load_from_disk
# import evaluate
import torch
import os, json

from spm_gpt.eval.cmd_eval_util import parse_line, build_command_schema, compare_cmd_list, add_data_to_json
from spm_gpt.eval.generate_util import get_eval_data, get_gpt_eval_data
schema = build_command_schema(command_str_csv)

max_seq_length = 1200  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
model_list = [
    ("Phi-4", "unsloth/Phi-4-unsloth-bnb-4bit", ModelType.Phi4, False),
    ("Phi-4(quantization)", "unsloth/Phi-4-unsloth-bnb-4bit", ModelType.Phi4, True),
    ("Phi-4(fine-tuned)", "../finetune/Phi-4_unsloth_spmcmd/checkpoint-348", ModelType.Phi4, True),
    ("Mistral-v0.3", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", ModelType.Mistral, False),
    ("Mistral-v0.3(quantization)", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", ModelType.Mistral, True),
    ("Mistral-v0.3(fine-tuned)", "../finetune/mistral_unsloth_spmcmd/checkpoint-348", ModelType.Mistral, True),
    ("Llama-3.2", "unsloth/Llama-3.2-3B-Instruct", ModelType.Llama, False),
    ("Llama-3.2(quantization)", "unsloth/Llama-3.2-3B-Instruct", ModelType.Llama, True),
    ("LLama-3.2(fine-tuned)", "../finetune/LLAMA_unsloth_spmcmd/checkpoint-348", ModelType.Llama, True),
]



def parse_text(text: str) :
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("</s>", "")
    text = text.replace("<|endoftext|>", "")

    text = re.sub(r"</?cmd>", "", text).split("\n")
    parsed = []
    for it in text:
        l = parse_line(it, schema)
        if l is not None:
            parsed.append(l)
    return parsed




def compute_exact_match(predictions, references):
    assert len(predictions) == len(references)
    correct_flags = []

    for pred, ref in zip(predictions, references):

        pred_norm = parse_text(pred)
        ref_norm = parse_text(ref)


        correct = compare_cmd_list(pred_norm, ref_norm)
        if not correct:
            print(pred)
            print(pred_norm)
            print(ref_norm)
            print("-"*10)
        correct_flags.append(correct)

    accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    metrics = {
        "accuracy": accuracy,
        "num_samples": len(correct_flags),
        "per_sample_correct": correct_flags,
        "gpu_usage": int(torch.cuda.memory_allocated() / 1024 ** 2)  # MB
    }
    return metrics





def run(model_index, _dataset, save_dir="eval_results_cmd"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_list[model_index][1],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=model_list[model_index][3],
        device_map="cuda",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    print("Load Finished")

    print("dataset count:", len(_dataset))
    tokenized_datasets = _dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "model_type": model_list[model_index][2]},
        batched=True,
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

    test_dataset = tokenized_datasets

    # test_dataset = tokenized_datasets.select(range(5))

    questions, predictions, references = get_eval_data(
        model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        system_prompt=system_prompt,
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
        "wrong_indices_0_based": wrong_indices,      #
        "wrong_cases": wrong_cases,                  #
    }
    result_dict.update(metrics_dict)

    os.makedirs(save_dir, exist_ok=True)
    add_data_to_json(
        f"{save_dir}/{model_list[model_index][0]}.json",
        result_dict,
    )

    del model, tokenizer
    gc.collect()



def run_gpt(_dataset, inference=True, save_dir="eval_results_cmd"):
    if inference == True:
        load_in_4bit = model_list[0][3]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_list[0][1],
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit, device_map="cuda"
        )

        print("dataset count:", len(_dataset))
        tokenized_datasets = _dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer, "model_type": model_list[0][2]}, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

        # test_dataset = tokenized_datasets.select(range(5))
        test_dataset = tokenized_datasets
        # for it in test_dataset:
        #     print(it["input_text"])
        #     print(it["target_text"])
        questions, predictions, references, metrics_dict = get_gpt_eval_data(test_dataset=test_dataset, system_prompt=system_prompt)
    else:
        with open(f"{save_dir}/gpt4.json", encoding='utf-8') as f:
            dict = json.loads(f.read())
            print(dict.keys())
            predictions = dict["predictions"]
            references = dict["ground truth"]



    metrics_dict = compute_exact_match(predictions, references)

    #  間違えたサンプルだけを抜き出す処理を追加
    per_sample = metrics_dict["per_sample_correct"]
    wrong_indices = [i for i, c in enumerate(per_sample) if c == 0]

    wrong_cases = []
    for idx in wrong_indices:
        wrong_cases.append({
            "index_0_based": idx,
            "index_1_based": idx + 1,
            # "question": questions[idx],
            "prediction": predictions[idx],
            "ground_truth": references[idx],
        })
    #
    result_dict = {
        "predictions": predictions,
        "ground truth": references,
        "wrong_indices_0_based": wrong_indices,  #
        "wrong_cases": wrong_cases,  #
    }
    result_dict.update(metrics_dict)

    os.makedirs(save_dir, exist_ok=True)
    add_data_to_json(
        f"{save_dir}/gpt4.json",
        result_dict,
    )
    if inference:
        del model, tokenizer
        gc.collect()


if __name__ == '__main__':
    dataset = load_from_disk("../finetune/spmcmd_test_dataset")
    for it in range(len(model_list)):
        run(it, dataset)

    # run_gpt(inference=True, _dataset=dataset)

