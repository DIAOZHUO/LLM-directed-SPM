import gc

from unsloth import FastLanguageModel
from dataset.save_dataset import system_prompt
from spm_gpt.tokenize_functions import tokenize_function, ModelType
from spm_gpt.eval.cmd_eval_util import add_data_to_json
from spm_gpt.eval.generate_util import get_gpt_eval_data
from datasets import load_from_disk
from util.val.metric import *
import time
import numpy as np
from tqdm import tqdm


max_seq_length = 2000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

default_setting = {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.0}



model_list = [
    # model name, model path, model type (for tokenizer), load in 4bit
    ("Phi-4", "unsloth/Phi-4-unsloth-bnb-4bit", ModelType.Phi4, False),
    ("Phi-4(quantization)", "unsloth/Phi-4-unsloth-bnb-4bit", ModelType.Phi4, True),
    ("Phi-4(fine-tuned)", "../finetune/Phi-4-unsloth-bnb-4bit_unsloth_results/checkpoint-3024", ModelType.Phi4, True),
    ("Phi-4(distilled)", "../finetune/Phi-4-unsloth-bnb-4bit_stage2_distill_unsloth_results/checkpoint-3024", ModelType.Phi4, True),

    ("Mistral-v0.3", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", ModelType.Mistral, False),
    ("Mistral-v0.3(quantization)", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", ModelType.Mistral, True),
    ("Mistral-v0.3(fine-tuned)", "../finetune/mistral-7b-instruct-v0.3-bnb-4bit_unsloth_results/checkpoint-3024", ModelType.Mistral, True),
    ("Mistral-v0.3(distilled)", "../finetune/mistral-7b-instruct-v0.3-bnb-4bit_distill_unsloth_results/checkpoint-3024", ModelType.Mistral, True),

    ("Llama-3.2", "unsloth/Llama-3.2-3B-Instruct", ModelType.Llama, False),
    ("Llama-3.2(quantization)", "unsloth/Llama-3.2-3B-Instruct", ModelType.Llama, True),
    ("Llama-3.2(fine-tuned)", "../finetune/Llama-3.2-3B-Instruct_unsloth_results/checkpoint-3024", ModelType.Llama, True),
    ("Llama-3.2(distilled)", "../finetune/Llama-3.2-3B-Instruct_distill_unsloth_results/checkpoint-3024", ModelType.Llama, True),
]


def get_eval_data(model, tokenizer, test_dataset):
    questions = []
    predictions = []
    references = []
    probs_list = []
    perplexity_list = []
    token_per_second_list = []
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    gpu_usage = int(torch.cuda.memory_allocated() / 1024 ** 2)  # MB

    for example in tqdm(test_dataset):
        input_text = example["input_text"].replace(tokenizer.eos_token, "")  # input prompt
        target_text = example["target_text"]  # ground truth output

        prompt_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        print(prompt_text)

        inputs = tokenizer.apply_chat_template(
            prompt_text,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")
        # print(inputs.dtype)
        inputs_text = tokenizer.decode(inputs[0])

        start_time = time.time()
        generated_ids = model.generate(input_ids=inputs, max_new_tokens=max_seq_length, use_cache=True,
                                       return_dict_in_generate=True,
                                       output_scores=True, **default_setting)
        end_time = time.time()
        # generated_text = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids.sequences[0]).split(inputs_text)[-1]

        tokens_per_second = len(generated_ids.sequences[0]) / (end_time - start_time)

        ppl, probs = compute_perplexity(inputs, generated_ids, skip_token_ids=[tokenizer.pad_token_id, tokenizer.eos_token_id])


        print(generated_text)
        # print("perplexity_score", perplexity_score)

        questions.append(input_text)
        predictions.append(generated_text)
        references.append(target_text)
        perplexity_list.append(ppl)
        probs_list.append(probs)
        token_per_second_list.append(tokens_per_second)

    metrics = {
        "mean perplexity": np.mean(perplexity_list),
        "probs_list": probs_list,
        "token_per_second": np.mean(token_per_second_list),
        "gpu_usage": gpu_usage,
    }

    return questions, predictions, references, metrics



def run(model_index):
    load_in_4bit = model_list[model_index][3]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_list[model_index][1],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit, device_map="cuda"
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    print("Load Finished")

    _dataset = load_from_disk("../finetune/spmknowledge_test_dataset")
    print("dataset count:", len(_dataset))
    tokenized_datasets = _dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer, "model_type": model_list[model_index][2]}, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

    # test_dataset = tokenized_datasets.select(range(5))
    test_dataset = tokenized_datasets
    # for it in test_dataset:
    #     print(it["input_text"])
    #     print(it["target_text"])

    questions, predictions, references, metrics_dict = get_eval_data(model, tokenizer=tokenizer, test_dataset=test_dataset)

    def safe_text(x):
        if x is None:
            return "EMPTY"
        if not isinstance(x, str):
            return str(x)
        if x.strip() == "":
            return "EMPTY"
        return x

    predictions = [safe_text(p) for p in predictions]
    references = [safe_text(r) for r in references]

    metrics_dict = compute_text_generation_metrics(predictions, references, metrics_dict)
    # define openai key in api_keys.openai.py
    metrics_dict = compute_g_eval_metrics(questions, predictions, references, metrics_dict)

    result_dict = {"predictions": predictions, "ground truth": references}
    result_dict.update(metrics_dict)
    add_data_to_json(f"eval_results/{model_list[model_index][0]}.json", result_dict)
    del model, tokenizer
    gc.collect()




def run_gpt():
    load_in_4bit = model_list[0][3]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_list[0][1],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit, device_map="cuda"
    )

    _dataset = load_from_disk("../finetune/spmknowledge_test_dataset")
    print("dataset count:", len(_dataset))
    tokenized_datasets = _dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer, "model_type": model_list[0][2]}, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

    # test_dataset = tokenized_datasets.select(range(5))
    test_dataset = tokenized_datasets
    # for it in test_dataset:
    #     print(it["input_text"])
    #     print(it["target_text"])

    questions, predictions, references, metrics_dict = get_gpt_eval_data(test_dataset=test_dataset, system_prompt=system_prompt)

    def safe_text(x):
        if x is None:
            return "EMPTY"
        if not isinstance(x, str):
            return str(x)
        if x.strip() == "":
            return "EMPTY"
        return x

    predictions = [safe_text(p) for p in predictions]
    references = [safe_text(r) for r in references]

    metrics_dict = compute_text_generation_metrics(predictions, references, metrics_dict)
    metrics_dict = compute_g_eval_metrics(questions, predictions, references, metrics_dict)

    result_dict = {"predictions": predictions, "ground truth": references}
    result_dict.update(metrics_dict)
    add_data_to_json(f"eval_results/gpt4.json", result_dict)
    gc.collect()




if __name__ == '__main__':
    # run(8)
    run_gpt()


    # for i in range(len(model_list)):
    #     run(i)



