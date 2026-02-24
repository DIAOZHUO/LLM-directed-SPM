import numpy as np
import time
import torch
from tqdm import tqdm
from openai import OpenAI
from api_keys.openai import openai_api_key



def get_eval_data(model, tokenizer, test_dataset, system_prompt, max_new_tokens=None):
    questions = []
    predictions = []
    references = []
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    for example in tqdm(test_dataset):
        input_text = example["input_text"].replace(tokenizer.eos_token, "").replace("[/INST]", "")
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
        inputs_text = tokenizer.decode(inputs[0])

        generated_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_text = tokenizer.decode(generated_ids.sequences[0]).split(inputs_text)[-1].replace(" ", "")

        print(generated_text)

        questions.append(input_text)
        predictions.append(generated_text)
        references.append(target_text)

    return questions, predictions, references



def get_gpt_eval_data(test_dataset, system_prompt):
    client = OpenAI(
        api_key=openai_api_key
    )

    questions = []
    predictions = []
    references = []
    token_per_second_list = []

    for example in tqdm(test_dataset):
        input_text = example["input_text"] # input prompt
        target_text = example["target_text"]  # ground truth output

        prompt_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        print(prompt_text[1])
        start_time = time.time()

        completion = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            reasoning_effort="high",
        )

        end_time = time.time()
        generated_text = completion.choices[0].message.content

        tokens_per_second = completion.usage.completion_tokens / (end_time - start_time)


        print(generated_text)
        # print("perplexity_score", perplexity_score)

        questions.append(input_text)
        predictions.append(generated_text)
        references.append(target_text)
        token_per_second_list.append(tokens_per_second)

    metrics = {
        "token_per_second": np.mean(token_per_second_list),
    }

    return questions, predictions, references, metrics



