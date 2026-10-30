import json
import copy
import re
from datasets import load_from_disk, load_dataset, Dataset

EOS_TOKEN = "<|eot_id|>"
system_prompt = ""
max_token_length = 2048


def save_to_jsonl(prompts, output_file="dataset.jsonl"):
    """Converts a list of prompt dictionaries to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            json.dump(prompt, f, ensure_ascii=False)
            f.write("\n")


def create_dialogue_dataset(json_dataset_path, save_path, data=None, is_append=False):
    text_list = []

    if data is None:
        with open(json_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    for key in data:
        for it in data[key]:

            if not is_append:
                lines = it.split("\n")

                # text = copy.deepcopy(default_prompt)
                text = None


                for line in lines:
                    if line.startswith("<user>:"):
                        text = [{"role": "user", "content": line.replace("<user>:", "")}]
                    if line.startswith("<assistant>:") and text:
                        text.append({"role": "assistant", "content": line.replace("<assistant>:", "")})
                        text_list.append(copy.deepcopy(text))

                        text = None
            else:
                pattern = r"<(user|assistant)>(.*?)(?=<(?:user|assistant)>|$)"
                text = []

                for role, message in re.findall(pattern, it, re.DOTALL):
                    text.append({"role": role, "content": message.strip()})

                    if role == "assistant":
                        # print(text)
                        text_list.append(copy.deepcopy(text))
                        text = []
            # print(text_list)
                # text_list.append(copy.deepcopy(text))

    dataset = Dataset.from_dict({"messages": text_list})
    dataset.save_to_disk(save_path)


def tokenize_function_llama(examples):
    texts = []
    for example in examples['messages']:
        if system_prompt == "":
            conversation = ""
        else:
            conversation = f"<|start_header_id|>system<|end_header_id|>{system_prompt}{EOS_TOKEN}"

        for message in example:
            if message['role'] == "user":
                conversation += f"<|start_header_id|>user<|end_header_id|>{message['content']}{EOS_TOKEN}"
            elif message['role'] == "assistant":
                conversation += f"<|start_header_id|>assistant<|end_header_id|>{message['content']}{EOS_TOKEN}"

        texts.append(conversation)
    return {"texts": texts}


def tokenize_function_llama_v2(examples, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []
    texts = []

    for example in examples['messages']:
        if system_prompt == "":
            conversation = ""
        else:
            conversation = f"<|start_header_id|>system<|end_header_id|>{system_prompt}{EOS_TOKEN}"

        for j, message in enumerate(example):
            if message['role'] == "user":
                conversation += f"<|start_header_id|>user<|end_header_id|>{message['content']}{EOS_TOKEN}"
            elif message['role'] == "assistant":
                conversation += f"<|start_header_id|>assistant<|end_header_id|>{message['content']}{EOS_TOKEN}"

        encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_token_length)
        # encoding['labels'] = [
        #     -100 if token == tokenizer.pad_token_id and i > 1 and encoding['input_ids'][
        #         i - 1] == tokenizer.eos_token_id else token for i, token in enumerate(encoding['input_ids'])
        # ]

        encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]
        # print(encoding["input_ids"])
        # print(encoding['labels'])
        # print("-------------")
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        texts.append(conversation)
        labels.append(encoding['labels'])

    # return {"texts": texts, "input_ids": labels, "attention_mask": attention_masks, "labels": labels}
    return {"input_ids": labels, "attention_mask": attention_masks, "labels": labels}
