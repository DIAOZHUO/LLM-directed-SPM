from enum import Enum
from dataset.save_dataset import system_prompt


class ModelType(Enum):
    Phi4 = "Phi4"
    Mistral = "Mistral"
    Llama = "Llama"



def tokenize_function(examples, *, tokenizer, model_type: ModelType, max_seq_length=4000):
    if model_type == ModelType.Phi4:
        return _phi4_tokenize_function(examples, tokenizer, max_seq_length)
    elif model_type == ModelType.Mistral:
        return _mistral_tokenize_function(examples, tokenizer, max_seq_length)
    elif model_type == ModelType.Llama:
        return _llama_tokenize_function(examples, tokenizer, max_seq_length)
    else:
        raise ValueError("Model type not supported")




def _phi4_tokenize_function(examples, tokenizer, max_seq_length=4000):
    input_ids = []
    attention_masks = []
    labels = []
    texts = []
    input_texts = []
    target_texts = []

    for example in examples['messages']:
        if system_prompt == "":
            conversation = ""
        else:
            # System prompt is included in the conversation but not in input_text
            conversation = f"<|im_start|>system<|im_sep|>{system_prompt}{tokenizer.eos_token}"

        # Track the actual conversation for input and output (no special tokens in input_text)
        user_input = ""
        assistant_output = ""

        for j, message in enumerate(example):
            if message['role'] == "user":
                user_input += f"{message['content']}{tokenizer.eos_token}"  # This is part of input_text
            elif message['role'] == "assistant":
                assistant_output = message['content']  # This is the model's target (output_text)

        # Now, conversation includes special tokens and is for model input
        conversation += user_input + f"<|im_start|>assistant<|im_sep|>{assistant_output}{tokenizer.eos_token}"

        # Set input_text as the user_input (excluding special tokens)
        input_text = user_input.strip()

        # Set target_text as the assistant's response (final message)
        target_text = assistant_output.strip()

        # Tokenize the entire conversation (with special tokens)
        encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_seq_length)
        encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        labels.append(encoding['labels'])
        texts.append(conversation)
        input_texts.append(input_text)  # Only include the user's message as input_text
        target_texts.append(target_text)  # Assistant's response as target_text

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "input_text": input_texts,  # Raw user input as input_text
        "target_text": target_texts  # Assistant's response as target_text
    }


# def _mistral_tokenize_function(examples, tokenizer, max_seq_length=4000):
#     input_ids = []
#     attention_masks = []
#     labels = []
#     texts = []
#     input_texts = []
#     target_texts = []
#
#     for example in examples['messages']:
#         # Track the actual conversation for input and output (no special tokens in input_text)
#         user_input = ""
#         assistant_output = ""
#
#         for j, message in enumerate(example):
#             if message['role'] == "user":
#                 user_input += f"{message['content']}"
#             elif message['role'] == "assistant":
#                 assistant_output = message['content']  # This is the model's target (output_text)
#
#         # Now, conversation includes special tokens and is for model input
#         conversation = (
#             "<s>[INST]\n"
#             f"{system_prompt}\n\n"
#             f"{user_input}\n"
#             "[/INST]\n"
#             f"{assistant_output}{tokenizer.eos_token}"
#         )
#
#         # Set input_text as the user_input (excluding special tokens)
#         input_text = user_input.strip()
#
#         # Set target_text as the assistant's response (final message)
#         target_text = assistant_output.strip()
#
#         # Tokenize the entire conversation (with special tokens)
#         encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_seq_length)
#         encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]
#
#         input_ids.append(encoding['input_ids'])
#         attention_masks.append(encoding['attention_mask'])
#         labels.append(encoding['labels'])
#         texts.append(conversation)
#         input_texts.append(input_text)  # Only include the user's message as input_text
#         target_texts.append(target_text)  # Assistant's response as target_text
#
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_masks,
#         "labels": labels,
#         "input_text": input_texts,  # Raw user input as input_text
#         "target_text": target_texts  # Assistant's response as target_text
#     }

def _mistral_tokenize_function(examples, tokenizer, max_seq_length=4000):
    input_ids = []
    attention_masks = []
    labels = []
    texts = []
    input_texts = []
    target_texts = []

    for example in examples['messages']:
        conversation = "<s>[INST]" + system_prompt

        # Track the actual conversation for input and output (no special tokens in input_text)
        user_input = ""
        assistant_output = ""

        for j, message in enumerate(example):
            if message['role'] == "user":
                user_input += f"{message['content']}[/INST]"
            elif message['role'] == "assistant":
                assistant_output = message['content']  # This is the model's target (output_text)

        # Now, conversation includes special tokens and is for model input
        conversation += user_input + f"{assistant_output}{tokenizer.eos_token}"

        # Set input_text as the user_input (excluding special tokens)
        input_text = user_input.strip()

        # Set target_text as the assistant's response (final message)
        target_text = assistant_output.strip()

        # Tokenize the entire conversation (with special tokens)
        encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_seq_length)
        encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        labels.append(encoding['labels'])
        texts.append(conversation)
        input_texts.append(input_text)  # Only include the user's message as input_text
        target_texts.append(target_text)  # Assistant's response as target_text

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "input_text": input_texts,  # Raw user input as input_text
        "target_text": target_texts  # Assistant's response as target_text
    }


def _llama_tokenize_function(examples, tokenizer, max_seq_length=4000):
    input_ids = []
    attention_masks = []
    labels = []
    texts = []
    input_texts = []
    target_texts = []

    for example in examples['messages']:
        if system_prompt == "":
            conversation = ""
        else:
            # System prompt is included in the conversation but not in input_text
            conversation = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                            f"{system_prompt}{tokenizer.eos_token}")

        # Track the actual conversation for input and output (no special tokens in input_text)
        user_input = ""
        assistant_output = ""

        for j, message in enumerate(example):
            if message['role'] == "user":
                user_input += f"{message['content']}{tokenizer.eos_token}"  # This is part of input_text
            elif message['role'] == "assistant":
                assistant_output = message['content']  # This is the model's target (output_text)

        # Now, conversation includes special tokens and is for model input
        conversation += user_input + (f"<|start_header_id|>assistant<|end_header_id|>"
                                      f"{assistant_output}{tokenizer.eos_token}")

        # Set input_text as the user_input (excluding special tokens)
        input_text = user_input.strip()

        # Set target_text as the assistant's response (final message)
        target_text = assistant_output.strip()

        # Tokenize the entire conversation (with special tokens)
        encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_seq_length)
        encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        labels.append(encoding['labels'])
        texts.append(conversation)
        input_texts.append(input_text)  # Only include the user's message as input_text
        target_texts.append(target_text)  # Assistant's response as target_text

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "input_text": input_texts,  # Raw user input as input_text
        "target_text": target_texts  # Assistant's response as target_text
    }





