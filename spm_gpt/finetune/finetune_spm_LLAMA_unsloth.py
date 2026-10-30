from unsloth import FastLanguageModel
import torch
# import evaluate

# metric_perplexity = evaluate.load("perplexity")
# metric_glue = evaluate.load("glue", "mrpc")

from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from dataset.save_dataset import system_prompt

max_seq_length = 4000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print(tokenizer.eos_token, tokenizer.pad_token)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    # "embed_tokens",
                    ],
    # modules_to_save=[],
    lora_alpha=64,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

def tokenize_function(examples, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []

    for example in examples['messages']:
        conversation_ids = []
        label_ids = []

        # 处理 system prompt
        if system_prompt != "":
            system_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}{tokenizer.eos_token}"
            system_tokens = tokenizer(system_text, add_special_tokens=False)['input_ids']
            conversation_ids.extend(system_tokens)
            label_ids.extend([-100] * len(system_tokens))  # 掩码 system

        # 处理每轮对话
        for message in example:
            if message['role'] == "user":
                user_text = f"<|start_header_id|>user<|end_header_id|>{message['content']}{tokenizer.eos_token}"
                user_tokens = tokenizer(user_text, add_special_tokens=False)['input_ids']
                conversation_ids.extend(user_tokens)
                label_ids.extend([-100] * len(user_tokens))  # 掩码 user 输入

            elif message['role'] == "assistant":
                # 分别处理 prompt 部分和回复内容
                assistant_prefix = f"<|start_header_id|>assistant<|end_header_id|>"
                prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)['input_ids']
                conversation_ids.extend(prefix_tokens)
                label_ids.extend([-100] * len(prefix_tokens))  # 掩码 assistant 前缀

                # 只有这部分参与 loss 计算
                content_with_eos = f"{message['content']}{tokenizer.eos_token}"
                content_tokens = tokenizer(content_with_eos, add_special_tokens=False)['input_ids']
                conversation_ids.extend(content_tokens)
                label_ids.extend(content_tokens)  # 不掩码,参与训练

        # 截断或填充
        if len(conversation_ids) > max_seq_length:
            # raise ValueError(f"Too many tokens. requires {len(conversation_ids)} max tokens length")
            print(f"cut tokens to {max_seq_length}. requires {len(conversation_ids)} max tokens length")
            conversation_ids = conversation_ids[:max_seq_length]
            label_ids = label_ids[:max_seq_length]

        else:
            padding_length = max_seq_length - len(conversation_ids)
            conversation_ids.extend([tokenizer.pad_token_id] * padding_length)
            label_ids.extend([-100] * padding_length)

        # 创建 attention mask
        attention_mask = [1 if token != tokenizer.pad_token_id else 0
                          for token in conversation_ids]

        input_ids.append(conversation_ids)
        attention_masks.append(attention_mask)
        labels.append(label_ids)
    # print(len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }



_dataset = load_from_disk("./spmknowledge_dataset")
# _dataset = load_from_disk("./spmknowledge_distill_dataset")
print("dataset count:", len(_dataset))

tokenized_datasets = _dataset.map(
    tokenize_function,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    remove_columns=_dataset.column_names,  # 移除所有原始列
    desc="Tokenizing"
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets,
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        save_strategy="steps",
        # eval_strategy='steps',
        save_steps=200,
        eval_steps=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # per_device_eval_batch_size=2,
        # do_eval=True,
        # warmup_ratio=0.03,
        num_train_epochs=8,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"{model_name.split('/')[-1]}_unsloth_results",
        report_to="none",  # Use this for WandB etc
    ),
)

trainer.train()



