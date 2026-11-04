import os

from unsloth import FastLanguageModel
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

os.makedirs("LLAMA_unsloth_spmcmd", exist_ok=True)
logging.basicConfig(
    filename="LLAMA_unsloth_spmcmd/training_loss.log",
    filemode='w',  # Overwrite log file each run. Use 'a' to append.
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from dataset.save_cmd_dataset import system_prompt

max_seq_length = 1200  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
model_name = "./Llama-3.2-3B-Instruct_distill_unsloth_results/checkpoint-3024"
# model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# from unsloth import add_new_tokens
# add_new_tokens(model, tokenizer, new_tokens=['<cmd>', '</cmd>'])


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
                     #"lm_head",
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

# special_tokens_dict = {'additional_special_tokens': ['<cmd>', '</cmd>']}
# num_added = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))
print(tokenizer.additional_special_tokens)
print(tokenizer.eos_token, tokenizer.pad_token)




class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")




def tokenize_function(examples, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []
    texts = []

    for example in examples['messages']:
        if system_prompt == "":
            conversation = ""
        else:
            conversation = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}{tokenizer.eos_token}"

        for j, message in enumerate(example):
            if message['role'] == "user":
                conversation += f"<|start_header_id|>user<|end_header_id|>{message['content']}{tokenizer.eos_token}"
            elif message['role'] == "assistant":
                conversation += f"<|start_header_id|>assistant<|end_header_id|>{message['content']}{tokenizer.eos_token}"

        encoding = tokenizer(conversation, padding='max_length', truncation=True, max_length=max_seq_length)
        encoding['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in encoding['input_ids']]

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        texts.append(conversation)
        labels.append(encoding['labels'])

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


_dataset = load_from_disk("./spmcmd_train_dataset")
print("dataset count:", len(_dataset))
tokenized_datasets = _dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["messages"])

# for it in tokenized_datasets:
#     print(tokenizer.decode(it["input_ids"]))

# tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.01, seed=42)
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
        # per_device_eval_batch_size=2,
        # do_eval=True,
        warmup_steps=40,
        num_train_epochs=6,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"LLAMA_unsloth_spmcmd",
        report_to="none",  # Use this for WandB etc
    ),
    callbacks=[LossLoggerCallback()],
)

trainer.train()
# tokenizer.save_pretrained(trainer.args.output_dir)