from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from dataset.save_cmd_dataset import system_prompt as cmd_system_prompt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../spm_gpt/finetune/Phi-4_unsloth_spmcmd/checkpoint-348",
    max_seq_length=1200,
    dtype=None,
    load_in_4bit=True,
)


user_msg = "move scan area to left 10um"

prompt_text = [
    {"role": "system", "content": cmd_system_prompt},
    {"role": "user", "content": user_msg},
]

inputs = tokenizer.apply_chat_template(
    prompt_text,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

generation_kwargs = dict(
    input_ids=inputs,
    streamer=streamer,
    max_new_tokens=1000,
    **{"temperature": 0, "do_sample": False},
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

generated_text = ""
for new_text in streamer:
    generated_text += new_text

print(generated_text)