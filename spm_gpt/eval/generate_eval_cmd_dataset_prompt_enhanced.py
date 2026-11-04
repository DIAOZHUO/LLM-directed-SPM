"""
Prompt-enhanced baselines of un-fine-tuned models on the command benchmark (Fig. 5 of the revised manuscript),
and the GPU energy / latency of each query, including the fine-tuned command SLM converted to GGUF.

The models are served by LM Studio through its OpenAI-compatible API. The models in MODEL_RUNS are evaluated one
after another; each model is loaded with the lms CLI (all other models are unloaded first), equivalent to:
    lms server start
    qwen : unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL)
           lms load qwen3.8-27b --identifier qwen3.8-27b --context-length 32768 --parallel 1 --gpu max
    phi4 : unsloth/phi-4-GGUF (Q4_K_M), same base model as the fine-tuned command SLM
           lms load phi-4 --identifier phi-4 --context-length 16384 --parallel 1 --gpu max
    phi4_ft : fine-tuned command SLM converted to GGUF (Q4_K_M), evaluated as in the paper (k = 0, prompt mode)
           lms load phi-4-spmcmd --identifier phi-4-spmcmd --context-length 16384 --parallel 1 --gpu max
           (LoRA adapter merged into the 16-bit base model with ../finetune/merge_lora_to_16bit.py, then converted
            with convert_hf_to_gguf.py and quantized to Q4_K_M with llama-quantize of llama.cpp)

Phi-4 has no thinking mode and its chat template has no native tool support, so in the function_calling mode
LM Studio falls back to its own prompt-based tool description.

Modes
    prompt            : same system prompt as the other local models / o4-mini (<cmd> text output)
    function_calling  : commands exposed as OpenAI tools, tool calls are converted back to <cmd> text
    structured        : JSON output constrained by a json_schema (response_format), converted back to <cmd> text
Each mode can be combined with k-shot examples drawn from the training set (k = 0/1/3/5/10/20, see run_list).
The k examples are the training samples most similar to the query (TF-IDF character n-grams), given as preceding
conversation turns in the format of each mode; they are identical for every model and recorded in raw_outputs.

Results: eval_results_cmd (Stage i) and eval_results_cmd_planningv2 (Stage ii), plotted by plot_prompt_enhanced_figure.py
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset.save_cmd_dataset import system_prompt, command_str_csv
from spm_gpt.eval.gpu_energy_meter import GPUEnergyMeter
from spm_gpt.eval.cmd_eval_util import parse_line, build_command_schema, compare_cmd_list, compare_cmd_list_unordered, add_data_to_json

schema = build_command_schema(command_str_csv)

base_url = "http://localhost:1235/v1"   # port printed by `lms server start`

lms_path = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms.exe")

# lms_key        : model key in LM Studio (`lms ls`)
# model_id       : identifier given by `lms load ... --identifier`
# context_length : context length used when the model is loaded
# model_name     : prefix of the result file names
# thinking       : evaluate in thinking mode
# has_thinking   : the model has a thinking switch (reasoning_effort)
MODEL_CONFIGS = {
    "qwen": dict(lms_key="qwen3.8-27b", model_id="qwen3.8-27b", context_length=32768,
                 model_name="Qwen3.8-27B", thinking=True, has_thinking=True),
    "phi4": dict(lms_key="phi-4", model_id="phi-4", context_length=16384,
                 model_name="Phi-4-GGUF", thinking=False, has_thinking=False),
    # fine-tuned command SLM: LoRA (checkpoint-348) merged into unsloth/phi-4 (16 bit), converted with llama.cpp b11602 (Q4_K_M)
    "phi4_ft": dict(lms_key="phi-4-spmcmd", model_id="phi-4-spmcmd", context_length=16384,
                    model_name="Phi-4-GGUF(fine-tuned)", thinking=False, has_thinking=False),
}

# model currently evaluated (set by select_model)
MODEL = "qwen"
model_id = MODEL_CONFIGS[MODEL]["model_id"]
model_name = MODEL_CONFIGS[MODEL]["model_name"]
seed = 0
max_attempts = 3


def select_model(key):
    global MODEL, model_id, model_name
    MODEL = key
    model_id = MODEL_CONFIGS[key]["model_id"]
    model_name = MODEL_CONFIGS[key]["model_name"]


def load_model(key):
    """unload all models and load the model with --parallel 1 (samples must not overlap for the energy measurement)"""
    cfg = MODEL_CONFIGS[key]
    subprocess.run([lms_path, "unload", "--all"], check=True)
    subprocess.run([lms_path, "load", cfg["lms_key"], "--identifier", cfg["model_id"],
                    "--context-length", str(cfg["context_length"]), "--parallel", "1", "--gpu", "max", "--yes"], check=True)
    select_model(key)


# ----------------------------
# Prompts for function calling / structured output.
# Constraints are identical to `system_prompt`; only the output format instructions differ.
# ----------------------------
_constraints = """The sample bias should not be changed unless there are instructions to change it.
The coordinate range accessible by the SPM tip and scan area on is -350 to 350 nm in the both x-direction and y-direction. The user command should NOT causes the probe to exceed the scanning area."""

fc_system_prompt = f"""
You are a robot controlling a scanning probe microscopy. Users will provide instructions in text format on how to control the device, and you'll need to carry out these instructions by calling the provided functions.
You can control and set the scan parameters in the scanning probe microscopy.
{_constraints}
If the user's instructions can be accomplished by multiple step commands, then call the functions in the order they should be executed.
If the user's instructions cannot be carried out by the provided functions alone, or the parameters are invalid, do NOT call any function. Instead, respond with 'None' first and then give a reason to user.
"""

structured_system_prompt = f"""
You are a robot controlling a scanning probe microscopy. Users will provide instructions in text format on how to control the device, and you'll need to translate these texts into specific programmatic commands.
You can control and set the scan parameters in the scanning probe microscopy.
Here is a list of commands in CSV format that you can invoke:\n
{str(command_str_csv.drop(columns=["callback"]))}\n
Respond with a JSON object with the fields "plan", "valid" and "commands".
"plan": a short explanation of how the instruction is carried out, or the reason why it cannot be carried out.
"valid": false if the user's instructions cannot be carried out by the commands provided above alone, or the parameters are invalid; otherwise true.
"commands": the commands to execute, in execution order, each given as {{"command": <name>, "arg": <value>}} ("arg" is omitted for commands without argument). Empty when "valid" is false.
{_constraints}
"""


def _json_type(arg_type):
    return {"float": "number", "int": "integer", "bool": "boolean"}[arg_type]


def build_tools():
    tools = []
    for _, row in command_str_csv.iterrows():
        name = row["programmatic commands"].split("(")[0].strip()
        if pd.notna(row["arg_type"]):
            parameters = {
                "type": "object",
                "properties": {"arg": {"type": _json_type(row["arg_type"].strip()), "description": row["arg_description"].strip()}},
                "required": ["arg"],
            }
        else:
            parameters = {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "function": {"name": name, "description": row["description"].strip(), "parameters": parameters},
        })
    return tools


def build_response_format():
    items = []
    for name, s in schema.items():
        item = {"type": "object", "properties": {"command": {"const": name}}, "required": ["command"], "additionalProperties": False}
        if s["arg_type"] is not None:
            item["properties"]["arg"] = {"type": _json_type(s["arg_type"])}
            item["required"].append("arg")
        items.append(item)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "spm_commands",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                    "valid": {"type": "boolean"},
                    "commands": {"type": "array", "items": {"anyOf": items}},
                },
                "required": ["plan", "valid", "commands"],
                "additionalProperties": False,
            },
        },
    }


tools = build_tools()
response_format = build_response_format()


# ----------------------------
# Conversion between <cmd> text and tool calls / JSON
# ----------------------------
def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def split_target(target_text):
    """target -> (is_none, plan_text, parsed commands)"""
    if target_text.startswith("None"):
        return True, re.sub(r"^None\.?\s*", "", target_text), []
    plan = target_text.split("<cmd>")[0].strip()
    return False, plan, parse_text(target_text)


def format_arg(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def commands_to_text(commands, plan=""):
    """[(name, arg or None)] -> <cmd> text that the evaluation parser understands"""
    lines = [f"{name}({'' if arg is None else format_arg(arg)})" for name, arg in commands]
    text = "<cmd>\n" + "\n".join(lines) + "\n</cmd>"
    return f"{plan}\n\n{text}" if plan else text


def target_to_fc_messages(target_text):
    is_none, plan, cmds = split_target(target_text)
    if is_none:
        return [{"role": "assistant", "content": target_text}]

    tool_calls = []
    for i, c in enumerate(cmds):
        args = {} if not c["args"] else {"arg": c["args"][0]}
        tool_calls.append({"id": f"call_{i}", "type": "function",
                           "function": {"name": c["method"], "arguments": json.dumps(args)}})
    messages = [{"role": "assistant", "content": plan or None, "tool_calls": tool_calls}]
    messages += [{"role": "tool", "tool_call_id": t["id"], "content": "OK"} for t in tool_calls]
    messages.append({"role": "assistant", "content": "Done."})
    return messages


def target_to_json(target_text):
    is_none, plan, cmds = split_target(target_text)
    commands = []
    for c in cmds:
        item = {"command": c["method"]}
        if c["args"]:
            item["arg"] = c["args"][0]
        commands.append(item)
    return json.dumps({"plan": plan, "valid": not is_none, "commands": commands}, ensure_ascii=False)


def json_to_text(content):
    try:
        d = json.loads(strip_think(content))
    except json.JSONDecodeError:
        return strip_think(content)
    if not d.get("valid", False):
        return f"None. {d.get('plan', '')}"
    return commands_to_text([(c["command"], c.get("arg")) for c in d.get("commands", [])], d.get("plan", ""))


# ----------------------------
# Few-shot example selection from the training set
# ----------------------------
class FewShotSelector:
    def __init__(self, train_dataset, selection="similar"):
        self.examples = [(m["messages"][0]["content"], m["messages"][1]["content"]) for m in train_dataset]
        self.selection = selection
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.train_matrix = self.vectorizer.fit_transform([q for q, _ in self.examples])

    def select_indices(self, query, k, index):
        """
        Indices of the k training examples for a test sample. The selection depends only on the query, the training
        set, k and the fixed seed, so every model receives exactly the same examples (recorded in raw_outputs).
          similar : TF-IDF similarity; stable sort, ties broken by the training-set order
          random  : fixed seed (seed + sample index)
        """
        if k == 0:
            return []
        if self.selection == "similar":
            sim = cosine_similarity(self.vectorizer.transform([query]), self.train_matrix)[0]
            idx = np.argsort(-sim, kind="stable")[:k][::-1]  # most similar example placed right before the query
        else:
            idx = np.random.default_rng(seed + index).choice(len(self.examples), size=k, replace=False)
        return [int(i) for i in idx]

    def select(self, query, k, index):
        return [self.examples[i] for i in self.select_indices(query, k, index)]


def build_messages(mode, query, shots):
    prompt = {"prompt": system_prompt, "function_calling": fc_system_prompt, "structured": structured_system_prompt}[mode]
    messages = [{"role": "system", "content": prompt}]
    for q, a in shots:
        messages.append({"role": "user", "content": q})
        if mode == "function_calling":
            messages += target_to_fc_messages(a)
        elif mode == "structured":
            messages.append({"role": "assistant", "content": target_to_json(a)})
        else:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    return messages


# ----------------------------
# Inference
# ----------------------------
def sampling_kwargs(thinking, attempt=0):
    # non-thinking: greedy decoding, same as the command SLM in the paper
    # thinking: sampling parameters recommended by Qwen for thinking mode, reasoning effort same as o4-mini
    # LM Studio ignores chat_template_kwargs; thinking is only switched off by reasoning_effort="none"
    # min_p / repeat_penalty are passed explicitly so that the LM Studio defaults are not applied
    # (a repeat penalty > 1 penalizes repeated tokens such as Aux1MaxVoltage(5) / Aux2MaxVoltage(5))
    if thinking:
        return dict(temperature=1.0, top_p=0.95, presence_penalty=0.0, seed=seed + attempt, max_tokens=16384,
                    reasoning_effort="high", extra_body={"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0})
    kwargs = dict(temperature=0, presence_penalty=0.0, seed=seed, max_tokens=1024,
                  extra_body={"min_p": 0.0, "repeat_penalty": 1.0})
    # models without a thinking switch (Phi-4) do not get the reasoning_effort parameter
    if MODEL_CONFIGS[MODEL]["has_thinking"]:
        kwargs["reasoning_effort"] = "none"
    return kwargs


def get_reasoning(message):
    extra = message.model_extra or {}
    return extra.get("reasoning_content") or extra.get("reasoning") or ""


def generate(client, mode, messages, thinking, max_tool_rounds=8, attempt=0):
    """returns (prediction text in <cmd> format, raw output, completion tokens)"""
    kwargs = sampling_kwargs(thinking, attempt)
    completion_tokens = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def add_usage(completion):
        usage["prompt_tokens"] += completion.usage.prompt_tokens
        usage["completion_tokens"] += completion.usage.completion_tokens
        return completion.usage.completion_tokens

    if mode == "structured":
        completion = client.chat.completions.create(model=model_id, messages=messages, response_format=response_format, **kwargs)
        content = completion.choices[0].message.content
        completion_tokens += add_usage(completion)
        return json_to_text(content), {"content": content, "reasoning": get_reasoning(completion.choices[0].message), "usage": usage}, completion_tokens

    if mode == "prompt":
        completion = client.chat.completions.create(model=model_id, messages=messages, **kwargs)
        content = completion.choices[0].message.content
        completion_tokens += add_usage(completion)
        return strip_think(content), {"content": content, "reasoning": get_reasoning(completion.choices[0].message), "usage": usage}, completion_tokens

    # function calling: tool results are mocked as "OK" until the model stops calling tools
    messages = list(messages)
    commands, contents, reasoning = [], [], []
    for _ in range(max_tool_rounds):
        completion = client.chat.completions.create(model=model_id, messages=messages, tools=tools, **kwargs)
        message = completion.choices[0].message
        completion_tokens += add_usage(completion)
        contents.append(strip_think(message.content))
        reasoning.append(get_reasoning(message))
        if not message.tool_calls:
            break

        for t in message.tool_calls:
            try:
                arg = json.loads(t.function.arguments or "{}").get("arg")
            except json.JSONDecodeError:
                arg = t.function.arguments
            commands.append((t.function.name, arg))
        messages.append({"role": "assistant", "content": message.content,
                         "tool_calls": [t.model_dump() for t in message.tool_calls]})
        messages += [{"role": "tool", "tool_call_id": t.id, "content": "OK"} for t in message.tool_calls]

    content = "\n".join(c for c in contents if c)
    prediction = commands_to_text(commands, content) if commands else content
    return prediction, {"content": contents, "tool_calls": commands, "reasoning": reasoning, "usage": usage}, completion_tokens


def get_lmstudio_eval_data(test_dataset, mode, k, selector, thinking, meter):
    client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=1800)

    questions, predictions, references, raw_outputs = [], [], [], []
    token_per_second_list = []

    meter.reset()
    meter.start()
    for i, example in enumerate(tqdm(test_dataset)):
        input_text = example["messages"][0]["content"]
        target_text = example["messages"][1]["content"]
        shot_indices = selector.select_indices(input_text, k, i)
        messages = build_messages(mode, input_text, [selector.examples[j] for j in shot_indices])

        # thinking mode occasionally ends the turn right after the reasoning with an empty answer;
        # such samples are re-sampled with a different seed (the number of attempts is recorded).
        # latency, energy and token counts include all attempts
        with meter.measure() as rec:
            rec["prompt_tokens"], rec["completion_tokens"] = 0, 0
            for attempt in range(max_attempts):
                try:
                    prediction, raw, completion_tokens = generate(client, mode, messages, thinking, attempt=attempt)
                except Exception as e:
                    prediction, raw, completion_tokens = "", {"error": repr(e)}, 0
                raw["num_attempts"] = attempt + 1
                rec["prompt_tokens"] += raw.get("usage", {}).get("prompt_tokens", 0)
                rec["completion_tokens"] += raw.get("usage", {}).get("completion_tokens", 0)
                if prediction.strip() != "" or not thinking:
                    break
        print(prediction)
        raw["few_shot_indices"] = shot_indices  # indices in the training set, identical for every model

        questions.append(input_text)
        predictions.append(prediction)
        references.append(target_text)
        raw_outputs.append(raw)
        token_per_second_list.append(rec["completion_tokens"] / rec["latency_s"] if rec["latency_s"] > 0 else 0)
    meter.stop()

    metrics = meter.summary()
    metrics["token_per_second"] = float(np.mean(token_per_second_list))
    return questions, predictions, references, raw_outputs, metrics


# ----------------------------
# Evaluation
# ----------------------------
def parse_text(text: str):
    text = strip_think(text)
    text = re.sub(r"</?cmd>", "", text).split("\n")
    parsed = []
    for it in text:
        l = parse_line(it, schema)
        if l is not None:
            parsed.append(l)
    return parsed


def compute_exact_match(predictions, references):
    """
    accuracy           : same metric as generate_eval_cmd_dataset_answer.py (order-aware, see compare_cmd_list)
    accuracy_unordered : previous metric (multiset of commands, order ignored)
    accuracy_strict    : accuracy, plus a 'None' reference requires an explicit 'None' answer
                         (an empty / unparsable prediction is not counted as a correct rejection)
    """
    assert len(predictions) == len(references)
    correct_flags, unordered_flags, strict_flags = [], [], []

    for pred, ref in zip(predictions, references):
        pred_norm = parse_text(pred)
        ref_norm = parse_text(ref)

        correct = compare_cmd_list(pred_norm, ref_norm)
        unordered_flags.append(compare_cmd_list_unordered(pred_norm, ref_norm))
        if ref.startswith("None"):
            strict = re.match(r"\s*None\b", strip_think(pred)) is not None and len(pred_norm) == 0
        else:
            strict = correct

        if not correct:
            print(pred)
            print(pred_norm)
            print(ref_norm)
            print("-" * 10)
        correct_flags.append(correct)
        strict_flags.append(strict)

    n = len(correct_flags)
    return {
        "accuracy": sum(correct_flags) / n if n else 0.0,
        "accuracy_unordered": sum(unordered_flags) / n if n else 0.0,
        "accuracy_strict": sum(strict_flags) / n if n else 0.0,
        "num_samples": n,
        "per_sample_correct": correct_flags,
        "per_sample_correct_unordered": unordered_flags,
        "per_sample_correct_strict": strict_flags,
        "gpu_usage": gpu_memory_used_mb(),
    }


def gpu_memory_used_mb():
    # total GPU memory in use (LM Studio runs in another process); includes other applications
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def get_run_name(mode, k, thinking, selection):
    name = model_name
    tags = []
    if mode != "prompt":
        tags.append(mode.replace("_", "-"))
    if k > 0:
        tags.append(f"{k}-shot" + ("" if selection == "similar" else f"-{selection}"))
    if thinking:
        tags.append("thinking")
    return f"{name}({', '.join(tags)})" if tags else name


def run(mode, k, _dataset, train_dataset, meter, save_dir="eval_results_cmd", thinking=False, selection="similar"):
    run_name = get_run_name(mode, k, thinking, selection)
    print(run_name, "dataset count:", len(_dataset))
    selector = FewShotSelector(train_dataset, selection)

    questions, predictions, references, raw_outputs, metrics = get_lmstudio_eval_data(_dataset, mode, k, selector, thinking, meter)
    metrics_dict = compute_exact_match(predictions, references)
    metrics_dict.update(metrics)

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

    result_dict = {
        "config": {"model": model_id, "mode": mode, "k": k, "selection": selection, "thinking": thinking,
                   "sampling": {key: v for key, v in sampling_kwargs(thinking).items()}},
        "questions": questions,
        "predictions": predictions,
        "ground truth": references,
        "raw_outputs": raw_outputs,
        "wrong_indices_0_based": wrong_indices,
        "wrong_cases": wrong_cases,
    }
    result_dict.update(metrics_dict)

    os.makedirs(save_dir, exist_ok=True)
    add_data_to_json(f"{save_dir}/{run_name}.json", result_dict)
    print(run_name, "accuracy:", metrics_dict["accuracy"], "strict:", metrics_dict["accuracy_strict"])


run_list = [
    # (mode, k)
    ("prompt", 0),
    ("prompt", 1),
    ("prompt", 3),
    ("prompt", 5),
    ("prompt", 10),
    ("prompt", 20),
    ("function_calling", 0),
    ("function_calling", 1),
    ("function_calling", 3),
    ("function_calling", 5),
    ("function_calling", 10),
    ("function_calling", 20),
    ("structured", 0),
    ("structured", 1),
    ("structured", 3),
    ("structured", 5),
    ("structured", 10),
    ("structured", 20),

]


if __name__ == '__main__':
    train_dataset = load_from_disk("../finetune/spmcmd_train_dataset")
    dataset = load_from_disk("../finetune/spmcmd_test_dataset")
    planning_dataset = load_from_disk("../finetune/spmcmd_planning_testv2_dataset")

    # models evaluated in this run, in this order, and the (mode, k) settings of each model
    MODEL_RUNS = {
        "phi4": run_list,
        "qwen": run_list,
        "phi4_ft": [("prompt", 0)],
    }

    # GPU energy is measured with the NVML energy counter; nothing else should run on the GPU
    meter = GPUEnergyMeter()

    for key, model_run_list in MODEL_RUNS.items():
        load_model(key)
        meter.measure_idle()  # idle power with this model loaded

        # Qwen is evaluated in thinking mode (reasoning_effort="high"), same as o4-mini; Phi-4 has no thinking mode
        thinking = MODEL_CONFIGS[key]["thinking"]
        for mode, k in model_run_list:
            run(mode, k, dataset, train_dataset, meter, save_dir="eval_results_cmd", thinking=thinking)
            run(mode, k, planning_dataset, train_dataset, meter, save_dir="eval_results_cmd_planningv2", thinking=thinking)
