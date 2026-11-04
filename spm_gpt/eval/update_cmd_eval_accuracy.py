"""
Recompute the command accuracy of existing evaluation results with the order-aware compare_cmd_list
(revised exact-match criterion; the results of the first version were scored with the order ignored).

Only the metric keys are rewritten; predictions / ground truth are left untouched, so the script can be re-run.
    accuracy, per_sample_correct                      : order-aware (see compare_cmd_list in cmd_eval_util.py)
    accuracy_unordered, per_sample_correct_unordered  : previous metric (order ignored)
    accuracy_strict, per_sample_correct_strict        : only updated in files that already have it (Qwen baselines)
    wrong_indices_0_based, wrong_cases                : rebuilt from the order-aware result
"""
import glob
import json
import os
import re

from dataset.save_cmd_dataset import command_str_csv
from spm_gpt.eval.cmd_eval_util import parse_line, build_command_schema, compare_cmd_list, compare_cmd_list_unordered, add_data_to_json

schema = build_command_schema(command_str_csv)

folder_list = ["eval_results_cmd", "eval_results_cmd_planning", "eval_results_cmd_planningv2"]


def parse_text(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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


def is_explicit_none(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.match(r"\s*None\b", text) is not None


def update_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    predictions = d["predictions"]
    references = d["ground truth"]
    questions = d.get("questions")

    correct_flags, unordered_flags, strict_flags = [], [], []
    for pred, ref in zip(predictions, references):
        pred_norm = parse_text(pred)
        ref_norm = parse_text(ref)

        correct = compare_cmd_list(pred_norm, ref_norm)
        correct_flags.append(correct)
        unordered_flags.append(compare_cmd_list_unordered(pred_norm, ref_norm))
        if ref.startswith("None"):
            strict_flags.append(is_explicit_none(pred) and len(pred_norm) == 0)
        else:
            strict_flags.append(correct)

    n = len(correct_flags)
    wrong_indices = [i for i, c in enumerate(correct_flags) if not c]
    wrong_cases = []
    for idx in wrong_indices:
        case = {"index_0_based": idx, "index_1_based": idx + 1}
        if questions is not None:
            case["question"] = questions[idx]
        case["prediction"] = predictions[idx]
        case["ground_truth"] = references[idx]
        wrong_cases.append(case)

    updates = {
        "accuracy": sum(correct_flags) / n,
        "per_sample_correct": correct_flags,
        "accuracy_unordered": sum(unordered_flags) / n,
        "per_sample_correct_unordered": unordered_flags,
        "wrong_indices_0_based": wrong_indices,
        "wrong_cases": wrong_cases,
    }
    if "accuracy_strict" in d:
        updates["accuracy_strict"] = sum(strict_flags) / n
        updates["per_sample_correct_strict"] = strict_flags

    # on the first run, the stored accuracy (order ignored) must equal the recomputed unordered accuracy
    old_accuracy = d.get("accuracy")
    if "accuracy_unordered" not in d and old_accuracy is not None and abs(old_accuracy - updates["accuracy_unordered"]) > 1e-9:
        print(f"[warning] {path}: stored accuracy {old_accuracy:.4f} != recomputed unordered accuracy {updates['accuracy_unordered']:.4f}")

    add_data_to_json(path, updates)

    changed = [i for i in range(n) if correct_flags[i] != unordered_flags[i]]
    return sum(correct_flags), sum(unordered_flags), n, changed


if __name__ == '__main__':
    for folder in folder_list:
        print(f"\n===== {folder} =====")
        print(f"{'model':<40}{'unordered':>14}{'order-aware':>14}   changed (0-based)")
        for path in sorted(glob.glob(f"{folder}/*.json")):
            correct, unordered, n, changed = update_json(path)
            name = os.path.basename(path).replace(".json", "")
            print(f"{name:<40}{f'{unordered}/{n}':>8} {unordered / n:6.1%}{f'{correct}/{n}':>8} {correct / n:6.1%}   {changed}")
