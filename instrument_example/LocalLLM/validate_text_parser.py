"""
Offline validation of the text parser (command_validator.py), without the instrument framework.

    1. reference answers : every command sequence of the training and test sets must be accepted
                           (answers starting with 'None' are rejections by the SLM itself and are only counted)
    2. unsafe sequences  : hand-written sequences that violate the command format, the numerical bounds,
                           or a state-dependent constraint must be rejected as a whole
    3. model outputs     : outputs of the evaluated models (spm_gpt/eval/eval_results_cmd*) that are accepted or
                           rejected by the parser, separately for correct and wrong outputs

The initial instrument state is the probe at the origin with a 10 x 10 nm scan size; the scan is running
if the first ScanEnabled command of the sequence stops it (e.g. "stop the scan, then ...").

Run from the repository root:
    python -m instrument_example.LocalLLM.validate_text_parser
"""
import glob
import json
import os
import re

import pandas as pd

from .cmd_eval_util import build_command_schema
from .command_validator import SPMState, extract_commands, validate_sequence

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
schema = build_command_schema(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spm_command.csv")))

dataset_list = [
    "dataset/cmd_train",
    "dataset/cmd_planning",
    "dataset/cmd_test",
    "dataset/cmd_planning_test",
    "dataset/cmd_planning_altered",
]
result_folder_list = [
    "spm_gpt/eval/eval_results_cmd",
    "spm_gpt/eval/eval_results_cmd_planningv2",
]

# (description, response of the SLM, initial state or None for the default); all must be rejected
unsafe_list = [
    ("unknown command", "<cmd>\nMoveTip(10)\n</cmd>", None),
    ("argument of the wrong type", "<cmd>\nScanEnabled(10 nm)\n</cmd>", None),
    ("absolute position out of range", "<cmd>\nStageOffset_X_Tube(400)\n</cmd>", None),
    ("relative move out of range from the current position",
     "<cmd>\nStageOffset_X_Tube_ADD(200)\n</cmd>", SPMState(x_nm=200.0, y_nm=0.0, size_x_nm=10.0, size_y_nm=10.0, scanning=False)),
    ("relative move out of range after a preceding move",
     "<cmd>\nStageOffset_Y_Tube(300)\nStageOffset_Y_Tube_ADD(100)\n</cmd>", None),
    ("scan area beyond the accessible range",
     "<cmd>\nStageOffset_X_Tube(340)\nAux1MaxVoltage(40)\nAux2MaxVoltage(40)\nScanEnabled(True)\n</cmd>", None),
    ("scan size not positive", "<cmd>\nAux1MaxVoltage(0)\nAux2MaxVoltage(0)\n</cmd>", None),
    ("scan speed not positive", "<cmd>\nScan_Speed(-10)\n</cmd>", None),
    ("scan started twice",
     "<cmd>\nScanEnabled(True)\n</cmd>", SPMState(x_nm=0.0, y_nm=0.0, size_x_nm=10.0, size_y_nm=10.0, scanning=True)),
    ("scan stopped while not scanning",
     "<cmd>\nScanEnabled(False)\n</cmd>", SPMState(x_nm=0.0, y_nm=0.0, size_x_nm=10.0, size_y_nm=10.0, scanning=False)),
    ("valid commands followed by an invalid one (no partial execution)",
     "<cmd>\nAux1MaxVoltage(10)\nAux2MaxVoltage(10)\nStageOffset_Y_Tube(-500)\nScanEnabled(True)\n</cmd>", None),
    ("rejection by the SLM", "None. Over the limit of the scan range.", None),
]


def default_state(commands):
    scan = [c.arg for c in commands if c.method == "ScanEnabled"]
    return SPMState(x_nm=0.0, y_nm=0.0, size_x_nm=10.0, size_y_nm=10.0, scanning=bool(scan) and scan[0] is False)


def is_slm_rejection(response):
    # same rule as extract_commands: a line starting with 'None' rejects the whole response
    return any(line.strip().startswith("None") for line in response.split("\n"))


def check(response, state=None):
    """(accepted, reason, is_slm_rejection)"""
    is_valid, reason, commands = extract_commands(response, schema)
    if not is_valid:
        return False, reason, is_slm_rejection(response)
    is_valid, reason = validate_sequence(state or default_state(commands), commands)
    return is_valid, reason, False


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(r'(assistant|user):\s*(.*?)(?=\n(?:assistant|user):|\Z)', text, re.DOTALL)
    user = [m.strip() for role, m in matches if role == "user"]
    assistant = [m.strip() for role, m in matches if role == "assistant"]
    return list(zip(user, assistant))


def validate_references():
    print("1. reference answers")
    n_failed = 0
    for path in dataset_list:
        accepted, slm_rejection, failed = 0, 0, []
        pairs = load_pairs(os.path.join(root, path))
        for query, answer in pairs:
            ok, reason, is_none = check(answer)
            if ok:
                accepted += 1
            elif is_none:
                slm_rejection += 1
            else:
                failed.append((query, reason))
        print(f"    {path:32s} {len(pairs):4d} samples: {accepted:4d} accepted, {slm_rejection:3d} 'None' answers, "
              f"{len(failed)} rejected")
        for query, reason in failed:
            print(f"        {query} -> {reason}")
        n_failed += len(failed)
    return n_failed == 0


def validate_unsafe():
    print("2. unsafe sequences")
    passed = True
    for description, response, state in unsafe_list:
        ok, reason, _ = check(response, state)
        passed &= not ok
        print(f"    [{'OK' if not ok else 'NOT REJECTED'}] {description}: {reason}")
    return passed


def validate_model_outputs():
    """
    Outputs that the benchmark scores as correct but that contain a 'None' line followed by commands
    (mainly un-fine-tuned models) are counted as 'None' here, because the parser does not execute them.
    """
    print("3. model outputs: executed / rejected by the SLM ('None') / rejected by the parser")
    for folder in result_folder_list:
        print(f"    {folder}")
        for path in sorted(glob.glob(os.path.join(root, folder, "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            count = {(c, r): 0 for c in (True, False) for r in ("executed", "none", "parser")}
            for prediction, correct in zip(d["predictions"], d["per_sample_correct"]):
                ok, _, slm_rejection = check(re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL))
                result = "executed" if ok else "none" if slm_rejection else "parser"
                count[(bool(correct), result)] += 1
            print(f"        {os.path.basename(path)[:-5]:50s}"
                  f" correct {count[(True, 'executed')]:3d} / {count[(True, 'none')]:3d} / {count[(True, 'parser')]:3d}"
                  f"    wrong {count[(False, 'executed')]:3d} / {count[(False, 'none')]:3d} / {count[(False, 'parser')]:3d}")


if __name__ == '__main__':
    references_ok = validate_references()
    unsafe_ok = validate_unsafe()
    validate_model_outputs()
    print("all reference answers accepted:", references_ok)
    print("all unsafe sequences rejected:", unsafe_ok)
