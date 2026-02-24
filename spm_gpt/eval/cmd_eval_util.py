import pandas as pd
import re
from collections import Counter
import json
import os




pattern = re.compile(r"""
    ^\s*
    (?P<method>\w+)      
    \s*
    \(
        (?P<args>[^)]*)  
    \)
    \s*$
""", re.VERBOSE)


def parse_line(line: str, schema: dict):
    if line.strip() == "":
        return None

    match = pattern.match(line)
    if not match:
        return None

    method = match.group("method")
    raw_args = match.group("args").strip()

    if method not in schema:
        return None

    arg_type = schema[method]["arg_type"]

    try:
        if raw_args == "":
            raw_arg_list = []
            norm_args = []
        else:
            raw_arg_list = [a.strip() for a in raw_args.split(",")]
            norm_args = [
                normalize_arg(a, arg_type) for a in raw_arg_list
            ]

        return {
            "method": method,
            "raw_args": raw_arg_list,
            "args": norm_args,
            "arg_type": arg_type
        }
    except Exception:
        return None



def build_command_schema(df: pd.DataFrame):
    schema = {}

    for _, row in df.iterrows():
        # e.g. "ScanEnabled(arg)" → "ScanEnabled"
        cmd = row["programmatic commands"]
        method = cmd.split("(")[0].strip()

        schema[method] = {
            "arg_type": row["arg_type"] if pd.notna(row["arg_type"]) else None,
            "arg_description": row["arg_description"],
            "callback": row["callback"]
        }

    return schema




def normalize_arg(arg, arg_type):
    if arg_type is None or arg == "":
        return None

    arg = arg.strip()

    if arg_type == "float":
        return float(arg)


    if arg_type == "int":
         return float(arg)


    if arg_type == "bool":
        if arg.lower() in ("true", "1"):
            return True
        if arg.lower() in ("false", "0"):
            return False
        raise ValueError(f"Invalid bool arg: {arg}")

    return arg


def command_to_key(cmd, float_tol=1e-6):
    """
    将一条 command 变成可 hash 的 canonical 表示
    """
    method = cmd["method"]
    arg_type = cmd["arg_type"]
    args = cmd["args"]

    if not args:
        return (method, ())

    if arg_type == "float":
        # 用整数化避免浮点误差
        args_key = tuple(int(round(a / float_tol)) for a in args)
    else:
        args_key = tuple(args)

    return (method, args_key)


def compare_cmd_list(parsed_cmd1, parsed_cmd2):
    keys1 = Counter([command_to_key(cmd) for cmd in parsed_cmd1])
    keys2 = Counter([command_to_key(cmd) for cmd in parsed_cmd2])
    return keys1 == keys2






def add_key_to_json(path, key, value):
    if not os.path.exists(path):
        data = {}
    else:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    data[key] = value
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def add_data_to_json(path, data: dict):
    if not os.path.exists(path):
        _data = {}
    else:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                _data = json.load(f)
            except json.JSONDecodeError:
                _data = {}

    _data.update(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_data, f, ensure_ascii=False, indent=4)




