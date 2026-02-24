from __future__ import annotations

import os.path
import re
import pandas as pd
import SPMUtil
from typing import Tuple, List
from pathlib import Path
from Framework.types.RemoteDataType import RemoteDataType
from Framework.LabviewRemoteManager import LabviewRemoteManager
from Framework.ScanEventManager import ScanEventManager
from SubModule.LocalLLM.cmd_eval_util import parse_line, build_command_schema

cmd_csv = pd.read_csv(os.path.join(str(Path(__file__).resolve().parent), "./spm_command.csv"))
cmd_callback_function_dict = {}
for i, it in enumerate(cmd_csv["programmatic commands"]):
    cmd = it.split("(")[0]
    if isinstance(cmd_csv["callback"][i], float):
        callback = "nan"
    else:
        callback = cmd_csv["callback"][i]

    cmd_callback_function_dict[cmd] = callback

data_tree_list = list(LabviewRemoteManager.get_remote_command_tree()["data_tree"].keys())
schema = build_command_schema(cmd_csv)


def text2command_ver3(client, language_code: str = None) -> Tuple[
    str, str | List[Tuple[RemoteDataType, str]], List[str]]:
    callback_function_list = []
    history = client.get_history()

    assistant_msg_list = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            assistant_msg_list.append(content)
    response = re.sub(r"</?cmd>", "", assistant_msg_list[-1])
    # response = re.sub(r"</?cmd>", "", client)
    lines = response.split("\n")
    li = []
    for it in lines:
        if it.startswith("None"):
            return "None", response[4:], []
        else:
            cmd_dict = parse_line(it, schema)
            if cmd_dict is not None:
                # {
                #     "method": method,
                #     "raw_args": raw_arg_list,
                #     "args": norm_args,
                #     "arg_type": arg_type
                # }
                # print(cmd_dict)
                cmd = cmd_dict["method"]
                arg = "" if len(cmd_dict["args"]) == 0 else cmd_dict["args"][0]

                if cmd in cmd_callback_function_dict:
                    callback = cmd_callback_function_dict[cmd]
                else:
                    callback = "nan"

                # print("="*20)
                # print(cmd, arg, callback)
                # print(cmd in RemoteDataType.__members__)
                # print(cmd in cmd_callback_function_dict)
                # print(cmd in data_tree_list)
                # print("=" * 20)
                if cmd in RemoteDataType.__members__:
                    remote_type = RemoteDataType[cmd]
                    if remote_type == RemoteDataType.ScanEnabled:
                        is_scanning = ScanEventManager.instance.is_scanning

                        if arg is True:
                            if is_scanning:
                                return "None", "scan has already started.", []

                        elif arg is False:
                            if not is_scanning:
                                return "None", "scan has not started.", []

                    elif remote_type == RemoteDataType.StageOffset_X_Tube_ADD or \
                            remote_type == RemoteDataType.StageOffset_X_Tube or \
                            remote_type == RemoteDataType.StageOffset_Y_Tube_ADD or \
                            remote_type == RemoteDataType.StageOffset_Y_Tube:
                        arg = float(arg) / 37.5

                    li.append((remote_type.name, str(arg)))
                    callback_function_list.append(callback)
                elif cmd in cmd_callback_function_dict:
                    if cmd in ["Aux1MaxVoltage", "Aux2MaxVoltage", "Aux1MinVoltage", "Aux2MinVoltage"]:
                        arg = float(arg) / 37.5
                    li.append((cmd, str(arg)))
                    callback_function_list.append(callback)
                elif cmd in data_tree_list:
                    if cmd in ["Aux1MaxVoltage", "Aux2MaxVoltage", "Aux1MinVoltage", "Aux2MinVoltage"]:
                        arg = float(arg) / 37.5
                    li.append((cmd, str(arg)))
                    callback_function_list.append(callback)
                else:
                    return "None", cmd + " is not a correct command.", []
            else:
                continue
    return "OK", li, callback_function_list


if __name__ == '__main__':
    pass
    # result = text2command_ver3(
    #     """
    #     Expert Experimental Plan:
    #
    #     Step edge makes the current region unsuitable
    #     Maintain atomic scan parameters
    #     Switch to a cleaner nearby area
    #     <cmd>
    #     TipFix(arg=nan)
    #     Aux1MaxVoltage(20)
    #     Aux2MaxVoltage(20)
    #     </cmd>
    #     """
    # )
    # print(result)
