from __future__ import annotations

import os.path
import pandas as pd
import SPMUtil
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
from SystemLoader import SystemLoader
from Framework.types.RemoteDataType import RemoteDataType
from Framework.LabviewRemoteManager import LabviewRemoteManager
from Framework.ScanEventManager import ScanEventManager
from SubModule.LocalLLM.cmd_eval_util import build_command_schema
from SubModule.LocalLLM.command_validator import SPMState, Command, extract_commands, validate_sequence

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

# 1 V of the tube scanner = 37.5 nm
NM_PER_VOLT = 37.5


@dataclass(frozen=True)
class QueuedCommand:
    remote_type: str
    arg: str            # argument sent to LabVIEW (converted to volts for positions and scan sizes)
    callback: str
    command: Command    # original command in nm, used for the execution-time check


def read_instrument_state() -> SPMState:
    stage = ScanEventManager.instance.stageConfigure
    param = SystemLoader.MODULE_DICT["ArrayBuilderManager"].PythonScanParam
    return SPMState(
        x_nm=stage.Tube_Scanner_Offset_X * NM_PER_VOLT,
        y_nm=stage.Tube_Scanner_Offset_Y * NM_PER_VOLT,
        size_x_nm=(param.Aux1MaxVoltage - param.Aux1MinVoltage) * NM_PER_VOLT,
        size_y_nm=(param.Aux2MaxVoltage - param.Aux2MinVoltage) * NM_PER_VOLT,
        scanning=ScanEventManager.instance.is_scanning,
    )


def text2command_ver4(response: str, state: SPMState) -> Tuple[str, str | List[QueuedCommand]]:
    """
    response of the command SLM -> ("OK", commands) or ("None", reason)
    The whole sequence is rejected if any command is malformed, out of bounds,
    or invalid for the instrument state simulated from `state`; nothing is executed in that case.
    """
    is_valid, reason, commands = extract_commands(response, schema)
    if not is_valid:
        return "None", reason

    is_valid, reason = validate_sequence(state, commands)
    if not is_valid:
        return "None", reason

    li = []
    for c in commands:
        cmd, arg = c.method, "" if c.arg is None else c.arg
        callback = cmd_callback_function_dict.get(cmd, "nan")

        if cmd in RemoteDataType.__members__:
            remote_type = RemoteDataType[cmd]
            if remote_type in (RemoteDataType.StageOffset_X_Tube_ADD, RemoteDataType.StageOffset_X_Tube,
                               RemoteDataType.StageOffset_Y_Tube_ADD, RemoteDataType.StageOffset_Y_Tube):
                arg = float(arg) / NM_PER_VOLT
            li.append(QueuedCommand(remote_type.name, str(arg), callback, c))
        elif cmd in cmd_callback_function_dict or cmd in data_tree_list:
            if cmd in ["Aux1MaxVoltage", "Aux2MaxVoltage", "Aux1MinVoltage", "Aux2MinVoltage"]:
                arg = float(arg) / NM_PER_VOLT
            li.append(QueuedCommand(cmd, str(arg), callback, c))
        else:
            return "None", cmd + " is not a correct command."
    return "OK", li


if __name__ == '__main__':
    pass
    # state = SPMState(x_nm=0, y_nm=0, size_x_nm=10, size_y_nm=10, scanning=False)
    # result = text2command_ver4(
    #     """
    #     Expert Experimental Plan:
    #     - Step edge makes the current region unsuitable
    #     - Switch to a cleaner nearby area
    #     <cmd>
    #     TipFix()
    #     SwitchScanarea()
    #     Aux1MaxVoltage(20)
    #     Aux2MaxVoltage(20)
    #     ScanEnabled(True)
    #     </cmd>
    #     """, state
    # )
    # print(result)
