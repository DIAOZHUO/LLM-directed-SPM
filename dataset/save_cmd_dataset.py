import re
import os
from util.dataset_format import add_content_to_dict_list
import util.dataset_util as dataset
import pandas as pd


def convert_to_function_tools_format(df):
    tools = []

    for _, row in df.iterrows():
        func_name = row['programmatic commands'].split('(')[0]
        func_description = row['description'].strip().rstrip('.')
        arg_type = row['arg_type'].strip().lower()
        arg_description = row['arg_description'].strip().rstrip('.')

        # Map to JSON Schema types
        json_type = "number" if arg_type == "float" else arg_type

        tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {
                            "type": json_type,
                            "description": arg_description
                        }
                    },
                    "required": ["arg"]
                }
            }
        }

        tools.append(tool)

    return tools


command_str_csv = pd.read_csv(os.path.dirname(__file__) + "../../ui/resources/spm_command.csv")
command_str = command_str_csv.drop(columns=["callback"])
# print(command_str)
# command_str = convert_to_function_tools_format(command_str)
command_str = str(command_str)
system_prompt = f"""
You are a robot controlling a scanning probe microscopy. Users will provide instructions in text format on how to control the device, and you'll need to translate these texts into specific programmatic commands.
You can control and set the scan parameters in the scanning probe microscopy.
Here is a list of commands in CSV format that you can invoke:\n
{command_str}\n
When writing commands, always enclose them between <cmd> and </cmd> tags. You should try your best to understand the instructions and use the list up functions to write. The function argument should follow the type I defined.
All commands must be enclosed inside a multi-line <cmd> block using the following exact format:\n
<cmd>
CommandA()
CommandB(True)
</cmd>
The sample bias should not be changed unless there are instructions to change it.
If the user's instructions can be accomplished by multiple step commands, then output them sequentially and separate each command with a new line.
The coordinate range accessible by the SPM tip and scan area on is -350 to 350 nm in the both x-direction and y-direction. The user command should NOT causes the probe to exceed the scanning area.
If the user's instructions cannot be carried out by the commands provided above alone, or the parameters are invalid, please respond with 'None' first and then give a reason to user. Otherwise, reply with the names of the corresponding programmatic commands and provide appropriate values within parentheses.
"""




print(system_prompt)


def get_formatted_dataset_dict(path):
    basename = os.path.basename(path)
    formatted_data_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        # print(file)

        pattern = r'(assistant|user):\s*(.*?)(?=\n(?:assistant|user):|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

        # Separate messages into user and assistant lists
        user_messages = [msg.strip() for role, msg in matches if role == 'user']
        assistant_messages = [msg.strip() for role, msg in matches if role == 'assistant']
        print(len(user_messages), len(assistant_messages))

        for i in range(len(user_messages)):
            # print("user:", user_messages[i])
            # print("assistant:", assistant_messages[i])
            add_content_to_dict_list(basename, formatted_data_dict,
                                     f"<user>{user_messages[i]}<assistant>{assistant_messages[i]}", append=True)

    return formatted_data_dict


if __name__ == '__main__':
    data_dict = get_formatted_dataset_dict("cmd_train")
    data_dict |= get_formatted_dataset_dict("cmd_planning")
    dataset.create_dialogue_dataset(None, save_path="../spm_gpt/finetune/spmcmd_train_dataset", data=data_dict, is_append=True)

    data_dict = get_formatted_dataset_dict("cmd_test")
    dataset.create_dialogue_dataset(None, save_path="../spm_gpt/finetune/spmcmd_test_dataset", data=data_dict, is_append=True)

    data_dict = get_formatted_dataset_dict("cmd_planning_test")
    dataset.create_dialogue_dataset(None, save_path="../spm_gpt/finetune/spmcmd_test_planning_dataset", data=data_dict, is_append=True)
