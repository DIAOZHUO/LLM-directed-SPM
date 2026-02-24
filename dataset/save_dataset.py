import re
import os
import ast
import json
from util.dataset_format import add_content_to_dict_list
import util.dataset_util as dataset


system_prompt = """
You are an expert in Scanning Probe Microscopy (SPM). Your primary role is to assist scientists who use SPM in their research. 
You will provide accurate, detailed, and professional answers to their questions related to all aspects of SPM, including experimental techniques, data analysis, instrumentation, and recent advances in the field. 
Your responses should reflect the depth of knowledge expected from a domain specialist and aim to support high-level scientific work.
"""




def get_formatted_dataset_dict(path):
    word_count = 0
    basename = os.path.basename(path)
    formatted_data_dict = {}

    with open(path, "r") as json_file:
        data = json.load(json_file)

    if data["type"] == "book" and data["answered"] is True:
        texts = data.pop("text")
        for chapter_idx, text in enumerate(texts):
            word_count += len(text.split())
            for i, it in enumerate(data[str(chapter_idx)]):
                # print(f"<user>:{it}<assistant>:{data[str(chapter_idx)+'_answers'][i]}")
                add_content_to_dict_list(basename, formatted_data_dict, f"<user>{it}<assistant>{data[str(chapter_idx)+'_answers'][i]}", append=True)

    elif data["type"] == "journal" and data["answered"] is True:
        q = data.pop("questions")
        a = data.pop("answers")
        for i, it in enumerate(q):
            # print(f"<user>:{it}\n<assistant>:{a[i]}\n")
            add_content_to_dict_list(basename, formatted_data_dict, f"<user>{it}<assistant>{a[i]}", append=True)
    print("word_count:", word_count)
    return formatted_data_dict



def get_formatted_distill_dataset_dict(path):
    word_count = 0
    basename = os.path.basename(path)
    formatted_data_dict = {}

    with open(path, "r") as json_file:
        data = json.load(json_file)

    if data["type"] == "book" and data["distilled"] is True:
        texts = data.pop("text")
        for chapter_idx, text in enumerate(texts):
            word_count += len(text.split())
            for i, it in enumerate(data[str(chapter_idx)]):
                answer = data[str(chapter_idx)+'_distilled_answers'][i]
                answer = re.sub(r'(?i)^improved answer:\s*\n*', '', answer)
                add_content_to_dict_list(basename, formatted_data_dict, f"<user>{it}<assistant>{answer}", append=True)
    else:
        print("Ignore:", path)
    print("word_count:", word_count)
    return formatted_data_dict




