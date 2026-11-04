import os
import json


class FormatArgs():
    def __init__(self, param_dict: dict = None):
        self.targetSpeaker = ["ムラサメ"]
        self.replace_speaker_name = "ムラサメ"
        self.merge_conversation = False
        self.ignore_non_speaker = False

        self.replace_word_pair = [("Genjuurou", "master")]

        self.is_multi_language_dataset = False
        self.language_idx = 0

        if param_dict is not None:
            for key, value in param_dict.items():
                exec(f'self.{key} = {repr(value)}')

    def replace_words_in_text(self, text: str):
        for it in self.replace_word_pair:
            if it[0] in text:
                text = text.replace(it[0], it[1])
        return text



def add_content_to_dict_list(key, dict, content, append=True):
    if key not in dict:
        dict[key] = [content]
    else:
        if append:
            dict[key].append(content)
        else:
            dict[key][-1] += content


def __load_single_language_dataset(dataset_path, ignore_non_speaker=False):
    dataset_dict = {}

    for curdir, dirs, files in os.walk(dataset_path):
        for fileName in files:
            with open(os.path.join(curdir, fileName), 'r', encoding='utf-8') as f:
                data = json.load(f)

                for k, conversation in enumerate(data):
                    dataset_dict[fileName + str(k)] = []

                    if type(conversation) is list and len(conversation) % 2 == 0:
                        speaker = "InitNoneSpeakerName"
                        for i in range(int(len(conversation) / 2)):
                            # print(f"{conversation[i*2]}: {conversation[i*2+1]}")
                            if conversation[i * 2] is None:
                                conversation[i * 2] = "None"
                                if ignore_non_speaker:
                                    continue


                            conversation[i * 2 + 1] = conversation[i * 2 + 1].replace("「", "").replace("」", "")

                            if speaker == conversation[i * 2]:
                                new_conversation = f"{speaker}: " + "".join(
                                    dataset_dict[fileName + str(k)][-1].split(": ")[1:]) + conversation[i * 2 + 1] + "。"
                                dataset_dict[fileName + str(k)][-1] = new_conversation
                            else:
                                dataset_dict[fileName + str(k)].append(
                                    f"{conversation[i * 2]}: {conversation[i * 2 + 1]}")
                            speaker = conversation[i * 2]
    return dataset_dict


def __load_multiple_language_dataset(dataset_path, language_idx=0, ignore_non_speaker=False):
    dataset_dict = {}

    for curdir, dirs, files in os.walk(dataset_path):
        for fileName in files:
            extension = os.path.splitext(fileName)[1]
            if extension != ".json":
                continue
            with open(os.path.join(curdir, fileName), 'r', encoding='utf-8') as f:
                data = json.load(f)

                for k, conversation in enumerate(data):
                    dataset_dict[fileName + str(k)] = []

                    if type(conversation) is list and len(conversation) % 2 == 0:
                        speaker = "InitNoneSpeakerName"
                        for i in range(int(len(conversation) / 2)):
                            # print(f"{conversation[i*2]}: {conversation[i*2+1]}")
                            if conversation[i * 2] is None:
                                conversation[i * 2] = "None"
                                if ignore_non_speaker:
                                    continue


                            # list with {speaker name (language), script}
                            # print(conversation[i*2+1][language_idx])

                            conversation[i * 2 + 1] = conversation[i * 2 + 1][language_idx][1].strip().replace("「", "").replace(
                                "」", "")

                            if speaker == conversation[i * 2]:
                                new_conversation = f"{speaker}: " + "".join(
                                    dataset_dict[fileName + str(k)][-1].split(": ")[1:]) + conversation[i * 2 + 1]
                                dataset_dict[fileName + str(k)][-1] = new_conversation
                            else:
                                dataset_dict[fileName + str(k)].append(
                                    f"{conversation[i * 2]}: {conversation[i * 2 + 1]}")
                            speaker = conversation[i * 2]
    return dataset_dict


def save_formatted_dataset(dataset_path_list, args: FormatArgs, save_name="outputs"):
    dataset_dict = {}
    for dataset_path in dataset_path_list:
        if args.is_multi_language_dataset:
            dataset_dict = dataset_dict | __load_multiple_language_dataset(dataset_path, language_idx=args.language_idx, ignore_non_speaker=args.ignore_non_speaker)
        else:
            dataset_dict = dataset_dict | __load_single_language_dataset(dataset_path, ignore_non_speaker=args.ignore_non_speaker)

    formatted_data_dict = {}

    count = 0
    for it in dataset_dict:
        prev_i = 0
        for i, data in enumerate(dataset_dict[it]):
            speaker = data.split(": ")[0]
            content = "".join(data.split(": ")[1:])
            # content = "".join(data.split(": ")[1:]).replace(speaker, args.replace_speaker_name)
            content = args.replace_words_in_text(content)
            # print(speaker, content)

            if args.targetSpeaker is not None and i > 0:
                if speaker in args.targetSpeaker:
                    prev_speaker = dataset_dict[it][i - 1].split(": ")[0]
                    prev_content = "".join(dataset_dict[it][i - 1].split(": ")[1:])
                    # prev_content = "".join(dataset_dict[it][i - 1].split(": ")[1:]).replace(speaker, args.replace_speaker_name)
                    prev_content = args.replace_words_in_text(prev_content)
                    # print(prev_speaker, speaker)
                    append = ((i - prev_i) == 2) and args.merge_conversation
                    add_content_to_dict_list(it, formatted_data_dict,
                                             f"<user>{prev_content}<assistant>{content}", append=append)
                    count += 1
                    prev_i = i

    print("find " + str(count) + " conversations")
    with open(f'{save_name}.json', 'w', encoding='utf-8') as file:
        json.dump(formatted_data_dict, file, ensure_ascii=False)


# dataset_dict = load_multiple_language_dataset(", language_idx=0)

if __name__ == '__main__':
    param = {
        "targetSpeaker": ["ムラサメ"],
        "replace_speaker_name": "ムラサメ",
        "merge_conversation": False,
        "is_multi_language_dataset": True,
        "language_idx": 0,
    }
    save_formatted_dataset("../dataset/senrenbanka_dataset/ムラサメ", FormatArgs(param))
