import time
import requests
import os
import torch
import torchaudio



def get_file_name(index=0):
    return time.strftime("%Y%m%d-%H%M%S") + "-" + str(index)


def save_audio(data: [], filepath=None):
    if filepath is None:
        filepath = "temp/"+get_file_name()
    torchaudio.save(filepath+".wav", torch.FloatTensor([data]), 16000, bits_per_sample=16)
    return os.path.abspath(filepath+".wav")


def read_audio(path: str, sampling_rate: int = 16000) -> []:
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0).to('cpu').detach().numpy().copy()



def slack_download_from_payload(payload, filename, slack_bot_token):
    url = payload["event"]["files"][0]["url_private_download"]
    r = requests.get(url, headers={'Authorization': 'Bearer %s' % slack_bot_token})
    with open(filename, 'w+b') as f:
        f.write(bytearray(r.content))
    return os.path.abspath(filename)


