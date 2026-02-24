import asyncio
import requests
from googletrans import Translator
from lingua import Language, LanguageDetectorBuilder
from apis.google_translate import do_translate
languages = [Language.ENGLISH, Language.JAPANESE, Language.CHINESE]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

translator = Translator()


def detect_language(text):
    try:
        lang = lang_detector.detect_language_of(text)
        if lang is None:
            return "xx"
        elif lang == Language.CHINESE:
            return "zh"
        elif lang == Language.ENGLISH:
            return "en"
        elif lang == Language.JAPANESE:
            return "ja"
        else:
            return "xx"
    except Exception as e:
        pass

def translate_language(text, target_language_code="en"):
    code = detect_language(text)
    if code == target_language_code:
        return text, code
    else:
        return do_translate(text, code, target_language_code), code




if __name__ == '__main__':
    print(translate_language("草泥马",  "ja"))
