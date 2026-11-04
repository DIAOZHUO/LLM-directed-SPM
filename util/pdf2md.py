import os.path

import pymupdf4llm
import re
import textwrap
from pathlib import Path



def __divide_section_forward(text, pattern):
    if type(text) != str:
        raise Exception("Not a string")
    if pattern in text:
        result = re.split(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if len(result) == 2:
            return result[1]

    return text


def __divide_section_backward(text, pattern):
    if type(text) != str:
        raise Exception("Not a string")

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if match:
        main_body = match.group(1).strip()
        return main_body
    else:
        return text


def md_paper_pipeline(file_path):
    result = pymupdf4llm.to_markdown(file_path, show_progress=True, ignore_images=True, ignore_graphics=True)
    # r'^(\#{1,6}\s+)?\*\*?ACKNOWLEDGMENTS\*\*?$'
    result = filter_contents(result)
    result = __divide_section_forward(result, "\nABSTRACT\n")
    return result.strip()


def filter_contents(result):
    # result = __divide_section_backward(contents, r"(.*)\nREFERENCES\n")
    # result = __divide_section_backward(result, r'(.*?)\b\**REFERENCES\**\b(.*)')
    result = __divide_section_backward(result, r"^(.*?)(?:\n\s*\*{0,2}REFERENCES\*{0,2}\s*\n?)")
    # result = __divide_section_backward(result, r'(.*?)\b\**ACKNOWLEDGMENTS\**\b(.*)')
    # result = __divide_section_backward(result, r"^(.*?)(?=\n\s*#{1,5}\s*\*{0,2}Acknowledg(e)?ment(s)?\*{0,2}\s*$)")
    result = __divide_section_backward(result, r"^(.*?)(?:\n\s*(?:#{1,5}\s*)?\*{0,2}Acknowledg(?:e)?ment(?:s)?\*{0,2}\s*\n?)")

    result = __divide_section_backward(result, r"(.*)\nAUTHOR DECLARATIONS\n")
    result = __divide_section_backward(result, r"(.*)\nConflict of Interest\n")
    result = __divide_section_backward(result, r"(.*)\nDATA AVAILABILITY\n")
    result = __divide_section_backward(result, r"(.*)\nAuthor Contributions\n")
    return result


def md_book_pipeline(file_path, filter_every_chapter=False) -> list[dict]:
    def is_chapter_heading(line: str):
        line = line.strip()

        patterns = [
            r'^(#{1,6})\s+\*\*Chapter\s+\d+\*\*$',  # ## **Chapter 1**
            # r'^(#{1,6})\s+\*\*[\d\.]+\s+.+\*\*$',  # ## **1.2 Title**
            # r'^(#{1,6})\s+[\d\.]+\s+.+$',  # ## 1.2 Title
            r'^(#{1,6})\s+\*\*[IVXLCDM]+\.\s+[A-Z\s\-]+\*\*$',  # ## **I. INTRODUCTION**
            r'^\*\*[IVXLCDM]+\.\s+[A-Z\s\-]+\*\*$',  # **I. INTRODUCTION**
            r'^[IVXLCDM]+\.\s+[A-Z\s\-]+$',  # I. INTRODUCTION
            r'^\*\*Chapter\s+\d+\*\*$',  # **Chapter 1**
            r'^CHAPTER\s+\d+\:?.*$',  # CHAPTER 3: RESULTS
            # ✅ NEW: Markdown header followed by multiple **bold blocks** (e.g., **2** **Method...**)
            r'^(#{1,6})\s+\*\*\d+\*\*\s+\*\*.+\*\*$',  # ##### **2** **Method...**
        ]

        for pattern in patterns:
            if re.match(pattern, line):
                return True
        return False

    # def is_acknowledgment_section(line: str) -> bool:
    #     if line.upper() == "ACKNOWLEDGMENTS":
    #         return True
    #     return bool(re.match(r'^(\#{1,6}\s+)?\*\*?ACKNOWLEDGMENTS\*\*?$', line.strip(), re.IGNORECASE))

    md_lines = pymupdf4llm.to_markdown(file_path, show_progress=True,
                                       ignore_graphics=True, ignore_images=True, force_text=True,
                                       fontsize_limit=9.5,
                                       table_strategy=None
                                       ).splitlines()

    chapters = []
    current_chapter = None
    stop_extracting = False

    for line in md_lines:
        print(line)
        # if is_acknowledgment_section(line):
        #     stop_extracting = True
        #     break

        if is_chapter_heading(line):
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = {
                "title": line.strip(),
                "content": ""
            }
        else:
            if current_chapter:
                if line.strip() != "":
                    current_chapter["content"] += line.strip() + "\n"

    if current_chapter and not stop_extracting:
        chapters.append(current_chapter)

    print("find %d chapters" % len(chapters))
    if filter_every_chapter:
        new_chapters = []
        for chapter in chapters:
            new = filter_contents(chapter["content"])
            new_chapters.append({"title": chapter["title"],
                                 "content": new})
        # for it in new_chapters:
        #     print(it["title"])
        #     print(it["content"])
        #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return new_chapters
    else:
        for it in chapters:
            print(it["title"])
            # print(it["content"])
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return chapters




def md_book_pipeline_seperate(folder_path) -> list[dict]:
    new_chapters = []


    folder_path = Path(folder_path)
    pdf_files = list(folder_path.glob("*.pdf"))
    # print(pdf_files)

    for file_path in pdf_files:
        chapter = md_paper_pipeline(file_path).replace("�", "f")
        new_chapters.append({"title": os.path.basename(file_path), "content": chapter})
    return new_chapters



if __name__ == '__main__':
    file_path = "../dataset/spm_paper/books/Introduction to Scanning Tunneling Microscopy (2nd edn)/1 Overview.pdf"
    # file_path = "../dataset/spm_paper/reviews/A 10 mK scanning probe microscopy facility.pdf"
    # file_path = "../dataset/spm_paper/reviews/Review of Scanning Probe Microscopy Techniques.pdf"
    # file_path = "../dataset/spm_paper/reviews/Scanning probe microscopy in the age of machine learning.pdf"
    # file_path = "../dataset/spm_paper/paper/Small Methods - 2024 - Diao - AI‐Equipped Scanning Probe Microscopy for Autonomous Site‐Specific Atomic‐Level.pdf"

    print(md_paper_pipeline(file_path))


    # file_path = "../dataset/spm_paper/books/Scanning Probe Microscopy Voigtlander.pdf"
    # file_path = "../dataset/spm_paper/books/Theories of scanning probe microscopes at the atomic scale.pdf"
    # file_path = "../dataset/spm_paper/reviews/Bian_et_al-2021-Nature_Reviews_Methods_Primers.pdf"
    # md_book_pipeline(file_path, filter_every_chapter=False)

# pathlib.Path("output.md").write_bytes(md_text.encode())
