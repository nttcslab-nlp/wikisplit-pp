import re

from src.utils import batch_sentence_tokenize, sentence_tokenize


def _preprocess(text: str) -> str:
    text = text.strip()
    text = text.strip("- ")

    if text.count('"') % 2 == 1:
        text = text.replace('"', "", 1)

    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("``", '"')

    text = text.replace("<::::>", "")
    text = text.replace("<::::> ", "")
    text = text.replace("''", '"')
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")

    text = text.replace(" - ", "-")
    text = text.replace(" -- ", "--")
    text = text.replace(" – ", "–")
    text = text.replace(" / ", "/")

    text = text.replace("(; ", "(")
    text = text.replace("محمدرضا رحيمی ", "")
    text = text.replace("(, ", "(")
    text = text.replace('( "', '("')
    text = text.replace(" (,)", "")
    text = text.replace(" ()", "")
    text = text.replace('( "', "(")
    text = text.replace(" :", ":")

    text = text.replace(" ²", "²")
    text = text.replace("<SEP> ", "")
    text = text.replace(" ​ ", " ")
    text = text.replace("‎", "")

    # remove spaces on both sides of an en-dash
    text = re.sub(r"\s*–⁠\s*", r"–⁠", text)
    # remove spaces on both sides of an en-dash (for a bit different char)
    text = re.sub(r"\s*–⁠\s*", r"–⁠", text)
    # remove spaces on both sides of an em-dash
    text = re.sub(r"\s*—\s*", r"—", text)

    text = re.sub(r'"(.*?)\s"', r'"\1"', text)
    text = re.sub(r"`\s*(.*?)\s*'", r"'\1'", text)
    text = re.sub(r"'\s*(.*?)\s*'", r"'\1'", text)
    text = re.sub(r"``\s*(.*?)\s*``", r'"\1"', text)

    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
    text = re.sub(r'""\s*(.*?)\s*""', r'"\1"', text)
    text = re.sub(r"‘\s*(.*?)\s*’", r"‘\1’", text)
    text = re.sub(r'\(\s*"(.*?)"\s*\)', r'("\1")', text)

    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
    text = re.sub(r"“\s*(.*?)\s*”", r'"\1"', text)
    text = re.sub(r'"(.*?)""(.*?)"', r'"\1" "\2"', text)
    text = re.sub(r'(^|[^\(])"(.*?)"', r'\1 "\2"', text)
    text = text.strip()

    text = re.sub(r"\s+", r" ", text)
    text = re.sub(r"\. (\d)", r".\1", text)
    text = re.sub(r"(\d), (\d)", r"\1,\2", text)

    text = text.replace("s ’ ", "s’ ")
    text = text.replace(" ’ ", "’")

    text = text.replace("i. e.", "i.e.")

    text = text.strip()
    return text


def preprocess(text: str) -> str:
    new = _preprocess(text)
    for _ in range(10):
        if new == text:
            break
        text = new
        new = _preprocess(text)
    return new


def validate_simple(text: str) -> bool:
    sentences = sentence_tokenize(text)
    return len(sentences) >= 2


def batch_validate_simple(texts: list[str]) -> list[bool]:
    texts = batch_sentence_tokenize(texts)
    return [len(sentences) >= 2 for sentences in texts]
