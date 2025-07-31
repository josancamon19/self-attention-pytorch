import re


def clean_text_for_fasttext(text):
    text = re.sub(r"^\d+ ===\n", "", text)  # Remove "1 ===\n"
    text = re.sub(r"^URL: .*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Content-Length: .*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{80,}\n", "", text, flags=re.MULTILINE)  # 80+ dashes
    text = re.sub(r"^={80,}\n*", "", text, flags=re.MULTILINE)  # 80+ equals

    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()
