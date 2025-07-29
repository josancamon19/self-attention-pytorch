import fasttext
import re

langid_model = fasttext.load_model(".models/lid.176.bin")


def language_identification(
    content: str = "Hi, my name is John",
    determine_is_english: bool = False,
    confidence_threshold: float = 0.9,
):
    # why does it ask to not use \n ~ cause fasttext train models by line
    prediction = langid_model.predict(re.sub(r"\n", "", content))
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1].item()
    if determine_is_english:
        return label == "en" and confidence_threshold >= confidence
    return label, confidence
