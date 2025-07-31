import fasttext
import re

hatespeech_model = fasttext.load_model(".models/jigsaw_fasttext_bigrams_hatespeech_final.bin")
nsfw_model = fasttext.load_model(".models/jigsaw_fasttext_bigrams_nsfw_final.bin")


def is_harmful(content: str, confidence_threshold: float = 0.8):
    nsfw_label, nsfw_confidence = check_nsfw(content)
    if nsfw_label == "nsfw" and nsfw_confidence > confidence_threshold:
        return True  # skip non-nsfw && confidence

    hs_label, hs_confidence = check_hatespeech(content)
    if hs_label == "toxic" and hs_confidence > confidence_threshold:
        return True
    return False


def check_nsfw(content):
    pred = nsfw_model.predict(re.sub(r"\n", "", content))
    return pred[0][0].replace("__label__", ""), pred[1].item()


def check_hatespeech(content):
    pred = hatespeech_model.predict(re.sub(r"\n", "", content))
    return pred[0][0].replace("__label__", ""), pred[1].item()


# - What problems do you think might arise downstream in a language model when these filters are
# applied to create the training set? How might you mitigate these issues?
# TODO: run on the whole data, look through 20 random examples, are the judgments corect,
# - what'd be a suitable classifier confidence
