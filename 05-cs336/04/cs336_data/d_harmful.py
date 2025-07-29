import fasttext

hatespeech_model = fasttext.load_model(".models/jigsaw_fasttext_bigrams_hatespeech_final.bin")
nsfw_model = fasttext.load_model(".models/jigsaw_fasttext_bigrams_nsfw_final.bin")


def check_nsfw(content):
    pred = nsfw_model.predict(content)
    return pred[0][0].replace("__label__", ""), pred[1].item()


def check_hatespeech(content):
    pred = hatespeech_model.predict(content)
    return pred[0][0].replace("__label__", ""), pred[1].item()


# - What problems do you think might arise downstream in a language model when these filters are
# applied to create the training set? How might you mitigate these issues?
# TODO: run on the whole data, look through 20 random examples, are the judgments corect,
# - what'd be a suitable classifier confidence
