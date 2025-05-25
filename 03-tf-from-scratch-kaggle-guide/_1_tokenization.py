import pandas as pd
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core concept in NLP."

tokenized_text = {}
# print(tokenizer(text))
# {'input_ids': [101, 19204, 6026, 3793, 2003, 1037, 4563, 4145, 1999, 17953, 2361, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# input_ids: numerical ids for each token
# token_type_ids: 0 for the first sentence, 1 for the second sentence
# attention_mask: 1 for the tokens that are not padding, 0 for the padding tokens (ignore)
#   - in BERT, always 1, cause is encoder only.


tokenized_text["Numerical Token"] = tokenizer(text)["input_ids"]
tokenized_text["Token"] = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
# print(tokenized_text)
# {'Numerical Token': [101, 19204, 6026, 3793, 2003, 1037, 4563, 4145, 1999, 17953, 2361, 1012, 102], 'Token': ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'concept', 'in', 'nl', '##p', '.', '[SEP]']}
# print("Tokenizer has a vocabulary size of", tokenizer.vocab_size, "words.")
# print(
#     "Tokenizer has a maximum sequence length of", tokenizer.model_max_length, "tokens."
# )
# print("\nOur text to tokenize:", text, "\n")
# print(pd.DataFrame(tokenized_text).T)


def tokenize_input(input_text: str, add_logs: bool = False):
    input_sequence = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",  # pytorch tensors instead of list
        padding="max_length",
        truncation=True,
        max_length=100,
    )["input_ids"]

    # encode_plus is same as encode, but adds:
    # - Adds special tokens (like [CLS] and [SEP] for BERT)
    # - Handles padding, truncation, and attention masks

    if add_logs:
        print(input_sequence)
        print("\nShape of output:", input_sequence.shape)
    return input_sequence


if __name__ == "__main__":
    sample_text = "We're going to reduce the maximum sequence length to 100 tokens, \
so we'll use a longer string here for demonstration purposes. We're not going to \
reach the full 100 tokens, so we'll pad our sequence with 0s."
    tokenize_input(sample_text, True)
