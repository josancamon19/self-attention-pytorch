import pandas as pd
import json

from transformers import AutoTokenizer
from _0_shared import *  # noqa: F403, F401
import torch


def load_dataset(path: str) -> pd.DataFrame:
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse each line as JSON
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Problematic line: {line[:100]}...")  # Show first 100 chars
                continue

    return pd.DataFrame(data)


# Data explained:
# https://github.com/DenisPeskoff/2020_acl_diplomacy/tree/master/data

# {
#     "messages": [
#         "Hi Italy! Just opening up communication, and I want to know what some of your initial thoughts on the game are and if/how we can work together",
#         "Well....that's a great question, and a lot of it comes down to how free I'll be left to play in the West, no?",
#         "Well, if you want to attack France in the Mediterranean while I attack through Burgundy you can have Marseille and Iberia while I take Brest and Paris, then with France out of the way you could focus on Turkey or Austria. Sound fair?",
#         "Hello, I'm just asking about your move to Tyrolia. It's making me more than a little concerned",
#         "Totally understandable - but did you notice the attempt at Trieste?  Tyrolia is the natural support position for that attempt \ud83d\ude42",
#     ],
#     "sender_labels": [True, True, True, True, True],
#     "receiver_labels": [True, True, False, True, False],
#     "speakers": ["germany", "italy", "germany", "germany", "italy"],
#     "receivers": ["italy", "germany", "italy", "italy", "germany"],
#     "absolute_message_index": [87, 132, 138, 207, 221],
#     "relative_message_index": [0, 1, 2, 3, 4],
#     "seasons": ["Spring", "Spring", "Spring", "Winter", "Winter"],
#     "years": ["1901", "1901", "1901", "1901", "1901"],
#     "game_score": ["3", "3", "3", "5", "4"],
#     "game_score_delta": ["0", "0", "0", "1", "-1"],
#     "players": ["italy", "germany"],
#     "game_id": 12,
# }

# this doesn't look like a Transformer only thing, it could be a part of a bigger architecture.
# language modeling, yes, it has it, but the other fields are even important as well as the message.

# a simpler/interesting way of doing this, is adding the countries/scores/game stats as part of the tokens/inputs.
# and have a classifier head, lie/truth, or have multiple classifier heads, deception, confidence, intent.


def parse_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in df.iterrows():
        for (
            message,
            sender_name,
            receiver_name,
            sender_label,
            receiver_label,  # ignored
            score,
            score_delta,
            season,
            years,
        ) in zip(
            row["messages"],
            row["speakers"],
            row["receivers"],
            row["sender_labels"],
            row["receiver_labels"],
            row["game_score"],
            row["game_score_delta"],
            row["seasons"],
            row["years"],
        ):
            # sender has the label, receiver is inferred
            # True, truthful, False, deceptive
            rows.append(
                (
                    f"[SCORE:{score}] [SCORE_DELTA:{score_delta}] [SEASON:{season}] {sender_name} to {receiver_name}: {message}",
                    int(sender_label),  # 0,1
                )
            )
    return pd.DataFrame(rows, columns=["text", "label"])


def _compute_max_length(
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> int:
    max_length = 0
    full_data = pd.concat([train_df, test_df])
    for index, row in full_data.iterrows():
        tokenized = tokenizer.encode(row["text"])
        if len(tokenized) > max_length:
            max_length = len(tokenized)

    print(f"Max length: {max_length}")
    return max_length


def _get_custom_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    max_tokens = 420  # 4.2x the previous one will be a lot, prob can't train here.
    # max_length = _compute_max_length(tokenizer, train_df, test_df)

    special_tokens = []
    for score in range(19):
        special_tokens.append(f"[SCORE:{score}]")
    for delta in range(-18, 19):
        special_tokens.append(f"[SCORE_DELTA:{delta}]")

    seasons = ["Spring", "Fall", "Winter"]
    for season in seasons:
        special_tokens.append(f"[SEASON:{season}]")

    # tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # TODO: Adding special tokens causes issues with Embedding matrix, why?
    # TODO: is batch size too big? what's going on?
    # TODO: why is train accuracy 0.031, then validation 0.912, wtf
    # TODO: how big the model has to be? should train in gpu?
    # TODO: play with hyperparameters

    return tokenizer, max_tokens


if __name__ == "__main__":
    base_path = "04-practice/_2_data/"
    train_df = load_dataset(f"{base_path}/train.jsonl")
    test_df = load_dataset(f"{base_path}/test.jsonl")

    train_df = parse_dataset(train_df)
    test_df = parse_dataset(test_df)
    train_df.to_csv(f"{base_path}/train.csv", index=False)
    test_df.to_csv(f"{base_path}/test.csv", index=False)

    tokenizer, max_tokens = _get_custom_tokenizer()

    config, train_dataloader, val_dataloader = get_model_config_and_data_loaders(  # noqa: F405
        train_path=f"{base_path}/train.csv",
        test_path=f"{base_path}/test.csv",
        max_tokens=max_tokens,
        custom_tokenizer=tokenizer,
        y_label="label",
    )
    model = Transformer(config).to(config.device)  # noqa: F405
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(  # noqa: F405
        config,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        n_epochs=10,
        name_model="best_model_2",
    )
