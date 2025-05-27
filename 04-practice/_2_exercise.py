import pandas as pd
import json
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
                    sender_label,
                )
            )
            break
    return pd.DataFrame(rows, columns=["text", "label"])


if __name__ == "__main__":
    train_df = load_dataset("04-practice/_2_data/test.jsonl")
    test_df = load_dataset("04-practice/_2_data/test.jsonl")

    train_df = parse_dataset(train_df)
    test_df = parse_dataset(test_df)

    print("Train [rows, cols]:", train_df.shape)
    print("Val [rows, cols]:", test_df.shape)
    print()
    print(train_df.head())

    # TODO: determine max_length param, to check tokens
    # TODO: implement tokenizer again, add special tokens
    # special_tokens = ['[SCORE:0]', '[SCORE:1]', ..., '[WINNING]', '[LOSING]', ...]
    # Create all score tokens (0-18), delta, season, sender / receiver?
    # TODO: set up config
    # TODO: set up Transformer, classifier output?
    # TODO: BCELoss expects probabilities, but your classifier outputs logits
    # Hint: Either:
    # - Use a different loss function that expects raw logits
    # - Or add sigmoid activation to your classifier output
    # Think about which is more standard for classification
    # TODO: train
    
