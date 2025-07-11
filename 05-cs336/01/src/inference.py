import torch

from src.train.transformer import get_args
from types import SimpleNamespace
from src.models.transformer import Transformer
from src.utils import data_loading, softmax
import numpy as np
from tqdm import tqdm


def generate(
    model_path: str,
    prompt: str,
    target_seq_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    assert top_p <= 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(model_path, map_location=device)
    args = SimpleNamespace(**data["args"]) if "args" in data else get_args()
    
    data = torch.load(model_path, map_location=device)
    model, tokenizer = Transformer.from_args(args, return_tokenizer=True)

    encoded = tokenizer.encode_batched([prompt], True, args.seq_length, True)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]

    assert (len(input_ids) + target_seq_length) < args.seq_length

    state_dict = data["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        for i in range(target_seq_length):
            logits = model(input_ids, attention_mask)[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            else:
                logits = logits / temperature
                probs = softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_values, sorted_indices = torch.sort(probs, descending=True)
                    cumm_sum = torch.cumsum(sorted_values, 1)
                    mask = cumm_sum > 0.8
                    mask[:, 0] = False  # set first item to False no matter what
                    sorted_values[mask] = 0.0
                    probs.zero_()
                    probs.scatter_(1, index=sorted_indices, src=sorted_values)
                    probs = probs / probs.sum(dim=1, keepdim=True)

                # TODO: check details, more than random.choice(weighted?)
                next_token = torch.multinomial(probs, 1)
                # print("next_token:",i, next_token)

            if next_token == 256:
                print("generate hit <|endoftext|> token.")
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1))], dim=1)

    generated_tokens = input_ids[0].tolist()
    decoded = tokenizer.decode(generated_tokens)
    print("decoded:", decoded)
    return decoded


def isolated_validation_check(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(model_path, map_location=device)
    args = SimpleNamespace(**data["args"]) if "args" in data else get_args()
    print("isolated_validation_check args:", args)

    model: Transformer = Transformer.from_args(args, return_tokenizer=False)
    valid_data = np.load(args.valid_dataset_path)

    if args.use_torch_compile:
        model = torch.compile(model)

    model.load_state_dict(data["model"])
    model.to(device)
    valid_loss = 0
    valid_steps = len(valid_data) // args.seq_length // args.batch_size
    with torch.inference_mode():
        pbar = tqdm(total=valid_steps, desc="valid-dataset")
        for _ in range(valid_steps):
            batch = data_loading(valid_data, args.batch_size, args.seq_length, device)
            input_ids, labels = batch
            output = model(input_ids, None)
            output_flatten = output.view(-1, output.shape[-1])
            labels = labels.contiguous().view(-1)
            valid_loss += F.cross_entropy(output_flatten, labels).item()
            pbar.update()
    print(valid_loss / valid_steps)


if __name__ == "__main__":
    generate(
        ".models/owt-epoch-8-lr-0.004-batch-64-arch-1024-768-6-12.pt",
        "So, as of today ",
        target_seq_length=920,
        temperature=1,
        top_p=1,
    )
