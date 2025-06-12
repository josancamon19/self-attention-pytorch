import dataset
import models
import utils
import torch

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

block_size = 128
text = open("wiki.txt", encoding="utf-8").read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)
mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256,
)
model_pretrained = models.GPT(mconf)
model_pretrained.to(device)

model_finetuned = models.GPT(mconf)
model_finetuned.to(device)

model_pretrained.load_state_dict(torch.load("vanilla.pretrain.params"))
model_finetuned.load_state_dict(torch.load("vanilla.finetune.params"))


def predict(text: str, include_mask: bool = False, model: models.GPT | None = None):
    text = text + "⁇" if include_mask else text
    x = torch.tensor([pretrain_dataset.stoi[s] for s in text], dtype=torch.long)[
        None, ...
    ].to(device)
    pred = utils.sample(model, x, 60, sample=False)[0]
    completion = "".join([pretrain_dataset.itos[int(i)] for i in pred])
    return completion


def exec_all(text: str):
    # mp_1 = predict(text, True, model_pretrained)
    # mp_2 = predict(text, False, model_pretrained)
    mf_1 = predict(text, True, model_finetuned)
    mf_2 = predict(text, False, model_finetuned)

    # print("Pretrained model with mask:", mp_1)
    # print("\nPretrained model without mask:", mp_2)
    print("\nFinetuned model with mask:")
    print(mf_1)
    print("\nFinetuned model without mask:")
    print(mf_2)

    # evaluate, when mask added, uses this as pred
    # pred = completion.split("⁇")[1]


# TODO: debug this more.
text = "Where was Thomas Wright Rudderow born?"
exec_all(text)
