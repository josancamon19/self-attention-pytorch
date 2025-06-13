import dataset
import models
import utils
import torch

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()

block_size = 128
pretrain_text = open("wiki.txt", encoding="utf-8").read()
finetune_text = open("birth_places_train.tsv", encoding="utf-8").read()

pretrain_dataset = dataset.CharCorruptionDataset(pretrain_text, block_size)
finetune_dataset = dataset.NameDataset(pretrain_dataset, finetune_text)

mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256,
)

model_nopretrain_finetuned = models.GPT(mconf)
model_nopretrain_finetuned.to(device)
model_nopretrain_finetuned.load_state_dict(torch.load("vanilla.model.params"))


model_pretrained = models.GPT(mconf)
model_pretrained.to(device)
model_pretrained.load_state_dict(torch.load("vanilla.pretrain.params"))

model_finetuned = models.GPT(mconf)
model_finetuned.to(device)
model_finetuned.load_state_dict(torch.load("vanilla.finetune.params"))

mconf.rope = True
model_pretrained_rope = models.GPT(mconf)
model_pretrained_rope.to(device)
model_pretrained_rope.load_state_dict(torch.load("rope.pretrain.params"))

model_finetuned_rope = models.GPT(mconf)
model_finetuned_rope.to(device)
model_finetuned_rope.load_state_dict(torch.load("rope.finetune.params"))


def predict(text: str, include_mask: bool = False, model: models.GPT | None = None):
    text = text + "⁇" if include_mask else text
    x = torch.tensor([pretrain_dataset.stoi[s] for s in text], dtype=torch.long)[
        None, ...
    ].to(device)
    pred = utils.sample(model, x, 32, sample=False)[0]
    return "".join([pretrain_dataset.itos[int(i)] for i in pred])[len(text) :]


def exec_pretrained(text: str):
    mp_1 = predict(text, True, model_pretrained)
    mp_2 = predict(text, False, model_pretrained)
    print("Pretrained model with mask:", mp_1)
    print("\nPretrained model without mask:", mp_2)

    # Rope models
    mrp_1 = predict(text, True, model_pretrained_rope)
    mrp_2 = predict(text, False, model_pretrained_rope)
    print("\nPretrained RoPE model with mask:", mrp_1)
    print("\nPretrained RoPE model without mask:", mrp_2)


def exec_finetuned(text: str):
    print("exec_finetuned text:", text)
    print()

    mp_1 = predict(text, True, model_nopretrain_finetuned)
    mp_2 = predict(text, False, model_nopretrain_finetuned)
    print("model_nopretrain_finetuned mask:", mp_1)
    print("model_nopretrain_finetuned no_mask:", mp_2)
    print()

    mp_1 = predict(text, True, model_finetuned)
    mp_2 = predict(text, False, model_finetuned)
    print("model_finetuned mask:", mp_1)
    print("model_finetuned no mask:", mp_2)
    print()

    # Rope models
    mrp_1 = predict(text, True, model_finetuned_rope)
    mrp_2 = predict(text, False, model_finetuned_rope)
    print("model_finetuned_rope mask:", mrp_1)
    print("model_finetuned_rope no mask:", mrp_2)


def explore_datasets():
    print("pretrain_dataset.data[0]:", pretrain_dataset.data[0])
    print("finetune_dataset.data[0]:", finetune_dataset.data[0])


print()

# This models have seen only wiki.txt in the corrupted dataset way.
# so depending on ⁇, we'll see how it behaves, try on finetuned data as well, and see
# TODO: understand key idea of cross entropy loss in the model ignore_index
# --- maybe now with proper pretrained you'll be able to see patterns properly

# TODO: play with datasets here as well, test loss in prediction manually.

# explore_datasets()
exec_finetuned("Where was John Owens born?")
