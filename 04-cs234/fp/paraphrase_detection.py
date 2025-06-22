import torch

from torch import nn, autocast
from torch.utils.data import DataLoader

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
import shared as utils
from peft import get_peft_model
from types import SimpleNamespace
import torch.multiprocessing as mp
from transformers import GPT2Tokenizer
import numpy as np

TQDM_DISABLE = False


class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    @utils.timeit
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        for param in self.gpt.parameters():
            param.requires_grad = True

        self.config = self.gpt.config
        self.generation_config = SimpleNamespace(temperature=0.7, top_p=0.9)

    def forward(self, input_ids, attention_mask, **kwargs):
        gpt_output: dict = self.gpt(input_ids, attention_mask)
        return self.gpt.hidden_state_to_token(gpt_output["last_token"])
        # self.gpt.hidden_state_to_token(gpt_output["last_hidden_state"])[:, -1, :]

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, weights_only=False)
    model = ParaphraseGPT(saved["args"])

    # model = ParaphraseGPT(args)
    if args.peft:
        model = get_peft_model(model, utils.get_lora_config(True))

    # model.load_state_dict(saved["model"])
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split="test")

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )
    para_test_dataloader = DataLoader(
        para_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn,
    )

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader, model, device, args.use_bf16
    )
    print(f"dev paraphrase acc :: {dev_para_acc:.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device, args.use_bf16
    )

    with open(args.para_dev_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")


def inference(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    yes_token_id = tokenizer.encode("yes")[0]
    no_token_id = tokenizer.encode("no")[0]
    print(yes_token_id, no_token_id)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # saved = torch.load(args.filepath, weights_only=False)
    # model = ParaphraseGPT(saved["args"])
    # model.load_state_dict(saved["model"])
    model = ParaphraseGPT(args)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    for i, item in enumerate(para_dev_data):
        s1, s2, same, sid = item[0], item[1], item[2], item[3]
        prompt = f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
        print("prompt:", prompt, "| answer:", same)
        encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"]).to(device)
        attention_mask = torch.LongTensor(encoding["attention_mask"]).to(device)

        print("tokens_id:", token_ids.shape)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
            logits = model(token_ids, attention_mask)

        logits = logits.float().detach().cpu().numpy()
        # print("logits:", logits.shape, logits[0, :10])
        # top_10_preds = np.argsort(logits, axis=1)[:, -10:]
        # print("top 10 preds:", top_10_preds.shape, top_10_preds)
        # top_10_values = np.take_along_axis(logits, top_10_preds, axis=1)
        # print(top_10_values)
        preds = np.argmax(logits, axis=1).flatten()
        print("prediction:", preds, f"\"{tokenizer.decode(preds)}\"")
        print("-------\n")
        if i == 100:
            break
        # break


# DISTRIBUTED LEARNINGS
# use torchrun instead of mp.spawn
# python process controlling other python processes.
# fuck, so DDP is a bad idea here, communication overhead is bigger than comp gain.
# python paraphrase_detection.py --use_gpu --batch_size 160 --distributed, 345 samples second
# python paraphrase_detection.py --use_gpu --batch_size 52, 100*52/16 = 325 samples per second

# with gradient accumulation 3, 360 samples per second
# single gpu, gradient accum: 338 samples per second

# maybe here, there's a slight 5/10% gain

# STOPPED TRYING STUFF, let's profile the model
# nvm profiling is so confusing,

# bf16, 2x as fast, 1/2 memory, wtf. (single GPU)
# now I'm confused, distributed is not the same? why, wtf
# now distributed doesn't work at all, after 10% fails, gpu memory full. Prob gradient accumulation, fuck

# TODO: I don't know yet if distributed communication/loss/training is working, solve

if __name__ == "__main__":
    # 0.86 default settings 10-1e-05-paraphrase.pt
    args = utils.get_args(utils.ModelTarget.paraphrase)
    # if args.distributed:
    #     gpus = torch.cuda.device_count()
    #     print("loading distributed training, gpus:", gpus)
    #     mp.spawn(utils.train_dist, args=(args, ParaphraseGPT, True), nprocs=gpus)
    # else:
    # ok, apparently gpt2-large has 0% accuracy, not trained.
    utils.train(args, ParaphraseGPT)
    # TODO: gpt2 saturated super fast (2nd epoch, 0.64 val loss), is train data, distributed? model seem to be predicting always 3919
    # -- imbalanced?
    # -- does a classifier head works best?
    test(args)
    # inference(args)
