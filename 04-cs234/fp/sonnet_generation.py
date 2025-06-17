import torch
from torch import nn
from transformers import GPT2Tokenizer
from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from peft import get_peft_model
from types import SimpleNamespace
from shared import ModelTarget, get_args, get_lora_config, train

TQDM_DISABLE = False


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # By default, fine-tune the full model. TODO: this is maybe not idea.
        for param in self.gpt.parameters():
            param.requires_grad = True

        self.generation_config = SimpleNamespace(temperature=0.7, top_p=0.9) # peft

    @property
    def config(self):
        """Forward the config from the underlying GPT2 model for PEFT compatibility."""
        return self.gpt.config

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        gpt_output = self.gpt(input_ids, attention_mask)
        return self.gpt.hidden_state_to_token(gpt_output["last_hidden_state"])

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
        Generates an original sonnet using top-p sampling and softmax temperature.

        TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
        In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
        there are many.
        """
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(
            self.get_device()
        )

        for _ in range(max_length):
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = (
                logits_sequence[:, -1, :] / temperature
            )  # Apply temperature scaling

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[
                ..., :-1
            ].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(
                dim=-1, keepdim=True
            )  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.int64).to(self.get_device()),
                ],
                dim=1,
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[
            3:
        ]
        return token_ids, generated_output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Required method for PEFT compatibility with causal LM."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, weights_only=False)

    model = SonnetGPT(saved["args"])
    if args.peft:
        model = get_peft_model(model, get_lora_config(False))
    # model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        )[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f"{decoded_output}\n\n"
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(batch[1], "\noutput:", decoded_output, "-\n----")

    with open(args.sonnet_out, "w+") as f:
        f.write("--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


if __name__ == "__main__":
    args = get_args(ModelTarget.sonnet)
    train(args, SonnetGPT)
    generate_submission_sonnets(args)
    # peft def underperforms here, very little data to train a lot
