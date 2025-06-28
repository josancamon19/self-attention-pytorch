import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# We expect you to build these components from scratch. In particular, you may not
# use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
# • torch.nn.Parameter
# • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1
# • The torch.optim.Optimizer base class


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        nn.init.normal_(self.embeddings, mean=0, std=0.02)

    def forward(self, token_ids: torch.tensor):
        return self.embeddings[token_ids]


def get_positional_encodings(embed_dim, max_positions=3):
    positions = torch.arange(max_positions).unsqueeze(1)
    dim_indices = torch.arange(0, embed_dim, 2)  # TODO: not sure if this part is clear
    div_term = 10000 ** (dim_indices / embed_dim)

    pe = torch.empty((max_positions, embed_dim))
    pe[:, 0::2] = torch.sin(positions / div_term)
    pe[:, 1::2] = torch.cos(positions / div_term)
    return pe


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        shape = (embedding_dim, head_size)
        # print("SelfAttention.__init__:", shape)
        self.Q = Parameter(torch.zeros(shape))
        self.K = Parameter(torch.zeros(shape))
        self.V = Parameter(torch.zeros(shape))

        nn.init.normal_(self.Q, std=0.02)
        nn.init.normal_(self.K, std=0.02)
        nn.init.normal_(self.V, std=0.02)

    def forward(self, x, padding_mask):
        # batch, seq_length, embedding_dim
        q = x @ self.Q
        k = x @ self.K
        v = x @ self.V

        # TODO: Causal Mask
        # print("SelfAttention.forward q.shape:", q.shape)
        attention_scores = q @ k.transpose(1, 2)

        mask = torch.tril(attention_scores) * (padding_mask.unsqueeze(1) if padding_mask is not None else 1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # print(attention_scores)
        # TODO: out of 7,7, which i,j refer to which tokens

        # print("SelfAttention.forward attention_scores:", attention_scores.shape)
        attention_weights = torch.softmax(attention_scores / math.sqrt(k.shape[-1]), dim=-1)
        print(attention_weights)
        # print("SelfAttention.forward attention_weights:", attention_weights.shape)
        output = attention_weights @ v
        # print("SelfAttention.forward output:", output.shape)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        # TODO: No need for SelfAttention block, just the other one is more annoying on dimensions, use this one for now
        self.attention = nn.Sequential(*[SelfAttention(embedding_dim, num_heads) for _ in range(num_heads)])
        self.W_O = nn.Parameter(torch.zeros((embedding_dim, embedding_dim)))

    def forward(self, x, padding_mask):
        # print(x.shape)
        output = [head(x, padding_mask) for head in self.attention]
        output = torch.cat(output, dim=-1)
        wo = output @ self.W_O
        print("MultiHeadSelfAttention.forward wo.shape:", wo.shape)
        return wo


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(embedding_dim, num_heads)
        #  3,5,128
        # pass each seq input through a ffn, `128, 128`?
        self.mlp_in = nn.Parameter(torch.zeros((embedding_dim, 4 * embedding_dim)))
        self.mlp_in_bias = nn.Parameter(torch.zeros((embedding_dim * 4,)))
        self.relu = F.relu  # TODO: should be implemented manually I guess, and uses swiGLU iirc
        self.mlp_out = nn.Parameter(torch.zeros((4 * embedding_dim, embedding_dim)))
        self.mlp_out_bias = nn.Parameter(torch.zeros((embedding_dim,)))

        nn.init.normal_(self.mlp_in, std=0.02)
        nn.init.normal_(self.mlp_out, std=0.02)

    def forward(self, x, padding_mask):
        # TODO: normalize
        attention = self.mhsa(x, padding_mask) + x
        # TODO: normalize
        proj_in = attention @ self.mlp_in + self.mlp_in_bias
        proj_in_activated = self.relu(proj_in)
        proj_out = proj_in_activated @ self.mlp_out + self.mlp_out_bias
        output = proj_out + attention
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.embeddings = EmbeddingLayer(vocab_size, embedding_dim)
        self.pe = get_positional_encodings(embedding_dim, max_sequence_length)
        # TODO: register pytorch buffer, device movement
        self.blocks = nn.ModuleList(TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers))
        # norm
        self.output = nn.Parameter(torch.zeros(embedding_dim, vocab_size))

    def forward(self, input_ids, padding_mask):
        print(self.pe[: len(input_ids), :].shape)
        tokens = self.embeddings(input_ids) + self.pe[: input_ids.shape[-1], :]
        for block in self.blocks:
            tokens = block(tokens, padding_mask)

        output = tokens @ self.output
        # print("output.shape:", output.shape)
        # return torch.softmax(output, dim=-1)
        return output  # output logits


tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
max_sequence_length = 100
embedding_dim = 128
num_layers = 1
num_heads = 1

tokenizer.pad_token = "[PAD]"
tokenized = tokenizer(
    ["Hi there, this is a test", "hey"],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
print(tokenized)
tf = Transformer(tokenizer.vocab_size, max_sequence_length, embedding_dim, num_layers, num_heads)
prob_dist = tf(tokenized["input_ids"], tokenized["attention_mask"])
