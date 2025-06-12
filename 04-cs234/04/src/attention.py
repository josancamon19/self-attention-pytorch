"""
Originally forked from Andrej Karpathy's minGPT.

CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###

    theta_i = 1 / 10000 ** (-2 * (torch.arange(dim // 2) - 1) / dim)
    # print(theta_i.shape)
    # for every position, 1, 100
    # for every dimension,
    # compute cos, sin

    positions = torch.arange(max_positions).unsqueeze(dim=1)
    theta_i = theta_i.unsqueeze(dim=0)
    # print(positions.shape, theta_i.shape)
    thetas = positions * theta_i
    # print(thetas.shape)
    cosines = torch.cos(thetas)
    sines = torch.sin(thetas)
    rope_cache = torch.stack([cosines, sines], dim=-1)
    # print(rope_cache.shape)
    # print(cosines.shape)

    # for pos in range(max_positions):
    #     thetas_pos = []
    #     for d in range(dim // 2):
    #         thetas_pos.append([math.cos(pos * theta_i[d]), math.sin(pos * theta_i[d])])
    #     rope_cache.append(thetas_pos)
    ### END YOUR CODE ###
    return rope_cache


# output = precompute_rotary_emb(128, 100)


def apply_rotary_emb_2dim(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g] # TODO: need to improve pytorch
    print("apply_rotary_emb x:", x.shape, "rope_cache:", rope_cache.shape)
    # You might find the following functions useful to convert
    # between real and complex numbers:
    # Reshape x to add a dimension for pairs of values
    rope_truncated = rope_cache[: x.shape[0],]
    print("rope_truncated", rope_truncated.shape)

    sequence_length = x.shape[0]
    x_pairs = x.view(sequence_length, -1, 2)
    x_pairs = x_pairs.to(torch.float16)
    # INSTEAD OF COMPLEX, could've done direct matrix approach
    # For each pair of dimesions [d1, d2] and rotation angle θ:
    #   d1_new = d1 * cos(θ) - d2 * sin(θ)
    #   d2_new = d1 * sin(θ) + d2 * cos(θ)
    # but apparently complex op just goes way faster than this and represents the same.
    print("x_pairs.shape:", x_pairs.shape)
    x_complex = torch.view_as_complex(x_pairs)
    print("x_complex", x_complex)
    print("x_complex.shape:", x_complex.shape)

    rope_complex = torch.view_as_complex(rope_truncated)
    print("rope_complex", rope_complex.shape)
    rotated_complex = x_complex * rope_complex
    print("rotated_complex", rotated_complex.shape)

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
    rotated_real = torch.view_as_real(rotated_complex)
    print("rotated_real", rotated_real.shape)
    rotated_x = rotated_real.flatten(-2)
    print("rotated_x", rotated_x.shape)
    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    ### YOUR CODE HERE ###
    ### END YOUR CODE ###
    return rotated_x


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # print("apply_rotary_emb x:", x.shape, "rope_cache:", rope_cache.shape)
    seq_len = x.shape[-2]
    rope_truncated = rope_cache[:seq_len]
    # print("apply_rotary_emb rope_truncated.shape:", rope_truncated.shape)

    *leading_dims, _ = x.shape
    # print("apply_rotary_emb *leading_dims, last_dim:", )

    x_pairs = x.view(*leading_dims, -1, 2) # or last_dim // 2, 2
    # print("apply_rotary_emb x_pairs.shape:", x_pairs.shape)
    x_pairs = x_pairs.to(torch.float16)

    x_complex = torch.view_as_complex(x_pairs)

    rope_complex = torch.view_as_complex(rope_truncated)
    rotated_complex = x_complex * rope_complex

    rotated_real = torch.view_as_real(rotated_complex)
    rotated_x = rotated_real.flatten(-2)
    return rotated_x


embed_dim = 12
n_heads = 2
batch_size = 5
head_size = embed_dim // n_heads

sequence_length = 16
max_length = 20

# 2d, x applied
# rope_cache = precompute_rotary_emb(embed_dim, max_length)
# x = torch.arange(0, embed_dim * sequence_length).reshape((sequence_length, -1))
# apply_rotary_emb_2dim(x, rope_cache)

# batch, q applied, not x
rope_cache = precompute_rotary_emb(embed_dim, max_length)
x = torch.arange(0, embed_dim * sequence_length * n_heads * batch_size).reshape(
    (batch_size, n_heads, sequence_length, -1)
)
apply_rotary_emb(x, rope_cache)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # TODO: [part g] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            ### YOUR CODE HERE ###
            # rope_cache = precompute_rotary_emb(config.n_embd, config.block_size)
            rope_cache = precompute_rotary_emb(
                config.n_embd // config.n_head, config.block_size
            )
            ### END YOUR CODE ###

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x):
        # embedding_dim = 256, //2 = 128
        # batch_size = 128
        # x sequence_length = 128

        B, T, C = x.size()
        # batch_size, sequence_length, embedding_size
        # print("forward x.shape:", x.shape)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nheads = 8, T = sequence_length, head_size = embed/heads = 256/8=32
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###
            # self.rope_cache = max_sequence_length, embed_dim // 2, 2
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)
            ### END YOUR CODE ###

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # keys of x1
        k = (
            self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2)
        )  # (B, nh, Tk, hs)

        # query with x2
        q = (
            self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2)
        )  # (B, nh, Tq, hs)

        # values from x1
        v = (
            self.value(x_kv)
            .view(Bk, Tk, self.n_head, Ck // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        B = max(Bk, Bq)

        att = att.masked_fill(self.mask[:, :, :Tq, :Tk] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, Tq, Cq)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
