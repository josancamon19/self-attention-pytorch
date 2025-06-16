import math
import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = rearrange(proj, "b t (h d) -> b t h d", h=self.num_attention_heads)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = rearrange(proj, "b t h d -> b h t d")
        return proj

    def attention(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
    ):
        # print("CausalSelfAttention.attention query.shape key.shape", key.shape)
        # print(
        #     "CausalSelfAttention.attention attention_mask.shape", attention_mask.shape
        # )

        batch_size, num_heads, sequence_length, att_head_size = query.shape

        # [batch, heads, from_pos, to_pos]
        attention_scores = query @ key.transpose(3, 2)
        attention_scores = attention_scores / math.sqrt(key.shape[-1])
        # print(attention_mask)
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length)).to(query.device)# .to(torch.bfloat16)
        causal_mask = (1.0 - causal_mask) * -10000.0 # match attention_mask base
        # print(attention_mask)
        attention_scores = attention_scores + attention_mask + causal_mask
        
        # print("CausalSelfAttention.attention scores.shape", attention_scores.shape)
        attention_weights = torch.softmax(attention_scores, -1)
        attention_weights = self.dropout(attention_weights)

        output = attention_weights @ value
        # print("CausalSelfAttention.attention output.shape", output.shape)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, sequence_length, num_heads * att_head_size)
        # print("CausalSelfAttention.attention output.shape", output.shape)
        # output: [bs, seq_len, hidden_state]
        return output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # print("CausalSelfAttention.forward", hidden_states.shape, attention_mask.shape)
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Calculate the multi-head attention.
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value
