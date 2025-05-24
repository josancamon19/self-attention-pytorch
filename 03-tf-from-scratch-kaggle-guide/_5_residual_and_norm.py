# First, we get our positional encoder output
# x = pos_enc_output

# Instantiate the layer normalization
# layer_norm = nn.LayerNorm(config.embedding_dimensions).to(config.device)

# Our output is then defined as the normalized output of the attention block
# plus the ouput of the positional encoder (skip connection)
# add_norm_output = layer_norm(x + multihead_attn(x))
