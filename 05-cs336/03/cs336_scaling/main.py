# FLOPs budget of 1e19
# operate in smaller scale first
# Be sure to carefully plan your runs before you get started
# once scaling laws budget of 2e18 is consumed, no further api requests
# HP requirement, batch size 128 or 256

# Q
# - Given 2e18 budget, how did you decide which runs to query?
# - Runs result, scaling laws fittinhg
# - - [ ] Fit for another method besides IsoFLOPs
# - Given the budget, what params/loss your scaling laws predict.
# - what hyper-parameters would you use given your predicted optimal params?
# - - To estimate num of non-embedding params for a model hp config, use 12n_{layer}*d_model^2.
# FORM https://docs.google.com/forms/d/e/1FAIpQLSegsTdt7uGATLAdk1NZlqOS14IefVIVeoHGyx9mu_5_Sdoc9A/viewform?usp=send_form


# - TODO: Request to API
# - TODO: save each result somewhere
# - TODO: how to setup using MuP initialization, or scale aware init
# - TODO: what sizes to try

# API
# /total_flops_used
# /loss 
# {
# d_model [64,1024], 
# num_layers [2, 24],
# num_heads [2, 16],
# batch_size {128,256},
# lr [1e-3, 1e-4],
# train_flops {1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18}
# }

# api_key header/post? wtf
# "http://hyperturing.stanford.edu:8000/docs",

# querying this endpoint with a previously queried doesn't discount FLOPs, tho save them and assert when calling
# endpoint returns
# loss, total_flops_used (new total)

# /previous_runs endpoint

# Consider the model.py architecture
# - abs pos instead of RoPE
# - LNorm instead of RMS
# - GeLU instead of SwiGLU, 2 Linear instead of 3, dff = 4d_model as usual
# - untied input/output embeddings
# - SlimPajama dataset
# - BPE 32k items on above dataset
# - seq_length 512
# - attn and residual dropout 0.1 ?
# - AdamW with 0.01 and gradient clipping 1.0
# - cos lr schedule to decay lr 10x, annealing steps = num of training steps, 
# - no lr warmup used.