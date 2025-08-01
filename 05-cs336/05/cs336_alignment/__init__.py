# vLLM use to generate, huggingface tf to load qwen 2.5 math 1.5B model and tokenizer, not use Trainer utilities
# teach llm's to reason on math problems,
# - won't use our pre-trained models, too weak, using qwen one, and work on top of that model
# - new metric != perplexity, close gap between downstream tasks by using MATH 12k dataset
# ~
# COT reasoning and reasoning RL
# things like "step by step", STaR (Self-taught reasoner), bootrstraping loop, pre-trained samples, ones that lead to correctness are kept, and then finetuned
# verifiable rewards (O1, R1), policy gradient methods, pure RL with verifiable rewards
# https://arxiv.org/pdf/2501.12599
# https://github.com/huggingface/open-r1
# https://arxiv.org/pdf/2503.18892
# not available MATH dataset, can use GSM8k, or Tulu 3 SFT Math
# some of this don't provide exact response, so use https://github.com/huggingface/Math-Verify
# start by comparing base model performance against dataset using `r1_zero` prompt, check <think><answer> tags
# Prompt choice
# r1_zero is not best for qwen, only the question perform best, due to it had been trained on this data, but stay with r1 (you'll try `question_only` prompt later)

# using vLLM, inference for RL requires to be high performance,

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)
# Create an LLM.
llm = LLM(model="")  # TODO: include whatever model, hf parsed downloaded right away
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# • Qwen 2.5 Math 1.5B Base (for reasoning experiments):
# /data/a5-alignment/models/Qwen2.5-Math-1.5B
# • Llama 3.1 8B Base (for optional instruction tuning experiments):
# /data/a5-alignment/models/Llama-3.1-8B
# • Llama 3.3 70B Instruct (for optional instruction tuning experiments):
# /data/a5-alignment/models/Llama-3.3-70B-Instruct

# Check https://github.com/sail-sg/understand-r1-zero

# Zero shot
# 