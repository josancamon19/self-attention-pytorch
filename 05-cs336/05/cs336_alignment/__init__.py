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
import importlib.metadata

__version__ = importlib.metadata.version("alignment")