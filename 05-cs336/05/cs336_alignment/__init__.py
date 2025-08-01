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
from typing import Callable, List
import json
from .drgrpo_grader import r1_zero_reward_fn
# Sample prompts.

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,  # stop=["</answer>"], include_stop_str_in_output=True
)
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True


# • Qwen 2.5 Math 1.5B Base (for reasoning experiments):
# /data/a5-alignment/models/Qwen2.5-Math-1.5B
# • Llama 3.1 8B Base (for optional instruction tuning experiments):
# /data/a5-alignment/models/Llama-3.1-8B
# • Llama 3.3 70B Instruct (for optional instruction tuning experiments):
# /data/a5-alignment/models/Llama-3.3-70B-Instruct

# Check https://github.com/sail-sg/understand-r1-zero

# use an answer parser, if 1/2 or 0.5 or she sold 10, or 10... r1_zero_reward_fn
# temperature 1, top p 1, stop token </answer>,

# Task 1, write a script to evaluate qwen 2.5 Math 1.5B zero shto performance on the dataset


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def evaluate_model(
    model: str = "Qwen/Qwen2.5-Math-1.5B",
    dataset_path: str = "data/gsm8k/test.jsonl",
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt",
):
    llm = LLM(model=model)

    # Load prompt template
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    # Create r1_zero format prompts
    prompts = []
    ground_truths = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        gt_answer = answer.split("#### ")[-1].strip()

        prompts.append(prompt_template.format(question=question))
        ground_truths.append(gt_answer)

    # Generate responses
    print(f"Evaluating {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate responses
    results = []
    for output, gt in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward_result = r1_zero_reward_fn(response, gt, fast=True)
        results.append(reward_result)

    # Calculate statistics
    total_examples = len(results)
    format_correct = sum(1 for r in results if r["format_reward"] > 0)
    answer_correct = sum(1 for r in results if r["answer_reward"] > 0)
    overall_correct = sum(1 for r in results if r["reward"] > 0)

    format_accuracy = format_correct / total_examples * 100
    answer_accuracy = answer_correct / total_examples * 100
    overall_accuracy = overall_correct / total_examples * 100

    # Print statistics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Dataset: {dataset_path}")
    print(f"Total examples: {total_examples}")
    print("-" * 50)
    print(
        f"Format accuracy: {format_correct}/{total_examples} ({format_accuracy:.2f}%)"
    )
    print(
        f"Answer accuracy: {answer_correct}/{total_examples} ({answer_accuracy:.2f}%)"
    )
    print(
        f"Overall accuracy: {overall_correct}/{total_examples} ({overall_accuracy:.2f}%)"
    )
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()
