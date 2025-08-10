from vllm import LLM, SamplingParams
from src.drgrpo_grader import r1_zero_reward_fn
import json
from tqdm.auto import tqdm


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


def evaluate_model_against_dataset(
    model: str = "Qwen/Qwen2.5-Math-1.5B",
    dataset_path: str = "data/gsm8k/test.jsonl",
    prompt_template_path: str = "src/prompts/r1_zero.prompt",
    sampling_temperature: float = 0.0,
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
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_model(llm, sampling_params, prompts, ground_truths)
    compute_eval_stats(results)


def evaluate_model(llm, sampling_params, prompts, ground_truths):
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate responses
    results = []
    for output, gt in tqdm(
        list(zip(outputs, ground_truths)),
        total=len(ground_truths),
        desc="Scoring",
        dynamic_ncols=True,
        disable=True,
    ):
        response = output.outputs[0].text
        reward_result = r1_zero_reward_fn(response, gt, fast=True)
        results.append(reward_result)
    return results


def compute_eval_stats(results, print_results: bool = True):
    # Calculate statistics
    total_examples = len(results)
    format_correct = sum(1 for r in results if r["format_reward"] > 0)
    answer_correct = sum(1 for r in results if r["answer_reward"] > 0)
    overall_correct = sum(1 for r in results if r["reward"] > 0)

    format_accuracy = format_correct / total_examples * 100
    answer_accuracy = answer_correct / total_examples * 100
    overall_accuracy = overall_correct / total_examples * 100
    if not print_results:
        return format_accuracy, answer_accuracy, overall_accuracy

    # Print statistics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    # print(f"Model: {model}")
    # print(f"Dataset: {dataset_path}")
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
    return format_accuracy, answer_accuracy, overall_accuracy


if __name__ == "__main__":
    evaluate_model_against_dataset()
# Model: Qwen/Qwen2.5-Math-1.5B
# Dataset: data/gsm8k/test.jsonl
# Total examples: 1319
# --------------------------------------------------
# Format accuracy: 258/1319 (19.56%)
# Answer accuracy: 32/1319 (2.43%)
# Overall accuracy: 32/1319 (2.43%)
# ==================================================
