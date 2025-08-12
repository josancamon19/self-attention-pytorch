import enum
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

{
    "problem": "Chandra has four bowls.  Each one is a different color (red, blue, yellow, green).  She also has exactly one glass the same color as each bowl.  If she chooses a bowl and a glass from the cupboard, how many pairings are possible?  One such pairing is a blue bowl and a yellow glass.",
    "level": "Level 2",
    "subject": "Prealgebra",
    "unique_id": "test/prealgebra/947.json",
    "answer": "16",
}
{
    "problem": 'In the diagram, the three concentric circles have radii of $4,$ $6,$ and $7.$ Three regions are labeled $X,$ $Y,$ or $Z$ below. Of these three regions, what is the difference between the area of the region with the greatest area and the area of the region with the smallest area? Express your answer in exact form.\n\n[asy]\nimport graph;\nfilldraw(circle((0,0),7), lightgray, black+linewidth(1));\nfilldraw(circle((0,0),6), gray, black+linewidth(1));\nfilldraw(circle((0,0),4), white, black+linewidth(1));\ndot((0,0));\nlabel("$X$",(2,0));\nlabel("$Y$",(5,0));\nlabel("$Z$",(6.5,0));\n[/asy]',
    "level": "Level 5",
    "subject": "Prealgebra",
    "unique_id": "test/prealgebra/1512.json",
    "answer": "7\\pi",
}


class EvalDataset(enum.Enum):
    GSM8K = "gsm8k"
    MATH = "math"


datasets = {"gsm8k": "data/math/test.jsonl", "math": "data/math/validation.jsonl"}


def evaluate_model_against_dataset(
    model: str = "Qwen/Qwen2.5-Math-1.5B",
    dataset: EvalDataset = EvalDataset.MATH,
    prompt_template_path: str = "src/prompts/r1_zero.prompt",
):
    llm = LLM(model=model)

    # Load prompt template
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    dataset_path = datasets[dataset.value]

    with open(dataset_path, "r") as f:
        dataset_items = [json.loads(line.strip()) for line in f]

    # Create r1_zero format prompts
    prompts = []
    ground_truths = []
    for item in dataset_items:
        if dataset == EvalDataset.MATH:
            question = item["problem"]
            gt_answer = item["answer"]
        else:
            question = item.get("question")
            answer = item["answer"]
            gt_answer = answer.split("#### ")[-1].strip()

        prompts.append(prompt_template.format(question=question))
        ground_truths.append(gt_answer)

    # Generate responses
    print(f"Evaluating {len(prompts)} examples...")
    sampling_params = SamplingParams(
        temperature=1.0,
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
    for output, gt in zip(outputs, ground_truths):
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

# MATH
# ==================================================
# EVALUATION RESULTS
# ==================================================
# Total examples: 5000
# --------------------------------------------------
# Format accuracy: 852/5000 (17.04%)
# Answer accuracy: 141/5000 (2.82%)
# Overall accuracy: 141/5000 (2.82%)
# ==================================================
