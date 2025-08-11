import os
import csv
import glob
from vllm import LLM, SamplingParams
import re


def format_mmlu_question(question, choices):
    """Format MMLU question with multiple choice options"""
    formatted = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    return formatted


def create_mmlu_instruction(subject, question, choices):
    """Create instruction for MMLU question with subject context"""
    formatted_question = format_mmlu_question(question, choices)
    instruction = f'Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n{formatted_question}'
    return instruction


def extract_answer_letter(response):
    """Extract the answer letter from the model response"""
    # Look for patterns like "The correct answer is A" or just "A" at the end
    patterns = [
        r"[Tt]he correct answer is ([ABCD])",
        r"([ABCD])\s*\.?\s*$",
        r"Answer:\s*([ABCD])",
        r"^([ABCD])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            return match.group(1).upper()

    # If no clear pattern, look for any single letter A, B, C, or D
    letters = re.findall(r"\b([ABCD])\b", response)
    if letters:
        return letters[-1]  # Take the last occurrence

    return None


def evaluate_mmlu_accuracy(predictions, ground_truths):
    """Evaluate MMLU accuracy by comparing predicted letters with ground truth"""
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        extracted = extract_answer_letter(pred)
        if extracted == gt:
            correct += 1

    return correct / total * 100 if total > 0 else 0


def load_mmlu_csv(csv_path):
    """Load MMLU CSV file and return questions, choices, and answers"""
    questions = []
    choices = []
    answers = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # question, 4 choices, answer
                questions.append(row[0])
                choices.append(row[1:5])  # A, B, C, D choices
                answers.append(row[5])

    return questions, choices, answers


def mmlu_baseline(
    model: str = "meta-llama/Llama-3.1-8B",
    data_dir: str = "data/mmlu/dev",
    prompt_template_path: str = "src/prompts/zero_shot_system_prompt.prompt",
):
    """Evaluate model on MMLU dev set"""
    llm = LLM(model=model)

    # Load prompt template
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_results = {}
    total_correct = 0
    total_questions = 0

    for csv_file in csv_files:
        subject = os.path.basename(csv_file).replace("_dev.csv", "").replace("_", " ")
        print(f"\nEvaluating {subject}...")

        # Load the CSV data
        questions, choices, answers = load_mmlu_csv(csv_file)

        if not questions:
            print(f"No valid questions found in {csv_file}")
            continue

        # Create prompts
        prompts = []
        for question, choice_list in zip(questions, choices):
            instruction = create_mmlu_instruction(subject, question, choice_list)
            prompt = prompt_template.format(instruction=instruction)
            prompts.append(prompt)

        # Generate responses
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=256, stop=None
        )

        outputs = llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        # Evaluate accuracy
        accuracy = evaluate_mmlu_accuracy(responses, answers)
        correct_count = int(accuracy * len(answers) / 100)

        all_results[subject] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(answers),
            "responses": responses,
            "ground_truths": answers,
        }

        total_correct += correct_count
        total_questions += len(answers)

        print(f"{subject}: {correct_count}/{len(answers)} ({accuracy:.2f}%)")

    # Print overall results
    overall_accuracy = (
        total_correct / total_questions * 100 if total_questions > 0 else 0
    )
    print(f"\n{'=' * 60}")
    print("MMLU EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(
        f"Overall accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2f}%)"
    )
    print(f"{'=' * 60}")

    # Print per-subject breakdown
    print("\nPer-subject results:")
    for subject, result in sorted(all_results.items()):
        print(
            f"{subject:30}: {result['correct']:3}/{result['total']:3} ({result['accuracy']:6.2f}%)"
        )

    return all_results


if __name__ == "__main__":
    results = mmlu_baseline()
