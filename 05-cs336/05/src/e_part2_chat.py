import argparse
import os
import glob
from vllm import LLM, SamplingParams


def load_prompts(prompts_dir: str = "src/prompts"):
    """Load available prompt templates"""
    prompt_files = glob.glob(os.path.join(prompts_dir, "*.prompt"))
    prompts = {}
    
    for file_path in prompt_files:
        prompt_name = os.path.basename(file_path).replace(".prompt", "")
        try:
            with open(file_path, "r") as f:
                prompts[prompt_name] = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return prompts


def display_prompt_menu(prompts):
    """Display available prompts and get user selection"""
    print("\nAvailable prompts:")
    print("0. No prompt (direct input)")
    
    prompt_names = list(prompts.keys())
    for i, name in enumerate(prompt_names, 1):
        print(f"{i}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect prompt (0-{len(prompt_names)}): ").strip()
            
            if choice == "0":
                return None, "direct"
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(prompt_names):
                selected_name = prompt_names[choice_idx]
                return prompts[selected_name], selected_name
            else:
                print(f"Please enter a number between 0 and {len(prompt_names)}")
        
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None, None


def format_with_prompt(user_input: str, prompt_template: str) -> str:
    """Format user input with the selected prompt template"""
    if "{instruction}" in prompt_template and "{response}" in prompt_template:
        # For training templates like alpaca_sft, remove the {response} part for inference
        template_parts = prompt_template.split("{response}")
        formatted_template = template_parts[0].format(instruction=user_input)
        return formatted_template.rstrip()
    elif "{instruction}" in prompt_template:
        return prompt_template.format(instruction=user_input)
    elif "{question}" in prompt_template:
        return prompt_template.format(question=user_input)
    else:
        # If no placeholder, append user input to the end
        return f"{prompt_template}\n{user_input}"


def chat_loop(model_path: str, temperature: float = 0.7, max_tokens: int = 512):
    """Run interactive chat loop with the specified model"""

    print(f"Loading model: {model_path}")
    try:
        llm = LLM(model=model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load available prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompt templates")

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop=None
    )

    print("\nChat interface ready. Type 'quit' or 'exit' to stop.")
    print(f"Settings: temperature={temperature}, max_tokens={max_tokens}")
    print("-" * 50)

    while True:
        try:
            # Select prompt template
            selected_prompt, prompt_name = display_prompt_menu(prompts)
            
            if prompt_name is None:  # User pressed Ctrl+C during selection
                print("\nGoodbye!")
                break
            
            print(f"\nUsing prompt: {prompt_name}")
            if selected_prompt:
                print(f"Template preview: {selected_prompt[:100]}{'...' if len(selected_prompt) > 100 else ''}")
            
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Format with selected prompt
            if selected_prompt:
                final_prompt = format_with_prompt(user_input, selected_prompt)
            else:
                final_prompt = user_input
            
            # Generate response
            outputs = llm.generate([final_prompt], sampling_params)
            response = outputs[0].outputs[0].text

            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with language models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model path (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    chat_loop(
        model_path=args.model, temperature=args.temperature, max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
