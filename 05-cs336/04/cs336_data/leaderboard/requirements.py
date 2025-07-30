# Check token requirements based on training config
train_steps = 100_000
batch_size = 128
sequence_length = 512

total_tokens_needed = train_steps * batch_size * sequence_length
print("Training token requirements:")
print(f"  Steps: {train_steps:,}")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {sequence_length}")
print(f"  Total tokens needed: {total_tokens_needed:,} ({total_tokens_needed / 1e9:.1f}B)")

records_per_file = 25_000
survival_rate = 0.10  # rough estimate
chars_per_record = 3202  # From their example
tokens_per_char = 1 / 4  # GPT-2 tokenizer ratio

tokens_per_file = records_per_file * survival_rate * chars_per_record * tokens_per_char
print("\nWARC file analysis:")
print(f"  Records per file: {records_per_file:,}")
print(f"  Survival rate: {survival_rate:.1%}")
print(f"  Chars per record: {chars_per_record:,}")
print(f"  Tokens per char: {tokens_per_char:.3f}")
print(f"  Tokens per file: {tokens_per_file:,.0f}")

files_needed = total_tokens_needed / tokens_per_file
print(f"\nWARC needed: {files_needed:.0f}")
